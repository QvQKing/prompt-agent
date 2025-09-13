#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat API runner with batch input support + evaluation.

默认行为（无需参数）：
- 扫描 eval/ 目录下所有 *.parquet
- 对每个数据集逐条推理，保存到：
    chat-results/<dataset>.json                # 全量对话记录（含 trace）
    results-5/<dataset>/results.json           # 评测最小结构（含对话全文、最终回答）
    results-5/<dataset>/results_eval.json      # {question, ground_truth, answer}
    results-5/<dataset>/eval.json              # 指标（EM/F1/Substring/Semantic）

备注：
- dataset 名优先使用 parquet 的 data_source 唯一值，否则退回文件名（去扩展名）。
- 仍支持单条/多条/单文件模式；评测文件仅在目录批量模式下生成（符合你的需求）。
"""

import argparse
import json
import importlib
import os
import sys
import glob
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config


# ======== 默认值（按你的要求写入）========
DEFAULT_INPUT_DIR = "eval"
DEFAULT_OUTPUT_DIR = "chat-results-eval"   # 对话记录（全量）
DEFAULT_RESULTS_DIR = "results-eval"     # 评测三件套
DEFAULT_INPUT_FIELD = "prompt"        # 2Wiki 类数据的字段
# ========================================


# ANSI color codes for colored output
COLORS: Dict[str, str] = {
    "user": "\033[1;34m",       # Bold Blue
    "assistant": "\033[1;32m",  # Bold Green
    "tool": "\033[1;33m",       # Bold Yellow
    "tool_call": "\033[1;35m",  # Bold Purple
    "reset": "\033[0m",         # Reset to default
    "bg_user": "\033[44m",      # Blue background
    "bg_assistant": "\033[42m", # Green background
    "bg_tool": "\033[43m",      # Yellow background
    "bg_tool_call": "\033[45m", # Purple background
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run chat inference with configurable parameters + evaluation')

    # 环境与 API
    parser.add_argument('--tools', type=str, nargs='*', default=default_config.TOOLS,
                        help='Tool names (list). Example: --tools web_search calculator')
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY,
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE,
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME,
                        help='Model name for inference')

    # 采样参数
    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY,
                        help='Repetition penalty for generation')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='Max assistant/tool turns to avoid infinite loops')

    # 输入（单条/多条/单文件）——可选
    parser.add_argument('--question', type=str, default=None,
                        help='Single question')
    parser.add_argument('--questions', nargs='+', default=None,
                        help='Multiple questions (space-separated, wrap each in quotes)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to a single .txt/.jsonl/.parquet file for batch run')
    parser.add_argument('--input-field', type=str, default=DEFAULT_INPUT_FIELD,
                        help=f'Field containing question or messages (default "{DEFAULT_INPUT_FIELD}" for 2Wiki parquet)')

    # 目录批量模式（默认按你的要求）
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'Directory to scan .parquet files when no explicit inputs are given (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to write per-dataset JSON files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help=f'Directory to write evaluation JSON files (default: {DEFAULT_RESULTS_DIR})')

    # 配置
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config .py file (override defaults)')

    # 输出（仅单条/单文件模式会用到）
    parser.add_argument('--output-file', type=str, default=None,
                        help='Write predictions as a single JSON file (list). Ignored in folder-batch mode.')

    # 其他
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')

    return parser.parse_args()


def load_custom_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(custom_config)  # type: ignore[attr-defined]
    return custom_config


def print_user(use_colors: bool, text: str) -> None:
    if use_colors:
        print(f"{COLORS['bg_user']} User {COLORS['reset']} {COLORS['user']}{text}{COLORS['reset']}")
    else:
        print(f"User: {text}")


def print_assistant(use_colors: bool, text: str) -> None:
    if use_colors:
        print(f"\n{COLORS['bg_assistant']} Assistant {COLORS['reset']} {COLORS['assistant']}{text}{COLORS['reset']}")
    else:
        print(f"\nAssistant: {text}")


def print_tool_call(use_colors: bool, fn_name: str, args_str: str) -> None:
    if use_colors:
        print(f"\n{COLORS['bg_tool_call']} Tool Call {COLORS['reset']} {COLORS['tool_call']}Function: {fn_name}{COLORS['reset']}")
        print(f"{COLORS['tool_call']}Arguments:{COLORS['reset']}\n{args_str}")
    else:
        print(f"\n[Tool Call] Function: {fn_name}")
        print(f"Arguments:\n{args_str}")


def print_tool_result(use_colors: bool, result: str) -> None:
    if use_colors:
        print(f"\n{COLORS['bg_tool']} Tool {COLORS['reset']} {COLORS['tool']}{result}{COLORS['reset']}")
    else:
        print(f"\nTool: {result}")


def _as_py(obj: Any) -> Any:
    """Convert numpy/pandas scalars to native Python, and arrays to python lists."""
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def normalize_to_messages(obj: Any, instruction_following: str) -> List[Dict[str, str]]:
    """
    支持：
      - list[{'role','content'}] -> 直接返回（确保都是字符串）
      - str -> 包装成一条 user 消息，并拼接 INSTRUCTION_FOLLOWING
      - 其他 -> str(obj) 后按上面处理
    """
    obj = _as_py(obj)

    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and 'role' in obj[0] and 'content' in obj[0]:
        msgs: List[Dict[str, str]] = []
        for m in obj:
            role = str(m.get('role', 'user'))
            content = "" if m.get('content') is None else str(m['content'])
            msgs.append({"role": role, "content": content})
        return msgs

    if isinstance(obj, str):
        return [{
            "role": "user",
            "content": "Question: " + obj + '\n' + instruction_following
        }]

    return [{
        "role": "user",
        "content": "Question: " + str(obj) + '\n' + instruction_following
    }]


def iter_inputs_from_file(path: str, input_field: str, instruction_following: str
                          ) -> Iterable[Tuple[str, List[Dict[str, str]], Dict[str, Any]]]:
    """
    从 .txt/.jsonl/.parquet 逐条产出 (item_id, messages, meta)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                q = line.strip()
                if not q:
                    continue
                item_id = f"line_{i}"
                messages = normalize_to_messages(q, instruction_following)
                yield item_id, messages, {"source": os.path.basename(path)}

    elif ext == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"question": line.strip()}

                payload = (obj.get('messages', None)
                           or obj.get('prompt', None)
                           or obj.get('question', None)
                           or obj.get('text', None)
                           or obj)

                messages = normalize_to_messages(payload, instruction_following)
                item_id = str(obj.get('id', f"row_{i}"))
                meta = {k: _as_py(v) for k, v in obj.items() if k not in ('messages', 'prompt')}
                meta["source"] = os.path.basename(path)
                yield item_id, messages, meta

    elif ext == '.parquet':
        df = pd.read_parquet(path)
        # 若没传或字段不存在，尝试 prompt/question
        if input_field not in df.columns:
            if 'prompt' in df.columns:
                input_field = 'prompt'
            elif 'question' in df.columns:
                input_field = 'question'
            else:
                raise KeyError(f"Input field '{input_field}' not in parquet columns: {list(df.columns)}")

        for i, row in df.iterrows():
            raw = row[input_field]
            messages = normalize_to_messages(raw, instruction_following)
            item_id = str(row.get('id', f"row_{i}"))  # type: ignore[attr-defined]

            meta: Dict[str, Any] = {}
            # 附带常见评测字段
            for key in ['data_source', 'ability', 'reward_model', 'ground_truth', 'style']:
                if key in df.columns:
                    meta[key] = _as_py(row[key])
            meta["source"] = os.path.basename(path)

            yield item_id, messages, meta

    else:
        raise ValueError(f"Unsupported input file extension: {ext}")


def yield_inputs(args: argparse.Namespace, instruction_following: str
                 ) -> Iterable[Tuple[str, List[Dict[str, str]], Dict[str, Any]]]:
    """
    统一产出 (item_id, messages, meta)
    """
    if args.input_file:
        yield from iter_inputs_from_file(args.input_file, args.input_field, instruction_following)
        return

    if args.questions:
        for i, q in enumerate(args.questions):
            item_id = f"q_{i}"
            messages = normalize_to_messages(q, instruction_following)
            yield item_id, messages, {"source": "cmd:--questions"}
        return

    if args.question:
        item_id = "q_0"
        messages = normalize_to_messages(args.question, instruction_following)
        yield item_id, messages, {"source": "cmd:--question"}
        return

    print("ERROR: No input provided. Use --question, --questions, or --input-file.", file=sys.stderr)
    sys.exit(2)


def extract_preview_user_text(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return "\n".join(m.get("content", "") for m in messages)


# 正则用于兜底校验
RE_THINK = re.compile(r"<think>.*?</think>", re.S)
RE_ANSWER = re.compile(r"<answer>.*?</answer>", re.S)


def run_inference_for_item(
    client: OpenAI,
    model_name: str,
    tool_descs: List[Dict[str, Any]],
    env: NousToolEnv,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    stop_tokens: List[str],
    base_messages: List[Dict[str, str]],
    use_colors: bool,
    max_steps: int,
) -> Dict[str, Any]:
    """
    对单条样本跑对话工具循环，返回 {output, trace}
    """
    messages: List[Dict[str, Any]] = [dict(m) for m in base_messages]
    preview = extract_preview_user_text(base_messages)
    print_user(use_colors, preview)

    steps = 0

    while True:
        steps += 1
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,         # type: ignore[arg-type]
            tools=tool_descs,          # list of tool descriptions
            tool_choice="auto",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"repetition_penalty": repetition_penalty},
            stop=stop_tokens
        )

        response_message = response.choices[0].message

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": response_message.content
        }

        if response_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in response_message.tool_calls
            ]

        messages.append(assistant_message)
        print_assistant(use_colors, response_message.content or "")

        if response_message.tool_calls:
            # 执行所有工具调用
            for tool_call in response_message.tool_calls:
                # pretty args
                try:
                    args_dict = json.loads(tool_call.function.arguments)
                    formatted_args = json.dumps(args_dict, indent=2, ensure_ascii=False)
                except Exception:
                    args_dict = tool_call.function.arguments
                    formatted_args = str(tool_call.function.arguments)

                print_tool_call(use_colors, tool_call.function.name, formatted_args)

                # 执行工具
                exec_args = args_dict if isinstance(args_dict, dict) else json.loads(args_dict)
                tool_result = env.tool_map[tool_call.function.name].execute(exec_args)
                tool_content = tool_result.get("content", "")

                print_tool_result(use_colors, tool_content)

                messages.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call.id
                })

            # 工具返回后，强制下一轮先 <think> 再 <answer>
            messages.append({
                "role": "system",
                "content": (
                    "You MUST update your reasoning based on the latest tool results. "
                    "Before giving your final answer, FIRST write your reasoning inside <think>...</think>, "
                    "then provide the final answer inside <answer>...</answer>. "
                    "Output both tags now."
                )
            })
            # 继续下一轮生成或早停
            if steps < max_steps:
                continue
            else:
                print("\n[WARN] Reached --max-steps right after tool calls; returning latest assistant message.")
                return {"output": response_message.content or "", "trace": messages}

        else:
            # 无工具调用 -> 最终
            final_output = response_message.content or ""
            has_think = bool(RE_THINK.search(final_output))
            has_answer = bool(RE_ANSWER.search(final_output))

            if not (has_think and has_answer):
                messages.append({
                    "role": "system",
                    "content": (
                        "Your output must include both <think>...</think> and <answer>...</answer>. "
                        "Now, produce them (first <think>, then <answer>) based on all context and tool results."
                    )
                })
                if steps < max_steps:
                    continue
                else:
                    print("\n[WARN] Reached --max-steps while enforcing tag format; returning latest assistant message.")
                    return {"output": final_output, "trace": messages}

            return {"output": final_output, "trace": messages}


# ---------- 目录批量模式（对话与评测） ----------

_SANITIZE_RE = re.compile(r'[^A-Za-z0-9._-]+')


def sanitize_filename(name: str) -> str:
    name = name.strip()
    if not name:
        return "dataset"
    return _SANITIZE_RE.sub('_', name)


def infer_dataset_name_from_parquet(path: str) -> str:
    """
    优先 data_source 唯一值；否则使用文件名（去扩展名）
    """
    try:
        df = pd.read_parquet(path, columns=None)
        if 'data_source' in df.columns:
            vals = pd.unique(df['data_source'].astype(str))
            vals = [v for v in vals if v and v.lower() != 'nan']
            if len(vals) == 1:
                return sanitize_filename(vals[0])
    except Exception:
        pass
    return sanitize_filename(os.path.splitext(os.path.basename(path))[0])


def process_parquet_file(
    path: str,
    input_field: str,
    instruction_following: str,
    run_one_item_fn,
    client: OpenAI,
    model_name: str,
    tool_descs: List[Dict[str, Any]],
    env: NousToolEnv,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    use_colors: bool,
    max_steps: int,
) -> List[Dict[str, Any]]:
    """
    对单个 parquet 文件运行推理，返回记录列表（不写盘）。
    """
    result_records: List[Dict[str, Any]] = []

    print(f"\n[FILE] {path}")
    for item_id, messages, meta in iter_inputs_from_file(path, input_field, instruction_following):
        print(f"\n===== Item {item_id} =====")
        out = run_one_item_fn(
            client=client,
            model_name=model_name,
            tool_descs=tool_descs,
            env=env,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop_tokens=["</tool_call>"],
            base_messages=messages,
            use_colors=use_colors,
            max_steps=max_steps,
        )
        rec: Dict[str, Any] = {
            "id": item_id,
            "model": model_name,
            "input": messages,
            "output": out["output"],
            "trace": out["trace"],   # 保存完整轨迹以便渲染对话全文
            "meta": meta,
        }
        result_records.append(rec)

    return result_records


def write_dataset_json(output_dir: str, dataset_name: str, records: List[Dict[str, Any]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}.json")
    with open(out_path, 'w', encoding='utf-8') as wf:
        json.dump(records, wf, ensure_ascii=False, default=_as_py, indent=2)
    return out_path


# ---------- 评测工具：对话渲染 & 指标 ----------

def _serialize_tool_calls_for_plain(resp_tool_calls) -> List[Dict[str, Any]]:
    out = []
    if not resp_tool_calls: return out
    for tc in resp_tool_calls:
        out.append({
            "id": tc.get("id"),
            "type": tc.get("type"),
            "function": {
                "name": (tc.get("function") or {}).get("name"),
                "arguments": (tc.get("function") or {}).get("arguments"),
            }
        })
    return out


def render_messages_plain(messages: List[Dict[str, Any]]) -> str:
    """
    把整段 messages 渲染成“纯文本对话全文”，无任何键值字段或重复快照：
    - 角色行：system / user / assistant
    - assistant 若含 tool_calls：追加 <tool_call>JSON</tool_call>
    - tool 结果：渲染为 user + <tool_response>...</tool_response>
    """
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role in ("system", "user", "assistant"):
            lines.append(role)
            lines.append(content)
            if role == "assistant" and msg.get("tool_calls"):
                for tc in _serialize_tool_calls_for_plain(msg["tool_calls"]):
                    name = (tc.get("function") or {}).get("name")
                    raw_args = (tc.get("function") or {}).get("arguments")
                    try:
                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_obj = raw_args
                    payload = {"name": name, "arguments": args_obj}
                    lines.append("<tool_call>")
                    lines.append(json.dumps(payload, ensure_ascii=False))
                    lines.append("</tool_call>")
        elif role == "tool":
            lines.append("user")
            lines.append("<tool_response>")
            lines.append(str(content))
            lines.append("</tool_response>")
        else:
            lines.append(role or "user")
            lines.append(content)
    return "\n".join(lines)


def normalize_answer(s):
    if s is None: s = ""
    elif isinstance(s, np.ndarray):
        try: s = " ".join(map(str, s.tolist()))
        except Exception: s = str(s)
    elif isinstance(s, list): s = " ".join(map(str, s))
    elif not isinstance(s, str): s = str(s)

    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    def lower(text): return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution_from_text(text: str):
    if not text: return None
    m = re.search(r"<\s*answer\s*>(.*?)</\s*answer\s*>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def cal_f1_score(prediction: str, ground_truth: str) -> float:
    at = normalize_answer(prediction).split()
    bt = normalize_answer(ground_truth).split()
    common = set(at) & set(bt)
    num_same = sum(min(at.count(w), bt.count(w)) for w in common)
    if num_same == 0: return 0.0
    precision = num_same / len(at) if at else 0.0
    recall = num_same / len(bt) if bt else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def semantic_similarity_pairs(answers: List[str], truths: List[str], max_workers: int = 8) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        corpus = [a or "" for a in answers] + [t or "" for t in truths]
        emb = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        n = len(answers)
        ans_emb = emb[:n]; gt_emb = emb[n:]

        def _cosine(u, v):
            nu = np.linalg.norm(u); nv = np.linalg.norm(v)
            if nu == 0 or nv == 0: return 0.0
            return float(np.dot(u, v) / (nu * nv))

        sims = [0.0] * n
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_cosine, ans_emb[i], gt_emb[i]): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                sims[i] = float(fut.result()) if fut.exception() is None else 0.0
        return sims

    except Exception:
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [a or "" for a in answers] + [t or "" for t in truths]
        vec = TfidfVectorizer(lowercase=True)
        X = vec.fit_transform(texts)
        n = len(answers); A = X[:n]; B = X[n:]
        sims = [0.0] * n

        def _row_cosine(i):
            ai = A.getrow(i); bi = B.getrow(i)
            num = (ai.multiply(bi)).sum()
            denom = (ai.multiply(ai)).sum() ** 0.5 * (bi.multiply(bi)).sum() ** 0.5
            return float(num / denom) if denom != 0 else 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_row_cosine, i): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                sims[i] = float(fut.result()) if fut.exception() is None else 0.0
        return sims


def extract_question_from_messages(messages: List[Dict[str, str]]) -> str:
    # 取最后一条 user 的首行；若以 "Question:" 开头则去前缀
    txt = ""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            txt = m.get("content") or ""
            break
    first_line = txt.split("\n", 1)[0]
    if first_line.strip().lower().startswith("question:"):
        return first_line.split(":", 1)[1].strip()
    return first_line.strip()


def pick_ground_truth(meta: Dict[str, Any]) -> str:
    # 优先 meta['ground_truth']，其次 meta['reward_model']['ground_truth']
    gt = meta.get("ground_truth")
    if gt is None and isinstance(meta.get("reward_model"), dict):
        gt = meta["reward_model"].get("ground_truth")
    # 统一转字符串
    if isinstance(gt, list) and gt:
        return str(gt[0])
    if isinstance(gt, np.ndarray) and gt.size > 0:
        try:
            return str(gt.tolist()[0])
        except Exception:
            return str(gt)
    return "" if gt is None else str(gt)


def make_results_triplet(records: List[Dict[str, Any]], results_dir: str, dataset_name: str) -> None:
    """
    依据 records（含 trace）生成并写出：
      - results.json
      - results_eval.json
      - eval.json
    """
    subset: List[Dict[str, Any]] = []
    results_eval: List[Dict[str, str]] = []

    for rec in records:
        messages = rec.get("trace") or rec.get("input") or []
        dialog_text = render_messages_plain(messages)
        final_text = rec.get("output") or ""
        question = extract_question_from_messages(messages if isinstance(messages, list) else [])
        ground_truth = pick_ground_truth(rec.get("meta", {}))

        # 从最终文本中提取 <answer>，若没有就从对话全文再尝试一次
        answer = extract_solution_from_text(final_text) or extract_solution_from_text(dialog_text) or ""

        subset.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "final_assistant_text": final_text,
            "dialog_text": dialog_text,
        })
        results_eval.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer
        })

    # 写盘
    out_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    results_eval_path = os.path.join(out_dir, "results_eval.json")
    with open(results_eval_path, "w", encoding="utf-8") as f:
        json.dump(results_eval, f, ensure_ascii=False, indent=2)

    # 计算指标
    gts = [normalize_answer(r["ground_truth"]) for r in results_eval]
    ans = [normalize_answer(r["answer"]) for r in results_eval]
    em = [float(a == b) for a, b in zip(ans, gts)]

    def _f1(a, b):
        at = a.split(); bt = b.split()
        common = set(at) & set(bt)
        num_same = sum(min(at.count(w), bt.count(w)) for w in common)
        if num_same == 0: return 0.0
        precision = num_same / len(at) if at else 0.0
        recall = num_same / len(bt) if bt else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    f1 = [_f1(a, b) for a, b in zip(ans, gts)]

    # 子串匹配：gold 在 ans 中
    sub = [1.0 if (r["ground_truth"] and normalize_answer(r["ground_truth"]) in a) else 0.0
           for a, r in zip(ans, results_eval)]

    sim = semantic_similarity_pairs(ans, gts, max_workers=8)

    metrics = {
        "count": len(results_eval),
        "exact_match": {"mean": float(np.mean(em) if em else 0.0), "sum": int(np.sum(em) if em else 0)},
        "substring_match": {"mean": float(np.mean(sub) if sub else 0.0), "sum": int(np.sum(sub) if sub else 0)},
        "f1": {"mean": float(np.mean(f1) if f1 else 0.0)},
        "semantic_similarity": {"mean": float(np.mean(sim) if sim else 0.0)}
    }
    eval_path = os.path.join(out_dir, "eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[SAVE] results.json       -> {results_path}")
    print(f"[SAVE] results_eval.json  -> {results_eval_path}")
    print(f"[SAVE] eval.json          -> {eval_path}")


def main() -> None:
    args = parse_args()
    use_colors = not args.no_color

    # 装载自定义 config
    config = default_config
    if args.config:
        try:
            config = load_custom_config(args.config)  # type: ignore[assignment]
            print(f"Loaded custom config from {args.config}")
        except Exception as e:
            print(f"Error loading custom config: {e}")
            print("Falling back to default config")

    # 覆盖配置
    TOOLS = args.tools
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    REPETITION_PENALTY = args.repetition_penalty
    INSTRUCTION_FOLLOWING = getattr(config, "INSTRUCTION_FOLLOWING", "")

    # OpenAI 客户端
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    # 工具环境
    tool_instances = []
    for tool_name in TOOLS:
        tool_instances.append(_default_tool(tool_name))
    env = NousToolEnv(tools=tool_instances, max_tool_response_length=MAX_TOKENS)
    tool_descs: List[Dict[str, Any]] = [t.tool_description for t in tool_instances]

    print(f"Running inference with model: {MODEL_NAME}")

    # --- 判定运行模式 ---
    single_or_file_mode = bool(args.input_file or args.question or args.questions)

    if single_or_file_mode:
        # 单条/多条/单文件（仅保存到 --output-file；不生成评测三件套）
        results: List[Dict[str, Any]] = []

        for item_id, messages, meta in yield_inputs(args, INSTRUCTION_FOLLOWING):
            print(f"\n===== Item {item_id} =====")

            out = run_inference_for_item(
                client=client,
                model_name=MODEL_NAME,
                tool_descs=tool_descs,
                env=env,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                stop_tokens=["</tool_call>"],
                base_messages=messages,
                use_colors=use_colors,
                max_steps=args.max_steps,
            )

            record: Dict[str, Any] = {
                "id": item_id,
                "model": MODEL_NAME,
                "input": messages,
                "output": out["output"],
                "trace": out["trace"],
                "meta": meta,
            }
            results.append(record)

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as wf:
                json.dump(results, wf, ensure_ascii=False, default=_as_py, indent=2)
            print(f"\nSaved {len(results)} record(s) to {args.output_file}")
        else:
            print(f"\nProcessed {len(results)} record(s). Use --output-file to save JSON.")
        return

    # ===== 默认目录批量模式 =====
    input_dir = args.input_dir            # 默认 eval/
    output_dir = args.output_dir          # 默认 chat-results/
    results_dir = args.results_dir        # 默认 results-5/
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        print(f"[WARN] No .parquet files found under: {input_dir}")
        return

    total_files = 0
    total_records = 0

    for p in parquet_files:
        total_files += 1
        dataset_name = infer_dataset_name_from_parquet(p)

        # 1) 跑推理
        records = process_parquet_file(
            path=p,
            input_field=args.input_field,                    # 默认 prompt
            instruction_following=INSTRUCTION_FOLLOWING,
            run_one_item_fn=run_inference_for_item,
            client=client,
            model_name=MODEL_NAME,
            tool_descs=tool_descs,
            env=env,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            use_colors=use_colors,
            max_steps=args.max_steps,
        )
        total_records += len(records)

        # 2) 保存全量对话记录
        chat_path = write_dataset_json(output_dir, dataset_name, records)
        print(f"[SAVE] {len(records)} record(s) -> {chat_path}")

        # 3) 生成并保存评测三件套（results.json / results_eval.json / eval.json）
        make_results_triplet(records, results_dir, dataset_name)

    print(f"\nDone. Processed {total_files} file(s), {total_records} record(s).")
    print(f"Dialog JSONs in: {output_dir}, metrics JSONs in: {results_dir}")


if __name__ == "__main__":
    main()
