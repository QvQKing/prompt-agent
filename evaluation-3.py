#!/usr/bin/env python3
"""
Async batch evaluation against a vLLM OpenAI-compatible endpoint.

增强点：
1) 完整记录多轮对话与工具调用；
2) 额外生成 transcripts.json：每条样本输出一段“纯文本对话串”，
   采用你指定的格式（system/user/assistant + <tool_call>/<tool_response> 文本块）。

其它输出：
- results.json / results_eval.json / eval.json 与原先一致。
"""

import argparse
import json
import os
import sys
import pandas as pd
import asyncio
import re
import string
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import time
from typing import List, Dict, Any

from openai import AsyncOpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

import numpy as np


# ================== JSON 安全转换 ==================

def json_safe(obj):
    """递归地把对象转换成可 JSON 序列化的 Python 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


# ================== 语义相似度工具 ==================

def _cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def semantic_similarity_pairs(answers, truths, max_workers: int = 8) -> list:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        corpus = [a if a is not None else "" for a in answers] + [t if t is not None else "" for t in truths]
        emb = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)

        n = len(answers)
        ans_emb = emb[:n]
        gt_emb = emb[n:]

        sims = [0.0] * n
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_cosine, ans_emb[i], gt_emb[i]): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                sims[i] = float(fut.result()) if fut.exception() is None else 0.0
        return sims

    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [a if a is not None else "" for a in answers] + [t if t is not None else "" for t in truths]
        vec = TfidfVectorizer(lowercase=True)
        X = vec.fit_transform(texts)
        n = len(answers)
        A = X[:n]
        B = X[n:]
        sims = [0.0] * n

        def _row_cosine(i):
            ai = A.getrow(i)
            bi = B.getrow(i)
            num = (ai.multiply(bi)).sum()
            denom = (ai.multiply(ai)).sum() ** 0.5 * (bi.multiply(bi)).sum() ** 0.5
            return float(num / denom) if denom != 0 else 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_row_cosine, i): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                sims[i] = float(fut.result()) if fut.exception() is None else 0.0
        return sims


# ================== 参数 ==================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="batch",
                        help="运行模式：single=单文件，batch=批量处理 eval 目录")
    parser.add_argument("--parquet-file", type=str, default='hotpot-512-test.parquet',
                        help="单文件模式下输入 parquet 文件路径")
    parser.add_argument("--eval-dir", type=str, default="./eval",
                        help="批量模式下输入目录（包含 parquet 文件）")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="批量模式下结果根目录")

    # 模型参数
    parser.add_argument('--tools', type=str, nargs='*', default=default_config.TOOLS)
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY)
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE)
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME)
    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE)
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P)
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS)
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY)

    # 控制参数
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sim-workers", type=int, default=8)

    # 记录控制
    parser.add_argument("--save-trace-file", action="store_true", default=True,
                        help="是否把每条样本的 trace 单独保存到 traces/{id}.json")
    parser.add_argument("--max-iterations", type=int, default=10, help="每条样本最大迭代轮数")

    return parser.parse_args()


# ================== 消息与格式化 ==================

def extract_question_from_prompt_list(prompt_list):
    if not prompt_list:
        return ""
    content = prompt_list[0].get("content", "")
    if isinstance(content, str) and content.startswith("Question:"):
        first_line = content.split("\n", 1)[0]
        return first_line.replace("Question:", "").strip()
    return content.strip() if isinstance(content, str) else str(content)


def build_initial_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从 parquet 的 row 中恢复完整初始多轮消息。
    期望 row['prompt'] 是 [{'role':'system'|'user'|'assistant','content':'...'}, ...]。
    否则退化为单轮 user。
    """
    prompt_list = row.get("prompt", [])
    messages = []
    if isinstance(prompt_list, list) and prompt_list:
        for m in prompt_list:
            role = m.get("role") or m.get("speaker") or "user"
            content = m.get("content") or m.get("text") or ""
            messages.append({"role": role, "content": content})
    else:
        messages.append({"role": "user", "content": str(prompt_list)})
    return messages


def _serialize_tool_calls(resp_tool_calls) -> List[Dict[str, Any]]:
    out = []
    if not resp_tool_calls:
        return out
    for tc in resp_tool_calls:
        item = {
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", None),
            "function": {
                "name": getattr(getattr(tc, "function", None), "name", None),
                "arguments": getattr(getattr(tc, "function", None), "arguments", None),
            }
        }
        out.append(item)
    return out


def render_plain_transcript(messages: List[Dict[str, Any]]) -> str:
    """
    输出严格的纯文本对话串：
    - 行首是角色小写：system / user / assistant（tool 结果渲染为 user + <tool_response> 块）
    - assistant 如有 tool_calls，则追加 <tool_call>JSON</tool_call>
    - 不做任何结构化包装
    """
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role in ("system", "user", "assistant"):
            lines.append(role)
            lines.append(content)
            # 如果 assistant 含 function-calling 工具调用，按文本块补充 <tool_call>
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = (tc.get("function") or {}).get("name")
                    raw_args = (tc.get("function") or {}).get("arguments")
                    # 尽量把 arguments 解析为对象，否则按原字符串输出
                    try:
                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_obj = raw_args
                    payload = {"name": name, "arguments": args_obj}
                    lines.append("<tool_call>")
                    lines.append(json.dumps(payload, ensure_ascii=False))
                    lines.append("</tool_call>")
        elif role == "tool":
            # 工具结果作为 user 的 <tool_response> 文本块输出，贴合你的示例
            lines.append("user")
            lines.append("<tool_response>")
            lines.append(str(content))
            lines.append("</tool_response>")
        else:
            # 未知角色，按原样输出
            lines.append(role or "user")
            lines.append(content)
    return "\n".join(lines)


# ================== OpenAI 请求与工具执行 ==================

async def get_model_response(client, model_name, messages, tools, temperature, top_p, max_tokens, repetition_penalty):
    return await client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"repetition_penalty": repetition_penalty},
        stop=["</tool_call>"],  # 兼容部分走 XML 协议的模型
    )


async def execute_tool_safely(tool_fn, args_dict):
    try:
        t0 = time.time()
        if hasattr(tool_fn, '_execute_async'):
            result = await tool_fn._execute_async(args_dict)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool_fn.execute, args_dict)
        t1 = time.time()
        return {"content": result.get("content", str(result)), "t_start": t0, "t_end": t1, "error": None}
    except Exception as e:
        t1 = time.time()
        return {"content": f"Tool execution error: {e}", "t_start": None, "t_end": t1, "error": str(e)}


# ================== 主对话循环（返回 messages 以生成纯文本） ==================

async def run_one_dialog(client, model_name, tools_schema, env, initial_messages,
                         temperature, top_p, max_tokens, repetition_penalty, max_iterations=10):
    """
    返回:
      messages(list[dict])   # 最终完整消息序列，用于转成纯文本对话串
      last_assistant_text(str)
      rounds(int)            # 轮数统计（assistant 回复次数）
    """
    messages = deepcopy(initial_messages)
    last_assistant_text = ""
    rounds = 0

    for iteration in range(max_iterations):
        try:
            response = await get_model_response(
                client, model_name, messages, tools_schema,
                temperature, top_p, max_tokens, repetition_penalty
            )
        except Exception as e:
            # 在消息序列里追加错误提示（assistant）
            messages.append({"role": "assistant", "content": f"[ERROR] chat.completions.create failed: {e}"})
            break

        choice = response.choices[0]
        resp_msg = choice.message
        assistant_message = {
            "role": "assistant",
            "content": resp_msg.content or ""
        }
        rounds += 1  # 每产生一次 assistant 就计一轮

        # 如果触发 function-calling
        if resp_msg.tool_calls:
            assistant_message["tool_calls"] = _serialize_tool_calls(resp_msg.tool_calls)
            messages.append(assistant_message)

            # 逐个执行工具，把工具返回作为 role=tool 的消息注入
            for tc in resp_msg.tool_calls:
                raw_args = getattr(getattr(tc, "function", None), "arguments", None)
                try:
                    args_dict = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args_dict = {}

                tool_fn = env.tool_map.get(tc.function.name) if hasattr(tc, "function") else None
                if tool_fn:
                    tool_exec = await execute_tool_safely(tool_fn, args_dict)
                    tool_content = tool_exec["content"]
                else:
                    tool_content = "Tool not found"

                messages.append({
                    "role": "tool",
                    "content": str(tool_content),
                    "tool_call_id": getattr(tc, "id", None)
                })

            # 工具后进入下一轮
            continue
        else:
            # 无工具调用则终止
            messages.append(assistant_message)
            last_assistant_text = assistant_message["content"] or ""
            break

    return messages, last_assistant_text, rounds


# ================== 指标与抽取（保持不变） ==================

def normalize_answer(s):
    if s is None:
        s = ""
    elif isinstance(s, np.ndarray):
        try:
            s = " ".join(map(str, s.tolist()))
        except Exception:
            s = str(s)
    elif isinstance(s, list):
        s = " ".join(map(str, s))
    elif not isinstance(s, str):
        s = str(s)

    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def cal_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def subem_check(prediction: str, golden_answers) -> float:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    npred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(g) in npred for g in golden_answers) else 0.0


def extract_solution_from_formatted(solution_str: str):
    if not solution_str:
        return None
    blocks = re.split(r"^===\s*ASSISTANT\s*===\s*$", solution_str, flags=re.MULTILINE)
    assistant_blocks = [b.strip() for b in blocks[1:] if b.strip()]
    pattern = re.compile(r"<\s*answer\s*>(.*?)</\s*answer\s*>", re.IGNORECASE | re.DOTALL)
    for block in reversed(assistant_blocks):
        m = pattern.search(block)
        if m:
            return m.group(1).strip()
    return None


def extract_solution(final_assistant_text: str, formatted_fallback: str):
    pattern = re.compile(r"<\s*answer\s*>(.*?)</\s*answer\s*>", re.IGNORECASE | re.DOTALL)
    if final_assistant_text:
        m = pattern.search(final_assistant_text)
        if m:
            return m.group(1).strip()
    return extract_solution_from_formatted(formatted_fallback)


# ================== 单条任务 ==================

async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty, rid):
    initial_messages = build_initial_messages(row)
    question = extract_question_from_prompt_list(row.get("prompt", []))
    ground_truth = row.get("reward_model", {}).get("ground_truth", "")

    try:
        messages, final_text, rounds = await run_one_dialog(
            client, model_name, tools_schema, env,
            initial_messages, temp, top_p, max_tokens, rep_penalty, max_iterations=10
        )
    except Exception as e:
        messages, final_text, rounds = initial_messages + [{"role": "assistant", "content": f"[ERROR] {e}"}], "", 0

    # 生成纯文本对话串
    dialog_text = render_plain_transcript(messages)

    # 为了兼容原有评测逻辑，这里仍返回结构，评测时会从最终 assistant 文本抽取 <answer>
    return {
        "id": rid,
        "question": question,
        "ground_truth": ground_truth,
        "rounds": rounds,
        "final_assistant_text": final_text,
        "dialog_text": dialog_text  # 提供给 main_single 写入 transcripts.json
    }


# ================== 单文件主逻辑 ==================

async def main_single(parquet_file, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    # 准备工具
    tool_objs = []
    for t in args.tools:
        try:
            tool_objs.append(_default_tool(t))
        except Exception:
            pass
    env = NousToolEnv(tools=tool_objs, max_tool_response_length=args.max_tokens)
    tools_schema = [t.tool_description for t in tool_objs]

    # 读取数据
    df = pd.read_parquet(parquet_file)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]

    # 并发评测
    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_worker(r):
        async with sem:
            return await worker(
                r, client, args.model, tools_schema, env,
                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty, r["id"]
            )

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {os.path.basename(parquet_file)}"):
        results.append(await fut)

    # 1) transcripts.json：仅包含 id 与纯文本对话串 dialog
    transcripts = [{"id": rec["id"], "dialog": rec["dialog_text"]} for rec in results]
    transcripts_file = os.path.join(out_dir, "transcripts.json")
    with open(transcripts_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(transcripts), f, ensure_ascii=False, indent=2)

    # 2) results.json：保留评测所需的最小字段
    results_file = os.path.join(out_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)

    # 3) 评测抽取与指标
    results_eval_file = os.path.join(out_dir, "results_eval.json")
    eval_file = os.path.join(out_dir, "eval.json")

    results_eval = []
    for rec in results:
        final_text = rec.get("final_assistant_text", "") or ""
        # 回退：如果 final_text 为空，你也可以从 rec["dialog_text"] 里尝试匹配 <answer>
        answer = extract_solution(final_text, rec.get("dialog_text", ""))

        record_id = rec.get('id', 'unknown')
        if answer is None:
            text = (final_text or rec.get("dialog_text", "") or "")
            if "<answer>" in text.lower():
                print(f"[ERROR] Failed to extract answer from record {record_id}")
                a_start = text.lower().find("<answer>")
                a_end = text.lower().find("</answer>")
                if a_start != -1 and a_end != -1:
                    manual_extract = text[a_start+8:a_end].strip()
                    print(f"[ERROR] Found tags at positions: start={a_start}, end={a_end}")
                    print(f"[ERROR] Manual extraction would give: '{manual_extract}'")
                else:
                    print(f"[ERROR] Could not find both opening and closing tags")
            else:
                print(f"[INFO] No answer tags found in record {record_id}")
        else:
            print(f"[SUCCESS] Extracted answer for record {record_id}: '{answer}'")

        results_eval.append({
            "id": record_id,
            "question": rec.get("question", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "answer": answer or ""
        })

    with open(results_eval_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results_eval), f, ensure_ascii=False, indent=2)

    gts = [normalize_answer(r["ground_truth"]) for r in results_eval]
    ans = [normalize_answer(r["answer"]) for r in results_eval]
    em = [exact_match_score(a, b) for a, b in zip(ans, gts)]
    sub = [subem_check(a, b) for a, b in zip(ans, gts)]
    f1 = [cal_f1_score(a, b) for a, b in zip(ans, gts)]
    sim = semantic_similarity_pairs(ans, gts, max_workers=args.sim_workers)

    metrics = {
        "count": len(results_eval),
        "exact_match": {"mean": float(np.mean(em) if em else 0.0), "sum": int(np.sum(em) if em else 0)},
        "substring_match": {"mean": float(np.mean(sub) if sub else 0.0), "sum": int(np.sum(sub) if sub else 0)},
        "f1": {"mean": float(np.mean(f1) if f1 else 0.0)},
        "semantic_similarity": {"mean": float(np.mean(sim) if sim else 0.0)}
    }
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[INFO] Wrote transcripts to: {transcripts_file}")


# ================== Batch 模式 ==================

async def main_batch(args):
    parquet_files = sorted(glob.glob(os.path.join(args.eval_dir, "*.parquet")))
    if not parquet_files:
        print(f"[ERROR] No parquet found in {args.eval_dir}")
        return
    for pf in parquet_files:
        dataset_name = os.path.splitext(os.path.basename(pf))[0]
        out_dir = os.path.join(args.results_dir, dataset_name)
        print(f"\n[INFO] ==== 开始评测 {dataset_name} ====")
        await main_single(pf, out_dir, args)
        print(f"[INFO] ==== 完成 {dataset_name}, 结果存入 {out_dir} ====\n")


# ================== 入口 ==================

def main():
    args = parse_args()
    if args.mode == "single":
        if not args.parquet_file:
            print("[ERROR] --parquet-file 必须指定")
            sys.exit(1)
        out_dir = os.path.join(args.results_dir, os.path.splitext(os.path.basename(args.parquet_file))[0])
        asyncio.run(main_single(args.parquet_file, out_dir, args))
    else:
        asyncio.run(main_batch(args))


if __name__ == "__main__":
    main()
