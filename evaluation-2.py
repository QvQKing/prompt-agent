#!/usr/bin/env python3
"""
Async batch evaluation against a vLLM OpenAI-compatible endpoint.

增强版：完整记录多轮对话的“每次模型请求/响应”和“每次工具调用”的输入输出与时间戳。
输出：
- results.json：含每条样本的结构化 trace（内联）与可读串。
- traces/{id}.json：每条样本的完整轨迹单独存档。
- results_eval.json / eval.json：评测结果与汇总指标。
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


# ================== 消息/格式化 ==================

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
    期望 row['prompt'] 是形如 [{'role':'system'|'user'|'assistant','content':'...'}, ...] 的列表。
    如果不是，则退化为单轮 user。
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


def format_conversation_history(messages: List[Dict[str, str]]) -> str:
    """把消息历史格式化成人类可读文本，保留 system/user/assistant/tool"""
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            formatted_parts.append(f"=== SYSTEM ===\n{content}")
        elif role == "user":
            formatted_parts.append(f"=== USER ===\n{content}")
        elif role == "assistant":
            formatted_parts.append(f"=== ASSISTANT ===\n{content}")
        elif role == "tool":
            formatted_parts.append(f"=== TOOL RESULT ===\n{content}")
        else:
            formatted_parts.append(f"=== {role.upper()} ===\n{content}")
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                func_name = tc.get("function", {}).get("name", "unknown")
                func_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args_dict = json.loads(func_args)
                    formatted_args = json.dumps(args_dict, indent=2, ensure_ascii=False)
                except Exception:
                    formatted_args = func_args
                formatted_parts.append(f"=== TOOL CALL ===\nFunction: {func_name}\nArguments:\n{formatted_args}")
    return "\n\n".join(formatted_parts)


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


# ================== OpenAI 请求 ==================

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


# ================== 主对话循环（完整记录，含 rounds） ==================

async def run_one_dialog(client, model_name, tools_schema, env, initial_messages,
                         temperature, top_p, max_tokens, repetition_penalty, max_iterations=10):
    """
    返回:
      formatted_history(str),
      last_assistant_text(str),
      trace(list[dict])  # 每轮请求/响应/工具调用
      header(dict)       # run_meta：模型、工具、tools_schema 等（含 rounds / tool_calls_total / messages_final_count）
    """
    messages = deepcopy(initial_messages)
    last_assistant_text = ""
    trace: List[Dict[str, Any]] = []

    header = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "tools_schema": json_safe(tools_schema),  # 完整保存
        "tools_available": [t.get("function", {}).get("name") for t in tools_schema if isinstance(t, dict)],
        "t_start": time.time(),
    }

    for iteration in range(max_iterations):
        req_snapshot = {
            "messages": deepcopy(messages),
            "params": {
                "model": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty
            }
        }

        t_req = time.time()
        try:
            response = await get_model_response(
                client, model_name, messages, tools_schema,
                temperature, top_p, max_tokens, repetition_penalty
            )
            t_resp = time.time()
        except Exception as e:
            trace.append({
                "iteration": iteration,
                "t_request": t_req,
                "t_response": time.time(),
                "request": req_snapshot,
                "error": f"chat.completions.create failed: {e}"
            })
            break

        choice = response.choices[0]
        resp_msg = choice.message
        assistant_message = {
            "role": "assistant",
            "content": resp_msg.content or ""
        }
        model_usage = getattr(response, "usage", None)
        response_meta = {
            "response_id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": (model_usage.model_dump() if model_usage else None),
        }

        tool_runs = []
        if resp_msg.tool_calls:
            assistant_message["tool_calls"] = _serialize_tool_calls(resp_msg.tool_calls)
            messages.append(assistant_message)

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
                    t0, t1, tool_err = tool_exec["t_start"], tool_exec["t_end"], tool_exec["error"]
                else:
                    tool_content = "Tool not found"
                    t0, t1, tool_err = None, None, "Tool not found"

                messages.append({
                    "role": "tool",
                    "content": str(tool_content),
                    "tool_call_id": getattr(tc, "id", None)
                })

                tool_runs.append({
                    "tool_call_id": getattr(tc, "id", None),
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "raw_arguments": raw_args,
                    "parsed_arguments": json_safe(args_dict),
                    "tool_started_at": t0,
                    "tool_finished_at": t1,
                    "result": tool_content,
                    "error": tool_err
                })
        else:
            messages.append(assistant_message)
            last_assistant_text = assistant_message["content"] or ""

        trace.append({
            "iteration": iteration,
            "t_request": t_req,
            "t_response": t_resp,
            "latency_sec": (t_resp - t_req) if (t_resp and t_req) else None,
            "request": req_snapshot,
            "response": {"assistant": assistant_message},
            "response_meta": response_meta,
            "tool_runs": json_safe(tool_runs)
        })

        if not resp_msg.tool_calls:
            break  # 终止

    # === 在这里统计 rounds / tool_calls_total / messages_final_count ===
    header["t_end"] = time.time()
    header["total_latency_sec"] = header["t_end"] - header["t_start"]
    header["rounds"] = len(trace)  # 交互轮数：每产生一次 assistant 回复就+1
    header["tool_calls_total"] = int(sum(len(x.get("tool_runs", []) or []) for x in trace))
    header["messages_final_count"] = len(messages)

    formatted = format_conversation_history(messages)
    return formatted, last_assistant_text, trace, header


# ================== 指标计算与抽取 ==================

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

async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty, rid, save_trace_file, trace_dir, max_iterations):
    initial_messages = build_initial_messages(row)
    question = extract_question_from_prompt_list(row.get("prompt", []))
    ground_truth = row.get("reward_model", {}).get("ground_truth", "")

    try:
        formatted_history, final_text, trace, header = await run_one_dialog(
            client, model_name, tools_schema, env,
            initial_messages, temp, top_p, max_tokens, rep_penalty, max_iterations=max_iterations
        )
    except Exception as e:
        formatted_history, final_text, trace, header = f"[ERROR] {e}", "", [], {"error": str(e)}

    model_output = {
        "header": json_safe(header),
        "formatted": formatted_history,
        "final_assistant_text": final_text,
        "trace": json_safe(trace)
    }

    trace_path = None
    if save_trace_file:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"{rid}.json")
        with open(trace_path, "w", encoding="utf-8") as tf:
            json.dump(json_safe(model_output), tf, ensure_ascii=False, indent=2)
        model_output["trace_file"] = trace_path

    return {"id": rid, "question": question, "ground_truth": ground_truth, "model_output": model_output}


# ================== 单文件主逻辑 ==================

async def main_single(parquet_file, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    if args.save_trace_file:
        os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    tool_objs = []
    for t in args.tools:
        try:
            tool_objs.append(_default_tool(t))
        except Exception:
            pass
    env = NousToolEnv(tools=tool_objs, max_tool_response_length=args.max_tokens)
    tools_schema = [t.tool_description for t in tool_objs]

    df = pd.read_parquet(parquet_file)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_worker(r):
        async with sem:
            return await worker(
                r, client, args.model, tools_schema, env,
                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty,
                r["id"],
                args.save_trace_file,
                os.path.join(out_dir, "traces"),
                args.max_iterations
            )

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {os.path.basename(parquet_file)}"):
        results.append(await fut)

    results_file = os.path.join(out_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)

    results_eval_file = os.path.join(out_dir, "results_eval.json")
    eval_file = os.path.join(out_dir, "eval.json")

    results_eval = []
    for rec in results:
        mo = rec.get("model_output", {})
        formatted = mo.get("formatted", "") if isinstance(mo, dict) else str(mo)
        final_text = mo.get("final_assistant_text", "") if isinstance(mo, dict) else ""
        answer = extract_solution(final_text, formatted)

        record_id = rec.get('id', 'unknown')
        if answer is None:
            text = (final_text or formatted or "")
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
