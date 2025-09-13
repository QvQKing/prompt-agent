#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标：生成 transcripts.json（纯字符串数组），每个元素是一条样本从头到尾的“对话全文”，
只包含以下段落/块，不添加任何键值字段或调试头：
- 角色行：system / user / assistant
- 内容行：紧随其后
- assistant 的 function-calling 工具调用：<tool_call>{...}</tool_call>
- 工具返回（role=tool）以用户块呈现：user + <tool_response>...</tool_response>

同时保留 results.json / results_eval.json / eval.json 以便评测。
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
from copy import deepcopy
from typing import List, Dict, Any

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AsyncOpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config


# ================== Utils ==================

def json_safe(obj):
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


def _cosine(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0: return 0.0
    return float(np.dot(u, v) / (nu * nv))


def semantic_similarity_pairs(answers, truths, max_workers: int = 8) -> list:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        corpus = [a if a is not None else "" for a in answers] + [t if t is not None else "" for t in truths]
        emb = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)

        n = len(answers)
        ans_emb = emb[:n]; gt_emb = emb[n:]

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
        n = len(answers); A = X[:n]; B = X[n:]
        sims = [0.0] * n

        def _row_cosine(i):
            ai = A.getrow(i); bi = B.getrow(i)
            num = (ai.multiply(bi)).sum()
            denom = (ai.multiply(ai)).sum() ** 0.5 * (bi.multiply(bi)).sum() ** 0.5
            return float(num / denom) if denom != 0 else 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_row_cosine, i): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                sims[i] = float(fut.result()) if fut.exception() is None else 0.0
        return sims


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="batch", help="single | batch")
    p.add_argument("--parquet-file", type=str, default='hotpot-512-test.parquet')
    p.add_argument("--eval-dir", type=str, default="./eval")
    p.add_argument("--results-dir", type=str, default="./results-5")
    # 模型参数
    p.add_argument('--tools', type=str, nargs='*', default=default_config.TOOLS)
    p.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY)
    p.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE)
    p.add_argument('--model', type=str, default=default_config.MODEL_NAME)
    p.add_argument('--temperature', type=float, default=default_config.TEMPERATURE)
    p.add_argument('--top-p', type=float, default=default_config.TOP_P)
    p.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS)
    p.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY)
    # 控制参数
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=10)
    return p.parse_args()


# ================== 消息构建 & 渲染为“无字段对话全文” ==================

def extract_question_from_prompt_list(prompt_list):
    if not prompt_list: return ""
    content = prompt_list[0].get("content", "")
    if isinstance(content, str) and content.startswith("Question:"):
        first_line = content.split("\n", 1)[0]
        return first_line.replace("Question:", "").strip()
    return content.strip() if isinstance(content, str) else str(content)


def build_initial_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
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
    if not resp_tool_calls: return out
    for tc in resp_tool_calls:
        out.append({
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", None),
            "function": {
                "name": getattr(getattr(tc, "function", None), "name", None),
                "arguments": getattr(getattr(tc, "function", None), "arguments", None),
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
                for tc in msg["tool_calls"]:
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


# ================== OpenAI & Tool ==================

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
        stop=["</tool_call>"],  # 便于保留 <tool_call> 块
    )


async def execute_tool_safely(tool_fn, args_dict):
    try:
        if hasattr(tool_fn, '_execute_async'):
            result = await tool_fn._execute_async(args_dict)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool_fn.execute, args_dict)
        return {"content": result.get("content", str(result))}
    except Exception as e:
        return {"content": f"Tool execution error: {e}"}


# ================== 对话：只保留一次性的线性对话全文（无重复） ==================

async def run_dialog_return_messages(client, model_name, tools_schema, env, initial_messages,
                                     temperature, top_p, max_tokens, repetition_penalty,
                                     max_iterations=10):
    """
    返回：
      messages: 最终的完整消息序列（线性，无重复快照）
      final_text: 最后一条 assistant 文本
    """
    messages = deepcopy(initial_messages)
    final_text = ""

    for _ in range(max_iterations):
        # 模型回复
        try:
            response = await get_model_response(
                client, model_name, messages, tools_schema,
                temperature, top_p, max_tokens, repetition_penalty
            )
        except Exception as e:
            messages.append({"role": "assistant", "content": f"[ERROR] chat.completions.create failed: {e}"})
            break

        choice = response.choices[0]
        resp_msg = choice.message

        assistant_message = {
            "role": "assistant",
            "content": resp_msg.content or ""
        }

        # 有工具调用：记录 assistant（含 tool_calls），随后注入 tool 响应，再进入下一轮
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
                else:
                    tool_content = "Tool not found"

                # 注入 tool 结果（role=tool），渲染时会变为 user/<tool_response>
                messages.append({
                    "role": "tool",
                    "content": str(tool_content),
                    "tool_call_id": getattr(tc, "id", None)
                })
            # 继续下一轮
            continue

        # 无工具：终止
        messages.append(assistant_message)
        final_text = assistant_message["content"] or ""
        break

    return messages, final_text


# ================== 评测 & 提取 ==================

def normalize_answer(s):
    if s is None: s = ""
    elif isinstance(s, np.ndarray):
        try: s = " ".join(map(str, s.tolist()))
        except Exception: s = str(s)
    elif isinstance(s, list): s = " ".join(map(str, s))
    elif not isinstance(s, str): s = str(s)
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution_from_text(text: str):
    if not text: return None
    m = re.search(r"<\s*answer\s*>(.*?)</\s*answer\s*>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


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


# ================== Worker / 主流程 ==================

async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty):
    initial_messages = build_initial_messages(row)
    question = extract_question_from_prompt_list(row.get("prompt", []))
    ground_truth = row.get("reward_model", {}).get("ground_truth", "")

    try:
        messages, final_text = await run_dialog_return_messages(
            client, model_name, tools_schema, env,
            initial_messages, temp, top_p, max_tokens, rep_penalty,
            max_iterations=10
        )
    except Exception as e:
        messages, final_text = initial_messages + [{"role": "assistant", "content": f"[ERROR] {e}"}], ""

    # —— 无重复，仅一次性“对话全文” ——
    dialog_text = render_messages_plain(messages)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_assistant_text": final_text,
        "dialog_text": dialog_text
    }


async def main_single(parquet_file, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    # 工具
    tool_objs = []
    for t in args.tools:
        try: tool_objs.append(_default_tool(t))
        except Exception: pass
    env = NousToolEnv(tools=tool_objs, max_tool_response_length=args.max_tokens)
    tools_schema = [t.tool_description for t in tool_objs]

    # 数据
    df = pd.read_parquet(parquet_file)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]

    # 并发
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def sem_worker(r):
        async with sem:
            return await worker(
                r, client, args.model, tools_schema, env,
                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty
            )

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {os.path.basename(parquet_file)}"):
        results.append(await fut)

    # 1) transcripts.json —— 纯字符串数组（每个元素是一整段“无重复”的对话全文）
    transcripts = [rec["dialog_text"] for rec in results]
    transcripts_file = os.path.join(out_dir, "transcripts.json")
    with open(transcripts_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(transcripts), f, ensure_ascii=False, indent=2)

    # 2) results.json —— 供评测用的最小结构
    results_file = os.path.join(out_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)

    # 3) 评测
    results_eval_file = os.path.join(out_dir, "results_eval.json")
    eval_file = os.path.join(out_dir, "eval.json")

    results_eval = []
    for rec in results:
        final_text = rec.get("final_assistant_text", "") or ""
        answer = extract_solution_from_text(final_text) or extract_solution_from_text(rec.get("dialog_text", ""))

        results_eval.append({
            "question": rec.get("question", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "answer": answer or ""
        })

    with open(results_eval_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results_eval), f, ensure_ascii=False, indent=2)

    gts = [normalize_answer(r["ground_truth"]) for r in results_eval]
    ans = [normalize_answer(r["answer"]) for r in results_eval]
    em = [float(a == b) for a, b in zip(ans, gts)]
    sub = [1.0 if (r["ground_truth"] and normalize_answer(r["ground_truth"]) in a) else 0.0 for a, r in zip(ans, results_eval)]

    def _f1(a, b):
        at = a.split(); bt = b.split()
        common = set(at) & set(bt)
        num_same = sum(min(at.count(w), bt.count(w)) for w in common)
        if num_same == 0: return 0.0
        precision = num_same / len(at) if at else 0
        recall = num_same / len(bt) if bt else 0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    f1 = [_f1(a, b) for a, b in zip(ans, gts)]
    sim = semantic_similarity_pairs(ans, gts, max_workers=8)

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
    print(f"[INFO] transcripts.json written to: {transcripts_file}")


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


def main():
    args = parse_args()
    if args.mode == "single":
        if not args.parquet_file:
            print("[ERROR] --parquet-file 必须指定"); sys.exit(1)
        out_dir = os.path.join(args.results_dir, os.path.splitext(os.path.basename(args.parquet_file))[0])
        asyncio.run(main_single(args.parquet_file, out_dir, args))
    else:
        asyncio.run(main_batch(args))


if __name__ == "__main__":
    main()
