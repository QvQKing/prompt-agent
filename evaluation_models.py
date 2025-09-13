#!/usr/bin/env python3
"""
Async batch evaluation against a vLLM OpenAI-compatible endpoint.

功能：
1. --mode single : 评测一个 parquet 文件
   - 输出 results.json, results_eval.json, eval.json
2. --mode batch : 批量评测 eval 目录下的所有 parquet 文件
   - 每个数据集的结果存到 results/{model_name}/{dataset_name}/ 下
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

from openai import AsyncOpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

import numpy as np


# ================== 从 vllm.sh 文件读取 MODEL_NAME ==================

def get_model_name_from_script(script_path="vllm_serve.sh"):
    # 检查文件是否存在
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} 文件不存在！")
    
    # 读取脚本内容
    with open(script_path, "r") as file:
        content = file.read()

    # 使用正则表达式提取 MODEL_NAME 变量的值
    match = re.search(r'export\s+MODEL_NAME=["\']([^"\']+)["\']', content)
    
    if match:
        model_name = match.group(1)
        
        # 去掉路径前缀（例如 "merge_model/"）
        model_name = model_name.replace("merge_model/", "").strip()

        print(f"从 vllm.sh 读取到的模型名称: {model_name}")  # 打印模型名称
        return model_name
    else:
        raise ValueError("在 vllm.sh 文件中未找到 MODEL_NAME 变量！")


# ================== JSON 安全转换 ==================

def json_safe(obj):
    """递归地把对象转换成可 JSON 序列化的 Python 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
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

    parser.add_argument("--mode", choices=["single", "batch"], default="single",
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

    return parser.parse_args()


# ================== 评测工具函数 ==================

def extract_question_from_prompt_list(prompt_list):
    if not prompt_list:
        return ""
    content = prompt_list[0].get("content", "")
    if content.startswith("Question:"):
        first_line = content.split("\n", 1)[0]
        return first_line.replace("Question:", "").strip()
    return content.strip()


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
        stop=["</tool_call>"],
    )


async def execute_tool_safely(tool_fn, args_dict):
    try:
        if hasattr(tool_fn, '_execute_async'):
            return await tool_fn._execute_async(args_dict)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, tool_fn.execute, args_dict)
    except Exception as e:
        return {"content": f"Tool execution error: {e}"}


async def run_one_dialog(client, model_name, tools_schema, env, user_prompt_msg,
                         temperature, top_p, max_tokens, repetition_penalty):
    messages = [{"role": "user", "content": user_prompt_msg}]
    final_text_parts = []
    for _ in range(10):
        response = await get_model_response(client, model_name, messages, tools_schema,
                                            temperature, top_p, max_tokens, repetition_penalty)
        resp_msg = response.choices[0].message
        if resp_msg.content:
            final_text_parts.append(resp_msg.content)
        if not resp_msg.tool_calls:
            break
        for tc in resp_msg.tool_calls:
            try:
                args_dict = json.loads(tc.function.arguments)
            except Exception:
                args_dict = {}
            tool_fn = env.tool_map.get(tc.function.name)
            tool_output = await execute_tool_safely(tool_fn, args_dict) if tool_fn else {"content": "Tool not found"}
            messages.append({"role": "tool", "content": str(tool_output), "tool_call_id": tc.id})
    return "\n".join(final_text_parts)


async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty, rid):
    prompt_list = row.get("prompt", [])
    user_prompt = prompt_list[0]["content"] if prompt_list else ""
    question = extract_question_from_prompt_list(prompt_list)
    ground_truth = row.get("reward_model", {}).get("ground_truth", "")

    try:
        model_output = await run_one_dialog(client, model_name, tools_schema, env,
                                            user_prompt, temp, top_p, max_tokens, rep_penalty)
    except Exception as e:
        model_output = f"[ERROR] {e}"

    return {"id": rid, "question": question, "ground_truth": ground_truth, "model_output": model_output}


# ================== 指标计算 ==================

def normalize_answer(s):
    """保证输入是字符串"""
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


def extract_solution(solution_str: str):
    if not solution_str: return None
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    return match.group(1).strip() if match else None


# ================== 单文件主逻辑 ==================

async def main_single(parquet_file, model_name, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
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
    if args.limit: rows = rows[:args.limit]

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_worker(r):
        async with sem:
            return await worker(r, client, args.model, tools_schema, env,
                                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty, r["id"])

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {os.path.basename(parquet_file)}"):
        results.append(await fut)

    results_file = os.path.join(out_dir, "results.json")
    results_eval_file = os.path.join(out_dir, "results_eval.json")
    eval_file = os.path.join(out_dir, "eval.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2)

    results_eval = []
    for rec in results:
        answer = extract_solution(rec.get("model_output", "")) or ""
        results_eval.append({
            "id": rec["id"],
            "question": rec["question"],
            "ground_truth": rec["ground_truth"],
            "answer": answer
        })
    with open(results_eval_file, "w", encoding="utf-8") as f:
        json.dump(json_safe(results_eval), f, ensure_ascii=False, indent=2)

    # --- 计算指标 ---
    gts = [normalize_answer(r["ground_truth"]) for r in results_eval]
    ans = [normalize_answer(r["answer"]) for r in results_eval]

    em = [exact_match_score(a, b) for a, b in zip(ans, gts)]
    sub = [subem_check(a, b) for a, b in zip(ans, gts)]
    f1 = [cal_f1_score(a, b) for a, b in zip(ans, gts)]
    sim = semantic_similarity_pairs(ans, gts, max_workers=args.sim_workers)

    metrics = {
        "count": len(results_eval),
        "exact_match": {"mean": float(np.mean(em)), "sum": int(np.sum(em))},
        "substring_match": {"mean": float(np.mean(sub)), "sum": int(np.sum(sub))},
        "f1": {"mean": float(np.mean(f1))},
        "semantic_similarity": {"mean": float(np.mean(sim))}
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
    
    model_name = get_model_name_from_script()  # 从 vllm.sh 中读取模型名称
    for pf in parquet_files:
        dataset_name = os.path.splitext(os.path.basename(pf))[0]
        
        # 生成带模型名称的文件夹
        out_dir = os.path.join(args.results_dir, model_name, dataset_name)
        print(f"\n[INFO] ==== 开始评测 {dataset_name} ====")
        await main_single(pf, model_name, out_dir, args)
        print(f"[INFO] ==== 完成 {dataset_name}, 结果存入 {out_dir} ====\n")


# ================== 入口 ==================

def main():
    args = parse_args()
    if args.mode == "singal":
        if not args.parquet_file:
            print("[ERROR] --parquet-file 必须指定")
            sys.exit(1)
        model_name = get_model_name_from_script()  # 从 vllm.sh 中读取模型名称
        out_dir = os.path.join(args.results_dir, model_name, os.path.splitext(os.path.basename(args.parquet_file))[0])
        asyncio.run(main_single(args.parquet_file, model_name, out_dir, args))
    else:
        asyncio.run(main_batch(args))


if __name__ == "__main__":
    main()
