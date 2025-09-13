#!/usr/bin/env python3
"""
Async batch evaluation against a vLLM OpenAI-compatible endpoint using test.parquet.
- Reads prompts & ground truths from test.parquet
- Calls the model concurrently (asyncio)
- Iteratively resolves tool calls until none remain
- Saves results to JSON

新增：
(1) 在 results.json 中保存 id
(2) 生成 results_eval.json，包含 id / question / ground_truth / answer（从 <answer>...</answer> 中提取）
(3) 计算 Exact Match、子串匹配、F1（与 Bytedance 代码一致）与语义相似度（多线程），保存 eval.json 并打印
"""

import argparse
import json
import os
import sys
import pandas as pd
import asyncio
import inspect
import re
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AsyncOpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

# ========== 语义相似度计算工具（与指标对齐无关，保持不变，仅在聚合阶段被调用） ==========
def _cosine(u, v):
    import numpy as np
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def semantic_similarity_pairs(answers, truths, max_workers: int = 8) -> list:
    """
    返回与 (answers[i], truths[i]) 成对的相似度列表。
    优先使用 sentence_transformers；若不可用则使用 TF-IDF 余弦相似度。
    逐对相似度计算使用多线程。
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        corpus = [a if a is not None else "" for a in answers] + [t if t is not None else "" for t in truths]
        emb = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)

        n = len(answers)
        ans_emb = emb[:n]
        gt_emb  = emb[n:]

        sims = [0.0] * n
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_cosine, ans_emb[i], gt_emb[i]): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    sims[i] = float(fut.result())
                except Exception:
                    sims[i] = 0.0
        return sims

    except Exception:
        # 回退：TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [a if a is not None else "" for a in answers] + [t if t is not None else "" for t in truths]
        vec = TfidfVectorizer(lowercase=True)
        X = vec.fit_transform(texts)
        n = len(answers)
        A = X[:n]
        B = X[n:]
        sims = [0.0] * n

        # 稀疏向量的行向量余弦
        def _row_cosine(i):
            ai = A.getrow(i)
            bi = B.getrow(i)
            num = (ai.multiply(bi)).sum()
            denom = (ai.multiply(ai)).sum()**0.5 * (bi.multiply(bi)).sum()**0.5
            if denom == 0:
                return 0.0
            return float(num / denom)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_row_cosine, i): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    sims[i] = float(fut.result())
                except Exception:
                    sims[i] = 0.0
        return sims

# ========== 原脚本（仅在“计算指标”块有变化） ==========

def parse_args():
    parser = argparse.ArgumentParser(description="Async batch evaluate from test.parquet via vLLM OpenAI-compatible API")

    parser.add_argument('--tools', type=str, nargs='*', default=default_config.TOOLS,
                        help='Tools for selection, e.g. --tools web_search browser')
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY,
                        help='API key (dummy if local)')
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE,
                        help='API base URL, e.g. http://localhost:8000/v1')
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME,
                        help='Model name (must match --served-model-name in vLLM)')

    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE)
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P)
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS)
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY)

    parser.add_argument('--parquet-file', type=str, default='data/hotpotqa/test.parquet',
                        help='Input parquet file')
    parser.add_argument('--output-file', type=str, default='results/results.json',
                        help='Output JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only evaluate first N rows')
    parser.add_argument('--concurrency', type=int, default=8,
                        help='Number of concurrent requests')

    # 新增参数：评测输出、线程数
    parser.add_argument('--results-eval-file', type=str, default='results/results_eval.json',
                        help='Per-sample (id, question, ground_truth, answer)')
    parser.add_argument('--eval-file', type=str, default='results/eval.json',
                        help='Aggregate metrics output')
    parser.add_argument('--sim-workers', type=int, default=8,
                        help='Threads for semantic similarity')
    return parser.parse_args()

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
    """安全地执行工具函数，处理同步和异步情况"""
    try:
        if hasattr(tool_fn, '_execute_async'):
            result = await tool_fn._execute_async(args_dict)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool_fn.execute, args_dict)
            if inspect.iscoroutine(result):
                result = await result
        return result
    except Exception as e:
        print(f"[ERROR] Tool execution failed: {type(e).__name__}: {e}")
        return {"content": f"Tool execution error: {str(e)}"}

async def run_one_dialog(client, model_name, tools_schema, env, user_prompt_msg,
                         temperature, top_p, max_tokens, repetition_penalty):
    messages = [{"role": "user", "content": user_prompt_msg}]
    final_text_parts = []
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        try:
            response = await get_model_response(
                client, model_name, messages, tools_schema,
                temperature, top_p, max_tokens, repetition_penalty
            )
            resp_msg = response.choices[0].message
        except Exception as e:
            print(f"[ERROR] Model response failed: {type(e).__name__}: {e}")
            return f"[ERROR] Model response failed: {str(e)}"

        if resp_msg.content:
            final_text_parts.append(resp_msg.content)

        assistant_message = {"role": "assistant", "content": resp_msg.content}
        if resp_msg.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in resp_msg.tool_calls
            ]
        messages.append(assistant_message)

        if resp_msg.tool_calls:
            for tc in resp_msg.tool_calls:
                try:
                    args_dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args_dict = tc.function.arguments

                if tc.function.name not in env.tool_map:
                    tool_output = f"Error: Tool '{tc.function.name}' not found"
                else:
                    tool_fn = env.tool_map[tc.function.name]
                    result = await execute_tool_safely(tool_fn, args_dict)
                    tool_output = result.get("content", str(result)) if isinstance(result, dict) else str(result)

                messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "tool_call_id": tc.id
                })
            continue
        else:
            break

    if iteration >= max_iterations:
        print(f"[WARN] Dialog reached max iterations ({max_iterations}), stopping.")

    return "\n".join([t for t in final_text_parts if t is not None])

async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty, rid):
    prompt_list = row.get("prompt", [])
    user_prompt = prompt_list[0]["content"] if prompt_list else ""
    question = extract_question_from_prompt_list(prompt_list)
    ground_truth = row.get("reward_model", {}).get("ground_truth", "")

    try:
        model_output = await run_one_dialog(
            client, model_name, tools_schema, env, user_prompt,
            temp, top_p, max_tokens, rep_penalty
        )
    except Exception as e:
        print(f"[ERROR] Worker failed for question: {question[:50]}...")
        model_output = f"[ERROR] {type(e).__name__}: {e}"

    # 返回时带上 id
    return {"id": rid, "question": question, "ground_truth": ground_truth, "model_output": model_output}

# ========== 计算指标：按 Bytedance 实现对齐（唯一定制变化块） ==========

def normalize_answer(s: str):
    """Bytedance 版本：小写、去冠词(a|an|the)、去标点、空白规整"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = s if s is not None else ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def cal_f1_score(prediction: str, ground_truth: str) -> float:
    """Bytedance 版本 F1（基于 token 交并、词频最小计数）"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Bytedance 版本 EM"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def subem_check(prediction: str, golden_answers) -> float:
    """Bytedance 版本 子串匹配（单向：golden in prediction）"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            return 1.0
    return 0.0

def extract_solution(solution_str: str):
    """Bytedance 版本：从文本中抽取 <answer>...</answer>"""
    if solution_str is None:
        return None
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def has_answer_tags(solution_str: str) -> bool:
    """是否存在带内容的 <answer> 标签（供需要时使用）"""
    if solution_str is None:
        return False
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    return bool(match and match.group(1).strip())

# ========== 主流程 ==========

async def main_async(args):
    print("[INFO] Initializing async client...")
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    # tools
    print(f"[INFO] Loading tools: {args.tools}")
    tool_objs = []
    for t in args.tools:
        try:
            tool_objs.append(_default_tool(t))
            print(f"[INFO] Loaded tool: {t}")
        except NotImplementedError:
            print(f"[WARN] Tool {t} not implemented, skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load tool {t}: {e}")
    
    env = NousToolEnv(tools=tool_objs, max_tool_response_length=args.max_tokens)
    tools_schema = [t.tool_description for t in tool_objs]
    print(f"[INFO] Initialized {len(tool_objs)} tools")

    # data
    print(f"[INFO] Loading data from {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)

    # 尝试获取 id 列；若没有则用索引
    id_col = None
    for cand in ["id", "qid", "uid", "example_id"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        df = df.reset_index().rename(columns={"index": "id"})
        id_col = "id"

    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]
    print(f"[INFO] Processing {len(rows)} rows with concurrency {args.concurrency}")

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_worker(r):
        async with sem:
            rid = r.get(id_col, None)
            return await worker(r, client, args.model, tools_schema, env,
                                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty, rid)

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        try:
            result = await fut
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task failed: {e}")
            results.append({"id": None, "question": "FAILED", "ground_truth": "", "model_output": f"[ERROR] {str(e)}"})

    # 确保输出目录存在
    for out_path in [args.output_file, args.results_eval_file, args.eval_file]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # (1) 保存带 id 的 results.json
    print(f"[INFO] Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # (2) 生成 results_eval.json（id, question, ground_truth, answer）
    results_eval = []
    for rec in results:
        answer = extract_solution(rec.get("model_output", "")) or ""
        results_eval.append({
            "id": rec.get("id"),
            "question": rec.get("question", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "answer": answer
        })

    print(f"[INFO] Saving extracted eval samples to {args.results_eval_file}")
    with open(args.results_eval_file, "w", encoding="utf-8") as f:
        json.dump(results_eval, f, ensure_ascii=False, indent=2)

    # (3) 计算指标并保存 eval.json —— 这里使用 Bytedance 的 EM / 子串 / F1 定义
    gts = [r["ground_truth"] for r in results_eval]
    ans = [r["answer"] for r in results_eval]

    em_list  = [exact_match_score(a, b) for a, b in zip(ans, gts)]
    sub_list = [subem_check(a, b) for a, b in zip(ans, gts)]
    f1_list  = [cal_f1_score(a, b) for a, b in zip(ans, gts)]
    sim_list = semantic_similarity_pairs(ans, gts, max_workers=args.sim_workers)

    import numpy as np
    metrics = {
        "count": len(results_eval),
        "exact_match": {
            "mean": float(np.mean(em_list)) if em_list else 0.0,
            "sum": int(np.sum(em_list)),
        },
        "substring_match": {
            "mean": float(np.mean(sub_list)) if sub_list else 0.0,
            "sum": int(np.sum(sub_list)),
        },
        "f1": {
            "mean": float(np.mean(f1_list)) if f1_list else 0.0,
        },
        "semantic_similarity": {
            "mean": float(np.mean(sim_list)) if sim_list else 0.0,
        }
    }

    print("[RESULT] Metrics summary:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"[INFO] Saving metrics to {args.eval_file}")
    with open(args.eval_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(results)} results, {len(results_eval)} eval rows, metrics to {args.eval_file}")

def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("[INFO] Running in existing event loop")
            loop = asyncio.get_event_loop()
            task = loop.create_task(main_async(args))
            loop.run_until_complete(task)
        else:
            raise e
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
