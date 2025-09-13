#!/usr/bin/env python3
"""
Async batch evaluation against a vLLM OpenAI-compatible endpoint using test.parquet.
- Reads prompts & ground truths from test.parquet
- Calls the model concurrently (asyncio)
- Iteratively resolves tool calls until none remain
- Saves results to JSON
"""

import argparse
import json
import os
import sys
import pandas as pd
import asyncio
import inspect
from tqdm import tqdm

from openai import AsyncOpenAI
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config


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
        # 检查工具是否有异步执行方法
        if hasattr(tool_fn, '_execute_async'):
            # 直接调用异步方法，避免嵌套事件循环
            result = await tool_fn._execute_async(args_dict)
        else:
            # 对于普通的同步工具，在线程池中执行以避免阻塞
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool_fn.execute, args_dict)
            
            # 如果返回的是协程，则等待它
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
    max_iterations = 10  # 防止无限循环
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


async def worker(row, client, model_name, tools_schema, env, temp, top_p, max_tokens, rep_penalty):
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

    return {"question": question, "ground_truth": ground_truth, "model_output": model_output}


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
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]
    print(f"[INFO] Processing {len(rows)} rows with concurrency {args.concurrency}")

    results = []
    sem = asyncio.Semaphore(args.concurrency)

    async def sem_worker(r):
        async with sem:
            return await worker(r, client, args.model, tools_schema, env,
                                args.temperature, args.top_p, args.max_tokens, args.repetition_penalty)

    tasks = [asyncio.create_task(sem_worker(r)) for r in rows]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        try:
            result = await fut
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task failed: {e}")
            results.append({"question": "FAILED", "ground_truth": "", "model_output": f"[ERROR] {str(e)}"})

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved {len(results)} records to {args.output_file}")


def main():
    args = parse_args()
    
    # 简化事件循环处理
    try:
        # 直接运行异步函数
        asyncio.run(main_async(args))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # 如果已经在事件循环中，创建任务
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