#!/usr/bin/env python3
"""
Batch evaluation against a vLLM OpenAI-compatible endpoint using test.parquet.
- Reads prompts & ground truths from test.parquet
- Calls the model (with tools, auto tool-choice, repetition_penalty, etc.)
- Iteratively resolves tool calls until none remain
- Saves [{question, ground_truth, model_output}] to JSON
"""

import argparse
import json
import os
import sys
import importlib
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# import your project pieces to match the interactive script behavior
from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate from test.parquet via vLLM OpenAI-compatible API")

    # Keep flags aligned with your interactive script
    parser.add_argument('--tools', type=str, nargs='*', default=default_config.TOOLS,
                        help='Tools for selection (space-separated), e.g. --tools web_search browser')
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY,
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE,
                        help='OpenAI API base URL, e.g. http://localhost:8000/v1')
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME,
                        help='Model name served by vLLM (e.g. --served-model-name)')

    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS,
                        help='Maximum tokens to generate')
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY,
                        help='Repetition penalty')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config (python file) to override defaults')

    # Batch settings
    parser.add_argument('--parquet-file', type=str, default='/home/luohaoran/wenjin/new/Agent-R1/data/hotpotqa/test.parquet',
                        help='Input parquet file path')
    parser.add_argument('--output-file', type=str, default='eval_results/results.json',
                        help='Output JSON file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate only first N rows (for smoke tests)')

    return parser.parse_args()


def load_custom_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)
    return custom_config


def get_model_response(client: OpenAI, model_name: str, messages, tools, temperature, top_p, max_tokens, repetition_penalty):
    """Single call to the chat.completions endpoint (mirrors your interactive code)."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={
            "repetition_penalty": repetition_penalty,
        },
        stop=["</tool_call>"],  # consistent with your script
    )
    return resp


def extract_question_from_prompt_list(prompt_list):
    """
    prompt_list is like: [{"role":"user","content":"Question: ...\\n<instructions>"}]
    We return the first line after "Question: "
    """
    if not prompt_list:
        return ""
    content = prompt_list[0].get("content", "")
    if content.startswith("Question:"):
        first_line = content.split("\n", 1)[0]
        return first_line.replace("Question:", "").strip()
    return content.strip()


def run_one_dialog(client, model_name, tools_schema, env: NousToolEnv, user_prompt_msg,
                   temperature, top_p, max_tokens, repetition_penalty):
    """
    Run a single dialog until no tool calls remain.
    Returns final assistant content (concatenated over iterations).
    """
    messages = []
    # Use the prompt exactly as stored in parquet (already includes instructions)
    messages.append({
        "role": "user",
        "content": user_prompt_msg
    })

    final_text_parts = []

    while True:
        response = get_model_response(
            client, model_name, messages, tools_schema,
            temperature, top_p, max_tokens, repetition_penalty
        )
        response_message = response.choices[0].message

        # Record the assistant textual content
        if response_message.content:
            final_text_parts.append(response_message.content)

        # Prepare assistant message to add (including potential tool_calls)
        assistant_message = {"role": "assistant", "content": response_message.content}
        if response_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response_message.tool_calls
            ]
        messages.append(assistant_message)

        # Handle tool calls
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                # Parse args; vLLM hermes parser returns JSON in .arguments
                try:
                    args_dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # If not strict JSON, pass raw string to tool; your env likely expects dict
                    args_dict = tc.function.arguments

                # Execute tool via your environment
                tool_fn = env.tool_map[tc.function.name]
                result = tool_fn.execute(args_dict)
                tool_output = result["content"] if isinstance(result, dict) and "content" in result else str(result)

                # Add tool result for the assistant to consume
                messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "tool_call_id": tc.id
                })
            # Continue the loop to let the model read tool outputs
            continue
        else:
            # No tool calls -> dialog is done
            break

    return "\n".join([t for t in final_text_parts if t is not None])


def main():
    args = parse_args()

    # Load config if provided
    config = default_config
    if args.config:
        try:
            config = load_custom_config(args.config)
            print(f"[INFO] Loaded custom config from {args.config}")
        except Exception as e:
            print(f"[WARN] Failed to load custom config: {e}")
            print("[WARN] Falling back to default_config")

    # Resolve runtime settings (aligned with your interactive script)
    TOOLS = args.tools
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    REPETITION_PENALTY = args.repetition_penalty

    # Init OpenAI-compatible client
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    # Build tool runtime & JSON schemas (exactly as your script)
    tool_objs = [_default_tool(t) for t in TOOLS]
    env = NousToolEnv(tools=tool_objs, max_tool_response_length=MAX_TOKENS)
    tools_schema = [t.tool_description for t in tool_objs]

    # Load parquet
    df = pd.read_parquet(args.parquet_file)
    rows = df.to_dict(orient="records")
    if args.limit:
        rows = rows[:args.limit]

    results = []

    for row in tqdm(rows, desc="Evaluating"):
        # parquet schema built by your preprocessing:
        #  - prompt: list[{"role":"user","content":"Question: ...\n<instructions>"}]
        #  - reward_model: {"style": "rule", "ground_truth": "..."}
        # We use the prompt exactly (no reformat), and extract question for reporting.
        prompt_list = row.get("prompt", [])
        user_prompt = prompt_list[0]["content"] if prompt_list else ""
        question = extract_question_from_prompt_list(prompt_list)

        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "")

        try:
            model_output = run_one_dialog(
                client=client,
                model_name=MODEL_NAME,
                tools_schema=tools_schema,
                env=env,
                user_prompt_msg=user_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                repetition_penalty=REPETITION_PENALTY
            )
        except Exception as e:
            model_output = f"[ERROR] {type(e).__name__}: {e}"

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_output": model_output
        })

    # Save JSON
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(results)} records to {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Goodbye!")
        sys.exit(130)
