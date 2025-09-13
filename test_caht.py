from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

messages = [{"role": "user", "content": "hi"}]

dummy_tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on the internet using Wikipedia as a knowledge source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

print(tok.apply_chat_template(
    messages,
    tools=dummy_tools,          # ★ 必须：非空列表，才能进 {%- if tools %} 分支
    add_generation_prompt=True, # 可选：看起来更像推理首轮
    tokenize=False
))
