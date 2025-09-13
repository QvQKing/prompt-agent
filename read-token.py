from transformers import AutoTokenizer

# 加载 Qwen2.5-1.5B-Instruct 的 tokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 打印词表大小
print("Vocab size:", len(tok))

# 打印 eos、pad、bos 等特殊 token 信息
print("EOS token:", tok.eos_token, "ID:", tok.eos_token_id)
print("BOS token:", tok.bos_token, "ID:", tok.bos_token_id)
print("PAD token:", tok.pad_token, "ID:", tok.pad_token_id)
