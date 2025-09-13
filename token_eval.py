from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/luohaoran/wenjin/new/Agent-R1/Qwen/Qwen2.5-3B-Instruct")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Max token id: {max(tokenizer.get_vocab().values())}")