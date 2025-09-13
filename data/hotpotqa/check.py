import os
import datasets
import tiktoken  # pip install tiktoken

# 读取 parquet
train_file = "train.parquet"
val_file = "validation.parquet"

train_ds = datasets.Dataset.from_parquet(train_file)
val_ds = datasets.Dataset.from_parquet(val_file)

# 使用 OpenAI 的 cl100k_base 分词器 (GPT-4 / GPT-3.5)
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

def check_dataset(ds, name, max_len=8192):
    count_exceed = 0
    for sample in ds:
        try:
            q = sample["prompt"][0]["content"]  # question 在 prompt 里
        except Exception:
            continue
        num_tokens = count_tokens(q)
        if num_tokens > max_len:
            count_exceed += 1
    
    print(f"=== {name} 数据集 ===")
    print(f"⚠️ 超过 {max_len} tokens 的样本数: {count_exceed}\n")

# 检查 train 和 validation
check_dataset(train_ds, "train")
check_dataset(val_ds, "validation")
