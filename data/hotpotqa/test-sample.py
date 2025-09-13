#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datasets import Dataset

def sample_parquet(src_path: str, dst_path: str, k: int, seed: int = 42):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到文件: {src_path}")

    ds = Dataset.from_parquet(src_path)
    n = len(ds)
    if n == 0:
        raise ValueError(f"{src_path} 为空")

    # 若数据不足，最多就取 n 条
    k_eff = min(k, n)

    # 打乱后取前 k_eff 条，保证可复现
    ds_small = ds.shuffle(seed=seed).select(range(k_eff))
    ds_small.to_parquet(dst_path)

    print(f"✅ {os.path.basename(src_path)} -> {os.path.basename(dst_path)}: 选取 {k_eff}/{n} 条")

    # 打印第一条数据
    first = ds_small[0]
    ground_truth = None
    if "reward_model" in first and isinstance(first["reward_model"], dict):
        ground_truth = first["reward_model"].get("ground_truth", None)

    print(f"第一条数据 (来自 {os.path.basename(dst_path)}):")
    print(first)

    # 检查 ground_truth 是否为空
    if (ground_truth is None 
        or (isinstance(ground_truth, list) and all((not x or str(x).strip()=="") for x in ground_truth)) 
        or (isinstance(ground_truth, str) and str(ground_truth).strip()=="")):
        print("⚠️ ground_truth 为空")
    else:
        print("✅ ground_truth 不为空")

    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="包含 parquet 文件的目录")
    parser.add_argument("--train_in", default="train-or.parquet")
    parser.add_argument("--val_in", default="validation-or.parquet")
    parser.add_argument("--train_out", default="train.parquet")
    parser.add_argument("--val_out", default="validation.parquet")
    parser.add_argument("--train_k", type=int, default=1280)
    parser.add_argument("--val_k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_src = os.path.join(args.dir, args.train_in)
    val_src   = os.path.join(args.dir, args.val_in)
    train_dst = os.path.join(args.dir, args.train_out)
    val_dst   = os.path.join(args.dir, args.val_out)

    sample_parquet(train_src, train_dst, args.train_k, args.seed)
    sample_parquet(val_src,   val_dst,   args.val_k,   args.seed)
