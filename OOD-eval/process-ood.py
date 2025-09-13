# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

"""
读取 test_ood/ 目录下的所有 .json 文件，将形如
{question, golden_answers, context} 的数据
预处理为 parquet（每个输入 json -> 一个同名 parquet），并映射成以下结构：
{
  "data_source": "hotpotqa/hotpot_qa",  # 可通过 --data_source 覆盖
  "prompt": [{"role": "user", "content": "Question: <q>\\n<instruction_following>"}],
  "ability": "multihop_qa",
  "reward_model": {"style": "rule", "ground_truth": ["答案1", "答案2", ...]}
}
输出目录固定为 eval_ood/（可通过 --local_dir 覆盖，默认 eval_ood）
"""

import os
import glob
import json
import argparse
import datasets
from typing import List, Dict, Any, Iterable

# 公司内部 HDFS 工具（可选）
from verl.utils.hdfs_io import copy, makedirs  # 如果环境中无此库，请自行处理或去掉相关逻辑

def read_json_any(path: str) -> List[Dict[str, Any]]:
    """
    读取 JSON 或 NDJSON（每行一个 JSON 对象）。
    - 若文件整体是一个 JSON 列表 -> 直接返回
    - 若是 JSON 对象且包含 'data' 且为列表 -> 返回 data
    - 若解析失败 -> 按 NDJSON 逐行解析
    - 其他情况 -> 返回空列表
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            data = obj.get('data')
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            # 某些数据可能直接就是一个条目，包装成列表
            if 'question' in obj:
                return [obj]
        return []
    except json.JSONDecodeError:
        # 尝试 NDJSON
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    # 忽略坏行
                    continue
        return items

def extract_answers(item: Dict[str, Any]) -> List[str]:
    """
    支持两种情况：
    1) 新数据：{"golden_answers": [...]} -> 返回整个 list（可能为空）
    2) 兼容旧数据：{"answer": "..."} -> 返回 [answer]
    其他情况统一返回 []
    """
    if 'golden_answers' in item:
        ga = item.get('golden_answers', [])
        if isinstance(ga, list):
            return [str(ans) for ans in ga]
        return []
    if 'answer' in item:
        return [str(item['answer'])]
    return []

instruction_following = (
    r'You MUST FIRST think about the question, explain and analyze it, and rephrase it into a clear, explanatory statement, making it easier to understand so that large language models can interpret and answer it more accurately. '
    r'The thinking process MUST BE enclosed within <think> </think> tags. '
    r'The FINAL answer obtained by using the tool large language model must be placed in <answer> </answer> tags.'
)

def build_records(items: Iterable[Dict[str, Any]], data_source: str) -> List[Dict[str, Any]]:
    """将原始条目映射为目标结构列表"""
    records = []
    for it in items:
        q = str(it.get('question', ''))
        answers = extract_answers(it)
        question = "Question: " + q + "\n" + instruction_following
        rec = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "multihop_qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": answers
            }
        }
        records.append(rec)
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./test_ood', help='输入目录（默认 ./test_ood）')
    parser.add_argument('--local_dir', default='./eval_ood', help='输出 parquet 目录（默认 ./eval_ood）')
    parser.add_argument('--hdfs_dir', default=None, help='可选：将输出目录拷贝到的 HDFS 目标目录')
    parser.add_argument('--data_source', default='hotpotqa/hotpot_qa', help='输出字段 data_source 的值')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 遍历 input_dir 下所有 .json（不递归）
    json_paths = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not json_paths:
        print(f'[WARN] 在目录 {input_dir} 未发现 .json 文件')
        return

    print(f'发现 {len(json_paths)} 个 JSON 文件，开始处理...')
    total_files = 0
    total_records = 0

    for path in json_paths:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        print(f'--> 处理文件：{base}')

        items = read_json_any(path)
        if not items:
            print(f'    [WARN] 文件 {base} 未解析出有效条目，跳过')
            continue

        recs = build_records(items, data_source=args.data_source)
        if not recs:
            print(f'    [WARN] 文件 {base} 映射后记录为空，跳过')
            continue

        ds = datasets.Dataset.from_list(recs)
        out_path = os.path.join(local_dir, f'{stem}.parquet')
        ds.to_parquet(out_path)
        print(f'    写出：{out_path}（{len(ds)} 条）')

        total_files += 1
        total_records += len(ds)

    print(f'完成！成功处理 {total_files} 个文件，共 {total_records} 条记录。输出目录：{local_dir}')

    if args.hdfs_dir is not None:
        print(f'拷贝到 HDFS：{args.hdfs_dir}')
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print('HDFS 拷贝完成。')

if __name__ == '__main__':
    main()
