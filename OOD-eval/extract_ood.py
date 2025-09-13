import os
import json
import random

# 输入数据目录（同级目录）
data_dir = "./"
# 输出目录
output_dir = "test_ood"
os.makedirs(output_dir, exist_ok=True)

# 每个数据集抽取的数量
sample_size = 128

# 遍历文件夹下的所有 json 文件（不递归）
for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(data_dir, filename)
    # 生成输出文件名：优先去掉 _raw 后缀，再去掉 .json
    stem, _ = os.path.splitext(filename)
    dataset_name = stem[:-4] if stem.endswith("_raw") else stem
    out_name = f"{dataset_name}-128-eval_ood.json"
    out_path = os.path.join(output_dir, out_name)

    # 读取数据
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 {filename} 失败：{e}")
        continue

    if not isinstance(data, list):
        print(f"❌ 数据集 {filename} 不是列表格式，已跳过。")
        continue

    # 如果数据量不足，跳过该数据集
    if len(data) < sample_size:
        print(f"❌ 数据集 {filename} 数据不足（{len(data)} 条），需要 {sample_size} 条，已跳过。")
        continue

    # 抽样（不修改原列表顺序）
    sampled = random.sample(data, sample_size)

    # 保存
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
        print(f"✅ 已处理 {filename} -> {os.path.join(output_dir, out_name)}")
    except Exception as e:
        print(f"❌ 写出 {out_name} 失败：{e}")

print("🎉 处理完成！")
