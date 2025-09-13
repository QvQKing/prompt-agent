import pandas as pd

# 让 pandas 显示所有列和完整内容
pd.set_option("display.max_columns", None)   # 显示所有列
pd.set_option("display.max_rows", None)      # 显示所有行（这里只有2行）
pd.set_option("display.max_colwidth", None)  # 每个单元格显示完整内容

# 读取 parquet 文件
df = pd.read_parquet("MathQA-128-eval_ood.parquet")

# 打印前两条数据
print(df.head(1))
