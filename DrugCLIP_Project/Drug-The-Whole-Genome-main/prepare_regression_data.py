import pandas as pd
import numpy as np

# 读取 API 下载的数据
df = pd.read_csv('wrn_chembl_ic50_api.csv')

print(f"原始数据: {len(df)} 条")

# 筛选有效数据：IC50 类型、数值有效、有 SMILES
df = df[df['standard_type'] == 'IC50']
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
df = df[df['standard_value'].notna()]
df = df[df['canonical_smiles'].notna()]

# 去除重复 SMILES（保留 IC50 最低的，即活性最强的）
df = df.sort_values('standard_value').drop_duplicates('canonical_smiles', keep='first')

print(f"去重后: {len(df)} 条")

# 计算 pIC50 = -log10(IC50)，注意 IC50 单位是 nM，需要转换为 M
# pIC50 = -log10(IC50_in_M) = 9 - log10(IC50_in_nM)
df['pIC50'] = 9 - np.log10(df['standard_value'])

# 过滤异常值（pIC50 通常在 4-10 之间）
df = df[(df['pIC50'] >= 4) & (df['pIC50'] <= 10)]
print(f"过滤后: {len(df)} 条")
print(f"pIC50 范围: {df['pIC50'].min():.2f} ~ {df['pIC50'].max():.2f}")

# 保存训练数据
df[['canonical_smiles', 'pIC50']].to_csv('./data/wrn_pic50_train.csv', index=False)
print(f"训练数据已保存: ./data/wrn_pic50_train.csv")