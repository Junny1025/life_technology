# check_wrn_data.py
from chembl_webresource_client.new_client import new_client
import pandas as pd

print("正在从 ChEMBL 获取 WRN 数据...")

# 获取 WRN 靶点的所有活性数据
activities = new_client.activity.filter(target_chembl_id='CHEMBL2146312').only(
    ['standard_value', 'standard_type', 'standard_relation', 'canonical_smiles']
)

df = pd.DataFrame(activities)

print(f"总共获取到 {len(df)} 条记录")
print(f"数据类型分布: {df['standard_type'].value_counts().to_dict()}")
print(f"数据关系分布: {df['standard_relation'].value_counts().to_dict()}")

# 只看 IC50 类型
df_ic50 = df[df['standard_type'] == 'IC50'].copy()
df_ic50['standard_value'] = pd.to_numeric(df_ic50['standard_value'], errors='coerce')
df_ic50 = df_ic50[df_ic50['standard_value'].notna()]

print(f"\nIC50 数据共 {len(df_ic50)} 条")
if len(df_ic50) > 0:
    print(f"IC50 数值范围: {df_ic50['standard_value'].min():.2f} ~ {df_ic50['standard_value'].max():.2f} nM")
    print(f"IC50 平均值: {df_ic50['standard_value'].mean():.2f} nM")
    print(f"IC50 中位数: {df_ic50['standard_value'].median():.2f} nM")
    
    # 按阈值分类
    positive = df_ic50[df_ic50['standard_value'] < 10000]  # <10 μM
    negative = df_ic50[df_ic50['standard_value'] > 100000]  # >100 μM
    mid = df_ic50[(df_ic50['standard_value'] >= 10000) & (df_ic50['standard_value'] <= 100000)]
    
    print(f"\n阳性 (<10 μM): {len(positive)} 条")
    print(f"阴性 (>100 μM): {len(negative)} 条")
    print(f"中等活性 (10-100 μM): {len(mid)} 条")
    
    # 打印前20条数值供参考
    print(f"\n前20条 IC50 值: {df_ic50['standard_value'].head(20).tolist()}")
else:
    print("没有找到 IC50 数据，请检查靶点 ID 是否正确")