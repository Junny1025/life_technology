from chembl_webresource_client.new_client import new_client
import pandas as pd

print("正在从 ChEMBL API 获取 WRN (CHEMBL2146312) 的活性数据...")

# 获取靶点所有活性数据
activities = new_client.activity.filter(target_chembl_id='CHEMBL2146312').only(
    ['standard_value', 'standard_type', 'standard_relation', 'canonical_smiles', 'molecule_chembl_id']
)

df = pd.DataFrame(activities)
print(f"总共获取到 {len(df)} 条记录")

# 筛选 IC50 类型
df_ic50 = df[df['standard_type'] == 'IC50'].copy()
print(f"其中 IC50 类型: {len(df_ic50)} 条")

# 转换数值并去除无效值
df_ic50['standard_value'] = pd.to_numeric(df_ic50['standard_value'], errors='coerce')
df_ic50 = df_ic50[df_ic50['standard_value'].notna()]
print(f"有效数值: {len(df_ic50)} 条")

# 查看数值分布
if len(df_ic50) > 0:
    print(f"数值范围: {df_ic50['standard_value'].min():.2f} ~ {df_ic50['standard_value'].max():.2f} nM")
    print(f"中位数: {df_ic50['standard_value'].median():.2f} nM")

# 保存为 CSV
df_ic50.to_csv('wrn_chembl_ic50_api.csv', index=False)
print("数据已保存为 wrn_chembl_ic50_api.csv")

# 按阈值简单分类
positive = df_ic50[df_ic50['standard_value'] < 10000]
negative = df_ic50[df_ic50['standard_value'] > 100000]
print(f"阳性 (<10 µM): {len(positive)} 条")
print(f"阴性 (>100 µM): {len(negative)} 条")