# analyze_hits.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

# 读取筛选结果
df = pd.read_csv('tcmbank_screening_results.csv')
top20 = df.head(20)

print("="*60)
print("Top 20 候选分子分析")
print("="*60)

# 计算每个分子的属性
results = []
for idx, row in top20.iterrows():
    smi = row['SMILES']
    pic50 = row['pred_pIC50']
    mol = Chem.MolFromSmiles(smi)
    
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # 生成骨架
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold) if scaffold else ""
        
        results.append({
            'rank': idx+1,
            'SMILES': smi[:80] + '...' if len(smi) > 80 else smi,
            'pIC50': pic50,
            'MolWt': mw,
            'LogP': logp,
            'HBD': hbd,
            'HBA': hba,
            'RotBonds': rot_bonds,
            'TPSA': tpsa,
            'Scaffold': scaffold_smi[:60] + '...' if len(scaffold_smi) > 60 else scaffold_smi
        })

result_df = pd.DataFrame(results)
print(result_df.to_string())

# 生成Top 10结构图（仅当有至少1个分子时）
mols = []
for smi in top20['SMILES'][:10]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)

if mols:
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250,250))
    img.save('top10_hits.png')
    print("\n[完成] Top 10 结构图已保存: top10_hits.png")
else:
    print("[警告] 没有有效的分子可以绘图")