# extract_smiles_fixed.py
# 修复版本：跳过无法解析的 mol2 文件

import os
import glob
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# 路径基于脚本自身位置解析，保证在任意机器/目录下可直接运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOL2_DIR = os.path.join(SCRIPT_DIR, "data", "tcmbank", "mol2_files")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "data", "tcmbank", "tcm_smiles.csv")

def extract_smiles_safe(mol2_path):
    """安全提取 SMILES，出错则返回 None"""
    try:
        mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=False)
        if mol is not None:
            # 尝试标准化（有些分子可能需要）
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            return Chem.MolToSmiles(mol)
        return None
    except Exception as e:
        # 静默跳过有问题的分子
        return None

def main():
    print("查找 mol2 文件...")
    mol2_files = glob.glob(os.path.join(MOL2_DIR, '**', '*.mol2'), recursive=True)
    print(f"找到 {len(mol2_files)} 个 mol2 文件")

    if not mol2_files:
        print("[警告] 未找到 mol2 文件")
        return

    smiles_list = []
    failed = 0

    for mol2_path in tqdm(mol2_files, desc="提取 SMILES"):
        smi = extract_smiles_safe(mol2_path)
        if smi:
            smiles_list.append({'SMILES': smi, 'source_file': os.path.basename(mol2_path)})
        else:
            failed += 1

    print(f"[完成] 成功: {len(smiles_list)}, 失败: {failed}")

    if smiles_list:
        df = pd.DataFrame(smiles_list)
        df = df.drop_duplicates('SMILES')
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[完成] 已保存: {OUTPUT_CSV}，共 {len(df)} 个唯一化合物")
    else:
        print("[错误] 没有提取到任何 SMILES")

if __name__ == "__main__":
    main()