# run_docking_final.py
# 完整自动对接脚本 - 保存所有结果
# 配体过滤保留 ATOM/ROOT/ENDROOT/BRANCH/ENDBRANCH/TORSDOF 行（TORSDOF 为 Vina 必需）
import os
import subprocess
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# 路径基于脚本自身位置解析，保证在任意机器/目录下可直接运行
DOCKING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DOCKING_DIR)          # DrugCLIP_Project
PROJECT_ROOT = os.path.dirname(PROJECT_DIR)         # 项目根目录
VINA_EXE = os.path.join(DOCKING_DIR, "vina_1.2.5_win.exe")
SCREENING_RESULTS = os.path.join(PROJECT_ROOT, "result", "筛选结果tcmbank_screening_results.csv")
if not os.path.exists(SCREENING_RESULTS):
    SCREENING_RESULTS = os.path.join(PROJECT_DIR, "Drug-The-Whole-Genome-main", "tcmbank_screening_results.csv")
PROTEIN_PDB = os.path.join(DOCKING_DIR, "wrn_receptor.pdb")
PROTEIN_ORIGINAL = os.path.join(DOCKING_DIR, "8PFL.pdb")  # 大小写修正，Linux兼容
TOP_N = 3

OUTPUT_DIR = os.path.join(DOCKING_DIR, "vina_final_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")

def clean_receptor(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    keep = []
    for line in lines:
        if not line.strip().startswith(('ROOT','ENDROOT','BRANCH','ENDBRANCH','TORSDOF')):
            keep.append(line)
    with open(output_file, 'w') as f:
        f.writelines(keep)
    log(f"受体已清理: {output_file}")
    return output_file

def smiles_to_pdb(smiles, out):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, out)
    return out

def pdb_to_pdbqt(pdb_file, pdbqt_file, is_ligand=True):
    # 预先检查 Open Babel 是否可用，给出友好提示
    if shutil.which('obabel') is None:
        log("未找到 Open Babel (obabel)！请先安装: conda install openbabel -c conda-forge")
        log("提示: 提交包已附带完成的对接结果 (vina_final_results/)，跳过对接不影响其他步骤")
        raise SystemExit(1)
    if is_ligand:
        cmd = f'obabel "{pdb_file}" -O "{pdbqt_file}" -h --gen3d --addcharges --writecharges'
    else:
        cmd = f'obabel "{pdb_file}" -O "{pdbqt_file}"'
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    return pdbqt_file

# 1. 生成干净的受体
log("生成干净的受体PDBQT...")
protein_raw = os.path.join(OUTPUT_DIR, "wrn_receptor_raw.pdbqt")
protein_clean = os.path.join(OUTPUT_DIR, "wrn_receptor.pdbqt")
pdb_to_pdbqt(PROTEIN_PDB, protein_raw, is_ligand=False)
clean_receptor(protein_raw, protein_clean)

# 2. 读取Top候选
df = pd.read_csv(SCREENING_RESULTS)
top = df.head(TOP_N)

# 3. 对接盒子中心
center = (-27.641, -2.078, 25.281)
size = (22, 22, 22)

all_results = []

for idx, row in top.iterrows():
    rank = idx + 1
    smi = row['SMILES']
    pic50 = row['pred_pIC50']
    log(f"对接候选 #{rank}...")

    mol_dir = os.path.join(OUTPUT_DIR, f"candidate_{rank:02d}")
    os.makedirs(mol_dir, exist_ok=True)

    try:
        # 生成配体
        ligand_pdb = os.path.join(mol_dir, "ligand.pdb")
        smiles_to_pdb(smi, ligand_pdb)

        ligand_pdbqt_raw = os.path.join(mol_dir, "ligand_raw.pdbqt")
        ligand_pdbqt = os.path.join(mol_dir, "ligand.pdbqt")
        pdb_to_pdbqt(ligand_pdb, ligand_pdbqt_raw, is_ligand=True)

        # 清理配体中的非标准原子（保留ROOT/ENDROOT及Vina必需的TORSDOF行）
        with open(ligand_pdbqt_raw, 'r') as f:
            lines = f.readlines()
        fixed_lines = []
        for line in lines:
            if line.strip().startswith(('ATOM','ROOT','ENDROOT','BRANCH','ENDBRANCH','TORSDOF')):
                fixed_lines.append(line)
        with open(ligand_pdbqt, 'w') as f:
            f.writelines(fixed_lines)

        # 运行Vina对接
        out_pdbqt = os.path.join(mol_dir, "out.pdbqt")
        log_file = os.path.join(mol_dir, "vina.log")

        cmd = (
            f'"{VINA_EXE}"'
            f' --receptor "{protein_clean}"'
            f' --ligand "{ligand_pdbqt}"'
            f' --out "{out_pdbqt}"'
            f' --center_x {center[0]:.3f}'
            f' --center_y {center[1]:.3f}'
            f' --center_z {center[2]:.3f}'
            f' --size_x {size[0]:.1f}'
            f' --size_y {size[1]:.1f}'
            f' --size_z {size[2]:.1f}'
            f' --exhaustiveness 8'
            f' --num_modes 9'
            f' --verbosity 1'
        )

        with open(log_file, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

        # 解析结合能
        with open(log_file, 'r') as f:
            content = f.read()
        affinities = []
        for line in content.split('\n'):
            if 'kcal/mol' in line and 'affinity' in line.lower():
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        affinities.append(float(parts[1]))
                    except:
                        pass

        if affinities:
            best_aff = affinities[0]
            log(f"  候选 #{rank} 最佳结合能: {best_aff:.3f} kcal/mol")
        else:
            best_aff = None
            log(f"  候选 #{rank} 未能解析结合能")

        all_results.append({
            'Rank': rank,
            'pIC50': pic50,
            'Vina_Affinity': best_aff,
            'Num_Modes': len(affinities)
        })

    except Exception as e:
        log(f"  候选 #{rank} 失败: {e}")
        all_results.append({'Rank': rank, 'pIC50': pic50, 'Vina_Affinity': None, 'Num_Modes': 0})

# 保存汇总结果
result_df = pd.DataFrame(all_results)
result_df.to_csv(os.path.join(OUTPUT_DIR, "docking_summary.csv"), index=False)

log("=" * 50)
log("对接完成！汇总结果：")
print(result_df.to_string(index=False))
log(f"\n结果已保存至: {OUTPUT_DIR}")
