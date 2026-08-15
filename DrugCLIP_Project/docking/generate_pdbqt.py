# generate_pdbqt.py
# 使用 RDKit 直接生成符合 Vina 1.2.7 格式的 PDBQT

import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PyMol
import os

def generate_vina_pdbqt(input_pdb, output_pdbqt, is_protein=False):
    """
    生成 Vina 1.2.7 兼容的 PDBQT 文件
    """
    # 读取分子
    if is_protein:
        mol = Chem.MolFromPDBFile(input_pdb, removeHs=False, proximityBonding=False)
        if mol is None:
            print(f"无法读取蛋白: {input_pdb}")
            return False
    else:
        mol = Chem.MolFromPDBFile(input_pdb, removeHs=True)
        if mol is None:
            print(f"无法读取配体: {input_pdb}")
            return False
    
    # 获取构象
    conf = mol.GetConformer()
    
    # 原子类型映射 (Vina 支持的原子类型)
    ATOM_TYPE = {
        6: 'C',   # 碳
        7: 'N',   # 氮
        8: 'O',   # 氧
        15: 'P',  # 磷
        16: 'S',  # 硫
        9: 'F',   # 氟
        17: 'Cl', # 氯
        35: 'Br', # 溴
        53: 'I',  # 碘
    }
    
    lines = []
    lines.append('ROOT')
    
    atom_idx = 0
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_num = atom.GetAtomicNum()
        
        # 获取原子类型
        atom_type = ATOM_TYPE.get(atom_num, 'C')
        
        # 获取原子名称 (PDB 格式)
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info:
            atom_name = pdb_info.GetName().strip()
            res_name = pdb_info.GetResidueName().strip()
            res_num = pdb_info.GetResidueNumber()
            chain = pdb_info.GetChainId()
        else:
            # 如果没有 PDB 信息，使用默认值
            atom_name = f"{atom_type}{atom.GetIdx()+1}"
            res_name = "LIG"
            res_num = 1
            chain = "A"
        
        # 获取电荷 (如果没有，设为0)
        charge = 0.0
        try:
            charge = atom.GetDoubleProp('_GasteigerCharge')
        except:
            pass
        
        # 格式化 PDBQT ATOM 行
        # 格式: ATOM  index atom_name res_name chain res_num x y z charge atom_type
        line = f"ATOM{atom_idx+1:7d} {atom_name:<4s} {res_name:>3s} {chain:1s}{res_num:4d}    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{charge:8.3f} {atom_type:>2s}"
        lines.append(line)
        atom_idx += 1
    
    lines.append('ENDROOT')
    
    # 写入文件，确保没有 BOM
    with open(output_pdbqt, 'w', encoding='ascii') as f:
        f.write('\n'.join(lines))
    
    print(f"[完成] 生成完成: {output_pdbqt}")
    print(f"   原子数: {atom_idx}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python generate_pdbqt.py <输入.pdb> <输出.pdbqt> [--protein]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    is_protein = "--protein" in sys.argv
    
    generate_vina_pdbqt(input_file, output_file, is_protein)