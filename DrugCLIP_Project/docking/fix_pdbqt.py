# fix_pdbqt.py
# 修复 PDBQT 文件格式，使其兼容 Vina 1.2.7
# 使用方法：python fix_pdbqt.py input.pdbqt output.pdbqt

import sys
import re

# Vina 1.2.7 支持的原子类型（元素符号）
SUPPORTED_ELEMENTS = {'C', 'N', 'O', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I'}

def fix_pdbqt(input_file, output_file):
    """
    修复 PDBQT 文件：
    1. 过滤掉非标准原子类型（如 Na、K、Mg 等）
    2. 确保 ROOT/ENDROOT 标签完整
    3. 检查原子记录格式
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    atom_count = 0
    root_found = False
    endroot_found = False
    
    for line in lines:
        # 保留 ROOT 和 ENDROOT 标签
        if line.startswith('ROOT'):
            root_found = True
            fixed_lines.append(line)
            continue
        if line.startswith('ENDROOT'):
            endroot_found = True
            fixed_lines.append(line)
            continue
        
        # 保留 BRANCH 和 ENDBRANCH 标签（如果有）
        if line.startswith('BRANCH') or line.startswith('ENDBRANCH'):
            fixed_lines.append(line)
            continue
        
        # 处理 ATOM 行
        if line.startswith('ATOM'):
            # 检查原子类型（第 77-78 列，PDBQT 格式）
            # 示例：ATOM      1  C   LIG A   1      -1.234   2.345   3.456  0.000  0.000
            if len(line) >= 78:
                atom_type = line[76:78].strip()  # 提取第 77-78 列
                # 有些文件原子类型在第 77 列，有些在第 78 列，更稳健的方法是用正则
                # 使用正则提取原子类型
                match = re.search(r'ATOM\s+\d+\s+(\S+)\s+', line)
                if match:
                    atom_name = match.group(1)
                    # 提取元素符号（原子名称通常是元素符号 + 编号，如 C1, C2, N3, O4）
                    element = re.sub(r'[0-9]', '', atom_name).strip()
                    if element:
                        element = element[:1].upper() + element[1:].lower()
                        # 检查是否为支持的元素
                        if element in SUPPORTED_ELEMENTS:
                            fixed_lines.append(line)
                            atom_count += 1
                        else:
                            print(f"  跳过非标准原子: {atom_name} (元素: {element})")
                    else:
                        # 无法提取元素，检查原子类型列
                        if atom_type in SUPPORTED_ELEMENTS:
                            fixed_lines.append(line)
                            atom_count += 1
                        else:
                            print(f"  跳过非标准原子类型: {atom_type}")
                else:
                    # 正则匹配失败，保留原行（保守处理）
                    fixed_lines.append(line)
                    atom_count += 1
            else:
                # 行太短，可能是畸形，尝试保留
                fixed_lines.append(line)
                atom_count += 1
        else:
            # 非 ATOM 行（如 TORSDOF、REMARK 等）保留
            fixed_lines.append(line)
    
    # 如果缺少 ROOT 或 ENDROOT，补上
    if not root_found:
        print("警告: 未找到 ROOT 标签，正在添加")
        fixed_lines.insert(0, 'ROOT\n')
    if not endroot_found:
        print("警告: 未找到 ENDROOT 标签，正在添加")
        fixed_lines.append('ENDROOT\n')
    
    # 写入修复后的文件
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"修复完成: {output_file}")
    print(f"  保留原子数: {atom_count}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python fix_pdbqt.py <输入文件> <输出文件>")
        print("示例: python fix_pdbqt.py ligand.pdbqt ligand_fixed.pdbqt")
    else:
        fix_pdbqt(sys.argv[1], sys.argv[2])