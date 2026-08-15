# clean_protein.py
# 在 D:\DrugCLIP_Project\docking 文件夹下创建这个文件并运行

import os

input_file = "8pfl.pdb"
output_file = "wrn_receptor.pdb"

if not os.path.exists(input_file):
    print(f"[错误] 错误：找不到 {input_file}，请先下载并放入当前文件夹")
else:
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            # 只保留 ATOM 记录（蛋白质原子）
            if line.startswith('ATOM'):
                f.write(line)
    
    print(f"[完成] 蛋白文件已清理并保存为: {output_file}")