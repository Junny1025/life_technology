# force_fix_pdbqt.py
import sys
import re

def force_fix_pdbqt(input_file, output_file):
    # 以二进制模式读取，去除 BOM
    with open(input_file, 'rb') as f:
        content = f.read()
    
    # 去除 UTF-8 BOM (EF BB BF)
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    
    # 转为文本，忽略无法解码的字符
    text = content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    # 过滤：去除空行和注释行（以 # 开头），并去除首尾空白
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            filtered.append(stripped)
    
    # 确保第一行是 ROOT
    if not filtered or filtered[0] != 'ROOT':
        filtered.insert(0, 'ROOT')
    
    # 确保最后一行是 ENDROOT
    if not filtered or filtered[-1] != 'ENDROOT':
        filtered.append('ENDROOT')
    
    # 移除重复的 ROOT/ENDROOT
    final_lines = []
    for line in filtered:
        if line == 'ROOT':
            if final_lines and final_lines[-1] == 'ROOT':
                continue
        if line == 'ENDROOT':
            if final_lines and final_lines[-1] == 'ENDROOT':
                continue
        final_lines.append(line)
    
    # 写入新文件
    with open(output_file, 'w') as f:
        f.write('\n'.join(final_lines))
    
    print(f"[完成] 修复完成: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python force_fix_pdbqt.py <输入.pdbqt> <输出.pdbqt>")
    else:
        force_fix_pdbqt(sys.argv[1], sys.argv[2])