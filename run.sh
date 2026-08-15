#!/bin/bash
# DrugCLIP-WRN 一键运行脚本 (Linux/Mac)
# 用法:
#   bash run.sh full      # 运行全流程
#   bash run.sh train     # 仅训练模型
#   bash run.sh screen    # 仅虚拟筛选
#   bash run.sh dock      # 仅分子对接
#   bash run.sh report    # 仅生成报告


set -e

MODE=${1:-full}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


echo "DrugCLIP-WRN: WRN抑制剂虚拟筛选系统"
echo "运行模式: ${MODE}"


# 激活conda环境（如果存在）
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate drugclip 2>/dev/null || echo "警告: drugclip环境未找到，使用当前Python环境"
fi

# 运行
python "${SCRIPT_DIR}/main.py" --mode "${MODE}" "$@"


echo "运行完成"

