#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugCLIP-WRN 预测/推理入口
===========================
使用已训练好的模型对化合物库进行pIC50预测。

使用示例:
  # 基本用法
  python predict.py --model_path ./wrn_model/best_model.pth --scaler_path ./wrn_model/scaler.pkl --input compounds.csv --output predictions.csv

  # 使用result目录中的预训练模型
  python predict.py --model_path ./result/训练好的模型权重best_model.pth --input ./data/your_library.csv

  # 单分子预测
  python predict.py --model_path ./wrn_model/best_model.pth --scaler_path ./wrn_model/scaler.pkl --smiles "CCO"

输入格式:
  - CSV文件需包含 'SMILES' 列
  - 每行一个化合物

输出格式:
  - CSV文件 (UTF-8): SMILES, pred_pIC50 (按pIC50降序排列)
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')


# 模型定义（需与训练时保持一致）
class MLPRegressor(nn.Module):
    """MLP回归器 — 与 finetune_wrn_regression.py 保持一致"""
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


# 日志工具
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# 核心预测函数
def load_model(model_path, scaler_path, device='auto'):
    """加载模型和标准化器"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = MLPRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载标准化器
    scaler = joblib.load(scaler_path)

    log(f"模型已加载 (设备: {device})")
    return model, scaler, device


def predict_single(smiles, model, scaler, device):
    """预测单个SMILES的pIC50"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    X = scaler.transform([np.array(fp, dtype=np.float32)])

    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device)).item()

    return pred


def predict_batch(smiles_list, model, scaler, device, batch_size=128, quiet=False):
    """批量预测pIC50"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid_smiles = []
    failed = 0

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp, dtype=np.float32))
                valid_smiles.append(smi)
            else:
                failed += 1
        except Exception:
            failed += 1

    if not quiet:
        log(f"有效SMILES: {len(valid_smiles)} / {len(smiles_list)} (失败: {failed})")

    if len(valid_smiles) == 0:
        log("无有效化合物！", "WARNING")
        return pd.DataFrame(columns=['SMILES', 'pred_pIC50'])

    # 标准化
    X = scaler.transform(np.array(fps))

    # 分批预测
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            all_preds.extend(preds)

    # 整理结果
    results = pd.DataFrame({
        'SMILES': valid_smiles,
        'pred_pIC50': all_preds
    }).sort_values('pred_pIC50', ascending=False).reset_index(drop=True)

    return results



# 主函数
def main():
    parser = argparse.ArgumentParser(
        description='DrugCLIP-WRN: pIC50预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量预测化合物库
  python predict.py --model_path ./wrn_model/best_model.pth --scaler_path ./wrn_model/scaler.pkl --input compounds.csv

  # 使用预训练模型
  python predict.py --model_path ./result/训练好的模型权重best_model.pth --input your_compounds.csv --output results.csv

  # 预测单分子
  python predict.py --model_path ./wrn_model/best_model.pth --smiles "CC1=CC=C(C=C1)C(=O)NO"
        """
    )

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='标准化器文件路径 (.pkl)，默认与模型同目录的scaler.pkl')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='运行设备 (默认: auto)')

    # 输入参数（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, default=None,
                             help='输入化合物库CSV文件（需包含SMILES列）')
    input_group.add_argument('--smiles', type=str, default=None,
                             help='单分子SMILES字符串（用于快速测试）')

    # 输出参数
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果CSV路径 (默认: ./prediction_results.csv)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批大小 (默认: 128)')
    parser.add_argument('--top_n', type=int, default=20,
                        help='显示的Top候选数量 (默认: 20)')

    args = parser.parse_args()

    # 标准化器路径默认值
    if args.scaler_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.scaler_path = os.path.join(model_dir, 'scaler.pkl')

    # 输出路径默认值
    if args.output is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(base_dir, 'prediction_results.csv')

    log("=" * 60)
    log("DrugCLIP-WRN: pIC50预测")
    log("=" * 60)

    # 验证文件存在
    for path, name in [(args.model_path, "模型文件"), (args.scaler_path, "标准化器")]:
        if not os.path.exists(path):
            log(f"{name}不存在: {path}", "ERROR")
            sys.exit(1)

    # 加载模型
    try:
        model, scaler, device = load_model(args.model_path, args.scaler_path, args.device)
    except Exception as e:
        log(f"模型加载失败: {e}", "ERROR")
        sys.exit(1)

    # 预测
    if args.smiles:
        # 单分子预测
        log(f"预测单分子: {args.smiles}")
        pred = predict_single(args.smiles, model, scaler, device)
        if pred is not None:
            log(f"预测 pIC50 = {pred:.4f}")
            log(f"对应 IC50 ≈ {10 ** (9 - pred):.2f} nM")
        else:
            log("无效的SMILES字符串", "ERROR")

    elif args.input:
        # 批量预测
        if not os.path.exists(args.input):
            log(f"输入文件不存在: {args.input}", "ERROR")
            sys.exit(1)

        log(f"读取化合物库: {args.input}")
        df = pd.read_csv(args.input)

        # 查找SMILES列
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        if smiles_col is None:
            smiles_col = df.columns[0]
            log(f"未找到SMILES列，使用第一列: {smiles_col}", "WARNING")

        smiles_list = df[smiles_col].dropna().tolist()
        log(f"化合物数量: {len(smiles_list)}")

        # 预测
        results = predict_batch(smiles_list, model, scaler, device, args.batch_size)

        if len(results) > 0:
            # 保存结果
            results.to_csv(args.output, index=False, encoding='utf-8')
            log(f"结果已保存: {args.output}")
            log(f"pIC50范围: {results['pred_pIC50'].min():.2f} ~ {results['pred_pIC50'].max():.2f}")
            log(f"pIC50均值: {results['pred_pIC50'].mean():.2f}")
            log(f"pIC50中位数: {results['pred_pIC50'].median():.2f}")

            # 显示Top候选
            log(f"\n{'=' * 60}")
            log(f"Top {min(args.top_n, len(results))} 候选分子:")
            log(f"{'=' * 60}")
            for i, row in results.head(args.top_n).iterrows():
                smi_short = row['SMILES'][:70] + '...' if len(row['SMILES']) > 70 else row['SMILES']
                log(f"  #{i+1}: pIC50 = {row['pred_pIC50']:.4f} | {smi_short}")

    log("\n预测完成!")


if __name__ == "__main__":
    main()
