#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugCLIP-WRN 全流程一键运行入口
===============================
支持模式:
  - full:  完整流程（数据获取 → 训练 → 筛选 → 对接 → 报告）
  - train: 仅模型训练
  - screen: 仅虚拟筛选（需已训练模型）
  - dock: 仅分子对接（需筛选结果）
  - report: 仅生成报告

使用示例:
  python main.py --mode full
  python main.py --mode train --data_path ./data/wrn_pic50_train.csv
  python main.py --mode screen --input ./data/tcmbank/tcm_smiles.csv
  python main.py --mode dock --top_n 10
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Scaffolds

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "DrugCLIP_Project", "Drug-The-Whole-Genome-main")
DOCK_DIR = os.path.join(BASE_DIR, "DrugCLIP_Project", "docking")
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_DIR = os.path.join(SRC_DIR, "wrn_model")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# 日志工具
# =============================================================================
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


# =============================================================================
# 步骤1: 数据获取
# =============================================================================
def step_fetch_data():
    """从ChEMBL API获取WRN活性数据"""
    log("=" * 60)
    log("步骤1: 获取WRN活性数据")
    log("=" * 60)
    script = os.path.join(SRC_DIR, "fetch_wrn_api.py")
    if os.path.exists(script):
        try:
            subprocess.run([sys.executable, script], cwd=SRC_DIR, check=True)
        except Exception as e:
            log(f"数据获取失败（可能无网络或API不可用）: {e}", "WARNING")
            log("将使用随提交包附带的数据文件继续运行", "WARNING")
    else:
        log(f"未找到脚本: {script}，跳过", "WARNING")
        log("假设已有数据文件 wrn_chembl_ic50_api.csv", "INFO")


# =============================================================================
# 步骤2: 数据预处理
# =============================================================================
def step_prepare_data():
    """预处理训练数据（pIC50回归）"""
    log("=" * 60)
    log("步骤2: 数据预处理")
    log("=" * 60)
    script = os.path.join(SRC_DIR, "prepare_regression_data.py")
    if os.path.exists(script):
        subprocess.run([sys.executable, script], cwd=SRC_DIR, check=True)
    else:
        log(f"未找到脚本: {script}，跳过", "WARNING")


# =============================================================================
# 步骤3: 模型训练
# =============================================================================
def step_train(data_path=None, epochs=50, batch_size=16, lr=1e-4, seed=48):
    """训练MLP回归模型（默认 seed=48，可复现最佳验证 RMSE=2.89）"""
    log("=" * 60)
    log("步骤3: 模型训练")
    log("=" * 60)

    if data_path is None:
        data_path = os.path.join(SRC_DIR, "data", "wrn_pic50_train.csv")

    if not os.path.exists(data_path):
        log(f"训练数据不存在: {data_path}", "ERROR")
        log("请先运行步骤1和步骤2获取数据", "ERROR")
        return False

    script = os.path.join(SRC_DIR, "finetune_wrn_regression.py")
    cmd = [
        sys.executable, script,
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--save_dir", MODEL_DIR,
    ]
    subprocess.run(cmd, cwd=SRC_DIR, check=True)
    log(f"模型已保存至: {MODEL_DIR}")
    return True


# =============================================================================
# 步骤4: 虚拟筛选
# =============================================================================
def step_screen(input_csv=None, output_csv=None, model_path=None, scaler_path=None):
    """对化合物库进行虚拟筛选"""
    log("=" * 60)
    log("步骤4: 虚拟筛选")
    log("=" * 60)

    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if scaler_path is None:
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if input_csv is None:
        input_csv = os.path.join(SRC_DIR, "data", "tcmbank", "tcm_smiles.csv")
    if output_csv is None:
        output_csv = os.path.join(RESULT_DIR, "筛选结果tcmbank_screening_results.csv")

    # 检查必要文件
    for path, name in [(model_path, "模型文件"), (scaler_path, "标准化器"), (input_csv, "输入化合物库")]:
        if not os.path.exists(path):
            log(f"{name}不存在: {path}", "ERROR")
            return None

    # 动态加载模型
    sys.path.insert(0, SRC_DIR)
    from finetune_wrn_regression import MLPRegressor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"使用设备: {device}")

    # 加载模型
    model = MLPRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载标准化器
    scaler = joblib.load(scaler_path)

    # 读取化合物库
    df = pd.read_csv(input_csv)
    log(f"加载化合物库: {len(df)} 个化合物")

    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    if smiles_col is None:
        smiles_col = df.columns[0]
        log(f"未找到SMILES列，使用第一列: {smiles_col}", "WARNING")

    smiles_list = df[smiles_col].dropna().tolist()

    # 生成指纹
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp, dtype=np.float32))
                valid_smiles.append(smi)
        except Exception:
            continue

    log(f"有效化合物: {len(valid_smiles)} / {len(smiles_list)}")

    # 标准化
    X = scaler.transform(np.array(fps))

    # 批量预测
    all_preds = []
    batch_size = 128
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

    results.to_csv(output_csv, index=False, encoding='utf-8')
    log(f"筛选结果已保存: {output_csv}")
    log(f"pIC50范围: {results['pred_pIC50'].min():.2f} ~ {results['pred_pIC50'].max():.2f}")

    # 显示Top 10
    log("\nTop 10 候选分子:")
    for i, row in results.head(10).iterrows():
        smi_short = row['SMILES'][:80] + '...' if len(row['SMILES']) > 80 else row['SMILES']
        log(f"  #{i+1}: pIC50={row['pred_pIC50']:.4f} | {smi_short}")

    return results


# =============================================================================
# 步骤5: 分子对接
# =============================================================================
def step_dock(screening_csv=None, top_n=3):
    """对Top候选进行分子对接验证"""
    log("=" * 60)
    log("步骤5: 分子对接")
    log("=" * 60)

    script = os.path.join(DOCK_DIR, "run_docking_final.py")
    if os.path.exists(script):
        log(f"运行对接脚本: {script}")
        log("注意: 对接需要 AutoDock Vina 可执行文件和 Open Babel")
        log("如果尚未安装，请参考 README.md 中的说明")
        try:
            subprocess.run([sys.executable, script], cwd=DOCK_DIR, check=True)
        except Exception as e:
            log(f"分子对接失败（自动跳过，不影响其他步骤）: {e}", "WARNING")
            log("提示: Linux/Mac 需自行下载对应平台的 Vina 可执行文件放入 docking/ 目录", "WARNING")
    else:
        log(f"未找到对接脚本: {script}", "WARNING")


# =============================================================================
# 步骤6: 生成报告
# =============================================================================
def step_generate_report(screening_csv=None, docking_csv=None, top_n=10):
    """生成最终结果报告和标准化结果文件

    默认输出 Top 10 候选（与交付的 result/results.csv 格式一致）。
    top_n=0 表示输出全部候选。
    """
    log("=" * 60)
    log("步骤6: 生成最终报告")
    log("=" * 60)

    if screening_csv is None:
        screening_csv = os.path.join(RESULT_DIR, "筛选结果tcmbank_screening_results.csv")
    if docking_csv is None:
        docking_csv = os.path.join(DOCK_DIR, "vina_final_results", "docking_summary.csv")

    # 读取筛选结果
    if not os.path.exists(screening_csv):
        log(f"筛选结果不存在: {screening_csv}", "ERROR")
        return

    df_screen = pd.read_csv(screening_csv)

    # 读取对接结果（如果有）
    df_dock = None
    if os.path.exists(docking_csv):
        df_dock = pd.read_csv(docking_csv)

    # 构建标准化的结果文件（默认 Top N 候选，与交付的 results.csv 格式一致）
    df_top = df_screen if top_n and top_n <= 0 else df_screen.head(top_n)
    results = []
    for idx, row in df_top.iterrows():
        smi = row['SMILES']
        pic50 = row['pred_pIC50']
        rank = int(idx) + 1

        # 计算分子属性
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(mol) if mol else None
        logp = Descriptors.MolLogP(mol) if mol else None
        hbd = Descriptors.NumHDonors(mol) if mol else None
        hba = Descriptors.NumHAcceptors(mol) if mol else None
        tpsa = Descriptors.TPSA(mol) if mol else None

        # 对接结合能（如果有）
        vina_aff = None
        vina_eval = ""
        if df_dock is not None and rank <= len(df_dock):
            dock_row = df_dock[df_dock['Rank'] == rank]
            if len(dock_row) > 0 and pd.notna(dock_row.iloc[0].get('Vina_Affinity')):
                vina_aff = dock_row.iloc[0]['Vina_Affinity']

        # 结合评估
        if vina_aff is not None:
            if vina_aff <= -7.0:
                vina_eval = "强结合"
            elif vina_aff <= -5.0:
                vina_eval = "中等结合"
            else:
                vina_eval = "较弱结合"

        # 备注
        note = ""
        if vina_aff is not None and vina_aff <= -7.0:
            note = "推荐优先研究"

        results.append({
            '候选编号': rank,
            '候选来源': 'TCMBank',
            'SMILES': smi,
            '分子量': round(mw, 2) if mw else None,
            'LogP': round(logp, 2) if logp else None,
            'HBD': hbd,
            'HBA': hba,
            'TPSA': round(tpsa, 2) if tpsa else None,
            '预测pIC50': round(float(pic50), 4) if pd.notna(pic50) else None,
            '预测IC50(nM)': round(10 ** (9 - float(pic50)), 2) if pd.notna(pic50) else None,
            'Vina结合能(kcal/mol)': round(float(vina_aff), 3) if vina_aff else None,
            '结合评估': vina_eval,
            '模型版本': 'v1.0-best_model.pth',
            '备注': note,
        })

    df_results = pd.DataFrame(results)

    # 保存CSV
    csv_output = os.path.join(RESULT_DIR, "results.csv")
    df_results.to_csv(csv_output, index=False, encoding='utf-8-sig')
    log(f"CSV结果已保存: {csv_output}")

    # 保存Excel
    try:
        xlsx_output = os.path.join(RESULT_DIR, "results.xlsx")
        df_results.to_excel(xlsx_output, index=False, engine='openpyxl')
        log(f"Excel结果已保存: {xlsx_output}")
    except Exception as e:
        log(f"Excel保存失败（可能缺少openpyxl）: {e}", "WARNING")

    # 生成Top10结构图
    top10 = df_results.head(10)
    mols = []
    legends = []
    for _, row in top10.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"#{row['候选编号']} pIC50={row['预测pIC50']:.3f}")

    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 200), legends=legends)
        img_path = os.path.join(RESULT_DIR, "top10_hits.png")
        img.save(img_path)
        log(f"Top10结构图已保存: {img_path}")

    # 打印汇总
    log("\n" + "=" * 60)
    log("最终筛选结果汇总")
    log("=" * 60)
    display_cols = ['候选编号', '预测pIC50', 'Vina结合能(kcal/mol)', '结合评估', '分子量', 'LogP']
    available_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[available_cols].head(20).to_string(index=False))

    log(f"\n完整结果已保存至: {csv_output}")

    return df_results


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='DrugCLIP-WRN: WRN解旋酶抑制剂虚拟筛选全流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode full                        # 一键运行全流程
  python main.py --mode train                       # 仅训练模型
  python main.py --mode screen                      # 仅虚拟筛选
  python main.py --mode dock --top_n 10             # 对接Top 10
  python main.py --mode report                      # 仅生成报告
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'screen', 'dock', 'report'],
                        help='运行模式 (默认: full)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (内含 wrn_pic50_train.csv 等，默认: Drug-The-Whole-Genome-main/data)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录 (默认: ./result)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='训练数据路径 (优先于 --data_dir)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入化合物库CSV (用于筛选模式)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果路径 (优先于 --output_dir)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='标准化器路径')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--seed', type=int, default=48,
                        help='随机种子 (默认: 48，可复现最佳验证RMSE 2.89)')
    parser.add_argument('--top_n', type=int, default=3,
                        help='对接候选数量 (默认: 3)')
    parser.add_argument('--report_top_n', type=int, default=10,
                        help='报告模式输出的候选数量 (默认: 10，0=全部)')

    args = parser.parse_args()

    log("=" * 60)
    log("DrugCLIP-WRN: WRN抑制剂虚拟筛选系统")
    log(f"运行模式: {args.mode}")
    log(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # ---- 解析数据目录和输出目录 ----
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        train_data_default = os.path.join(data_dir, "wrn_pic50_train.csv")
        screen_input_default = os.path.join(data_dir, "tcmbank", "tcm_smiles.csv")
        # 如果指定目录下没有数据文件，回退到内置默认位置
        if not os.path.exists(train_data_default):
            fallback = os.path.join(SRC_DIR, "data", "wrn_pic50_train.csv")
            if os.path.exists(fallback):
                log(f"[警告] {train_data_default} 不存在，回退到: {fallback}", "WARNING")
                train_data_default = fallback
        if not os.path.exists(screen_input_default):
            fallback = os.path.join(SRC_DIR, "data", "tcmbank", "tcm_smiles.csv")
            if os.path.exists(fallback):
                log(f"[警告] {screen_input_default} 不存在，回退到: {fallback}", "WARNING")
                screen_input_default = fallback
    else:
        train_data_default = None  # 使用 step_train 内置默认值
        screen_input_default = None  # 使用 step_screen 内置默认值

    if args.output_dir:
        result_dir = os.path.abspath(args.output_dir)
        os.makedirs(result_dir, exist_ok=True)
        global RESULT_DIR
        RESULT_DIR = result_dir
        screen_output_default = os.path.join(result_dir, "筛选结果tcmbank_screening_results.csv")
    else:
        screen_output_default = None  # 使用 step_screen 内置默认值

    try:
        if args.mode == 'full':
            step_fetch_data()
            step_prepare_data()
            step_train(data_path=args.data_path or train_data_default,
                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
            step_screen(input_csv=args.input or screen_input_default,
                        output_csv=args.output or screen_output_default,
                        model_path=args.model_path, scaler_path=args.scaler_path)
            step_dock(top_n=args.top_n)
            step_generate_report()
            log("\n[完成] 全流程完成!")

        elif args.mode == 'train':
            step_train(data_path=args.data_path or train_data_default,
                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)

        elif args.mode == 'screen':
            step_screen(input_csv=args.input or screen_input_default,
                        output_csv=args.output or screen_output_default,
                        model_path=args.model_path, scaler_path=args.scaler_path)

        elif args.mode == 'dock':
            step_dock(top_n=args.top_n)

        elif args.mode == 'report':
            step_generate_report(top_n=args.report_top_n)

    except KeyboardInterrupt:
        log("\n[警告] 用户中断", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"\n[错误] 运行出错: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
