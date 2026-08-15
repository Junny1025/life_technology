# tcmbank_pipeline.py
# 全自动：从 TCMBank 下载数据 → 提取 SMILES → 用训练好的模型预测 pIC50

import os
import sys
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import glob
import shutil
from urllib.parse import urlparse

# ==================== 配置 ====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TCM_DIR = os.path.join(DATA_DIR, 'tcmbank')
MOL2_DIR = os.path.join(TCM_DIR, 'mol2_files')
MODEL_DIR = os.path.join(PROJECT_DIR, 'wrn_model')

# TCMBank 下载链接（根据官网页面推断）
# 注意：实际链接可能需要从官网获取，这里提供常用格式
TCMBANK_BASE = "https://www.tcmbank.cn"
DOWNLOAD_PAGE = "https://www.tcmbank.cn/Download"

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TCM_DIR, exist_ok=True)
os.makedirs(MOL2_DIR, exist_ok=True)


# ==================== 第一步：下载数据 ====================
def download_file(url, dest_path, description="下载中"):
    """通用下载函数，带进度条"""
    if os.path.exists(dest_path):
        print(f"文件已存在: {dest_path}")
        return True
    
    print(f"{description}: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"[完成] 下载完成: {dest_path}")
        return True
    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        return False


def download_tcmbank_data():
    """
    从 TCMBank 下载数据
    注意：由于 TCMBank 的下载链接可能需要动态获取，
    这里提供两种方式：
    1. 直接下载（如果知道具体链接）
    2. 提示用户手动下载（最可靠）
    """
    print("\n" + "="*60)
    print("第一步：获取 TCMBank 数据")
    print("="*60)
    
    # TCMBank 的 mol2 文件通常是打包成 zip 的
    # 常见的下载链接格式（实际使用时可能需要从页面解析）
    mol2_zip_path = os.path.join(TCM_DIR, 'all_mol2.zip')
    
    # 尝试多个可能的下载链接
    possible_urls = [
        "https://www.tcmbank.cn/Download/All-mol2.zip",
        "https://tcmbank.cn/Download/All-mol2.zip",
        "https://www.tcmbank.cn/data/All-mol2.zip",
    ]
    
    downloaded = False
    for url in possible_urls:
        print(f"尝试: {url}")
        if download_file(url, mol2_zip_path, "下载 All-mol2"):
            downloaded = True
            break
        time.sleep(1)
    
    if not downloaded:
        print("\n" + "="*60)
        print("[警告] 自动下载失败，请手动下载")
        print("="*60)
        print(f"1. 打开浏览器访问: {DOWNLOAD_PAGE}")
        print("2. 点击下载 'All-mol2' 文件")
        print(f"3. 将下载的 zip 文件放到: {mol2_zip_path}")
        print("4. 按 Enter 继续...")
        input()
    
    # 检查文件是否存在
    if not os.path.exists(mol2_zip_path):
        print("[错误] 未找到 mol2 文件，请确保已下载并放到正确位置")
        return False
    
    return mol2_zip_path


# ==================== 第二步：解压并提取 SMILES ====================
def extract_smiles_from_mol2(mol2_path):
    """从单个 mol2 文件提取 SMILES"""
    try:
        mol = Chem.MolFromMol2File(mol2_path, removeHs=True)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return None
    except:
        return None


def extract_smiles_from_mol2_zip(zip_path, output_csv):
    """从 mol2 zip 包中批量提取 SMILES"""
    print("\n" + "="*60)
    print("第二步：从 mol2 文件提取 SMILES")
    print("="*60)
    
    if os.path.exists(output_csv):
        print(f"SMILES 文件已存在: {output_csv}")
        df = pd.read_csv(output_csv)
        print(f"已加载 {len(df)} 个化合物")
        return df
    
    # 解压
    extract_dir = MOL2_DIR
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 100:
        print(f"解压 {zip_path} 到 {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("[完成] 解压完成")
        except Exception as e:
            print(f"[错误] 解压失败: {e}")
            # 尝试使用更宽松的模式
            print("尝试使用更宽松的解压模式...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    try:
                        zip_ref.extract(member, extract_dir)
                    except Exception as e2:
                        print(f"跳过文件 {member.filename}: {e2}")
    
    # 查找所有 mol2 文件
    mol2_files = glob.glob(os.path.join(extract_dir, '**', '*.mol2'), recursive=True)
    if not mol2_files:
        mol2_files = glob.glob(os.path.join(extract_dir, '**', '*.mol2.gz'), recursive=True)
    
    print(f"找到 {len(mol2_files)} 个 mol2 文件")
    
    if not mol2_files:
        print("[警告] 未找到 mol2 文件，尝试从 Excel 文件获取 SMILES")
        return extract_from_excel()
    
    # 提取 SMILES
    smiles_list = []
    failed = 0
    
    for mol2_path in tqdm(mol2_files, desc="提取 SMILES"):
        smi = extract_smiles_from_mol2(mol2_path)
        if smi:
            smiles_list.append({
                'SMILES': smi,
                'source_file': os.path.basename(mol2_path)
            })
        else:
            failed += 1
    
    print(f"[完成] 成功提取 {len(smiles_list)} 个 SMILES，失败 {failed} 个")
    
    df = pd.DataFrame(smiles_list)
    
    # 去重
    df = df.drop_duplicates('SMILES')
    print(f"去重后: {len(df)} 个化合物")
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"[完成] SMILES 已保存: {output_csv}")
    
    return df


def extract_from_excel():
    """如果 mol2 文件无法提取，尝试从 Excel 文件获取"""
    print("\n尝试从 Excel 文件获取 SMILES...")
    
    excel_files = glob.glob(os.path.join(TCM_DIR, '*.xlsx'))
    ingredient_file = None
    
    for f in excel_files:
        if 'ingredient' in f.lower() or '成分' in f:
            ingredient_file = f
            break
    
    if not ingredient_file and excel_files:
        ingredient_file = excel_files[0]
    
    if ingredient_file:
        print(f"读取: {ingredient_file}")
        try:
            df = pd.read_excel(ingredient_file)
            print(f"列名: {df.columns.tolist()}")
            
            # 尝试找到 SMILES 列
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower() or 'SMILES' in col or '结构' in col:
                    smiles_col = col
                    break
            
            if smiles_col:
                smiles_list = df[smiles_col].dropna().tolist()
                result_df = pd.DataFrame({'SMILES': smiles_list})
                result_df = result_df.drop_duplicates()
                print(f"[完成] 从 Excel 提取 {len(result_df)} 个 SMILES")
                return result_df
        except Exception as e:
            print(f"[错误] 读取 Excel 失败: {e}")
    
    print("[警告] 无法提取 SMILES，请检查数据文件")
    return pd.DataFrame()


# ==================== 第三步：预测 ====================
def load_model_and_scaler(model_dir):
    """加载训练好的模型和标准化器"""
    model_path = os.path.join(model_dir, 'best_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        return None, None
    
    if not os.path.exists(scaler_path):
        print(f"[错误] 标准化器不存在: {scaler_path}")
        return None, None
    
    # 导入模型类
    try:
        from finetune_wrn_regression import MLPRegressor
    except ImportError:
        # 如果导入失败，使用内联定义
        import torch.nn as nn
        class MLPRegressor(nn.Module):
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = MLPRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    
    print("[完成] 模型和标准化器加载成功")
    return model, scaler, device


def smiles_to_fingerprint(smiles_list, batch_size=128):
    """将 SMILES 列表转换为 ECFP4 指纹"""
    fps = []
    valid_smiles = []
    
    for smi in tqdm(smiles_list, desc="生成指纹"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp, dtype=np.float32))
                valid_smiles.append(smi)
        except:
            continue
    
    return valid_smiles, np.array(fps)


def predict_batch(model, scaler, fingerprints, device, batch_size=128):
    """批量预测 pIC50"""
    if len(fingerprints) == 0:
        return []
    
    # 标准化
    X = scaler.transform(fingerprints)
    
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            all_preds.extend(preds)
    
    return all_preds


def run_screening(smiles_df, model, scaler, device, output_path):
    """执行虚拟筛选"""
    print("\n" + "="*60)
    print("第三步：虚拟筛选")
    print("="*60)
    
    smiles_list = smiles_df['SMILES'].tolist()
    print(f"待筛选化合物: {len(smiles_list)} 个")
    
    # 生成指纹
    valid_smiles, fingerprints = smiles_to_fingerprint(smiles_list)
    print(f"有效化合物: {len(valid_smiles)} 个")
    
    if len(valid_smiles) == 0:
        print("[错误] 没有有效化合物")
        return None
    
    # 预测
    preds = predict_batch(model, scaler, fingerprints, device)
    
    # 整理结果
    results = pd.DataFrame({
        'SMILES': valid_smiles,
        'pred_pIC50': preds
    })
    
    # 按 pIC50 降序排列（越高越好）
    results = results.sort_values('pred_pIC50', ascending=False).reset_index(drop=True)
    
    # 保存
    results.to_csv(output_path, index=False)
    print(f"[完成] 筛选结果已保存: {output_path}")
    
    # 统计
    print(f"\n预测统计:")
    print(f"   pIC50 范围: {results['pred_pIC50'].min():.2f} ~ {results['pred_pIC50'].max():.2f}")
    print(f"   pIC50 平均值: {results['pred_pIC50'].mean():.2f}")
    print(f"   pIC50 中位数: {results['pred_pIC50'].median():.2f}")
    
    # 显示 Top 20
    print(f"\nTop 20 候选分子:")
    print(results.head(20).to_string(index=False))
    
    return results


# ==================== 主流程 ====================
def main():
    print("\n" + "="*60)
    print("TCMBank 中药化合物库 → WRN 抑制剂虚拟筛选 全自动流程")
    print("="*60)
    
    # 检查模型是否存在
    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print("\n[错误] 未找到训练好的模型!")
        print(f"请先运行: python finetune_wrn_regression.py --data_path ./data/wrn_pic50_train.csv --epochs 50")
        return
    
    # 1. 下载数据
    zip_path = download_tcmbank_data()
    if not zip_path:
        print("[错误] 数据下载失败")
        return
    
    # 2. 提取 SMILES
    smiles_csv = os.path.join(TCM_DIR, 'tcm_smiles.csv')
    smiles_df = extract_smiles_from_mol2_zip(zip_path, smiles_csv)
    
    if len(smiles_df) == 0:
        print("[错误] 未能提取任何 SMILES")
        return
    
    # 3. 加载模型
    model, scaler, device = load_model_and_scaler(MODEL_DIR)
    if model is None:
        return
    
    # 4. 虚拟筛选
    output_path = os.path.join(PROJECT_DIR, 'tcmbank_screening_results.csv')
    results = run_screening(smiles_df, model, scaler, device, output_path)
    
    if results is not None:
        print("\n" + "="*60)
        print("[完成] 全流程完成!")
        print(f"[文件] 结果文件: {output_path}")
        print("="*60)


if __name__ == "__main__":
    main()