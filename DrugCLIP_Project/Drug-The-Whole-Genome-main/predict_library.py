# predict_library.py
import torch
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from finetune_wrn_regression import MLPRegressor
import os

def predict_smiles(smiles_list, model_path, scaler_path, device='cuda', batch_size=128):
    """对 SMILES 列表进行批量预测"""
    
    # 加载模型
    model = MLPRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载标准化器
    scaler = joblib.load(scaler_path)
    
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
        except:
            continue
    
    print(f"有效 SMILES: {len(valid_smiles)} / {len(smiles_list)}")
    
    # 标准化
    X = scaler.transform(np.array(fps))
    
    # 分批预测
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            preds = model(batch).cpu().numpy()
            all_preds.extend(preds)
    
    # 排序结果
    results = pd.DataFrame({
        'SMILES': valid_smiles,
        'pred_pIC50': all_preds
    }).sort_values('pred_pIC50', ascending=False)
    
    return results

def main():
    # 检查是否有待预测的库文件
    input_file = './compound_library.csv'  # 需包含 SMILES 列
    output_file = './wrn_screening_results.csv'
    
    if not os.path.exists(input_file):
        print(f"请准备化合物库文件: {input_file}")
        print("格式: 包含 'SMILES' 列的 CSV 文件，每行一个分子")
        return
    
    print("加载化合物库...")
    df = pd.read_csv(input_file)
    
    if 'SMILES' not in df.columns:
        print("错误: 文件需要包含 'SMILES' 列")
        return
    
    smiles_list = df['SMILES'].dropna().tolist()
    print(f"化合物数量: {len(smiles_list)}")
    
    # 预测
    results = predict_smiles(
        smiles_list,
        './wrn_model/best_model.pth',
        './wrn_model/scaler.pkl'
    )
    
    # 保存结果
    results.to_csv(output_file, index=False)
    print(f"结果已保存至: {output_file}")
    print(f"Top 20 候选分子:")
    print(results.head(20))
    print(f"预测 pIC50 范围: {results['pred_pIC50'].min():.2f} ~ {results['pred_pIC50'].max():.2f}")

if __name__ == "__main__":
    main()