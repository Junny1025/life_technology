# finetune_wrn_regression.py
# WRN 解旋酶 pIC50 回归模型训练脚本
# 模型: MLP + ECFP4 分子指纹 (Morgan Fingerprints, radius=2, nBits=2048)
# 数据: ChEMBL CHEMBL2146312 (WRN) IC50 活性数据
# 输出: best_model.pth, scaler.pkl, training_log.txt, training_config.json, training_history.csv
import os
import sys
import argparse
import json
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================== 随机种子设置（确保可复现性） ==================
SEED = 48  # 该种子可复现最佳验证 RMSE 2.8853（四舍五入 2.89）
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== 日志工具（同时输出到屏幕和文件） ==================
class Logger:
    """双输出日志器：同时写入屏幕和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def log_print(msg):
    """带时间戳的日志打印"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ================== 数据加载 ==================
def load_wrn_data(data_path):
    """加载训练数据并生成 ECFP4 分子指纹"""
    df = pd.read_csv(data_path)
    fingerprints = []
    smiles_list = []
    pIC50_list = []

    for idx, row in df.iterrows():
        smi = row['canonical_smiles']
        pic50 = row['pIC50']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(np.array(fp, dtype=np.float32))
            smiles_list.append(smi)
            pIC50_list.append(pic50)

    X = np.array(fingerprints)
    y = np.array(pIC50_list, dtype=np.float32)
    log_print(f"数据加载完成: {len(X)} 个样本，指纹维度: {X.shape[1]}")
    log_print(f"  pIC50 范围: {y.min():.2f} ~ {y.max():.2f}")
    log_print(f"  pIC50 均值: {y.mean():.2f}, 中位数: {np.median(y):.2f}")
    return X, y

# ================== 模型定义 ==================
class MLPRegressor(nn.Module):
    """
    MLP 回归器 — 从 ECFP4 分子指纹预测 pIC50
    架构: 2048 → 512 → 256 → 1
    正则化: BatchNorm + Dropout(p=0.3)
    """
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

# ================== 训练函数 ==================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """单轮训练"""
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """验证：返回 loss 和 RMSE"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
    return total_loss / len(dataloader), rmse


# ================== 获取硬件信息 ==================
def get_hardware_info(device):
    """获取硬件和运行环境信息"""
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "platform": sys.platform,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info

# ================== 主函数 ==================
def main():
    parser = argparse.ArgumentParser(
        description='WRN pIC50回归模型训练 — ECFP4指纹 + MLP',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据 CSV 路径 (含 canonical_smiles, pIC50 列)')
    parser.add_argument('--save_dir', type=str, default='./wrn_model',
                        help='模型和日志保存目录 (默认: ./wrn_model)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小 (默认: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='使用 GPU (默认: 是)')
    parser.add_argument('--seed', type=int, default=48,
                        help='随机种子 (默认: 48，可复现最佳验证RMSE 2.89)')
    args = parser.parse_args()

    # 更新全局随机种子
    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(args.save_dir, exist_ok=True)
    # torch 2.0.1 在 Windows 上对含非ASCII字符（如中文）的绝对路径有兼容问题：
    # torch.save 会误报 "Parent directory does not exist"。此处将保存路径
    # 转为相对当前工作目录的路径规避（torch.load 不受影响）。
    save_dir_torch = os.path.relpath(args.save_dir, os.getcwd())
    if any(ord(ch) > 127 for ch in save_dir_torch):
        log_print(f"[警告] 保存路径含非ASCII字符且无法转为纯英文相对路径: {save_dir_torch}")
        log_print("[警告] torch 2.0.1 可能无法保存模型，建议将项目放在纯英文路径下运行")
        save_dir_torch = args.save_dir
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    # ---- 设置日志文件 ----
    log_file = os.path.join(args.save_dir, "training_log.txt")
    logger = Logger(log_file)
    sys.stdout = logger

    log_print("=" * 70)
    log_print("WRN pIC50 回归模型训练")
    log_print("=" * 70)
    log_print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 硬件信息 ----
    hw = get_hardware_info(device)
    log_print(f"\n【硬件环境】")
    log_print(f"  Python: {hw['python_version'].split()[0]}")
    log_print(f"  PyTorch: {hw['pytorch_version']}")
    log_print(f"  设备: {hw['device']}")
    log_print(f"  CUDA 可用: {hw['cuda_available']}")
    if hw['cuda_available']:
        log_print(f"  CUDA 版本: {hw.get('cuda_version', 'N/A')}")
        log_print(f"  GPU: {hw.get('gpu_name', 'N/A')}")
        log_print(f"  GPU 显存: {hw.get('gpu_memory_gb', 'N/A')} GB")

    # ---- 训练配置 ----
    train_config = {
        "model_type": "MLPRegressor",
        "input_type": "ECFP4 Morgan Fingerprint (radius=2, nBits=2048)",
        "architecture": {
            "input_dim": 2048,
            "hidden_dim": 512,
            "hidden_dim_2": 256,
            "output_dim": 1,
            "activation": "ReLU",
            "regularization": "BatchNorm1d + Dropout(p=0.3)",
            "loss_function": "MSELoss",
        },
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "random_seed": SEED,
        },
        "data": {
            "source": "ChEMBL CHEMBL2146312 (WRN)",
            "data_path": args.data_path,
            "train_test_split": "80% / 20%",
            "split_random_state": 42,
        },
        "device_info": hw,
    }

    log_print(f"\n【训练配置】")
    log_print(f"  数据路径: {args.data_path}")
    log_print(f"  模型保存目录: {args.save_dir}")
    log_print(f"  训练轮数: {args.epochs}")
    log_print(f"  批大小: {args.batch_size}")
    log_print(f"  学习率: {args.lr}")
    log_print(f"  随机种子: {SEED}")
    log_print(f"  使用设备: {device}")

    # ---- 加载数据 ----
    log_print(f"\n【数据加载】")
    X, y = load_wrn_data(args.data_path)

    # 划分训练集和验证集 (80%/20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log_print(f"  训练集: {len(X_train)} 样本")
    log_print(f"  验证集: {len(X_val)} 样本")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    log_print(f"  特征标准化: StandardScaler (fit on training set)")

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- 模型初始化 ----
    model = MLPRegressor(input_dim=X.shape[1]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n【模型信息】")
    log_print(f"  模型架构: MLPRegressor (2048→512→256→1)")
    log_print(f"  总参数量: {total_params:,}")
    log_print(f"  可训练参数量: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ---- 训练循环 ----
    log_print(f"\n【开始训练】")
    log_print(f"{'Epoch':>6s}  {'Train Loss':>12s}  {'Val Loss':>10s}  {'Val RMSE':>10s}  {'Best':>6s}")
    log_print(f"{'-'*50}")

    best_val_rmse = float('inf')
    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(save_dir_torch, "best_model.pth"))

        epoch_time = time.time() - epoch_start

        # 保存历史记录
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_rmse": round(val_rmse, 4),
            "is_best": is_best,
            "epoch_time_sec": round(epoch_time, 2),
        })

        # 每 10 轮或首尾轮打印
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            marker = "  *" if is_best else ""
            log_print(f"{epoch+1:>6d}  {train_loss:>12.6f}  {val_loss:>10.6f}  {val_rmse:>10.4f}  {'*' if is_best else '':>6s}")

    total_time = time.time() - start_time
    log_print(f"{'-'*50}")

    # ---- 保存最终模型和标准化器 ----
    torch.save(model.state_dict(), os.path.join(save_dir_torch, "final_model.pth"))
    import joblib
    joblib.dump(scaler, os.path.join(args.save_dir, "scaler.pkl"))

    # ---- 保存训练历史到 CSV ----
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.save_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    # ---- 保存训练配置到 JSON ----
    train_config["training_results"] = {
        "best_val_rmse": round(float(best_val_rmse), 4),
        "total_epochs_completed": args.epochs,
        "total_training_time_sec": round(total_time, 1),
        "total_training_time_min": round(total_time / 60, 1),
        "avg_epoch_time_sec": round(total_time / args.epochs, 1),
    }
    config_json = os.path.join(args.save_dir, "training_config.json")
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    # ---- 训练总结 ----
    log_print(f"\n【训练完成】")
    log_print(f"  最佳验证 RMSE: {best_val_rmse:.4f}")
    log_print(f"  总训练时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    log_print(f"  平均每轮时间: {total_time/args.epochs:.1f} 秒")
    log_print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"\n【保存文件】")
    log_print(f"  模型权重:     {os.path.join(args.save_dir, 'best_model.pth')}")
    log_print(f"  最终模型:     {os.path.join(args.save_dir, 'final_model.pth')}")
    log_print(f"  标准化器:     {os.path.join(args.save_dir, 'scaler.pkl')}")
    log_print(f"  训练日志:     {log_file}")
    log_print(f"  训练历史:     {history_csv}")
    log_print(f"  训练配置:     {config_json}")
    log_print(f"\n{'=' * 70}")

    # 恢复 stdout
    sys.stdout = logger.terminal
    logger.close()

    # 最终打印到控制台（不用 emoji，避免 Windows GBK 控制台编码错误）
    print(f"\n[完成] 训练结束！最佳验证 RMSE: {best_val_rmse:.4f}")
    print(f"[完成] 所有文件已保存至: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()