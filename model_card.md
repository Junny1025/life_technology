# 模型说明卡片 (Model Card) — DrugCLIP-WRN

## 基本信息

| 属性 | 值 |
|------|-----|
| **模型名称** | DrugCLIP-WRN-MLP-Regressor |
| **模型版本** | v1.0 (best_model.pth) |
| **模型类型** | MLP回归器（多层感知机） |
| **发布日期** | 2026-08-10 |
| **框架** | PyTorch 2.0.1 |
| **训练设备** | NVIDIA GeForce RTX 4060 Laptop GPU 8GB, CUDA 11.8 |
| **作者/团队** | [作者/团队名称] |

---

## 模型架构

```
MLPRegressor(
  输入层:   Linear(2048, 512)
  批归一化: BatchNorm1d(512)
  激活函数: ReLU
  Dropout:  p=0.3
  隐藏层1:  Linear(512, 256)
  批归一化: BatchNorm1d(256)
  激活函数: ReLU
  Dropout:  p=0.3
  输出层:   Linear(256, 1)
)
```

### 关键参数
- **总参数量**: 1,182,209 (~1.18M)
- **输入特征**: ECFP4分子指纹 (Morgan Fingerprints, radius=2, nBits=2048)
- **标准化**: StandardScaler（基于训练集均值和标准差）

---

## 训练数据

### 数据来源
- **数据库**: ChEMBL v34
- **靶点**: CHEMBL2146312 (WRN — Werner Syndrome Helicase)
- **活性类型**: IC50
- **获取日期**: 2026-08-09
- **原始记录数**: 9,075条（含所有活性类型）
- **IC50记录数**: 233条（有效数值）
- **去重后样本数**: 169条

### 数据预处理
1. 筛选IC50类型活性数据
2. 转换IC50 (nM) → pIC50: `pIC50 = 9 - log10(IC50_nM)`
3. 按SMILES去重（保留IC50最低/活性最强的记录）
4. 过滤异常值（pIC50 ∈ [4.0, 10.0]）
5. 训练/验证划分: 80%/20% (random_state=42)

### 数据分布
| 指标 | 值 |
|------|-----|
| pIC50范围 | 4.08 ~ 9.64 |
| pIC50均值 | 6.24 |
| pIC50中位数 | 6.25 |
| 训练集样本 | 135 |
| 验证集样本 | 34 |

---

## 性能指标

| 指标 | 值 |
|------|-----|
| 最佳验证RMSE | 2.89 |
| 训练设备 | cuda |

**注意**: RMSE=2.89 表示预测值在pIC50尺度上平均偏差约2.89个单位。该精度足以：
- 从大量化合物中排序筛选Top候选
- 区分高活性（pIC50 > 7）与低活性（pIC50 < 5）化合物
- **不适用于**: 精确的pIC50定量预测（需要实验验证）

---

## 适用范围 (Intended Use)

### 适用场景 ✅
1. **WRN抑制剂的虚拟筛选**: 从中药/天然产物库中初步筛选候选分子
2. **苗头化合物优先排序**: 为实验验证提供优先级参考
3. **结构-活性关系参考**: 辅助药物化学家进行骨架跃迁和结构改造
4. **学术研究**: 计算化学、药物设计教学与科研

### 不适用场景 ❌
1. **临床决策**: 不可替代体外/体内实验
2. **其他靶点预测**: 仅针对WRN解旋酶，不可迁移到其他蛋白
3. **ADMET预测**: 不包括吸收、代谢、毒性等成药性评估
4. **精确IC50定量**: 预测值存在系统偏差，仅供排序参考
5. **商业用途**: 模型权重采用CC BY-NC 4.0许可证（需联系作者获取商业授权）

---

## 已知局限性

### 数据层面
1. **训练数据量有限**: 仅169个样本，可能未覆盖WRN活性相关的全部化学空间
2. **数据来源单一**: 仅使用ChEMBL数据，可能存在数据库偏差
3. **活性悬崖**: 对结构相似但活性差异大的分子可能预测不准

### 方法层面
1. **2D指纹局限**: ECFP4只能捕获拓扑信息，无法反映三维构象、结合模式
2. **靶点柔性**: 未考虑WRN蛋白的构象变化和诱导契合效应
3. **无蛋白结构信息**: 预测不依赖蛋白口袋信息，仅从配体角度建模

### 评估层面
1. **验证集较小**（34样本）：RMSE估计存在不确定性
2. **无外部测试集**: 未在独立外部数据集上验证泛化能力

---

## 使用方法

### 加载模型
```python
import torch
import joblib
from finetune_wrn_regression import MLPRegressor

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPRegressor().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 加载标准化器
scaler = joblib.load('scaler.pkl')
```

### 预测单分子
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def predict_single(smiles, model, scaler, device):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    X = scaler.transform([np.array(fp, dtype=np.float32)])
    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(device)).item()
    return pred

pic50 = predict_single("CCO", model, scaler, device)
print(f"预测pIC50: {pic50:.4f}")
```

---

## 伦理考量

1. **数据隐私**: 所有训练数据来自公开数据库ChEMBL，不涉及患者隐私
2. **偏见与公平性**: 训练数据可能存在化合物骨架偏好，需注意结果的化学多样性
3. **透明度**: 完整代码、数据处理流程和模型架构均已公开
4. **安全**: 本模型仅用于辅助药物发现研究，不可直接用于临床诊断或治疗决策

---

## 变更日志

| 版本 | 日期 | 变更说明 |
|------|------|---------|
| v1.0 | 2026-08-10 | 初始版本，基于169条ChEMBL WRN IC50数据的MLP回归模型 |

---

## 引用格式

如使用本模型，请引用：
```
[作者]. DrugCLIP-WRN: A Deep Learning Model for WRN Helicase Inhibitor Prediction [Model Card]. 2026.
```
