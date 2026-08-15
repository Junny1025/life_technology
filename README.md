# DrugCLIP-WRN: 基于深度学习的WRN解旋酶抑制剂虚拟筛选系统

## 项目概述

本项目构建了一套完整的 **WRN（Werner Syndrome Helicase）抑制剂虚拟筛选流程**，结合深度学习模型预测与分子对接验证，从大规模中药化合物库中筛选潜在的WRN抑制剂候选分子。

### 靶点信息
- **靶点名称**: WRN 解旋酶 (Werner Syndrome ATP-dependent Helicase)
- **ChEMBL ID**: CHEMBL2146312
- **UniProt ID**: Q14191
- **疾病关联**: MSI-H（微卫星高度不稳定）胃癌等多种癌症
- **PDB结构**: 8PFL

### 研究策略
1. **数据获取**: 从ChEMBL数据库获取WRN靶点的生物活性数据（IC50）
2. **模型训练**: 基于ECFP4分子指纹，训练MLP回归模型预测pIC50
3. **虚拟筛选**: 对TCMBank中药成分库（~37,000化合物）进行高通量预测
4. **分子对接**: 使用AutoDock Vina对Top候选分子进行结合模式验证

---

## 目录结构

```
DrugCLIP_Project/
│
├── README.md                              # 项目说明文档（本文件）
├── requirements.txt                       # Python依赖列表
├── main.py                                # 全流程一键运行入口
├── predict.py                             # 预测/推理入口（加载已训练模型）
├── model_card.md                          # 模型说明卡片
├── run.sh                                 # Linux/Mac 一键运行脚本
├── run.bat                                # Windows 一键运行脚本
│
├── Drug-The-Whole-Genome-main/            # 核心源代码
│   ├── fetch_wrn_api.py                   # 从ChEMBL API获取WRN活性数据
│   ├── check_wrn_data.py                  # 检查和统计WRN数据
│   ├── prepare_regression_data.py         # 数据预处理（pIC50回归训练集）
│   ├── finetune_wrn_regression.py         # 回归模型训练（MLP + ECFP4，含训练日志）
│   ├── predict_library.py                 # 化合物库批量预测
│   ├── tcmbank_pipeline.py                # TCMBank全自动流程（下载→提取→筛选）
│   ├── extract_smiles_fixed.py            # 从mol2文件提取SMILES
│   ├── analyze_hits.py                    # 命中化合物分析与结构可视化
│   ├── data/                              # 数据目录
│   │   ├── wrn_pic50_train.csv            # pIC50训练数据（169条）
│   │   └── tcmbank/                       # TCMBank化合物mol2文件
│   ├── wrn_model/                         # 训练产出
│   │   ├── best_model.pth                 # 最佳模型权重
│   │   ├── final_model.pth                # 最终模型权重
│   │   ├── scaler.pkl                     # 特征标准化器
│   │   ├── training_log.txt               # 训练日志（硬件、配置、每轮指标）
│   │   ├── training_history.csv           # 训练历史记录（CSV格式）
│   │   └── training_config.json           # 训练配置（JSON格式）
│   ├── docs/                              # 许可证文件
│   ├── wrn_chembl_ic50_api.csv            # ChEMBL API原始数据
│   ├── datachembl_wrn_raw.csv             # ChEMBL导出原始数据
│   └── tcmbank_screening_results.csv      # TCMBank虚拟筛选结果
│
├── docking/                               # 分子对接模块
│   ├── run_docking_final.py               # 对接主脚本（完整自动对接：受体清理+配体修复+Vina对接+汇总）
│   ├── clean_protein.py                   # 蛋白结构清理（PDB→受体）
│   ├── generate_pdbqt.py                  # PDBQT文件生成（RDKit）
│   ├── fix_pdbqt.py                       # PDBQT格式修复
│   ├── force_fix_pdbqt.py                 # PDBQT强制修复（BOM处理）
│   ├── vina_1.2.5_win.exe                 # AutoDock Vina可执行文件
│   ├── 8PFL.pdb                           # WRN蛋白原始结构（PDB）
│   ├── wrn_receptor.pdb                   # 清理后的受体结构（仅ATOM行）
│   └── vina_final_results/                # 对接成功结果（提交包附带，含完整日志）
│       ├── wrn_receptor.pdbqt             # 受体PDBQT
│       ├── docking_summary.csv            # 对接汇总表
│       ├── FINAL_REPORT.txt               # 对接完整报告
│       ├── candidate_01/                  # 候选1对接结果（结合能-7.033）
│       ├── candidate_02/                  # 候选2对接结果（结合能-5.180）
│       └── candidate_03/                  # 候选3对接结果（结合能-7.735）★最佳
│
├── result/                                # 结果输出目录
│   ├── results.csv                        # 标准化最终结果（提交格式，14字段，Top10）
│   ├── results.xlsx                       # 标准化最终结果（Excel版）
│   ├── 筛选结果tcmbank_screening_results.csv  # 虚拟筛选完整结果（29,857条）
│   ├── 分子对接报告FINAL_REPORT.txt        # 对接完整报告
│   ├── 结合能对比docking_summary.csv        # 对接汇总
│   ├── top10_hits.png                     # Top10化合物结构图
│   ├── 模型流程图framework.png             # 模型流程图
│   ├── 训练好的模型权重best_model.pth       # 训练好的模型权重
│   ├── WRN数据datachembl_wrn_raw.csv        # 原始ChEMBL数据
│   └── training/                          # 训练数据
│
└── notebooks/
    └── DrugCLIP_WRN_Pipeline.ipynb        # Jupyter Notebook全流程演示
```

---

## 环境配置

### 系统要求
- **操作系统**: Windows 10/11 64位 或 Linux
- **GPU**: NVIDIA显卡（推荐，显存 ≥ 4GB），CUDA 11.8+
- **Python**: 3.9
- **磁盘空间**: ≥ 10 GB（含TCMBank化合物库）

### 快速安装

#### 1. 安装Miniconda
从 https://docs.conda.io/en/latest/miniconda.html 下载并安装

#### 2. 创建虚拟环境
```bash
conda create -n drugclip python=3.9 -y
conda activate drugclip
```

#### 3. 安装PyTorch（GPU版）
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### 4. 安装RDKit
```bash
pip install rdkit-pypi==2022.09.5
```

#### 5. 安装其余依赖
```bash
pip install -r requirements.txt
```

#### 6. 安装Open Babel（用于分子对接格式转换）
```bash
conda install openbabel -c conda-forge -y
```

#### 7. 下载AutoDock Vina（分子对接）
从 https://github.com/ccsb-scripps/AutoDock-Vina/releases 下载对应平台的Vina可执行文件，放入 `docking/` 目录。

---

## 使用方法

### 方式一：一键运行全流程
```bash
python main.py --mode full --data_dir ./data --output_dir ./result
```

### 方式二：分步运行

#### 步骤1：获取WRN活性数据
```bash
cd DrugCLIP_Project/Drug-The-Whole-Genome-main
python fetch_wrn_api.py
```
输出: `wrn_chembl_ic50_api.csv` — WRN靶点的IC50数据

#### 步骤2：数据预处理
```bash
python prepare_regression_data.py
```
输出: `./data/wrn_pic50_train.csv` — 训练用pIC50数据（169条）

#### 步骤3：模型训练
```bash
python finetune_wrn_regression.py --data_path ./data/wrn_pic50_train.csv --epochs 50 --batch_size 16 --lr 1e-4 --seed 48 --use_gpu
```
输出: `./wrn_model/best_model.pth`、`scaler.pkl`（seed 48 可复现最佳验证 RMSE 2.89）

#### 步骤4：化合物库准备
```bash
python tcmbank_pipeline.py
```
自动下载TCMBank数据并提取SMILES

#### 步骤5：虚拟筛选
```bash
python predict.py --model_path ./wrn_model/best_model.pth --scaler_path ./wrn_model/scaler.pkl --input ./data/tcmbank/tcm_smiles.csv --output ./result/screening_results.csv
```
输出: 按pIC50降序排列的筛选结果

#### 步骤6：分子对接（可选）
```bash
cd ../docking
python run_docking_final.py  # 完整自动对接脚本（受体清理+配体修复+Vina对接）
# 需要 Open Babel: conda install openbabel -c conda-forge
```
输出: `vina_final_results/docking_summary.csv`

> **说明**: 提交包已附带完成的成功对接结果于 `docking/vina_final_results/`（含 vina.log 与 FINAL_REPORT.txt），无需重新对接即可评审。重跑对接会覆盖 `vina_final_results/`，如需保留原始结果请先备份。

### 方式三：仅预测（使用预训练模型）
```bash
python predict.py --model_path ./result/训练好的模型权重best_model.pth --input your_compounds.csv --output predictions.csv
```
输入CSV需包含 `SMILES` 列。

---

## 提交包运行说明（评审必读）

提交包解压后**不依赖任何本机绝对路径**：所有脚本均基于自身文件位置解析路径，解压到任意目录（含中文路径）均可直接运行。

### 环境准备
```bash
conda create -n drugclip python=3.9 -y
conda activate drugclip
# GPU版 PyTorch（无GPU可省略 --index-url 安装CPU版）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 方式A：Jupyter Notebook 全流程演示
```bash
cd notebooks
jupyter notebook DrugCLIP_WRN_Pipeline.ipynb     # 打开后 Kernel → Restart & Run All
```
- 已配置通用 `python3` 内核，任意 Python 3.9 环境均可运行，无需特殊配置
- 第一个单元格会自动定位项目根目录并打印，无需手工修改任何路径

### 方式B：命令行运行
```bash
python main.py --mode report    # 用提交包内已有结果生成标准化 results.csv/xlsx（Top10，无需网络；--report_top_n 0 可输出全部）
python main.py --mode train     # 用自带 169 条训练数据重新训练（内置数据，无需网络）
python main.py --mode screen    # 用提交包内置模型权重对 TCMBank 库虚拟筛选（无需网络）
python main.py --mode full      # 全流程（见下方说明）
```

### 关于 --mode full 的说明
- 步骤1（ChEMBL 数据获取）需要联网；**网络不可用时自动跳过，使用提交包内已附带的数据文件继续运行**
- 步骤5（分子对接）依赖 AutoDock Vina：Windows 已内置 `docking/vina_1.2.5_win.exe`；Linux/Mac 需自行从 [AutoDock-Vina releases](https://github.com/ccsb-scripps/AutoDock-Vina/releases) 下载对应平台二进制放入 `docking/` 目录，**缺失时自动跳过对接步骤，其余步骤不受影响**
- 模型权重、筛选结果、对接结果均已随提交包附带，无需重新计算即可评审

---

## 模型信息

### 模型架构
- **类型**: MLP回归器（多层感知机）
- **输入**: ECFP4分子指纹（Morgan指纹，半径2，2048位）
- **结构**: 2048 → 512 → 256 → 1（三层全连接 + BatchNorm + Dropout）
- **输出**: 预测pIC50值（-log10(IC50)，单位为M）

### 训练数据
- **来源**: ChEMBL数据库 (CHEMBL2146312)
- **样本数**: 169条（去重后）
- **pIC50范围**: 4.08 ~ 9.64
- **训练/验证划分**: 80%/20%（random_state=42）

### 性能指标
- **最佳验证RMSE**: 2.89（随机种子48复现，详见 wrn_model/training_log.txt）
- **设备**: NVIDIA GeForce RTX 4060 Laptop GPU 8GB / CUDA 11.8

### 适用范围
- WRN解旋酶pIC50/IC50活性预测
- 中药及天然产物虚拟筛选
- 苗头化合物发现与先导化合物优化参考

### 已知局限性
1. 训练数据量有限（169条），模型泛化能力受限于ChEMBL中已有的化学空间
2. 使用ECFP4指纹，对三维构象信息利用有限
3. pIC50预测RMSE约2.89，预测值仅供排序筛选参考，不能替代实验测定
4. 模型未考虑ADMET性质（吸收、分布、代谢、排泄、毒性）

更多模型细节请参见: [model_card.md](model_card.md)

---

## 创新贡献

1. **首个WRN靶点专用虚拟筛选模型**: 利用ChEMBL公开数据，构建了针对WRN解旋酶的pIC50回归预测模型
2. **中药数据库覆盖**: 将TCMBank（36,951种中药成分）纳入筛选范围，探索中药活性成分的WRN抑制潜力
3. **模型+对接双验证策略**: 深度学习预测结合AutoDock Vina分子对接，提高虚拟筛选的可靠性
4. **全流程自动化**: 从数据获取→模型训练→虚拟筛选→分子对接→结果报告，提供一键运行的完整pipeline

---

## 开源模型与工具使用说明

| 组件 | 名称/版本 | 用途 | 许可证 |
|------|----------|------|--------|
| 分子指纹 | RDKit 2022.09.5 | 化学信息学计算 | BSD |
| 深度学习框架 | PyTorch 2.0.1 | 模型训练与推理 | BSD |
| 分子对接 | AutoDock Vina 1.2.5 | 结合模式预测 | Apache 2.0 |
| 数据源 | ChEMBL | 生物活性数据 | CC BY-SA 3.0 |
| 化合物库 | TCMBank | 中药成分库 | 学术使用 |
| 上游框架 | Uni-Mol / DrugCLIP | 参考架构 | MIT / CC BY-NC 4.0 |
| 格式转换 | Open Babel | 化学文件格式转换 | GPL 2.0 |

**本项目模型**: 使用ECFP4指纹+MLP架构从头训练（未使用预训练权重），训练代码与模型权重均为本项目原创产出。

---

## 结果文件说明

最终筛选结果文件 `result/results.csv` 包含以下字段：

| 字段名 | 说明 | 示例 |
|--------|------|------|
| 候选编号 | 按pIC50降序排列的序号 | 1 |
| 候选来源 | 化合物原始来源 | TCMBank |
| SMILES | 规范SMILES结构表示 | [H]OC(=O)... |
| 分子量 | 化合物分子量（RDKit计算） | 763.84 |
| LogP | 脂水分配系数（RDKit计算） | 2.26 |
| HBD / HBA | 氢键供体/受体数 | 5 / 13 |
| TPSA | 拓扑极性表面积 | 215.22 |
| 预测pIC50 | 模型预测的pIC50值 | 5.3193 |
| 预测IC50(nM) | 换算的IC50（10^(9-pIC50)） | 4794.02 |
| Vina结合能(kcal/mol) | 分子对接结合自由能（Top3） | -7.033 |
| 结合评估 | 结合强度分级（≤-7.0强/≤-5.0中） | 强结合 |
| 模型版本 | 使用的模型版本 | best_model.pth |
| 备注 | 附加说明 | 推荐优先研究 |

---

## 引用

如使用本项目代码或结果，请引用：
- ChEMBL Database: https://www.ebi.ac.uk/chembl/
- TCMBank: https://www.tcmbank.cn/
- AutoDock Vina: Trott, O., & Olson, A. J. (2010). *J. Comput. Chem.*
- RDKit: https://www.rdkit.org/
- DrugCLIP: Gao, B. et al. "DrugCLIP: A Contrastive Protein-Molecule Representation Learning Framework for Virtual Screening"

---

## 许可证

- 本项目源代码: [Apache 2.0](DrugCLIP_Project/Drug-The-Whole-Genome-main/docs/LICENSE.md)
- 模型权重: 仅限学术研究使用
- TCMBank数据: 需遵守TCMBank使用条款

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目主页: [GitHub]
- 邮箱: [作者邮箱]
