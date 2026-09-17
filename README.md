# 基于深度学习的牛津宠物细粒度分类

本项目用于完成《科研项目组本科生考核任务书与操作指南》中的 72 小时考核任务，目标是在 Oxford-IIIT Pet 数据集上完成可复现的 ResNet-18 分类基线、消融实验、结果分析和工程交付。

## 当前状态

当前阶段：A5 工程化与复现

当前状态：EXECUTING

A4 已验收。最终模型测试集 Top-1 为 92.66%，混淆矩阵和 Grad-CAM 已归档。
A5 正在整理运行说明、日志、依赖和最终工程检查，不会重复评估测试集。

## 项目目标

- 完成 37 类猫狗品种的细粒度图像分类；
- 跑通 ResNet-18 Baseline；
- 完成 1 到 2 组单变量消融实验；
- 生成 Accuracy、Macro-F1、混淆矩阵和 Grad-CAM；
- 提供模块化代码、README、训练日志和 2 到 3 页报告。

## 计划结构

```text
.
├── data/                  数据集加载、划分和 Transform
│   ├── manifest.py        官方清单解析与固定划分
│   ├── transforms.py      图片 Transform
│   ├── dataset.py         PyTorch Dataset
│   └── dataloaders.py     DataLoader 构造
├── models/                模型定义
├── utils/                 指标、日志和可视化工具
├── scripts/               可重复运行的检查脚本
├── data/oxford-iiit-pet/  原始数据，不提交到 Git
├── artifacts/             实验证据和交付物
├── docs/                  项目管理文档
├── train.py               训练入口
├── evaluate.py            评估入口
├── train_label_smoothing.py Label Smoothing 消融入口
├── requirements.txt       依赖清单
└── README.md
```

代码已经完成数据划分、Baseline、Label Smoothing 消融、最终测试评估和 Grad-CAM。
原始数据和模型权重不会提交到 Git，但训练日志、指标和可视化结果会进入 artifacts。

## 项目文档

- [协作与流程管理](00_项目协作与流程管理.md)
- [需求清单](docs/REQUIREMENTS.md)
- [系统架构](docs/ARCHITECTURE.md)
- [当前状态](docs/STATUS.md)
- [阶段对齐](docs/ALIGNMENT.md)
- [技术决策](docs/DECISIONS.md)
- [实验记录](docs/EXPERIMENTS.md)
- [AI 使用记录](docs/AI_USAGE_LOG.md)
- [术语表](docs/GLOSSARY.md)

## 快速开始

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

当前验证环境是 Windows、Python 3.14.4 和 CPU 版 PyTorch 2.14.0。

### 2. 准备数据

如果 `data/oxford-iiit-pet` 已经存在：

```bash
python -m data.manifest
```

如果本地缺少数据，允许 torchvision 下载：

```bash
python -m data.manifest --download
```

### 3. 检查数据和模型

```bash
python -m scripts.check_a1_data
python -m scripts.check_a2_model
```

### 4. 训练实验

训练普通 Baseline：

```bash
python train.py --epochs 15 --batch-size 32 --device cpu
```

训练 Label Smoothing 0.1 消融实验：

```bash
python train_label_smoothing.py --epochs 10 --batch-size 32 --device cpu
```

### 5. 验证集评估

```bash
python evaluate.py `
  --checkpoint artifacts/runs/label_smoothing_0.1/best_model.pth `
  --split val `
  --device cpu
```

### 6. 最终测试分析

测试集已经评估过一次。`scripts/analyze_model.py` 会检查
`artifacts/analysis/final_model/test_metrics.json`，存在时拒绝重复运行：

```bash
python -m scripts.analyze_model --allow-test --device cpu
```

## 最终结果

| 模型 | 验证 Top-1 | 测试 Top-1 | 测试 Macro-F1 |
| --- | ---: | ---: | ---: |
| ResNet-18 Baseline | 92.47% | 未使用 | 未使用 |
| ResNet-18 + Label Smoothing 0.1 | 93.19% | 92.66% | 0.9258 |

最终图表位于 `artifacts/analysis/final_model/`，训练日志位于
`artifacts/runs/baseline/` 和 `artifacts/runs/label_smoothing_0.1/`。

训练时默认显示 batch 级进度条、当前 Loss 和平均 Loss；使用 `--no-progress` 可以关闭。

如果本地数据缺失，可以运行：

```bash
python -m data.manifest --download
```
