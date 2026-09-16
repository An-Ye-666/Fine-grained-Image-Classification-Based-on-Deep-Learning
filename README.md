# 基于深度学习的牛津宠物细粒度分类

本项目用于完成《科研项目组本科生考核任务书与操作指南》中的 72 小时考核任务，目标是在 Oxford-IIIT Pet 数据集上完成可复现的 ResNet-18 分类基线、消融实验、结果分析和工程交付。

## 当前状态

当前阶段：A1 数据与评估协议

当前状态：ACCEPTED

数据划分、图片 Transform、Dataset、DataLoader 和基础指标已经实现并通过自动检查。
A1 已由项目负责人验收。本地 PyTorch 是 CPU 版本；Colab 云端 Tesla T4 已验证可用，
但项目代码和数据在 Colab 上的部署留到 A2 处理。

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
├── datasets/              原始数据，不提交到 Git
├── artifacts/             实验证据和交付物
├── docs/                  项目管理文档
├── train.py               训练入口
├── evaluate.py            评估入口
├── requirements.txt       依赖清单
└── README.md
```

模型与训练代码将在 A2 阶段实现。A1 当前只负责数据入口和评价协议。

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

```bash
pip install -r requirements.txt
python -m data.manifest
python -m scripts.check_a1_data
```

如果本地数据缺失，可以运行：

```bash
python -m data.manifest --download
```
