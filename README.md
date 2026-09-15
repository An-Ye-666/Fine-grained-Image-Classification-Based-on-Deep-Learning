# 基于深度学习的牛津宠物细粒度分类

本项目用于完成《科研项目组本科生考核任务书与操作指南》中的 72 小时考核任务，目标是在 Oxford-IIIT Pet 数据集上完成可复现的 ResNet-18 分类基线、消融实验、结果分析和工程交付。

## 当前状态

当前阶段：A0 工程骨架与项目控制台

当前状态：VERIFYING

本项目尚未开始数据下载、模型训练和实验。当前步骤先建立工程结构、需求映射、架构说明和项目状态记录。

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
├── models/                模型定义
├── utils/                 指标、日志和可视化工具
├── datasets/              原始数据，不提交到 Git
├── artifacts/             实验证据和交付物
├── docs/                  项目管理文档
├── train.py               训练入口
├── evaluate.py            评估入口
├── requirements.txt       依赖清单
└── README.md
```

Python 文件将在后续阶段逐步实现，当前只建立包边界，不提前写未经验证的算法代码。

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

环境和依赖将在 A1、A2 阶段确认后补充。README 的命令必须在全新环境中验证通过，避免写入未经测试的一键复现步骤。
