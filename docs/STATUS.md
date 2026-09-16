# 项目状态

## 当前快照

```text
当前阶段：A1 数据与评估协议
阶段状态：ACCEPTED
当前目标：确认数据划分、Transform、DataLoader 和指标均符合约定
已完成：固定划分、Dataset、Transform、DataLoader、基础指标和自动检查
已知问题：Colab 端重新运行和项目部署暂缓；依赖版本策略留到 A2
下一步：建立 A1 提交，然后进入 A2 Baseline 对齐
最后提交：7f9fc84
```

## 阶段进度

| 阶段 | 状态 | 产出 |
| --- | --- | --- |
| A0 工程骨架 | ACCEPTED | Git、目录、项目控制文档 |
| A1 数据与评估协议 | ACCEPTED | 固定划分、数据管线、指标和检查脚本 |
| A2 Baseline | DRAFT | 待生成阶段对齐说明 |
| A3 消融实验 | TODO | 待 A2 通过 |
| A4 评估与可解释性 | TODO | 待 A3 通过 |
| A5 工程化与复现 | TODO | 待 A4 通过 |
| A6 报告与答辩 | TODO | 待 A5 通过 |

## A0 检查点

| ID | 检查项 | 状态 | 证据 |
| --- | --- | --- | --- |
| C0.1 | 建立协作流程文档 | DONE | `00_项目协作与流程管理.md` |
| C0.2 | 初始化 Git 仓库 | DONE | `.git/` |
| C0.3 | 建立 Python 包边界 | DONE | `data/`、`models/`、`utils/` |
| C0.4 | 建立项目控制文档 | DONE | `docs/` |
| C0.5 | 验证仓库和文档 | DONE | 文件清单、Git 检查、Python 3.14.4 和包编译通过 |
| C0.6 | 建立 A0 提交 | DONE | `b9eddd9` |

## A0 验证记录

```text
验证日期：2026-09-15
文件清单：通过
Git 仓库识别：通过
空白格式检查：通过
Python 版本：3.14.4，通过
pip 版本：26.0.1，通过
Python 包编译：通过
说明：Codex 终端需要通过绝对路径调用 Python；训练依赖与 PyTorch 兼容性留到 A1 验证
```

## A1 检查点

| ID | 检查项 | 状态 | 证据 |
| --- | --- | --- | --- |
| C1.1 | 读取官方清单 | DONE | 官方有效样本 7,349 |
| C1.2 | 创建固定分层划分 | DONE | `artifacts/splits/oxford_pet_seed42.json` |
| C1.3 | 保存划分文件 | DONE | JSON 重复生成 SHA-256 一致 |
| C1.4 | 实现 Transform | DONE | `data/transforms.py` |
| C1.5 | 实现 Dataset 与 DataLoader | DONE | `data/dataset.py`、`data/dataloaders.py` |
| C1.6 | 实现评价指标 | DONE | `utils/metrics.py` |
| C1.7 | 自动检查数据协议 | DONE | `python -m scripts.check_a1_data` |
| C1.8 | 建立 A1 提交 | DONE | `7f9fc84` |

## A1 验证记录

```text
验证日期：2026-09-16
划分数量：5144 / 1102 / 1103
官方样本总数：7349
训练、验证、测试重叠：0
类别覆盖：每份数据均覆盖 37 类
单张图片形状：[3, 224, 224]
DataLoader batch：[32, 3, 224, 224]
标签范围：0 到 36
验证集 Transform 可重复：通过
Top-1、Top-5、Macro-F1：通过
本地训练设备：CPU，torch 2.14.0+cpu，CUDA 不可用
Colab 环境：Linux，torch 2.11.0+cu128，Tesla T4，CUDA 可用
阶段验收：项目负责人确认 A1 未发现问题
延后事项：在 Colab 上重新运行 A1 和部署项目
```

## 需求状态

| 需求 | 状态 |
| --- | --- |
| R1 数据与分层划分 | DONE |
| R2 数据预处理 | DONE |
| R3 Baseline | TODO |
| R4 训练记录 | TODO |
| R5 消融实验 | TODO |
| R6 评估与分析 | TODO |
| R7 Grad-CAM | TODO |
| R8 工程交付 | DOING |
| R9 技术报告 | TODO |
| R10 答辩准备 | TODO |
