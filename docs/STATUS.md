# 项目状态

## 当前快照

```text
当前阶段：A3 消融实验
阶段状态：VERIFYING
当前目标：确认 Label Smoothing 0.1 的消融结果
已完成：独立实验脚本、10 轮训练和最佳模型复算
已知问题：仅完成单次实验；测试集仍未评估
下一步：项目负责人验收 A3，然后进入 A4 对齐
最后提交：d45581a
```

## 阶段进度

| 阶段 | 状态 | 产出 |
| --- | --- | --- |
| A0 工程骨架 | ACCEPTED | Git、目录、项目控制文档 |
| A1 数据与评估协议 | ACCEPTED | 固定划分、数据管线、指标和检查脚本 |
| A2 Baseline | ACCEPTED | 15 轮训练、最佳模型、训练日志 |
| A3 消融实验 | VERIFYING | Label Smoothing 0.1 对比结果 |
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

## A2 检查点

| ID | 检查项 | 状态 | 证据 |
| --- | --- | --- | --- |
| C2.1 | 构建预训练 ResNet-18 | DONE | `models/model.py` |
| C2.2 | 替换为 37 类输出层 | DONE | 输出形状 `[2, 37]` |
| C2.3 | 实现训练与验证循环 | DONE | `utils/engine.py` |
| C2.4 | 实现训练入口和日志 | DONE | `train.py` |
| C2.5 | 建立 A2 代码提交 | DONE | `1c80674` |
| C2.6 | 本地运行完整 Baseline | DONE | 完成 15 轮 |
| C2.7 | 重新加载最佳模型复算 | DONE | Top-1 92.47% |
| C2.8 | 建立 A2 结果提交 | DONE | `c9237f2` |

## A2 验证记录

```text
验证日期：2026-09-16
Python 编译：通过
ImageNet 预训练权重：加载成功
模型输入：[2, 3, 224, 224]
模型输出：[2, 37]
可训练参数：11,195,493
训练入口帮助信息：通过
batch 级训练/验证进度条：通过
实际训练：本地 CPU，15 轮完成
最佳轮次：4
最佳验证 Top-1：0.9247
最佳验证 Macro-F1：0.9244
最佳模型复算：一致
最终验证 Top-1：0.9056
最终训练 Top-1：0.9944
结论：第 4 轮后出现明显过拟合
```

## A3 检查点

| ID | 检查项 | 状态 | 证据 |
| --- | --- | --- | --- |
| C3.1 | 创建独立 Label Smoothing 脚本 | DONE | `train_label_smoothing.py` |
| C3.2 | 运行 10 轮单变量实验 | DONE | `artifacts/runs/label_smoothing_0.1` |
| C3.3 | 重新加载最佳模型复算 | DONE | Top-1 93.19% |
| C3.4 | 完成 Baseline 对比 | DONE | `docs/EXPERIMENTS.md` |
| C3.5 | 建立 A3 结果提交 | DONE | `d45581a` |

## A3 验证记录

```text
Baseline：Top-1 92.47%，Macro-F1 0.9244
Label Smoothing 0.1：Top-1 93.19%，Macro-F1 0.9315
Top-1 变化：+0.73 个百分点
Macro-F1 变化：+0.71 个百分点
最佳轮次：9
最佳模型复算：一致
结论：当前单次实验下有小幅提升
```

## 需求状态

| 需求 | 状态 |
| --- | --- |
| R1 数据与分层划分 | DONE |
| R2 数据预处理 | DONE |
| R3 Baseline | DONE |
| R4 训练记录 | DONE |
| R5 消融实验 | DONE |
| R6 评估与分析 | TODO |
| R7 Grad-CAM | TODO |
| R8 工程交付 | DOING |
| R9 技术报告 | TODO |
| R10 答辩准备 | TODO |
