# 阶段对齐记录

## A0 工程骨架与项目控制台

```text
阶段编号：A0
阶段状态：ACCEPTED
对齐结论：项目负责人已确认开始建立工程骨架
完成提交：b9eddd9
```

### 阶段目标

建立仓库、目录、需求映射、架构、状态看板和后续阶段的工作规则，使项目能够持续推进并保持上下文连贯。

### 必须完成

- 初始化 Git 仓库；
- 建立 `data/`、`models/`、`utils/` 和 `artifacts/` 边界；
- 建立需求、架构、状态、对齐、决策、实验和 AI 使用记录；
- 验证文档存在、内容可读、结构无冲突；
- 建立一个只包含 A0 内容的 Git 提交。

### 明确不做

- 不下载 Oxford-IIIT Pet；
- 不实现 Dataset 和 DataLoader；
- 不加载或训练 ResNet-18；
- 不运行消融实验；
- 不生成 Grad-CAM；
- 不写技术报告正文。

### 输出物

```text
README.md
.gitignore
00_项目协作与流程管理.md
docs/REQUIREMENTS.md
docs/ARCHITECTURE.md
docs/STATUS.md
docs/ALIGNMENT.md
docs/DECISIONS.md
docs/EXPERIMENTS.md
docs/AI_USAGE_LOG.md
docs/GLOSSARY.md
artifacts/README.md
data/__init__.py
models/__init__.py
utils/__init__.py
```

### 验收标准

- 文件结构和文档内容可读；
- 任务书中的必做项已映射到需求 ID；
- 架构图能够说明主要数据流；
- 状态文件明确当前阶段、问题和下一步；
- Git 仓库可以正常识别新增文件；
- A0 提交不包含算法实验代码。

### 验证结果

```text
文件清单：通过
Git 仓库识别：通过
git diff --check：通过
Python 版本：3.14.4，通过
pip 版本：26.0.1，通过
Python 包编译：通过
```

Python 训练依赖和实际训练环境属于 A1 的环境决策。

### 风险与预案

- 训练环境尚未确定：A0 保持环境无关，环境选择推迟到 A1；
- 数据划分协议存在歧义：推迟到 A1，通过数据和实验规则一起对齐；
- 文档过多可能增加负担：只维护必要内容，不复制任务书全文。

## A1 数据与评估协议

```text
阶段编号：A1
阶段状态：ACCEPTED
对齐结论：项目负责人确认执行完整 A1，并要求代码对 Python 入门学习者可读
验收结论：项目负责人确认 A1 未发现问题
```

### 阶段目标

建立固定、无重叠、可重复的 70/15/15 数据划分，完成图片 Transform、PyTorch Dataset、DataLoader 和统一评价指标，并通过自动检查。

### 必须完成

- 使用官方 7,349 个样本；
- 合并官方 `trainval` 与 `test` 清单后重新分层；
- 使用 `seed=42` 生成 70/15/15；
- 保存固定 JSON 划分文件；
- 训练集使用随机增强；
- 验证集和测试集使用确定性 Transform；
- 实现单张图片读取和 batch 读取；
- 实现 Top-1、Top-5 和 Macro-F1；
- 自动检查无重叠、类别覆盖、形状、标签和指标。

### 明确不做

- 不训练 ResNet-18；
- 不选择 Baseline 超参数；
- 不运行消融实验；
- 不生成 Grad-CAM；
- 不撰写报告正文。

### 输出物

```text
data/manifest.py
data/transforms.py
data/dataset.py
data/dataloaders.py
utils/metrics.py
scripts/check_a1_data.py
artifacts/splits/oxford_pet_seed42.json
requirements.txt
```

### 验收标准

- 官方样本总数为 7,349；
- 划分数量为 5144 / 1102 / 1103；
- 三份数据无重叠且覆盖全部 37 类；
- 单张图片形状为 `[3, 224, 224]`；
- batch 形状为 `[32, 3, 224, 224]`；
- 标签范围为 0 到 36；
- 验证集读取具备确定性；
- 指标函数可以正确计算；
- 相同 seed 重复生成相同划分文件。

### 验证结果

```text
python -m data.manifest：通过
python -m scripts.check_a1_data：全部通过
划分文件 SHA-256：13C4BD919AB1D5DF0FBB783727429B11327C13BD3D877F1EE8DEF1A897FA496C
本地 CUDA：不可用
Colab 环境：Linux、Tesla T4、CUDA 可用
延后事项：在 Colab 上重新运行 A1 和部署项目
```

### 阶段验收

项目负责人已确认 A1 未发现问题。建立 A1 提交后，进入 A2 Baseline 对齐。

## 后续阶段模板

```text
阶段编号：
阶段状态：
阶段目标：
必须完成：
明确不做：
技术方案：
输入与依赖：
输出物：
验收标准：
风险与预案：
需要确认的决定：
```
