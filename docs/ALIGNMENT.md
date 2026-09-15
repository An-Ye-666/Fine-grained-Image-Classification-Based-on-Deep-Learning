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
