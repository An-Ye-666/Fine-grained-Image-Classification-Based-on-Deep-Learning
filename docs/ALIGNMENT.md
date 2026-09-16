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
实现提交：`7f9fc84`

## A2 Baseline

```text
阶段编号：A2
阶段状态：ACCEPTED
对齐结论：项目负责人确认先执行模型构建和训练代码，再处理 Colab 实际运行
```

### 当前范围

- 构建 ImageNet 预训练 ResNet-18；
- 替换为 37 类输出层；
- 实现训练和验证循环；
- 实现命令行训练入口；
- 保存 checkpoint、TensorBoard 日志和指标 JSON；
- 验证模型输入输出形状。

### 暂不执行

- 不启动 15 轮真实训练；
- 不运行测试集；
- 不进行消融实验；
- 不生成 Grad-CAM；
- 暂不部署 Colab 代码和数据。

### 当前产出

```text
models/model.py
utils/engine.py
train.py
scripts/check_a2_model.py
```

### 已通过验证

```text
ImageNet 预训练权重：加载成功
输入形状：[2, 3, 224, 224]
输出形状：[2, 37]
可训练参数：11,195,493
train.py --help：通过
```

### 下一检查点

提交当前 A2 代码。之后对齐 Colab 的数据、代码同步、输出存储和依赖环境，再运行真实 Baseline。
实现提交：`1c80674`

### 实际执行结果

项目负责人最终选择本地 CPU，而非 Colab。15 轮训练已完成：

```text
最佳轮次：4
最佳验证 Top-1：0.9247
最佳验证 Top-5：0.9982
最佳验证 Macro-F1：0.9244
最终训练 Top-1：0.9944
最终验证 Top-1：0.9056
最佳模型复算：Top-1 0.9247，一致
```

第 4 轮后训练指标继续提升，但验证指标下降，说明存在过拟合。

## A3 消融实验

```text
阶段编号：A3
阶段状态：ACCEPTED
对齐结论：使用独立脚本比较普通 CrossEntropy 与 Label Smoothing 0.1
```

### 实验设置

```text
Baseline 比价范围：前 10 轮
实验组：Label Smoothing 0.1，训练 10 轮
模型：ResNet-18
Seed：42
Batch Size：32
学习率：1e-4
Weight Decay：1e-4
设备：本地 CPU
唯一变量：损失函数
```

### 结果

```text
Baseline Top-1：0.9247
Label Smoothing Top-1：0.9319
Baseline Macro-F1：0.9244
Label Smoothing Macro-F1：0.9315
最佳 Label Smoothing 轮次：9
最佳模型复算：一致
```

### 结论

在当前单次实验设置下，Label Smoothing 0.1 对 Top-1 和 Macro-F1 都带来了约
0.7 个百分点的小幅提升。该结果只代表当前固定数据划分、随机种子和训练预算，
不能直接推广为普遍结论。

### 阶段验收

项目负责人已确认 A3 阶段门通过。最终模型确定为 Label Smoothing 0.1 的第 9 轮
checkpoint，测试集获准在 A4 中评估一次。

## A4 评估与可解释性

```text
阶段编号：A4
阶段状态：EXECUTING
对齐结论：使用最终模型完成一次性测试评估、混淆矩阵和 Grad-CAM
```

### 范围

- 只评估测试集一次；
- 记录测试集 Loss、Top-1、Top-5 和 Macro-F1；
- 生成 37 类混淆矩阵；
- 找出主要混淆品种对；
- 选择预测正确和错误的样本各一张；
- 生成 Grad-CAM 热力图并分析关注区域。

### 暂不执行

- 不重新训练模型；
- 不根据测试集继续调参；
- 不加入第二组消融实验；
- 不撰写最终报告正文。

### 验收标准

- 测试集只被评估一次；
- 混淆矩阵覆盖全部 37 类；
- Grad-CAM 明确标注目标层和预测类别；
- 至少分析一个真实错误案例；
- 所有图和指标都保存到固定目录。

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
