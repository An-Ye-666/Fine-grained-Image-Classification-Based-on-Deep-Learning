# 实验记录

Baseline 已在本地 CPU 完成 15 轮训练。
最终模型已选择 Label Smoothing 0.1 的第 9 轮 checkpoint。

## 记录规则

- 同一组对比必须复用完全相同的训练、验证和测试划分；
- 一次只改变一个变量；
- 实验结论必须对应具体指标和日志路径；
- 失败的实验也要记录，不能只保留成功结果。

## 实验表

| Run ID | Commit | 配置 | Seed | Top-1 Acc | Top-5 Acc | Macro-F1 | 耗时 | 日志路径 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline-01 | `d48b6f3` | ResNet-18 + 基础增强 + CrossEntropy + AdamW | 42 | 92.47% | 99.82% | 0.9244 | 约 93 分钟 | `artifacts/runs/baseline` | 第 4 轮最佳，之后明显过拟合 |
| label-smoothing-01 | `d45581a` | ResNet-18 + 基础增强 + CrossEntropy(LS=0.1) + AdamW | 42 | 93.19% | 99.09% | 0.9315 | 约 27 分钟 | `artifacts/runs/label_smoothing_0.1` | 第 9 轮最佳，Top-1 和 Macro-F1 小幅提升 |
| final-test-01 | `7dc7fc7` | Label Smoothing 第 9 轮 + 标准测试评估 | 42 | 92.66% | 99.37% | 0.9258 | 约 19 秒 | `artifacts/analysis/final_model` | 测试集只评估一次；主要错误集中在相似品种对 |

## 计划实验

| 编号 | 实验 | 状态 |
| --- | --- | --- |
| E1 | Baseline：ResNet-18 + 基础增强 + CrossEntropy | DONE |
| E2 | 消融 A：Label Smoothing 0.1 | DONE |
| E3 | 消融 B：待 A3 对齐后确定 | TODO |
