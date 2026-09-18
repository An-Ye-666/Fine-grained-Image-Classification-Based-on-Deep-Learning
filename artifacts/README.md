# Artifacts

此目录保存可复查的实验证据和最终交付物。

```text
artifacts/
├── splits/       固定数据划分
├── runs/         训练指标和 TensorBoard 日志
├── analysis/     测试评估、混淆矩阵、预测明细和 Grad-CAM
└── report/       最终报告和报告图表
```

模型参数 `*.pth` 和原始数据集默认不提交到 Git。
