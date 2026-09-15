# Artifacts

此目录用于保存可复查的实验证据和交付物。

计划使用的子目录：

```text
artifacts/
├── checkpoints/   模型权重，文件较大，默认不提交到 Git
├── logs/          训练日志
├── figures/       训练曲线、混淆矩阵、Grad-CAM
└── reports/       最终报告和导出文件
```

每个实验必须在 `docs/EXPERIMENTS.md` 中记录实验编号、配置、结果和产物路径。
