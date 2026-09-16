# 最终模型测试分析

## 测试指标

- Loss：0.4036
- Top-1：0.9266
- Top-5：0.9937
- Macro-F1：0.9258

## 主要混淆品种对

| 真实类别 | 预测类别 | 数量 | 占真实类别比例 |
| --- | --- | ---: | ---: |
| Egyptian_Mau | Bengal | 5 | 0.1724 |
| american_pit_bull_terrier | staffordshire_bull_terrier | 5 | 0.1667 |
| Siamese | Birman | 4 | 0.1333 |
| american_pit_bull_terrier | american_bulldog | 3 | 0.1000 |
| Bengal | Abyssinian | 2 | 0.0667 |
| British_Shorthair | Ragdoll | 2 | 0.0667 |
| Persian | Maine_Coon | 2 | 0.0667 |
| great_pyrenees | samoyed | 2 | 0.0667 |
| miniature_pinscher | american_pit_bull_terrier | 2 | 0.0667 |
| miniature_pinscher | chihuahua | 2 | 0.0667 |

## 案例

- 正确案例：`english_setter_21`，真实类别 `english_setter`，预测类别 `english_setter`
- 错误案例：`american_pit_bull_terrier_46`，真实类别 `american_pit_bull_terrier`，预测类别 `staffordshire_bull_terrier`

## 图表

- `confusion_matrix.png`
- `gradcam_correct.png`
- `gradcam_error.png`
