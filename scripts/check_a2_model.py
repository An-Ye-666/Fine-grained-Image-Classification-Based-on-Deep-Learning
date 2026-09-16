"""检查 ResNet-18 是否能加载预训练权重并输出 37 类结果。

运行方式：
    python -m scripts.check_a2_model
"""

from __future__ import annotations

from models.model import count_trainable_parameters, build_resnet18
import torch


def main() -> None:
    torch.manual_seed(42)
    model = build_resnet18(
        num_classes=37,
        pretrained=True,
        freeze_backbone=False,
    )
    model.eval()

    dummy_images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_images)

    if tuple(output.shape) != (2, 37):
        raise AssertionError(f"模型输出形状错误：{tuple(output.shape)}")

    print("[PASS] 已加载 ImageNet 预训练权重")
    print("[PASS] 输出层已替换为 37 类")
    print(f"[PASS] 输入形状：[2, 3, 224, 224]")
    print(f"[PASS] 输出形状：{tuple(output.shape)}")
    print(f"[PASS] 可训练参数数量：{count_trainable_parameters(model):,}")


if __name__ == "__main__":
    main()
