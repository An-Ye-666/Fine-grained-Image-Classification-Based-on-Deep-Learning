"""ResNet-18 模型构建函数。

这个文件只负责创建模型：
1. 加载 ImageNet 预训练权重；
2. 把原来的 1000 类输出层替换为 37 类输出层；
3. 返回一个可以直接训练的 PyTorch 模型。
"""

from __future__ import annotations

from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

NUMBER_OF_PET_CLASSES = 37


def build_resnet18(
    num_classes: int = NUMBER_OF_PET_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """创建用于 Oxford-IIIT Pet 分类的 ResNet-18。

    参数：
    - num_classes：输出类别数，本项目为 37；
    - pretrained：是否加载 ImageNet 预训练权重；
    - freeze_backbone：是否冻结特征提取部分，只训练最后的分类层。
    """

    if num_classes <= 0:
        raise ValueError(f"num_classes 必须大于 0，实际为 {num_classes}")

    # 使用 torchvision 的新式 weights 参数，避免使用已过时的 pretrained=True。
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # 原始 ResNet-18 最后输出 1000 类；本数据集需要输出 37 类。
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    if freeze_backbone:
        # 只允许 fc 层更新，其他参数保持 ImageNet 预训练值。
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad = parameter_name.startswith("fc.")

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """统计当前会被优化器更新的参数数量。"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
