"""ResNet Grad-CAM 可视化。

Grad-CAM 会记录目标卷积层的前向特征和反向梯度，再用梯度权重对特征图加权，
最终得到一张与输入图片同尺寸的热力图。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """为指定卷积层生成 Grad-CAM 热力图。"""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """保存目标层的输出特征图。"""

        self.activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """保存目标层的反向梯度。"""

        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """删除 hook，避免影响后续模型运行。"""

        self.forward_handle.remove()
        self.backward_handle.remove()

    @torch.enable_grad()
    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成指定类别的 Grad-CAM。

        返回值：
        - cam：形状 [224, 224]，数值范围 0 到 1；
        - probabilities：模型对 37 类的 Softmax 概率。
        """

        if image_tensor.ndim != 3:
            raise ValueError(
                f"单张图片应为 [3, H, W]，实际为 {image_tensor.shape}"
            )
        if not 0 <= target_class < 37:
            raise ValueError(f"target_class 越界：{target_class}")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor.unsqueeze(0))
        logits[0, target_class].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("没有捕获到目标层的特征或梯度")

        # 每个通道的梯度取平均，作为该通道特征图的重要程度权重。
        channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        return cam.detach().cpu(), probabilities

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args: object) -> None:
        self.remove_hooks()
