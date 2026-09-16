"""本项目统一使用的分类评价指标。

指标函数只接收模型输出 logits 和真实标签，不负责加载模型或读取数据。
这样训练、验证和测试阶段都可以调用同一套逻辑，避免指标口径不一致。
"""

from __future__ import annotations

import torch
from sklearn.metrics import f1_score
from torch import Tensor

NUMBER_OF_PET_CLASSES = 37


def _validate_logits_and_labels(logits: Tensor, labels: Tensor) -> None:
    """检查预测和标签的基本形状，尽早发现维度错误。"""

    if logits.ndim != 2:
        raise ValueError(f"logits 应为二维 [样本数, 类别数]，实际为 {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels 应为一维 [样本数]，实际为 {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits 和 labels 的样本数不一致："
            f"{logits.shape[0]} != {labels.shape[0]}"
        )
    if logits.shape[1] != NUMBER_OF_PET_CLASSES:
        raise ValueError(
            f"类别数应为 {NUMBER_OF_PET_CLASSES}，实际为 {logits.shape[1]}"
        )
    if logits.shape[0] == 0:
        raise ValueError("不能对空 batch 计算指标")


def top_k_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    """计算 Top-k Accuracy。

    Top-1 表示“最高分是否猜对”。
    Top-5 表示“正确类别是否出现在前五个候选中”。
    """

    _validate_logits_and_labels(logits, labels)
    if not 1 <= k <= NUMBER_OF_PET_CLASSES:
        raise ValueError(f"k 必须在 1 到 {NUMBER_OF_PET_CLASSES} 之间，实际为 {k}")

    # topk_indices 的形状是 [样本数, k]。
    topk_indices = logits.topk(k=k, dim=1).indices
    correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


@torch.no_grad()
def predict_labels(logits: Tensor) -> Tensor:
    """从每个样本的 37 个分数中选出最高分对应的类别。"""

    if logits.ndim != 2 or logits.shape[1] != NUMBER_OF_PET_CLASSES:
        raise ValueError(
            f"logits 形状应为 [样本数, {NUMBER_OF_PET_CLASSES}]，"
            f"实际为 {logits.shape}"
        )
    return logits.argmax(dim=1)


@torch.no_grad()
def compute_classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """统一计算 Top-1、Top-5 和 Macro-F1。"""

    _validate_logits_and_labels(logits, labels)
    predictions = predict_labels(logits).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()

    # Macro-F1 会先分别计算 37 类，再做等权平均。
    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(range(NUMBER_OF_PET_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return {
        "top1_accuracy": top_k_accuracy(logits, labels, k=1),
        "top5_accuracy": top_k_accuracy(logits, labels, k=5),
        "macro_f1": float(macro_f1),
    }
