"""训练和验证一个 epoch 的通用流程。

把循环放在这里，可以让 train.py 更专注于参数解析、模型创建和流程控制。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.metrics import compute_classification_metrics


@dataclass(frozen=True)
class EpochResult:
    """保存一个 epoch 的损失、指标和耗时。"""

    loss: float
    metrics: dict[str, float]
    elapsed_seconds: float


def _move_batch_to_device(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把一批图片和标签移动到 CPU 或 GPU。"""

    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> EpochResult:
    """训练一个 epoch，并返回平均损失和分类指标。"""

    model.train()
    start_time = time.perf_counter()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress_bar = (
        tqdm(
            dataloader,
            desc="Train",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if show_progress
        else dataloader
    )

    for images, labels in progress_bar:
        images, labels = _move_batch_to_device(images, labels, device)

        # 清空上一个 batch 留下的梯度。
        optimizer.zero_grad(set_to_none=True)

        # 前向传播：图片经过模型得到 [batch, 37] 的 logits。
        logits = model(images)
        loss = criterion(logits, labels)

        # 反向传播并更新参数。
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        # 保存到 CPU，避免把所有中间结果长期留在 GPU 显存中。
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{total_loss / total_samples:.4f}",
                refresh=False,
            )

    if show_progress:
        progress_bar.close()

    if total_samples == 0:
        raise ValueError("训练 DataLoader 没有产生任何样本")

    metrics = compute_classification_metrics(
        logits=torch.cat(all_logits, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=metrics,
        elapsed_seconds=time.perf_counter() - start_time,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> EpochResult:
    """在当前数据集上评估模型，不更新任何参数。"""

    model.eval()
    start_time = time.perf_counter()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress_bar = (
        tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if show_progress
        else dataloader
    )

    for images, labels in progress_bar:
        images, labels = _move_batch_to_device(images, labels, device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{total_loss / total_samples:.4f}",
                refresh=False,
            )

    if show_progress:
        progress_bar.close()

    if total_samples == 0:
        raise ValueError("验证 DataLoader 没有产生任何样本")

    metrics = compute_classification_metrics(
        logits=torch.cat(all_logits, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=metrics,
        elapsed_seconds=time.perf_counter() - start_time,
    )
