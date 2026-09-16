"""Label Smoothing 消融实验入口。

这个脚本与 Baseline 的 train.py 分开：
- train.py：普通 CrossEntropyLoss Baseline；
- train_label_smoothing.py：只把损失函数改成 Label Smoothing 0.1。

除此之外，模型、数据划分、随机种子、优化器和训练轮数保持一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data.dataloaders import create_oxford_pet_dataloaders
from data.manifest import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_PATH
from models.model import NUMBER_OF_PET_CLASSES, build_resnet18
from train import config_to_dict, resolve_device, save_checkpoint, set_random_seed
from utils.engine import evaluate, train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "runs" / "label_smoothing_0.1"
)
LABEL_SMOOTHING = 0.1


def parse_args() -> argparse.Namespace:
    """定义 Label Smoothing 实验的参数。

    为了让实验公平，默认配置与 Baseline 保持一致；训练轮数默认 10。
    """

    parser = argparse.ArgumentParser(
        description="训练 CrossEntropy + Label Smoothing 0.1 实验组"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="关闭 batch 级训练和验证进度条。",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    # 保存到 checkpoint 后，可以明确知道这组实验只改变了损失函数。
    args.label_smoothing = LABEL_SMOOTHING
    return args


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"找不到数据目录：{args.data_root}")
    if not args.split_file.is_file():
        raise FileNotFoundError(f"找不到划分文件：{args.split_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pth"
    metrics_path = args.output_dir / "metrics.json"

    print(f"实验：Label Smoothing {LABEL_SMOOTHING}")
    print(f"训练设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    dataloaders = create_oxford_pet_dataloaders(
        dataset_root=args.data_root,
        split_path=args.split_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18(
        num_classes=NUMBER_OF_PET_CLASSES,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    # 这是本实验唯一改变的训练因素。
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    best_validation_accuracy = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            show_progress=args.show_progress,
        )
        validation_result = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            show_progress=args.show_progress,
        )

        writer.add_scalar("Loss/train", train_result.loss, epoch)
        writer.add_scalar("Loss/validation", validation_result.loss, epoch)
        writer.add_scalar(
            "Accuracy/train_top1",
            train_result.metrics["top1_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "Accuracy/validation_top1",
            validation_result.metrics["top1_accuracy"],
            epoch,
        )
        writer.add_scalar(
            "F1/validation_macro",
            validation_result.metrics["macro_f1"],
            epoch,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_top1_accuracy": train_result.metrics["top1_accuracy"],
                "train_top5_accuracy": train_result.metrics["top5_accuracy"],
                "train_macro_f1": train_result.metrics["macro_f1"],
                "validation_loss": validation_result.loss,
                "validation_top1_accuracy": validation_result.metrics[
                    "top1_accuracy"
                ],
                "validation_top5_accuracy": validation_result.metrics[
                    "top5_accuracy"
                ],
                "validation_macro_f1": validation_result.metrics["macro_f1"],
                "train_seconds": train_result.elapsed_seconds,
                "validation_seconds": validation_result.elapsed_seconds,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_result.loss:.4f} | "
            f"train top1 {train_result.metrics['top1_accuracy']:.4f} | "
            f"val loss {validation_result.loss:.4f} | "
            f"val top1 {validation_result.metrics['top1_accuracy']:.4f}"
        )

        if (
            validation_result.metrics["top1_accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = validation_result.metrics["top1_accuracy"]
            save_checkpoint(
                output_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_accuracy=best_validation_accuracy,
                args=args,
            )
            print(f"保存新的最佳模型：{checkpoint_path}")

    writer.close()
    metrics_path.write_text(
        json.dumps(
            {
                "config": config_to_dict(args),
                "best_validation_top1_accuracy": best_validation_accuracy,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"训练完成，指标文件：{metrics_path}")


if __name__ == "__main__":
    main()
