"""Oxford-IIIT Pet 的 ResNet-18 训练入口。

这个脚本负责：
- 读取命令行参数；
- 创建 DataLoader；
- 创建预训练 ResNet-18；
- 训练和验证；
- 保存最佳模型、TensorBoard 日志和指标 JSON。

它不会读取测试集，测试集要留到最终评估阶段。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data.dataloaders import create_oxford_pet_dataloaders
from data.manifest import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_PATH
from models.model import NUMBER_OF_PET_CLASSES, build_resnet18
from utils.engine import evaluate, train_one_epoch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs" / "baseline"


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """把命令行中的 auto、cpu、cuda 转换为 PyTorch 设备。"""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境没有可用的 CUDA GPU")
    return torch.device(requested_device)


def config_to_dict(args: argparse.Namespace) -> dict[str, object]:
    """把命令行配置转换为可以写入 JSON 和 checkpoint 的普通字典。"""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args() -> argparse.Namespace:
    """定义训练脚本支持的命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 Oxford-IIIT Pet Baseline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Oxford-IIIT Pet 的图片和 annotations 所在目录。",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="A1 阶段生成的固定划分 JSON。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="checkpoint、日志和指标输出目录。",
    )
    parser.add_argument("--epochs", type=int, default=15)
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
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="不加载 ImageNet 预训练权重。",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 ResNet-18 特征提取层，只训练最终分类层。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="关闭 batch 级训练和验证进度条。",
    )
    parser.set_defaults(pretrained=True, show_progress=True)
    return parser.parse_args()


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_accuracy: float,
    args: argparse.Namespace,
) -> None:
    """保存模型参数、优化器状态和训练配置。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_accuracy": best_validation_accuracy,
            "number_of_classes": NUMBER_OF_PET_CLASSES,
            "config": config_to_dict(args),
        },
        output_path,
    )


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
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
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
