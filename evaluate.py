"""使用保存的最佳模型评估验证集。

这个脚本不会训练模型，也不会更新参数：
1. 读取 best_model.pth；
2. 重建结构相同的 ResNet-18；
3. 加载最佳权重；
4. 在验证集上计算 Loss、Top-1、Top-5 和 Macro-F1。

测试集在本阶段保持锁定，脚本会拒绝使用 test 划分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataloaders import create_dataloader
from data.dataset import OxfordPetDataset
from data.manifest import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_PATH
from data.transforms import build_eval_transform
from models.model import NUMBER_OF_PET_CLASSES, build_resnet18
from utils.engine import evaluate as evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "baseline" / "best_model.pth"
)
DEFAULT_RESULT_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "baseline" / "validation_metrics.json"
)


def resolve_device(requested_device: str) -> torch.device:
    """把 auto、cpu 或 cuda 转换为 PyTorch 设备。"""

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境没有可用的 CUDA GPU")
    return torch.device(requested_device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已保存的 Baseline 模型")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="训练阶段保存的最佳模型文件。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Oxford-IIIT Pet 数据目录。",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="固定数据划分 JSON。",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="本阶段只允许 val，test 保持锁定。",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="关闭验证进度条。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="验证指标 JSON 的输出路径。",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split == "test":
        raise RuntimeError(
            "测试集当前保持锁定，不能使用 evaluate.py 进行评估。"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到模型文件：{args.checkpoint}")

    device = resolve_device(args.device)
    print(f"评估设备：{device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = build_resnet18(
        num_classes=NUMBER_OF_PET_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    validation_dataset = OxfordPetDataset(
        dataset_root=args.data_root,
        split_path=args.split_file,
        split_name="val",
        transform=build_eval_transform(),
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = torch.nn.CrossEntropyLoss()
    result = evaluate_model(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
        show_progress=args.show_progress,
    )

    checkpoint_best_accuracy = float(
        checkpoint.get("best_validation_accuracy", -1.0)
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_validation_top1_accuracy": checkpoint_best_accuracy,
        "validation_loss": result.loss,
        "top1_accuracy": result.metrics["top1_accuracy"],
        "top5_accuracy": result.metrics["top5_accuracy"],
        "macro_f1": result.metrics["macro_f1"],
        "elapsed_seconds": result.elapsed_seconds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Checkpoint epoch: {summary['checkpoint_epoch']}")
    print(f"Validation loss: {summary['validation_loss']:.4f}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.4f}")
    print(f"Top-5 accuracy: {summary['top5_accuracy']:.4f}")
    print(f"Macro-F1: {summary['macro_f1']:.4f}")
    print(f"结果文件：{args.output}")


if __name__ == "__main__":
    main()
