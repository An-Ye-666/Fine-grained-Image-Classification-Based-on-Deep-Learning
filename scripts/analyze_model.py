"""对最终模型执行一次测试集评估和可解释性分析。

运行后生成：
- 测试集指标；
- 37 类混淆矩阵；
- 每类 Precision、Recall、F1；
- 最容易混淆的品种对；
- 一张正确案例和一张错误案例的 Grad-CAM 热力图。

测试集默认锁定。只有显式传入 `--allow-test` 且尚未生成 `test_metrics.json`
时才允许运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from data.dataset import OxfordPetDataset
from data.manifest import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_PATH
from data.transforms import build_eval_transform
from models.model import NUMBER_OF_PET_CLASSES, build_resnet18
from utils.gradcam import GradCAM
from utils.metrics import compute_classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "label_smoothing_0.1"
    / "best_model.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "final_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行最终测试集评估、混淆矩阵和 Grad-CAM 分析"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        help="关闭测试集进度条。",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="明确确认本次只评估测试集一次。",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前环境没有可用的 CUDA GPU")
    return torch.device(requested_device)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_resnet18(
        num_classes=NUMBER_OF_PET_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "model": model.to(device),
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_accuracy": float(
            checkpoint.get("best_validation_accuracy", -1.0)
        ),
    }


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """收集测试集所有 logits、标签和标准交叉熵 Loss。"""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    progress_bar = (
        tqdm(
            dataloader,
            desc="Test",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if show_progress
        else dataloader
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
        raise ValueError("测试集没有产生任何样本")

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        total_loss / total_samples,
    )


def save_confusion_outputs(
    matrix: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """保存混淆矩阵图片和 CSV。"""

    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Oxford-IIIT Pet Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=90, fontsize=6)
    axis.set_yticklabels(class_names, fontsize=6)
    figure.tight_layout()
    figure.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    with (output_dir / "confusion_matrix.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_class", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def find_top_confusions(
    matrix: np.ndarray,
    class_names: list[str],
    limit: int = 10,
) -> list[dict[str, object]]:
    """找出最容易被混淆的品种对。"""

    pairs: list[dict[str, object]] = []
    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count == 0:
                continue
            support = int(matrix[true_index].sum())
            pairs.append(
                {
                    "true_class": class_names[true_index],
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                    "true_class_support": support,
                    "ratio_of_true_class": count / support if support else 0.0,
                }
            )

    pairs.sort(key=lambda item: (-int(item["count"]), str(item["true_class"])))
    return pairs[:limit]


def save_predictions_csv(
    image_ids: list[str],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_id",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "confidence",
                "correct",
            ]
        )
        for image_id, true_label, predicted_label, confidence in zip(
            image_ids,
            true_labels,
            predicted_labels,
            confidences,
        ):
            writer.writerow(
                [
                    image_id,
                    int(true_label),
                    class_names[int(true_label)],
                    int(predicted_label),
                    class_names[int(predicted_label)],
                    f"{float(confidence):.6f}",
                    int(true_label == predicted_label),
                ]
            )


def load_display_image(dataset: OxfordPetDataset, index: int) -> Image.Image:
    """读取与验证/测试 Transform 对应的中心裁剪图片，供热力图叠加。"""

    record = dataset.get_record(index)
    image_path = dataset.images_dir / f"{record.image_id}.jpg"
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        resized = TF.resize(rgb_image, 256)
        return TF.center_crop(resized, 224)


def save_gradcam_case(
    dataset: OxfordPetDataset,
    model: nn.Module,
    index: int,
    target_class: int,
    probabilities: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    """保存原图、热力图和叠加图。"""

    image_tensor, true_label = dataset[index]
    image_tensor = image_tensor.to(next(model.parameters()).device)

    target_layer = model.layer4[-1].conv2
    with GradCAM(model, target_layer) as gradcam:
        cam, _ = gradcam(image_tensor, target_class=target_class)

    raw_image = load_display_image(dataset, index)
    raw_array = np.asarray(raw_image).astype(np.float32) / 255.0
    heatmap = plt.get_cmap("jet")(cam.numpy())[..., :3]
    overlay = np.clip(0.55 * raw_array + 0.45 * heatmap, 0.0, 1.0)
    confidence = float(probabilities[target_class].item())

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(raw_image)
    axes[0].set_title(f"Original\nTrue label: {true_label}")
    axes[1].imshow(cam.numpy(), cmap="jet")
    axes[1].set_title(f"Grad-CAM\nTarget class: {target_class}")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nConfidence: {confidence:.4f}")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_analysis_summary(
    output_path: Path,
    metrics: dict[str, float],
    top_confusions: list[dict[str, object]],
    correct_case: dict[str, object],
    error_case: dict[str, object],
) -> None:
    lines = [
        "# 最终模型测试分析",
        "",
        "## 测试指标",
        "",
        f"- Loss：{metrics['loss']:.4f}",
        f"- Top-1：{metrics['top1_accuracy']:.4f}",
        f"- Top-5：{metrics['top5_accuracy']:.4f}",
        f"- Macro-F1：{metrics['macro_f1']:.4f}",
        "",
        "## 主要混淆品种对",
        "",
        "| 真实类别 | 预测类别 | 数量 | 占真实类别比例 |",
        "| --- | --- | ---: | ---: |",
    ]
    for pair in top_confusions:
        lines.append(
            f"| {pair['true_class']} | {pair['predicted_class']} | "
            f"{pair['count']} | {pair['ratio_of_true_class']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 案例",
            "",
            f"- 正确案例：`{correct_case['image_id']}`，"
            f"真实类别 `{correct_case['true_class']}`，"
            f"预测类别 `{correct_case['predicted_class']}`",
            f"- 错误案例：`{error_case['image_id']}`，"
            f"真实类别 `{error_case['true_class']}`，"
            f"预测类别 `{error_case['predicted_class']}`",
            "",
            "## 图表",
            "",
            "- `confusion_matrix.png`",
            "- `gradcam_correct.png`",
            "- `gradcam_error.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.allow_test:
        raise RuntimeError(
            "测试集保持锁定。确认只评估一次后，请添加 --allow-test。"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到模型文件：{args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_metrics_path = args.output_dir / "test_metrics.json"
    if test_metrics_path.exists():
        raise RuntimeError(
            f"测试集已经评估过：{test_metrics_path}\n"
            "为避免测试集泄漏，不再重复评估。"
        )

    device = resolve_device(args.device)
    print(f"评估设备：{device}")
    checkpoint_data = load_checkpoint(args.checkpoint, device)
    model = checkpoint_data["model"]
    assert isinstance(model, nn.Module)

    test_dataset = OxfordPetDataset(
        dataset_root=args.data_root,
        split_path=args.split_file,
        split_name="test",
        transform=build_eval_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    start_time = time.perf_counter()
    logits, labels, loss = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        show_progress=args.show_progress,
    )
    metric_values = compute_classification_metrics(logits, labels)
    metrics = {
        "loss": loss,
        "top1_accuracy": metric_values["top1_accuracy"],
        "top5_accuracy": metric_values["top5_accuracy"],
        "macro_f1": metric_values["macro_f1"],
    }

    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    true_labels = labels.numpy()
    predicted_labels = predictions.numpy()
    confidence_values = probabilities.max(dim=1).values.numpy()
    correct_mask = true_labels == predicted_labels

    class_names = [
        test_dataset.manifest.class_names[class_id]
        for class_id in range(1, NUMBER_OF_PET_CLASSES + 1)
    ]

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUMBER_OF_PET_CLASSES)),
    )
    save_confusion_outputs(matrix, class_names, args.output_dir)
    top_confusions = find_top_confusions(matrix, class_names)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUMBER_OF_PET_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with (args.output_dir / "per_class_metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in class_names:
            row = report[class_name]
            writer.writerow(
                [
                    class_name,
                    row["precision"],
                    row["recall"],
                    row["f1-score"],
                    row["support"],
                ]
            )

    image_ids = [
        test_dataset.get_record(index).image_id
        for index in range(len(test_dataset))
    ]
    save_predictions_csv(
        image_ids=image_ids,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidence_values,
        class_names=class_names,
        output_path=args.output_dir / "test_predictions.csv",
    )

    correct_indices = np.flatnonzero(correct_mask)
    error_indices = np.flatnonzero(~correct_mask)
    if correct_indices.size == 0:
        raise RuntimeError("测试集中没有正确样本，无法生成正向案例")
    if error_indices.size == 0:
        raise RuntimeError("测试集中没有错误样本，无法生成错误案例")

    correct_index = int(
        correct_indices[np.argmax(confidence_values[correct_indices])]
    )
    error_index = int(error_indices[np.argmax(confidence_values[error_indices])])

    correct_case = {
        "index": correct_index,
        "image_id": image_ids[correct_index],
        "true_label": int(true_labels[correct_index]),
        "true_class": class_names[int(true_labels[correct_index])],
        "predicted_label": int(predicted_labels[correct_index]),
        "predicted_class": class_names[int(predicted_labels[correct_index])],
        "confidence": float(confidence_values[correct_index]),
    }
    error_case = {
        "index": error_index,
        "image_id": image_ids[error_index],
        "true_label": int(true_labels[error_index]),
        "true_class": class_names[int(true_labels[error_index])],
        "predicted_label": int(predicted_labels[error_index]),
        "predicted_class": class_names[int(predicted_labels[error_index])],
        "confidence": float(confidence_values[error_index]),
    }

    save_gradcam_case(
        dataset=test_dataset,
        model=model,
        index=correct_index,
        target_class=int(true_labels[correct_index]),
        probabilities=probabilities[correct_index],
        output_path=args.output_dir / "gradcam_correct.png",
        title="Correct prediction",
    )
    save_gradcam_case(
        dataset=test_dataset,
        model=model,
        index=error_index,
        target_class=int(predicted_labels[error_index]),
        probabilities=probabilities[error_index],
        output_path=args.output_dir / "gradcam_error.png",
        title="Incorrect prediction",
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_data["epoch"],
        "split": "test",
        "sample_count": len(test_dataset),
        "evaluation_loss": "standard CrossEntropyLoss",
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start_time,
        "top_confusions": top_confusions,
        "correct_case": correct_case,
        "error_case": error_case,
    }
    test_metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_analysis_summary(
        output_path=args.output_dir / "analysis_summary.md",
        metrics=metrics,
        top_confusions=top_confusions,
        correct_case=correct_case,
        error_case=error_case,
    )

    print(f"测试样本数：{len(test_dataset)}")
    print(f"测试 Loss：{metrics['loss']:.4f}")
    print(f"测试 Top-1：{metrics['top1_accuracy']:.4f}")
    print(f"测试 Top-5：{metrics['top5_accuracy']:.4f}")
    print(f"测试 Macro-F1：{metrics['macro_f1']:.4f}")
    print("主要混淆品种对：")
    for pair in top_confusions[:5]:
        print(
            f"  {pair['true_class']} -> {pair['predicted_class']}: "
            f"{pair['count']} 张"
        )
    print(f"分析结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
