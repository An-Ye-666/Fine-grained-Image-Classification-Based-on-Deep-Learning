"""A1 数据阶段的可执行检查脚本。

运行方式：
    python -m scripts.check_a1_data

脚本会真实读取图片，而不是只检查 JSON 文件，因此能够发现路径错误、
Transform 错误、标签越界和 batch 形状错误。
"""

from __future__ import annotations

from pathlib import Path

import torch

from data.dataloaders import create_oxford_pet_dataloaders
from data.dataset import OxfordPetDataset
from data.manifest import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_PATH,
    EXPECTED_SAMPLE_COUNT,
    SPLIT_NAMES,
    load_split_manifest,
)
from data.transforms import build_eval_transform
from utils.metrics import compute_classification_metrics


def require(condition: bool, message: str) -> None:
    """如果条件不成立，就立即停止并报告清晰错误。"""

    if not condition:
        raise AssertionError(message)


def main() -> None:
    split_path = Path(DEFAULT_SPLIT_PATH)
    dataset_root = Path(DEFAULT_DATASET_ROOT)
    manifest = load_split_manifest(split_path)

    counts = manifest.counts
    require(
        sum(counts.values()) == EXPECTED_SAMPLE_COUNT,
        f"总样本数错误：{sum(counts.values())}",
    )
    require(
        counts["train"] == 5144,
        f"训练集数量错误：{counts['train']}，预期 5144",
    )
    require(
        counts["val"] == 1102,
        f"验证集数量错误：{counts['val']}，预期 1102",
    )
    require(
        counts["test"] == 1103,
        f"测试集数量错误：{counts['test']}，预期 1103",
    )
    print("[PASS] 划分数量：5144 / 1102 / 1103")

    split_ids = {
        split_name: {
            sample.image_id for sample in manifest.splits[split_name]
        }
        for split_name in SPLIT_NAMES
    }
    for first_name, second_name in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        require(
            not (split_ids[first_name] & split_ids[second_name]),
            f"{first_name} 与 {second_name} 存在重复样本",
        )
    print("[PASS] 训练、验证、测试之间没有重复图片")

    for split_name in SPLIT_NAMES:
        class_ids = {
            sample.class_id for sample in manifest.splits[split_name]
        }
        require(
            class_ids == set(range(1, 38)),
            f"{split_name} 没有覆盖全部 37 个类别",
        )
    print("[PASS] 每份数据都覆盖 37 个类别")

    train_dataset = OxfordPetDataset(
        dataset_root=dataset_root,
        split_path=split_path,
        split_name="train",
        transform=build_eval_transform(),
    )
    val_dataset = OxfordPetDataset(
        dataset_root=dataset_root,
        split_path=split_path,
        split_name="val",
        transform=build_eval_transform(),
    )
    test_dataset = OxfordPetDataset(
        dataset_root=dataset_root,
        split_path=split_path,
        split_name="test",
        transform=build_eval_transform(),
    )

    sample_image, sample_label = train_dataset[0]
    require(
        tuple(sample_image.shape) == (3, 224, 224),
        f"单张图片形状错误：{tuple(sample_image.shape)}",
    )
    require(
        0 <= sample_label <= 36,
        f"标签超出范围：{sample_label}",
    )
    print("[PASS] 单张图片形状为 [3, 224, 224]")

    # 验证集 Transform 必须没有随机性：同一张图片连续读取两次应完全相同。
    first_val_image, _ = val_dataset[0]
    second_val_image, _ = val_dataset[0]
    require(
        torch.equal(first_val_image, second_val_image),
        "验证集 Transform 不是确定性的，同一张图片得到了不同结果",
    )
    print("[PASS] 验证集 Transform 可重复")

    dataloaders = create_oxford_pet_dataloaders(
        dataset_root=dataset_root,
        split_path=split_path,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )
    train_images, train_labels = next(iter(dataloaders["train"]))
    require(
        tuple(train_images.shape) == (32, 3, 224, 224),
        f"训练 batch 形状错误：{tuple(train_images.shape)}",
    )
    require(
        tuple(train_labels.shape) == (32,),
        f"标签 batch 形状错误：{tuple(train_labels.shape)}",
    )
    require(
        train_images.dtype == torch.float32,
        f"图片 Tensor 类型错误：{train_images.dtype}",
    )
    require(
        int(train_labels.min()) >= 0 and int(train_labels.max()) <= 36,
        "batch 中存在超出 0 到 36 的标签",
    )
    print("[PASS] DataLoader batch 形状为 [32, 3, 224, 224]")
    print("[PASS] 标签 batch 形状为 [32]，范围在 0 到 36")

    # 使用一组人工构造的预测检查指标函数本身是否按预期工作。
    metric_logits = torch.zeros(4, 37)
    metric_logits[0, 5] = 10.0
    metric_logits[1, 6] = 10.0
    metric_logits[2, 7] = 10.0
    metric_logits[3, 8] = 10.0
    metric_labels = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    metric_values = compute_classification_metrics(metric_logits, metric_labels)
    require(
        metric_values["top1_accuracy"] == 1.0,
        "Top-1 指标检查失败",
    )
    require(
        metric_values["top5_accuracy"] == 1.0,
        "Top-5 指标检查失败",
    )
    require(
        0.0 <= metric_values["macro_f1"] <= 1.0,
        "Macro-F1 超出 0 到 1 的范围",
    )
    print("[PASS] Top-1、Top-5 和 Macro-F1 指标可用")

    print("A1 数据检查全部通过。")


if __name__ == "__main__":
    main()
