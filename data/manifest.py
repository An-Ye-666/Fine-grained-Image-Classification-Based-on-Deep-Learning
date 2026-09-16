"""官方数据清单解析、下载检查和固定分层划分。

这个文件只回答两个问题：
1. 哪些图片属于官方有效的 7,349 张样本？
2. 哪些图片应该进入训练集、验证集和测试集？

它不负责读取图片，也不负责把图片变成 Tensor。那些工作由 dataset.py 完成。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "oxford-iiit-pet"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "oxford_pet_seed42.json"

EXPECTED_SAMPLE_COUNT = 7349
EXPECTED_CLASS_IDS = tuple(range(1, 38))
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class PetSample:
    """一张图片的官方标签信息。

    属性含义：
    - image_id：文件名去掉 .jpg 后的部分，例如 Abyssinian_100。
    - class_id：官方类别编号，范围是 1 到 37。
    - class_name：类别名称，例如 Abyssinian。
    - source_split：这张图片来自官方 trainval.txt 还是 test.txt。
    """

    image_id: str
    class_id: int
    class_name: str
    source_split: str


@dataclass(frozen=True)
class SplitManifest:
    """已经保存到 JSON 文件中的固定数据划分。"""

    seed: int
    ratios: dict[str, float]
    counts: dict[str, int]
    class_names: dict[int, str]
    splits: dict[str, tuple[PetSample, ...]]
    excluded_unlisted_image_ids: tuple[str, ...]


def _parse_manifest_file(path: Path, source_split: str) -> list[PetSample]:
    """读取官方清单文件。

    官方清单每一行有四个字段：
    image-id class-id species-id breed-id
    """

    samples: list[PetSample] = []
    seen_image_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_number}: 需要 4 个字段，实际为 {len(fields)}"
                )

            image_id = fields[0]
            try:
                class_id = int(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: 非法类别编号 {fields[1]!r}"
                ) from error

            if image_id in seen_image_ids:
                raise ValueError(
                    f"{path}:{line_number}: 重复图片编号 {image_id!r}"
                )

            # 例如 Abyssinian_100 -> Abyssinian
            class_name_match = re.fullmatch(r"(.+)_\d+", image_id)
            if class_name_match is None:
                raise ValueError(
                    f"{path}:{line_number}: 无法从 {image_id!r} 推导类别名称"
                )

            seen_image_ids.add(image_id)
            samples.append(
                PetSample(
                    image_id=image_id,
                    class_id=class_id,
                    class_name=class_name_match.group(1),
                    source_split=source_split,
                )
            )

    return samples


def ensure_dataset_available(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    download: bool = False,
) -> None:
    """确认官方数据文件存在；必要时使用 torchvision 下载。

    torchvision 会把数据解压到 dataset_root 中。参数 download=False 时不会联网，
    所以已经手动下载好数据后，可以直接使用本地文件。
    """

    required_paths = (
        dataset_root / "annotations" / "trainval.txt",
        dataset_root / "annotations" / "test.txt",
        dataset_root / "images",
    )
    if all(path.exists() for path in required_paths):
        return

    if not download:
        missing = [str(path) for path in required_paths if not path.exists()]
        raise FileNotFoundError(
            "缺少本地数据文件，且没有启用自动下载：\n"
            + "\n".join(missing)
        )

    # 这里使用 torchvision 官方下载入口；导入放在函数内部，避免平时执行划分时
    # 无谓加载较大的 torchvision 库。
    from torchvision.datasets import OxfordIIITPet

    torchvision_root = dataset_root.parent
    OxfordIIITPet(
        root=str(torchvision_root),
        split="trainval",
        download=True,
    )
    OxfordIIITPet(
        root=str(torchvision_root),
        split="test",
        download=True,
    )


def load_official_samples(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    download: bool = False,
) -> tuple[list[PetSample], dict[int, str], list[str]]:
    """读取并检查官方 trainval 与 test 清单。

    返回值依次是：
    - 官方 7,349 个样本；
    - 类别编号到类别名称的映射；
    - 图片目录中存在、但不在官方清单中的额外图片编号。
    """

    ensure_dataset_available(dataset_root, download=download)

    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"
    trainval_path = annotations_dir / "trainval.txt"
    test_path = annotations_dir / "test.txt"

    samples = [
        *_parse_manifest_file(trainval_path, "official_trainval"),
        *_parse_manifest_file(test_path, "official_test"),
    ]

    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"官方样本应为 {EXPECTED_SAMPLE_COUNT} 张，实际为 {len(samples)} 张"
        )

    image_ids = [sample.image_id for sample in samples]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("官方 trainval 与 test 清单之间存在重复图片")

    # 同一个类别编号只能对应一个类别名称，反之亦然。
    class_names_by_id: dict[int, set[str]] = defaultdict(set)
    class_ids_by_name: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        class_names_by_id[sample.class_id].add(sample.class_name)
        class_ids_by_name[sample.class_name].add(sample.class_id)

    if tuple(sorted(class_names_by_id)) != EXPECTED_CLASS_IDS:
        raise ValueError(
            "官方类别编号应为 1 到 37，实际为 "
            f"{sorted(class_names_by_id)}"
        )

    ambiguous_classes = {
        class_id: sorted(names)
        for class_id, names in class_names_by_id.items()
        if len(names) != 1
    }
    if ambiguous_classes:
        raise ValueError(f"同一个类别编号对应多个名称：{ambiguous_classes}")

    reused_names = {
        class_name: sorted(class_ids)
        for class_name, class_ids in class_ids_by_name.items()
        if len(class_ids) != 1
    }
    if reused_names:
        raise ValueError(f"同一个类别名称对应多个编号：{reused_names}")

    missing_images = [
        sample.image_id
        for sample in samples
        if not (images_dir / f"{sample.image_id}.jpg").is_file()
    ]
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"官方清单中有 {len(missing_images)} 张图片缺失，例如：{preview}"
        )

    class_names = {
        class_id: next(iter(class_names_by_id[class_id]))
        for class_id in EXPECTED_CLASS_IDS
    }

    # 官方 trainval + test 合计 7,349 张；images 目录中还可能有额外图片。
    # 这些额外文件没有进入官方清单，因此不参与训练、验证或测试。
    official_image_ids = set(image_ids)
    all_image_ids = {path.stem for path in images_dir.glob("*.jpg")}
    unlisted_image_ids = sorted(all_image_ids - official_image_ids)

    return samples, class_names, unlisted_image_ids


def _allocate_counts_by_class(
    class_sizes: dict[int, int],
    ratio: float,
    target_total: int,
) -> dict[int, int]:
    """按类别分配数量，并使用稳定的“最大余数法”补齐总数。

    例如某类按 70% 计算得到 139.5 张，基础分配先取 139，
    剩余名额再根据小数部分大小分配给各个类别。
    """

    ideal_counts = {
        class_id: size * ratio for class_id, size in class_sizes.items()
    }
    allocated_counts = {
        class_id: int(ideal_count)
        for class_id, ideal_count in ideal_counts.items()
    }
    remaining = target_total - sum(allocated_counts.values())

    if remaining < 0:
        raise ValueError("目标数量小于基础分配数量")

    ranked_class_ids = sorted(
        class_sizes,
        key=lambda class_id: (
            -(ideal_counts[class_id] - allocated_counts[class_id]),
            class_id,
        ),
    )

    for class_id in ranked_class_ids[:remaining]:
        allocated_counts[class_id] += 1

    if remaining > len(ranked_class_ids):
        raise ValueError("需要补齐的数量超过了类别数量")

    for class_id, count in allocated_counts.items():
        if not 1 <= count <= class_sizes[class_id]:
            raise ValueError(
                f"类别 {class_id} 的分配数量非法：{count} / {class_sizes[class_id]}"
            )

    if sum(allocated_counts.values()) != target_total:
        raise ValueError("按类别分配后的总数与目标总数不一致")

    return allocated_counts


def create_stratified_split(
    samples: Iterable[PetSample],
    class_names: dict[int, str],
    seed: int = 42,
) -> dict[str, object]:
    """创建固定的 70/15/15 分层划分。

    “分层”表示每个类别都按接近 70/15/15 的比例划分，而不是只保证
    全局总数符合比例。这样可以避免某些类别集中在某一个数据集中。
    """

    sample_list = list(samples)
    samples_by_class: dict[int, list[PetSample]] = defaultdict(list)
    for sample in sample_list:
        samples_by_class[sample.class_id].append(sample)

    class_sizes = {
        class_id: len(samples_by_class[class_id])
        for class_id in EXPECTED_CLASS_IDS
    }
    total = sum(class_sizes.values())
    target_counts = {
        "train": round(total * SPLIT_RATIOS["train"]),
        "val": round(total * SPLIT_RATIOS["val"]),
    }
    target_counts["test"] = total - target_counts["train"] - target_counts["val"]

    train_counts = _allocate_counts_by_class(
        class_sizes,
        SPLIT_RATIOS["train"],
        target_counts["train"],
    )
    val_counts = _allocate_counts_by_class(
        class_sizes,
        SPLIT_RATIOS["val"],
        target_counts["val"],
    )
    test_counts = {
        class_id: class_sizes[class_id]
        - train_counts[class_id]
        - val_counts[class_id]
        for class_id in EXPECTED_CLASS_IDS
    }

    if any(count <= 0 for count in test_counts.values()):
        raise ValueError("至少有一个类别没有测试样本")

    split_samples: dict[str, list[PetSample]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }

    # 为每个类别分别打乱。排序后再使用固定随机种子，结果不会被文件系统顺序影响。
    random_generator = random.Random(seed)
    for class_id in EXPECTED_CLASS_IDS:
        class_samples = sorted(
            samples_by_class[class_id],
            key=lambda sample: sample.image_id,
        )
        random_generator.shuffle(class_samples)

        train_end = train_counts[class_id]
        val_end = train_end + val_counts[class_id]
        split_samples["train"].extend(class_samples[:train_end])
        split_samples["val"].extend(class_samples[train_end:val_end])
        split_samples["test"].extend(class_samples[val_end:])

    for split_name in SPLIT_NAMES:
        split_samples[split_name].sort(key=lambda sample: sample.image_id)

    all_split_ids = [
        sample.image_id
        for split_name in SPLIT_NAMES
        for sample in split_samples[split_name]
    ]
    if len(all_split_ids) != total or len(set(all_split_ids)) != total:
        raise ValueError("划分结果不是完整且互不重叠的数据集")

    return {
        "version": 1,
        "dataset": "Oxford-IIIT Pet",
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "counts": {
            split_name: len(split_samples[split_name])
            for split_name in SPLIT_NAMES
        },
        "class_names": {
            str(class_id): class_names[class_id]
            for class_id in EXPECTED_CLASS_IDS
        },
        "splits": {
            split_name: [
                asdict(sample) for sample in split_samples[split_name]
            ]
            for split_name in SPLIT_NAMES
        },
    }


def write_split_file(split_data: dict[str, object], output_path: Path) -> None:
    """将固定划分写入 JSON 文件。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        split_data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output_path.write_text(f"{serialized}\n", encoding="utf-8", newline="\n")


def build_and_write_split(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path = DEFAULT_SPLIT_PATH,
    seed: int = 42,
    download: bool = False,
) -> dict[str, object]:
    """读取官方清单、创建划分并保存固定 JSON 文件。"""

    samples, class_names, unlisted_image_ids = load_official_samples(
        dataset_root,
        download=download,
    )
    split_data = create_stratified_split(samples, class_names, seed=seed)
    split_data["excluded_unlisted_image_count"] = len(unlisted_image_ids)
    split_data["excluded_unlisted_image_ids"] = unlisted_image_ids
    write_split_file(split_data, output_path)
    return split_data


def load_split_manifest(split_path: Path = DEFAULT_SPLIT_PATH) -> SplitManifest:
    """读取已经生成的 JSON 划分文件，并转换成容易使用的 Python 对象。"""

    if not split_path.is_file():
        raise FileNotFoundError(
            f"找不到划分文件：{split_path}\n"
            "请先运行：python -m data.manifest"
        )

    raw_data = json.loads(split_path.read_text(encoding="utf-8"))
    splits: dict[str, tuple[PetSample, ...]] = {}
    for split_name in SPLIT_NAMES:
        split_records = raw_data["splits"][split_name]
        splits[split_name] = tuple(
            PetSample(
                image_id=record["image_id"],
                class_id=int(record["class_id"]),
                class_name=record["class_name"],
                source_split=record["source_split"],
            )
            for record in split_records
        )

    class_names = {
        int(class_id): class_name
        for class_id, class_name in raw_data["class_names"].items()
    }
    manifest = SplitManifest(
        seed=int(raw_data["seed"]),
        ratios={
            split_name: float(ratio)
            for split_name, ratio in raw_data["ratios"].items()
        },
        counts={
            split_name: int(count)
            for split_name, count in raw_data["counts"].items()
        },
        class_names=class_names,
        splits=splits,
        excluded_unlisted_image_ids=tuple(
            raw_data.get("excluded_unlisted_image_ids", [])
        ),
    )
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: SplitManifest) -> None:
    """检查 JSON 划分文件是否完整、无重叠且覆盖全部 37 类。"""

    split_ids = {
        split_name: {sample.image_id for sample in manifest.splits[split_name]}
        for split_name in SPLIT_NAMES
    }

    for split_name, records in manifest.splits.items():
        if len(records) != manifest.counts[split_name]:
            raise ValueError(
                f"{split_name} 的实际数量和 JSON 记录数量不一致"
            )

    for first_name, second_name in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = split_ids[first_name] & split_ids[second_name]
        if overlap:
            raise ValueError(
                f"{first_name} 与 {second_name} 存在 {len(overlap)} 个重复样本"
            )

    all_image_ids = set().union(*split_ids.values())
    if len(all_image_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"划分总样本应为 {EXPECTED_SAMPLE_COUNT}，实际为 {len(all_image_ids)}"
        )

    for split_name, records in manifest.splits.items():
        class_ids = {sample.class_id for sample in records}
        if class_ids != set(EXPECTED_CLASS_IDS):
            missing = sorted(set(EXPECTED_CLASS_IDS) - class_ids)
            raise ValueError(f"{split_name} 缺少类别：{missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成并检查 Oxford-IIIT Pet 的固定分层划分。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="包含 annotations 和 images 的目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="划分 JSON 的输出路径。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="本地数据缺失时，允许 torchvision 自动下载。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_data = build_and_write_split(
        dataset_root=args.dataset_root,
        output_path=args.output,
        seed=args.seed,
        download=args.download,
    )
    counts = split_data["counts"]
    print("官方有效样本：", sum(counts.values()))
    print("训练集：", counts["train"])
    print("验证集：", counts["val"])
    print("测试集：", counts["test"])
    print("排除的额外图片：", split_data["excluded_unlisted_image_count"])
    print("划分文件：", args.output)


if __name__ == "__main__":
    main()
