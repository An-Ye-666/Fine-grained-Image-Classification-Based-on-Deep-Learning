"""根据固定数据划分创建 PyTorch DataLoader。"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import OxfordPetDataset
from data.manifest import DEFAULT_DATASET_ROOT, DEFAULT_SPLIT_PATH
from data.transforms import build_eval_transform, build_train_transform


def create_dataloader(
    dataset: OxfordPetDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """把 Dataset 包装成按 batch 输出数据的 DataLoader。

    输入：一次只能返回一张图片的 Dataset。
    输出：一次返回一批图片，例如 [32, 3, 224, 224]。
    """

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if num_workers < 0:
        raise ValueError("num_workers 不能小于 0")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_oxford_pet_dataloaders(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_path: Path = DEFAULT_SPLIT_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """一次创建 train、val、test 三个 DataLoader。"""

    train_dataset = OxfordPetDataset(
        dataset_root=dataset_root,
        split_path=split_path,
        split_name="train",
        transform=build_train_transform(),
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

    return {
        # 训练集打乱顺序可以减少连续样本高度相似带来的训练波动。
        "train": create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        # 验证和测试必须保持固定顺序，方便逐样本比较和复现实验。
        "val": create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
