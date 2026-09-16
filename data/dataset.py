"""把 Oxford-IIIT Pet 图片转换成 PyTorch 可以读取的 Dataset。

Dataset 的职责非常单一：
- 根据划分文件找到一张图片；
- 打开图片；
- 应用 Transform；
- 返回图片 Tensor 和从 0 开始的标签。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from data.manifest import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_PATH,
    PetSample,
    load_split_manifest,
)

# 输入是 PIL 图片，输出是模型可以读取的 Tensor。
ImageTransform = Callable[[Image.Image], Tensor]


class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet 的 PyTorch Dataset。

    参数：
    - dataset_root：包含 images 和 annotations 的目录；
    - split_path：固定的 JSON 划分文件；
    - split_name：train、val 或 test；
    - transform：图片来源模型前执行的处理流程。
    """

    def __init__(
        self,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
        split_path: Path = DEFAULT_SPLIT_PATH,
        split_name: str = "train",
        transform: ImageTransform | None = None,
    ) -> None:
        if split_name not in {"train", "val", "test"}:
            raise ValueError(
                f"split_name 必须是 train、val 或 test，实际为 {split_name!r}"
            )

        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.split_name = split_name
        self.transform = transform

        self.manifest = load_split_manifest(Path(split_path))
        self.records: tuple[PetSample, ...] = self.manifest.splits[split_name]

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"找不到图片目录：{self.images_dir}")

    def __len__(self) -> int:
        """告诉 PyTorch 这个数据集一共有多少张图片。"""

        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """返回第 index 张图片和对应标签。

        返回值：
        - image_tensor：形状为 [3, 224, 224] 的图片 Tensor；
        - label：范围 0 到 36 的整数，供 CrossEntropyLoss 使用。

        注意：官方类别编号是 1 到 37，因此这里减 1 转为 PyTorch 常用的
        0 到 36 标签。标签名称仍保留在 record 中，方便查看和解释。
        """

        record = self.records[index]
        image_path = self.images_dir / f"{record.image_id}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"找不到图片：{image_path}")

        # Image.open 采用“用时才读取”的方式。with 语句可以确保文件及时关闭。
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                # Dataset 本身不默认做数值转换，防止调用者误以为它已经完成预处理。
                raise ValueError(
                    "OxfordPetDataset 需要传入 transform，例如 "
                    "build_train_transform() 或 build_eval_transform()。"
                )

        label = record.class_id - 1
        return image_tensor, label

    def get_record(self, index: int) -> PetSample:
        """返回某张图片的原始元数据，方便做错误分析。"""

        return self.records[index]

    def get_class_name(self, label: int) -> str:
        """把 0 到 36 的模型标签转换回可读的类别名称。"""

        if not 0 <= label < 37:
            raise ValueError(f"标签必须在 0 到 36 之间，实际为 {label}")
        return self.manifest.class_names[label + 1]
