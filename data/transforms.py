"""图片进入模型前的 Transform。

Transform 不会修改磁盘上的原图。它每次在内存中生成一份模型输入。
"""

from torchvision import transforms

IMAGE_SIZE = 224
RESIZE_SIZE = 256

# ResNet-18 使用 ImageNet 预训练权重，因此归一化参数也沿用 ImageNet 统计值。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    """训练集 Transform：加入随机性，帮助模型学习更稳健的特征。"""

    return transforms.Compose(
        [
            # 先缩放到 256，再随机裁出 224×224。
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            # 猫狗图片左右翻转后仍属于同一类别，因此可以作为训练增强。
            transforms.RandomHorizontalFlip(),
            # PIL 图片转成 [3, 224, 224] 的 Tensor，并把像素缩放到 0 到 1。
            transforms.ToTensor(),
            # 按通道进行标准化，输入数值更稳定。
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """验证集和测试集 Transform：必须是确定性的。"""

    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            # 中心裁剪不使用随机数，保证同一张图片每次得到相同结果。
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
