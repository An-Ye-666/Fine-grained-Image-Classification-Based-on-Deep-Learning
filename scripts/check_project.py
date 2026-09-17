"""检查最终工程文件和关键交付物是否完整。

运行方式：
    python -m scripts.check_project
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少{description}：{path}")
    print(f"[PASS] {description}：{path.relative_to(PROJECT_ROOT)}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"缺少{description}：{path}")
    print(f"[PASS] {description}：{path.relative_to(PROJECT_ROOT)}")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    required_files = [
        (PROJECT_ROOT / "README.md", "项目说明"),
        (PROJECT_ROOT / "requirements.txt", "依赖清单"),
        (PROJECT_ROOT / "train.py", "Baseline 训练入口"),
        (PROJECT_ROOT / "train_label_smoothing.py", "消融训练入口"),
        (PROJECT_ROOT / "evaluate.py", "验证集评估入口"),
        (PROJECT_ROOT / "models" / "model.py", "模型定义"),
        (PROJECT_ROOT / "data" / "manifest.py", "数据划分代码"),
        (
            PROJECT_ROOT
            / "artifacts"
            / "splits"
            / "oxford_pet_seed42.json",
            "固定数据划分",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "baseline"
            / "metrics.json",
            "Baseline 指标",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "label_smoothing_0.1"
            / "metrics.json",
            "Label Smoothing 指标",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "final_model"
            / "test_metrics.json",
            "最终测试指标",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "final_model"
            / "confusion_matrix.png",
            "混淆矩阵",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "final_model"
            / "gradcam_correct.png",
            "正确案例 Grad-CAM",
        ),
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "final_model"
            / "gradcam_error.png",
            "错误案例 Grad-CAM",
        ),
    ]
    for path, description in required_files:
        require_file(path, description)

    require_directory(
        PROJECT_ROOT / "data" / "oxford-iiit-pet" / "images",
        "原始图片目录",
    )

    baseline_logs = list(
        (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "baseline"
            / "tensorboard"
        ).glob("events.out.tfevents.*")
    )
    smoothing_logs = list(
        (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "label_smoothing_0.1"
            / "tensorboard"
        ).glob("events.out.tfevents.*")
    )
    if not baseline_logs:
        raise FileNotFoundError("缺少 Baseline TensorBoard 日志")
    if not smoothing_logs:
        raise FileNotFoundError("缺少 Label Smoothing TensorBoard 日志")
    print(f"[PASS] Baseline TensorBoard 日志：{len(baseline_logs)} 个")
    print(f"[PASS] Label Smoothing TensorBoard 日志：{len(smoothing_logs)} 个")

    split_data = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "oxford_pet_seed42.json"
    )
    if int(split_data["counts"]["train"]) != 5144:
        raise ValueError("训练集数量不是 5144")
    if int(split_data["counts"]["val"]) != 1102:
        raise ValueError("验证集数量不是 1102")
    if int(split_data["counts"]["test"]) != 1103:
        raise ValueError("测试集数量不是 1103")
    print("[PASS] 数据划分为 5144 / 1102 / 1103")

    test_data = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "analysis"
        / "final_model"
        / "test_metrics.json"
    )
    test_metrics = test_data["metrics"]
    if float(test_metrics["top1_accuracy"]) < 0.90:
        raise ValueError("测试集 Top-1 低于 90%")
    print(
        "[PASS] 最终测试 Top-1："
        f"{float(test_metrics['top1_accuracy']):.4f}"
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "label_smoothing_0.1"
        / "best_model.pth"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError("最佳模型 checkpoint 缺少 model_state_dict")
    if int(checkpoint.get("number_of_classes", 0)) != 37:
        raise ValueError("checkpoint 的类别数不是 37")
    print(f"[PASS] 最终模型可加载，训练轮次：{checkpoint.get('epoch')}")

    print("A5 工程完整性检查全部通过。")


if __name__ == "__main__":
    main()
