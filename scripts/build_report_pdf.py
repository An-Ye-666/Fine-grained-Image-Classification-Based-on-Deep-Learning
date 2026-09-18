"""Build the final 2-3 page technical report PDF."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "report"
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_PDF = REPORT_DIR / "【考核】叶安_X124306049_宠物分类.pdf"

STUDENT_NAME = "叶安"
STUDENT_ID = "X124306049"
GITHUB_URL = (
    "https://github.com/An-Ye-666/"
    "Fine-grained-Image-Classification-Based-on-Deep-Learning"
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def register_chinese_font() -> str:
    """Register a font that contains Chinese glyphs."""

    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), "MicrosoftYaHei", 0),
        (Path("C:/Windows/Fonts/simhei.ttf"), "SimHei", None),
    ]
    for path, name, subfont_index in candidates:
        if not path.is_file():
            continue
        if subfont_index is None:
            pdfmetrics.registerFont(TTFont(name, str(path)))
        else:
            pdfmetrics.registerFont(
                TTFont(name, str(path), subfontIndex=subfont_index)
            )
        return name
    raise FileNotFoundError("没有找到可用的中文字体")


def image_flowable(path: Path, width_cm: float) -> Image:
    """Scale an image while preserving its aspect ratio."""

    with PillowImage.open(path) as image:
        width_px, height_px = image.size
    width = width_cm * cm
    height = width * height_px / width_px
    return Image(str(path), width=width, height=height)


def make_table(
    rows: list[list[object]],
    widths: list[float],
    font_name: str,
    header: bool = True,
) -> Table:
    table = Table(rows, colWidths=[width * cm for width in widths])
    commands: list[tuple[object, ...]] = [
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("LEADING", (0, 0), (-1, -1), 10.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C3D0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def build_story(font_name: str) -> list[object]:
    baseline = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "baseline"
        / "validation_metrics.json"
    )
    smoothing = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "label_smoothing_0.1"
        / "validation_metrics.json"
    )
    test_data = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "analysis"
        / "final_model"
        / "test_metrics.json"
    )
    split_data = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "oxford_pet_seed42.json"
    )

    test_metrics = test_data["metrics"]
    baseline_top1 = float(baseline["top1_accuracy"])
    baseline_f1 = float(baseline["macro_f1"])
    smoothing_top1 = float(smoothing["top1_accuracy"])
    smoothing_f1 = float(smoothing["macro_f1"])
    test_top1 = float(test_metrics["top1_accuracy"])
    test_top5 = float(test_metrics["top5_accuracy"])
    test_f1 = float(test_metrics["macro_f1"])

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ChineseBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.7,
        leading=11.3,
        spaceAfter=4,
        wordWrap="CJK",
    )
    heading = ParagraphStyle(
        "ChineseHeading",
        parent=body,
        fontSize=11.2,
        leading=14,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=5,
        spaceAfter=4,
    )
    title = ParagraphStyle(
        "ChineseTitle",
        parent=body,
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=7,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=7.7,
        leading=9.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
    )

    story: list[object] = [
        Paragraph("基于深度学习的牛津宠物细粒度分类实验报告", title),
        Paragraph(
            f"姓名：{STUDENT_NAME} | 学号：{STUDENT_ID} | "
            f"<link href='{GITHUB_URL}'>GitHub 仓库</link>",
            ParagraphStyle(
                "Metadata",
                parent=body,
                alignment=TA_CENTER,
                fontSize=8.5,
                leading=11,
            ),
        ),
        Spacer(1, 4),
        Paragraph("1. 任务背景与实验设置", heading),
        Paragraph(
            "本实验使用 Oxford-IIIT Pet 数据集完成 37 类猫狗品种的细粒度图像分类。"
            "官方 trainval 与 test 清单合并后共有 7,349 张有效图片。为保证类别比例一致，"
            "全部样本按类别进行 70%/15%/15% 分层划分，随机种子固定为 42。测试集仅在"
            "最终配置冻结后评估一次。",
            body,
        ),
        make_table(
            [
                ["数据集", "训练集", "验证集", "测试集", "总样本"],
                [
                    "Oxford-IIIT Pet",
                    int(split_data["counts"]["train"]),
                    int(split_data["counts"]["val"]),
                    int(split_data["counts"]["test"]),
                    sum(int(value) for value in split_data["counts"].values()),
                ],
            ],
            [3.8, 2.5, 2.5, 2.5, 2.7],
            font_name,
        ),
        Spacer(1, 5),
        Paragraph(
            "模型使用 ImageNet 预训练 ResNet-18，并将最后全连接层替换为 37 类输出。"
            "输入尺寸为 224×224，训练集采用 Resize、RandomCrop 和水平翻转，验证与测试集"
            "仅使用 Resize、CenterCrop 和归一化。优化器为 AdamW，学习率为 1e-4，"
            "Weight Decay 为 1e-4，Batch Size 为 32。",
            body,
        ),
        make_table(
            [
                ["配置项", "设置"],
                ["模型", "预训练 ResNet-18，全参数微调"],
                ["Baseline 损失", "CrossEntropyLoss"],
                ["消融损失", "CrossEntropyLoss(label_smoothing=0.1)"],
                ["选择标准", "验证集 Top-1 Accuracy"],
                ["训练设备", "本地 CPU，PyTorch 2.14.0+cpu"],
            ],
            [4.5, 10.0],
            font_name,
        ),
        PageBreak(),
        Paragraph("2. 实验结果与消融分析", heading),
        Paragraph(
            "Baseline 共训练 15 轮，最佳结果出现在第 4 轮，之后训练集准确率继续上升而"
            "验证集表现下降，说明出现过拟合。Label Smoothing 实验使用独立脚本训练 10 轮，"
            "最佳结果出现在第 9 轮。两组使用相同数据划分、随机种子和优化配置，唯一变量"
            "为损失函数。",
            body,
        ),
        make_table(
            [
                ["实验", "Top-1", "Top-5", "Macro-F1", "最佳轮次"],
                [
                    "Baseline，验证集",
                    percentage(baseline_top1),
                    percentage(float(baseline["top5_accuracy"])),
                    f"{baseline_f1:.4f}",
                    4,
                ],
                [
                    "Label Smoothing 0.1，验证集",
                    percentage(smoothing_top1),
                    percentage(float(smoothing["top5_accuracy"])),
                    f"{smoothing_f1:.4f}",
                    9,
                ],
                [
                    "Label Smoothing 0.1，测试集",
                    percentage(test_top1),
                    percentage(test_top5),
                    f"{test_f1:.4f}",
                    9,
                ],
            ],
            [6.0, 2.1, 2.1, 2.2, 2.1],
            font_name,
        ),
        Spacer(1, 3),
        Table(
            [
                [
                    image_flowable(
                        FIGURE_DIR / "baseline_training_curves.png",
                        7.6,
                    ),
                    image_flowable(
                        FIGURE_DIR / "label_smoothing_training_curves.png",
                        7.6,
                    ),
                ]
            ],
            colWidths=[8.0 * cm, 8.0 * cm],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        ),
        Paragraph("左：Baseline 训练曲线；右：Label Smoothing 训练曲线。", caption),
        image_flowable(FIGURE_DIR / "final_comparison.png", 16.0),
        Paragraph("图 1：最佳验证集与最终测试集指标对比。", caption),
        Paragraph(
            "相对于 Baseline，Label Smoothing 在验证集上使 Top-1 提升约 0.73 个百分点，"
            "Macro-F1 提升约 0.71 个百分点。该提升幅度较小，且目前只完成一次随机种子"
            "实验，因此结论应限定在当前数据划分和训练配置下。",
            body,
        ),
        PageBreak(),
        Paragraph("3. 错误案例与 Grad-CAM", heading),
        Paragraph(
            "最终模型在测试集上的主要错误集中在毛色和身体结构相近的品种。混淆矩阵显示，"
            "Egyptian_Mau 最容易被预测为 Bengal，american_pit_bull_terrier 最容易与"
            "staffordshire_bull_terrier 混淆，Siamese 则容易被预测为 Birman。",
            body,
        ),
        make_table(
            [
                ["真实类别", "预测类别", "数量", "占真实类别比例"],
                ["Egyptian_Mau", "Bengal", 5, "17.24%"],
                [
                    "american_pit_bull_terrier",
                    "staffordshire_bull_terrier",
                    5,
                    "16.67%",
                ],
                ["Siamese", "Birman", 4, "13.33%"],
            ],
            [5.5, 6.0, 2.0, 3.0],
            font_name,
        ),
        Spacer(1, 3),
        image_flowable(
            PROJECT_ROOT
            / "artifacts"
            / "analysis"
            / "final_model"
            / "confusion_matrix.png",
            8.0,
        ),
        Paragraph("图 2：37 类测试集混淆矩阵。", caption),
        Table(
            [
                [
                    image_flowable(
                        PROJECT_ROOT
                        / "artifacts"
                        / "analysis"
                        / "final_model"
                        / "gradcam_correct.png",
                        7.7,
                    ),
                    image_flowable(
                        PROJECT_ROOT
                        / "artifacts"
                        / "analysis"
                        / "final_model"
                        / "gradcam_error.png",
                        7.7,
                    ),
                ]
            ],
            colWidths=[8.0 * cm, 8.0 * cm],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        ),
        Paragraph("图 3：正确案例与错误案例 Grad-CAM。", caption),
        Paragraph("4. AI 辅助编程说明与 Bug 复盘", heading),
        Paragraph(
            "AI 工具主要参与项目结构设计、数据检查、训练脚本、评估脚本和 Grad-CAM"
            "实现。关键训练由项目负责人在本地执行，实验指标均来自实际运行结果。"
            "开发过程中发现过一个真实问题：训练脚本最初直接序列化 argparse 中的 Path"
            "对象，导致保存 metrics.json 时会失败。修复方式是先通过 config_to_dict 将"
            "Path 转为字符串，再加入 JSON 写出流程。修复后重新运行并确认指标文件正常生成。",
            body,
        ),
        Paragraph("5. 局限与结论", heading),
        Paragraph(
            "项目完成了 37 类细粒度分类的 Baseline、单变量消融、测试集评估、混淆矩阵和"
            "Grad-CAM 分析。最终测试 Top-1 为 "
            f"{percentage(test_top1)}，达到任务目标。主要局限是只使用单一随机种子，"
            "Label Smoothing 的提升幅度较小，且本地 CPU 训练存在明显计算成本。后续可"
            "通过多随机种子实验、更合理的数据增强和早停策略进一步验证稳定性。",
            body,
        ),
    ]
    return story


def build_pdf() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = register_chinese_font()

    document = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="基于深度学习的牛津宠物细粒度分类实验报告",
        author=STUDENT_NAME,
    )

    def add_footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            1.5 * cm,
            0.72 * cm,
            "Oxford-IIIT Pet Fine-grained Classification",
        )
        canvas.drawRightString(
            A4[0] - 1.5 * cm,
            0.72 * cm,
            f"第 {doc.page} 页",
        )
        canvas.restoreState()

    document.build(
        build_story(font_name),
        onFirstPage=add_footer,
        onLaterPages=add_footer,
    )
    return OUTPUT_PDF


if __name__ == "__main__":
    output = build_pdf()
    print(output)
