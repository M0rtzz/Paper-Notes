---
title: >-
  [论文解读] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline
description: >-
  [CVPR 2025][医学图像][交互式医学图像分割] 本文提出 IMed-361M，一个包含 640 万张图像和 3.61 亿个 mask（平均每张 56 个）的大规模交互式医学图像分割基准数据集，覆盖 14 种成像模态和 204 个分割目标，并基于此开发了支持点击、边框、文本及组合交互的 IMIS 基线网络，在多个场景下超越现有视觉基础模型。
tags:
  - CVPR 2025
  - 医学图像
  - 交互式医学图像分割
  - 大规模数据集
  - SAM微调
  - 密集标注
  - 多模态医学影像
---

# Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline

**会议**: CVPR 2025  
**arXiv**: [2411.12814](https://arxiv.org/abs/2411.12814)  
**代码**: https://github.com/uni-medical/IMIS-Bench  
**领域**: 医学图像 / 交互式分割  
**关键词**: 交互式医学图像分割, 大规模数据集, SAM微调, 密集标注, 多模态医学影像

## 一句话总结
本文提出 IMed-361M，一个包含 640 万张图像和 3.61 亿个 mask（平均每张 56 个）的大规模交互式医学图像分割基准数据集，覆盖 14 种成像模态和 204 个分割目标，并基于此开发了支持点击、边框、文本及组合交互的 IMIS 基线网络，在多个场景下超越现有视觉基础模型。

## 研究背景与动机

1. **领域现状**：SAM 及其衍生模型（MedSAM、SAM-Med2D）推动了交互式分割的发展，但在医学影像中的"分割一切"能力仍受限。SA-1B 自然图像数据集平均每张 100 个 mask，而医学数据集如 COSMOS 平均仅 5.7 个。
2. **现有痛点**：(a) 现有医学分割数据集标注稀疏——每张图像仅有少量 mask，限制了密集分割和细粒度交互能力；(b) 数据集通常局限于特定模态或任务，泛化能力差；(c) 不同 IMIS 方法的评估缺乏统一标准，可比性和可靠性差。
3. **核心矛盾**：医学影像基础模型需要类似 SA-1B 的大规模密集标注数据集，但医学标注成本极高，且现有整合数据集的标注密度远不够。
4. **本文目标** (a) 构建大规模、多样、密集标注的 IMIS 基准数据集；(b) 建立支持多种交互策略的统一基线模型；(c) 提供公平一致的 IMIS 模型评估框架。
5. **切入角度**：利用 SAM 的物体识别能力自动生成密集 interactive masks，再通过 GT 引导的质量控制确保标注质量。
6. **核心 idea**：整合 110+ 公开数据集并用 SAM 自动生成高密度交互标注（每张图 56 个 mask），构建覆盖 14 个模态的 3.61 亿 mask 基准数据集，训练统一的多策略交互式分割基线。

## 方法详解

### 整体框架
数据构建流水线：收集 110+ 数据集 → 标准化预处理 → 冲突消歧 → SAM 自动生成 interactive masks → GT 引导的质量控制 → 最终得到 640 万图像 + 3.61 亿 mask。模型架构采用 ViT-Base 编码器 + 多策略提示编码器（点+框+文本）+ Transformer 解码器，通过模拟连续交互的训练策略进行端到端训练。

### 关键设计

1. **自动 Mask 生成 + GT 引导的质量控制**:

    - 功能：为每张图像生成密集、高质量的交互式 mask
    - 核心思路：用 SAM 的 32×32 点网格生成候选 mask，通过三重筛选：IoU 置信度 > 0.85 保留、NMS（IoU > 0.7 去重）、移除覆盖 > 80% 的背景 mask。关键的质量控制——用原始 GT 校正生成的 mask：对 GT 中的多连通区域直接替换对应生成 mask；对单连通区域，若生成 mask 的最小外接框与 GT 重叠 > 95% 则保留生成版，否则用 GT 替换。最后用形态学操作去噪填洞
    - 设计动机：SAM 自动生成的 mask 常无法正确分离边界模糊的结构（如心脏的心房心室），且分散结构（如肠道）常被识别为多个独立对象。GT 引导的校正解决了这些医学特有的分割粒度问题

2. **多策略提示编码器**:

    - 功能：支持点击、边框、文本三种交互输入及其组合
    - 核心思路：点和框通过位置编码+可学习嵌入表示；文本通过 CLIP 文本编码器编码，模板为"A segmentation area of a [category]"，覆盖 200+ 器官和病变类别。三种提示可以自由组合——文本提供全局语义引导，点击/框提供局部空间定位
    - 设计动机：现有方法通常只支持单一交互策略，无法评估不同交互方式的影响。统一多策略设计为未来多模态分割研究提供了基准

3. **模拟连续交互训练策略**:

    - 功能：训练模型通过多轮交互逐步改善分割结果
    - 核心思路：对每张图像和目标 mask，先模拟初始交互（随机前景采样点、目标最小包围框+5px偏移），产生初始预测。然后进行 K=8 轮迭代：根据预测与 GT 的错误区域生成校正交互（新的正/负点击），同时提供上一轮的低分辨率预测 mask 作为额外线索。图像编码器仅编码一次，后续迭代只更新提示编码器和解码器参数
    - 设计动机：真实临床使用中，医生会通过多次点击逐步修正分割结果。模拟这一过程使模型学会从错误中学习，提高交互效率

### 损失函数 / 训练策略
使用 Focal Loss + Dice Loss 的线性组合（20:1 比例）。Adam 优化器，学习率 $2 \times 10^{-5}$，72 块 4090 GPU，batch size 2。每张图随机选 5 个目标（不足则重复选取），图像统一到 1024×1024，像素值 20% 概率随机缩放偏移。训练 12 个 epoch。评估使用 Dice score。

## 实验关键数据

### 主实验

**外部数据集评估（bbox 提示）：**

| 数据集 | SAM | SAM-2 | MedSAM | SAM-Med2D | **IMIS-Net** |
|--------|-----|-------|--------|-----------|-------------|
| ISLES (脑卒中) | 55.92 | 60.14 | 59.90 | 68.22 | **71.78** |
| SegThor (平均) | 84.46 | 85.86 | 60.55 | 86.43 | **89.27** |
| TotalSeg MRI (平均) | 75.45 | 77.62 | 59.52 | 75.92 | **79.06** |

**内部测试集（图像级/mask级统计）IMIS-Net 在点击和框两种交互下均取得最佳表现。**

### 消融实验

| 解码器维度 | 图像分辨率 | 点击 Dice | 框 Dice | 可训练参数 |
|-----------|----------|----------|---------|-----------|
| 768 | 256×256 | 0.8214 | 0.8469 | 29.68M |
| 768 | 512×512 | 0.8673 | 0.8968 | 29.68M |
| 256 | 1024×1024 | 0.8366 | 0.8497 | 5.52M |
| 512 | 1024×1024 | 0.8563 | 0.8729 | 15.19M |
| **768** | **1024×1024** | **0.8848** | **0.9060** | 29.68M |

**文本+组合提示效果：**

| 交互策略 | Dice |
|---------|------|
| 仅文本 | 76.30% |
| 文本+点击 | 88.25% (+11.95%) |
| 文本+点击+3轮校正 | 89.69% |

### 关键发现
- 密集 mask 对性能至关重要：仅用 GT 训练效果差，加入 interactive masks 后 Dice 快速提升
- 数据规模效应显著：训练数据量增加持续带来性能提升，验证了方法的可扩展性
- 输入分辨率影响最大：256→1024 分辨率提升使 Dice 从 84.69% 提升到 90.60%
- 解码器维度从 256 到 768 仅增加 24.16M 参数但 Dice 提升约 5.6%，说明当数据足够大时，模型容量是瓶颈
- 框交互始终优于点击（提供更多边界信息），多轮交互持续缩小不同模型间的差距
- 点击位置越靠近质心效果越好，SAM-2 的 Dice 提升 2.84%；bbox 偏移导致所有方法下降 0.85%-3.94%

## 亮点与洞察
- **用 SAM 为医学图像自动生成密集标注**是一个聪明的策略：利用通用模型的物体感知能力弥补医学标注的不足，再用现有 GT 校正保证质量。这个思路可以迁移到其他标注稀缺的专业领域。
- **GT 引导的粒度管理**解决了 SAM 在医学影像中的核心问题：处理多连通区域和边界模糊结构的替换策略很实用，体现了对医学影像特殊性的深入理解。
- **数据规模+解码器容量的 scaling 分析**提供了清晰的实践指导：即使模型性能接近饱和，扩展解码器仍能获得提升，为未来工作提供了简单有效的改进路径。

## 局限与展望
- 部分 interactive masks 缺乏直接临床意义（如 X 光背景字母、皮肤镜图像中的毛发），虽增加多样性但可能引入噪声
- 仅使用 ViT-Base 编码器，更大的编码器可能进一步提升性能
- 自动生成 mask 的语义信息缺失，未来需探索如何有效获取 interactive masks 的语义标签
- 3D 医学影像仅以 2D 切片处理，未充分利用体积信息
- 质量评估显示 18 个子集的 mask 质量较差，虽已清理但可能仍有残留问题

## 相关工作与启发
- **vs MedSAM**: MedSAM 在大规模医学数据上微调 SAM 但仅支持 bbox 交互，且标注密度不足。IMIS-Net 支持多策略交互且 mask 密度 56/image vs MedSAM 的 ~5/image
- **vs SAM-Med2D**: SAM-Med2D 缺乏骨骼结构数据导致该类解剖结构分割差，IMIS-Net 通过更全面的数据覆盖解决了这一偏差
- **vs SAM/SAM-2**: 自然图像预训练的模型在医学影像上表现大幅落后（单点击 Dice 仅 60%），证明了领域适配的必要性

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集构建思路有创新（SAM 自动生成+GT 校正），但模型架构基本是标准的 SAM 微调
- 实验充分度: ⭐⭐⭐⭐⭐ 多角度评估（模态、解剖结构、交互策略、点击位置、框偏移），外部数据集验证全面
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程清晰，但部分内容可以更简洁
- 价值: ⭐⭐⭐⭐⭐ 3.61 亿 mask 的基准数据集将成为医学影像基础模型的重要资源

<!-- RELATED:START -->

## 相关论文

- [Interactive Medical Image Analysis with Concept-based Similarity Reasoning](interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](../../ACL2025/medical_imaging/afrimed_qa_pan_african.md)
- [Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
