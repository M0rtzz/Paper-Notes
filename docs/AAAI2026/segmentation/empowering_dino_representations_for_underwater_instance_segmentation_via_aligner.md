---
title: >-
  [论文解读] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter
description: >-
  [AAAI2026][图像分割][图像分割] 首次将 DINOv2 引入水下实例分割任务，通过 AquaStyle Aligner（傅里叶频域风格注入）和 ObjectPrior Prompter（二值掩码先验提示）两个模块实现高效领域适配，在 UIIS 和 USIS10K 数据集上以更少参数大幅超越 SAM 基方法。
tags:
  - AAAI2026
  - 图像分割
  - DINOv2
  - 域适应
  - Fourier style transfer
  - foundation model fine-tuning
---

# Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter

**会议**: AAAI2026  
**arXiv**: [2511.08334](https://arxiv.org/abs/2511.08334)  
**代码**: [ettof/Diveseg](https://github.com/ettof/Diveseg)  
**领域**: segmentation  
**关键词**: underwater instance segmentation, DINOv2, domain adaptation, Fourier style transfer, foundation model fine-tuning

## 一句话总结

首次将 DINOv2 引入水下实例分割任务，通过 AquaStyle Aligner（傅里叶频域风格注入）和 ObjectPrior Prompter（二值掩码先验提示）两个模块实现高效领域适配，在 UIIS 和 USIS10K 数据集上以更少参数大幅超越 SAM 基方法。

## 背景与动机

水下实例分割（Underwater Instance Segmentation, UIS）需要同时完成像素级分类和实例级区分，是海洋探索、生态监测和水下机器人导航的核心技术。水下图像面临独特挑战：

- **光吸收与散射**：长波光被水吸收，导致图像呈蓝绿色调偏移
- **前向散射**造成模糊，**后向散射**降低能见度
- 退化效果不均匀且依赖深度，实例外观变化大

早期 CNN 方法（如 WaterMask）受限于表征能力；SAM 基方法（USIS-SAM）虽引入视觉基础模型，但依赖大规模水下标注数据且改善有限。DINOv2 通过自监督学习获得任务无关的通用特征，泛化能力强，在标注数据稀缺的水下场景更有优势。然而 PCA 可视化表明，直接迁移 DINOv2 到水下任务时特征受背景噪声影响严重，且可能漏检目标。

## 核心问题

如何从两个层面高效适配 DINOv2 到水下场景：

1. **场景级适配**：消除水下色彩偏移与预训练域的不对齐
2. **目标级适配**：使模型泛化到珊瑚、水母、海龟等在 LVD-142M 预训练数据中罕见的水下目标

## 方法详解

### 整体框架 DiveSeg

基于冻结的 DINOv2 ViT-L 骨干网络 + Mask2Former 分割头，外加两个核心适配模块。ViT 层被均分为四个块，每块第一层插入 AquaStyle Aligner，每块之后插入 ObjectPrior Prompter。

### AquaStyle Aligner（水下风格对齐器）

**目标**：从场景层面消除水下色彩域偏移。

**Style Extraction（风格提取）**：

- 对输入图像做 Fourier 变换，分离振幅分量（包含低层统计特征，如色彩信息）和相位分量（包含内容/结构信息）
- 将相位固定为平均值，仅保留振幅信息，通过逆 Fourier 变换重建"风格图像"——去除了目标内容仅保留水下色彩特征
- 用多层卷积 + 全局平均池化将风格图像编码为紧凑的风格向量 $p_x$

**Style Injection（风格注入）**：

- 作为 ViT 中 Multi-head Attention 的并行分支，使用交叉注意力机制：ViT 特征作 query，风格向量经 MLP 后作 key/value
- 交叉注意力的输出与原 MHA 输出相加：$\omega_1 = MHA(V_{in}) + CrossAttn(V_{in}, MLP(p_x))$
- 在后续 Feed-Forward 层同样并行一个 bottleneck MLP 结构做更深层的特征融合
- MHA 和 FF 的所有原始参数冻结，仅训练注入模块的参数

### ObjectPrior Prompter（目标先验提示器）

**目标**：从目标层面提供实例无关的前景先验，降低直接学习实例分割的难度。

**多尺度编码器**：三层卷积提取特征，步长为 2 下采样，输出三尺度特征金字塔 $\{f_M^1, f_M^2, f_M^3\}$（分辨率 1/8², 1/16², 1/32²）。

**伪掩码生成**：各尺度特征经 1×1 卷积 + Sigmoid 生成伪掩码 $P_{mask}^k$，由二值前景掩码（ground truth 合并所有实例得到）监督。

**特征增强**：伪掩码与原始特征逐元素相乘过滤前景，再通过卷积和残差连接融合：$f_{MT}^k = Conv(P_{mask}^k \cdot f_M^k) + f_M^k$

**先验注入**：多尺度增强特征展平拼接为 $O_{prompt}$，通过交叉注意力与 ViT 特征交互（$O_{prompt}$ 作 key/value，ViT 特征作 query），输出与原始 ViT 特征相加后送入解码器。

### 训练设置

- 骨干：DINOv2 ViT-L（冻结）
- 解码头：Mask2Former
- 优化器：AdamW，weight decay 0.05，初始 lr 1e-4，warmup
- 30,000 次迭代，在第 23,000 和 27,000 次衰减 lr 至 1/10
- 损失：分类损失 + 掩码损失（Mask2Former）+ BCE + IoU + L1 损失（伪掩码）
- 硬件：NVIDIA A100，batch size 8

## 实验关键数据

### UIIS 数据集（7类，3937 训练 / 691 测试）

| 方法 | 骨干 | 参数量 | mAP | AP50 | AP75 |
|------|------|--------|-----|------|------|
| WaterMask | ResNet-101 | 67M | 27.2 | 43.7 | 29.3 |
| USIS-SAM | ViT-H | 701M | 29.4 | 45.0 | 32.3 |
| **DiveSeg** | **ViT-L** | **390M** | **35.6** | **52.0** | **38.5** |

相比 USIS-SAM：mAP +21.1%，AP50 +15.6%，AP75 +19.2%，且参数量仅为其 55.6%。

### USIS10K 数据集（class-agnostic / multi-class）

| 方法 | class-agnostic mAP | multi-class mAP |
|------|---------------------|-----------------|
| USIS-SAM (ViT-H, 701M) | 59.7 | 43.1 |
| **DiveSeg (ViT-L, 390M)** | **64.1** | **48.4** |

### 消融实验

| 配置 | mAP | AP50 | AP75 |
|------|-----|------|------|
| DINOv2 + Mask2Former（baseline） | 30.9 | 44.6 | 32.2 |
| + AquaStyle Aligner | 34.1 | 50.8 | 37.8 |
| + ObjectPrior Prompter | 34.8 | 50.6 | 37.6 |
| **完整模型** | **35.6** | **52.0** | **38.5** |

### 适配策略对比（替代 AquaStyle Aligner）

| 策略 | mAP |
|------|-----|
| Frozen（不适配） | 30.9 |
| Full Fine-tuning | 31.1 |
| LoRA | 31.8 |
| Adapter | 32.7 |
| **AquaStyle Aligner** | **34.1** |

Full Fine-tuning 效果差可能因灾难性遗忘；AquaStyle Aligner 通过显式建模水下风格信息，优于通用参数高效微调策略。

## 亮点

- **首次将 DINOv2 引入水下实例分割**，证明自监督预训练基础模型可高效适配水下场景
- **AquaStyle Aligner 设计精巧**：利用 Fourier 频域分解捕捉水下色彩特征，通过交叉注意力注入 ViT，物理直觉清晰
- **ObjectPrior Prompter 思路巧妙**：将实例分割解耦为"前景感知 → 实例区分"两阶段，降低学习难度
- **参数效率极高**：仅 390M 参数（ViT-L）超越 701M 参数的 USIS-SAM（ViT-H），且大部分参数冻结
- 定性结果显示在阴影中鱼群分割、重叠实例区分、误分类纠正等困难场景均有明显优势

## 局限与展望

- 仅在 UIIS 和 USIS10K 两个数据集上验证，水下场景多样性有限
- ObjectPrior Prompter 训练时依赖 ground truth 合并的二值掩码，推理时使用预测伪掩码，预测质量直接影响最终效果
- 仅使用 ViT-L，未探索 ViT-B（更轻量）或 ViT-G（更强）的扩展性
- 未讨论推理速度和实时性，对水下机器人等实时应用场景的适用性存疑
- 风格提取依赖全局平均相位，不同深度/水域条件下风格变化可能更复杂

## 与相关工作的对比

| 维度 | WaterMask | USIS-SAM | DiveSeg |
|------|-----------|----------|---------|
| 基础模型 | CNN (ResNet) | SAM (ViT-H) | DINOv2 (ViT-L) |
| 参数量 | 67M | 701M | 390M |
| 预训练方式 | 监督 | 监督（掩码标注） | 自监督 |
| 域适配策略 | 专用模块 | LoRA + Adapter | 风格注入 + 先验提示 |
| 核心思路 | 端到端学习 | 提示工程 | 双层适配（场景+目标） |

## 启发与关联

- **Fourier 频域风格迁移**的思路可推广到其他域适配场景（如医学图像、遥感图像），其核心洞察是振幅包含域相关的低层统计特征
- **解耦适配策略**（场景级 + 目标级）是一种通用的基础模型迁移范式，可用于其他领域特定的分割任务
- ObjectPrior Prompter 的先验注入方式与 SAM 的 prompt engineering 形成互补思路：前者学习隐式先验，后者需显式设计 prompt
- 在基础模型时代，参数高效微调 + 领域知识注入的组合可能是小数据域迁移的最佳实践

## 评分
- 新颖性: 4/5（首次引入 DINOv2 到 UIS，两个模块设计有洞察力）
- 实验充分度: 4/5（两个数据集、充分消融、多策略对比，但缺速度分析）
- 写作质量: 4/5（结构清晰，动机阐述充分）
- 价值: 4/5（为水下视觉领域提供了强基线，迁移思路有通用性）

<!-- RELATED:START -->

## 相关论文

- [Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [A Mixed Diet Makes DINO An Omnivorous Vision Encoder](../../CVPR2026/segmentation/mixed_diet_dino_omnivorous_encoder.md)
- [UniFS: Universal Few-Shot Instance Perception with Point Representations](../../ECCV2024/segmentation/unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](../../CVPR2026/segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)
- [Phrase-Instance Alignment for Generalized Referring Segmentation](../../CVPR2026/segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)

<!-- RELATED:END -->
