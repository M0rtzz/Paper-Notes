---
title: >-
  [论文解读] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception
description: >-
  [CVPR 2025][语义分割][开放词汇分割] DeCLIP 发现 CLIP 的自注意力中存在"代理 token"现象导致图像 token 无法聚合空间相关信息，提出将自注意力模块解耦为"内容"和"上下文"特征并分别用 CLIP 自蒸馏和视觉基础模型蒸馏进行优化的框架，在开放词汇目标检测和语义分割上全面超越现有方法。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "开放词汇分割"
  - "CLIP"
  - "解耦注意力"
  - "知识蒸馏"
  - "密集预测"
---

# DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception

**会议**: CVPR 2025  
**arXiv**: [2505.04410](https://arxiv.org/abs/2505.04410)  
**代码**: [https://github.com/xiaomoguhz/DeCLIP](https://github.com/xiaomoguhz/DeCLIP)  
**领域**: 图像分割  
**关键词**: 开放词汇分割, CLIP, 解耦注意力, 知识蒸馏, 密集预测

## 一句话总结
DeCLIP 发现 CLIP 的自注意力中存在"代理 token"现象导致图像 token 无法聚合空间相关信息，提出将自注意力模块解耦为"内容"和"上下文"特征并分别用 CLIP 自蒸馏和视觉基础模型蒸馏进行优化的框架，在开放词汇目标检测和语义分割上全面超越现有方法。

## 研究背景与动机
传统密集预测方法（目标检测、图像分割）依赖预定义类别，无法处理无界的视觉概念。视觉语言模型 CLIP 的出现为开放词汇密集预测提供了可能，但直接将 CLIP 应用于密集预测存在域偏移问题。

**CLIP 为什么不适合密集预测？** 作者通过分析 CLIP 不同层的注意力图发现了关键问题 —— **"代理 token"现象**：
- 在浅层，CLS token 的注意力广泛分布在整个图像上
- 在深层（第 9 层之后），CLS token 不再关注主要物体，而是高度关注背景中的特定 token
- 更严重的是，**图像 token 也表现出与 CLS token 类似的行为** —— 无论自身位置在哪，都高度关注那些"代理" token
- 这些"代理" token 充当 CLS token 的信息中转站，虽然有助于图像级分类，但**破坏了图像 token 之间的空间/语义关联**

核心矛盾：密集预测需要图像 token 具有**局部判别性**（能区分不同区域的语义）和**空间一致性**（同语义区域的 token 相关联），但 CLIP 的"代理 token"现象使这两个能力都受损。直接将自蒸馏和 VFM 蒸馏同时施加在同一特征上会导致优化冲突（区域分类下降 3.9 mAcc）。

DeCLIP 的核心 idea：**解耦** CLIP 最后一层自注意力模块中的 Q（上下文特征）和输出（内容特征），分别用不同的教师模型蒸馏，避免优化冲突。

## 方法详解

### 整体框架
DeCLIP 是一种无监督的预微调方法，输入原始图像和裁剪子图。CLIP 的最后一层自注意力被重新解释：Q 投影输出作为"上下文"特征，负责空间一致性；注意力加权后的输出作为"内容"特征，负责局部判别性。CLIP 自身（冻结副本）作为内容特征的教师进行自蒸馏，视觉基础模型（如 DINOv2）作为上下文特征的教师。

### 关键设计
1. **解耦注意力 (Decoupled Attention)**:

    - 从 CLIP 最后一层的自注意力中提取两种特征：
    - **上下文特征** $\mathbf{X}_{context}$：直接取 Q 投影输出 $\text{Proj}_q(\mathbf{X})$
    - **内容特征** $\mathbf{X}_{content}$：用上下文特征的自注意力（$\text{Attn}_{context} = \text{SoftMax}(\mathbf{X}_{context} \mathbf{X}_{context}^T / \sqrt{d})$）对 V 加权求和，再投影
    - 灵感来源：之前的训练无关 OVS 方法（如 SCLIP、ClearCLIP）将 $\text{Attn}_{qk}$ 改为 $\text{Attn}_{qq}$ 并去掉残差连接来改善分割效果，DeCLIP 将这个洞察推广为可训练的解耦蒸馏框架
    - 通过解耦，可以对两种特征施加不同约束而不干扰

2. **内容特征蒸馏 (Content Feature Distillation)**:

    - 教师模型：CLIP 冻结副本（自蒸馏）
    - 将输入图像切成 k 个子区域，裁剪为子图
    - 学生：从内容特征图上用 RoI Align 提取区域特征 $\mathbf{f}_i^s$
    - 教师：将裁剪子图输入冻结 CLIP，获取对应 CLS token $\mathbf{f}_i^t$
    - 损失：余弦相似度对齐 $\mathcal{L}_{content} = \frac{1}{k} \sum_{i=1}^k (1 - \cos(\mathbf{f}_i^t, \mathbf{f}_i^s))$
    - 直觉：CLIP 用图像裁剪分类（CLS token）比用区域特征分类准确率更高，因此对齐可以提升区域特征的判别性

3. **上下文特征蒸馏 (Context Feature Distillation)**:

    - 教师模型：视觉基础模型（DINOv2 效果最佳）
    - 对 VFM 和 CLIP 的上下文特征分别计算 token 间的余弦相似度矩阵（相关体积）
    - 用 L2 损失对齐两者的相关体积：$\mathcal{L}_{context} = \frac{1}{HW} \sum_i \sum_j \|r_{ij}^{VFM} - r_{ij}^{CLIP}\|_2$
    - VFM 不存在"代理 token"现象，且语义相关 token 之间有更好的关联
    - 通过蒸馏 VFM 的相关性，改善 CLIP token 的空间一致性

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{content} + \lambda \mathcal{L}_{context}$
- 无监督预微调：不需要标注数据
- 微调完成后，增强的 CLIP 可即插即用于各种下游密集预测框架

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | DeCLIP | 基线 | 提升 |
|--------|------|------|----------|------|
| OV-COCO 检测 (F-ViT, ViT-B) | AP50 Novel | 41.1 | 37.6 (CLIPSelf) | +3.5 |
| OV-COCO 检测 (OV-DQUO, ViT-B) | AP50 Novel | 46.1 | 39.2 (OV-DQUO) | +6.9 |
| OV-COCO 检测 (OV-DQUO, ViT-L) | AP50 Novel | 48.3 | 45.6 (OV-DQUO) | +2.7 |
| OV-LVIS 检测 (OV-DQUO, ViT-L) | mAP rare | 41.5 | 39.3 (OV-DQUO) | +2.2 |
| OV 语义分割 (CAT-Seg+DeCLIP, ViT-B) | ADE150 mIoU | 36.3 | 31.8 (CAT-Seg) | +4.5 |
| OV 语义分割 (CAT-Seg+DeCLIP, ViT-L) | ADE150 mIoU | 40.7 | 37.9 (CAT-Seg) | +2.8 |
| VLM 特征分割 (8 benchmarks) | Avg mIoU | 41.9 | 38.2 (SCLIP) | +3.7 |

### 消融实验

| 配置 | 区域分类 mAcc (Thing/Stuff) | 语义分割 mIoU (Context59/CityScape) |
|------|---------|------|
| 自蒸馏 only (CLIPSelf) | 69.5 / 44.6 | 29.4 / 25.6 |
| 自蒸馏 + VFM 蒸馏（不解耦） | 65.6 / 41.3 (-3.9/-3.3) | 32.4 / 28.7 (+3.0/+3.1) |
| 自蒸馏 + VFM + 解耦 (DeCLIP) | 75.0 / 51.8 (+5.5/+7.2) | 35.3 / 32.3 (+5.9/+6.7) |

| VFM 选择 | 区域分类 (Thing) | Context59 | ADE | 特点 |
|----------|-----------------|-----------|-----|------|
| DINO ViT-B/16 | 67.6 | 38.1 | 20.4 | 分割中等，分类较弱 |
| SAM ViT-B/16 | 75.0 | 35.3 | 18.5 | 分类强，分割弱 |
| DINOv2 ViT-B/14 | 77.2 | 39.2 | 21.9 | 两者兼顾最佳 |

### 关键发现
- **不解耦直接蒸馏会产生优化冲突**：区域分类性能反而下降 3.9 mAcc，这是提出解耦策略的核心动机
- 解耦后两个能力同时大幅提升：区域分类 +5.5，语义分割 +5.9，证明解耦有效避免了冲突
- DINOv2 作为 VFM 教师效果最佳：SAM 缺乏语义关联能力，DINO 对所有主要物体无差别关注
- 在 ViT-B 尺度上，CAT-Seg+DeCLIP 就几乎超越了所有使用更大编码器（ConvNeXt-L）的现有方法

## 亮点与洞察
1. **"代理 token"现象的发现**：深入分析了 CLIP 在密集预测上失败的原因，不是简单的域偏移，而是注意力机制中特定 token 充当信息中转站导致图像 token 间空间关联被破坏
2. **解耦蒸馏避免优化冲突**：不解耦时自蒸馏和 VFM 蒸馏相互干扰，解耦后两个目标可以独立优化，这一洞察具有重要方法论价值
3. **无监督预微调范式**：DeCLIP 不需要标注数据，微调后可直接用于任何下游检测/分割框架，通用性强
4. **VFM 选择的分析**：揭示了不同 VFM（DINO vs SAM vs DINOv2）在空间一致性引导上的差异，DINOv2 兼顾语义和空间最优

## 局限与展望
- 目前仅解耦最后一层的自注意力，是否解耦更多层以获取多尺度信息值得探索
- 上下文蒸馏使用简单的 L2 损失对齐相关体积，更精细的蒸馏策略可能有进一步提升
- VFM 教师是固定的，能否动态调整蒸馏权重或使用多个 VFM 集成
- 内容蒸馏依赖均匀网格裁剪，可能对非网格分布的物体不够灵活
- 额外的 VFM 前传引入计算开销，尽管仅在预微调阶段

## 相关工作与启发
- CLIPSelf 提出了自蒸馏提升区域分类的思路，DeCLIP 在此基础上加入 VFM 引导并解决了优化冲突
- ClearCLIP/SCLIP 等训练无关方法通过修改注意力机制提升分割效果，启发了 DeCLIP 的 Q-Q attention 设计
- CLIM 使用马赛克拼图作为伪区域，DeCLIP 使用实际裁剪更加精确
- 对开放词汇分割的启发：预训练模型的特征质量是上限，在特征层面的解耦增强比下游框架的复杂设计更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ "代理 token"现象的发现有洞察力，解耦蒸馏的设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 检测(OV-COCO, OV-LVIS) + 分割(6 OVS benchmarks) + VLM 特征分割(8 benchmarks) + 区域分类 + 跨数据集迁移
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析深入，图表直观，从"代理 token"发现到优化冲突到解耦方案的论证逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 在开放词汇密集预测任务上具有很强的实用价值，通用性框架可即插即用于多种下游方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)

</div>

<!-- RELATED:END -->
