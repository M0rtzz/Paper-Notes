---
title: >-
  [论文解读] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment
description: >-
  [ECCV 2024][图像生成] 发现正负描述文本在数据集层面存在分布偏差（如词频差异），提出用纯文本分类器过滤偏差数据，微调 LLaVA-1.5 获得 SOTA 图文对齐评分模型 LLaVA-score。
tags:
  - ECCV 2024
  - 图像生成
---

# Removing Distributional Discrepancies in Captions Improves Image-Text Alignment

**会议**: ECCV 2024  
**arXiv**: [2410.00905](https://arxiv.org/abs/2410.00905)  
**领域**: 图像生成

## 一句话总结

发现正负描述文本在数据集层面存在分布偏差（如词频差异），提出用纯文本分类器过滤偏差数据，微调 LLaVA-1.5 获得 SOTA 图文对齐评分模型 LLaVA-score。

## 研究背景与动机

- 自动评估图像和文本的语义对齐对于数据清洗、模型评估和生成模型改进至关重要
- CLIP 等模型以"词袋"方式工作，无法区分 "horse eating grass" 和 "grass eating horse"
- 现有方法通过构造负面描述进行对比训练，但存在两个层面的问题：
  1. **实例层面**：已有方法（NegCLIP）通过随机打乱词序生成不流畅的负面描述，可被语法模型轻易区分
  2. **分布层面**（本文首次发现）：即使每条负面描述语法正确、语义合理，正负描述在词频分布上仍存在系统性偏差
- **典型案例**：COCO 数据集中 "giraffe" 出现频率远高于 "elephant"，但 GPT 生成负描述时倾向将 "giraffe" 替换为 "elephant"，导致模型仅凭文本就能区分正负样本

## 方法详解

### 整体框架

三步流程：构造多样化负描述 → 检测并过滤分布偏差 → 微调视觉语言模型

### 关键设计

**1. 混合负描述生成**

使用 GPT 生成两种类型的负面描述：

- **替换型（Replacing）**：将描述中某个语言元素替换为合理的替代品
    - 例："a photo of a broken down stop sign" → "a photo of a brand new stop sign"
    - 增强模型的**感知能力**

- **交换型（Swapping）**：重新排列原始描述中的语言组件
    - 例："an airplane is flying in the blue sky" → "a blue airplane is flying in the sky"
    - 增强模型的**推理能力**

**2. 分布偏差检测与过滤**

核心创新——用纯文本分类器暴露偏差：

1. 训练一个仅看文本（不看图像）的 BERT 二分类器
2. 将数据集分为 $N$ 个分区，交叉验证方式训练并预测
3. **移除被分类器高置信度正确预测的 top $k\%$ 样本**（正负各 $k\%$）
4. 保留的数据确保纯文本分类器无法区分正负样本，迫使图文对齐模型必须同时看图和文

**直觉**：如果纯文本模型能判断一条描述是正还是负，说明该样本包含分布偏差特征（如特定词频模式），移除这些偏差样本后，训练出的模型才真正学会"看图说话"。

**3. 微调 LLaVA-1.5**

采用简洁的 prompt 格式："Does this image I match the following caption T. Answer Yes or No directly."

对齐分数计算：

$$\text{Score} = \frac{e^{P(\text{Yes}|prompt)}}{e^{P(\text{Yes}|prompt)} + e^{P(\text{No}|prompt)}}$$

### 损失函数

标准交叉熵损失，标签为 Yes（正样本）或 No（负样本）。也可应用于 BLIP2 的 ITM 头。

训练配置：batch size=64，8×A100，1 epoch，lr=2e-6，过滤比例 $k=30\%$，分区数 $N=5$。

## 实验关键数据

### 主实验

多数据集综合比较：

| 方法 | Winoground-img | Winoground-text | Winoground-group | SeeTRUE-Draw | SeeTRUE-Edit | SugarCrepe-replace | SugarCrepe-swap | MagicBrush |
|------|------|------|------|------|------|------|------|------|
| CLIP-ViT-L-14 | 10.50 | 28.50 | 7.75 | 61.4 | 62.1 | 79.4 | 61.4 | 52.89 |
| NegCLIP | 11.75 | 30.75 | 8.25 | 63.2 | 66.0 | 85.3 | 75.3 | 61.12 |
| BLIP2-ITM | 24.25 | 41.75 | 19.00 | 60.8 | 67.5 | 88.9 | 83.9 | 75.32 |
| Image-Reward | 15.25 | 43.00 | 12.75 | 70.4 | 70.2 | 88.2 | 81.0 | 70.28 |
| VQ2 (PaLI) | 42.25 | 47.00 | 30.50 | 82.6 | 73.6 | — | — | — |
| LLaVA-1.5 (零样本) | 49.75 | 51.00 | 34.25 | 86.9 | 78.3 | 93.5 | 88.3 | 82.61 |
| **LLaVA-score** | **68.00** | **53.75** | **47.25** | **88.8** | 77.7 | **95.3** | **94.9** | **87.28** |

属性/计数/空间推理的细粒度评估：

| 方法 | 属性 avg | 计数 avg | 空间关系 avg |
|------|----------|----------|-------------|
| CLIP-ViT-L-14 | 63 | 58 | 53 |
| NegCLIP | 65 | 59 | 57 |
| BLIP2-ITM | 58 | 53 | 51 |
| Image-Reward | 70 | 61 | 57 |
| LLaVA-1.5 | 71 | 62 | 57 |
| **LLaVA-score** | **81** | **71** | **81** |

### 消融实验

数据过滤策略的重要性：

| 训练设置 | Winoground-group↑ | SugarCrepe-replace↑ | SugarCrepe-swap↑ | MagicBrush↑ |
|----------|------------|----------|----------|------------|
| LLaVA-1.5 基线 | 34.25 | 93.5 | 88.3 | 82.61 |
| 仅 replace + 过滤 | 38.50 | 95.0 | 89.4 | 84.50 |
| 仅 swap + 过滤 | 40.25 | 93.2 | 93.0 | 84.75 |
| 混合 无过滤 | 42.00 | 94.8 | 93.8 | 81.50 |
| 随机子采样 (等量数据) | 39.75 | 94.1 | 92.5 | 83.00 |
| **混合 + 过滤 (完整)** | **47.25** | **95.3** | **94.9** | **87.28** |

过滤比例 $k$ 的影响：$k$ 从 0% 增加到 90%，性能在 30%-40% 时达到峰值。滤后数据的纯文本分类准确率从 75.9% 降至约 50%（理想值），验证了偏差确实被消除。

### 关键发现

1. **Winoground group score 从 34.25 飙升至 47.25**（+37.9%），该基准以高难度著称
2. **空间关系理解从 57 提升至 81**（+42%），该项提升最为显著
3. 数据过滤是关键——无过滤时在 MagicBrush 上甚至低于基线（81.50 vs 82.61），说明偏差数据确实伤害模型
4. 替换型和交换型负描述互补——单独使用各有所长，组合效果最佳
5. LLaVA-1.5 的零样本表现已是第二好（34.25），但微调后仍有巨大提升空间
6. 分布偏差不是 GPT 特有的——其他负描述生成方法同样存在此问题

## 亮点与洞察

- **核心洞察简洁深刻**：用纯文本分类器暴露正负描述的分布偏差，这个思路可推广到任何多模态学习场景
- **方法极其简单实用**：实质上就是"训练文本分类器 + 过滤高置信样本"，无需复杂架构设计
- **超越语法层面的偏差发现**：不同于先前工作只检查语法流畅性，本文关注词频等隐式统计特征
- **实际应用明确**：可直接用于 T2I 生成图像的排序和质量评估

## 局限性

- 过滤过程需要多次交叉验证训练 BERT 分类器，增加数据准备开销
- 过滤可能丢弃部分有价值的困难样本，存在信息损失
- 主要在 COCO 数据集上构造训练数据，领域泛化性待验证
- LLaVA-1.5 模型较大，推理效率不如 CLIP 等轻量模型
- 偏差过滤的最优比例 $k$ 需要对每个数据集重新调优

## 评分

- 创新性：⭐⭐⭐⭐ — 分布偏差视角新颖
- 实用性：⭐⭐⭐⭐⭐ — 方法简单且效果显著
- 表现力：⭐⭐⭐⭐⭐ — 多数据集全面领先
- 综合评分：8.5/10

<!-- RELATED:START -->

## 相关论文

- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)
- [Latent Guard: a Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)

<!-- RELATED:END -->
