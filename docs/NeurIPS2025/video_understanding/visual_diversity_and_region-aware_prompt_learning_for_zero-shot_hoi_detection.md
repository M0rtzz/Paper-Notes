---
title: >-
  [论文解读] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection
description: >-
  [NeurIPS 2025][视频理解][HOI检测] 提出 VDRP 框架，通过视觉多样性感知的 prompt 学习（注入组级方差 + 高斯扰动）和区域感知的 prompt 增强（基于 LLM 生成的区域概念检索），解决零样本 HOI 检测中类内视觉多样性和类间视觉纠缠两大挑战。
tags:
  - NeurIPS 2025
  - 视频理解
  - HOI检测
  - 零样本学习
  - 提示学习
  - CLIP
  - 视觉多样性
---

# VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection

**会议**: NeurIPS 2025  
**arXiv**: [2510.25094](https://arxiv.org/abs/2510.25094)  
**代码**: https://github.com/mlvlab/VDRP (有)  
**领域**: 视频/图像理解  
**关键词**: HOI检测, 零样本学习, Prompt Learning, CLIP, 视觉多样性

## 一句话总结
提出 VDRP 框架，通过视觉多样性感知的 prompt 学习（注入组级方差 + 高斯扰动）和区域感知的 prompt 增强（基于 LLM 生成的区域概念检索），解决零样本 HOI 检测中类内视觉多样性和类间视觉纠缠两大挑战。

## 研究背景与动机
人-物交互（HOI）检测需要定位人和物体并识别它们之间的交互。零样本 HOI 检测要求模型泛化到训练时未见过的动词-物体组合，这带来了两个核心视觉挑战：

**类内视觉多样性**：同一个动词（如"拿着棒球手套"）在不同姿势、视角和场景下视觉差异巨大。作者量化发现动词类的多样性得分（0.364±0.060）显著高于物体类（0.274±0.048），说明单一静态 prompt 无法覆盖动词的视觉变化。

**类间视觉纠缠**：语义不同的动词（如"吃"、"舔"、"坐在旁边"）在全局/联合区域特征下产生高度相似的视觉模式，t-SNE 可视化显示不同动词类出现大量重叠。

**现有方法不足**：大多数 CLIP prompt 方法（GEN-VLKT、ADA-CM）每个动词仅用一个静态 prompt；CMMP 加入空间线索但文本 prompt 仍不感知区域；EZ-HOI 用 LLM 描述但忽略类内变化。

**本文核心 idea**：在 prompt 嵌入中同时编码视觉变化统计量（方差注入 + 扰动）和区域特定语义（概念检索 + 增强），两者互补地解决上述两个挑战。

## 方法详解

### 整体框架
采用两阶段 HOI 检测 pipeline：(1) 冻结的 DETR 检测器定位人和物体；(2) 基于 CLIP 的交互分类，通过轻量级 adapter 提取人 ($\mathbf{x}_h$)、物体 ($\mathbf{x}_o$)、联合区域 ($\mathbf{x}_u$) 特征。关键创新在文本 prompt 端：先生成视觉多样性感知 prompt，再用区域概念增强为区域感知 prompt，最终用三个区域的 logit 平均做动词分类。

### 关键设计

1. **视觉多样性感知 Prompt 学习**：

    - 从训练集提取每个动词 $v$ 的联合区域 CLS 特征，计算方差 $\boldsymbol{\sigma}_v^2$
    - 按 CLIP 文本嵌入余弦相似度构建语义相近动词组 $\mathcal{G}(v)$，计算组级方差 $\bar{\boldsymbol{\sigma}}_v^2 = \frac{1}{|\mathcal{G}(v)|}\sum_{v' \in \mathcal{G}(v)} \boldsymbol{\sigma}_{v'}^2$（稳定化估计，对罕见动词尤为重要）
    - 用 MLP 将组级方差转为调制向量 $\mathbf{d}_v$，注入共享上下文嵌入：$\hat{\mathbf{E}}_v = \mathbf{E} + \mathbf{d}_v \alpha$
    - 经 CLIP 文本编码器后，再施加方差引导的高斯扰动：$\tilde{\mathbf{t}}^v = \mathbf{t}^v + (\epsilon \odot \tilde{\boldsymbol{\sigma}}_v)\beta$

2. **区域感知 Prompt 增强**：

    - 用 LLM（LLaMA-7B / GPT-4）为每个动词的每个区域（人/物体/联合）生成 $K$ 个视觉概念描述
    - 用 CLIP 文本编码器编码为概念池 $\mathcal{C}_{(\cdot)}^v$
    - 给定区域特征 $\mathbf{x}_{(\cdot)}$，计算与概念的余弦相似度，通过 Sparsemax（而非 Softmax）产生稀疏权重，仅保留最相关概念
    - 加权聚合得区域概念向量 $\bar{\mathbf{c}}_{(\cdot)}^v$，增强至多样性 prompt：$\hat{\mathbf{t}}_{(\cdot)}^v = \mathbf{t}^v + \bar{\mathbf{c}}_{(\cdot)}^v \gamma$

3. **空间增强的联合区域特征**：通过 SpatialHead 融合联合区域特征与人、物体特征及其边界框，引入空间先验。

### 损失函数 / 训练策略
- Focal Loss 用于多标签动词分类
- 轻量级 adapter 插入 CLIP 视觉编码器的多个 Transformer 块中，仅训练 4.50M 参数

## 实验关键数据

### 主实验

| 零样本设置 | 指标 | 本文(VDRP) | 之前SOTA(EZ-HOI) | 提升 |
|-----------|------|-----------|-----------------|------|
| NF-UC | HM / Unseen | 33.85 / 36.45 | 31.76 / 33.66 | +2.09 / +2.79 |
| RF-UC | HM / Unseen | 32.77 / 31.29 | 31.18 / 29.02 | +1.59 / +2.27 |
| UO | HM / Unseen | 34.41 / 36.13 | 32.14 / 33.28 | +2.27 / +2.85 |
| UV | HM / Unseen | 29.80 / 26.69 | 29.09 / 25.10 | +0.71 / +1.59 |

### 消融实验

| 配置 | NF-UC Unseen | RF-UC Unseen | UO Unseen | UV Unseen |
|------|-------------|-------------|-----------|-----------|
| BASE | 28.32 | 25.64 | 28.60 | 22.41 |
| + VDP（多样性prompt） | 32.19 | 29.16 | 33.29 | 23.78 |
| + RAP（区域prompt） | 34.93 | 26.46 | 33.90 | 24.53 |
| + VDRP（完整） | 36.45 | 31.29 | 36.13 | 26.69 |

### 关键发现
- VDP 和 RAP 各自都有显著提升，两者结合效果最佳，说明类内多样性和类间判别性是互补的两个维度
- 在 NF-UC 设置下，VDRP 对 Unseen 类提升达 +8.13（从28.32到36.45），远超各单独模块
- 仅需 4.50M 可训练参数，远少于 CLIP4HOI (56.7M) 和 HOICLIP (66.18M)

## 亮点与洞察
- **方差即信息**：将类内视觉方差从"噪声"变为"信号"，注入 prompt 中指导学习，是一个优雅的设计思路
- **Sparsemax for 概念检索**：相比 Softmax，Sparsemax 能给不相关概念精确的零权重，避免噪声干扰
- **定量分析驱动设计**：先通过 diversity score 和 t-SNE 定量分析问题，再针对性设计方案，方法论值得学习

## 局限与展望
- 区域概念依赖 LLM 生成，概念质量受 LLM 能力限制
- 组级方差的"相近动词组"定义依赖 CLIP 文本嵌入相似度，可能引入偏差
- 仅在 HICO-DET 上评估，缺乏 V-COCO 等其他 HOI 基准验证
- 扰动强度 $\alpha, \beta, \gamma$ 等超参需要仔细调节

## 相关工作与启发
- **vs EZ-HOI**: EZ-HOI 用 LLM 描述区分动词语义差异但忽略类内变化，VDRP 同时处理两个维度
- **vs CMMP**: CMMP 加入空间线索但文本 prompt 不感知区域，VDRP 的区域概念检索更细粒度
- **vs CoOp/CoCoOp**: 将 prompt learning 从分类任务扩展到了 HOI 检测的多区域场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 方差注入 prompt 和区域概念检索的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 四种零样本设置全覆盖，消融充分，但缺少跨数据集验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析定量清晰，方法图示直观
- 价值: ⭐⭐⭐⭐ 对零样本 HOI 检测有实际推动，方差注入思路可推广到其他视觉任务

<!-- RELATED:START -->

## 相关论文

- [RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)
- [Locality-Aware Zero-Shot Human-Object Interaction Detection](../../CVPR2025/video_understanding/locality-aware_zero-shot_human-object_interaction_detection.md)
- [INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning](inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)
- [SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)

<!-- RELATED:END -->
