---
title: >-
  [论文解读] StickMotion: Generating 3D Human Motions by Drawing a Stickman
description: >-
  [CVPR 2025][人体理解][动作生成] 提出 StickMotion 框架，通过用户手绘的火柴人图作为细粒度动作控制条件，结合文本描述实现全局+局部的 3D 人体动作生成，并设计多条件模块（MCM）高效处理条件组合，节省用户 51.5% 的动作创意表达时间。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "动作生成"
  - "火柴人条件"
  - "多条件扩散"
  - "动态监督"
  - "文本到动作"
---

# StickMotion: Generating 3D Human Motions by Drawing a Stickman

**会议**: CVPR 2025  
**arXiv**: [2503.04829](https://arxiv.org/abs/2503.04829)  
**代码**: 即将发布  
**领域**: 人体理解/动作生成  
**关键词**: 动作生成, 火柴人条件, 多条件扩散, 动态监督, 文本到动作

## 一句话总结

提出 StickMotion 框架，通过用户手绘的火柴人图作为细粒度动作控制条件，结合文本描述实现全局+局部的 3D 人体动作生成，并设计多条件模块（MCM）高效处理条件组合，节省用户 51.5% 的动作创意表达时间。

## 研究背景与动机

文本到动作生成虽然取得显著进展，但简单的文本描述（如"向前高踢"）难以精确传达用户想象中的复杂四肢姿态。现有方法试图通过更详细的文本描述来控制细节（如 FineMoGen 按身体部位描述），但这要求用户撰写冗长精确的文字，增加了使用门槛。

核心问题：
- **文本描述精度不足**：自然语言难以精确描述 3D 空间中的肢体位置和角度
- **多条件融合效率低**：传统 self-attention 方法处理多条件组合时引入不必要的计算和性能下降
- **固定帧索引不自然**：直接将火柴人姿态硬绑定到指定帧会破坏动作序列的自然流畅性

火柴人图作为控制条件的优势：用户画一个简单的火柴人比写一段精确的文本描述快得多且更直观，同时提供了文本无法表达的精确肢体位置信息。

## 方法详解

### 整体框架

StickMotion 基于扩散模型，接受文本描述和最多 3 个火柴人图（分别位于序列开头、中间、结尾附近）作为条件。通过多条件模块（MCM）高效处理 4 种条件组合 $(text), (text, stick), (stick), ()$，输出预测噪声和火柴人帧索引评分。

### 关键设计一：火柴人生成算法（SGA）

- **功能**：从现有数据集的 3D 关节坐标自动生成模拟手绘风格的火柴人训练数据
- **核心思路**：考虑手绘特征：(1) 笔画平滑度（模拟不同设备的抖动）；(2) 落笔偏差（全局位置偏移）；(3) 缩放不一致（不同身体部位的比例偏差）。从正面视角观察姿态，要求用户画 6 条单笔线段（头、躯干、四肢），用 Transformer encoder 编码
- **设计动机**：手绘火柴人数据收集耗时且受标注者风格影响大；SGA 可自动从任意动作数据集生成多样化训练数据

### 关键设计二：多条件模块（MCM）

- **功能**：高效处理文本和火柴人的所有可能条件组合
- **核心思路**：在 batch 维度将数据分为 4 组 $(B_1, B_2, B_3, B_4)$ 对应 4 种条件组合。Condition Fusion 中使用两个 Feat Decoder 分别处理文本条件和火柴人条件，通过在 batch 维度选择性应用实现所有组合的输出，无需注意力掩码。Latent Encoder 进一步融合信息
- **设计动机**：传统方法用 self-attention 加掩码处理多条件组合，会计算 masked token 的无用注意力，且不同表示空间（火柴人 vs 文本）的 attention 相互干扰导致性能下降

### 关键设计三：动态监督策略（Dynamic Supervision）

- **功能**：允许网络在指定位置附近自动调整火柴人对应帧的精确索引
- **核心思路**：用户只需指定火柴人的大致位置（开头/中间/结尾），网络输出每帧的索引评分 $\hat{I}_l$。训练时在各位置范围（如中间位置在 $[3L/8, 5L/8]$）内随机采样帧作为 GT 火柴人，用 softmax 加权的 index loss 监督：$\mathcal{L}_{index} = M \cdot \sum_l softmax(\hat{I}_l) \cdot \|\hat{x}_l - x_i\|^2$
- **设计动机**：固定帧索引会导致动作不自然（如某帧突然过渡到指定姿态）。允许网络在附近帧中选择最自然的位置插入火柴人姿态

### 损失函数

$\mathcal{L}_{total} = \mathcal{L}_{index}^{start} + \mathcal{L}_{index}^{middle} + \mathcal{L}_{index}^{end} + \mathcal{L}_{motion}$，其中 $\mathcal{L}_{motion}$ 为标准扩散噪声预测损失。推理时使用 classifier-free guidance 控制文本/火柴人条件的偏好权重。

## 实验关键数据

### 主实验：HumanML3D 测试集

| 方法 | R Precision Top3 ↑ | FID ↓ | MM Dist ↓ | Diversity ↑ |
|------|-------------------|-------|----------|------------|
| Real motions | 0.797 | 0.002 | 2.974 | 9.503 |
| MDM | 0.611 | 0.544 | 5.566 | 9.559 |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 |
| **StickMotion** | **~0.78** | **~0.3** | **~3.1** | ~9.5 |

### 用户研究

| 指标 | StickMotion 优势 |
|------|----------------|
| 与想象一致性 | 比纯文本方法显著提升 |
| 时间节省 | 比文本描述节省 **51.5%** 时间 |
| 交互满意度 | 高于纯文本方法 |

### 消融实验

| 模块 | FID | R-Precision |
|------|-----|-------------|
| Self-Attention 基线 | 较高 | 较低 |
| MCM (本文) | **更低** | **更高** |
| 无动态监督 (固定帧) | 不自然 | - |
| 有动态监督 | **自然** | - |

### 关键发现

- StickMotion 在 text-to-motion 指标上与 SOTA 方法可比，同时额外提供了精确的姿态控制
- MCM 比 self-attention 基线减少了计算复杂度且提升性能
- 动态监督显著改善了火柴人条件下生成动作的自然度
- 用户研究表明画火柴人比写详细文本节省 51.5% 时间，生成结果更符合用户想象

## 亮点与洞察

1. **火柴人作为控制条件的直觉性**：画一个简笔画远比用文字描述"右手抬到平肩高度、左脚向前伸..."更直观高效
2. **MCM 的批处理设计**：利用 batch 维度分组巧妙实现多条件组合，避免了 attention mask 的计算浪费
3. **动态监督是关键**：允许网络小幅调整火柴人位置，是保证自然性和用户友好性的平衡点

## 局限与展望

- 火柴人表达的信息有限（无手指细节、无面部表情）
- 当前仅支持 3 个火柴人位置（开头/中间/结尾），更密集的关键帧控制有待探索
- 从 2D 火柴人推断 3D 姿态存在歧义（相同的 2D 投影可能对应不同的 3D 姿态）
- 未探索交互式实时生成场景

## 相关工作与启发

- **MDM, MLD**：基于扩散模型的文本到动作生成基线
- **FineMoGen**：通过详细文本控制身体部位
- **Flame**：允许追加文本修改动作序列
- 火柴人条件的思路可推广到其他需要精确空间控制的生成任务（如手势生成、舞蹈编排）

## 评分

⭐⭐⭐⭐ — 火柴人条件的设计直觉且实用，用户研究验证了其实际价值。MCM 和动态监督都是解决实际问题的好设计。51.5% 的时间节省是有说服力的实用指标。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2025\] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[ICLR 2026\] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](../../ICLR2026/human_understanding/quamo_quaternion_motion_kinematics.md)

</div>

<!-- RELATED:END -->
