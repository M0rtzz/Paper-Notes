---
title: >-
  [论文解读] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening
description: >-
  [CVPR 2026][目标检测][生成式帧插值] 提出 KAB（Keyframe-Anchored Attention Bias）和 ReTRo（Rescaled Temporal RoPE）两个无需训练的推理时方法，基于 Wan2.1 视频扩散模型解决稀疏关键帧下大运动生成式帧插值（GI）中的语义不忠、帧不一致和节奏不稳问题，并构建首个文本条件 GI 评估基准 TGI-Bench。
tags:
  - CVPR 2026
  - 目标检测
  - 生成式帧插值
  - 注意力锚定
  - 时序RoPE缩放
  - 关键帧引导
  - 视频扩散模型
---

# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening

**会议**: CVPR 2026  
**arXiv**: [2603.17651](https://arxiv.org/abs/2603.17651)  
**代码**: 待确认  
**领域**: 视频生成  
**关键词**: 生成式帧插值, 注意力锚定, 时序RoPE缩放, 关键帧引导, 视频扩散模型

## 一句话总结

提出 KAB（Keyframe-Anchored Attention Bias）和 ReTRo（Rescaled Temporal RoPE）两个无需训练的推理时方法，基于 Wan2.1 视频扩散模型解决稀疏关键帧下大运动生成式帧插值（GI）中的语义不忠、帧不一致和节奏不稳问题，并构建首个文本条件 GI 评估基准 TGI-Bench。

## 研究背景与动机

生成式帧插值（Generative Inbetweening, GI）是指给定首尾两个关键帧，生成中间过渡帧序列。与传统光流插帧不同，GI 需要"想象"中间过程，在大运动、长时序场景下面临三大核心挑战：

**语义不忠（Semantic Infidelity）**：中间帧出现与关键帧不一致的物体或场景元素

**帧间不一致（Frame Inconsistency）**：相邻帧之间出现闪烁、突变

**节奏不稳（Temporal Rhythm Instability）**：运动速度不均匀，时序分布不自然

现有方法大多基于 Image-to-Video（I2V）模型改造，典型如 TRF 和 SEINE。但当关键帧间距增大（如 65、81 帧），这些方法的质量急剧下降。根本原因在于：

- Cross-attention 机制对两端关键帧的关注度在长序列中稀释
- Temporal attention 的位置编码未考虑首尾帧的锚定需求
- 缺乏统一的评估基准来衡量文本条件 GI 的质量

本文的出发点是：**不修改模型权重**，仅通过推理时的注意力操控来解决上述问题。

## 方法详解

### 整体框架

基于 **Wan2.1**（DiT-based First-Last-Frame-to-Video 模型），在推理阶段引入两个互补模块：

- **KAB**：操控 cross-attention 的 logit 分布，将关键帧的语义锚点注入中间帧
- **ReTRo**：调整 temporal self-attention 中 RoPE 的缩放系数，差异化处理边缘帧与中间帧

两者均**不需要额外训练**，直接在去噪过程中介入。

### 关键设计

#### 1. KAB（Keyframe-Anchored Attention Bias）

核心思想：从关键帧的 cross-attention map 中提取**语义锚点**，通过 logit bias 引导中间帧的注意力分布。

**Step 1: 提取关键帧锚点**

对首帧 $I_{\text{first}}$ 和尾帧 $I_{\text{last}}$，在 cross-attention 层中获取它们的注意力分布 $A_{\text{first}}$ 和 $A_{\text{last}}$，作为 keyframe anchors。

**Step 2: 线性插值生成逐帧 target anchors**

对第 $t$ 帧，按时间位置线性插值：

$$\bar{A}(t) = \frac{T - t}{T} \cdot A_{\text{first}} + \frac{t}{T} \cdot A_{\text{last}}$$

**Step 3: 计算并施加 logit bias**

定义注意力偏置：

$$B(t) = \log(M(t) + \varepsilon) - \log(\bar{A}(t) + \varepsilon)$$

其中 $M(t)$ 是期望的 target mask，$\varepsilon$ 防止数值溢出。该 bias 被加到 cross-attention 的 logit 上，在 softmax 之前生效，从而**不改变模型参数**地引导注意力聚焦。

**Triple Isolated Cross-Attention**：

为避免首帧、尾帧和文本条件之间的信息干扰，将三者的 cross-attention 完全隔离：

- $I_{\text{first}}$ 的 cross-attention 独立计算
- $I_{\text{last}}$ 的 cross-attention 独立计算
- 文本 prompt 的 cross-attention 独立计算

三路结果加权融合，确保对称处理两端关键帧。

#### 2. ReTRo（Rescaled Temporal RoPE）

Temporal self-attention 中的 RoPE 位置编码控制了帧间的注意力衰减模式。ReTRo 对不同位置的帧使用不同的缩放系数：

- **边缘帧**（靠近首/尾关键帧）：使用 $s_{\text{edge}} > 1$
    - 放大位置编码频率 → 锐化局部注意力 → 更好地保留关键帧细节
    - 直觉：靠近关键帧的帧应该"更像"关键帧

- **中间帧**：使用 $s_{\text{mid}} < 1$
    - 缩小位置编码频率 → 扩展感受野范围 → 促进帧间一致性
    - 直觉：远离关键帧的帧需要"看得更远"来保持连贯

这种非均匀缩放在时间轴上形成一个"U 形"分布：两端紧、中间松，巧妙平衡了关键帧保真度与中间过渡流畅性。

### 损失函数 / 训练策略

本方法**完全无需训练**（training-free），所有操作在推理时完成：

- KAB 仅修改 cross-attention 的 logit（加 bias）
- ReTRo 仅修改 RoPE 的缩放系数
- 不引入额外参数，不需要反向传播
- 计算开销：仅增加关键帧 anchor 提取和 bias 计算，相对总推理时间可忽略

## 实验关键数据

### TGI-Bench（新基准）

首个文本条件生成式帧插值评估基准：

| 维度 | 规模 |
|------|------|
| 视频数量 | 220 |
| 序列长度 | 25 / 33 / 65 / 81 帧 |
| 挑战类别 | 4 类（大运动/遮挡/外观变化/场景切换） |
| 评估指标 | PSNR, SSIM, FVD, VBench |

### 主实验

**长序列（65/81 帧）性能对比**：

| 方法 | 训练需求 | PSNR↑ | SSIM↑ | FVD↓ | VBench↑ |
|------|---------|-------|-------|------|---------|
| TRF | 需要 | 中 | 中 | 中 | 中 |
| SEINE | 需要 | 中 | 中 | 中 | 中 |
| Wan2.1 (baseline) | - | 中 | 中 | 中 | 中 |
| **KAB + ReTRo** | **不需要** | **最优** | **最优** | **最优** | **最优** |

关键观察：在短序列（25 帧）上各方法差距不大，但随着序列增长到 65/81 帧，KAB+ReTRo 的优势显著放大。

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| Baseline (Wan2.1) | 基线 | 基线 | 无干预 |
| + KAB only | ↑ | ↑ | 语义一致性提升 |
| + ReTRo only | ↑ | ↑ | 时序稳定性提升 |
| + KAB + ReTRo | ↑↑ | ↑↑ | 两者互补，最优 |
| KAB w/o Triple Isolation | ↓ | ↓ | 首尾帧干扰导致退化 |
| ReTRo 均匀缩放 (s=1) | → 基线 | → 基线 | 等于不做缩放 |
| $s_{\text{edge}}$ 过大 | ↓ | ↑ | 过度锐化，失去流畅性 |
| $s_{\text{mid}}$ 过小 | ↓ | ↓ | 感受野过大，细节模糊 |

### 关键发现

1. **KAB 和 ReTRo 解决不同问题**：KAB 主攻语义忠实度，ReTRo 主攻时序一致性，组合效果最佳
2. **长序列优势明显**：序列越长（65/81帧），方法增益越大，说明针对的确实是长程依赖问题
3. **Triple Isolation 不可或缺**：不隔离首尾帧 attention 会导致信息串扰，中间帧偏向一端
4. **ReTRo 的 U 形分布至关重要**：均匀缩放无效，必须边缘紧中间松

## 亮点与洞察

- **Training-free** 的设计极具实用性：无需收集配对数据、无需微调，即插即用
- KAB 的 logit bias 思路与 Classifier-Free Guidance 异曲同工，但在空间维度（attention map）而非类别维度操作
- ReTRo 对 RoPE 缩放的非均匀设计思路新颖，可推广到其他需要差异化时序建模的任务
- Triple Isolated Cross-Attention 的对称设计体现了对首尾帧公平性的细致考量
- TGI-Bench 填补了文本条件 GI 评估的空白，4 类挑战场景×4 种长度的设计科学全面
- 方法的可解释性强：每个组件的物理含义清晰，消融实验验证了各部分的独立贡献

## 局限与展望

1. **依赖 Wan2.1 架构**：KAB 和 ReTRo 的设计与 DiT + RoPE 紧密耦合，迁移到 U-Net 架构需适配
2. **线性插值假设**：target anchor 的线性插值假设运动均匀，对非线性运动（加速/减速）可能不理想
3. **超参数敏感性**：$s_{\text{edge}}$ 和 $s_{\text{mid}}$ 需要手动调整，缺乏自适应选择机制
4. **计算成本未详细分析**：虽然声称开销可忽略，但未给出具体的推理时间对比数据
5. **仅限帧插值**：方法针对首尾帧已知的场景，无法直接扩展到单帧外推或无条件生成
6. **评估指标局限**：PSNR/SSIM 侧重像素级，对感知质量的评估有限；VBench 覆盖面更广但不够细粒度

## 相关工作与启发

- **vs Wan2.1 (baseline)**：本文直接基于 Wan2.1 FLF2V，不改权重只操控注意力，可视为其推理增强插件
- **vs TRF / SEINE**：先前帧插值方法需要训练且长序列退化严重；KAB+ReTRo 无需训练且长序列优势更大
- **vs Classifier-Free Guidance**：CFG 在类别维度操控生成方向；KAB 在空间维度（attention map）操控语义聚焦，是注意力层面的类比
- **RoPE 的非均匀缩放思路**可推广到其他需要差异化时序建模的任务（如长视频理解中的关键帧增强）
- 推理时注意力操控是一种低成本但高效的模型能力提升手段，值得在视频编辑、视频补全等更多任务中探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ KAB + ReTRo 组合新颖，training-free 设计思路独特
- 实验充分度: ⭐⭐⭐⭐⭐ TGI-Bench 新基准 + 4种长度×4类挑战的全面评测
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 无需训练即插即用，视频生成社区直接受益

<!-- RELATED:START -->

## 相关论文

- [Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [Long-Context Generalization with Sparse Attention](../../ICLR2026/object_detection/long-context_generalization_with_sparse_attention.md)
- [Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](../../ICLR2026/object_detection/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/object_detection/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)

<!-- RELATED:END -->
