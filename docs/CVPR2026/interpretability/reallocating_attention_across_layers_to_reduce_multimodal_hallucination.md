---
title: >-
  [论文解读] Reallocating Attention Across Layers to Reduce Multimodal Hallucination
description: >-
  [CVPR 2026][多模态幻觉] 提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。
tags:
  - CVPR 2026
  - 多模态幻觉
  - 注意力头
  - 感知-推理分层
  - 训练无关插件
  - 注意力重分配
---

# Reallocating Attention Across Layers to Reduce Multimodal Hallucination

**会议**: CVPR 2026  
**arXiv**: [2510.10285](https://arxiv.org/abs/2510.10285)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态幻觉, 注意力头, 感知-推理分层, 训练无关插件, 注意力重分配

## 一句话总结

提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。

## 研究背景与动机

**多模态幻觉日益严重**：多模态大推理模型（MLRM）频繁生成与视觉证据矛盾或与自身推理链不一致的结论，严重削弱了模型的可靠性和可部署性。

**现有方法假设偏颇**：主流方法（如更强监督、细粒度对齐、外部视觉先验）默认幻觉主要源自视觉信息利用不足，忽略了模型内部感知与推理之间的分配失衡。

**解释性研究揭示分层机制**：已有工作发现 Transformer 注意力存在分阶段特性——浅层依赖视觉信号提取证据，深层转向基于文本的符号推理，提示幻觉可能来自跨层功能失调。

**两种互补的失败模式**：浅层的"感知偏差"（Perceptual Bias）导致视觉注意力分散、关键证据被稀释；深层的"推理漂移"（Reasoning Drift）导致推理链偏离中间步骤的前提。

**200例手动检查**：约16%幻觉源自感知偏差，20%源自推理漂移，10%为两者共现，表明这是一个多阶段复合问题。

**模型内部已有潜力**：模型可能已经包含具备感知或推理专长的注意力头，但它们在当前状态下并未发挥主导作用，需要被显式识别和放大。

## 方法详解

### 整体框架

方法由两步组成，作为即插即用的轻量插件：

1. **功能头识别（Functional Head Identification）**：利用注意力权重的模态分配比例和层深信息，将注意力头分类为感知头或推理头。
2. **类别条件缩放（Class-Conditioned Rescaling）**：对识别出的功能头施加乘法增益，放大其贡献以抵消感知偏差和推理漂移，同时不改变底层注意力计算机制。

整个过程不需要重新训练，不修改模型架构，仅在注意力输出投影前进行轻微缩放。

### 关键设计

**模态注意力比例（Visual Attention Ratio）**：对每个层 $\ell$ 和头 $h$，计算视觉注意力分配比例 $S_v^{(\ell)}(h) = \sum_{j \in \mathcal{T}_v} a_{i^* j}^{(h,\ell)}$，衡量该头对视觉 token 的关注程度。与文本比例互补：$S_v + S_t = 1$。

**功能头分类规则**：引入两个比例阈值 $\tau_{\text{perc}}$ 和 $\tau_{\text{reas}}$（$\tau_{\text{reas}} < \tau_{\text{perc}}$），以及层边界 $\ell_{\text{perc}}$（感知层上界）和 $\ell_{\text{reas}}$（推理层下界）：

- 感知头：$\mathcal{H}_{\text{perc}}^{(\ell)} = \{h : \ell \le \ell_{\text{perc}} \wedge S_v^{(\ell)}(h) \ge \tau_{\text{perc}}\}$
- 推理头：$\mathcal{H}_{\text{reas}}^{(\ell)} = \{h : \ell \ge \ell_{\text{reas}} \wedge S_v^{(\ell)}(h) \le \tau_{\text{reas}}\}$

**缩放策略**：为感知头和推理头分别赋予全局增益 $g_{\text{perc}} \ge 1$ 和 $g_{\text{reas}} \ge 1$，其余头保持不变（增益为1）。在 per-head 输出计算完毕后、输出投影前施加：

$$Y_{\text{out}}^{(\ell)} = \text{Concat}(g^{(1,\ell)} O^{(1,\ell)}, \ldots, g^{(H,\ell)} O^{(H,\ell)}) W_O^{(\ell)}$$

**默认超参数（Ocean-R1）**：$\ell_{\text{perc}}=7$, $\ell_{\text{reas}}=3$, $g_{\text{reas}}=1.30$, $\tau_{\text{reas}}=0.01$, $g_{\text{perc}}=1.16$, $\tau_{\text{perc}}=0.22$。

### 损失函数

本方法无需训练，不涉及优化损失函数。其理论目标是最小化幻觉强度：

$$\mathcal{I} = \lambda_1 \mathbb{E}_{\ell \in \mathcal{L}_{\text{perc}}}[\mathcal{E}_{\text{perc}}^{(\ell)}] + \lambda_2 \mathbb{E}_{\ell \in \mathcal{L}_{\text{reas}}}[\mathcal{E}_{\text{reas}}^{(\ell)}]$$

其中感知偏差 $\mathcal{E}_{\text{perc}}$ 和推理漂移 $\mathcal{E}_{\text{reas}}$ 分别衡量浅层视觉特征和深层推理表示与理想目标的L2偏差。

## 实验

在3个MLRM（Kimi-VL A3B-Thinking, Ocean-R1 7B-Instruct, R1-Onevision 7B）和5个基准上评测。

### 主实验结果（Table 1，与Vanilla和3种baseline对比）

| 方法 | MathVista | MathVision | HallusionBench Acc | MMStar Acc | SEED-Bench Acc |
|------|-----------|------------|---------------------|------------|----------------|
| Kimi-VL Vanilla | 63.48 | 56.24 | 64.76 | 59.76 | 66.26 |
| Kimi-VL + VCD | 63.51 | 56.14 | 65.68 | 59.48 | 66.52 |
| Kimi-VL + AGLA | 67.32 | 58.88 | 61.36 | 63.76 | 69.27 |
| **Kimi-VL + Ours** | **69.78** | **60.54** | **68.19** | **66.49** | **69.74** |
| Ocean-R1 Vanilla | 54.58 | 20.05 | 49.41 | 45.24 | 59.76 |
| **Ocean-R1 + Ours** | **59.32** | **26.01** | **53.64** | **50.77** | **66.51** |
| R1-OneVision Vanilla | 59.92 | 33.54 | 58.26 | 56.26 | 68.48 |
| **R1-OneVision + Ours** | **60.09** | **39.12** | **60.77** | **58.02** | **69.52** |

### 消融实验

- **去除推理缩放（w/o Reason）**：视觉任务受益更多，数学推理任务提升有限。
- **去除感知缩放（w/o Percept）**：数学推理提升更明显，但视觉任务下降。
- **非加法效应**：R1-OneVision 上 MathVision 单独增强任一类头反而下降 −3.91%，同时增强则上升 +5.58%，说明感知和推理需要协同优化。
- **模型异质性**：不同架构对感知/推理头的依赖不同，例如 Kimi-VL 在 MMStar 上仅增强感知头就能 +6.71%，而 Ocean-R1 同样设置却 −1.51%。

### 关键发现

1. **约95%任务取得最优**，平均提升 4.2%，困难任务最高提升 7%。
2. **效率极佳**：仅增加约 2 秒推理时间（约 9% 基线延迟），远低于 VCD/CGD/AGLA 的 1.2×–6.6× 开销。
3. **层边界存在任务依赖的"带状"区域**而非单一分割点，视觉任务偏好浅层边界，推理任务偏好深层边界。
4. **最优增益 $g \approx 1.14$**，过大增益会导致性能退化；推理增益 $g_{\text{reas}}$ 表现更稳定，感知增益 $g_{\text{perc}}$ 波动更大。
5. **稀疏干预最有效**：约选中 6.4% 的头（50–150个）时效果最佳，选取比例升至 18% 后性能下降。
6. **中间层（10–17层）存在过渡区**：该区间内感知和推理功能高度交织，单独增强任一方向效果不佳，佐证了分层假设。
7. **跨模型迁移性有限**：最优超参配置在不同架构间差异显著，不同MLRM对感知/推理头的依赖模式各异。

## 亮点

- **即插即用**：无需训练、无需改架构，直接作用于推理阶段的注意力头输出，实用性极强。
- **理论-实验一致性好**：从感知偏差/推理漂移的形式化分析出发，方法设计直接对应理论目标，且实验全面验证了每个组件的贡献。
- **超低开销**：相比其他推理时方法（VCD 1.2×, CGD 6.6×），本方法仅增约 1% 计算量，几乎免费。
- **解释性视角**：提供了一种从跨层功能动态角度理解和调控多模态推理可靠性的新视角。

## 局限性

- 层边界和阈值需要针对不同模型/任务调优，搜索空间较大（150+边界配置、24+缩放策略），缺乏自动化选择方法。
- 仅在推理阶段干预，无法修复训练阶段引入的系统性偏差。
- 实验仅覆盖 7B 级别模型，未验证更大规模（如 70B+）模型上的效果和可扩展性。
- $\ell_{\text{perc}}$ 和 $\ell_{\text{reas}}$ 可以重叠或留空隙，但如何自动确定最优区间仍是开放问题。
- 增益为全局常数（layer-agnostic），未探索逐层自适应增益或 sample-wise 动态增益的可能收益。
- 感知/推理头的识别依赖单一查询位置 $i^*$ 的注意力分布，聚合策略（如多位置平均）可能更鲁棒。
- 未分析在生成式任务（如 image captioning）而非判别式基准上的表现。

## 相关工作

- **对比解码方法**：VCD 通过对比原始/反事实图像视图抑制幻觉，CGD 用 CLIP 引导解码，AGLA 融合全局/局部视图增强视觉锚定。这些方法虽然有效但引入显著推理开销（1.2×–6.6×）。
- **对齐/偏好学习**：通过微调提升跨模态对齐质量，但需要额外训练数据和计算资源，不适用于黑盒部署场景。
- **多模态可解释性**：已有工作揭示注意力头的视觉专注度在浅层/中层更高（如视觉头分析），本文在此基础上将可解释性发现转化为可操作的干预方案。
- **注意力头剪枝/分析**：前人发现剪除高视觉分配头对视觉任务影响更大，本文反其道而行之，通过"放大"而非"剪枝"来利用这些功能头。
- **链式思维推理**：CoT 类方法通过提示或标注推理链来提升多模态推理，但依赖手工设计的 prompt 或重度监督，本文方法与之互补。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从感知-推理分层角度切入幻觉问题，提出功能头识别+缩放的新范式
- 实验充分度: ⭐⭐⭐⭐⭐ — 3模型×5基准×4baseline，加上大规模超参数搜索和消融
- 写作质量: ⭐⭐⭐⭐ — 理论分析清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 即插即用+几乎零开销，工程落地价值高

<!-- RELATED:START -->

## 相关论文

- [Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [LLM Circuit Analyses Are Consistent Across Training and Scale](../../AAAI2026/interpretability/llm_circuit_analyses_consistent_across_training_and_scale.md)
- [Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)

<!-- RELATED:END -->
