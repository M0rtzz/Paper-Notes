---
title: >-
  [论文解读] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage
description: >-
  [ICLR 2026][多模态][GRPO] 提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。
tags:
  - ICLR 2026
  - 多模态
  - GRPO
  - 强化学习
  - 多模态推理
  - 难度自适应
  - advantage vanishing
  - 变体增强
---

# DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage

**会议**: ICLR 2026  
**arXiv**: [2603.01106](https://arxiv.org/abs/2603.01106)  
**代码**: [Siaaaaaa1/DIVA-GRPO](https://github.com/Siaaaaaa1/DIVA-GRPO)  
**领域**: multimodal_vlm  
**关键词**: GRPO, 强化学习, 多模态推理, 难度自适应, advantage vanishing, 变体增强

## 一句话总结

提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。

## 研究背景与动机

**GRPO 在多模态推理中广泛应用**：GRPO 通过组内相对 advantage 估计实现无 critic 模型的长链推理训练，已成为增强 MLLM 推理能力的主流方法。

**Advantage vanishing 是核心瓶颈**：当问题对当前模型过于简单或过于困难时，组内所有回答全对或全错，导致 advantage 为零，优化信号消失，训练效率骤降。

**Reward sparsity 加剧问题**：在训练早期或面对困难问题时，只有极少数推理路径获得正奖励，正向反馈稀缺导致学习缓慢。

**现有方法各有局限**：(a) 样本增强扩展法（如添加 prompt、生成变体）未控制难度分布，可能加剧 advantage vanishing；(b) 选择性样本利用法丢弃部分数据，减少多样性；(c) 间接奖励设计法可能引入与最终目标不对齐的偏差。

**难度动态变化被忽视**：随着训练推进，模型能力增强，原本中等难度的问题变简单，advantage vanishing 持续恶化，但现有方法均未考虑难度的动态演变。

**核心洞察**：关键在于保证每个问题的组内奖励分布具有足够的方差，从而产生清晰的优化信号——这需要根据问题难度动态调整变体的难度分布。

## 方法详解

### 整体框架

DIVA-GRPO 由三个核心模块组成：(1) 基于历史 rollout 的动态难度评估；(2) 难度自适应的变体生成；(3) 难度加权的局部-全局 advantage 平衡与 reward-range 重缩放。训练时，先评估每个问题的难度，再根据难度采样不同类型的变体，最后在原始问题及其变体组成的扩展空间中计算 advantage 并更新策略。

### 关键设计 1：动态难度评估

- **功能**：为每个训练问题维护一个动态难度分数 $D_q \in [D_{\min}, D_{\max}]$，根据模型在该问题上的历史表现实时更新。
- **核心思路**：统计 rollout 的经验正确率 $\alpha$，通过 $D^{\text{new}} = \text{clip}(D^{\text{old}} + \eta \cdot (0.5 - \alpha))$ 更新难度——正确率高则难度降低，正确率低则难度升高，正确率约 50% 时保持稳定。
- **设计动机**：问题难度不是固有属性，而是相对于当前模型能力的动态量。通过每个 epoch 重新校准难度，确保变体生成策略始终与模型当前水平匹配，避免训练后期所有问题变简单导致的 advantage 消失。

### 关键设计 2：难度自适应变体生成

- **功能**：根据问题难度等级，生成保持答案不变但难度不同的语义一致变体。
- **核心思路**：三级策略——
    - **简单问题** ($D_q < D_{\text{mid}}$)：同时扰动文本和图像（旋转、噪声、模糊等），增大难度以产生负样本
    - **中等问题** ($D_q \approx D_{\text{mid}}$)：仅生成文本改写变体，保持难度但增加表达多样性
    - **困难问题** ($D_q > D_{\text{mid}}$)：添加部分推理步骤作为提示（think-step），降低难度以产生正样本
- **设计动机**：确保每个问题的变体组内同时包含正确和错误回答，使奖励分布具有足够方差，从根本上解决 advantage vanishing。

### 关键设计 3：难度加权的局部-全局 Advantage 平衡

- **功能**：分别计算局部（单个问题组内）和全局（问题及其所有变体组内）advantage，通过 batch z-score 归一化和难度加权缩放合并。
- **核心思路**：先对局部和全局 advantage 分别做 batch-level z-score 归一化消除量级差异，再用 $\hat{A} = \exp(k \cdot (D_q^{(i)} - \bar{D}_q) \cdot \text{sgn}(\tilde{A})) \cdot \tilde{A}$ 进行难度加权——对于高于平均难度的变体，放大正确回答的 advantage、抑制错误回答；反之亦然。
- **设计动机**：(1) 局部和全局 advantage 因样本量不同导致量级不一致（全局通常更大），归一化使两者可比；(2) 难度加权鼓励模型在困难问题上的正确答案获得更大收益，实现难度自适应优化。

### 损失函数与训练

- 基础损失为标准 GRPO 策略梯度损失，advantage 替换为上述难度加权、归一化后的值
- 额外引入 **Reward-Range-Based Advantage Rescaling (RRB)**：$\hat{A}_{\text{range}} = \Delta r_q \cdot \tilde{A}$，其中 $\Delta r_q = (\max(\mathcal{R}_q) - \min(\mathcal{R}_q)) / R_{\max}$，防止奖励高度集中时 z-score 归一化放大微小差异
- 基座模型 Qwen2.5-VL-7B-Instruct，AdamW 优化器，学习率 $10^{-6}$，难度初始化 $D_q=5$（范围 1-9），$\eta=4$
- 文本变体和推理提示由 GPT-o3 离线生成，图像扰动在线施加

## 实验关键数据

### 表1：六个多模态数学推理基准上的主实验结果

| 模型 | MathVista | MathVerse | MathVision | OlympiadBench | WeMath | MMK12test | Avg. |
|---|---|---|---|---|---|---|---|
| GPT-4o | 63.8 | 50.2 | 30.4 | 35.0 | 68.8 | 49.9 | 49.68 |
| Qwen2.5-VL-7B (base) | 68.2 | 47.9 | 25.4 | 20.2 | 62.1 | 53.6 | 46.23 |
| Qwen2.5-VL-72B | 74.8 | 57.6 | 38.1 | 40.4 | 72.4 | 70.5 | 59.0 |
| R1-ShareVL-7B | 73.5 | 52.8 | 29.5 | 21.3 | 67.9 | 68.8 | 52.30 |
| MM-Eureka-7B | 71.7 | 50.3 | 26.9 | 20.1 | 66.1 | 64.5 | 49.93 |
| **DIVA-GRPO-7B (Ours)** | **74.2** | **57.6** | **32.1** | **23.1** | **69.3** | **70.2** | **54.58** |

- 7B 规模下六个基准均达 SOTA，平均 54.58 分
- 在 MathVista/MathVerse/WeMath 上已接近 72B 级别模型
- 相比基座 Qwen2.5-VL-7B 平均提升 **+8.35** 分

### 表2：消融实验结果

| 方法 | MathVista | MathVerse | MMK12test | Avg. |
|---|---|---|---|---|
| w/o Variant Generation | 70.0 | 53.7 | 61.1 | 61.6 |
| w/o Difficulty-Weighting | 69.9 | 55.7 | 66.5 | 64.0 |
| w/o RRB-Rescaling | 71.5 | 55.2 | 64.7 | 63.8 |
| w/o G-L Balance | 70.8 | 55.4 | 66.0 | 64.1 |
| **Full DIVA-GRPO** | **73.2** | **56.3** | **68.8** | **66.1** |

- 移除任一组件均导致性能下降，变体生成的影响最大（-4.5 avg）
- 训练效率方面：达到最优性能所需步数减少 **2.55×**，端到端加速 **1.76×**

## 亮点

- **问题定义精准**：从"如何保证组内奖励方差充足"的角度统一理解 advantage vanishing，提供了比现有三类方法更本质的解决思路
- **难度自适应闭环**：难度评估→变体生成→advantage 加权形成完整闭环，且难度随训练动态演化
- **理论支撑充分**：提供了梯度方差降低加速收敛的定理证明，以及正负样本比约 1:1 时优化信号最强的数学分析
- **训练效率显著提升**：2.55× 步数减少 + 1.76× 端到端加速，实用价值高
- **RRB-Rescaling 通用性强**：可独立于 DIVA-GRPO 应用到任何 GRPO 框架

## 局限与展望

- 变体的文本推理提示依赖 GPT-o3 离线生成，引入了对闭源模型的依赖和额外成本
- 在竞赛级数学任务（OlympiadBench 23.1 vs o1 的 68.0）上仍有很大差距，7B 模型容量限制明显
- 图像扰动方式（旋转、噪声等）相对简单，对需要精细视觉理解的场景可能不够
- 难度评估基于正确率，对于部分正确或推理过程正确但最终答案错误的情况缺乏区分

## 与相关工作的对比

- **vs GRPO/DAPO**：标准 GRPO 和 DAPO 未考虑难度自适应，在训练后期 advantage 信号衰减；DIVA-GRPO 通过变体生成维持奖励方差
- **vs GSPO**：GSPO 引入语义一致变体但未动态调整难度分布；DIVA-GRPO 根据模型当前能力动态匹配变体难度
- **vs Adora/MM-Eureka**：这些方法通过样本选择或间接奖励缓解问题，但分别存在数据浪费和优化方向偏差的风险
- **vs R1-ShareVL**：同为 7B 规模 SOTA 对手，DIVA-GRPO 在 MathVerse (+4.8) 和 MMK12test (+1.4) 上优势明显

## 评分

- 新颖性: ⭐⭐⭐⭐ — 难度自适应变体生成+三级策略+RRB rescaling 组合新颖
- 实验充分度: ⭐⭐⭐⭐ — 六个基准+详细消融+效率分析+理论证明，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 问题阐述清晰，方法动机层层递进
- 价值: ⭐⭐⭐⭐ — 解决 GRPO 训练的实际痛点，RRB 组件可即插即用

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)

<!-- RELATED:END -->
