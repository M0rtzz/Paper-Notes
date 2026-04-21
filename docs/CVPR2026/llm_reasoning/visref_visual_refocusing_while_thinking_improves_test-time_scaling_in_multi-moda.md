---
title: >-
  [论文解读] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models
description: >-
  [CVPR 2026][LLM推理][视觉重聚焦] 本文提出 VisRef，一种免训练的视觉重聚焦框架，通过在多模态大推理模型（MLRM）的推理过程中使用行列式点过程（DPP）动态选择并重新注入与当前推理上下文语义相关且多样化的视觉 token，解决了长链推理中视觉注意力逐渐衰减的问题，在 MathVista 等基准上提升高达 6.4%。
tags:
  - CVPR 2026
  - LLM推理
  - 视觉重聚焦
  - 测试时缩放
  - 多模态推理
  - 行列式点过程
  - 免训练
---

# VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models

**会议**: CVPR 2026  
**arXiv**: [2603.00207](https://arxiv.org/abs/2603.00207)  
**代码**: 无  
**领域**: LLM推理 / 多模态VLM  
**关键词**: 视觉重聚焦、测试时缩放、多模态推理、行列式点过程、免训练

## 一句话总结

本文提出 VisRef，一种免训练的视觉重聚焦框架，通过在多模态大推理模型（MLRM）的推理过程中使用行列式点过程（DPP）动态选择并重新注入与当前推理上下文语义相关且多样化的视觉 token，解决了长链推理中视觉注意力逐渐衰减的问题，在 MathVista 等基准上提升高达 6.4%。

## 研究背景与动机

1. **领域现状**：多模态大推理模型（MLRMs）如 InternVL、Qwen-VL 等通过生成 chain-of-thought 推理链在视觉推理任务上取得了出色表现。测试时缩放（test-time scaling）通过增加推理计算来提升性能。

2. **现有痛点**：随着推理链变长，视觉 token 在不断扩展的上下文窗口中被逐渐稀释，模型的注意力从图像内容转向文本先验，导致视觉信息丢失。现有的文本自反思方法只延长了文本推理但无法维持视觉知觉。

3. **核心矛盾**：基于 RL 微调的视觉重聚焦方法（如 Look-Back）虽然效果好，但需要大量训练数据和计算资源。而文本类的测试时缩放方法对视觉需求高的任务反而可能损害性能。

4. **本文目标**：能否在不做任何微调的情况下，纯粹在测试时恢复视觉 grounding？核心是两个问题——选哪些视觉 token 重新注入，以及何时停止推理。

5. **切入角度**：模仿人类解题时"看一眼图→推理→再看一眼图"的交替行为，在每个推理步骤自适应注入一组精心选取的视觉 token 核心子集。

6. **核心 idea**：用 DPP 选取与当前推理状态相关且视觉覆盖多样的 token 子集来重注入，确保推理全程保持视觉 grounding。

## 方法详解

### 整体框架

VisRef 在 MLRM 的推理过程中工作：给定图文输入 $x_{\text{input}} = [I, T]$，模型生成推理步骤 $z_k$，VisRef 在每一步选择一个视觉 token 子集 $V_k \subseteq \mathcal{V}$ 并重新注入到上下文中，形成视觉增强的推理轨迹 $\tau_{1:k} = \{(z_1, V_1), ..., (z_k, V_k)\}$。当基于熵的停止准则满足时，生成最终答案。

### 关键设计

1. **DPP 视觉 token 选择（Determinantal Point Process-based Selection）**

    - 功能：在每个推理步骤选择与推理上下文相关且相互多样化的视觉 token 子集
    - 核心思路：首先计算当前推理步骤的文本 token 嵌入的二阶矩矩阵 $M_k = \sum_{i} z_k^{(i)} (z_k^{(i)})^\top$ 来捕捉推理子空间几何。然后定义核函数 $L_k(v_i, v_j) = v_i^\top M_k v_j$，将视觉 token 投影到文本推理子空间中计算相似性。DPP 的行列式 $\det(L_k^{V_k})$ 自然平衡了相关性（对角线项高 = token 与推理状态对齐）和多样性（非对角线项低 = 视觉覆盖广）。通过 $\log\det$ 的分解，目标明确为相关性项 $\sum \log(r_i^2)$ 加多样性项 $\log\det(\bar{L}_k^{V_k})$
    - 设计动机：朴素方案——重注入全部视觉 token——计算代价高昂（推理延迟增加 2.3x）且引入冗余。DPP 提供了一个理论上优雅且实际可行的子集选择框架，同时考虑相关性和多样性

2. **贪心近似求解**

    - 功能：高效求解 NP-hard 的 DPP 子集选择问题
    - 核心思路：从空集出发，每次选择能带来最大边际增益的 token：$v_{k,i} = \arg\max_{v} \log(\det(L_k^{V_k^{(i-1)} \cup \{v\}}) / \det(L_k^{V_k^{(i-1)}}))$，迭代 $m$ 次得到大小为 $m$ 的子集。设置 token 预算 $m = \lfloor 0.3|\mathcal{V}|\rfloor$（即选 30% 的视觉 token）
    - 设计动机：贪心算法在 DPP 优化中有经典的 $(1-1/e)$ 近似保证，实际效果与精确解接近

3. **基于熵的自适应停止准则**

    - 功能：判断何时终止推理并输出最终答案
    - 核心思路：在每个推理步骤计算模型回答分布的熵 $H_k = -\mathbb{E}[\log \pi_\theta(y | x_{\text{input}}, \tau_{1:k})]$，当 $H_k < \delta_{\text{entropy}}$ 时停止（默认 $\delta_{\text{entropy}} = 0.25$）。同时设置最大推理步数 $K_{\max} = 10$ 防止无限推理
    - 设计动机：低熵意味着模型已经足够自信，继续推理可能导致"过度思考"而无法提升；高熵意味着仍有不确定性，需要更多推理。这自然适应了问题难度

### 损失函数 / 训练策略

VisRef 是完全免训练的框架，不需要任何参数更新或微调，直接在推理时使用。

## 实验关键数据

### 主实验

在三个视觉推理基准上的准确率（%）：

| 模型 | 方法 | MathVision | MathVista | MM-Star |
|------|------|-----------|-----------|---------|
| InternVL3.5-8B | 标准推理 | 39.2 | 68.1 | 57.2 |
| | 文本自反思 | 40.1 | 73.9 | 58.3 |
| | **VisRef** | **44.6 (+5.4)** | **79.3 (+11.2)** | **63.1 (+5.9)** |
| Qwen3-VL-8B | 标准推理 | 53.8 | 74.1 | 66.5 |
| | **VisRef** | **56.6 (+2.8)** | **77.1 (+3.0)** | **69.1 (+2.6)** |
| SAIL-VL2-8B | 标准推理 | 29.8 | 73.1 | 47.7 |
| | **VisRef** | **37.3 (+7.5)** | **78.2 (+5.1)** | **55.3 (+7.6)** |

与训练基方法对比（InternVL3.5-8B）：

| 方法 | MathVista | MathVision | MM-Star |
|------|-----------|-----------|---------|
| Look-Back (RL微调) | 80.8 | 44.2 | 63.7 |
| VisRef (免训练) | 79.3 | 44.6 | 63.1 |
| Look-Back + VisRef | **83.1** | **48.2** | **66.0** |

### 消融实验

相关性 vs 多样性消融（InternVL3.5-8B）：

| 相关性 | 多样性 | MathVista | MathVision | MM-Star |
|--------|--------|-----------|-----------|---------|
| ✓ | ✗ | 75.6 | 43.3 | 61.0 |
| ✗ | ✓ | 77.4 | 42.9 | 62.8 |
| **✓** | **✓** | **79.3** | **44.6** | **63.1** |

### 关键发现

- 文本自反思（TSR）对视觉任务的提升不稳定（-0.6% ~ +2.1%），而 VisRef 始终大幅提升
- 仅用相关性选 token 性能大幅下降，多样性项对避免冗余至关重要
- Token 预算 30% 是最优点，20% 太少（76.1%），40% 无额外增益
- VisRef 与训练基方法（Look-Back）正交，组合使用还能额外提升 2-4%
- 在固定 token 预算条件下，VisRef 的多链并行推理在所有预算水平上都优于纯文本并行推理

## 亮点与洞察

- **完全免训练的 plug-and-play 设计**：不需要任何微调、数据集构建或架构修改，可以直接应用于任何预训练 MLRM，实用性极强
- **DPP 的使用既优雅又有效**：将视觉 token 选择形式化为相关性和多样性的联合优化，比启发式方法有理论保证。$\log\det$ 分解为相关性+多样性的证明非常漂亮
- **注意力可视化验证了直觉**：VisRef 确实使模型在推理过程中持续关注任务相关的视觉区域，而非逐渐丢失视觉信息
- **与训练方法正交**：这意味着 VisRef 可以作为任何 MLRM 的即插即用增强，不论模型是否经过特殊推理训练

## 局限与展望

- DPP 核矩阵的计算需要 $O(N^2 d)$，当视觉 token 数量很大时开销不可忽略
- 当前是均匀地在每步注入固定比例的 token，可以考虑根据问题难度自适应调整注入量
- 仅在 8B 模型上验证，更大模型（70B+）的效果未知
- 停止准则依赖于熵阈值超参，不同任务可能需要不同的最优值

## 相关工作与启发

- **vs Look-Back (RL微调)**: Look-Back 需要 60 GPU 小时的 A6000 微调，VisRef 零训练达到可比性能，二者组合效果更好
- **vs 文本自反思 (Budget Forcing)**: TSR 只延长文本推理，不恢复视觉注意力，在视觉任务上效果不稳定
- **vs 视觉工具调用方法**: 需要架构修改或 SFT/RL 训练，VisRef 更轻量且通用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ DPP 用于推理时视觉 token 选择是全新思路
- 实验充分度: ⭐⭐⭐⭐ 三模型三基准，消融全面，但缺少大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导优雅，从问题定义到解决方案逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 免训练且即插即用，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](red_rationale_enhanced_decoding_cot.md)
- [Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)
- [ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)

<!-- RELATED:END -->
