---
title: >-
  [论文解读] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models
description: >-
  [CVPR 2026][LLM推理][visual refocusing] 提出 VisRef，一个免训练的视觉重聚焦框架——在多模态大推理模型（MLRM）的推理过程中，通过行列式点过程（DPP）在每步自适应选择与当前推理状态语义相关且视觉覆盖多样的 token 子集并重新注入，同时用基于熵的停止准则防止过度推理，在固定计算预算下将视觉推理准确率提升最高 6.4%。
tags:
  - CVPR 2026
  - LLM推理
  - visual refocusing
  - test-time scaling
  - 多模态
  - DPP
  - visual token selection
  - training-free
---

# VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models

**会议**: CVPR 2026  
**arXiv**: [2603.00207](https://arxiv.org/abs/2603.00207)  
**代码**: 无  
**领域**: 多模态推理 / Test-time Scaling  
**关键词**: visual refocusing, test-time scaling, multimodal reasoning, DPP, visual token selection, training-free

## 一句话总结

提出 VisRef，一个免训练的视觉重聚焦框架——在多模态大推理模型（MLRM）的推理过程中，通过行列式点过程（DPP）在每步自适应选择与当前推理状态语义相关且视觉覆盖多样的 token 子集并重新注入，同时用基于熵的停止准则防止过度推理，在固定计算预算下将视觉推理准确率提升最高 6.4%。

## 研究背景与动机

**领域现状**：多模态大推理模型（MLRMs）如 InternVL-3.5、Qwen-3-VL、SAIL-VL2 通过扩展 Chain-of-Thought 推理到视觉语言任务上取得了显著进展。然而近期研究（Chu et al., Yang et al.）发现一个关键问题：随着推理链长度增加，模型对视觉 token 的注意力逐步衰减，越来越依赖文本先验而非图像内容。

**现有痛点**：(1) 基于 RL 微调的方法（如 Look-Back）可以教模型自主"回看"视觉输入，但需要 60 GPU 小时的微调和大规模标注数据集构建，可扩展性差；(2) 现有 test-time scaling 方法（如 Budget Forcing、L1）纯文本导向——通过"Wait"/"Think more"等指令让模型继续推理，但不主动维护视觉接地，视觉信息继续衰减；(3) 简单地重新注入所有视觉 token 计算上不可行——InternVL-3.5-8B 上每张图约 1772 个视觉 token vs 每步约 615 个文本 token，全量注入导致 2.3x 推理延迟。

**核心矛盾**：人类解决多模态问题时会自然地在"看图"和"推理"之间交替往返，但当前 MLRM 在初始处理视觉 token 后就不再回看——推理链越长视觉接地越弱。训练式解决方案效果好但代价高，纯文本 test-time scaling 不解决根本问题。

**本文目标**：能否完全在测试时恢复视觉接地，不需要任何重训练或微调？

**切入角度**：在每步推理时自适应选择一小部分（30%）与当前推理上下文最相关且覆盖最广的视觉 token 重新注入，用 DPP 框架同时优化相关性和多样性。

**核心 idea**：将视觉 token 选择形式化为最大化 DPP 行列式的优化问题，实现推理过程中自适应的视觉重聚焦，无需训练即可在任意预训练 MLRM 上即插即用。

## 方法详解

### 整体框架

给定图文输入 $x_{\text{input}} = [I, T]$ 和视觉 token 集合 $\mathcal{V} = \{v_1, \ldots, v_N\}$，VisRef 在每步推理 $k$ 产生文本推理步 $z_k$ 后：(1) 基于 DPP 从 $\mathcal{V}$ 中选择 $m$ 个视觉 token 子集 $V_k$；(2) 将 $V_k$ 注入下一步的上下文中；(3) 检查模型响应分布的熵是否低于阈值 $\delta_{\text{entropy}}$，决定是否终止推理。最终答案 $y \sim \pi_\theta(\cdot | x_{\text{input}}, \tau_{1:k})$，其中 $\tau_{1:k} = \{(z_1, V_1), \ldots, (z_k, V_k)\}$。

### 关键设计

1. **DPP-based 视觉 token 选择**
    - 功能：在每步推理时选择一个既与当前推理状态相关、又视觉覆盖多样的 token 子集
    - 核心思路：定义文本子空间几何 $M_k = \sum_{j=1}^{T_k} z_k^{(j)}(z_k^{(j)})^\top$，构造核函数 $L_k(v_i, v_j) = v_i^\top M_k v_j$。优化目标为最大化核矩阵行列式 $\tilde{V}_k = \arg\max_{V_k \subseteq \mathcal{V}} \det(L_k^{V_k})$。此行列式自然分解为两项：$\log\det(L_k^{V_k}) = \underbrace{\sum_{v_i \in V_k} \log(r_i^2)}_{\text{relevance}} + \underbrace{\log\det(\bar{L}_k^{V_k})}_{\text{diversity}}$，其中 $r_i^2 = \sum_{j=1}^{T_k}(v_i^\top z_k^{(j)})^2$ 衡量相关性
    - 设计动机：朴素注入所有视觉 token 导致 2.3x 延迟；只选相关的会冗余；DPP 同时优化相关性和多样性是最优平衡

2. **贪心近似求解**
    - 功能：高效求解 NP-hard 的子集选择问题
    - 核心思路：从空集开始，每次选择边际增益最大的 token $v_{k,i} = \arg\max_{v \in \mathcal{V} \setminus V_k^{(i-1)}} \log\frac{\det(L_k^{V_k^{(i-1)} \cup \{v\}})}{\det(L_k^{V_k^{(i-1)}})}$，迭代 $m$ 次。在实验中 token 预算 $m = \lfloor 0.3|\mathcal{V}| \rfloor$
    - 设计动机：精确求解最大行列式子集选择是 NP-hard 的，贪心算法提供 $(1-1/e)$ 近似比保证

3. **基于熵的自适应停止准则**
    - 功能：防止过度推理（overthinking）或推理不足
    - 核心思路：在每步推理 $k$ 后计算模型的响应分布熵 $H_k = -\mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y | x_{\text{input}}, \tau_{1:k})]$，当 $H_k < \delta_{\text{entropy}} = 0.25$ 时终止，表示模型已收敛到高置信答案。同时设最大步数 $K_{\max} = 10$ 防止无限推理
    - 设计动机：简单问题快速达到低熵而提前终止（节省计算），复杂问题利用更多推理步。$\delta_{\text{entropy}} = 0.25$ 在所有模型上一致最优

### 损失函数 / 训练策略

VisRef 完全不需要训练。所有操作在推理时完成：视觉 token 选择通过 DPP 贪心算法实现，重新注入通过修改上下文序列实现，停止通过熵计算判定。该方法即插即用，适用于任何预训练的 MLRM。

## 实验关键数据

### 主实验（三个视觉推理基准，三个 MLRM）

| 模型 | 方法 | MathVision | MathVista | MM-Star |
|------|------|-----------|-----------|---------|
| InternVL3.5-8B | Standard Thinking | 39.2 | 68.1 | 57.2 |
| InternVL3.5-8B | Textual Self-Reflection | 40.1 | 73.9 | 58.3 |
| InternVL3.5-8B | **VisRef** | **44.6 (+5.4)** | **79.3 (+11.2)** | **63.1 (+5.9)** |
| Qwen3-VL-8B | Standard Thinking | 53.8 | 74.1 | 66.5 |
| Qwen3-VL-8B | Textual Self-Reflection | 54.3 | 74.2 | 65.9 |
| Qwen3-VL-8B | **VisRef** | **56.6 (+2.8)** | **77.1 (+3.0)** | **69.1 (+2.6)** |
| SAIL-VL2-8B | Standard Thinking | 29.8 | 73.1 | 47.7 |
| SAIL-VL2-8B | Textual Self-Reflection | 31.9 | 73.8 | 48.9 |
| SAIL-VL2-8B | **VisRef** | **37.3 (+7.5)** | **78.2 (+5.1)** | **55.3 (+7.6)** |

### 消融实验

相关性 vs 多样性消融（InternVL-3.5-8B）：

| 相关性 | 多样性 | MathVista | MathVision | MM-Star |
|--------|--------|-----------|-----------|---------|
| ✓ | ✗ | 75.6 | 43.3 | 61.0 |
| ✗ | ✓ | 77.4 | 42.9 | 62.8 |
| ✓ | ✓ | **79.3** | **44.6** | **63.1** |

与训练式方法 Look-Back 的对比（InternVL-3.5-8B）：

| 方法 | MathVista | MathVision | MM-Star |
|------|-----------|-----------|---------|
| Standard Thinking | 68.1 | 39.2 | 57.2 |
| Look-Back (需 60 GPU-hr) | 80.8 | 44.2 | 63.7 |
| VisRef (免训练) | 79.3 | 44.6 | 63.1 |
| Look-Back + VisRef | **83.1** | **48.2** | **66.0** |

### 关键发现

- 纯文本自反思（TSR）收益不稳定（0.1%-2.1%），甚至在 Qwen3-VL-8B 的 MM-Star 上出现 0.6% 下降，说明纯文本推理延长对视觉任务帮助有限
- VisRef 在所有 9 个（3模型×3基准）配置上一致胜出，最大提升 11.2%（InternVL3.5 on MathVista）
- 仅用相关性选择比仅用多样性差（MathVista 75.6 vs 77.4），说明多样性对覆盖视觉信息至关重要
- VisRef 免训练即达到与 Look-Back（60 GPU-hr 微调）接近的性能（MathVista 79.3 vs 80.8），且两者正交——结合后进一步提升到 83.1
- Token 预算 $m=30\%$ 是最优折中点：20% 不足（76.1%），30% 最优（79.2%），40% 无额外收益
- 在固定 token 预算（如 14K thinking tokens）的并行链场景下，VisRef 比无视觉重聚焦的并行推理高约 6% 准确率

## 亮点与洞察

- **理论优雅**：将视觉 token 选择形式化为 DPP 行列式最大化问题，并证明其自然分解为相关性+多样性两项——数学上清晰解释了"为什么 DPP 适合这个问题"
- **即插即用**：无需训练数据、无需微调、无需架构修改——可以立即应用于任何预训练的 MLRM，这在实际部署中极具价值
- **与训练式方法互补**：VisRef + Look-Back 的组合在所有基准上都超过单独使用任一方法，说明两者捕获了不同的视觉接地信号
- **注意力可视化**：Figure 5 直观展示了 VisRef 如何在推理步中逐步将注意力从弥散状态重聚焦到任务关键的视觉区域

## 局限与展望

- 每步都需要计算 DPP 选择和重新注入，虽然只选 30% token 但仍增加了上下文长度和计算量
- 当前的 DPP 核函数 $L_k$ 基于简单的文本子空间投影，更复杂的跨模态对齐可能带来进一步提升
- 熵阈值 $\delta_{\text{entropy}} = 0.25$ 虽然跨模型一致，但对不同难度和领域的问题可能需要更细粒度的调整
- 仅评估了 8B 参数量级的模型，对更大模型（如 70B+）的 scaling 行为未验证
- 在 Qwen3-VL-8B 上收益（2-3%)相对较小，可能因为该模型本身视觉接地就更强

## 相关工作与启发

- **Budget Forcing (Muennighoff et al.)**：通过"Wait"指令扩展推理链的 test-time scaling 方法，但纯文本导向
- **Look-Back (Chu et al.)**：基于 RL 微调教模型自主回看视觉输入，效果好但代价高（60 GPU-hr）
- **L1 (Aggarwal & Welleck)**：长度可控的策略优化，精确控制推理链长度
- 启发：VisRef 展示了"主动维护视觉接地"比"被动延长推理"更有效，核心洞察是——对于视觉任务，问题不在于想得不够多，而在于看得不够仔细

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐（DPP 框架用于视觉 token 选择是首创，理论推导优雅）
- **实验充分度**: ⭐⭐⭐⭐⭐（3模型×3基准，丰富消融包含相关性/多样性分解、训练式方法对比、token 预算分析、test-time scaling 曲线）
- **写作质量**: ⭐⭐⭐⭐⭐（问题动机明确，方法推导严谨，图表信息量大）
- **价值**: ⭐⭐⭐⭐⭐（免训练即插即用 + 显著且一致的提升 + 与训练式方法正交互补，实用价值很高）

<!-- RELATED:START -->

## 相关论文

- [Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)
- [Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)
- [Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)

<!-- RELATED:END -->
