---
title: >-
  [论文解读] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning
description: >-
  [NeurIPS 2025][多模态推理] Open Vision Reasoner（OVR）通过"语言冷启动 + 大规模多模态 RL"两阶段训练范式，将语言模型中的认知行为（如回溯、验证）有效迁移到视觉推理中，基于 Qwen2.5-VL-7B 在 MathVision 上首次突破 50%（51.8%），成为同规模 SOTA。
tags:
  - NeurIPS 2025
  - 多模态推理
  - 认知行为迁移
  - 强化学习
  - 视觉推理
  - 冷启动微调
---

# Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2507.05255](https://arxiv.org/abs/2507.05255)  
**代码**: 已开源（模型、数据、训练动态）  
**领域**: 强化学习  
**关键词**: 多模态推理, 认知行为迁移, 强化学习, 视觉推理, 冷启动微调

## 一句话总结

Open Vision Reasoner（OVR）通过"语言冷启动 + 大规模多模态 RL"两阶段训练范式，将语言模型中的认知行为（如回溯、验证）有效迁移到视觉推理中，基于 Qwen2.5-VL-7B 在 MathVision 上首次突破 50%（51.8%），成为同规模 SOTA。

## 研究背景与动机

从 RLHF 到 RLVR（基于可验证奖励的强化学习）的转变赋予了 LLM 强大的推理能力，其核心在于模型习得了**认知行为模式**——如回溯（backtracking）、子目标分解（subgoal decomposition）等。多模态领域天然适合 RLVR 范式，因为视觉事实提供了可验证的基准。然而，现有工作面临以下未解决的核心问题：

**语言认知行为如何迁移到视觉推理？** 现有方法要么依赖人工构建的行为轨迹数据集（复杂且不可扩展），要么采用 RLHF 式的学习型奖励模型（容易被 reward hacking）。

**冷启动和 RL 各自的角色是什么？** 之前的工作缺乏对训练各阶段中认知行为如何涌现、演化和迁移的系统性分析。

**训练规模不足**：现有开源 RL 实践在 MLLM 上的规模有限，最长 RL 训练不过几百步。

OVR 的核心动机是：在一个强大的测试平台上（Qwen2.5-VL-7B），通过前所未有的训练规模（200 万冷启动数据 + 30 万 RL 数据、约 1000 步 RL），系统揭示认知行为跨模态迁移的机制。

## 方法详解

### 整体框架

OVR 采用广泛使用的"RL with cold start"范式，分两阶段训练：

- **阶段一（语言冷启动）**：仅对 LLM 模块进行 SFT，在大规模语言推理数据上建立核心认知行为
- **阶段二（多模态 RL）**：在文本和多模态混合任务上用 PPO 进行强化学习，促进推理泛化并将认知模式对齐到视觉上下文

### 关键设计

1. **四类视觉认知行为定义**：
   作者从语言认知行为出发定义了四种视觉对应行为：
    - **视觉反思（Visual Reflection）**：模型在发现推理不一致时主动回看图像，对应语言中的"回溯"（如"让我再看一下图像"）
    - **分治（Divide-and-Conquer）**：将复杂视觉问题分解为子区域逐一处理，对应"子目标设定"
    - **视觉验证（Visual Verification）**：确认中间结论是否有视觉依据，对应"验证"
    - **目标驱动的视觉追溯（Goal-driven Visual Tracing）**：从期望结论反向推理找到图像中的证据，对应"反向推导"

2. **大规模数据策划**：

    - 冷启动数据：从 DeepSeek-R1 蒸馏约 200 万条推理轨迹，涵盖 AIME、MATH、Numina-Math 等数学题及合成逻辑题
    - RL 数据：约 30 万多模态样本，包含几何问题（Geometry3k、GeoQA）、视觉判别（IconQA、ChartQA）、视觉谜题（PuzzleVQA）、STEM 等
    - 多步策划：预训练模型过滤高 loss 噪声样本→规则和模型辅助移除不良模式→重加权平衡类别→移除不兼容奖励函数的题目

3. **PPO + GAE 优化**：
   采用轻量级 PPO 算法，结合广义优势估计（GAE）。优势估计：
    $\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_{t'} = r_{t'} + \gamma V_\phi(s_{t'+1}) - V_\phi(s_{t'})$
   
   策略更新使用标准 PPO clip 目标：
    $\mathcal{J}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_{\pi_{\text{old}}}\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$

### 损失函数 / 训练策略

- **奖励函数**：极简规则奖励——从模型输出的 `\boxed{}` 中提取答案，与参考答案精确匹配得 1 分，否则 0 分。不使用格式或风格偏好奖励。
- **冷启动训练**：5 轮 epoch，batch size 640，序列长度 64k，学习率 $2 \times 10^{-4}$，采用激进的大 batch + 高学习率策略打破模型固有约束
- **RL 训练**：900 步迭代，采用序列长度课程学习策略——前 300 步 24k → 300-700 步 32k → 700 步后 48k。严格遵循 on-policy 更新。
- **最终模型**：多个代表性中间检查点的均匀平均，确保各基准上的平衡性能

## 实验关键数据

### 主实验：视觉推理基准

| 模型 | MathVista | MathVision | MathVerse | DynaMath | WeMath | MMMU-Pro |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B | 69.2 | 25.5 | 41.1 | 31.2 | 53.1 | 47.9 |
| R1-OneVision-7B | 64.1 | 29.9 | 40.0 | - | - | - |
| ReVisual-R1 | 73.1 | 48.8 | 53.6 | 42.0 | - | 52.3 |
| MM-Eureka-7B | 72.6 | 28.1 | 45.4 | 21.8 | - | 46.3 |
| **OVR-7B** | **72.1** | **51.8** | **54.6** | **44.6** | **64.8** | **54.8** |

OVR 在 MathVision（+3.0）和 MathVerse（+1.0）上均超越此前 SOTA ReVisual-R1，MMMU-Pro 也提升 +2.5，是首个在 Qwen2.5-VL-7B 基础上 MathVision 突破 50% 的模型。

### 语言推理基准

| 模型 | AIME 2024 | AIME 2025 | MATH500 | GPQA | MMLU | MMLU-Pro |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B | 6.7 | 6.7 | 67.4 | 31.8 | 69.6 | 51.7 |
| DeepSeek-R1-Distill-7B | 55.5 | 39.2 | 92.8 | 49.1 | - | - |
| ReVisual-R1 | 53.3 | 43.3 | 89.2 | 47.5 | - | - |
| **OVR-7B** | **63.5** | **52.1** | **95.3** | **49.8** | **77.2** | **67.9** |

OVR 在 AIME 2024/2025 上超越同规模模型 10% 以上，MATH500 达到 95.3%，接近GPT-4o 水平。

### 消融实验 / 认知行为分析

| 训练阶段 | 视觉反思涌现 | 回溯迁移率 | 验证迁移率 |
|----------|:---:|:---:|:---:|
| 基线 (Qwen2.5-VL-7B) | 低 | ~2.5% | ~0% |
| 冷启动后 | 显著增加 | ~10% | ~0% |
| RL 后 | 大幅增加 | ~17.3% | ~0% |

| 序列长度课程 | 效果 |
|-------------|------|
| 24k（前 300 步） | 奖励和序列长度稳步增长 |
| 32k（300-700 步） | 突破平台期，催化新一轮增长 |
| 48k（700 步后） | 继续提升至最终性能 |

### 关键发现

- **认知行为迁移极早涌现**：得益于 DeepSeek-R1 中的"心理意象"（如"让我可视化…"），视觉反思行为在冷启动一开始就大量出现
- **冷启动广泛记忆，RL 关键筛选**：冷启动阶段不加区分地记忆所有模式；RL 阶段则作为"去粗取精"的过滤器，放大关键行为
- **迁移具有策略性**：回溯的迁移率持续增长（2.5%→17.3%），而验证的迁移率始终接近零，说明迁移优先选择高效用行为
- **感知能力先降后恢复**：语言冷启动引入感知退化，但多模态 RL 能有效恢复

## 亮点与洞察

- **语言中的"内心视觉"**：DeepSeek-R1 在数学推理中自发使用"心理意象"，这在 MLLM 中天然转化为真实视觉交互
- **训练动态的深刻分析**：从行为涌现、迁移率、序列长度课程等多个维度揭示了训练的内在机制
- **超大规模开源 RL 实践**：在 Qwen2.5-VL-7B 上进行约 1000 步 RL 训练，是已知最大规模的开源多模态 RL 实验

## 局限与展望

- 仅 RL 训练感知任务时可扩展性有限——奖励信号增长但推理复杂度不增
- 验证（Verification）行为的跨模态迁移率接近零，说明某些认知模式需要更原生的视觉支持
- 最终模型需要多检查点平均，缺乏单一最优检查点
- 尚未探索多轮或智能体式 RL 框架在视觉操作和想象力方面的潜力

## 相关工作与启发

- **DeepSeek-R1**：核心灵感来源，提供冷启动数据蒸馏基础
- **Open-Reasoner-Zero**：RL 训练框架的基础
- **Perception-R1, ReVisual-R1**：同期多模态 RL 工作，OVR 在规模和系统性分析上超越
- 启发：语言推理→视觉推理的迁移不是简单的模态扩展，而是认知行为层面的结构性传递

## 评分

- **新颖性**: ⭐⭐⭐⭐ 系统揭示语言认知行为如何迁移到视觉推理，分析深入
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 8+ 个视觉基准和 6 个语言基准，训练动态分析详尽
- **写作质量**: ⭐⭐⭐⭐ 概念清晰，认知行为表格定义简洁明了
- **价值**: ⭐⭐⭐⭐⭐ 最大规模开源多模态 RL 实践 + 深层机制分析，对领域发展有重要推动

<!-- RELATED:START -->

## 相关论文

- [NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)

<!-- RELATED:END -->
