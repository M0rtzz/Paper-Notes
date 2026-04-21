---
title: >-
  [论文解读] Training Large Reasoning Models Efficiently via Progressive Thought Encoding
description: >-
  [ICLR 2026][LLM推理][大推理模型] 提出 Progressive Thought Encoding，在 KV 缓存受限条件下将被驱逐的思维 token 编码进 LoRA 权重，使大推理模型在 RL 训练时显存减半的同时推理准确率反超全缓存 LoRA（AIME2024/2025 上最高提升 +23.4%）。
tags:
  - ICLR 2026
  - LLM推理
  - 大推理模型
  - 强化学习训练效率
  - KV缓存压缩
  - 参数高效微调
  - 渐进式思维编码
---

# Training Large Reasoning Models Efficiently via Progressive Thought Encoding

**会议**: ICLR 2026  
**arXiv**: [2602.16839](https://arxiv.org/abs/2602.16839)  
**代码**: 无公开代码  
**领域**: LLM推理  
**关键词**: 大推理模型, 强化学习训练效率, KV缓存压缩, 参数高效微调, 渐进式思维编码

## 一句话总结

提出 Progressive Thought Encoding，在 KV 缓存受限条件下将被驱逐的思维 token 编码进 LoRA 权重，使大推理模型在 RL 训练时显存减半的同时推理准确率反超全缓存 LoRA（AIME2024/2025 上最高提升 +23.4%）。

## 研究背景与动机

**RL 训练的核心瓶颈**：大推理模型（LRM）通过 RL（如 GRPO）进行后训练，需要长 rollout 序列获取 outcome-based reward。自回归解码使 rollout 阶段成为时间和显存的主要瓶颈——困难任务需要更长的思维链，进一步加剧资源消耗。

**滑动窗口的困境**：直觉上可以用滑动窗口限制 KV 缓存大小来降低显存。但实验显示这会严重损害推理质量——丢弃中间思维 token 破坏了长距离上下文理解能力，导致 rollout 样本质量下降，进而影响训练效果。例如 Qwen2.5-3B 在滑动窗口下平均准确率从 28.2% 降至 25.6%。

**核心问题**：能否在严格的显存预算下训练 LRM，同时不牺牲推理准确率？即让模型在有限缓存窗口下仍能"看到"所有历史 token。

## 方法详解

### 整体框架

Progressive Thought Encoding 的核心思想是：**不丢弃被驱逐的 token，而是先从中学习再丢弃**。具体而言，当 KV 缓存满时，被驱逐 token 的信息被编码为固定大小的向量表示，动态注入轻量级 LoRA 适配器中，使模型在有限缓存下保持长上下文理解能力。

工作流程：
1. 给定问题 x，在 rollout 阶段持续解码思维 token 直到 KV 缓存满
2. 根据驱逐策略 D 选定待驱逐的 token {y_e1, ..., y_em}
3. 利用全局查询向量 q_g 从被驱逐 token 计算上下文状态 S_e
4. 通过 S_e 更新 LoRA 权重：ΔW = A · S_e · B
5. 模型在更新后的策略下继续解码，每次缓存满时重复此过程

### 关键设计

**上下文状态计算**：引入可学习的全局查询向量 q_g 作为所有被驱逐上下文的摘要载体。通过注意力机制聚合被驱逐 token 的信息：

- S_e = (W_Q^a · q_g) · (W_K^a · K_e)^T · (W_V^a · V_e)

其中 K_e 和 V_e 是被驱逐 token 的键值向量，W_Q^a、W_K^a、W_V^a 是将全局查询和被驱逐 token 映射到压缩潜空间的权重矩阵。

**累积更新机制**：每次新一批 token 被驱逐时，计算新的 S_e'，通过 S_e ← Normalize(S_e + S_e') 累积更新，再重新计算 ΔW。这实现了流式适应——模型在整个生成过程中持续"记住"被驱逐的 token。

**全局 token 初始化**：在处理任何被驱逐 token 之前，用可学习的全局 token h_g 初始化上下文状态，使 q_g 从一开始就是被驱逐上下文信息的显式载体。

### 训练策略

- **驱逐策略**：训练时对问题 token 永久保留（类似 sink token 机制），仅对思维 token 采用滑动窗口驱逐，缓存饱和时驱逐 25% 的 token
- **RL 算法**：基于 GRPO，使用 DAPO-Math-17K 数据集，全局 batch size 512
- **参数设置**：LoRA rank 32，全局 token 数 32，学习率 1e-5
- **缓存大小**：设为当前 micro-batch 中最大问题长度

## 实验关键数据

### 主实验

在 3 个模型 × 6 个数学推理基准上的对比（最大生成长度 3072）：

| 方法 | Peak GPU Mem | Math500 | Olympiad | AMC | AIME24 (p@16) | AIME25 (p@16) | 平均 |
|------|-------------|---------|----------|-----|---------------|---------------|------|
| **Qwen2.5-3B** | | | | | | | |
| Baseline | - | 50.8 | 27.2 | 34.3 | 20.0 | 13.3 | 26.9 |
| LoRA | 82.8% | 53.2 | 27.8 | 35.9 | 20.0 | 16.7 | 28.2 |
| LoRA_c (滑动窗口) | 38.0% | 50.0 | 27.7 | 33.1 | 16.7 | 10.0 | 25.6 |
| **Ours** | **45.3%** | **54.0** | **29.0** | **45.0** | **20.0** | **16.7** | **30.1** |
| **DeepSeek-R1-Distill-8B** | | | | | | | |
| Baseline | - | 53.6 | 28.7 | 42.5 | 20.0 | 20.0 | 30.1 |
| LoRA | 88.7% | 57.4 | 35.3 | 55.0 | 23.3 | 20.0 | 34.9 |
| LoRA_c | 59.1% | 54.2 | 31.9 | 45.0 | 36.7 | 26.7 | 35.1 |
| **Ours** | **59.8%** | **57.6** | **39.7** | **60.0** | **56.7** | **43.3** | **45.6** |

### 消融实验

全局 token 与驱逐策略的影响（DeepSeek-R1-Distill-8B, MATH-500）：

| 配置 | 缓存768 | 缓存1K | 缓存2K | 说明 |
|------|---------|--------|--------|------|
| Baseline | 34.4 | 39.6 | 47.8 | 无 RL 训练 |
| #Global-0（无全局 token） | 36.2 | 41.0 | 48.6 | 仅驱逐编码，提升有限 |
| Global-Only（无驱逐更新） | 46.8 | 50.2 | 54.0 | 全局 token 有效但不足 |
| **Ours (#Global-32)** | **48.4** | **52.2** | **55.4** | 全局+驱逐编码最优 |
| Ours + HeadKV | 50.7 | 53.4 | 55.8 | 更好的驱逐策略有帮助 |

**可扩展性**：固定 1K 缓存窗口，将最大生成长度从 3K 扩展到 64K，本方法在整个长度范围内持续缩放提升，而 LoRA 和 LoRA_c 逐步饱和。

### 关键发现

1. **显存减半、精度反超**：在 DeepSeek-R1-8B 上，Peak GPU 从 88.7% 降至 59.8%（-28.9%），平均准确率从 34.9% 升至 45.6%（+10.7%）
2. **AIME 上表现惊人**：DeepSeek-R1-8B 在 AIME2024 上从 23.3 提升到 56.7（+33.4），AIME2025 从 20.0 到 43.3（+23.3）
3. **更长推理 = 更好效果**：渐进编码使训练时可安全增大 rollout 长度（4K→6K），显存几乎不变但 MATH-500 从 57.6 升至 60.2
4. **长序列推理更稳定**：本方法在长响应上表现尤为突出，而 LoRA_c 的增益主要来自短响应

## 亮点与洞察

- **化废为宝**：将 KV 缓存驱逐从"信息丢失"转变为"在线学习机会"，是一个极为巧妙的视角转换
- **推理 RL 的实际可行性**：显存消耗是阻碍 RL 训练扩展的主要障碍，本方法直接解决了这个工程痛点
- **test-time learning 的新形式**：渐进编码本质上是推理时的在线自适应——模型在生成过程中不断学习自己的中间思维

## 局限与展望

1. **仅验证了数学推理**：6 个基准全部是数学任务，代码生成、科学推理等场景未覆盖
2. **驱逐策略保守**：训练时仅用简单滑动窗口，作者自己指出更优的 token 选择策略（如 HeadKV）可进一步提升但会增加 37% 运行时间
3. **全局 token 数敏感**：#Global-64 反而不如 #Global-32，超参选择需要调优
4. **未与 token-level reward 方法对比**：当前仅考虑 outcome-based reward，process reward 可能进一步提升

## 相关工作与启发

- 与 **test-time training**（如 entropy minimization）互补——本方法是"生成时训练"
- 与 **KV 缓存压缩**（PyramidKV, H2O, HeadKV）正交——可直接集成更优的驱逐策略
- 启发：**RL 训练中的效率优化**是当前 LRM 研究的关键方向，本方法提供了一条从"缓存管理"切入的新路径

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将缓存驱逐转化为在线学习信号，思路极具创意
- 实验充分度: ⭐⭐⭐⭐ 3 模型 × 6 基准 + 丰富消融，但限于数学领域
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐⭐ 直击 LRM RL 训练的核心痛点，工程意义重大

<!-- RELATED:START -->

## 相关论文

- [Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)
- [When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_llms_compression_o.md)

<!-- RELATED:END -->
