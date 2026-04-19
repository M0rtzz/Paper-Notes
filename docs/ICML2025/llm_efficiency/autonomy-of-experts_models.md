---
title: >-
  [论文解读] Autonomy-of-Experts Models (AoE)
description: >-
  [ICML 2025][LLM效率][Mixture-of-Experts] AoE 提出让 MoE 中的 expert 基于自身内部激活范数自主决定是否处理输入（而非由外部 router 决定），通过低秩权重分解降低预计算开销，在 700M-4B 参数语言模型预训练中超越传统 MoE。
tags:
  - ICML 2025
  - LLM效率
  - Mixture-of-Experts
  - 专家自主选择
  - 激活范数
  - 低秩分解
  - 大语言模型
---

# Autonomy-of-Experts Models (AoE)

**会议**: ICML 2025  
**arXiv**: [2501.13074](https://arxiv.org/abs/2501.13074)  
**代码**: https://github.com/trestad/Autonomy-of-Experts  
**领域**: LLM效率  
**关键词**: Mixture-of-Experts, 专家自主选择, 激活范数, 低秩分解, 大语言模型

## 一句话总结

AoE 提出让 MoE 中的 expert 基于自身内部激活范数自主决定是否处理输入（而非由外部 router 决定），通过低秩权重分解降低预计算开销，在 700M-4B 参数语言模型预训练中超越传统 MoE。

## 研究背景与动机

**领域现状**：Mixture-of-Experts (MoE) 是当前大语言模型的核心架构之一（Mixtral、DeepSeek-MoE、Qwen-MoE 等）。MoE 将大型 FFN 分解为多个小型 FFN（expert），通过 router 为每个 token 选择 Top-K 个 expert 进行处理，实现稀疏激活以提升效率。Router 通常是一个简单的 MLP 分类器，根据输入 hidden state 输出 expert 选择概率。

**现有痛点**：传统 MoE 存在一个被广泛忽视的关键问题——**router 的决策和 expert 的执行之间的分离**。具体来说：(a) Router 无法直接评估 expert 的能力，它的选择本质上是一个"没有标签的预测"；(b) 如果 router 做了错误预测，被选中的 expert 可能难以有效处理 token，导致训练 loss 增加；(c) expert 可能被迫调整参数去处理不擅长的 token，与其原有专长冲突；(d) router 只能通过试错来学习更好的决策，浪费大量训练 step。

**核心矛盾**：在传统 MoE 中，"谁来处理这个 token"的决策权在 router 手中，但 router 对 expert 的实际能力一无所知。这种决策-执行分离导致了**次优的 expert 选择**和**低效的训练**。同时，MoE 中的 load balancing 辅助损失虽然缓解了 expert 负载不均的问题，但没有从根本上解决选择质量的问题。

**本文目标**：(a) 如何让 expert 选择更准确——基于 expert 自身的能力判断而非外部 router 的猜测？(b) 如何在消除 router 的同时保持计算效率？

**切入角度**：作者从 FFN-as-key-value-memory 的视角（Geva et al., 2021）出发，提出一个关键洞察：**expert 的内部激活范数反映了其处理输入的能力**。如果 expert 能有效处理某个输入，其"key"向量（内部激活）应该被高度激活。验证实验表明，在预训练的 Mixtral 8×7B 上删除 router、仅按 expert 内部激活范数选择 Top-K，无需任何参数更新就能保留高达 95% 的原始性能。

**核心 idea**：移除 router，让每个 expert 先对输入做低秩预计算得到内部激活，按激活范数排名选 Top-K 继续前向传递，实现 expert 自主选择。

## 方法详解

### 整体框架

AoE 修改了 MoE 层的工作流程：

- **传统 MoE**：输入 $\mathbf{x}$ → Router $R(\mathbf{x})$ 输出概率 → 选 Top-K expert → 被选 expert 做完整 FFN 计算 → 加权求和
- **AoE**：输入 $\mathbf{x}$ → 所有 expert 做低秩下投影 $\mathbf{x}\mathbf{W}_{down}^i$ → 缓存激活并计算 $L^2$ 范数 → 按范数选 Top-K → 被选 expert 从缓存继续前向计算 → 加权求和

核心区别：没有 router，expert 的选择完全由其自身的内部激活决定。未被选中的 expert 在第一步低秩投影后就终止计算。

### 关键设计

1. **Expert 自主选择机制**：

    - 功能：每个 expert 对输入做预计算，产生内部激活，按 $L^2$ 范数排名决定是否继续处理。
    - 核心思路：传统 expert 的计算公式为 $E_i(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_g^i) \odot (\mathbf{x}\mathbf{W}_p^i))\mathbf{W}_o^i$。AoE 的关键观察是 $\mathbf{x}\mathbf{W}_g^i$ 的范数能反映 expert $i$ 处理 $\mathbf{x}$ 的能力。范数大表示 expert 的"key"被高度激活，意味着它擅长处理这个输入；范数小则表示不匹配。
    - 设计动机：FFN 可以被理解为 key-value 记忆网络。如果 expert 能有效处理输入，对应的"key"（$\mathbf{x}\mathbf{W}_g$）应该被高度激活，从而通过与"value"（$\mathbf{W}_o$）的匹配实现有效的知识检索。这个self-evaluation 机制让 expert 基于自身对输入的"理解"来决策，消除了 router 决策和 expert 执行之间的信息鸿沟。

2. **低秩权重分解（Low-Rank Weight Factorization）**：

    - 功能：将 $\mathbf{W}_g$ 分解为 $\mathbf{W}_{down} \in \mathbb{R}^{d_{model} \times d_{low}}$ 和 $\mathbf{W}_{up} \in \mathbb{R}^{d_{low} \times d_{wide}}$，AoE expert 公式变为 $E_i(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_{down}^i \mathbf{W}_{up}^i) \odot (\mathbf{x}\mathbf{W}_p^i))\mathbf{W}_o^i$。
    - 核心思路：有两个关键好处：(a) 所有 expert 的 $\mathbf{W}_{down}^i$ 可以拼接成一个大矩阵 $\hat{\mathbf{W}}_{down} = [\mathbf{W}_{down}^1, \cdots, \mathbf{W}_{down}^n] \in \mathbb{R}^{d_{model} \times (n \cdot d_{low})}$，通过单次矩阵乘法 $\mathbf{C} = \mathbf{x}\hat{\mathbf{W}}_{down}$ 同时得到所有 expert 的低维激活缓存；(b) 未被选中的 expert 只做了低维投影（$d_{low} \ll d_{model}$），计算和缓存开销都很小。
    - 设计动机：如果不做分解，所有 expert 都要计算完整的 $\mathbf{x}\mathbf{W}_g^i \in \mathbb{R}^{d_{ffn}}$（如 Mixtral 中 $d_{ffn}=14336$），缓存和计算开销巨大。LLM 的权重本身就是低秩的（LoRA 等工作已验证），因此这种分解不损害表达能力。最优的 $d_{low}$ 约为 $d_{model}/3$。

3. **$d_{low}$ 与 $d_{wide}$ 的参数预算约束**：

    - 功能：在总参数量与传统 MoE 对齐的前提下，根据 $d_{low}$ 自动计算 $d_{wide}$。
    - 核心思路：$d_{wide} = \frac{3 \cdot d_{model} \cdot d_{ffn} - d_{low} \cdot d_{model}}{d_{low} + 2 \cdot d_{model}}$，确保 AoE 和 MoE 在总参数量上可比。
    - 设计动机：公平比较需要控制参数量。$d_{low}$ 减小会使 $d_{wide}$ 增大，反之亦然。

4. **可选的辅助 load-balancing loss**：

    - AoE 可以兼容传统 MoE 的 $\mathcal{L}_{aux}$，只需将 router 输出替换为 $L^2$-Norm($\mathbf{x}\mathbf{W}_{down}^i$) 的 softmax。实验表明 AoE 即使不用 $\mathcal{L}_{aux}$ 也比传统 MoE 更均衡，但加上后效果更好。

### 损失函数/训练策略

- **主损失**：标准语言模型的 NLL loss
- **辅助损失**（可选）：$\mathcal{L}_{aux} = \alpha_{aux} \cdot n \cdot \sum_{i=1}^{n} \mathbf{f}_i \cdot \mathbf{P}_i$，$\alpha_{aux}=0.01$
- 优化器：AdamW，$(\beta_1, \beta_2) = (0.9, 0.95)$，权重衰减 0.1
- 训练数据：RedPajama，100B tokens（小模型）/ 更大规模（大模型）
- 学习率：$2 \times 10^{-4}$（小模型）/ $3.2 \times 10^{-4}$（大模型），线性 warmup + cosine decay

## 实验关键数据

### 主实验

732M 参数（247M 激活参数）语言模型，100B tokens 训练，8 任务平均准确率：

| 模型配置 | ARC-E | PIQA | SIQA | WINO | HELLA | MNLI | QNLI | SST2 | AVG |
|---------|-------|------|------|------|-------|------|------|------|-----|
| Traditional MoE | 39.90 | 58.43 | 35.67 | 52.09 | 27.98 | 33.09 | 49.28 | 49.66 | 43.28 |
| MoE + $\mathcal{L}_{aux}$ | 40.74 | 58.49 | 36.13 | 51.30 | 28.11 | 32.67 | 50.23 | 51.83 | 43.68 |
| AoE ($d_{low}$=256) | 40.70 | **59.41** | 36.64 | 52.09 | 28.06 | 34.38 | 50.69 | 53.21 | **44.39** |
| AoE ($d_{low}$=256) + $\mathcal{L}_{aux}$ | 41.33 | 58.65 | 36.80 | 50.75 | 28.40 | 33.71 | 49.55 | 53.10 | 44.04 |

4B 参数（1.18B 激活参数）大模型对比：

| 模型 | ARC-E | PIQA | SIQA | WINO | HELLA | MNLI | QNLI | SST2 | AVG |
|------|-------|------|------|------|-------|------|------|------|-----|
| Traditional MoE | 53.70 | 65.40 | 39.10 | 51.54 | 35.80 | 32.19 | 49.77 | 57.00 | 48.06 |
| AoE | **55.98** | **65.61** | **39.87** | **52.57** | **36.77** | **35.39** | **50.05** | **61.93** | **49.80** |

4B 模型上 AoE 优势更加显著（+1.74 平均准确率），说明 AoE 的优势随模型规模扩大而增强。

### 消融实验

| 配置 | AVG 准确率 | 说明 |
|------|-----------|------|
| MoE baseline | 43.28 | 传统 MoE 无 $\mathcal{L}_{aux}$ |
| MoE + factorized $\mathbf{W}_g$ | 43.70 | 仅分解权重，不改选择机制 → 几乎无提升 |
| MoE + large router | 43.71 | 增大 router 参数量到 AoE 同等级 → 也无明显提升 |
| AoE ($d_{low}$=64) | 43.81 | 过度压缩，近似损失大 |
| AoE ($d_{low}$=128) | 44.12 | 较好 |
| AoE ($d_{low}$=256) | **44.39** | 最优，约为 $d_{model}/3$ |
| AoE ($d_{low}$=512) | 44.12 | 偏大，激活噪声增加 |

### 关键发现

- **AoE 的提升不来自权重分解**：MoE + factorized $\mathbf{W}_g$（Config 3）与原始 MoE（Config 2）性能几乎相同，证明提升源于自主选择机制本身。
- **AoE 的提升不来自更多参数参与选择**：即使给传统 MoE 增大 router 到与 AoE 对等的参数量（Config 4），仍不如 AoE。
- **$d_{low} \approx d_{model}/3$ 最优**：过小（$d_{low}=64$）导致低秩近似误差大，过大（$d_{low}=512$）导致激活噪声增加。
- **AoE 天然更均衡**：即使不用 $\mathcal{L}_{aux}$，AoE 的 expert 负载分布熵（$\text{Ent}_{load}$）也高于传统 MoE + $\mathcal{L}_{aux}$。
- **AoE 选择更自信**：AoE 的选择置信熵（$\text{Ent}_{conf}$）显著低于 MoE，且从浅层到深层递减，符合"浅层做通用处理、深层做专业任务"的直觉。
- **AoE 训练更高效**：NLL loss 曲线显示 AoE 在训练过程中始终低于传统 MoE，说明 expert 学习更有效率。
- **效率开销可接受**：AoE ($d_{low}$=256) 达到传统 MoE 97% 的吞吐量，额外内存约 7GB（57.32 vs 50.61 GB）。
- **兼容多种选择策略**：AoE 与 Top-P 和 Expert-Choice 策略结合后仍优于对应的传统 MoE。

## 亮点与洞察

- **"Expert 知道自己擅长什么"的洞察极其精准**：这是一个简洁、直觉且经过实验验证的观察。在 Mixtral 8×7B 上不做任何训练，仅用激活范数选 expert 就保留 95% 性能，这个 pilot 实验非常有说服力。
- **消除 router 的思路有深远意义**：Router 作为一个与 expert 分离的模块，其决策本质上是"盲选"。AoE 将选择能力内化到 expert 自身，从根本上解决了决策-执行分离的问题。这一设计哲学可以推广到更广泛的模块化/路由式架构。
- **低秩分解一举两得**：既解决了计算效率问题（压缩预计算维度），又提供了紧凑的自评估信号（低维激活范数），技术上非常干净。
- **Expert 自评估标准的自发对齐**（Figure 4）：训练过程中同一层的 expert 会自动对齐各自的激活范数尺度，不需要额外约束，这说明方法的自组织能力很强。

## 局限与展望

- **内存开销随 expert 数量和序列长度增加**：所有 expert 都要做低秩投影并缓存结果，当 expert 数量 $n$ 很大或序列很长时，缓存 $\mathbf{C} \in \mathbb{R}^{n \times d_{low}}$ per token 的开销不可忽视
- **分布式场景下的通信模式**：传统 MoE 在分布式部署时 expert 分布在不同设备上，AoE 需要所有 expert 先计算低秩投影再比较范数，这可能改变通信模式——论文未深入讨论大规模分布式训练的实践细节
- **仅验证了基于 Llama FFN 架构**：$\mathbf{W}_g$ 分解方案针对 SiLU-gated FFN 设计，对其他 FFN 架构（如 GLU 变体）需要适配
- **$d_{low}$ 的选择依赖经验**：最优值约为 $d_{model}/3$ 是实验观察而非理论推导
- **实验规模相对有限**：最大模型 4B 参数，100B tokens。在 DeepSeek-V3（671B）或 Mixtral（47B）等真正大规模模型上表现如何尚未验证
- **未与 router 蒸馏等替代方案对比**：例如用 expert 激活范数作为 router 的训练标签（Pham et al., 2024 的思路），是否能在保留 router 的同时获得类似收益？

## 相关工作与启发

- **vs 传统 MoE（Shazeer et al.、Switch Transformer）**：传统方法用独立的 router MLP 做选择，AoE 将选择能力内化到 expert 内部。本质区别是 AoE 的选择信号来自 expert 对输入的"理解"，而非 router 的外部预测。
- **vs CompeteSMoE（Pham et al., 2024）**：CompeteSMoE 也利用 expert 输出的范数作为 router 训练标签，但它仍保留 router 且需要所有 expert 做完整计算。AoE 直接移除 router 并通过低秩分解避免完整计算。
- **vs Expert-Choice routing（Zhou et al., 2022）**：Expert-Choice 沿 token 维度做 Top-K，让 expert 选择 token 而非 token 选 expert。AoE 与 Expert-Choice 正交且兼容——AoE 改变的是选择信号的来源（从 router 到激活范数），Expert-Choice 改变的是选择维度。
- **vs Mixture-of-Depths（Raposo et al., 2024）**：MoD 动态决定 token 是否需要在某层做计算。AoE 和 MoD 可以互补——前者决定"哪个 expert 处理"，后者决定"是否需要处理"。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "expert 自主选择"的范式转换极具创新性，洞察深刻且实验支持充分
- 实验充分度: ⭐⭐⭐⭐ 消融非常详细（8个研究问题），但模型规模限于4B，缺少超大规模验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、实验通过层层设问推进、toy实验直观、图表精美
- 价值: ⭐⭐⭐⭐⭐ 提出了 MoE 架构的重要改进方向，代码开源，对后续 MoE 研究有深远影响

<!-- RELATED:START -->

## 相关论文

- [Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)
- [On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](../../NeurIPS2025/llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [UMoE: Unifying Attention and FFN with Shared Experts](../../NeurIPS2025/llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)
- [Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)

<!-- RELATED:END -->
