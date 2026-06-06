---
title: >-
  [论文解读] Steering Pretrained Drafters during Speculative Decoding
description: >-
  [AAAI 2026][模型压缩][推测解码] 提出 SD²，通过从验证器隐藏状态中提取转向向量（steering vector）并注入预训练 drafter 的 MLP 层，实现推测解码中 drafter-verifier 的动态对齐，在标准采样下将被接受 token 数提升高达 35%…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "推测解码"
  - "动态对齐"
  - "转向向量"
  - "LLM推理加速"
  - "预训练drafter"
---

# Steering Pretrained Drafters during Speculative Decoding

**会议**: AAAI 2026  
**arXiv**: [2511.09844](https://arxiv.org/abs/2511.09844)  
**代码**: [github.com/ETH-DISCO/SD-square](https://github.com/ETH-DISCO/SD-square)  
**领域**: 模型压缩  
**关键词**: 推测解码, 动态对齐, 转向向量, LLM推理加速, 预训练drafter

## 一句话总结

提出 SD²，通过从验证器隐藏状态中提取转向向量（steering vector）并注入预训练 drafter 的 MLP 层，实现推测解码中 drafter-verifier 的动态对齐，在标准采样下将被接受 token 数提升高达 35%，同时计算开销可忽略不计。

## 研究背景与动机

推测解码（Speculative Decoding）通过"快速草拟 + 并行验证"的范式来加速 LLM 推理。其核心瓶颈在于 **drafter-verifier 的不对齐**：drafter 生成的候选 token 越多被验证器拒绝，加速效果越差。

目前存在两大类 drafter 方案：

**依赖型 drafter**（如 EAGLE、Medusa）：小型投机头直接挂载在验证器上，虽然速度快但生成质量有限，当验证延迟主导总延迟时效率下降。

**独立型 drafter**（如预训练小模型）：因其独立训练的强大生成能力，接受率更高，但缺乏与验证器的动态对齐手段。

现有离线对齐方法（如蒸馏）虽能提升一定接受率，但存在在非训练分布（OOD）数据上退化的问题。作者观察到 **LLM 的中间表示隐含编码了未来多步 token 的信息**，因此可以从验证器的隐藏状态中提取这种预测信号，动态引导 drafter 生成，从而在保留预训练 drafter 泛化能力的同时提升对齐质量。

## 方法详解

### 整体框架

SD² 遵循标准推测解码流程：drafter 草拟 k 个候选 token → 验证器并行验证 → 拒绝/接受。关键创新是在验证步骤中**额外生成一个转向向量**，并将其注入 drafter 以引导下一轮生成。

整个方法包含三个阶段：**验证阶段**（生成转向向量）、**草拟阶段**（被转向向量引导的自回归生成）、**训练阶段**（学习转向机制参数）。

### 关键设计

1. **转向向量提取（Steering Vector Extraction）**

   在验证阶段，从验证器的三个不同层（高层、中层、低层）提取隐藏状态 $h_t, m_t, l_t$，通过线性投影生成转向向量：

    $g_t = W_{hml}[h_t, m_t, l_t]^\top$

   层的选择参考 EAGLE-2，分别取第 3 层、$L/2$ 层和 $L-2$ 层。这种多层信息融合使转向向量能同时捕获高层语义、中层特征和低层模式。

2. **MLP 偏置注入（Bias Injection in MLP）**

   转向向量通过线性映射 $W_s$ 转化为偏置，注入 drafter 每一层 MLP 的上投影（up-projection）中，修改 SwiGLU 激活为：

    $a^{(l)}_{t+i}, g_t \mapsto W_d((W_u a^{(l)}_{t+i} + W_s g_t) \odot \sigma(W_g a^{(l)}_{t+i}))$

   这一设计的优点是：$W_s g_t$ 对草拟位置 $i$ 不变，只需在每次草拟开始时计算一次；同时不破坏 KV-Cache 兼容性；偏置通过门控机制（gating）自然地调节控制强度。

3. **初始化策略**

   $W_s$ 初始化为零矩阵（确保初始时转向不改变 drafter 行为），$W_{hml}$ 初始化为使输出等于 $h_t + m_t + l_t$ 的投影。转向向量后接 LayerNorm 以稳定训练。

### 损失函数 / 训练策略

训练使用验证器 $\pi_V$ 在合成数据上生成的概率分布作为目标（合成数据比真实数据更能反映验证器行为）。

- 随机采样偏移量 $\delta \in [1, k]$，模拟草拟第 $\delta$ 个 token 的场景
- 使用 KL 散度作为训练损失：$D_{KL}(\pi_V(\cdot|x_{1:t-1}) \| \pi_D(\cdot|x_{1:t-1}, g_{t-\delta}))$
- **同时微调 drafter 参数和转向机制参数**，验证器全程冻结
- 训练 6 个 epoch + 1 个 epoch 的 ShareGPT 数据微调
- 使用 AdamW 优化器和余弦学习率调度

## 实验关键数据

### 主实验

在 4 对 verifier-drafter 组合、5 个数据集上进行测试（k=8），指标为 Block Efficiency（τ，每块接受 token 数）和 Speedup（α，相对预训练 drafter 的加速比）。

| Verifier & Drafter | 方法 | τ (T=1 均值) | α (T=1 均值) | τ (T=0 均值) | α (T=0 均值) |
|---|---|---|---|---|---|
| Vicuna 13B + Llama 160M | Pretrained | 1.88 | 1.00 | 2.36 | 1.00 |
| | Distilled | 2.45 | 1.32 | 2.86 | 1.24 |
| | **SD²** | **2.96** | **1.61** | **3.27** | **1.43** |
| Qwen3 14B + Qwen3 0.6B | Pretrained | 3.86 | 1.00 | 4.03 | 1.00 |
| | Distilled | 3.97 | 1.05 | 4.21 | 1.06 |
| | **SD²** | **4.26** | **1.11** | **4.49** | **1.12** |
| Llama 3.1 8B + Llama 3.2 1B | Pretrained | 4.91 | 1.00 | 5.12 | 1.00 |
| | Distilled | 4.78 | 0.97 | — | — |
| | **SD²** | **5.00** | **1.00** | **5.31** | **1.02** |

### 消融实验

| 配置 | Block Efficiency 变化 | 说明 |
|------|---------|------|
| SD² (MLP 内偏置 + 解冻 drafter) | 最优 | 最终采用方案 |
| MLP 后偏置 | 略低 | 控制力不足 |
| 条件偏置 (基于 $g_t$ 和 $f^{(l-1)}$) | 略低 | 更多参数但无额外增益 |
| SD² + 冻结 drafter | 比 pretrained +100% | 说明转向向量本身有价值 |
| SD² + 解冻 drafter | 最优 | 转向+微调协同效果最佳 |
| 随机偏移 vs 块偏移 | 差异很小 | 随机偏移略优 |

### 关键发现

- 对于**容量差距大**的组合（Vicuna 13B + Llama 160M），SD² 在标准采样下平均提升 **61% 吞吐量**和 **57% Block Efficiency**
- 即便在已经高度对齐的组合（Llama 3.1 8B + Llama 3.2 1B）上，SD² 也不会降低性能
- 蒸馏在分布外数据上经常退化（如 GSM8K、HumanEval 上低于 pretrained），而 SD² 始终保持或超越 pretrained
- 转向向量本身（冻结 drafter）就能将接受 token 数提升 100%，说明验证器隐藏状态包含丰富的未来预测信息
- SD² 在长序列生成中保持稳定的优势，不随位置增加而衰减

## 亮点与洞察

- **设计精妙的轻量级干预**：只通过在已有 MLP 的上投影中加入偏置来实现转向，避免了额外的自注意力计算，开销可忽略
- **初始化策略巧妙**：$W_s = 0$ 确保训练初期不干扰原始 drafter 行为
- **可追加到现有架构**：无需改变预训练过程，可后置添加到任何使用 SwiGLU 的预训练 drafter 上
- 揭示了一个重要现象：**LLM 中间层隐含编码了多步未来 token 信息**，这一发现具有广泛的研究价值

## 局限与展望

- 性能依赖训练数据分布质量，分布外效果虽优于蒸馏但仍有下降
- 对已充分对齐的 drafter（如 Llama 3.1 家族）提升有限
- 需要访问验证器隐藏状态，不适用于黑盒远程验证器场景
- 仅在 SwiGLU 激活函数上验证，其他激活函数的适用性待探索
- 未探索专为动态转向设计的新 drafter 训练方案
- 未扩展到树状验证等更复杂的推测解码范式

## 相关工作与启发

- **EAGLE-3** 是本文主要对比的依赖型 drafter，通过直接拼接验证器的隐藏状态来引导轻量头；SD² 的转向方式更加模块化和可追加
- **DistillSpec** 提出的离线蒸馏是本文的主要基线；SD² 的动态转向在分布外场景的鲁棒性明显优于蒸馏
- **激活转向（Activation Steering）**文献启发了本文的设计；不过之前的工作都是静态且面向可解释性的，SD² 首次将其应用于推测解码的动态场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将激活转向引入推测解码是新颖的融合
- 实验充分度: ⭐⭐⭐⭐⭐ — 4 对模型、5 个数据集、多种采样模式、详细消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、公式严谨
- 价值: ⭐⭐⭐⭐ — 实用性强，可追加到现有系统，提供显著加速

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SSSD: Simply-Scalable Speculative Decoding](../../ACL2026/model_compression/sssd_simply-scalable_speculative_decoding.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](../../ICML2026/model_compression/lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](../../ICML2026/model_compression/speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)

</div>

<!-- RELATED:END -->
