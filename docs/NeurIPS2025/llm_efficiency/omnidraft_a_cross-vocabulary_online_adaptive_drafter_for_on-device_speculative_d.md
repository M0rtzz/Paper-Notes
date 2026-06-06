---
title: >-
  [论文解读] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding
description: >-
  [NeurIPS 2025][LLM效率][推测解码] 提出 OmniDraft 框架，通过在线 n-gram 缓存实现跨词表推测解码、混合蒸馏损失在线对齐草稿模型与目标模型、并结合自适应起草长度控制…
tags:
  - "NeurIPS 2025"
  - "LLM效率"
  - "推测解码"
  - "跨词表"
  - "在线蒸馏"
  - "自适应起草"
  - "端侧推理"
---

# OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2507.02659](https://arxiv.org/abs/2507.02659)  
**代码**: 暂无  
**领域**: LLM效率  
**关键词**: 推测解码, 跨词表, 在线蒸馏, 自适应起草, 端侧推理

## 一句话总结

提出 OmniDraft 框架，通过在线 n-gram 缓存实现跨词表推测解码、混合蒸馏损失在线对齐草稿模型与目标模型、并结合自适应起草长度控制，使单个轻量 Llama-68M 模型可为 Vicuna-7B、Qwen2-7B、Llama3-8B 等不同目标模型提供推测解码加速（1.5-2x）。

## 研究背景与动机

推测解码（Speculative Decoding）通过小模型（草稿模型）预测多个后续 token，再由大模型（目标模型）一次性验证来加速 LLM 推理。然而目前面临两个核心难题：

**草稿与目标模型的紧耦合**：现有方法要求草稿模型与目标模型来自同一模型家族（如都是 Llama 系列），共享相同的分词器和词表。一旦目标模型更换为其他家族（如 Qwen），草稿模型就无法使用。

**在线部署场景的动态需求**：用户在端侧使用时可能切换不同目标模型，且期望延迟随使用时间逐渐改善。

现有工作 UAG 提出了词表交集映射，但只处理直接映射的 token，无法解决"假拒绝"问题——草稿模型提出的多个子 token 在目标词表中对应一个合并 token 时会被错误拒绝。此外，离线蒸馏对齐方法假设目标模型固定，无法适应动态切换场景。

**核心 idea**：构建一个"one drafter for all"范式——通过 n-gram 缓存解决跨词表映射、通过在线蒸馏实现动态对齐、通过自适应起草控制效率。

## 方法详解

### 整体框架

OmniDraft 包含三个核心组件：（1）跨词表 n-gram 缓存用于草稿/目标词表间的token翻译；（2）混合蒸馏损失用于在线对齐草稿模型；（3）自适应起草头用于动态调整提议长度。整个流程：草稿模型生成 token → n-gram 缓存翻译到目标词表空间 → 目标模型验证 → 接受/拒绝结果反馈用于更新缓存和蒸馏草稿模型。

### 关键设计

1. **跨词表 N-gram 缓存**：核心思路是维护一个缓存 $\mathcal{C} = \{(t_i, [d_j^i]_{j=1:n})\}$，记录目标 token $t_i$ 与草稿 token 序列 $[d_1^i, d_2^i, \cdots, d_n^i]$ 之间的映射关系。在提议阶段，扫描草稿 token 序列并查找 n-gram 缓存进行合并映射，概率计算为：

    $q'(t_i) = \begin{cases} q(d_i), & \text{直接映射} \\ \prod_j q(d_j^i), & \text{n-gram 映射} \end{cases}$

   对于修正阶段的残差分布计算，需要在整个目标词表上定义 $q'$，对前缀子 token 做概率调整：$q'(d_1^i) = q(d_1^i) - \prod_j q(d_j^i)$，确保概率质量的正确分配。缓存在推理过程中实时更新——每当出现新的未见映射实例就加入。设计动机：相比 UAG 仅处理词表交集，n-gram 缓存能处理合并 token 的情况，避免"假拒绝"，提高接受率。

2. **跨词表混合蒸馏损失**：在线蒸馏分为两部分——对直接映射 token 使用反向 KL 散度以获得丰富的监督信号，对 n-gram token 使用负对数似然（NLL）因为只有可靠的点概率估计。总损失：

    $\mathcal{L}_{\text{cross\_vocab\_distill}}(\theta) = \mathcal{L}_{\text{DM}}(\theta) + \lambda \mathcal{L}_{\text{N-gram}}(\theta)$

   其中 $\mathcal{L}_{\text{DM}}$ 对直接映射 token 计算 KL 散度，$\mathcal{L}_{\text{N-gram}}$ 对 n-gram token 计算 NLL。$\lambda$ 可设为固定超参或动态权重（如目标模型对该 n-gram 的验证概率）。该设计使草稿模型能在在线推理过程中持续与（可能变化的）目标模型对齐。

3. **在线自适应起草**：使用轻量头网络 $f_\phi$ 预测当前提议 token 的接受率。通过累积拒绝概率控制是否提前终止提议：

    $P(\exists 1 \leq i \leq k, \text{s.t. } y_i \text{ rejected}) > \gamma \Rightarrow \text{exit}$

   提出两种训练变体：**联合训练**（蒸馏 + 自适应头同步更新）和**交替训练**（自适应头多次更新/蒸馏一次更新，使用更大 buffer 缓解分布漂移）。

### 损失函数 / 训练策略

- 蒸馏采用在策略（on-policy）数据，即草稿模型自身生成的数据
- 固定 $\lambda = 0.2$ 对所有任务/实验
- LoRA 微调作为轻量替代方案支持动态适配器切换
- 自适应头使用加权 BCE 损失，以接受率 $\min(1, p/q)$ 为标签

## 实验关键数据

### 主实验：跨词表在线蒸馏

| 目标模型 | 方法 | GSM8K Acc/Speed | MBPP+HE Acc/Speed | Alpaca Acc/Speed | XSum Acc/Speed |
|---|---|---|---|---|---|
| Llama3-8B | SpD_DM (baseline) | 0.10 / 0.94x | 0.09 / 1.03x | 0.09 / 0.96x | 0.11 / 0.91x |
| Llama3-8B | $\mathcal{L}_{\text{DM}}$ + $\lambda\mathcal{L}_{\text{N-gram}}$ | **0.42 / 1.70x** | **0.27 / 1.33x** | **0.20 / 1.30x** | **0.24 / 1.24x** |
| Qwen2-7B | SpD_DM (baseline) | 0.14 / 1.04x | 0.09 / 0.91x | 0.13 / 1.01x | 0.12 / 0.96x |
| Qwen2-7B | $\mathcal{L}_{\text{DM}}$ + $\lambda\mathcal{L}_{\text{N-gram}}$ | **0.37 / 1.61x** | **0.26 / 1.36x** | **0.20 / 1.30x** | **0.22 / 1.22x** |

### 消融实验：自适应起草（Vicuna-7B 目标）

| 方法 | GSM8K Acc/Speed | MBPP+HE Acc/Speed | Alpaca Acc/Speed | XSum Acc/Speed |
|---|---|---|---|---|
| SpD (vanilla) | 0.21 / 1.44x | 0.14 / 1.22x | 0.20 / 1.44x | 0.20 / 1.42x |
| Distill Only | 0.42 / 2.20x | 0.35 / 1.92x | 0.25 / 1.57x | 0.23 / 1.53x |
| Joint Distill+Adapt | **0.61 / 2.08x** | **0.51 / 1.91x** | **0.44 / 1.61x** | **0.42 / 1.59x** |
| Interleaved Distill+Adapt | 0.52 / **2.15x** | 0.48 / **1.94x** | 0.41 / 1.60x | 0.38 / 1.58x |

### 关键发现

- N-gram 缓存即使不训练也能带来显著提升（cache hit 0.87），配合蒸馏效果最佳
- 缓存大小很小（1-5 MB），适合端侧部署
- 框架可扩展到更大目标模型（Qwen2.5-32B），加速可达 2.05x
- 交替训练变体在加速上略优于联合训练，但联合训练接受率更高
- LoRA 微调性能接近全参微调，支持多目标模型动态切换

## 亮点与洞察

- "One drafter for all" 范式极具实际价值——端侧只需部署一个 68M 的草稿模型即可服务所有目标 LLM
- N-gram 缓存是优雅的工程设计，将跨词表映射问题转化为在线缓存查找
- 混合蒸馏损失对直接映射和 n-gram token 采用不同损失函数，体现了对问题结构的深入理解

## 局限与展望

- 在线适应仅一次遍历数据流，对全新数据可能不稳定
- 尚未解决特殊 token（如多模态 token）的跨词表映射
- 自适应起草头在线训练不够稳定，可能低估最优提议长度
- 缓存没有淘汰策略，内存受限设备需要优化

## 相关工作与启发

- 与 UAG 相比，n-gram 缓存从词表交集扩展到多对一映射，解决"假拒绝"
- 与 OSD（Online Speculative Decoding）相比，增加了跨词表能力
- 启发：端侧推理场景下，轻量+通用+可适应的设计比重量级但专用的方案更有价值

## 评分

- 新颖性: ⭐⭐⭐⭐ N-gram 缓存解决跨词表是新颖贡献，但自适应起草借鉴 SpecDec++
- 实验充分度: ⭐⭐⭐⭐ 多任务多目标模型，消融完整，但缺少与更多基线对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，公式推导详细，图示直观
- 价值: ⭐⭐⭐⭐⭐ 端侧通用草稿模型是重要实际需求，框架完整且可落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)
- [\[ICML 2025\] DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding](../../ICML2025/llm_efficiency/dssd_efficient_edge-device_llm_deployment_and_collaborative_inference_via_distri.md)
- [\[NeurIPS 2025\] MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](../../ACL2025/llm_efficiency/sam_decoding_speculative_decoding_via_suffix_automaton.md)

</div>

<!-- RELATED:END -->
