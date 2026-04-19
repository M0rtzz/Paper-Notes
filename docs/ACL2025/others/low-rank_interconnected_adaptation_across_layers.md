---
title: >-
  [论文解读] Low-Rank Interconnected Adaptation across Layers
description: >-
  [ACL 2025][LoRA] 提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。
tags:
  - ACL 2025
  - LoRA
  - 参数高效微调
  - 跨层适配
  - 混合专家
  - 高秩更新
---

# Low-Rank Interconnected Adaptation across Layers

**会议**: ACL 2025  
**arXiv**: [2407.09946](https://arxiv.org/abs/2407.09946)  
**代码**: [有](https://github.com/) (GitHub 链接在论文中标注)  
**领域**: NLP / 参数高效微调  
**关键词**: LoRA, 参数高效微调, 跨层适配, 混合专家, 高秩更新

## 一句话总结

提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。

## 研究背景与动机

LoRA 是最流行的参数高效微调（PEFT）方法，通过低秩投影矩阵 A 和 B 近似权重更新 ΔW = AB。但 LoRA 存在根本性限制：**每层的 A 和 B 紧密耦合**，且参数预算均匀分配到每层，导致每层的权重更新秩受限。

核心问题：能否在相同参数预算下，实现更动态、更具表现力的高秩权重更新？

观察发现：LoRA 为每层分配相同预算，忽略了层的重要性差异。如果减少适配器数量但增大其秩，就可以在总参数不变的情况下获得更高秩的更新。关键在于如何让少量大秩适配器有效服务于所有层。

## 方法详解

### 整体框架

Lily 将传统 LoRA 的层内 A-B 紧耦合解构为跨层的互联结构：

- **A 适配器**（下投影）：在相邻层间局部共享，数量少于模型层数
- **B 适配器**（上投影）：全局共享，所有层都可以访问所有 B 专家
- **路由器 R**：数据依赖的门控机制，决定每层使用哪些 B 专家及其权重

### 关键设计

#### 1. 下投影与选择性权重分配

**功能**：用共享的 A 将输入投影到低维空间，然后通过路由器动态选择 B 专家的组合权重。

**核心思路**：

- 输入 x 经过局部共享的 A 得到低维表示 $x' = xA$
- 路由器 $R \in \mathbb{R}^{N_e \times d}$ 基于 $x'$ 计算 B 专家的权重分布：

$$S = \text{softmax}\left(\sum_{i=1}^{N}(x'R^T)_i\right)$$

**设计动机**：路由器使得 A-B 的连接是数据驱动的，不同输入会激活不同的 B 专家组合，避免 B 专家行为趋同，增强表达能力。

#### 2. 加权专家混合与上投影

**功能**：将多个 B 专家的输出按路由权重混合，得到最终的权重更新。

**核心思路**：数学等价但高效的实现——先混合 B 再投影，避免对每个 B 分别计算：

$$x_\Delta = x'\left(\sum_{i=1}^{N_e} S_i \cdot B^i\right)$$

最终输出：$y = xW_0 + s \cdot x_\Delta$

**设计动机**：由于 $S_i$ 是标量，先加权混合 B 矩阵再做矩阵乘法，计算量与单个 LoRA 相当，但秩可以大幅提升。

#### 3. 参数效率与高秩更新

**关键洞察**：传统 LoRA 每层一对 (A, B)，每个秩为 r。Lily 用少量共享适配器（如 2 个 A、4 个 B），每个秩可以设得更大（如 32），在总参数更少的情况下实现更高的有效更新秩。

### 损失函数 / 训练策略

沿用各基线任务的标准训练策略，Lily 本身不引入额外损失项。缩放因子 s 控制适配更新对原始权重的影响幅度。

## 实验关键数据

### 主实验（表格）

**常识推理（LLaMA3-8B，8 个任务平均准确率）**：

| 方法 | 参数量 | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
|------|--------|-------|------|------|-----------|------------|-------|-------|------|------|
| LoRA | 56M | 70.8 | 85.2 | 79.9 | 91.7 | 84.3 | 84.2 | 71.2 | 79.0 | 80.8 |
| MiLoRA | 56.6M | 68.8 | 86.7 | 77.2 | 92.9 | 85.6 | 86.8 | 75.5 | 81.8 | 81.9 |
| **Lily** | **1.2M** | 72.9 | 85.6 | 77.8 | 92.7 | 83.3 | 89.7 | 77.6 | 82.8 | **82.8** |

注意：Lily 仅 **1.2M** 参数（LoRA 的 1/46），平均准确率反超 LoRA 2 个百分点。

**自然语言理解（RoBERTa-Base，GLUE 6 个任务）**：

| 方法 | 参数量 | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Avg. |
|------|--------|-------|------|------|------|-----|-------|------|
| LoRA | 0.3M | 94.8 | 89.8 | 63.3 | 92.9 | 78.2 | 91.5 | 85.1 |
| AdaLoRA | 0.3M | 94.5 | 88.7 | 62.0 | 93.1 | 81.0 | 90.5 | 85.0 |
| **Lily** | **0.3M** | **95.0** | **90.2** | **66.0** | 92.5 | **81.6** | 90.8 | **86.0** |

### 消融实验（表格）

**Falcon-Mamba-7B 常识推理**：

| 方法 | 参数量 | Avg. |
|------|--------|------|
| LoRA (3.7M) | 3.7M | 32.7 |
| Lily (Δ + in) | 3.7M | 57.0 |
| Lily (in) | 3.3M | 59.5 |

Lily 在 Mamba 架构上也大幅超越 LoRA，验证了跨架构的通用性。

### 关键发现

1. **参数效率极致**：在 LLaMA3-8B 上，Lily 用 1.2M 参数超越 56M 参数的 LoRA，参数效率提升 46 倍
2. **跨架构通用**：在 Transformer（LLaMA3、RoBERTa）、Mamba（Falcon-Mamba）和扩散模型（SDXL）上均显著优于 LoRA
3. **跨模态有效**：NLU、常识推理、图像生成、视觉适配（VTAB-1K）均覆盖
4. **路由机制关键**：数据依赖的路由防止 B 专家行为退化为相同，确保了多样化的知识组合
5. **高秩更新本身是收益的关键来源**：减少适配器数量、增大秩，比增加适配器数量更有效

## 亮点与洞察

- **核心洞察深刻**：LoRA 的瓶颈不在于方法本身，而在于参数预算的分配方式——均匀分配导致每层秩受限
- **设计简洁高效**：Lily 不引入额外损失、不修改预训练权重、不增加推理延迟（B 可预先混合）
- **MoE 思想的精妙应用**：将 B 视为专家是自然的，加权混合在标量级别完成，避免了传统 MoE 的路由开销
- **消除冗余**：传统 LoRA 每层一对 AB 有大量冗余，Lily 证明共享+互联可以大幅减少冗余

## 局限与展望

1. 路由器增加了少量参数和计算开销，对极端资源受限场景可能不适用
2. 共享 A 的最佳分组策略（哪些层共享一个 A）需要启发式设计，缺乏自动搜索机制
3. B 专家数量的选择（Ne）对性能有影响但缺乏理论指导
4. 论文中图像生成实验仅展示定性结果，缺乏定量指标（如 FID、CLIP Score）
5. 未与更新型的 PEFT 方法（如 DoRA、GaLore）进行比较

## 相关工作与启发

- **LoRA 系列改进**：PiSSA（主子空间初始化）、MiLoRA（次要成分初始化）、AdaLoRA（自适应秩分配）等都关注秩的利用，但都保持层内 A-B 紧耦合
- **MoE + PEFT**：MoLORA、MOLA 将整个 LoRA 视为专家，而 Lily 将 A 和 B 解耦为不同层次的专家
- **HydraLoRA**：并发工作，也探索非对称设计，但仅在单层内不对称，Lily 是模型全局的跨层互联

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 跨层解耦 A/B 并互联的思路非常新颖，从根本上改变了 LoRA 的参数分配范式
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 NLU/推理/生成/视觉多模态多架构，但图像生成缺定量评测
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法公式完整，图示直观（Fig.1 很好地解释了核心思想）
- **价值**: ⭐⭐⭐⭐⭐ — 极致的参数效率和跨架构通用性使其有很高的实用价值，有望成为新一代 PEFT 基线

<!-- RELATED:START -->

## 相关论文

- [MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)
- [MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection](macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)
- [Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)

<!-- RELATED:END -->
