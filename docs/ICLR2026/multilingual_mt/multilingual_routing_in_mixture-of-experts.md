---
title: >-
  [论文解读] Multilingual Routing in Mixture-of-Experts
description: >-
  [ICLR 2026][mixture-of-experts] 系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。
tags:
  - ICLR 2026
  - mixture-of-experts
  - multilingual routing
  - cross-lingual transfer
  - expert steering
  - interpretability
---

# Multilingual Routing in Mixture-of-Experts

**会议**: ICLR 2026  
**arXiv**: [2510.04694](https://arxiv.org/abs/2510.04694)  
**作者**: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng (UCLA, Fudan University)  
**代码**: 未开源  
**领域**: LLM Efficiency / MoE可解释性与多语言  
**关键词**: mixture-of-experts, multilingual routing, cross-lingual transfer, expert steering, interpretability

## 一句话总结

系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。

## 背景与动机

1. **MoE成为主流但多语言机制不清**: Mixture-of-Experts架构是LLM扩展的核心范式，能在保持合理推理开销的同时实现巨大的参数量扩展，但其稀疏路由动态如何响应多语言数据，此前几乎没有系统研究
2. **Dense LLM已有发现但未迁移到MoE**: 已有大量工作揭示dense LLM中间层存在语言无关（language-universal）表示空间，早/晚层负责语言特异映射，但MoE的稀疏激活机制是否呈现类似层级模式尚未被探索
3. **预训练的英语中心性**: 现有MoE模型的预训练和后训练数据高度以英语为中心，虽然模型规模增长带来了隐式的多语言能力，但在多数语言上仍存在显著性能差距
4. **MoE天然适合可解释性分析**: MoE的离散专家激活机制使得分析"哪些模型组件负责哪些能力"更加直观，但此前这一优势未被充分利用于多语言分析
5. **跨语言迁移的瓶颈待揭示**: 理解MoE中多语言路由的机制，可以为改进跨语言能力迁移提供指导性insights

## 方法

### 路由散度分析（Routing Divergence Analysis）

- 使用FLoRes-200平行翻译数据集，该数据集包含200+语言的平行文本，覆盖多种主题
- 对每个非英语序列，计算序列内所有token的路由权重均值，得到专家重要性分布 $\bm{q}_i^{(\text{lang},l)}$
- 使用**熵归一化Jensen-Shannon散度**（$D_{\text{H-JS}}$）量化每层中非英语序列与英语平行序列的路由差异
- 之所以需要熵归一化，是因为路由熵在不同层之间变化很大（越深的层熵越低），直接比较JS散度会引入偏差
- 最终得到每语言每层的路由散度指标 $\text{Div}^{(\text{lang},l)}$

### 模型覆盖

实验覆盖四个代表性开源MoE LLM：
- **Qwen3-30B-A3B**: 48层，多语言能力强
- **Phi-3.5-MoE**: 32层，微软出品
- **GPT-OSS-20B**: 24层，OpenAI开源
- **OLMoE**: 老旧较小的英文模型，多语言能力差（作为对照）

四个模型在架构宽度、稀疏度、深度上各不相同，提供了多样的验证场景。

### 专家识别（Expert Identification）

- 计算每个专家在特定领域/语言数据上相对于通用基线（FLoRes英文）的**激活频率差异** $\Delta_k$
- 使用离散激活计数而非路由权重，因其更能精确识别最负责的专家
- 设置正值阈值 $\tau$，当 $\Delta_k > \tau$ 时该专家被认定为该领域/语言的特化专家
- 多语言专家定义为：对**任何一种**非英语语言满足 $\Delta_k > \tau$
- 任务专家使用GSM8K-Instruct（数学）和AlpaCare MedInstruct（医学）数据识别

### 路由干预（Routing Interventions）

**软干预（Soft Intervention）**:
$$z'_k \leftarrow z_k + \lambda \cdot s(\bm{z})$$
在softmax之前对目标专家的logit加减标准差的 $\lambda$ 倍，$|\lambda| \leq 1.0$ 效果最佳

**硬干预（Hard Intervention）**:
$$z'_k \leftarrow \max(\bm{z}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 10^{-6})$$
将目标专家logit强制设为所有专家中的最大值（激活）或最小值（抑制）

干预仅在**中间层**进行，层范围基于每个模型的路由散度U型曲线确定。

## 核心发现

### 1. U型路由散度——中间层跨语言共享

所有模型中，早期层和晚期层的路由呈现**语言特异性**，而中间层路由在不同语言之间**高度对齐**——形成清晰的U形曲线。这意味着MoE模型也像dense模型一样，在中间层学到了语言无关的表示空间，并且以更加模块化、更清晰的方式呈现。

### 2. 语言性能与路由对齐度强相关

语言理解能力（Belebele准确率）与该语言在中间层路由对英语的对齐程度之间存在**强负相关**：
- OLMoE: $r \in [-0.95, -0.80]$（极强相关）
- Qwen3和Phi-3.5-MoE: 中等到强相关
- GPT-OSS: $r \in [-0.40, -0.60]$（最弱但仍显著）

模型不理解的语言（如Bambara）无法将输入映射到中间层的共享空间，全程保持高路由散度。

### 3. 语言-任务完全功能分离

当 $\tau \geq 0.3$ 时，**零个专家**同时特化于任务和多语言——两组专家集合完全不相交。这一发现为Mahowald等人提出的LLM中"语言与思维功能解耦"假说提供了极强的实证支持：处理语言形式（语言特化专家）与处理任务内容（任务特化专家）由不同的参数组件负责。

### 4. 路由熵和一致性的语言差异

- 路由熵随层深度降低，非英语语言下降更剧烈，最后一层出现显著跳降——暗示存在少量非英语生成专家
- Token间路由一致性（Jaccard相似度）与语言资源量负相关：低资源语言token间路由更一致（依赖更少的专家）

## 实验

### 主要干预结果

| 模型 | 任务 | 目标层 | τ | 干预方式 | 专家数 | 基线 | 干预后 | 提升 |
|------|------|--------|---|----------|--------|------|--------|------|
| Qwen3-30B-A3B | MGSM | (8,35) | 0.4 | soft, λ=0.5 | 22 | 76.4% | 78.0% | +1.6% |
| Phi-3.5-MoE | MGSM | (8,17) | 0.3 | soft, λ=0.5 | 12 | 57.5% | 58.9% | +1.4% |
| GPT-OSS-20B | MGSM | (4,19) | 0.3 | hard | 9 | 68.9% | 71.5% | +2.6% |
| Qwen3-30B-A3B | MMLU医学 | (8,35) | 0.5 | hard | 23 | 68.2% | 69.1% | +0.9% |
| Phi-3.5-MoE | MMLU医学 | (8,17) | 0.25 | soft, λ=0.5 | 2 | 57.8% | 58.8% | +1.0% |
| GPT-OSS-20B | MMLU医学 | (4,19) | 0.3 | soft, λ=0.5 | 6 | 63.8% | 64.5% | +0.7% |

### 低资源语言改善更显著

- Swahili MGSM: GPT-OSS 52.4%→62.0%（**+9.6%**）
- Bengali MGSM: Phi-3.5 20.8%→23.2%（+2.4%）
- Yoruba MMLU医学: Phi-3.5 40.0%→42.9%（+2.9%）
- 低资源语言平均提升普遍高于高资源语言

### 英语性能基本不受影响

干预几乎不影响英语性能（变化幅度 <1%），偶有轻微下降，说明干预精确定位于跨语言迁移瓶颈而非损害原有能力。

### 对照和消融

- **中间层之外干预** → 性能大幅下降（早/晚层的语言特化路由被破坏）
- **激活多语言专家而非任务专家** → 性能下降（验证了语言-任务分离假设）
- **随机专家干预** → 性能下降
- **去激活（而非激活）** → 仅有损害，无正向增益
- **层范围敏感性** → 超出最优层范围哪怕几层也会导致退化，验证了路由散度可视化的实用性

## 亮点与贡献

- **首次系统揭示MoE LLM中的多语言路由动态**，发现与dense模型一致但更清晰的中间层语言无关空间
- **语言-任务完全分离**的模块性发现（$\tau \geq 0.3$时零交叉专家），是对"语言与思维功能解耦"假说的强力实证
- **推理时极简路由干预**即可一致性提升3个模型×2个任务×15+语言的多语言性能——方法简单但效果稳健
- 干预仅修改1-2个专家的top-K选择（K通常为4或8），不改变大部分路由行为
- 大量细致的消融实验（层选择、专家类型、干预强度、硬/软方式）验证了因果关系
- 路由散度可视化本身即可作为确定干预层范围的实用工具

## 局限性

1. **增益幅度有限**: 1-2%的提升虽然统计显著且跨条件一致，但绝对幅度较小
2. **需要模型特定调参**: 每个模型的最优 $\tau$、$\lambda$、目标层范围不同，需要针对性调整
3. **专家识别依赖领域数据**: 数学专家用GSM8K-Instruct识别、医学专家用MedInstruct识别，数据选择影响结果
4. **仅推理时干预**: 未探索训练时促进跨语言专家共享的方法，可能有更大潜力
5. **模型覆盖有限**: 仅4个MoE模型，更大规模（如DeepSeek-V3）或不同架构的MoE可能行为不同
6. **任务覆盖有限**: 仅测试了数学推理和医学问答两类任务

## 相关工作

- **Dense LLM多语言中间层**: Kojima/Wendler/Bandarkar(2024-2025)发现dense模型中间层的语言无关空间 → 本文在MoE中发现更清晰的模块化对应
- **跨语言表示对齐**: Kargaran/Ravisankar(2025)发现中间层对齐度与多语言性能相关 → 本文将此关系从表示空间扩展到路由空间
- **推理时干预**: Mahmoud/Lu(2025)引导dense模型向语言共享表示靠拢 → 本文在MoE路由层面实现类似效果
- **Fayyaz et al.(2026)**: 专家激活/去激活干预 → 本文发现在多语言上下文中激活任务专家有效
- **语言-思维解耦**: Mahowald et al.(2024) "functional dissociation"假说 → MoE中任务/语言专家零交叉提供最强实证之一
- **MoE多语言训练**: Zheng et al.(2025)通过最后层MoE upcycling扩展多语言能力 → 与本文分析中晚层语言特化的发现一致

<!-- RELATED:START -->

## 相关论文

- [Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](../../ACL2025/multilingual_mt/group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)
- [Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](../../ACL2025/multilingual_mt/less_but_better_efficient_multilingual_expansion.md)
- [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](../../ACL2025/multilingual_mt/thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_for_perplexity.md)

<!-- RELATED:END -->
