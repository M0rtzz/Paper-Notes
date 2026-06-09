---
title: >-
  [论文解读] Multilingual Routing in Mixture-of-Experts
description: >-
  [ICLR 2026][多语言/翻译][mixture-of-experts] 系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。
tags:
  - "ICLR 2026"
  - "多语言/翻译"
  - "mixture-of-experts"
  - "multilingual routing"
  - "cross-lingual transfer"
  - "expert steering"
  - "interpretability"
---

# Multilingual Routing in Mixture-of-Experts

**会议**: ICLR 2026  
**arXiv**: [2510.04694](https://arxiv.org/abs/2510.04694)  
**作者**: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng (UCLA, Fudan University)  
**代码**: 未开源  
**领域**: 多语言翻译  
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

### 整体框架

这项工作先用一套路由散度指标把MoE模型每一层、每种语言的专家激活模式量化出来，定位出"哪些层、哪些专家"在跨语言上行为一致或分化；再用激活频率差异挑出语言特化专家和任务特化专家两组；最后在中间层做一次极简的推理时路由干预，强行激活英语任务专家，从而提升非英语性能。所有分析都横跨Qwen3-30B-A3B（48层）、Phi-3.5-MoE（32层）、GPT-OSS-20B（24层）以及作为多语言能力对照的小模型OLMoE，四者在宽度、稀疏度、深度上各不相同，保证结论不是单一模型的偶然现象。

### 关键设计

**1. 路由散度分析：量化每层的跨语言路由差异。** 要研究多语言路由，第一步得有一个能跨层、跨语言比较的尺度。作者用FLoRes-200平行翻译语料（200+语言、内容对齐），对每个非英语序列把序列内所有token的路由权重取均值，得到专家重要性分布 $\bm{q}_i^{(\text{lang},l)}$，再用**熵归一化Jensen-Shannon散度** $D_{\text{H-JS}}$ 度量该层非英语序列与对应英语平行序列之间的路由差异，得到每语言每层的散度 $\text{Div}^{(\text{lang},l)}$。之所以要做熵归一化，是因为路由熵本身随层深变化剧烈（越深的层熵越低），不归一化直接比JS散度会让深层的差异被系统性低估、引入偏差。这条指标既是后续所有发现的度量基础，也直接给出了"该在哪些层做干预"的可视化依据。

**2. 专家识别：用激活频率差挑出语言专家和任务专家。** 干预之前要先知道哪个专家负责什么。作者计算每个专家在特定领域或语言数据上、相对于通用基线（FLoRes英文）的**激活频率差异** $\Delta_k$，并刻意用离散的激活计数而非连续路由权重——因为计数更能精确锁定"最常被选中"的那批专家。给定正阈值 $\tau$，凡 $\Delta_k > \tau$ 的专家即被判为该领域/语言的特化专家；多语言专家放宽为只要对**任一**非英语语言满足 $\Delta_k > \tau$ 即可，任务专家则分别用GSM8K-Instruct（数学）和AlpaCare MedInstruct（医学）数据识别。阈值 $\tau$ 同时充当严格度旋钮：调高它能逼出真正高度特化的专家，这也正是后文"语言与任务专家零交叉"现象的判定条件。

**3. 路由干预：中间层激活英语任务专家。** 拿到目标专家后，干预本身极轻量，只动top-K里的一两个专家。软干预在softmax之前给目标专家的logit加上标准差的 $\lambda$ 倍，$z'_k \leftarrow z_k + \lambda \cdot s(\bm{z})$，实测 $|\lambda| \leq 1.0$ 时最稳；硬干预则直接把目标logit钉到所有专家的最大值（强制激活）或最小值（强制抑制），即 $z'_k \leftarrow \max(\bm{z}) + \varepsilon,\ \varepsilon \sim \mathcal{N}(0, 10^{-6})$。关键约束是干预只在**中间层**进行，具体层范围由该模型路由散度的U型曲线读出——因为只有中间层承载跨语言共享的语义空间，在早/晚层动手反而会破坏语言特异的映射。正因为命中的是跨语言迁移的瓶颈而非通用能力，干预才能在几乎不损英语性能的前提下持续抬高非英语表现。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](../../ACL2025/multilingual_mt/group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)
- [\[ACL 2025\] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](../../ACL2025/multilingual_mt/less_but_better_efficient_multilingual_expansion.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](../../ACL2026/multilingual_mt/no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](../../ACL2026/multilingual_mt/routelmt_learned_sample_routing_for_hybrid_llm_translation_deployment.md)

</div>

<!-- RELATED:END -->
