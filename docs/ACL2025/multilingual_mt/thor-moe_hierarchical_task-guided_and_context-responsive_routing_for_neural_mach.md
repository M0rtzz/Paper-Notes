---
title: >-
  [论文解读] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation
description: >-
  [ACL 2025][Mixture-of-Experts] 提出THOR-MoE框架，通过层级任务引导路由（自动预测领域/语言并生成混合任务表示来选任务级专家子集）和上下文响应路由（将全局上下文注入token表示以辅助专家选择），在多领域和多语言翻译中以更少激活参数获得显著性能提升。
tags:
  - ACL 2025
  - Mixture-of-Experts
  - 神经机器翻译
  - 层级路由
  - 上下文感知路由
  - 多领域翻译
  - 多语言翻译
---

# THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation

**会议**: ACL 2025  
**arXiv**: [2505.14173](https://arxiv.org/abs/2505.14173)  
**作者**: Yunlong Liang, Fandong Meng, Jie Zhou (腾讯微信AI)
**代码**: 未公开  
**领域**: nlp_generation  
**关键词**: Mixture-of-Experts, 神经机器翻译, 层级路由, 上下文感知路由, 多领域翻译, 多语言翻译

## 一句话总结

提出THOR-MoE框架，通过层级任务引导路由（自动预测领域/语言并生成混合任务表示来选任务级专家子集）和上下文响应路由（将全局上下文注入token表示以辅助专家选择），在多领域和多语言翻译中以更少激活参数获得显著性能提升。

## 研究背景与动机

### 问题背景
稀疏MoE架构通过条件计算在不增加推理开销的前提下扩大模型容量，已在NMT中取得显著进展。现有MoE路由方案主要分两类：(1) 引入任务特定知识（领域/语言标签）设计专门路由模块；(2) 通过减少激活专家数提升效率。

### 已有工作的不足

**依赖显式任务标签**：现有方法（如Lingual-MoE等）直接使用hard语言/领域标签来引导路由，但实际测试时这些标签往往不可用；对于code-mixed句子或跨领域文本，单一标签无法准确描述输入特性

**路由仅依赖局部token表示**：传统路由机制仅根据当前token的局部表示决定专家分配，忽略了全局上下文信息——上下文能反映每个token的难度和语境角色，对最优路由决策至关重要

**缺乏层级设计**：直接从全量专家池中为每个token选专家，未利用任务级别的天然分组特性来缩小候选范围

### 核心动机
设计一个即插即用的MoE路由框架，(a) 自动获取而非依赖给定的任务知识，用混合表示处理模糊边界；(b) 层级化先选任务级专家子集再做token级路由；(c) 将全局上下文融入token路由过程。

## 方法详解

### 整体框架
THOR-MoE在标准MoE路由之上增加两个模块，形成三层级路由流程：
1. **任务预测与混合表示生成**：自动预测输入所属领域/语言，生成软混合任务表示
2. **层级任务引导路由**：用混合任务表示从全量专家中选出任务级候选专家子集 $\mathcal{S}^t$
3. **上下文响应token路由**：将全局上下文注入token表示后，从候选子集 $\mathcal{S}^t$ 中做最终token级专家选择

该框架兼容Top-k和Top-p两种现有路由策略。

### 关键设计1：层级任务引导路由

**任务预测器**：在输入前添加[CLS]特殊token，经Transformer编码后通过MaxPooling和全连接层得到任务分布预测：

$$\mathcal{P}^t = \text{Softmax}(\mathbf{W}^p \cdot \text{MaxPooling}(\mathbf{H}^{cls}))$$

**混合任务表示**：不直接取argmax的硬标签，而是用预测概率分布对任务嵌入矩阵加权求和：

$$\mathbf{E}_p = \sum(\mathcal{P}^t \cdot \mathbf{EMB1})$$

这种软混合表示对code-mixed输入和跨领域文本有天然的容错能力——实验证明它甚至优于使用golden标签。

**层级路由**：混合任务表示输入专门的任务路由器 $g^t$，通过TopK从全量专家中选出任务级候选子集 $\mathcal{S}^t$。后续token级路由仅在 $\mathcal{S}^t$ 中进行，大幅缩小搜索空间。

### 关键设计2：上下文响应路由

在token级路由之前，用门控机制将全局上下文融入每个token的表示：

$$\mathbf{x}_i = g \odot \mathbf{x}_i + (1-g) \odot \mathbf{H}_{ctx}$$

其中 $g = \sigma([\mathbf{x}_i; \mathbf{H}_{ctx}]\mathbf{W}^g + \mathbf{b}^g)$ 是可学习门控，$\mathbf{H}_{ctx}$ 为序列中所有token的平均隐状态。在解码端，上下文随解码步动态更新（使用已生成前缀的表示）。

这使得路由器能从全局视角判断每个token的难度和角色，为其分配更合适的专家。

### 训练目标

总损失由四/五部分组成：
- $\mathcal{L}_{NMT}$：标准翻译损失
- $\mathcal{L}_{tp}$：任务预测交叉熵损失
- $\mathcal{L}_{bd}$：任务级负载均衡损失，确保不同任务均匀使用专家
- $\mathcal{L}_{bt}$：token级负载均衡损失，在候选子集内均衡
- $\mathcal{L}_{topp}$：（仅Top-p）动态路由熵约束，防止激活过多专家

## 实验关键数据

### 实验1：多领域翻译（De→En，Decoder-only架构）

基于Qwen1.5-MoE裁剪版（Trim-MoE，3.5B总参/2.3B激活参）在5个领域上微调。

| 模型 | IT | Koran | Medical | Law | Subtitles | Avg. |
|------|-----|-------|---------|-----|-----------|------|
| Dense SFT-3B | 40.65 | 20.40 | 51.40 | 54.80 | 28.33 | 39.12 |
| Trim-MoE (Top-2) | 45.10 | 22.68 | 51.84 | 57.12 | 29.02 | 41.15 |
| Trim-MoE (Top-p) | 39.39 | 19.21 | 55.67 | 60.18 | 29.21 | 40.73 |
| **THOR-MoE (Top-2)** | **46.00** | **23.35** | **55.79** | **61.06** | 28.23 | **42.89** |
| **THOR-MoE (Top-p)** | 44.63 | 22.53 | 53.58 | 58.65 | 27.99 | 41.48 |

THOR-MoE (Top-2) 比 Trim-MoE (Top-2) 平均提升 **+1.74 BLEU**，且多项显著性检验 p<0.01。

### 实验2：多语言翻译（OPUS-16, Encoder-Decoder架构）

16种语言（8高/4中/4低资源），基于Transformer-Base + 32专家。

| 模型 | En→XX Avg. | XX→En Avg. | 总Avg. |
|------|-----------|-----------|--------|
| Dense Transformer-base | 26.16 | 30.27 | 28.21 |
| ST-MoE (Top-1) | 29.09 | 33.71 | 31.40 |
| Lingual-MoE | 30.95 | 33.81 | 32.38 |
| **THOR-MoE (Top-2)** | **31.98** | **34.64** | **33.31** |
| THOR-MoE (Top-p) | 31.55 | 34.26 | 32.91 |

THOR-MoE (Top-2) 比最强基线 Lingual-MoE 在总Avg.上提升 **+0.93 BLEU**（p<0.05）。低资源语言提升最大（+1.51 vs Lingual-MoE）。

### 实验3：效率分析

| 路由策略 | IT | Koran | Medical | Law | Subtitles | 平均激活专家数 |
|---------|-----|-------|---------|-----|-----------|------------|
| Top-p（原始） | 1.87 | 1.95 | 1.82 | 1.92 | 1.77 | 1.87 |
| Top-p + 上下文 | 1.37 | 1.61 | 1.42 | 1.56 | 1.31 | **1.45** |

上下文响应路由使平均激活专家数从1.87降至1.45（减少22%），同时性能更优——上下文帮助模型更自信地路由到少量专家。

## 关键发现

1. **混合表示 > 金标签**：用预测概率加权的混合任务表示效果优于直接使用golden标签（41.48 vs 41.32 BLEU），证明软分布的容错优势
2. **层级设计 > 直接融合**：将任务知识层级化地用于先筛候选再token路由，效果显著优于直接将任务信息拼入token表示（41.48 vs 40.95）
3. **上下文使路由更高效**：引入上下文后模型用更少专家（1.45 vs 1.87）达到更好效果，训练时也更快收敛到低激活数
4. **框架通用性强**：在Top-1、Top-2、Top-p三种路由策略上一致有效，在Decoder-only和Encoder-Decoder两种架构上均成立

## 亮点与洞察

- **即插即用设计**：THOR-MoE作为模块化组件可无缝集成到任何使用Top-k或Top-p路由的MoE架构，无需修改基础模型结构
- **自动任务知识获取**：通过任务预测器自动获取领域/语言知识，摆脱了对显式标签的依赖，增强实际部署可行性
- **混合表示的优雅处理**：软概率加权的任务表示自然处理了code-mixed和跨领域模糊输入，比硬标签更鲁棒
- **上下文的双重收益**：全局上下文不仅提升路由精度（更好的专家匹配），还提升效率（更少的激活专家数），体现了"知难而精"的路由理念
- **层级路由的先验紧缩效应**：先在任务级缩小候选范围，再在token级精选，类似于贝叶斯框架中先验约束搜索空间的思路

## 局限性

- **依赖任务数先验**：设计需要预设领域/语言组数量，可扩展性受限——面对开放域或大量细粒度任务时不灵活
- **仅验证翻译任务**：所有实验限于NMT，未验证在摘要、对话等其他生成任务或判别任务上的效果
- **上下文表示过于简单**：全局上下文仅采用平均池化，未探索注意力加权等更精细的上下文聚合方式
- **额外计算开销未充分讨论**：任务预测器和门控机制引入的额外参数和计算量未做详细profiling
- **多领域实验规模较小**：多领域翻译仅涉及5个领域、单一语言对(De→En)，规模有限

## 相关工作与启发

- **Lingual-MoE (Zhao et al., 2024)**：最直接的对比方法，也采用层级语言引导+动态路由，但使用hard语言ID嵌入且不含上下文信息。THOR-MoE通过混合表示和上下文注入实现全面超越
- **Top-p路由 (Huang et al., 2024)**：动态激活专家数的路由策略，THOR-MoE在其上叠加后以更少激活参数获得更好性能
- **Hybrid-MoE (Kudugunta et al., 2021)**：编码端token路由、解码端语言路由的混合方案，性能不如统一的层级设计
- **启发**：该工作的核心思路——先粗选再精选的层级路由 + 融入全局信息——可迁移到LLM预训练阶段的MoE路由设计中，特别是多任务指令微调场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层级路由+上下文响应的组合设计新颖，混合任务表示优于golden标签是有趣发现
- 实验充分度: ⭐⭐⭐⭐ — 两种架构、两个benchmark、丰富消融，但缺乏翻译之外的验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，图表配合良好
- 价值: ⭐⭐⭐⭐ — 即插即用的MoE路由增强方案，对NMT实用价值高，但局限于翻译场景

<!-- RELATED:START -->

## 相关论文

- [Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)
- [Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation](registering_source_tokens_to_target_language_spaces_in_multilingual_neural_machi.md)

<!-- RELATED:END -->
