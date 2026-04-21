---
title: >-
  [论文解读] Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages
description: >-
  [ACL 2025][LLM/NLP][多语言模型] 提出权重共享框架，通过在编码器和解码器之间共享和交错权重，将多语言编码器高效适配为文本生成模型，在藏语、维吾尔语、哈萨克语和蒙古语四种极低资源语言上显著超越基线。
tags:
  - ACL 2025
  - LLM/NLP
  - 多语言模型
  - 低资源语言
  - 权重共享
  - 编码器-解码器
  - 中国少数民族语言
---

# Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages

**会议**: ACL 2025  
**arXiv**: [2502.10852](https://arxiv.org/abs/2502.10852)  
**代码**: [GitHub](https://github.com/asd765973346/xlm-swcm)  
**领域**: LLM/NLP  
**关键词**: 多语言模型, 低资源语言, 权重共享, 编码器-解码器, 中国少数民族语言

## 一句话总结

提出权重共享框架，通过在编码器和解码器之间共享和交错权重，将多语言编码器高效适配为文本生成模型，在藏语、维吾尔语、哈萨克语和蒙古语四种极低资源语言上显著超越基线。

## 研究背景与动机

多语言预训练模型如 XLM-R 虽然声称支持上百种语言，但对极低资源语言（如中国少数民族语言）的支持很差。更先进的 LLM（如 LLaMA、Qwen）支持的语言数量甚至更少。以藏语（1000万+使用者）、维吾尔语（1100万+）、哈萨克语（300万+）、蒙古语（700万+）为例，这些语言在主流多语言语料库（如 OSCAR）中几乎没有数据。

问题进一步复杂化的原因在于，中国少数民族语言使用的文字系统与同一语系中其他地区使用的不同（如维吾尔语在中国用阿拉伯文，在俄罗斯用西里尔文），导致现有语料库中的数据无法直接利用。

核心挑战是：**如何在极其有限的数据下，将已有的多语言编码器扩展为支持文本生成的模型？** 这促使作者提出了一种高效的编码器到编码器-解码器的适配框架。

## 方法详解

### 整体框架

基于 CINO（XLM-R 的中国少数民族语言增强版）作为编码器，通过权重共享机制构建解码器，形成编码器-解码器架构。模型命名为 XLM-SWCM（XLM-Shared Weight for Chinese Minorities），在 MC2 语料库上使用降噪自编码（DAE）和机器翻译双任务预训练，然后在下游任务上微调。

### 关键设计

**1. 双类型解码器层：CustomDecoderLayer + NormalDecoderLayer**

- **功能**: 平衡编码器预训练知识的复用与生成任务新知识的学习
- **核心思路**: CustomDecoderLayer 完全继承编码器权重，其自注意力和交叉注意力均从编码器的自注意力初始化，两个 FFN 从编码器的单个 FFN 初始化。NormalDecoderLayer 使用随机初始化权重，采用标准解码器结构。两种层交错排列——每 $X$ 个 Custom 层后插入一个 Normal 层，使 $n$ 层编码器对应 $n + \lfloor n/X \rfloor$ 层解码器
- **设计动机**: 仅复用编码器权重会导致模型过于依赖已学表示、缺乏生成能力；仅随机初始化则无法利用编码器的语义空间。交错设计实现了两者的最佳平衡

**2. 权重共享与初始化机制**

- **功能**: 最大化利用编码器预训练阶段学到的语义空间
- **核心思路**: 编码器仅有自注意力和 FFN，而解码器需要自注意力、交叉注意力和 FFN。初始化策略为：解码器的自注意力和交叉注意力均从编码器的自注意力块初始化；解码器的两个 FFN 均从编码器对应层的 FFN 初始化。这有效减少了需要从头学习的参数量
- **设计动机**: 在数据极度稀缺的条件下，从头训练解码器效率极低；通过权重复用，模型可以快速收敛并有效泛化

**3. 多任务预训练：DAE + 机器翻译**

- **功能**: 从词级完形填空平稳过渡到序列生成，同时增强跨语言迁移能力
- **核心思路**: DAE 任务使用 mBART 的去噪编码策略，帮助模型从编码器的掩码预测过渡到序列生成。机器翻译任务聚焦于中文与四种少数民族语言之间的双向翻译，增强跨语言对齐。上采样策略使用 $p_i = q_i^\alpha / \sum_j q_j^\alpha$（$\alpha = 0.3$）平衡各语言的采样比例
- **设计动机**: 单独的 DAE 无法充分建立跨语言联系，机器翻译作为辅助目标弥补了这一不足

### 损失函数 / 训练策略

预训练使用 AdamW 优化器，学习率 $1\text{e-4}$，全局 batch size 600，训练 8 个 epoch。序列最大长度 256 tokens，梯度裁剪 1.0，混合精度训练。在两块 A800 80GB GPU 上训练 92 小时。下游微调 50 epoch，batch size 200。

## 实验关键数据

### 主实验：藏语单语微调

| 模型 | 参数量 | 摘要(F1) | 阅读理解(F1) | 机器翻译(F1) |
|------|--------|---------|-------------|-------------|
| MC2-LLaMA-13B | 13B | 16.1 | 13.2 | 15.1 |
| mBART-CM | 611M | 8.6 | 7.9 | 11.5 |
| XLM-SWCM (ours) | 492M | **25.7** | **16.4** | **24.5** |

### 消融实验：组件贡献

| 移除模块 | 摘要 | 阅读理解 | 机器翻译 |
|---------|------|---------|---------|
| 完整 XLM-SWCM | **25.7** | **16.4** | **24.5** |
| 移除 MT | 25.6 | 15.1 | 20.3 |
| 移除 DAE | 22.4 | 12.2 | 18.7 |
| 移除 WS (权重共享) | 17.1 | 11.7 | 18.2 |
| 全部移除 | 15.9 | 10.8 | 16.5 |

### 关键发现

1. **权重共享是最关键组件**: 移除权重共享导致最大性能下降（摘要 F1 从 25.7 降至 17.1）
2. **492M 参数模型超越 13B**: XLM-SWCM 在所有任务上超越 MC2-LLaMA-13B，摘要 F1 高出 59%
3. **对 mBART-CM 提升巨大**: 摘要提升 198.8%，阅读理解提升 107.6%
4. **跨语言迁移优异**: 在中文微调后，藏语摘要 ROUGE-L 达到 17.1，超越所有基线
5. **Normal 层插入频率依赖数据量**: 小数据集（10K）适合大 $X$（少 Normal 层），大数据集（50K）适合小 $X$（多 Normal 层），中等数据集（20K）最佳 $X=3$

## 亮点与洞察

- 权重共享思路优雅且高效，以不到 500M 参数超越 13B 模型，对低资源语言 NLP 具有重大意义
- $X$ 值的灵活调节机制使框架可根据数据规模自适应调整解码器容量
- 跨语言迁移实验揭示了基线模型（mBART-CM 和 MC2-LLaMA）在语言切换上的系统性失败
- 编码器语义空间的可复用性远超预期，为"编码器 → 生成模型"的范式提供了有力证据

## 局限与展望

- 仅覆盖四种中国少数民族语言，受限于可用预训练模型和高质量语料
- 单语微调实验仅限于藏语（其他三种语言缺乏下游数据集）
- 翻译数据依赖 Google Translate，质量可能受限
- 未来应关注更多低资源语言的高质量数据集构建
- 编码器到生成模型的范式可推广到更多语言和任务

## 相关工作与启发

- **多语言模型**: mBART、mT5 支持多语言但未覆盖中国少数民族语言；LLaMA/Qwen 等 LLM 支持语言更少
- **MC2 语料库**: 首次为中国少数民族语言提供预训练语料，是本文的核心数据基础
- **启发**: 编码器的预训练知识可以通过巧妙的架构设计高效迁移到生成任务，这一发现对所有低资源场景下的 NLP 研究都有借鉴意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 权重共享从编码器适配到生成模型的方案简洁有效，Custom/Normal 交错设计有独创性
- **实验充分度**: ⭐⭐⭐⭐ 全面的消融实验（目标/结构/频率），但受限于单语微调仅有藏语
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，图表规范
- **价值**: ⭐⭐⭐⭐⭐ 对数千万级使用者的极低资源语言提供了可行的 NLP 解决方案

<!-- RELATED:START -->

## 相关论文

- [AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](lm_graph_search_supervision.md)
- [Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)
- [MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)

<!-- RELATED:END -->
