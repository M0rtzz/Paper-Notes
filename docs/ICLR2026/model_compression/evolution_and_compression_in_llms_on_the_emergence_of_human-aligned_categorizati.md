---
title: >-
  [论文解读] Evolution and compression in LLMs: On the emergence of human-aligned categorization
description: >-
  [ICLR2026][模型压缩][information bottleneck] 通过 Information Bottleneck (IB) 框架和迭代上下文语言学习 (IICLL) 范式，证明 LLM 能够在未经 IB 目标训练的情况下，自发涌现出与人类语义分类系统高度对齐的、近最优压缩效率的类别结构。
tags:
  - ICLR2026
  - 模型压缩
  - information bottleneck
  - color naming
  - iterated learning
  - semantic categories
  - LLM alignment
---

# Evolution and compression in LLMs: On the emergence of human-aligned categorization

**会议**: ICLR2026  
**arXiv**: [2509.08093](https://arxiv.org/abs/2509.08093)  
**代码**: [infocoglab/evolution-compression-llms](https://infocoglab.github.io/evolution-compression-llms)  
**领域**: model_compression  
**关键词**: information bottleneck, color naming, iterated learning, semantic categories, LLM alignment

## 一句话总结

通过 Information Bottleneck (IB) 框架和迭代上下文语言学习 (IICLL) 范式，证明 LLM 能够在未经 IB 目标训练的情况下，自发涌现出与人类语义分类系统高度对齐的、近最优压缩效率的类别结构。

## 背景与动机

人类语义分类系统（如颜色命名）被大量证据表明遵循 Information Bottleneck (IB) 原则——在词汇的信息复杂度（complexity）和沟通准确性（accuracy）之间实现近最优的权衡。这一理论框架由 Zaslavsky et al. (2018) 提出，并在 World Color Survey (WCS) 的 110 种语言中得到了广泛验证。

然而，LLM 的训练目标是语言建模（next-token prediction），并非 IB 目标函数。这就引出了核心疑问：LLM 是否仅仅在模仿训练数据中的分类模式，还是拥有一种内在的、类似人类的归纳偏置（inductive bias），能够自发驱动高效的语义压缩？

颜色命名是认知科学中研究分类的经典领域，拥有独一无二的跨语言人类数据（WCS 数据集）和文化演化实验数据（Xu et al., 2013），因此成为评估 LLM 是否与人类对齐的理想测试平台。

## 核心问题

1. LLM 的英语颜色命名系统在 IB 效率和人类对齐度上表现如何？
2. LLM 是否仅是模仿训练数据中的模式，还是拥有真正的 IB 效率归纳偏置？
3. 这种偏置在颜色以外的语义域是否也存在？

## 方法详解

### 实验一：英语颜色命名

- 使用 WCS 标准颜色网格（330 个颜色片段）作为刺激
- 测试 39 个模型，跨 6 个模型族（Gemini, Gemma 3, Llama 3, Qwen 2.5, Olmo 2, GPT-2）
- 输入方式：文本（sRGB 坐标）+ 图像（多模态模型）
- 约束生成：Gemini 使用 controlled generation，开源模型通过 log-probability 评分
- 评价指标：IB 效率损失 $\varepsilon = \min_\beta \frac{1}{\beta}(\mathcal{F}_\beta[q] - \mathcal{F}_\beta^*)$ 和归一化信息距离 (NID) 衡量对齐度

### 实验二：迭代上下文语言学习 (IICLL)

这是本文的核心方法创新，将认知科学中的迭代学习范式（Iterated Learning）与 LLM 的上下文学习能力结合：

1. **初始化**：用随机分区初始化伪颜色命名系统，类别数 $k \in \{2, 3, 4, 5, 6, 14\}$
2. **每一代**：从上一代的语言系统 $L_{t-1}$ 中采样少量"颜色-伪标签"对作为 in-context 示例 $d_{t-1}$
3. **推理**：LLM 在上下文中学习后，对全部 330 个颜色命名，产生新系统 $L_t$
4. **迭代**：重复此过程多代，观察系统演化轨迹

关键设计：使用伪造的标签词（非英语颜色词），且不告诉模型输入是颜色，仅称为具有"特征"的刺激。这确保模型无法直接调用训练数据中的颜色知识。

### IB 理论框架

IB 目标函数为：

$$\mathcal{F}_\beta[q] = I_q(M;W) - \beta \cdot I_q(W;U)$$

其中 $I_q(M;W)$ 是复杂度（说话者含义与词汇之间的互信息），$I_q(W;U)$ 是准确性（词汇保留的世界状态信息），$\beta \geq 0$ 控制权衡。最优系统落在 IB 理论界上。

## 实验关键数据

### 英语颜色命名结果

- LLM 在复杂度和英语对齐度上差异巨大
- **模型规模和指令微调**是两个关键因素：更大的指令微调模型达到更好的对齐和 IB 效率
- Gemini 2.0 和 Gemma 3 27B (inst.) 最接近人类英语命名系统
- 令人惊讶的是，许多 SOTA 模型无法重现英语颜色命名（如 Llama 3.3 70B inst.）
- 部分模型（Olmo 2 32B inst., Qwen 2.5 VL 7B inst.）产生的系统更像 WCS 中的低资源语言而非英语
- 图像输入并不总是优于文本输入；CIELAB 坐标表现普遍差于 sRGB

### IICLL 文化演化结果

- **Gemini 2.0**：唯一能覆盖人类语言中观察到的完整复杂度范围的模型，IICLL 链收敛到与 WCS 语言和人类迭代学习数据相似的近最优 IB 解
- **Gemma 3 27B, Qwen 2.5 32B, Llama 3.3 70B**：也收敛到 IB 高效解，但限于低复杂度区域
- 所有模型约在 4 代后收敛到 IB 界附近，与人类迭代学习的动态平行
- 旋转分析（rotation analysis）证实 Gemini 的效率和对齐非平凡——随机旋转颜色映射导致显著下降

### Shepard 圆形域扩展

- 在由半径和辐条角度定义的二维 Shepard 圆形空间（64 个刺激）上测试 Gemini
- 经过 IICLL 代际传递，类别逐渐变得空间紧凑且基于两个维度区分区域
- 初步证据表明 LLM 的 IB 偏置可能具有跨域通用性

## 亮点

1. **理论-实验深度结合**：将认知科学的 IB 框架和迭代学习范式无缝迁移到 LLM 研究，方法论极具说服力
2. **IICLL 范式创新**：使用伪标签消除了训练数据模仿的混淆因素，直接探测 LLM 的内在归纳偏置
3. **大规模模型比较**：39 个模型、6 个族系的系统比较，揭示了模型规模、指令微调与 IB 效率的清晰关系
4. **跨域泛化**：Shepard 圆实验提供了颜色以外的初步泛化证据
5. **人类-AI 对齐的新视角**：表明 IB 效率可能是智能行为的一种涌现属性，无论人类还是 LLM 都未被显式训练优化该目标

## 局限与展望

1. **仅 Gemini 2.0 达到完整复杂度范围**：其他 SOTA 模型限于低复杂度解，说明 IICLL 对 in-context learning 能力要求极高，结论的通用性有待验证
2. **偏置来源不明**：IB 效率偏置究竟来自训练数据分布、指令微调还是模型架构？论文未能解耦这些因素
3. **颜色域的特殊性**：颜色在互联网文本中有丰富的数值表示（hex, RGB），LLM 可能对此域有天然优势，跨域泛化仅有 Shepard 圆的初步结果
4. **缺乏通信压力**：IICLL 仅模拟文化传递，未整合实际通信的功能性压力，与真实语言演化仍有差距
5. **评估局限于英语**：虽然使用了 WCS 数据，但 LLM 的直接评测仅在英语颜色词上进行

## 与相关工作的对比

| 工作 | 关注点 | 本文区别 |
|------|--------|----------|
| Marjieh et al. (2024) | GPT-3/4 等少数模型的颜色命名 | 39 个模型的系统比较 + IB 分析 + IICLL |
| Abdou et al. (2021) | LLM 内部颜色表征 | 聚焦 prompt 交互下的命名行为 |
| Zhu & Griffiths (2024) I-ICL | LLM 的 in-context 先验 | 扩展为 IICLL，直接复现人类迭代语言学习实验 |
| Carlsson et al. (2024) | 神经网络 agent 的 IB 高效颜色命名 | 使用 LLM 而非从头训练的 agent |
| Ren et al. (2020) NIL | 神经迭代学习中的组合性语言 | 聚焦语义压缩效率而非组合性 |

## 启发与关联

- **对 LLM 对齐研究的启示**：IB 效率作为人类-AI 对齐的一个可量化维度，比传统的 benchmark 评测更深入地捕捉语义层面的对齐
- **对模型压缩的启示**：虽归类于 model_compression，本文实际讨论的是"语义压缩"而非参数压缩，但其信息论视角（IB 原则）对理解模型如何在有限表征中编码语义有重要参考价值
- **指令微调的认知效应**：指令微调不仅提升任务性能，还可能重塑模型的语义组织方式，使其更接近人类认知结构
- **文化演化 × AI**：IICLL 为研究 LLM 中的文化演化动态提供了可扩展的实验范式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (将认知科学的 IB 框架与 LLM 结合，IICLL 范式是重要方法创新)
- 实验充分度: ⭐⭐⭐⭐ (39 模型大规模比较，但跨域泛化证据较初步)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰、理论动机充分、图示精美)
- 价值: ⭐⭐⭐⭐ (为理解 LLM 的语义组织能力提供了全新理论视角)

<!-- RELATED:START -->

## 相关论文

- [LLM DNA: Tracing Model Evolution via Functional Representations](llm_dna_tracing_model_evolution_via_functional_representations.md)
- [Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](../../AAAI2026/model_compression/coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)
- [TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](../../NeurIPS2025/model_compression/tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)
- [Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy](../../CVPR2026/model_compression/preference-aligned_lora_merging_preserving_subspace_coverage_and_addressing_dire.md)

<!-- RELATED:END -->
