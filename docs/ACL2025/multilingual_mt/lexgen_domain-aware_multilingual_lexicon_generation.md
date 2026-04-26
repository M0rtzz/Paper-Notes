---
title: >-
  [论文解读] LexGen: Domain-aware Multilingual Lexicon Generation
description: >-
  [ACL 2025][词典生成] 本文提出LexGen框架，通过在预训练多语言翻译模型的decoder中引入可学习的"领域路由"（Domain Routing）层，实现领域特定和领域通用知识的动态融合，在6种印度语言、8个领域的词典生成任务上超越了NLLB和BLICEr等基线。
tags:
  - ACL 2025
  - 词典生成
  - 领域感知
  - 多语言翻译
  - 门控路由
  - 印度语言
---

# LexGen: Domain-aware Multilingual Lexicon Generation

**会议**: ACL 2025  
**arXiv**: [2405.11200](https://arxiv.org/abs/2405.11200)  
**代码**: 无  
**领域**: NLP理解 / 多语言翻译  
**关键词**: 词典生成, 领域感知, 多语言翻译, 门控路由, 印度语言

## 一句话总结

本文提出LexGen框架，通过在预训练多语言翻译模型的decoder中引入可学习的"领域路由"（Domain Routing）层，实现领域特定和领域通用知识的动态融合，在6种印度语言、8个领域的词典生成任务上超越了NLLB和BLICEr等基线。

## 研究背景与动机

**领域现状**：词典（lexicon）生成是一个具有重要社会价值的任务，特别是对低资源语言而言。目前该领域主要有两类方法：(1) 双语词典归纳（BLI），通过词嵌入对齐来发现翻译对应关系；(2) 基于生成模型的方法，使用NMT模型直接生成翻译。预训练的多语言NMT模型（如NLLB）在通用翻译上表现优异，但在特定专业领域（如生物技术、化学）的词汇翻译中力不从心。

**现有痛点**：(1) BLI方法依赖局部上下文和共现模式，无法增强领域特定词汇的特征；(2) 预训练NMT模型在通用语料上训练，对专业术语的翻译能力有限；(3) 大语言模型（如LLaMA 2）虽然在Hindi等高资源语言上有一定效果，但对其他印度语言的支持很差；(4) 现有的领域适应方法通常需要领域内平行数据，但专业术语的平行数据极为稀缺。

**核心矛盾**：专业领域词典的翻译既需要领域特定知识（理解术语的专业含义），又需要跨领域的通用语言知识（掌握语言的基本翻译模式），如何在一个模型中平衡这两者是核心难题。

**本文目标**：设计一种领域感知的端到端翻译框架，能够在有限的训练数据下生成高质量的多语言领域特定词典，并具备泛化到未见领域和未见语言的能力。

**切入角度**：作者观察到不同领域的术语虽然在表面形式上差异显著，但底层的翻译模式（如从英语到印度语言的词素对应）存在可共享的结构。关键是设计一个机制来动态决定"这个词应该走领域特定的翻译路径还是通用路径"。

**核心 idea**：在预训练Transformer的decoder层中嵌入领域路由（Domain Routing）层，通过可学习的门控机制为每个token动态选择领域特定或领域通用的变换矩阵。

## 方法详解

### 整体框架

LexGen基于预训练的多语言Transformer NMT模型（使用Samanantar数据集预训练）。输入为英语短语（附加目标语言标签），输出为目标印度语言的翻译。在原始decoder的每个self-attention层之后插入一个Domain Routing (DR) 层，该层根据当前token的表示动态决定信息流经领域特定通道还是共享通道。

### 关键设计

1. **领域路由层（Domain Routing Layer, DR）**:

    - 功能：为每个decoder token动态选择领域特定或领域通用的变换路径
    - 核心思路：DR层维护两个权重矩阵：$W_{dom}$（领域特定）和 $W_{shared}$（跨领域共享）。对于来自上一层的表示 $f(z_l)$，通过一个门控函数 $g(z_l)$ 来加权混合两条路径：$DR(f(z_l)) = g(z_l) \cdot W_{dom} f(z_l) + (1-g(z_l)) \cdot W_{shared} f(z_l)$。门控值由两层前馈网络生成：$g(z_l) = \sigma(\text{ReLU}(z_l W_1 + b) W_2)$，学习出一个二值化的硬门控
    - 设计动机：不同的词汇需要不同程度的领域特异性。通用词（如"the"、"is"）应走共享路径，而专业术语（如"biosynthesis"）应走领域特定路径。门控机制让模型自动学会这个分配

2. **DR层的参数共享策略**:

    - 功能：防止在小数据集上过拟合
    - 核心思路：所有decoder block中的DR层共享同一组参数（$W_{dom}$, $W_{shared}$, 门控网络参数），而非每层独立学习。这大幅减少了参数量，在训练样本有限的情况下（如某些领域仅有~2000个对照对）尤为关键
    - 设计动机：词典数据集通常很小（几千条），独立参数加上多个decoder层很容易过拟合

3. **梵语词根辅助信息融合**:

    - 功能：利用印度语言与梵语的亲缘关系来提升翻译质量
    - 核心思路：由于大多数印度语言共享梵语词根（词干相似但后缀不同），作者提取Hindi翻译对应的梵语词干+词缀分解，将其拼接到英语源输入中（如"biosynthesis [SEP] जैवसंश्लेषण"）。这为模型提供了跨语言的锚点
    - 设计动机：利用语言谱系学的先验知识，在训练语言和测试语言不同（零样本跨语言）的场景下，梵语作为"桥接语言"帮助知识迁移

### 损失函数 / 训练策略

使用标准的交叉熵损失训练，标签平滑系数0.1。优化器为Adam，学习率1e-4，4000步warmup。所有预训练参数都参与微调，但DR层从随机初始化。Beam search解码，beam size=5。

## 实验关键数据

### 主实验（IDST: 域内同语言测试）

| 领域 | 指标(ChrF) | LexGen | Base Transformer | NLLB | BLICEr | LLaMA |
|------|-----------|--------|-----------------|------|--------|-------|
| 行政管理 | 6语平均 | **56.84** | 55.56 | 50.30 | 42.98 | 20.53 |
| 生物技术 | 6语平均 | **64.94** | 61.05 | 42.65 | 47.06 | 18.37 |
| 化学 | 6语平均 | **56.11** | 53.60 | 38.17 | 42.07 | 18.83 |

### 消融实验（DR层位置对比）

| 配置 | 行政管理ChrF | 生物技术ChrF | 化学ChrF |
|------|-------------|-------------|---------|
| DR层放在self-attention后 (本文) | **56.84** | **64.94** | **56.11** |
| DR层放在cross-attention后 | 49.07 | 61.97 | 54.87 |
| 仅使用共享门控层 | 56.65 | 61.72 | 54.97 |

### 零样本跨语言测试（IDDT）

| 领域 | LexGen | Base | NLLB | LLaMA |
|------|--------|------|------|-------|
| 行政管理(3未见语言) | **51.12** | 47.35 | 45.82 | 15.90 |
| 生物技术(3未见语言) | **59.87** | 55.93 | 29.74 | 17.12 |
| 化学(3未见语言) | **46.63** | 43.23 | 28.28 | 16.82 |

### 关键发现

- **DR层在专业领域优势显著**：在行政管理（训练数据与NMT预训练语料重叠较多）领域LexGen优势较小（+1.28），但在生物技术（+3.89）和化学（+2.51）领域优势明显，说明DR层在领域差异大时最有价值
- **DR层位置很重要**：放在self-attention后效果最好，放在cross-attention后在行政管理领域大幅下降（56.84→49.07），可能因为干扰了源-目标的对齐学习
- **LLaMA在低资源语言上表现很差**：除Hindi/Marathi外几乎不可用，ChrF分数远低于其他方法，说明当前LLM在真正的低资源语言上能力有限
- **梵语辅助对NLLB有效但对LexGen无效**：NLLB+梵语在多数领域显著提升，但LexGen因其预训练模型仅支持英语作为源语言，无法处理梵语输入

## 亮点与洞察

- **简洁有效的领域适应机制**：DR层所做的本质上是"每个token自己决定走哪条路"，这个门控设计非常轻量但有效。与adapter、LoRA等参数高效方法相比，DR层的独特之处在于同时支持领域特定和领域共享两条路径，并通过数据驱动学习分配。这种设计可以迁移到任何需要领域感知的NLP任务。
- **零样本跨语言泛化能力**：在从未见过测试语言的情况下（IDDT），LexGen仍然显著优于基线，说明DR层学到的领域知识具有跨语言的可迁移性。这对低资源语言的技术术语词典构建有实际意义。
- **系统性的多维度评测**：论文设计了IDST/DDST/IDDT三种评测场景，分别测试域内泛化、跨域泛化和跨语言泛化，实验设计非常严谨。

## 局限与展望

- 仅在印度语言上实验，不确定DR层在差异更大的语言对（如中英、日英）上是否同样有效
- LexGen的预训练模型不支持非英语源语言输入，导致梵语辅助信息无法被利用
- 词典数据集的规模有限（几千条），在更大数据量下DR层的优势可能减小
- 仅评估了词或短语级别的翻译，未扩展到句子级别的领域适应
- 未与最新的多语言LLM（如GPT-4、Gemma、Aya等）进行比较

## 相关工作与启发

- **vs BLICEr (Li et al., 2022)**: BLICEr通过跨编码器重排序改进BLI，但仍受限于词嵌入对齐的局限。LexGen直接在生成模型中加入领域路由，效果更好且更灵活
- **vs NLLB (Costa-jussà et al., 2022)**: NLLB是大规模多语言NMT的代表，但缺乏领域适应能力。LexGen在NLLB基础上通过DR层获得了领域感知，且参数开销很小
- **vs Mixture-of-Experts (MoE)**: DR层的设计与MoE有相似之处（门控+多路径），但更轻量——仅两条路径（领域特定vs共享），不需要大量专家模块

## 评分

- 新颖性: ⭐⭐⭐ DR层的设计是现有门控路由技术的直接应用，创新有限但组合巧妙
- 实验充分度: ⭐⭐⭐⭐ 多语言多领域多设置的全面实验，含人工评估和消融分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设置描述详细
- 价值: ⭐⭐⭐ 主要面向印度语言低资源词典生成的特定应用场景

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] LangSAMP: Language-Script Aware Multilingual Pretraining](langsamp_multilingual_pretraining.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](../../ACL2026/multilingual_mt/morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)

<!-- RELATED:END -->
