---
title: >-
  [论文解读] Language Models Resist Alignment: Evidence From Data Compression
description: >-
  [ACL 2025 (Best Paper Award)][模型压缩][elasticity] 本文从压缩理论视角提出LLM的"弹性"(elasticity)概念，证明模型在受到微调扰动时压缩率变化与数据集大小成反比——因为预训练数据远大于对齐数据，对齐效果被优先"遗忘"，这从信息论角度根本性地解释了为什么LLM对齐如此脆弱。
tags:
  - ACL 2025 (Best Paper Award)
  - 模型压缩
  - elasticity
  - inverse alignment
  - compression theory
  - alignment fragility
  - Hooke's Law analogy
---

# Language Models Resist Alignment: Evidence From Data Compression

**会议**: ACL 2025 (Best Paper Award)  
**arXiv**: [2406.06144](https://arxiv.org/abs/2406.06144)  
**代码**: [pku-lm-resist-alignment.github.io](https://pku-lm-resist-alignment.github.io)  
**领域**: AI安全 / LLM对齐 / 信息论  
**关键词**: elasticity, inverse alignment, compression theory, alignment fragility, Hooke's Law analogy  

## 一句话总结

本文从压缩理论视角提出LLM的"弹性"(elasticity)概念，证明模型在受到微调扰动时压缩率变化与数据集大小成反比——因为预训练数据远大于对齐数据，对齐效果被优先"遗忘"，这从信息论角度根本性地解释了为什么LLM对齐如此脆弱。

## 研究背景与动机

**领域现状**：当前LLM对齐的主流做法是通过SFT、RLHF、DPO等方法，在预训练模型基础上用相对少量的高质量数据进行微调，使模型行为符合人类意图和安全规范。这些方法在实践中取得了显著效果，GPT-4、Llama 2等模型都经过了精心的安全对齐。

**现有痛点**：大量研究表明对齐的效果极其脆弱。Yang et al. (2023) 发现仅需少量恶意微调就能让安全模型重新变得不安全；Qi et al. (2024) 甚至发现在非恶意数据上微调也可能破坏安全机制。更令人担忧的是，Hubinger et al. (2024) 展示了"沉睡特工"现象——模型可以在安全训练后保留隐藏的有害行为。

**核心矛盾**：对齐数据量（通常数千到数万条）与预训练数据量（通常数万亿 tokens）之间存在数量级的差距。这种悬殊的数据量差异意味着什么？Qi et al. (2024) 提出了"浅层安全对齐"的概念，认为对齐未渗透到模型深层表示，但这只是现象描述，缺乏从根本机制上的理论解释。

**本文目标** (1) 对齐微调产生的效果到底是深层还是表面的？(2) 是否存在某种内在机制使LLM抵触对齐？(3) 如果存在，这个机制如何使"逆向对齐"成为可能？(4) 模型规模和预训练数据量如何影响这种抵触？

**切入角度**：作者从"语言建模即数据压缩"这一经典等价关系出发——最小化训练损失等价于最小化压缩率。如果把训练和对齐过程建模为对不同数据集的联合压缩，那么数据量差异会如何影响各数据集的压缩率变化？这个角度有希望，因为压缩理论提供了成熟的数学工具来分析这种非对称性。

**核心 idea**：通过将LLM训练建模为数据压缩过程，证明微调扰动下各数据集压缩率的变化与其大小成反比，因此预训练数据的"硬度"远大于对齐数据，导致模型本质上抵触对齐。

## 方法详解

### 整体框架

论文的方法论框架分为三层：(1) **建模层**——将LLM的训练和对齐过程等价建模为数据压缩问题，引入Token Tree和压缩协议；(2) **理论层**——在Pareto分布假设下推导出弹性定理(Theorem 4.2)，证明压缩率变化与数据集大小的反比关系；(3) **验证层**——通过多模型、多算法、多规模的实验系统验证抵抗性(resistance)和回弹性(rebound)两个弹性子现象。整体输入是预训练数据集 $\mathcal{D}_p$、对齐数据集 $\mathcal{D}_a$ 和扰动数据集 $\mathcal{D}_t$，输出是关于各数据集压缩率变化规律的理论和实验结论。

### 关键设计

1. **Token Tree 建模与压缩协议**:

    - 功能：为LLM的训练过程建立精确的压缩理论形式化框架
    - 核心思路：将数据集的所有可能响应表示为Token Tree $\mathcal{T}$，每个节点有 0/1 子节点和 EOS 叶节点，叶节点权重代表对应响应的概率。模型训练等价于学习树的节点权重。由于有限参数的模型无法精确建模任意深度的节点权重，假设可精确建模的深度 $d$ 与模型规模单调递增。压缩协议分两步：先将Token Tree裁剪到深度 $d$，再用 Huffman 编码压缩裁剪后的树。定义压缩率 $\gamma_{p_\theta}^{\mathcal{D}_i}$ 为压缩后编码长度与原始长度之比，最小化训练损失等价于最小化压缩率
    - 设计动机：传统的训练损失分析难以揭示不同数据集之间的竞争关系，而压缩视角天然支持多数据集联合分析——联合数据集的节点权重 $p_l^{\mathcal{D}} = \sum_i p_l^{\mathcal{D}_i} |\mathcal{D}_i| / \sum_i |\mathcal{D}_i|$ 明确体现了数据量的加权效应

2. **弹性定理 (Theorem 4.2) 的推导**:

    - 功能：从数学上严格证明微调扰动下压缩率变化与数据集大小成反比
    - 核心思路：定义归一化压缩率 $\gamma_{p_\theta}^{\mathcal{D}_i/\mathcal{D}} = \gamma_{p_\theta}^{\mathcal{D}_i} - \log M$（$M$ 为裁剪后叶节点数），引入质量分布(mass distribution)的概念将响应分布的熵转化为叶节点概率随机变量的熵。在假设质量分布服从 Pareto 分布（有 Zipf 定律支持）的条件下，推导出当扰动数据量 $|\mathcal{D}_t|$ 增加时，$d\gamma_{p_\theta}^{\mathcal{D}_a/\mathcal{D}} / dl = \Theta(-k \cdot d\gamma_{p_\theta}^{\mathcal{D}_p/\mathcal{D}} / dl)$，其中 $k = |\mathcal{D}_p|/|\mathcal{D}_a|$。即对齐数据集的压缩率变化速度是预训练数据集的 $k$ 倍（$k$ 通常为 $10^3 \sim 10^6$ 量级）
    - 设计动机：仅从直觉上认为"数据量大更重要"是不够的，需要严格的数学推导才能量化这种差异的具体程度。反比关系意味着对齐效果在扰动下的退化速度比预训练效果快若干个数量级，这是对齐脆弱性的根本原因

3. **胡克定律类比框架**:

    - 功能：为弹性定理提供直觉可理解的物理类比，并识别弹性不变量
    - 核心思路：将LLM的弹性类比为串联弹簧系统。数据集大小 $|\mathcal{D}_i|$ 对应弹簧常数 $k_i$，KL散度变化 $\Delta D_{KL}(\mathcal{P}_{p_\theta} \| \mathcal{P}_{\mathcal{D}_i})$ 对应弹簧形变 $\Delta l_i$。弹性力 $F \propto |\mathcal{D}_i| \cdot \Delta D_{KL}(\mathcal{P}_{p_\theta} \| \mathcal{P}_{\mathcal{D}_i})$ 对应胡克定律 $F = k \cdot \Delta l$。在串联弹簧系统中，同样外力下刚度小的弹簧形变大——对应数据量小的对齐数据集在扰动下变化更大
    - 设计动机：弹性定理的数学形式虽然严格，但物理类比使其变得直觉化。串联弹簧模型还揭示了一个重要的弹性不变量——$|\mathcal{D}_i| \cdot \Delta D_{KL}$ 在不同数据集间为常数，这为后续实验验证提供了可检验的具体预测

### 损失函数 / 训练策略

本文不提出新的训练方法，而是分析现有对齐训练的失效机制。实验中使用标准的SFT损失 $\mathcal{L}_{SFT}(\theta; \mathcal{D}) = -\mathbb{E}_{(x,y) \sim \mathcal{D}}[\log p_\theta(y|x)]$ 进行正向和逆向对齐，并在验证回弹性时额外测试了RLHF/PPO、DPO、KTO、SimPO等多种对齐算法，证明弹性现象与具体对齐算法无关。

## 实验关键数据

### 主实验

**抵抗性验证（Table 1）**：在Alpaca/TruthfulQA/Beavertails数据集上，比较正向对齐和逆向对齐的训练损失。

| 数据集 | 基座模型 | $\theta_2 \to \theta_1$ vs $\theta_1 \to \theta_2$ | $\theta_3 \to \theta_2$ vs $\theta_2 \to \theta_3$ | $\theta_3 \to \theta_1$ vs $\theta_1 \to \theta_3$ |
|--------|----------|------|------|------|
| Alpaca | Llama2-7B | 0.159↓ vs 0.202↑ | 0.195↓ vs 0.214↑ | 0.167↓ vs 0.235↑ |
| Alpaca | Llama2-13B | 0.177↓ vs 0.196↑ | 0.215↓ vs 0.241↑ | 0.184↓ vs 0.235↑ |
| Alpaca | Llama3-8B | 0.254↓ vs 0.257↑ | 0.227↓ vs 0.323↑ | 0.234↓ vs 0.323↑ |

所有实验中逆向对齐的训练损失均低于正向对齐，验证了抵抗性的存在。

### 消融实验

**KL散度回弹验证（Table 3）**：用不同数量安全数据对齐后，使KL散度降到 $\epsilon = 0.01$ 以下所需的不安全数据量。

| 基座模型 | 安全数据1000条 | 安全数据2000条 | 安全数据5000条 | 安全数据10000条 |
|----------|:---:|:---:|:---:|:---:|
| Llama2-7B 对齐后KL | 0.21 | 0.22 | 0.26 | 0.27 |
| Llama2-7B 逆转所需不安全数据 | 961 | 844 | 801 | 729 |
| Gemma-2B 对齐后KL | 0.18 | 0.21 | 0.24 | 0.25 |
| Gemma-2B 逆转所需不安全数据 | 923 | 853 | 709 | 598 |

用更多安全数据对齐的模型，反而需要更少的不安全数据就能逆转——完美符合弹性定理的预测。

### 关键发现

- **弹性与模型规模正相关**：Qwen 0.5B → 4B → 7B，模型越大回弹越快。这类似于一种"逆向缩放定律"——模型能力越强，对齐反而越容易被逆转
- **弹性与预训练数据量正相关**：TinyLlama系列从0.1T到3.0T，预训练数据越多弹性越强。在0.1T时几乎观察不到弹性，但在0.5T时弹性现象显著出现，暗示存在弹性涌现的临界数据量
- **弹性与对齐算法无关**：SFT、PPO、DPO、KTO、SimPO均观察到一致的回弹现象，说明弹性是模型的固有属性而非特定算法的缺陷
- **弹性具有双向对称性**：反向实验（先用负面数据训练，再用正面数据逆转）观察到同样的弹性，排除了实验设置偏差
- **回弹呈两阶段模式**：初始阶段性能快速下降（远离预训练分布→快速回弹），后续阶段下降减缓（接近预训练分布→趋于稳定），与弹簧模型的预测一致

## 亮点与洞察

- **物理类比极为精妙**：串联弹簧的类比让复杂的信息论推导变得直觉可理解——预训练数据是"硬弹簧"（高刚度），对齐数据是"软弹簧"（低刚度），同样的外力导致软弹簧形变远大于硬弹簧。这种跨学科的类比降低了理解门槛，也启发了新的研究方向
- **理论预测与实验验证的完美闭环**：弹性定理预测"变化量与数据集大小成反比"，实验在多个维度（模型规模、数据量、算法种类）精确验证了这一预测，而且发现了弹性涌现的临界点（0.1T~0.5T），这种理论驱动实验发现的范式值得借鉴
- **揭示了对齐研究的根本性困境**：如果弹性是由预训练-对齐数据量差异这一不可回避的结构性因素导致的，那么仅靠改进对齐算法可能无法根本解决问题。这个洞察对整个AI安全领域有战略性指导意义——要么大幅增加对齐数据规模使其与预训练数据可比，要么寻找完全不同的对齐范式

## 局限与展望

- **Pareto分布假设未经直接验证**：理论推导的核心假设（Assumption A.7）是Token Tree叶节点的质量分布服从Pareto分布，虽有Zipf定律间接支持，但对于具体的LLM训练数据是否严格满足这一假设尚未直接检验
- **未覆盖完整预训练生命周期**：受计算成本限制，实验未能在从零开始的完整预训练+对齐流程上系统验证弹性，而是使用已有的预训练模型进行后续实验
- **缓解措施停留在讨论层面**：Appendix C.2提出了基于弹性定理定制数据配比的方向来缓解逆向对齐风险，但未实际实现和验证这些方案
- **弹性临界点量化不足**：虽然发现弹性在0.1T~0.5T之间涌现，但由于缺乏细粒度的预训练数据样本，无法精确确定临界点
- **未探索多模态设置**：弹性是否同样存在于视觉-语言模型等多模态场景尚不清楚，这对多模态对齐的安全性评估至关重要

## 相关工作与启发

- **vs Qi et al. (2024) "浅层安全对齐"**: 他们提出对齐应超越表面token层面渗透到模型内部机制，本文从理论层面解释了为什么对齐注定是浅层的——预训练和对齐数据量的数量级差异是结构性根因，而非算法设计不当。本文优势在于提供了定量的理论框架
- **vs Hubinger et al. (2024) "Sleeper Agents"**: 他们从对抗性训练角度展示了对齐后模型仍可保留隐藏的有害行为，本文从信息论角度给出了更基础的解释——弹性是LLM的固有属性，不需要对抗性设计，仅靠数据量差异就足以使对齐脆弱
- **vs Wei et al. (2024) 基于权重归因的分析**: 他们从权重视角分离了安全关键区域和功能关键区域，是一种局部层面的分析；本文从全局的数据压缩视角给出了互补的解释框架，两者可以结合——弹性理论解释"为什么"，权重归因分析"在哪里"
- 弹性理论对开源LLM生态有深远影响：如果精心对齐的模型可以被极低成本逆转，开源社区的攻防平衡将面临根本性挑战，需要发展"不可微调逆转"的对齐方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从压缩理论视角给出对齐脆弱性的形式化解释，弹性概念及胡克定律类比极具创意
- 实验充分度: ⭐⭐⭐⭐⭐ 跨4家族模型(Llama2/3, Gemma, Qwen)、5种对齐算法(SFT/PPO/DPO/KTO/SimPO)、多数据规模的全面验证
- 写作质量: ⭐⭐⭐⭐ 理论部分需要一定数学基础，但物理类比和清晰的实验设计大大降低了理解门槛
- 价值: ⭐⭐⭐⭐⭐ ACL 2025 Best Paper，揭示了AI对齐面临的根本性挑战，对安全研究方向有战略指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](../../ICML2025/model_compression/from_language_models_over_tokens_to_language_models_over_characters.md)

<!-- RELATED:END -->
