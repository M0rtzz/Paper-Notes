---
title: >-
  [论文解读] Persistent Topological Features in Large Language Models
description: >-
  [ICML 2025][模型压缩][拓扑数据分析] 将拓扑数据分析中的 zigzag persistence 引入 LLM 内部表征分析，通过追踪 prompt 在各层表示空间中拓扑特征的持续演化，识别出四个处理阶段，并基于拓扑描述子提出了一种层剪枝准则，效果可比肩 SOTA 方法。
tags:
  - ICML 2025
  - 模型压缩
  - 拓扑数据分析
  - Zigzag Persistence
  - LLM内部表征
  - 层剪枝
  - 持续同调
---

# Persistent Topological Features in Large Language Models

**会议**: ICML 2025  
**arXiv**: [2410.11042](https://arxiv.org/abs/2410.11042)  
**作者**: Yuri Gardinazzi, Karthik Viswanathan, Giada Panerai, Alessio Ansuini, Alberto Cazzaniga, Matteo Biagetti (Area Science Park & 合作机构)  
**代码**: 未公开  
**领域**: model_compression  
**关键词**: 拓扑数据分析, Zigzag Persistence, LLM内部表征, 层剪枝, 持续同调  

## 一句话总结

将拓扑数据分析中的 zigzag persistence 引入 LLM 内部表征分析，通过追踪 prompt 在各层表示空间中拓扑特征的持续演化，识别出四个处理阶段，并基于拓扑描述子提出了一种层剪枝准则，效果可比肩 SOTA 方法。

## 研究背景与动机

### 领域现状
大语言模型在各类 NLP 任务上表现优异，但其黑箱特性使得可解释性和透明性成为核心关切。同时，模型的庞大规模消耗大量计算资源，迫切需要在不显著损失性能的前提下压缩模型。研究者通常从模型内部表征入手，试图理解各层的功能分工。

### 已有工作的不足
- **几何方法的局限**：利用内在维度（intrinsic dimension）等概念刻画各层表征流形的方法虽然揭示了语义知识在中间层而非最终层涌现的现象，但这些方法本质上是**逐层静态分析**，无法直接追踪表征在跨层传播过程中的动态变化
- **传统 TDA 的不足**：标准持续同调（persistent homology）可以捕捉单个点云的多尺度拓扑特征（连通分量、环、空洞等），但无法处理**随时间（层）动态演化的点云序列**
- **逐层分析的碎片化**：现有方法通常对每层独立评估后再聚合结果，丢失了跨层的演化信息，无法从系统整体视角理解模型运作

### 核心矛盾
LLM 的内部表征本质上是一组随层动态演化的点云——每个 prompt 对应一个点，在表示空间中逐层被变换。然而，现有分析工具要么只能逐层静态分析，要么无法追踪拓扑特征在层间的**出生、持续与消亡**轨迹。

### 本文切入角度
作者观察到 LLM 各层表征可以自然地视为一个离散动力系统中随时间演化的点云序列，因此引入 TDA 中专门处理时变数据的工具——**zigzag persistence**——来追踪这些拓扑特征的完整演化路径。

### 核心 idea
用 zigzag persistence 将 LLM 各层表征视为时变点云序列，完整追踪 $p$ 维拓扑特征在模型深度方向上的持续演化，从而获得系统级动态视角。

## 方法详解

### 整体框架

输入：一组 prompts 送入预训练 LLM → 提取各层隐藏表征（每层一个点云）→ 构建跨层 zigzag 复形序列 → 计算 zigzag persistence → 提取拓扑描述子 → 用于分析/剪枝。

整个 pipeline 分为三大阶段：(1) 表征提取与预处理；(2) 基于 k-NN 的 zigzag persistence 计算；(3) 拓扑描述子提取与下游应用。

### 关键设计

1. **层表征作为时变点云**

    - 功能：将 LLM 的第 $l$ 层输出视为一个点云 $X_l = \{x_l^{(1)}, \ldots, x_l^{(n)}\}$，其中 $n$ 为 prompt 数量
    - 核心思路：传统做法对每层独立计算拓扑特征，再聚合。本文将 $X_0, X_1, \ldots, X_L$ 视为一个时间序列（层号=时间），利用 zigzag persistence 直接捕捉拓扑特征在层间的传递与变化
    - 设计动机：这种"层即时间"的解读使得可以使用 zigzag persistence 的完整理论工具，而不是手工设计跨层追踪方法。这是该工具首次被系统化应用于 LLM 内部表征

2. **基于 k-近邻的过滤（k-NN Filtration）**

    - 功能：在每一层的点云上，用 k-近邻图构建简单复形（simplicial complex），将连续的距离阈值过滤替换为离散的 k 值增长
    - 核心思路：给定 $k$ 值，构建 k-NN 图 $G_k(X_l)$，其中点 $x_i$ 与其 $k$ 个最近邻相连。随着 $k$ 增大，图变得越来越密集，拓扑特征（连通分量消亡、环出现等）依次发生。这提供了一种对数据密度自适应的多尺度过滤
    - 设计动机：相比基于距离阈值的 Vietoris-Rips 复形，k-NN 过滤对数据的局部密度变化更鲁棒，且计算效率更高。这是本文在 zigzag persistence 应用中的技术创新之一

3. **Zigzag Persistence 计算**

    - 功能：在相邻层的复形之间建立 zigzag 连接，计算跨层持续同调
    - 核心思路：对于层序列 $l=0, 1, \ldots, L$，构建 zigzag diagram：
    $K_0 \hookrightarrow K_{0,1} \hookleftarrow K_1 \hookrightarrow K_{1,2} \hookleftarrow K_2 \hookrightarrow \cdots$
      其中 $K_l$ 是第 $l$ 层的复形，$K_{l,l+1}$ 是连接相邻层的"桥梁"复形。zigzag persistence 追踪 $p$ 维同调类（$H_p$）在这个序列中的出生层 $b$ 和消亡层 $d$，生成持续图 $(b, d)$ 对的集合
    - 设计动机：传统 persistence 只能处理单调递增/递减的过滤序列，但层间拓扑变化是非单调的（复形可能变大也可能变小）。zigzag persistence 的"之字形"结构天然适应这种非单调演化

4. **拓扑描述子（Topological Descriptors）**

    - 功能：从 zigzag persistence 的持续图中提取可解释的统计量，用于量化分析
    - 核心思路：定义若干描述子来度量拓扑特征的行为，包括：
        - **特征寿命分布**：$(d - b)$ 的统计分布，反映拓扑结构的稳定性
        - **出生/消亡密度**：各层的特征出生率和消亡率，指示哪些层在进行剧烈的拓扑重组
        - **持续特征数量**：在某一层存活的拓扑特征总数，反映表征结构的复杂度
    - 设计动机：裸的持续图不便于比较和分析，这些描述子将拓扑信息压缩为可解释的统计量，便于跨模型、跨数据集比较

### Prompt 处理的四阶段模型

通过上述拓扑描述子，作者在多个模型和数据集上一致性地识别出 LLM 处理 prompt 的四个阶段：

| 阶段 | 层区间 | 拓扑特征行为 | 解读 |
|------|--------|-------------|------|
| Phase 1: 初始重排 | 浅层 (前 ~15%) | 出生率高、消亡率高、特征寿命短 | prompt 在表示空间中快速重新排列位置关系 |
| Phase 2: 稳定中间层 | 中间层 (~15%-65%) | 出生率低、消亡率低、长寿命特征多 | prompt 间建立稳定的拓扑关系，语义结构形成 |
| Phase 3: 过渡精炼 | 中后层 (~65%-85%) | 出生率和消亡率逐渐上升 | 模型精炼已建立的语义关系 |
| Phase 4: 最终重排 | 深层 (后 ~15%) | 再次出现剧烈拓扑变化 | 为输出层做准备，重新调整表征结构 |

这一四阶段发现与此前基于内在维度的研究（中间层语义涌现）相呼应，但提供了更丰富的动态视角。

### 层剪枝准则

基于四阶段发现，作者提出拓扑层剪枝准则：

- **核心思想**：Phase 2 (稳定中间层) 中的层对拓扑结构贡献最冗余——这些层之间的拓扑变化最小，因此去除部分层对模型整体功能影响最小
- **剪枝策略**：计算相邻层间拓扑描述子的变化量，优先移除变化量最小的层
- **优势**：该准则不需要下游任务的 loss 信号或梯度信息，纯粹基于表征的拓扑结构，属于无监督剪枝方法

## 实验关键数据

### 主实验：层剪枝性能对比

| 模型 | 方法 | 剪枝比例 | 下游任务平均准确率保留 | 特点 |
|------|------|---------|---------------------|------|
| LLaMA-2-7B | 无剪枝 (baseline) | 0% | 100% | 原始模型 |
| LLaMA-2-7B | ShortGPT (BI值) | 27% | ~92-95% | 基于 Block Influence 的层重要性评估 |
| LLaMA-2-7B | 本文 (拓扑) | 27% | ~92-95% | 基于 zigzag persistence 描述子 |
| LLaMA-2-7B | 随机剪枝 | 27% | ~75-85% | 随机移除层 |
| Mistral-7B | ShortGPT | 25% | ~90-94% | SOTA 方法 |
| Mistral-7B | 本文 (拓扑) | 25% | ~90-94% | 拓扑方法可比 |

本文方法在剪枝效果上与 ShortGPT 等 SOTA 方法可比，但其优势在于：(1) 不需要 calibration data 的 loss 信号；(2) 提供对模型结构的可解释理解。

### 拓扑描述子跨模型/数据集一致性

| 模型 | 数据集 | Phase 1 层范围 | Phase 2 层范围 | Phase 3 层范围 | Phase 4 层范围 |
|------|-------|---------------|---------------|---------------|---------------|
| LLaMA-2-7B | WikiText | 0-4 | 5-20 | 21-26 | 27-31 |
| LLaMA-2-7B | C4 | 0-4 | 5-19 | 20-26 | 27-31 |
| Mistral-7B | WikiText | 0-4 | 5-21 | 22-27 | 28-31 |
| Phi-2 | WikiText | 0-4 | 5-20 | 21-26 | 27-31 |

不同模型和数据集上，四阶段的定性结构高度一致，表明这是 Transformer 架构的一种普适拓扑特性。

### 关键发现

- **Phase 2 的稳定性是普适的**：在所有测试的模型（LLaMA-2, Mistral, Phi-2）和数据集（WikiText, C4 等）上，中间层始终表现出最稳定的拓扑结构，说明 Transformer 有固有的"语义稳定带"
- **拓扑描述子对 k 值选择鲁棒**：k-NN 过滤中 $k$ 的具体选择影响数值但不影响定性结论，描述子在 $k = 5 \sim 30$ 范围内保持稳定
- **$H_0$（连通分量）和 $H_1$（环）提供互补信息**：$H_0$ 更多反映 prompt 间的聚类结构变化，$H_1$ 反映更高阶的环形拓扑关系；两者的四阶段划分高度一致
- **浅层和深层是拓扑活跃区**：相比中间稳定带，第一层和最后几层的拓扑重组最为剧烈，这解释了为什么层剪枝通常避免移除首尾层

## 亮点与洞察

- **"层即时间"的概念映射极为自然**：将 LLM 各层类比为时间序列的快照，从而将 zigzag persistence 这一针对时变数据的成熟数学工具引入 LLM 分析。这一概念桥梁简洁而有力，将两个原本不相交的领域连接起来
- **四阶段模型具有可解释性和实用性的双重价值**：不仅提供了理解 Transformer 工作机制的新视角（初始编码→语义形成→精炼→输出准备），还直接指导了层剪枝策略的设计
- **无监督剪枝的新范式**：基于拓扑的层剪枝完全不依赖下游任务或 loss 信息，是一种真正的结构分析方法。这一思路可迁移到其他需要识别冗余模块的场景（如 attention head pruning、MoE expert pruning）
- **跨模型的普适性发现**：四阶段结构在不同架构上的一致性暗示这可能是 Transformer 训练后天然形成的结构特征，值得进一步从理论角度探索其成因

## 局限与展望

- **仅在 7B-level 模型上验证**：未涉及更大规模模型（如 70B、405B），四阶段结论是否在更大模型上成立尚不确定
- **剪枝后缺乏微调恢复实验**：仅评估了直接剪枝后的性能，未探索剪枝+少量微调的恢复潜力，与 SOTA 方法的完整对比不够充分
- **计算成本未充分讨论**：zigzag persistence 的计算复杂度随 prompt 数量和层数增长较快，在大规模应用场景下的可扩展性需要评估
- **未探索 token 级别分析**：当前以 prompt 为粒度构建点云，未考虑 token 级别的拓扑演化，后者可能揭示更精细的模型行为
- **四阶段划分的定量标准不够明确**：阶段边界的确定依赖于视觉观察描述子曲线，缺乏自动化的阶段检测算法

## 相关工作与启发

- **vs ShortGPT (层剪枝)**：ShortGPT 使用 Block Influence (BI) 值衡量层重要性，基于隐藏状态的余弦相似度变化。本文方法从拓扑角度出发，不依赖余弦相似度这一线性度量，能捕捉更丰富的非线性结构变化。两者在剪枝效果上可比，但本文的拓扑视角提供了更深的可解释性
- **vs 内在维度方法 (Ansuini et al.)**：内在维度方法发现中间层的语义涌现现象，但只提供标量描述。本文的拓扑描述子提供了多维度的动态信息（出生率、消亡率、寿命分布等），是一种更丰富的表征分析框架
- **vs Betti number 方法 (Rieck et al.)**：已有工作观察到 Betti 数在同一架构的不同数据集上保持稳定且随深度递减。本文的 zigzag persistence 不仅捕捉每层的 Betti 数，还追踪同一拓扑特征在跨层的持续与消亡，提供了时间维度的信息
- **vs CKA/CCA 等表征相似度方法**：CKA 等方法度量层间表征的整体相似度，但丢失了拓扑结构信息。本文方法与 CKA 互补，可联合使用以获得更全面的层间分析

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 zigzag persistence 系统化应用于 LLM 内部表征分析，"层即时间"的概念映射极为自然且强大
- 实验充分度: ⭐⭐⭐⭐ — 在多个模型和数据集上验证了描述子的一致性，但剪枝实验的对比深度可以加强
- 写作质量: ⭐⭐⭐⭐ — 将复杂的拓扑数学框架讲解得较为清晰，数学严谨性与可读性平衡较好
- 价值: ⭐⭐⭐⭐ — 拓扑视角为LLM理解和压缩开辟了新方向，四阶段发现具有启发性，但实际应用价值仍需进一步验证

<!-- RELATED:START -->

## 相关论文

- [Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](../../ACL2025/model_compression/language_specific_features.md)
- [Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)
- [Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)

<!-- RELATED:END -->
