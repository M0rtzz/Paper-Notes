---
title: >-
  [论文解读] FreeMesh: Boosting Mesh Generation with Coordinates Merging
description: >-
  [ICML 2025][3D视觉][网格生成] 提出 Per-Token-Mesh-Entropy（PTME）度量来免训练评估网格tokenizer质量，并引入从NLP借鉴的 Rearrange & Merge Coordinates（RMC）坐标合并技术，在 MeshXL/MeshAnythingV2/EdgeRunner 三种tokenizer上实现最高21.2%的压缩率、显著增加可生成面片数和几何细节保留。
tags:
  - ICML 2025
  - 3D视觉
  - 网格生成
  - PTME
  - 坐标合并
  - BPE
  - Transformer
---

# FreeMesh: Boosting Mesh Generation with Coordinates Merging

**会议**: ICML 2025  
**arXiv**: [2505.13573](https://arxiv.org/abs/2505.13573)  
**代码**: 无  
**领域**: 3D视觉 / 网格生成  
**关键词**: 网格生成, PTME, 坐标合并, BPE, 自回归Transformer

## 一句话总结

提出 Per-Token-Mesh-Entropy（PTME）度量来免训练评估网格tokenizer质量，并引入从NLP借鉴的 Rearrange & Merge Coordinates（RMC）坐标合并技术，在 MeshXL/MeshAnythingV2/EdgeRunner 三种tokenizer上实现最高21.2%的压缩率、显著增加可生成面片数和几何细节保留。

## 研究背景与动机

**领域现状**：自回归网格生成已成为直接生成高质量三角网格的主流范式。MeshGPT 开创性地用 VQ-VAE + Transformer 生成网格；后续 MeshXL 转向坐标级直接生成，MeshAnythingV2 和 EdgeRunner 在几何维度引入压缩 tokenization，将每面片的坐标数从 9 个压缩到约 4-5 个。

**现有痛点**：(1) 缺乏免训练的tokenizer评估指标——只能通过昂贵的完整训练来比较不同tokenizer的优劣，且受随机性干扰不可控；(2) 坐标序列中存在大量高频重复模式（如相邻面共享顶点导致的坐标重复），但现有方法未利用这种统计冗余。

**核心矛盾**：自回归模型的序列长度限制了可生成面片数，而序列长度由tokenizer决定，但缺乏理论工具指导tokenizer设计——只能盲目训练来试错。

**本文目标** (1) 建立免训练的tokenizer评估理论框架；(2) 在现有tokenizer基础上进一步压缩序列长度。

**切入角度**：从信息论出发——信息量更低的序列更容易被自回归模型学习。NLP中BPE通过合并高频子词缩短序列的思想可以直接迁移到坐标序列。

**核心 idea**：信息熵×压缩率=PTME 度量tokenizer质量，BPE式坐标合并作为即插即用模块进一步压缩序列。

## 方法详解

### 整体框架

给定3D三角网格，pipeline为：选择基础tokenizer（RAW/AMT/EDR）→ 将网格序列化为1D坐标序列 → 规则化重排列（Rearrange）→ BPE合并高频坐标模式（Merge）→ 得到压缩后的token序列 → 训练自回归Transformer生成模型。推理时解码token，反向映射回坐标重建网格。整个坐标合并过程是即插即用的后处理步骤，可叠加在任意坐标级tokenizer之上。

### 关键设计

1. **Per-Token-Mesh-Entropy（PTME）度量**

    - 功能：免训练评估网格tokenizer质量的理论指标
    - 核心思路：对于一个tokenizer将网格 $\mathcal{M}$ 编码为token序列 $S$，定义 $\text{PTME} = H(S) \times CR$，其中 $H(S)=-\sum_i p_i \log p_i$ 是token序列的信息熵（统计所有token的频率），$CR = |S|/|S_{raw}|$ 是相对于原始RAW表示的压缩率。PTME越低意味着每个token承载的平均信息量越低、序列越短，因此自回归模型越容易学习。从Per-Coordinate-Mesh-Entropy(PCME)扩展而来，将基本单元从坐标推广到合并后的token
    - 设计动机：不同tokenizer产生不同长度的序列，单纯比较熵不公平。乘以压缩率后统一尺度比较。核心直觉：总信息量更低的序列更容易被序列模型拟合

2. **坐标重排列（Rearrange）**

    - 功能：将坐标排列顺序规则化，使高频模式更集中，为合并做准备
    - 核心思路：对序列化后的坐标进行规则化重排，使重复模式在序列中连续出现。观察到不同tokenizer输出的坐标排列方式不同（如RAW按 $x_1y_1z_1x_2y_2z_2x_3y_3z_3$ 排列），重排列通过调整坐标顺序（如将相邻面的共享坐标相邻放置），使BPE能更有效地发现并合并高频对
    - 设计动机：直接对未重排列的序列做BPE合并（MC方法）**无法降低PTME**——因为高频模式分散在序列各处。重排列是合并有效的**必要前提**，这是本文的关键实验发现

3. **BPE坐标合并（Merge）**

    - 功能：将高频坐标模式合并为新token，缩短序列
    - 核心思路：使用SentencePiece实现的BPE训练，从训练集坐标序列中统计最频繁的相邻token对，迭代合并为新token直到达到目标词汇表大小。例如若坐标对 $(x_i, y_i)$ 出现频繁就合并为单个token $[x_i, y_i]$。通过增大词汇表大小可压缩更多坐标对，持续降低PTME
    - 设计动机：NLP中BPE通过子词合并在词级和字符级之间取得平衡。3D网格的坐标序列与自然语言类似地存在统计冗余，BPE可以直接迁移

### 训练策略

使用标准自回归交叉熵损失训练Transformer。RMC是纯数据预处理步骤，不改变训练过程。在7-bit离散化设置（坐标量化到128级）下统一训练和测试。

## 实验关键数据

### 压缩率对比（7-bit离散化）

| Tokenizer | 原始压缩率 | +MC | +RMC | PTME变化 |
|-----------|-----------|-----|------|---------|
| RAW (MeshXL) | 100% | ~100%（无改善）| 显著降低 | 明显降低 |
| AMT (MeshAnythingV2) | ~50% | ~50% | 进一步降低 | 降低 |
| EDR (EdgeRunner) | ~45% | ~45% | **21.2%** | **最低** |

### 生成质量（Objaverse / Objaverse-XL，点云条件）

| 配置 | 最大面片数 | 几何细节 | 拓扑质量 |
|------|----------|---------|---------|
| EDR基线 | ~800 | 基线 | 基线 |
| EDR + MC | ≈800 | ≈基线 | ≈基线 |
| **EDR + RMC** | **~1600** | **显著提升** | **更好** |

### 消融实验

| 配置 | PTME变化 | 说明 |
|------|---------|------|
| MC only（无Rearrange） | 未降低或略升 | 重排列是合并有效的必要条件 |
| Rearrange only（无Merge） | 轻微降低 | 仅重排不够，需结合合并 |
| RMC（完整） | 显著降低 | 两步协同才有效 |
| 词汇表 256→1024→4096 | 持续降低 | 更多合并→更短序列→更多面 |

### 关键发现

- PTME与实际训练后的生成质量高度正相关（低PTME→高质量），验证了度量的有效性
- MC（无重排列的合并）在所有tokenizer上均失败——PTME不降反升
- RMC可将EdgeRunner的可生成最大面片数翻倍（~800→~1600），在不增加模型参数和计算预算的前提下实现

## 亮点与洞察

- **PTME是首个网格tokenizer的免训练理论评估指标**，信息熵×压缩率简洁优雅地统一了序列长度和学习难度两个维度——为tokenizer设计提供理论指引，避免了"训练才知好坏"的盲目试错
- **NLP→3D的跨领域方法迁移**：BPE子词分词在坐标序列上的成功应用展示了序列化表示下NLP技术对3D生成的通用性。Rearrange作为"使BPE生效"的必要预处理是重要洞察

## 局限与展望

- BPE合并是纯统计操作，不考虑几何语义——合并的坐标模式可能跨越不同几何结构
- 词汇表增大导致embedding参数量线性增长，存在压缩率与模型规模的权衡
- PTME假设token间独立同分布计算熵，忽略了序列中的条件依赖关系
- 仅验证了7-bit离散化，更高精度下的效果未知
- 推理时需要反向映射合并token，增加了解码复杂度

## 相关工作与启发

- **vs MeshGPT (Siddiqui et al., 2023)**：MeshGPT用VQ-VAE压缩到隐空间，FreeMesh在坐标级操作，两者正交
- **vs EdgeRunner (Tang et al., 2024a)**：EdgeRunner做几何维度压缩（减少每面片坐标数），RMC做统计维度压缩（合并高频模式），两者互补，叠加后压缩率达21.2%
- **vs NLP BPE (Sennrich et al., 2016)**：直接将BPE思想从自然语言迁移到3D网格，问题结构的同构性使迁移自然

## 评分

- 新颖性: ⭐⭐⭐⭐ PTME度量和BPE迁移到3D网格均属首创，Rearrange的必要性是非显然的洞察
- 实验充分度: ⭐⭐⭐ 三种tokenizer验证了一致性，但缺少下游应用评估和更多定量指标
- 写作质量: ⭐⭐⭐⭐ 概念清晰、图示直观，PTME的数学推导简洁
- 价值: ⭐⭐⭐⭐ 即插即用的压缩模块+免训练评估指标，对网格生成社区有直接实用价值
