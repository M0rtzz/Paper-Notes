---
title: >-
  [论文解读] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
description: >-
  [ICLR 2026][EEG 信号分析] 提出 TFM-Tokenizer，首个从单通道 EEG 学习时频 motif 词表并编码为离散 token 的框架，在事件分类、癫痫检测等任务上一致提升性能，且可作为即插即用组件增强现有 EEG 基础模型。
tags:
  - ICLR 2026
  - EEG 信号分析
  - 离散化 tokenization
  - 时频 motif
  - 向量量化
  - 基础模型
---

# Tokenizing Single-Channel EEG with Time-Frequency Motif Learning

**会议**: ICLR 2026  
**arXiv**: [2502.16060](https://arxiv.org/abs/2502.16060)  
**代码**: [https://github.com/Jathurshan0330/TFM-Tokenizer](https://github.com/Jathurshan0330/TFM-Tokenizer)  
**领域**: 模型压缩  
**关键词**: EEG 信号分析, 离散化 tokenization, 时频 motif, 向量量化, 基础模型

## 一句话总结

提出 TFM-Tokenizer，首个从单通道 EEG 学习时频 motif 词表并编码为离散 token 的框架，在事件分类、癫痫检测等任务上一致提升性能，且可作为即插即用组件增强现有 EEG 基础模型。

## 研究背景与动机

- **EEG 基础模型热潮**: 受 NLP 启发，EEG 分析领域正向任务无关的基础模型范式转变
- **Tokenization 缺失**: NLP 的核心是 tokenization，但现有 EEG 基础模型仅简单将连续信号分段为短时窗口，缺乏数据驱动的词表学习
    - LaBraM 虽提出神经 tokenizer，但仅作为训练目标而非实际输入，下游推理时丢弃
- **三大挑战**:
  1. **Tokenization 粒度**: 需要在单通道级别操作以实现设备无关性
  2. **Token 分辨率**: 需要表示底层 motif（短时重复模式），而非简单时间片段
  3. **学习目标**: 需要显式融合时频信息，仅靠时域无法捕获重要的频率模式

## 方法详解

### 整体框架

两阶段设计：
1. **TFM-Tokenizer 预训练**: 单通道无监督学习时频 motif 词表
2. **下游 Transformer 训练**: 使用离散 token 序列进行掩码预训练和微调

### 关键设计 1: 双路径时频编码

**局部频谱窗口编码器 (Localized Spectral Window Encoder)**:
- 将频谱图沿频率轴分为 $P$ 个不重叠的 patch
- 每个 patch 独立投影：$e_{(i,p)} = \text{GroupNorm}(\text{GeLU}(\mathbf{W}_p \mathbf{S}_{(i,p)}))$
- 频率 Transformer 建模跨频带依赖
- **门控逐 patch 聚合**: 使用 sigmoid 门控选择性强调重要频率 patch：

$$\mathbf{E}_i^F = \text{Concat}\left[\sigma(\mathbf{W}_{g1} \mathbf{e}_{(i,p)}) \mathbf{W}_{g2} \mathbf{e}_{(i,p)}\right]$$

**时间编码器**: 将原始 EEG patch 线性投影 + GELU + GroupNorm

**时间 Transformer**: 将频率嵌入 $\mathbf{E}_i^F$ 与时间嵌入 $\mathbf{E}_i^T$ 拼接后建模长程依赖

### 关键设计 2: VQ 词表学习

使用向量量化 (VQ-VAE) 将融合嵌入映射到离散码本：

$$q(\mathbf{z}_i) = \arg\min_{\mathbf{v}_k \in \mathcal{V}} \|\mathbf{z}_i - \mathbf{v}_k\|_2^2$$

### 关键设计 3: 时频掩码预测

联合频率-时间掩码策略：
- 频率轴分组随机掩码 $M_F$ + 时间轴随机掩码 $M_T$
- 对称掩码用于数据增强

总体损失：

$$\mathcal{L}_{\text{token}} = \sum_{(f,t)} \|\mathbf{S}(f,t) - \hat{\mathbf{S}}(f,t)\|_2^2 + \alpha \sum_i \|\text{sg}[E_i] - v_i\|_2^2 + \beta \sum_i \|E_i - \text{sg}[v_i]\|_2^2$$

- 重建损失 + 码本更新（commitment loss + 指数移动平均）
- 不使用位置编码（EEG 非平稳且可能混沌）

### 下游 Transformer

- 使用 VQ 码本初始化 token 嵌入查找表
- 线性注意力 Transformer（~0.7M 参数）
- 跨通道结合通道嵌入和位置嵌入
- 掩码 token 预测预训练 + 下游任务微调

## 实验

### 主实验：TUEV 事件分类

| 模型 | 参数量 | Cohen's Kappa（单数据集） | Cohen's Kappa（多数据集） |
|------|--------|------------------------|------------------------|
| SPaRCNet | 0.79M | 0.4233 | - |
| BIOT | 3.2M | 0.4482 | - |
| BIOT⋆ | 3.2M | 0.4890 | - |
| LaBraM⋆ | ~6M | - | 0.5588 |
| **TFM-Tokenizer** | **~0.7M** | **~0.53** | **0.6189 (+11%)** |

### IIIC 癫痫分类

| 模型 | Cohen's Kappa（多数据集） |
|------|------------------------|
| LaBraM | 0.3658 |
| CBraMod | 0.4792 |
| **TFM-Tokenizer** | **0.4979 (+36% vs LaBraM)** |

### 跨设备可扩展性：耳 EEG 睡眠分期

| 设置 | TFM-Tokenizer vs 基线 |
|------|---------------------|
| 耳 EEG（非标 10-20 系统） | **+14%** |

### 与现有基础模型集成

| 基础模型 | 原始 | + TFM-Tokenizer |
|----------|------|----------------|
| BIOT | baseline | **+~4% (TUEV)** |
| LaBraM | baseline | **+~4% (TUEV)** |

### 关键发现

- TFM-Tokenizer 以 3× 少于 LaBraM 和 1.5× 少于 BIOT 的参数量达到最优性能
- 作为即插即用组件可一致性提升 BIOT 和 LaBraM 等现有基础模型
- 跨设备实验（耳 EEG）证明单通道 tokenization 具有良好的设备无关性
- Token 分析显示学到的 token 具有类判别性、频率感知性和一致性
- 门控聚合机制有效聚焦任务相关频率带

## 亮点

- **首个真正的 EEG tokenization**: 学习离散 motif 词表并直接作为下游模型输入，而非仅用作训练目标
- **设备无关设计**: 单通道操作使 tokenizer 可适应任意通道配置和设备
- **极致轻量**: ~0.7M 参数的下游 Transformer 即可达到 SOTA
- **可解释性**: 离散 token 与具体神经生理事件对应，支持时间戳级检索

## 局限性

- VQ 码本大小 $K$ 需要预设，对不同 EEG 类型可能需调整
- 目前仅在分类任务上验证，生成式任务（如 EEG 重建、跨模态翻译）未探索
- 门控聚合的频率 patch 大小和分频策略可能需要针对不同采样率调整
- 多数据集预训练的规模仍远小于 NLP 语料库，tokenizer 的上限潜力未充分挖掘
- 耳 EEG 实验仅 10 名受试者，样本量有限

## 相关工作

- **EEG 基础模型**: BIOT（段级连续 tokenization）、LaBraM（VQ tokenizer 仅用于训练目标）、BRANT、MMM
- **VQ Tokenizer**: VQ-VAE 在图像（VQGAN）和 EEG（LaBraM）上的应用
- **EEG Motif 学习**: 仅少数工作（Schäfer & Leser 2022）关注时域 motif，时频联合 motif 为首创
- **信号 tokenization**: 参考 NLP tokenization（BPE/WordPiece）的设计理念应用于连续信号

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)

<!-- RELATED:END -->
