---
title: >-
  [论文解读] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis
description: >-
  [AAAI 2026][医学图像][Single-cell RNA-seq] 提出 PanFoMa，一种融合 Transformer 局部建模与 Mamba 全局整合的轻量级混合神经网络，用于泛癌单细胞转录组表示学习…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "Single-cell RNA-seq"
  - "Pan-Cancer"
  - "Transformer"
  - "foundation model"
  - "benchmark"
---

# PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis

**会议**: AAAI 2026  
**arXiv**: [2512.03111](https://arxiv.org/abs/2512.03111)  
**代码**: [GitHub](https://github.com/Xiaoshui-Huang/PanFoMa)  
**领域**: 计算生物学 / 单细胞转录组学 / 基础模型  
**关键词**: Single-cell RNA-seq, Pan-Cancer, Transformer-Mamba Hybrid, foundation model, benchmark

## 一句话总结

提出 PanFoMa，一种融合 Transformer 局部建模与 Mamba 全局整合的轻量级混合神经网络，用于泛癌单细胞转录组表示学习；同时构建了覆盖 33 种癌症亚型、350 万+ 细胞的大规模基准数据集 PanFoMaBench。

## 研究背景与动机

### 科学问题
单细胞 RNA 测序（scRNA-seq）为解析肿瘤异质性提供了单细胞分辨率的强大工具。从高维稀疏的转录组数据中学习有效的细胞和基因表示，是精准医学、生物标志物发现和药物靶点识别等应用的核心挑战。

### 现有方法的局限

**Transformer 类方法（scGPT、GeneFormer、scFoundation 等）**：
- 自注意力机制的计算复杂度为 $O(N^2)$，处理数万基因的完整转录组时计算代价过高
- 通常被迫只选择 top-K 高变基因（如 2048 个），可能丢失重要的低表达功能基因（如转录因子）
- HVG 选择策略引入分析偏差，影响泛癌场景下的泛化能力

**Mamba 类方法（GeneMamba 等）**：
- 提供 $O(N)$ 线性复杂度，突破了效率瓶颈
- 但 Mamba 本质上为序列模型，而基因表达谱是天然无序的集合——基因间的交互不依赖任何固有顺序
- 现有方法采用启发式固定排序（如按平均表达量），忽略了基因功能的上下文依赖性
- 固定维度的隐状态存在远程遗忘问题，在泛癌分析中需要捕获全局模式时性能受限

### 核心洞察
论文提出**解耦建模策略**：将转录组建模分解为两个独立子任务——局部基因交互的并行深度编码，和全局信息的高效序列化整合——分别交给最擅长的架构完成。

## 方法详解

### 整体框架
PanFoMa 采用分层的「局部到全局」处理范式（Local-to-Global），由两个核心模块构成：

1. **Local-context Encoder**：将输入基因分块后用参数共享的轻量 Transformer 并行处理
2. **Global Sequential Feature Decoder**：基于全局细胞状态动态排序基因，再用双向 Mamba 深度整合

整体计算复杂度为 $O(C \cdot M^2 + N \log N)$，其中 $N = C \cdot M$，实现了表达能力与效率的平衡。

### 关键设计

#### 1. 局部上下文编码器（Local-context Encoder）

**分块与表示（ICR）**：
- 每个训练 epoch 随机采样 3072 个基因
- 划分为 $C = 4$ 个不重叠块（chunk），每块包含 $M = 768$ 个基因
- 每个块前置一个可学习的 [CLS] token
- 基因的输入嵌入由基因 ID 嵌入和分箱表达值嵌入逐元素相加：

$$e_{k,i} = \text{Emb}_{\text{id}}(g_{\text{id}_{k,i}}) + \text{Emb}_{\text{val}}(g_{\text{val}_{k,i}})$$

**局部关系建模（LRM）**：
- 所有 $C$ 个块被送入轻量 Transformer 编码器**并行处理**
- 编码器由 $L = 6$ 层**参数共享**的 Transformer Block 堆叠而成
- 输出：每个块的基因级嵌入 $H_{\text{genes},k}^{(L)} \in \mathbb{R}^{M \times D}$ 和 CLS token 摘要向量 $h_{\text{[CLS]},k}^{(L)} \in \mathbb{R}^D$

**设计动机**：「分而治之」策略将全局 $O(N^2)$ 问题分解为 $C$ 个 $O(M^2)$ 的局部问题，大幅降低计算与内存开销，同时保持 Transformer 捕获复杂基因交互的表达能力。参数共享进一步减少模型大小。

#### 2. 全局序列特征解码器（Global Sequential Feature Decoder）

**全局感知动态排序（GDS）**：
- 对所有块的 CLS token 做平均池化，合成全局细胞状态向量：

$$h_{\text{global\_cls}} = \frac{1}{C} \sum_{k=1}^{C} h_{\text{[CLS]},k}^{(L)}$$

- 拼接所有块的基因嵌入后，用点积计算每个基因的重要性分数：

$$s_i = h_i \cdot h_{\text{global\_cls}}^T$$

- 按分数**降序排列**所有基因，得到动态排序的特征矩阵 $H_{\text{sorted}}^{(L)}$

**设计动机**：这是本文最关键的创新——不再使用固定的启发式排序，而是根据每个细胞的全局转录组上下文动态确定基因的输入顺序。这一机制反映了生物学事实：基因的重要性不是静态的，而是取决于其在特定细胞上下文中的动态功能角色。

**双向扫描与门控融合（BSGF）**：
- 排序后的基因序列送入 6 层双向 Mamba 模块
- 前向和反向 Mamba 分别处理正序和逆序序列
- 门控机制对每个基因的双向特征进行自适应加权融合：

$$h_{\text{fused},i} = \gamma_i \odot \overrightarrow{h}_{\text{mamba},i} + (1 - \gamma_i) \odot \overleftarrow{h}_{\text{mamba},i}$$

其中 $\gamma_i = \sigma(\text{Linear}(h_{\text{sorted},i}))$ 是学习到的门控向量。

### 基准数据集构建（PanFoMaBench）

论文通过系统检索 NCBI 数据库，从 83 项研究中整合了约 **350 万高质量细胞**，覆盖 **33 种癌症亚型**、**23 种组织类型**、**616 名患者**。数据经过严格的质控流程：
1. 去除表达基因数过少的细胞
2. 排除异常高基因/UMI 计数的潜在 doublet
3. 过滤线粒体基因比例过高的低活性细胞
4. 剔除低表达基因以降低噪声

### 训练策略
预训练采用大规模自监督学习策略，下游任务通过微调完成。具体的预训练目标在论文中未详细展开，主要关注架构创新和基准构建。

## 实验关键数据

### 主实验：泛癌诊断

| 模型 | Accuracy | Macro-F1 |
|------|----------|----------|
| scFoundation | 0.8876 | 0.8491 |
| scGPT | 0.9013 | 0.8732 |
| GeneMamba | 0.9026 | 0.8619 |
| GeneFormer | 0.9124 | 0.8851 |
| **PanFoMa** | **0.9474 (+3.5%)** | **0.9250 (+4.0%)** |

PanFoMa 在自建泛癌基准上以 94.74% 准确率大幅领先所有基线。

### 批次整合（Batch Integration）

| 数据集 | 指标 | GeneFormer | scGPT | GeneMamba | **PanFoMa** |
|--------|------|-----------|-------|----------|------------|
| Immune | Avg_batch | 0.8153 | 0.9194 | 0.9536 | **0.9641** |
| Immune | Avg_bio | 0.6983 | 0.7879 | 0.8131 | **0.8332** |
| BMMC | Avg_batch | 0.7720 | 0.8431 | 0.9157 | **0.9312** |
| BMMC | Avg_bio | 0.6324 | 0.6576 | 0.7628 | **0.8021** |
| Covid-19 | Avg_batch | 0.8240 | 0.8625 | 0.8742 | **0.9173** |

在 5 个批次整合数据集上，PanFoMa 在大多数指标上均取得最优表现。

### 细胞类型注释

| 数据集 | 模型 | Accuracy | Macro-F1 |
|--------|------|----------|----------|
| hPancreas | GeneMamba | 0.9713 | 0.7710 |
| hPancreas | **PanFoMa** | **0.9815** | **0.7760** |
| MS | scGPT | 0.8471 | 0.6630 |
| MS | **PanFoMa** | **0.8563 (+7.4% vs GeneMamba)** | **0.7016** |
| Myeloid_b | GeneMamba | 0.9603 | 0.9235 |
| Myeloid_b | **PanFoMa** | **0.9726** | 0.9351 |

### 多组学整合

| 数据集 | scGPT | scGLUE | **PanFoMa** |
|--------|-------|--------|------------|
| 10x Multiome PBMC | 0.758 | 0.747 | **0.789 (+3.1%)** |
| BMMC (RNA+Protein) | 0.697 | 0.600 | **0.721 (+2.4%)** |
| ASAP PBMC | **0.587** | 0.561 | 0.579 |

### 关键发现

1. **局部+全局建模的必要性**：纯 Transformer 方法受限于计算复杂度只能处理部分基因，纯 Mamba 方法受限于固定排序无法捕获真实的基因调控关系。PanFoMa 的解耦设计有效解决了这一矛盾。
2. **动态排序的生物学意义**：基于全局细胞状态的动态排序使同一基因在不同细胞中获得不同的位置编码，更好地反映了基因功能的上下文依赖性。
3. **基因调控网络推断**：可视化结果显示 PanFoMa 的注意力机制能以更高的置信度识别 MHC II 类分子相关的基因调控关系，比 scGPT 多恢复了一个相关基因。

## 亮点与洞察

- **架构设计的生物学合理性**：「局部调控信号 → 统一细胞状态」的信息流与真实的基因调控层级相呼应，使架构设计具有生物学解释性。
- **参数共享 Transformer**：6 层共享参数的设计大幅减少模型大小，同时利用了不同 chunk 间基因交互模式的共性。
- **动态排序机制**是连接 Transformer（无序集合建模）和 Mamba（有序序列建模）的桥梁，是本文最精妙的设计。
- **大规模基准数据集**：PanFoMaBench 覆盖 33 种癌症亚型、350 万+ 细胞，是目前最全面的泛癌单细胞基准之一。

## 局限与展望

1. **标题的误导性**：论文标题提到"Pathology Image Analysis"，但实际处理的是单细胞转录组数据而非病理图像，存在一定程度的不匹配。
2. **预训练细节缺失**：论文主要关注架构设计和基准构建，对预训练目标函数的描述不够详细。
3. **基因采样策略**：每个 epoch 随机采样 3072 个基因，剩余基因的信息可能被忽略，长期训练的覆盖率和稳定性需进一步分析。
4. **Myeloid 数据集上的弱势**：在 Myeloid 数据集上 PanFoMa 未能超越 GeneMamba（0.6515 vs 0.6607），说明在某些特定任务上混合架构的优势不是绝对的。
5. **计算开销分析不足**：虽然声称"轻量级"，但未与 GeneMamba 直接对比 FLOPs 和推理速度。

## 相关工作与启发

- **scGPT** 的 GPT 风格遮蔽预训练策略在单细胞转录组建模上取得了开创性成果，但受限于 $O(N^2)$ 复杂度。
- **GeneMamba** 用 Bi-Mamba 替代 Transformer 突破了效率瓶颈，但固定排序策略是硬伤。
- PanFoMa 的分块并行 + 动态排序 + 双向 Mamba 的范式可能启发其他「集合→序列」的建模问题（如点云处理、分子图）。
- 门控融合机制类似于 NLP 中的双向 LSTM 融合策略，但在 Mamba 上下文中的应用是新颖的。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.5 | Transformer-Mamba 混合架构 + 动态排序机制 |
| 技术深度 | 4 | 架构设计精巧，各模块动机清晰 |
| 实验充分性 | 4 | 泛癌诊断+批次整合+注释+多组学，五个基线 |
| 写作质量 | 3.5 | 结构清晰但标题与内容有偏差 |
| 实用性 | 4 | 代码开源+大规模基准贡献 |
| **综合** | **4** | 架构创新有亮点，基准贡献有价值，但部分细节待完善 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](../../CVPR2025/medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[ICML 2026\] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](../../ICML2026/medical_imaging/thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)
- [\[CVPR 2026\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](../../CVPR2026/medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)

</div>

<!-- RELATED:END -->
