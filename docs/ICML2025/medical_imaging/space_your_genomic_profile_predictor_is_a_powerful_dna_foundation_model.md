---
title: >-
  [论文解读] SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model
description: >-
  [ICML2025][医学图像][DNA基础模型] 提出 SPACE（Species-Profile Adaptive Collaborative Experts），论证**监督式基因组图谱预测**比无监督序列预训练能学到更有效的 DNA 表征，并通过物种感知 MoE 编码器和双门控解码器在 18 项 NT 下游任务中 11 项 SOTA。
tags:
  - ICML2025
  - 医学图像
  - DNA基础模型
  - 基因组图谱预测
  - Mixture of Experts
  - 跨物种建模
  - 监督预训练
---

# SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model

**会议**: ICML2025  
**arXiv**: [2506.01833](https://arxiv.org/abs/2506.01833)  
**代码**: [ZhuJiwei111/SPACE](https://github.com/ZhuJiwei111/SPACE)  
**领域**: 基因组学 / DNA基础模型  
**关键词**: DNA基础模型, 基因组图谱预测, Mixture of Experts, 跨物种建模, 监督预训练

## 一句话总结

提出 SPACE（Species-Profile Adaptive Collaborative Experts），论证**监督式基因组图谱预测**比无监督序列预训练能学到更有效的 DNA 表征，并通过物种感知 MoE 编码器和双门控解码器在 18 项 NT 下游任务中 11 项 SOTA。

## 研究背景与动机

- **DNA 基础模型困境**：现有 DNA 基础模型（DNABERT、HyenaDNA、Nucleotide Transformer）照搬 NLP 范式（MLM/NTP）做无监督预训练，但 DNA 序列本身不像自然语言那样自含语义——其功能受到染色质可及性、表观修饰、转录因子结合等**基因组图谱**的调控
- **监督信号的价值**：基因组图谱预测模型（GPPM）如 Enformer 通过预测实验可测的基因组图谱来学习 DNA 表征，天然包含更丰富的生物学上下文信息，但此前缺乏系统研究来挖掘其表征学习潜力
- **现有 GPPM 的局限**：
    - 共享编码器无法捕捉物种特异性特征（如人/鼠调控机制差异）
    - 独立预测头忽略了不同基因组图谱之间的功能依赖关系（如高表达区域通常伴随高染色质可及性）

## 方法详解

### 整体架构

SPACE 采用三阶段设计：

1. **CNN 局部上下文聚合**：沿用 Enformer 的 1D-CNN 模块压缩原始核苷酸序列，生成 128bp 分辨率的隐状态 $h_m \in \mathbb{R}^{L \times d_h}$
2. **物种感知 Transformer 编码器**：基于稀疏 MoE 的跨物种建模
3. **图谱分组增强解码器**：基于双门控 MoE 的跨图谱关系建模

### 物种感知编码器（Species-aware Encoder）

**物种特异嵌入**：为每个物种 $S_m$ 引入可学习嵌入 $e_m \in \mathbb{R}^{1 \times d_h}$，与序列隐状态拼接后送入 Transformer 层，显式引导模型区分物种。

**跨物种 MoE**：每个 MoE 层包含 $N$ 个共享专家 $\{E_1, \dots, E_N\}$ 和 $M$ 个物种特定门控网络 $\{G_1, \dots, G_M\}$。对物种 $S_m$ 的隐状态，输出为：

$$\hat{h}_m = \text{MHAttention}([e_m, h_m])$$

$$y_m = \sum_{k=1}^{N} G_m(\hat{h}_m)_k \cdot E_k(\hat{h}_m)$$

门控权重通过 TopK 稀疏路由 + 噪声注入计算：

$$G_m(\hat{h}_m) = \text{Softmax}(\text{TopK}(g(\hat{h}_m) + \mathcal{R}_{\text{noise}}))$$

**互信息损失**：最大化物种身份 $S$ 与专家选择 $E$ 之间的互信息，鼓励专家学习物种特异模式：

$$\mathcal{L}_{\text{MI}} = -MI(S; E) = -H(S) - H(E) + H(S, E)$$

### 图谱分组增强解码器（Profile-grouped Enhancement Decoder）

基于两个生物学原理设计：(P1) 演化保守性——跨物种同源图谱共享调控机制；(P2) 功能相互依赖——不同图谱受共同机制调控。

**图谱分类**：将初始预测 $o_{\text{base}}$ 按实验类型（DNase/ATAC-seq、TF ChIP-seq、Histone ChIP-seq、CAGE）分为 $Q$ 类。

**双门控专家加权聚合**：

- **组级门控**（第一层）：基于物种嵌入和序列上下文动态加权 $R$ 个专家组，捕捉演化保守模式

$$\hat{G}^q = \text{Softmax}(G^q_{\text{species}}(e) + G^q_{\text{sequence}}(\text{Pool}(y)))$$

- **专家级门控**（第二层）：基于预测模式在每组内选择特定专家，捕捉功能依赖

$$o^q_{\text{enhanced}} = \sum_{r=1}^{R} \hat{G}_r^q \cdot \left(\sum_{k=1}^{K} G_r^q(o^q)_k \cdot E_k(o^q)\right)$$

最终预测通过残差连接 $o_{\text{final}} = o_{\text{base}} + o_{\text{enhanced}}^T$。

### 训练目标

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Poisson}} - \alpha \sum_{d=1}^{D} MI(S; E_d)$$

其中 Poisson 负对数似然为主损失，$\alpha=0.01$ 控制互信息正则化强度。

## 实验关键数据

### NT 下游任务（18 个数据集）

| 模型 | H3K4me3 | H3K9ac | Enhancers | Enhancers(types) | Donors | Acceptors |
|------|---------|--------|-----------|------------------|--------|-----------|
| DNABERT-2 | 0.646 | 0.564 | 0.517 | 0.476 | 0.837 | 0.855 |
| NT-Multi (2.5B) | 0.618 | 0.527 | 0.527 | 0.484 | 0.958 | 0.964 |
| Enformer | 0.635 | 0.593 | 0.614 | 0.573 | 0.749 | 0.739 |
| **SPACE** | **0.661** | **0.635** | **0.631** | **0.583** | **0.942** | **0.902** |

SPACE 在 18 项任务中 **11 项 SOTA**，且参数量远少于 NT-Multispecies (2.5B)。

### GUE 跨物种验证（酵母 + 病毒）

| 任务 | Enformer | SPACE | 提升 |
|------|----------|-------|------|
| H3 | 70.65 | 79.53 | +8.88 |
| H3K14ac | 37.87 | 54.12 | +16.25 |
| H3K4me3 | 22.19 | 49.47 | **+27.28** |
| H4ac | 32.90 | 53.09 | +20.19 |
| Covid | 61.33 | 70.26 | +8.93 |

在 **与训练物种系统发育距离极远** 的酵母和病毒基因组上，SPACE 相比 Enformer 提升显著（最高 +27.28）。

### Genomic Benchmarks

| 任务 | Enformer | SPACE |
|------|----------|-------|
| Mouse Enhancers | 0.835 | **0.905** |
| Drosophila Enhancers | 0.613 | **0.721** |
| Human Enhancer Ensembl | 0.844 | **0.919** |
| Human Regulatory | 0.903 | **0.944** |

## 亮点与洞察

1. **范式挑战**：系统论证了"监督式基因组图谱预测 > 无监督序列预训练"这一重要结论，为 DNA 基础模型领域提供了新范式
2. **MoE 精妙应用**：编码器 MoE 用于跨物种知识路由，解码器双门控 MoE 用于跨图谱关系建模，两处 MoE 各有明确的生物学动机
3. **跨物种泛化**：通过互信息损失显式引导专家-物种对齐，在远缘物种（酵母/病毒）上泛化能力突出
4. **实验全面**：覆盖 NT 下游任务、GUE 跨物种基准、Genomic Benchmarks 三大类评测，令人信服

## 局限与展望

1. **训练物种有限**：仅在人类和小鼠上预训练，未纳入更多物种（如植物、昆虫），物种 MoE 的可扩展性尚未验证
2. **计算开销**：MoE 结构增加了模型复杂度，论文未详细分析推理效率和参数量对比
3. **长序列建模**：输入长度沿用 Enformer 的 ~196K bp 窗口，未探索更长范围的调控元件建模
4. **图谱分类依赖先验**：解码器的分组算子 $\Phi$ 基于人工定义的实验类型，可能限制对未知图谱关系的发现
5. **下游任务单一**：主要聚焦分类任务，缺乏对基因表达预测、变异效应预测等回归任务的深入评估

## 相关工作与启发

- **Enformer** (Avsec et al., 2021)：SPACE 的直接基线，共享 CNN 前端和 Poisson 损失
- **Nucleotide Transformer** (Dalla-Torre et al., 2024)：代表无监督路线的 2.5B 大模型，SPACE 以更小参数量超越
- **DNABERT-2** (Zhou et al., 2024)：改进 tokenization 的 MLM 预训练代表
- **Mod-Squad** (Chen et al., 2023)：互信息损失的灵感来源，用于多任务学习中的专家特化

## 评分
- 新颖性: ⭐⭐⭐⭐ — 监督预训练优于无监督的系统论证 + MoE 在基因组的双层应用
- 实验充分度: ⭐⭐⭐⭐⭐ — 三大基准、多物种、消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法推导严谨
- 价值: ⭐⭐⭐⭐ — 对 DNA 基础模型社区的范式选择具有重要参考意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](../../NeurIPS2025/medical_imaging/janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[ICML 2025\] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[ICML 2025\] Foundation Models for Clinical Records at Health System Scale](foundation_models_for_clinical_records_at_health_system_scale.md)
- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](../../NeurIPS2025/medical_imaging/neuript_foundation_model_for_neural_interfaces.md)

<!-- RELATED:END -->
