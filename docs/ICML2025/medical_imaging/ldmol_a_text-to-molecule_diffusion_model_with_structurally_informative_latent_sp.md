---
title: >-
  [论文解读] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models
description: >-
  [ICML2025][医学图像][潜在扩散模型] 提出 LDMol，通过 SMILES 枚举对比学习构建结构感知的潜在空间，在该空间上训练条件扩散模型实现文本到分子生成，首次让扩散模型在文本数据生成任务上超越自回归模型。
tags:
  - ICML2025
  - 医学图像
  - 潜在扩散模型
  - 文本到分子生成
  - 对比学习
  - SMILES
  - 结构感知表示
---

# LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models

**会议**: ICML2025  
**arXiv**: [2405.17829](https://arxiv.org/abs/2405.17829)  
**代码**: [jinhojsk515/LDMol](https://github.com/jinhojsk515/LDMol)  
**领域**: 分子生成 / 药物发现  
**关键词**: 潜在扩散模型, 文本到分子生成, 对比学习, SMILES, 结构感知表示

## 一句话总结

提出 LDMol，通过 SMILES 枚举对比学习构建结构感知的潜在空间，在该空间上训练条件扩散模型实现文本到分子生成，首次让扩散模型在文本数据生成任务上超越自回归模型。

## 研究背景与动机

- **扩散模型在分子生成中遇到困难**：分子数据具有不可避免的离散性（原子/键类型、SMILES token），直接在原始分子表示上训练扩散模型难以处理复杂条件（如自然语言）。
- **已有分子扩散模型的局限**：TGM-DLM 等直接在 SMILES token 序列上做连续高斯扩散，引入不合理的数值噪声，导致生成质量差、条件遵循能力弱。
- **潜在空间设计是关键**：简单的 autoencoder（如 β-VAE）虽能提供连续可重建的潜在空间，但其特征不保证反映分子结构特征。β-VAE 对同一分子的不同 SMILES 枚举，特征距离甚至与随机分子对无显著区别。
- **核心洞察**：精心设计的、化学结构信息丰富的潜在空间，能大幅提升扩散模型在分子生成中的表现。

## 方法详解

LDMol 包含三个阶段：(1) 对比预训练 SMILES 编码器，(2) 训练压缩层+解码器，(3) 训练文本条件潜在扩散模型。

### 阶段一：结构感知 SMILES 编码器（对比学习）

利用 **SMILES 枚举**（同一分子可通过不同节点遍历顺序得到多种等价 SMILES 表示）构造正样本对，不同分子的 SMILES 作为负样本对。

对比学习的关键论点：SMILES 枚举对之间的互信息极小（仅共享分子结构身份），迫使编码器必须理解分子图的完整连接性才能识别所有枚举变体，从而编码出**独特的结构特征**。

对称 InfoNCE 损失：

$$\mathcal{L}_{enc}(M, M') = \mathcal{L}_{con}(M, M') + \mathcal{L}_{con}(M', M)$$

其中：

$$\mathcal{L}_{con}(M, M') = -\sum_{k=1}^{N} \log \frac{\exp(v_k \cdot v_k' / \tau)}{\sum_{i=1}^{N} \exp(v_k \cdot v_i' / \tau)}$$

$v_k$ 和 $v_k'$ 分别为编码器对正样本对 $m_k, m_k'$ 输出的 [SOS] token 经线性投影和归一化后的向量。训练数据为 PubChem 的 1000 万分子。还引入**硬负例**策略：将立体异构体视为难区分的负样本以增强编码器对立体化学信息的敏感性。

### 阶段二：压缩层 + 自回归解码器

- **线性压缩层** $f(\cdot)$：将编码器输出从 $[L \times d_{enc}]$ 压缩到 $[L \times d_z]$，减少维度灾难，同时保留结构感知特征。故意保持简单（仅线性层），避免偏离已预训练好的特征空间。
- **自回归 Transformer 解码器**：通过交叉注意力从压缩潜在向量 $f(\mathcal{E}(m))$ 重建 SMILES，使用标准的 next-token prediction 损失：

$$\mathcal{L}_{dec} = -\sum_{i=1}^{n} \log p(t_n | t_{0:n-1}, f(\mathcal{E}(m)))$$

训练时冻结编码器参数，仅训练压缩层和解码器。重建准确率约 **98%**。

### 阶段三：文本条件潜在扩散模型

- **扩散目标域**：压缩后的潜在空间 $[L \times d_z]$。
- **架构**：采用 **DiT-base**（Transformer-based diffusion），而非 UNet，因为 UNet 的空间归纳偏置不适合 SMILES 潜在空间。通过交叉注意力注入文本条件。
- **文本编码器**：MolT5-large 的编码器部分。
- **条件扩散训练损失**：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_0, c, t, \epsilon} \| \epsilon - \epsilon_\theta(x_t, t, c) \|_2^2$$

- **采样**：DDIM 100 步 + classifier-free guidance（训练时 3% 概率用空文本替换条件，推理时 $\omega=2.5$）。
- **训练数据**：PubchemSTM + ChEBI-20 + PCdes 共约 32 万对，远小于 MolT5/bioT5 等基线的百万级数据量。

## 实验关键数据

### 文本到分子生成（ChEBI-20 测试集）

| 模型 | 类型 | Validity↑ | BLEU↑ | Exact Match↑ | FCD↓ | Morgan FTS↑ |
|------|------|-----------|-------|--------------|------|-------------|
| bioT5+ | AR | 1.000 | 0.872 | 0.522 | 0.35 | 0.779 |
| bioT5 | AR | 1.000 | 0.867 | 0.413 | 0.43 | 0.734 |
| MolT5-large | AR | 0.905 | 0.854 | 0.311 | 1.20 | 0.684 |
| TGM-DLM | DM | 0.871 | 0.826 | 0.242 | 0.77 | 0.688 |
| **LDMol** | **DM** | **0.941** | **0.926** | **0.530** | **0.20** | **0.931** |

LDMol 在几乎所有指标上大幅超越 AR 和 DM 基线。FCD 从 bioT5+ 的 0.35 降到 0.20，Morgan FTS 从 0.779 提升到 0.931。

### 分子到文本检索（64-way 准确率）

| 模型 | PCdes-段落 | MoMu-段落 |
|------|-----------|----------|
| MolCA | 86.4% | 73.4% |
| **LDMol (n=25)** | **90.3%** | **87.1%** |

### 消融实验

| 模型变体 | 重建准确率 | Validity | Match | FCD |
|---------|-----------|----------|-------|-----|
| 无对比学习 | 1.000 | 0.019 | 0.000 | 58.60 |
| β-VAE (β=0.001) | 0.999 | 0.847 | 0.492 | 0.34 |
| **LDMol（完整）** | 0.983 | **0.941** | **0.530** | **0.20** |

无对比预训练时扩散模型完全无法学习潜在分布（Validity 仅 0.019）；β-VAE 可工作但性能明显不如结构感知的 LDMol。

## 亮点与洞察

1. **首次让扩散模型超越自回归模型生成文本型数据**：SMILES 本质是一种文本，LDMol 在文本数据生成上超越 AR 模型，具有范式意义。
2. **SMILES 枚举作为对比学习增强的巧妙设计**：利用分子图的遍历不变性天然构造正样本对，理论上互信息最小且无信息损失。
3. **扩散模型的多功能性**：同一个训练好的 LDMol 无需额外训练即可用于分子-文本检索（利用噪声预测误差作为匹配度）和文本引导分子编辑（借鉴 DDS）。
4. **数据高效**：仅用 32 万对训练数据，远少于 AR 基线的百万级数据，但取得更好效果。
5. **潜在空间设计的方法论启示**：好的潜在空间不仅要可重建，更要编码语义/结构信息，这对所有潜在扩散模型都有指导价值。

## 局限与展望

1. **复杂生物属性的条件遵循仍有不足**：对复杂生物活性描述的生成准确率有待提升。
2. **Validity 未达 100%**：虽然 0.941 已很高，但 bioT5+ 可达 1.0，说明解码器在处理扩散模型生成的潜在向量时仍有鲁棒性问题。
3. **推理速度**：DDIM 100 步采样 + 解码器自回归生成，推理开销大于单次前向的 AR 模型。
4. **依赖 SMILES 表示**：未探索分子图等其他表示是否能进一步受益于此框架。
5. **文本编码器固定**：使用冻结的 MolT5-large，未探索与更强 LLM 结合的潜力。

## 相关工作与启发

- **TGM-DLM** (Gong et al., 2024)：直接在 SMILES token 上训练扩散模型，性能不佳，验证了原始离散空间的困难。
- **Stable Diffusion / LDM** (Rombach et al., 2022)：图像领域潜在扩散模型的范式来源。
- **DiT** (Peebles & Xie, 2023)：Transformer-based 扩散模型架构。
- **DDS** (Hertz et al., 2023)：文本引导图像编辑方法，被迁移到分子编辑。
- **对自然语言扩散模型的启示**：LDMol 的成功暗示，通过更好的潜在空间设计，扩散模型或许能在通用文本生成上追赶 AR 模型。

## 评分

- 新颖性: ⭐⭐⭐⭐ — SMILES 枚举对比学习 + 潜在扩散的组合非常巧妙
- 实验充分度: ⭐⭐⭐⭐ — 多任务评估 + 充分消融，但缺少与更多分子生成方法的比较
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观，逻辑连贯
- 价值: ⭐⭐⭐⭐ — 在分子生成和文本扩散模型两个方向都有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/medical_imaging/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](../../CVPR2025/medical_imaging/zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[ICML 2025\] SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model](space_your_genomic_profile_predictor_is_a_powerful_dna_foundation_model.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](../../NeurIPS2025/medical_imaging/atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)

<!-- RELATED:END -->
