---
title: >-
  [论文解读] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment
description: >-
  [医学图像] 提出 **RADM (Rotationally Aligned Diffusion Model)**，通过学习样本相关的 SO(3) 旋转变换构建对齐的潜空间，使非等变扩散模型能够有效生成 3D 分子，在生成质量上媲美 SOTA 等变模型，同时提供更好的可扩展性和采样效率。
tags:
  - 医学图像
---

# Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment

> **会议**: ICML 2025
> **arXiv**: [2506.10186](https://arxiv.org/abs/2506.10186)
> **代码**: [GitHub](https://github.com/skeletondyh/RADM)
> **领域**: 分子生成 / 扩散模型
> **关键词**: 3D分子生成, 非等变, 旋转对齐, 潜空间扩散, AutoEncoder

## 一句话总结

提出 **RADM (Rotationally Aligned Diffusion Model)**，通过学习样本相关的 SO(3) 旋转变换构建对齐的潜空间，使非等变扩散模型能够有效生成 3D 分子，在生成质量上媲美 SOTA 等变模型，同时提供更好的可扩展性和采样效率。

## 研究背景与动机

3D 分子生成中，分子在三维空间的旋转不改变化学性质（SE(3) 对称性）。主流方法通过等变网络（如 EGNN）满足此约束：

$$p(\mathbf{x}) = p(\mathbf{R}\mathbf{x}) \quad \forall \mathbf{R} \in \text{SO}(3)$$

但等变架构有明显缺点：

**参数化复杂**：EGNN 等需要特殊的消息传递规则来维持等变性

**缺乏标准实现**：不同于 Transformer 在 vision/NLP 中的统一地位

**效率和可扩展性差**：难以利用 FlashAttention 等现代加速技术

核心问题：**等变性是否必要？** 一个分子的概率由其所有可能 3D 位置的总概率决定，而不需要每个位置有相等概率。

## 方法详解

### 整体框架

分两阶段训练：(1) 训练带旋转对齐的自编码器 → 构建对齐潜空间；(2) 在对齐潜空间中训练非等变扩散模型。

### 关键设计

**1. 旋转参数化**

用任意矩阵 $\mathbf{M} \in \mathbb{R}^{3 \times 3}$ 通过 SVD 投影到 SO(3)：

$$\mathbf{R} = \text{SVD}^+(\mathbf{M}) = \mathbf{U}\text{diag}(1, 1, \det(\mathbf{U}\mathbf{V}^\top))\mathbf{V}^\top$$

该参数化在 $\det(\mathbf{M}) \neq 0$ 时光滑，适合梯度优化。

**2. 旋转网络**

使用 vanilla GNN（非等变）从分子 $(\mathbf{x}, \mathbf{h})$ 生成样本相关的旋转矩阵 $\mathbf{R}_\theta$。原子坐标和特征拼接后通过消息传递，最终平均池化后经 2 层 MLP 得到 $\mathbf{M}$。

**3. 非等变自编码器**

- **编码器**：1 层 EGNN（与 GeoLDM 相同，便于消融）
- **解码器**：非等变 GNN——**关键设计**：解码器必须非等变，以使重建损失对旋转敏感，从而为旋转网络提供梯度信号

重建损失：
$$\mathcal{L} = -\mathbb{E}_{q_{\theta,\eta}(\mathbf{z}_x, \mathbf{z}_h | \mathbf{x}, \mathbf{h})}[\log p_\psi(\mathbf{R}_\theta\mathbf{x}, \mathbf{h} | \mathbf{z}_x, \mathbf{z}_h)]$$

**4. 非等变潜空间扩散模型**

在对齐潜空间中训练标准去噪扩散模型，噪声预测网络可使用：
- Vanilla GNN（拼接坐标和特征）
- **DiT (Diffusion Transformer)**：直接复用视觉领域的 Transformer 加速实现

训练目标（标准 DDPM）：

$$\mathcal{L}(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}, t}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\phi\|^2]$$

**5. 平移处理**

减去质心将坐标投影到 $(N-1) \times 3$ 维子空间，在扩散的每一步也减去预测噪声的质心。

## 实验关键数据

### QM9 数据集

| 模型 | Atom Sta (%) | Mol Sta (%) | Valid (%) | Valid & Unique (%) |
|------|-------------|------------|-----------|-------------------|
| EDM (等变) | 98.7 | 82.0 | 91.9 | 90.7 |
| GeoLDM (等变) | 98.9 | 89.4 | 93.8 | 92.7 |
| GDM-aug (非等变) | 97.6 | 71.6 | 90.4 | 89.5 |
| **RADM (非等变)** | **~98.8** | **~87** | **~93** | **~92** |

### GEOM-Drugs 数据集

| 模型 | Atom Sta (%) | Valid (%) |
|------|-------------|-----------|
| EDM | 81.3 | 92.6 |
| GeoLDM | 84.4 | 99.3 |
| GDM-aug | 77.7 | 91.8 |
| **RADM** | 比 GDM-aug 大幅提升 | - |

### 效率对比

- RADM 采样速度显著快于等变扩散模型
- 使用 DiT 作为去噪网络可利用 FlashAttention 加速
- 非等变架构的参数效率更高

### 关键发现

- 非等变 RADM 显著超越此前所有非等变方法（GDM、GDM-aug、GraphLDM）
- 生成质量接近 SOTA 等变模型（GeoLDM）
- 旋转对齐是关键：消融实验证明去掉旋转网络后性能大幅下降
- 非等变解码器是必要的（等变解码器使重建损失对旋转不变→旋转网络无法学习）

## 亮点与洞察

1. **重新审视等变性的必要性**：概率论上等变约束并非必须，打破了领域惯性
2. **旋转对齐的灵感来源**：3D 视觉的数据集（如 ShapeNet）都是对齐的，分子为何不行？
3. **自编码器学习无监督对齐**：巧妙利用重建目标间接督促旋转网络对齐分子
4. **架构统一的可能性**：非等变模型可以直接使用 DiT 等通用架构，连接分子生成与视觉生成
5. **SVD 旋转参数化**：光滑且无约束，适合端到端梯度学习

## 局限性

1. 自编码器和扩散模型分开训练，可能未达到联合最优
2. 编码器仍使用 EGNN（等变），并非完全非等变框架
3. 仅在小分子数据集（QM9、GEOM-Drugs）上验证，未测试蛋白质等大分子
4. 旋转对齐仅处理 SO(3)，排列等变性仍由注意力机制保证
5. 潜空间维度与原始空间维度相同，未实现真正的维度压缩

## 相关工作

- 等变扩散：EDM, GeoLDM, MiDi
- 非等变方法：GDM, GDM-aug, GraphLDM
- 旋转表示：SVD 参数化, Euler 角, 指数坐标
- 潜扩散：LDM, GeoLDM

## 评分

⭐⭐⭐⭐ (4/5)

论点清晰有力——等变性非必须。旋转对齐的自编码器设计优雅，为连接分子生成与通用生成架构开辟了路径。实验充分但规模有限（仅小分子）。

<!-- RELATED:START -->

## 相关论文

- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](../../ACL2025/medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)
- [Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)

<!-- RELATED:END -->
