---
title: >-
  [论文解读] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration
description: >-
  [NeurIPS 2025][计算生物][单细胞多组学] 提出scMRDR框架，基于β-VAE将单细胞多组学数据的潜在表征解耦为模态共享和模态特异成分，通过等距正则化、对抗训练和掩码重建损失实现非配对多组学数据的可扩展整合。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "单细胞多组学"
  - "非配对数据整合"
  - "β-VAE"
  - "解耦表征"
  - "对抗训练"
---

# scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration

**会议**: NeurIPS 2025  
**arXiv**: [2510.24987](https://arxiv.org/abs/2510.24987)  
**代码**: [有](https://github.com/sjl-sjtu/scMRDR)  
**领域**: 生物信息学 / 多组学整合  
**关键词**: 单细胞多组学, 非配对数据整合, β-VAE, 解耦表征, 对抗训练

## 一句话总结
提出scMRDR框架，基于β-VAE将单细胞多组学数据的潜在表征解耦为模态共享和模态特异成分，通过等距正则化、对抗训练和掩码重建损失实现非配对多组学数据的可扩展整合。

## 研究背景与动机
单细胞测序技术已能从单细胞分辨率测量多种分子模态（scRNA、scATAC、蛋白质），但由于技术限制，大规模数据通常在不同模态间非配对。现有方法存在两类局限：(1) 联合降维方法（如MOFA、Seurat）依赖配对数据或先验对应；(2) 流形对齐方法（如UnionCom、SCOT）需计算全局配对耦合矩阵，限制了可扩展性且通常仅支持两种模态。本文目标是设计既灵活（完全非配对）又可扩展（大数据集+多模态）的整合框架。

## 方法详解

### 整体框架
使用单一编码器-解码器的β-VAE，将每个细胞的潜在表征解耦为模态共享($z_u$)和模态特异($z_s^{(m)}$，先验$\mathcal{N}(\mu_m, \sigma_m^2 I)$)成分。三项正则化确保共享空间的质量。

### 关键设计

**β-VAE解耦**：$\beta > 1$的KL散度加权鼓励$z_u$和$z_s$的条件独立。不同组学的观测被统一视为单一样本输入同一编码器，天然支持非配对和多模态。生成过程假设ZINB分布（零膨胀负二项分布），适配稀疏计数数据。

**等距正则化（Isometric Loss）**：在无细胞类型标签的情况下保留生物异质性。强制共享空间$z_u$保持全潜在空间$z=(z_u,z_s)$的距离结构：$\mathcal{L}_{\text{preserve}} = \sum_m \sum_{i,j \in X^{(m)}} [\|\mu_{z_u}(x_i)-\mu_{z_u}(x_j)\|_2 - \|\mu_z(x_i)-\mu_z(x_j)\|_2]^2$。

**对抗正则化**：引入m类判别器$D(z_u)$区分不同组学来源，编码器优化方向相反以促进跨模态对齐。

**掩码重建损失**：不同组学特征覆盖不同（如CITE-seq蛋白仅~134个 vs RNA ~13000基因），二元掩码$\mathbf{b}$防止梯度通过未测量特征回传，并按可用特征比例缩放以保持损失可比性。

### 损失函数 / 训练策略
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta\mathcal{L}_{\text{KL}} + \lambda\mathcal{L}_{\text{alignment}} + \gamma\mathcal{L}_{\text{preserve}}$

训练过程交替更新判别器（$\mathcal{L}_{\text{discriminator}}$）和VAE（$\mathcal{L}_{\text{total}}$）。默认超参数$\beta=2, \gamma=5, \lambda=5$。

## 实验关键数据

### 两组学整合（人类肾脏，scRNA+scATAC）

| 方法 | 总分 | 批次校正 | 生物保守 | 模态整合 |
|------|------|---------|---------|---------|
| scVI（基线） | 0.59 | 0.48 | 0.63 | 0.66 |
| Seurat v5 | - | - | - | - |
| JAMIE | - | - | - | - |
| **scMRDR(β=2,γ=5,λ=5)** | **0.66** | **0.52** | **0.74** | **0.70** |

### 消融实验

| 配置 | 总分 | 说明 |
|------|------|------|
| β=2,γ=5,λ=5 | **0.66** | 完整模型 |
| β=2,γ=5,λ=0 | 0.61 | 无对抗→模态整合下降 |
| β=2,γ=0,λ=5 | 0.61 | 无等距→生物保守下降 |
| β=1,γ=0,λ=0 | 0.59 | 基线（无正则化） |

### 大规模数据扩展性（小鼠初级运动皮层，124k细胞）
- JAMIE、UnionCom、Pamona：因内存/优化错误无法运行
- GLUE：预处理方法对大规模数据敏感，性能显著下降
- **scMRDR**：稳定表现，批次校正和生物保守均保持优秀

### 三组学整合（scRNA+scATAC+scProtein）
- Seurat v5不支持三组学
- GLUE无法对齐三种组学的分布（蛋白质特征数远少于其他两种）
- **scMRDR自然扩展到三+组学**，表现一致优秀

### 关键发现
- 等距正则化对生物保守至关重要，对抗训练对模态对齐至关重要
- 掩码损失在蛋白质组学特征稀少时防止性能退化
- 性能对超参数选择相对鲁棒（多种组合结果相近）
- 空间位置插补实验验证了整合的生物学意义（推断位置与皮层层注释吻合）

## 亮点与洞察
1. **设计优雅**：单一编码器-解码器架构天然支持任意数量的非配对组学
2. **理论驱动**：利用解耦空间的不可辨识性，通过正则化约束共享子空间为最大保留结构信息的那个
3. **掩码损失实用**：简洁地解决不同组学特征覆盖差异这一核心问题
4. **生物学验证**：空间位置插补发现了新的空间变异基因

## 局限与展望
- 对抗损失（min-max优化）增加训练难度和不稳定性
- 将所有模态映射到基因水平（如scATAC的peak聚合为gene activity score）可能损失信息
- 超参数$\beta, \lambda, \gamma$间的权衡仍需靠经验调试
- 未涉及空间多组学和扰动测序的时空动态整合

## 相关工作与启发
- 从scVI升级为β-VAE+正则化的路径简洁明了
- 等距损失类似于ISOMAP的思想，在潜在空间保距
- 掩码损失策略可推广到其他存在特征缺失的多模态学习任务

## 评分
- 新颖性：⭐⭐⭐⭐（β-VAE+三正则化的组合创新）
- 技术深度：⭐⭐⭐⭐（解耦表征+可扩展设计）
- 实验完整性：⭐⭐⭐⭐⭐（消融+大规模+三组学+生物验证）
- 实用价值：⭐⭐⭐⭐⭐（解决单细胞多组学整合的实际痛点）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)
- [\[ICML 2025\] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](../../ICML2025/computational_biology/scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](../../ICML2026/computational_biology/scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)

</div>

<!-- RELATED:END -->
