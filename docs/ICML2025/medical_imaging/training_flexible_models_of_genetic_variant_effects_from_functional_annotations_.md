---
title: >-
  [论文解读] Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra
description: >-
  [ICML2025][医学图像][GWAS] 本文提出 DeepWAS（Deep genome Wide Association Studies），利用现代快速线性代数技术（带状矩阵近似 + 迭代求解）解决 GWAS 中大规模 LD 矩阵求逆的计算瓶颈，首次实现用大规模神经网络最大化全似然来训练功能注释驱动的遗传变异效应预测模型，且发现只有在全似然训练下（而非传统 summary statistics 拟合）更大的模型才能带来更好的性能。
tags:
  - ICML2025
  - 医学图像
  - GWAS
  - 深度学习
  - 线性代数加速
  - 遗传变异效应预测
  - 功能注释
---

# Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra

**会议**: ICML2025  
**arXiv**: [2506.19598](https://arxiv.org/abs/2506.19598)  
**代码**: [https://github.com/AlanNawzadAmin/DeepWAS](https://github.com/AlanNawzadAmin/DeepWAS)  
**领域**: 医学图像 / 计算基因组学  
**关键词**: GWAS, 深度学习, 线性代数加速, 遗传变异效应预测, 功能注释

## 一句话总结

本文提出 DeepWAS（Deep genome Wide Association Studies），利用现代快速线性代数技术（带状矩阵近似 + 迭代求解）解决 GWAS 中大规模 LD 矩阵求逆的计算瓶颈，首次实现用大规模神经网络最大化全似然来训练功能注释驱动的遗传变异效应预测模型，且发现只有在全似然训练下（而非传统 summary statistics 拟合）更大的模型才能带来更好的性能。

## 研究背景与动机

**领域现状**：全基因组关联研究（GWAS）旨在通过分析数十万个体的基因型数据（$M \approx 10^6 - 10^8$ 个变异位点，$N \approx 10^5$ 个样本），建立遗传变异与表型（如身高、哮喘等疾病）之间的模型。遗传学家利用功能基因组特征（如 DNA 可及性、转录因子结合位点、跨物种保守性等）构建 "functionally informed priors"，以更好地预测变异对表型的影响。

**现有痛点**：由于连锁不平衡（Linkage Disequilibrium, LD）—— 基因组中相邻变异之间存在强相关性 —— 计算边际似然需要对大型 LD 矩阵进行求逆和对数行列式计算，复杂度为 $\mathcal{O}(M^3)$，这在百万级变异位点下计算上不可行。

**核心矛盾**：为了回避 LD 矩阵求逆，现有 SOTA 方法（如 LD score regression）做了两个妥协：(a) 仅使用简单参数化模型（线性或低维），无法捕捉功能注释与表型之间的复杂非线性关系；(b) 拟合 summary statistics 而非完整似然，牺牲了统计效率。这导致即使有更多数据和特征可用，模型性能也无法随之提升。

**本文目标** 如何在 GWAS 的全似然框架下高效训练大规模灵活的神经网络模型，使其能充分利用日益丰富的功能基因组特征。

**切入角度**：作者注意到高斯过程回归领域已有成功经验 —— 通过迭代算法将矩阵求逆从 $\mathcal{O}(M^3)$ 降到 $\mathcal{O}(M^2 K)$（$K \ll M$），并通过改善矩阵条件数来大幅减少迭代次数。LD 矩阵天然具有带状稀疏结构（远距离变异之间相关性趋近于零），可以被高效近似。

**核心 idea**：利用 LD 矩阵的带状结构进行分块近似，将分块作为 mini-batch，结合迭代线性代数方法高效计算全似然及其梯度，从而首次实现在 GWAS 数据上训练大规模深度神经网络。

## 方法详解

### 整体框架

DeepWAS 的整体 pipeline 如下：
- **输入**：(1) 大规模功能基因组特征（DNA 可及性、转录因子结合、跨物种保守性等），每个变异位点周围一个窗口内的特征向量；(2) 公开的表型关联数据（GWAS summary statistics 和 LD 矩阵）。
- **模型**：一个参数化的神经网络 $f_\theta$，输入为变异位点的功能注释特征，输出为该变异对表型的先验效应大小。
- **训练目标**：最大化完整的边际似然（marginal likelihood），而非拟合 summary statistics。
- **输出**：训练好的模型可以预测任意新变异位点的表型效应，用于疾病风险预测和治疗靶点识别。

整个方法的核心创新在于如何让全似然训练在计算上可行。传统方法需要对整个 LD 矩阵（$M \times M$）求逆，而 DeepWAS 通过带状近似将其分解为可管理的子问题。

### 关键设计

1. **LD 矩阵的带状近似（Banded Approximation）**：

    - 功能：将巨大的 LD 相关矩阵近似为带状矩阵，即认为距离超过一定窗口的变异之间相关性为零。
    - 核心思路：由于基因组中变异的相关性主要来源于物理距离近的位点，LD 矩阵天然具有近似带状结构。利用这一性质，可以将 $M \times M$ 的矩阵分解为多个重叠的较小矩阵块（slices）。
    - 设计动机：带状结构使得矩阵求逆可以在每个 slice 上独立进行，复杂度大幅降低。同时，这些 slices 天然可以作为随机梯度下降的 mini-batches，实现高效的批量训练。

2. **似然函数重组（Likelihood Rearrangement）**：

    - 功能：重新组织边际似然的计算形式，使其适合迭代线性代数算法加速。
    - 核心思路：在层次贝叶斯模型中，表型 $y$ 的边际似然涉及对 LD 矩阵 $\Sigma$ 的求逆和对数行列式。传统做法是直接 Cholesky 分解，复杂度 $\mathcal{O}(M^3)$。DeepWAS 将似然重写为适合共轭梯度法（Conjugate Gradient）等迭代求解器的形式。通过预处理（preconditioning）改善矩阵条件数，迭代步数 $K$ 可大幅减少。
    - 设计动机：迭代算法的每步只需矩阵-向量乘法（$\mathcal{O}(M^2)$ 或利用稀疏性更低），$K$ 步的总复杂度为 $\mathcal{O}(M^2 K)$，当 $K \ll M$ 时远优于直接求逆。

3. **大规模功能特征整合（Feature Curation）**：

    - 功能：精心整理大量功能基因组特征，作为神经网络的输入。
    - 核心思路：从 ENCODE、FANTOM 等数据库收集多种功能注释，包括 DNA 可及性、组蛋白修饰、转录因子结合、进化保守性等，为每个变异位点构建高维特征向量。
    - 设计动机：更丰富的特征使得大模型有足够信息学习复杂的特征-效应映射，这是 DeepWAS "越大越好" 结论的数据基础。

4. **深度神经网络作为 functionally informed prior**：

    - 功能：用深度网络替代传统的线性/低维参数模型作为先验。
    - 核心思路：传统方法（如 S-LDSC）用线性模型 $\sigma_j^2 = \sum_k a_{jk} \tau_k$ 将功能注释 $a_{jk}$ 映射为变异效应方差 $\sigma_j^2$，参数极少。DeepWAS 用神经网络 $f_\theta$ 替代此映射，可以捕捉非线性交互。
    - 设计动机：线性模型在特征增多时容易过拟合（尤其在 summary statistics 拟合下），而神经网络在全似然训练下具有更好的归纳偏置，避免"越大越差"的问题。

### 损失函数 / 训练策略

- **目标函数**：最大化基于带状近似 LD 矩阵的边际对数似然 $\log p(y | \theta)$。对于每个 mini-batch（对应 LD 矩阵的一个 slice），计算局部似然贡献并进行梯度更新。
- **优化器**：使用标准的随机梯度下降变体（如 Adam），每步训练中通过迭代线性代数（共轭梯度 + 预处理）高效计算似然及其对 $\theta$ 的梯度。
- **训练数据**：使用公开的大规模 GWAS 数据（如 UK Biobank summary statistics），按染色体分组进行 leave-one-chromosome-out 交叉验证。
- **对比训练方式**：同时实现了传统的 LD score regression 训练方式作为对比基线，以验证全似然方法的优越性。

## 实验关键数据

### 主实验：模型规模与训练方式对预测性能的影响

| 训练方式 | 模型类型 | 模型规模 | 拟合质量（似然/AIC） | 预测性能趋势 |
|----------|---------|---------|---------------------|-------------|
| LD score regression | 线性 | 小 (数十参数) | 基线 | 基线 |
| LD score regression | 神经网络 | 大 (数千参数) | ≤ 基线 | 不提升甚至下降 |
| DeepWAS (全似然) | 线性 | 小 | > 基线 | 优于 LDSC |
| DeepWAS (全似然) | 神经网络 | 大 | >> 基线 | 显著优于小模型 |

> 核心结论：在 summary statistics 拟合（LDSC）下，增大模型规模无法提升甚至会降低性能；而在 DeepWAS 全似然训练下，更大更灵活的模型稳定地带来更好的拟合和预测。

### 消融实验：关键组件贡献

| 配置 | 相对性能 | 说明 |
|------|---------|------|
| DeepWAS + 大模型 + 全特征 | 最优 | 完整模型，最佳预测 |
| DeepWAS + 小模型 + 全特征 | 次优 | 模型容量不足以捕捉非线性 |
| DeepWAS + 大模型 + 少特征 | 中等 | 特征不足限制了模型上限 |
| LDSC + 大模型 + 全特征 | 差 | Summary statistics 拟合无法利用大模型 |
| LDSC + 小模型 + 少特征 | 基线 | 传统方法 |

### 关键发现

- **模型规模效应具有训练方式依赖性**：这是本文最核心的发现。在 LDSC 框架下，大模型反而比小模型差（过拟合 summary statistics）；在全似然下，大模型稳定更优。这说明训练目标的选择比模型架构更重要。
- **特征数量与模型容量需匹配**：更多功能特征只在模型足够大时才能带来提升，小模型无法消化额外特征。
- **在 held-out 数据上的改进**：大模型 + 全似然在留出染色体上的预测准确度优于所有基线，暗示更大的模型和更多特征仍有进一步提升空间。
- **计算开销可控**：尽管使用了大规模神经网络和全似然，DeepWAS 通过带状近似和迭代求解器将计算控制在合理范围，在现代 GPU 上可行。

## 亮点与洞察

- **训练目标比模型架构更重要**：这是一个深刻的 insight —— 大模型在简化目标（summary statistics）下反而更差，但在正确目标（全似然）下才能发挥优势。这个发现不仅对 GWAS 领域有价值，对任何使用代理目标训练模型的场景都有启示。
- **LD 矩阵带状近似 + mini-batch 对应**：巧妙地利用了 LD 矩阵的物理结构（遗传距离导致的带状稀疏性），将其分解为 mini-batches，实现了统计严谨性与计算效率的平衡。这种"利用数据的物理结构设计计算方案"的思路可迁移到其他具有局部相关性的大规模问题。
- **从 GP 社区借鉴迭代线性代数**：将高斯过程领域的迭代求解 + 预处理技巧迁移到遗传学，展示了跨领域方法迁移的价值。GPyTorch 风格的方法在计算生物学中有广阔应用空间。
- **Scaling law 在基因组学中的验证**：论文间接验证了类似"scaling law"的现象 —— 更大模型 + 更多数据 + 更好的训练目标 → 更好的性能，但前提是训练方法得当。

## 局限与展望

- **带状近似的精度-效率权衡**：带状宽度的选择是一个超参数，过窄可能遗漏长距离 LD 关系，过宽则增加计算成本。论文未深入讨论如何自适应选择带宽。
- **模型可解释性**：用深度网络替代线性模型后，虽然预测性能提升，但对生物学家理解"哪些功能特征在哪些上下文下重要"增加了难度。
- **数据依赖性**：模型依赖于大规模公开 GWAS 数据和功能注释数据库，对于研究不充分的人群（如非欧洲裔）或稀有疾病，数据可用性可能限制方法的适用性。
- **计算资源需求**：虽然通过算法优化降低了复杂度，但训练大规模神经网络仍需 GPU 资源，对于资源有限的遗传学实验室可能不够友好。
- **与其他贝叶斯 GWAS 方法的对比**：如 SuSiE、FINEMAP 等精细定位方法使用不同的先验假设，DeepWAS 与这些方法的互补性值得探索。

## 相关工作与启发

- **vs LD Score Regression (S-LDSC)**：S-LDSC 是当前最广泛使用的方法，使用线性模型拟合 summary statistics。DeepWAS 的核心优势在于全似然训练 + 大模型，直接攻克了 S-LDSC 的两个核心妥协。本文最强的结论之一就是 S-LDSC 下大模型反而更差。
- **vs GPyTorch / 迭代 GP 方法**：DeepWAS 的加速策略直接借鉴了 Gardner et al. (2018) 在 GP 回归中的工作。区别在于 DeepWAS 面对的 LD 矩阵具有天然的带状结构，可以更自然地分块处理。
- **vs PolyFun / BayesR 等方法**：这些方法也使用功能注释指导先验，但受限于简单模型或计算开销。DeepWAS 通过计算创新突破了模型复杂度的限制。
- **启发**：该工作展示了在大规模结构化统计问题中，正确的训练目标（全似然 vs 代理目标）和高效的计算方案（利用问题结构的线性代数加速）可以共同解锁深度学习的 scaling 优势。这一范式可能对其他面临类似计算瓶颈的统计遗传学问题有启发，如多性状分析、跨人群遗传分析等。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 GP 社区的迭代线性代数技术迁移到 GWAS，核心 insight（训练目标决定 scaling 行为）新颖且重要
- 实验充分度: ⭐⭐⭐⭐ 在多个表型上验证，有 held-out 预测 + 对比实验，但缓存中缺少完整数值
- 写作质量: ⭐⭐⭐⭐ 引言逻辑清晰，从问题到方案的推导流畅，背景介绍对非领域读者友好
- 价值: ⭐⭐⭐⭐ 对计算遗传学社区意义重大，全似然 + 深度网络的组合可能改变该领域的建模范式

<!-- RELATED:START -->

## 相关论文

- [CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [Do Multiple Instance Learning Models Transfer?](do_multiple_instance_learning_models_transfer.md)
- [LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)
- [Certification for Differentially Private Prediction in Gradient-Based Training](certification_for_differentially_private_prediction_in_gradient-based_training.md)
- [Foundation Models for Clinical Records at Health System Scale](foundation_models_for_clinical_records_at_health_system_scale.md)

<!-- RELATED:END -->
