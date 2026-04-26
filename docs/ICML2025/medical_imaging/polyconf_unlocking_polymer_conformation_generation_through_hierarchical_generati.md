---
title: >-
  [论文解读] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models
description: >-
  [ICML 2025][医学图像][聚合物构象生成] 提出 PolyConf——首个专为聚合物构象生成设计的层次化生成框架：Phase 1 用掩码自回归模型（MAR）+ 扩散过程在随机顺序下生成各重复单元的局部构象，Phase 2 用 SO(3) 扩散模型生成朝向变换以将局部构象组装为完整聚合物构象；同时构建了首个聚合物构象基准 PolyBench（5万+聚合物，~2000原子/构象），在所有结构和能量指标上均大幅超越现有方法 25%+。
tags:
  - ICML 2025
  - 医学图像
  - 聚合物构象生成
  - 层次化生成模型
  - 掩码自回归模型
  - SO(3)扩散模型
  - 分子动力学
---

# PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models

**会议**: ICML 2025  
**arXiv**: [2504.08859](https://arxiv.org/abs/2504.08859)  
**代码**: [https://polyconf-icml25.github.io](https://polyconf-icml25.github.io)（代码、模型、数据均公开）  
**领域**: 分子建模 / AI4Science  
**关键词**: 聚合物构象生成, 层次化生成模型, 掩码自回归模型, SO(3)扩散模型, 分子动力学

## 一句话总结

提出 PolyConf——首个专为聚合物构象生成设计的层次化生成框架：Phase 1 用掩码自回归模型（MAR）+ 扩散过程在随机顺序下生成各重复单元的局部构象，Phase 2 用 SO(3) 扩散模型生成朝向变换以将局部构象组装为完整聚合物构象；同时构建了首个聚合物构象基准 PolyBench（5万+聚合物，~2000原子/构象），在所有结构和能量指标上均大幅超越现有方法 25%+。

## 研究背景与动机

聚合物（polymer）是由大量相同或相似单体通过共价键结合形成的大分子，在包装材料、电子器件等领域有广泛应用。**聚合物构象生成**——即在给定 2D 聚合物分子图的条件下生成稳定的 3D 聚合物结构——是研究聚合物性质的基础出发点。

现有构象生成方法面临的核心挑战：

**小分子方法难以扩展**：GeoDiff、TorsionalDiff 等方法设计用于小分子（原子数较少），直接应用于含数千原子的聚合物时性能显著下降

**蛋白质方法不适用**：蛋白质有统一的骨架框架（N-Cα-C-O 结构）和强方向性分子内相互作用，聚合物缺乏这些先验约束，具有更高的柔性和更少的有序性

**数据稀缺**：传统分子动力学（MD）模拟计算成本极高，导致聚合物构象数据集严重匮乏

**重复单元构象多样**：虽然同一聚合物的重复单元共享相同的 SMILES 字符串，但各单元在 3D 空间中的排列差异显著，不能简单地将聚合物建模为单一预定义重复单元构象的刚性组装

## 方法详解

### 整体框架

PolyConf 采用**层次化生成框架**，核心思想是将聚合物构象分解为两层：

$$p(\mathcal{C}|\mathcal{G}) = p(\mathcal{C}^u|\mathcal{G}) \cdot p(\mathcal{O}|\mathcal{G}, \mathcal{C}^u)$$

其中 $\mathcal{C}^u = \{C_i^u\}_{i=1}^{N_u}$ 是重复单元构象集合，$\mathcal{O} = \{\mathcal{O}_i\}_{i=1}^{N_u}$ 是对应的朝向变换集合。整个生成过程分为两阶段：

- **Phase 1**：基于 2D 分子图 $\mathcal{G}$ 生成各重复单元的局部构象 $\mathcal{C}^u$
- **Phase 2**：在已知 $\mathcal{G}$ 和 $\mathcal{C}^u$ 的条件下，生成朝向变换 $\mathcal{O}$ 以组装完整构象

### 基于帧的聚合物表示（Frame-based Representation）

每个聚合物 $N$ 个原子表示为 2D 图 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$，构象为 $C \in \mathbb{R}^{N \times 3}$。关键设计：

- 扩展标准重复单元定义：将相邻重复单元的两个关键原子（atom-1 和 atom-4）纳入当前单元的构象中，每个单元原子数为 $\frac{N}{N_u}+2$
- 受蛋白质残基建模策略启发，从重复单元构象中提取**帧（frame）**
- 朝向变换 $\mathcal{O}_i = (R_i, t_i)$：旋转 $R_i \in \mathbb{R}^{3\times3}$ 通过 Gram-Schmidt 正交化从关键原子向量计算，平移 $t_i \in \mathbb{R}^3$ 对应 atom-3 的 3D 坐标
- 相邻重复单元在键合原子处自然重叠（atom-1 与上一单元的 atom-3 对齐），因此 Phase 2 只需生成旋转，平移可由重叠原子坐标直接推导

### Phase 1：重复单元构象生成

该阶段集成了三个核心模块：

#### (1) 多模态重复单元编码器

$$X^u = \mathcal{M}(\mathcal{G}, \{C_i^u\}) = \text{Concat}_1(\mathcal{M}_{2d}(\mathcal{G}),\ \text{Concat}_0(\{\mathcal{M}_{3d}(C_i^u)\}))$$

- **2D 编码器** $\mathcal{M}_{2d}$：采用 MolCLR 架构，编码整个聚合物图 $\mathcal{G}$，得到 $X^{2d} \in \mathbb{R}^{N_u \times D_{2d}}$
- **3D 编码器** $\mathcal{M}_{3d}$：采用 Uni-Mol（SE(3)-不变），编码每个重复单元构象 $C_i^u$，得到 $X_i^{3d} \in \mathbb{R}^{1 \times D_{3d}}$
- 将 2D 和 3D 嵌入拼接得到多模态嵌入 $X^u \in \mathbb{R}^{N_u \times D_u}$
- 该编码器在 Phase 1 中可训练，在 Phase 2 中冻结

#### (2) 掩码自回归建模（MAR）

以**随机顺序**生成重复单元构象，捕获复杂交互而非简单序列依赖：

$$p(\mathcal{C}^u|\mathcal{G}) = \prod_{k=1}^{K} p(\mathcal{C}_k^u|\mathcal{G}, \{\mathcal{C}_i^u\}_{i=1}^{k-1})$$

训练时：

- 定义随机排列 $\pi$，随机采样遮掩率 $\tau \in [0,1]$
- 将遮掩后的已知嵌入 $X_{\text{known}}^u$ 送入 MAR 编码器 $\Phi$（标准 Transformer + 双向注意力）
- MAR 解码器 $\Psi$ 输出被遮掩重复单元的表示 $Z_{\text{mask}}^u \in \mathbb{R}^{\tau N_u \times D_m}$

$$Z_{\text{mask}}^u = \Psi(\Phi(X_{\text{known}}^u))$$

#### (3) 扩散损失

使用扩散模型表示被遮掩重复单元构象的条件概率分布。扩散过程定义在**扭转角空间**：

$$\mathcal{L}_{\text{phase-1}} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(C_t^u | t, z^u)\|^2\right]$$

$$C_t^u = \sqrt{\bar{\alpha}_t} C^u + \sqrt{1-\bar{\alpha}_t}\epsilon$$

- $z^u$ 是 $Z_{\text{mask}}^u$ 中对应行的解码表示，作为扩散过程的条件
- 去噪网络 $\epsilon_\theta$ 采用 TorsionalDiff 的架构
- 有效整合了自回归建模和扩散过程的能力

### Phase 2：朝向变换生成

#### SO(3) 扩散生成旋转

由于重叠原子的存在，只需生成旋转矩阵 $R$，平移可由重叠原子坐标直接推导：

$$\hat{R}^{(0)} = \varphi(\mathcal{O}^{(t)}, t, E^u), \quad \mathcal{O}^{(t)} = (R^{(t)}, T^{(t)})$$

- 去噪网络 $\varphi$ 采用与 FrameDiff (Yim et al., 2023) 相同的架构
- 条件信息 $E^u \in \mathbb{R}^{N_u \times D_e}$ 来自 MAR 编码器输出（代表 $\mathcal{G}$ 和 $\mathcal{C}^u$ 的信息）
- $R^{(t)}$ 通过 SO(3) 上的前向扩散获得，$T^{(t)}$ 由施加 $R^{(t)}$ 后重叠原子对齐计算

### 损失函数 / 训练策略

**Phase 1 损失**（扭转角空间扩散）：

$$\mathcal{L}_{\text{phase-1}} = \mathbb{E}_{\epsilon, t}\left[\|\epsilon - \epsilon_\theta(C_t^u | t, z^u)\|^2\right]$$

**Phase 2 损失**（SO(3) 旋转预测）：

$$\mathcal{L}_{\text{phase-2}} = \frac{1}{N_u}\sum_{i=1}^{N_u}\|\hat{R}_i^{(0)} - R_i\|^2$$

**两阶段训练**：Phase 1 训练编码器 + MAR + 扩散；Phase 2 冻结编码器，仅训练 SO(3) 扩散模型。

### 组装过程

1. 将生成的重复单元构象 $\hat{C}_i^u$ 通过逆变换回标准坐标系
2. 施加生成的旋转 $\hat{R}_i$ 得到旋转后构象 $\hat{C}_i^{u,\text{rot}}$
3. 通过对齐重叠原子计算平移 $\hat{t}_i$
4. 拼接去除重叠原子后得到完整聚合物构象 $\hat{C} \in \mathbb{R}^{N \times 3}$

## 实验关键数据

### 数据集 PolyBench

首个聚合物构象基准，通过分子动力学模拟构建：
- 50,000+ 聚合物，每个构象约 2,000 原子
- NVT 集成，298K/1atm，5ns (5M步) MD 模拟
- 力场：General AMBER Force Field (GAFF)，GROMACS 引擎
- 划分：训练 ~46k，验证 ~5k，测试 ~2k
- 重复单元数：大部分 20-100，少部分超过 100

### 主实验

| 方法 | S-MAT-R (Mean/Med) | S-MAT-P (Mean/Med) | E-MAT-R (Mean/Med) | E-MAT-P (Mean/Med) |
|------|---------------------|---------------------|---------------------|---------------------|
| GeoDiff | 93.119 / 89.767 | 95.259 / 91.869 | 21.249 / 18.106 | 64.871 / 58.711 |
| TorsionalDiff | 53.210 / 38.710 | 70.679 / 60.744 | 2.605 / 1.034 | 8.402 / 6.851 |
| MCF | 248.432 / 242.866 | 258.891 / 253.239 | — | — |
| ET-Flow | 94.057 / 90.475 | 96.896 / 92.877 | 6.733 / 5.186 | 53.528 / 30.125 |
| **PolyConf** | **35.021 / 24.279** | **46.861 / 37.996** | **0.933 / 0.359** | **6.191 / 4.122** |

- 比最佳基线 TorsionalDiff 在**所有指标上提升超 25%**
- 注意 TorsionalDiff 还额外使用了初始聚合物结构作为输入（biased advantage）

### 可扩展性实验（重复单元数翻倍，约 4000 原子）

| 方法 | S-MAT-R (Mean/Med) | S-MAT-P (Mean/Med) | E-MAT-R (Mean/Med) | E-MAT-P (Mean/Med) |
|------|---------------------|---------------------|---------------------|---------------------|
| GeoDiff | 184.668 / 175.607 | 186.861 / 177.645 | 52.614 / 47.872 | 112.883 / 105.197 |
| TorsionalDiff | 119.289 / 94.075 | 146.816 / 126.932 | 5.219 / 2.216 | 11.692 / 9.227 |
| ET-Flow | 186.132 / 176.370 | 188.725 / 178.977 | 15.331 / 12.465 | 65.116 / 41.642 |
| **PolyConf** | **65.040 / 41.992** | **84.626 / 64.445** | **1.259 / 0.609** | **5.785 / 4.434** |

- 能量指标相比 TorsionalDiff 提升超 **50%**
- 得益于 MAR 建模，PolyConf 展现出优异的可扩展性

### 效率对比

| 方法 | 平均生成时间 |
|------|-------------|
| GeoDiff | 3.54 min |
| MCF | 1.12 min |
| TorsionalDiff | 0.45 min |
| **PolyConf** | **0.40 min** |

### 关键发现

1. **层次化分解有效**：将聚合物构象分解为局部构象 + 朝向变换的两阶段策略，比直接端到端生成整体构象效果显著更好
2. **随机顺序生成优于固定顺序**：MAR 的随机排列机制捕获了重复单元间的复杂交互
3. **基线方法严重失效**：小分子方法（GeoDiff、ET-Flow）在聚合物上性能大幅下降，MCF 甚至无法计算能量指标
4. **生成质量的可视化证据**：PolyConf 生成的构象更接近参考构象的展开松弛状态，而 TorsionalDiff 即便有初始结构先验仍难以捕捉

## 亮点与洞察

1. **开创性的问题定义**：首次将聚合物构象生成作为独立任务提出，填补了小分子与蛋白质之间的重要空白
2. **优雅的层次化分解**：利用聚合物的重复单元结构特性，将复杂大分子问题自然分解为可处理的子问题
3. **MAR + Diffusion 的有效结合**：将最新的掩码自回归范式与扩散过程融合，在构象生成中首次应用
4. **SO(3) 扩散的巧妙应用**：利用重叠原子约束，将 SE(3) 变换简化为仅需生成旋转，降低了问题复杂度
5. **完整的基准贡献**：不仅提出方法，还投入大量资源构建了首个聚合物构象数据集和标准化评估协议

## 局限与展望

1. **仅限线性均聚物**：当前工作仅关注由相同单体组成的线性聚合物，未涉及共聚物、混合物等更复杂体系
2. **未考虑 2D 拓扑结构**：如交叉链接、分支等聚合物结构暂未建模
3. **力场精度限制**：使用 GAFF 力场而非 DFT 计算（受限于计算成本），在性质/能量预测精度上有一定局限
4. **可探索流匹配等新范式**：作者提到未来可引入 flow-based 生成模型进一步改进
5. **缺少下游任务验证**：未评估生成构象在聚合物性质预测等实际应用中的效果

## 相关工作与启发

- **小分子构象生成**：GeoDiff（欧氏空间扩散）、TorsionalDiff（扭转角扩散）、MCF、ET-Flow → 聚合物规模和柔性使其失效
- **蛋白质构象生成**：AlphaFold 系列、FrameDiff → 依赖蛋白质特有的进化信息和骨架约束，不适用于聚合物
- **关键启发**：MAR 范式 (Li et al., 2024) 及 SE(3) 扩散 (Yim et al., 2023) 的组合应用；聚合物的重复性结构特征天然适合层次化建模

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个聚合物构象生成方法，问题定义和层次化分解均有很强原创性
- **实验充分度**: ⭐⭐⭐⭐ — 多维度评估（结构/能量/效率/可扩展性），但缺少消融实验细节
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，公式推导流畅，图示直观
- **价值**: ⭐⭐⭐⭐⭐ — 开辟新领域，代码数据全开源，对聚合物建模研究影响深远

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Aligning Protein Conformation Ensemble Generation with Physical Feedback](aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[ICML 2025\] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models](deepseq_high-throughput_single-cell_rna_sequencing_data_labeling_via_web_search-.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](../../CVPR2026/medical_imaging/unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[ICML 2025\] ComRecGC: Global Graph Counterfactual Explainer through Common Recourse](comrecgc_global_graph_counterfactual_explainer_through_common_recourse.md)

<!-- RELATED:END -->
