---
title: >-
  [论文解读] SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling
description: >-
  [ICLR 2026][医学图像][可合成分子生成] SynCoGen 提出了一种结合掩码图扩散和流匹配的多模态生成框架，能够同时采样分子构建块反应图和3D原子坐标，在保证合成可行性的同时实现高质量的3D分子生成。
tags:
  - ICLR 2026
  - 医学图像
  - 可合成分子生成
  - 3D构象生成
  - 掩码图扩散
  - 流匹配
  - 药物发现
---

# SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling

**会议**: ICLR 2026  
**arXiv**: [2507.11818](https://arxiv.org/abs/2507.11818)  
**代码**: [GitHub](https://github.com/andreirekesh/SynCoGen)  
**领域**: 医学图像/分子生成  
**关键词**: 可合成分子生成, 3D构象生成, 掩码图扩散, 流匹配, 药物发现

## 一句话总结

SynCoGen 提出了一种结合掩码图扩散和流匹配的多模态生成框架，能够同时采样分子构建块反应图和3D原子坐标，在保证合成可行性的同时实现高质量的3D分子生成。

## 研究背景与动机

生成式分子设计在药物发现中具有重要价值，但一个关键瓶颈是**合成可及性**（synthetic accessibility）——生成的分子往往难以在实验室中合成。现有方法存在两大缺陷：

**基于模板的合成方法**（如 GFlowNet、Transformer）只能生成2D分子图，无法建模3D几何结构，而3D信息对药物的化学/生物性质至关重要。

**3D分子生成方法**（如 SemlaFlow、MiDi）可以生成原子坐标，但不考虑合成路径，生成的分子往往不可合成。

SynCoGen 旨在**弥合3D分子生成与合成可行性之间的鸿沟**，在一个统一框架中同时生成可合成的分子图和物理上合理的3D构象。

## 方法详解

### 整体框架

SynCoGen 将每个分子表示为三元组 $(X, E, C)$：
- $X \in \{0,1\}^{N \times (|\mathcal{B}|+1)}$：构建块（building block）身份编码
- $E \in \{0,1\}^{N \times N \times (|\mathcal{R}|V_{\max}^2+2)}$：构建块间的化学反应和反应中心编码
- $C \in \mathbb{R}^{N \times M \times 3}$：所有原子的3D坐标

训练将两种扩散方案统一在同一时间尺度上：对 $(X,E)$ 使用离散吸收（掩码）扩散，对 $C$ 使用连续的、可见性感知(visibility-aware)的流匹配。

### 关键设计

1. **SynSpace 数据集家族**: 构建了包含120万可合成分子和750万低能量构象的数据集。使用93个低成本构建块 + 19个高产率反应模板（核心集）和378个构建块 + 26个反应（扩展集）。通过迭代耦合构建块生成分子，并用半经验方法（GFN2-xTB）计算多个3D构象。

2. **掩码图扩散 + 流匹配联合生成**: 对离散的反应图使用 MDLM 框架的吸收扩散，对连续坐标使用条件流匹配（CFM），二者共享统一时间步。关键创新是设计了**可见性掩码** $S_t$，在训练时确保当构建块被掩码时，其对应原子坐标的信息也一致地被隐藏，使模型能正确处理可变原子数。

3. **训练时化学约束**: 

    - **无自环对角线**: 防止构建块与自身反应
    - **边数限制**: $n$ 个构建块最多 $n-1$ 条边
    - **兼容性掩码**: 一旦某条反应边确定，限制相连节点只能选择化学兼容的构建块，确保化学有效性

4. **SE(3)-等变骨干网络**: 基于 SemlaFlow 架构改造，支持同时预测构建块 logits $L_t^X, L_t^E$ 和坐标估计 $\hat{\tilde{C}}_0^t$。对药效团条件生成，将药效团作为"虚拟原子"引入，与真实原子在等变-不变动力学模块中交互。

### 损失函数 / 训练策略

总损失为加权和：
$$\mathcal{L} = \mathcal{L}_{\text{graph}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{pair}}$$

- $\mathcal{L}_{\text{graph}}$: 对 $(X, E)$ 的交叉熵损失
- $\mathcal{L}_{\text{MSE}}$: 掩码坐标的均方误差
- $\mathcal{L}_{\text{pair}}$: 短程成对距离正则项，保证3D结构合理性

## 实验关键数据

### 主实验

| 方法 | Valid.↑ | AiZyn.↑ | Synth.↑ | GFN-FF↓ | xTB↓ | PB↑ | FCD↓ |
|------|---------|---------|---------|---------|------|-----|------|
| **SynCoGen** | **96.7** | **50** | **72** | **3.01** | **-0.91** | **87.2** | **2.91** |
| SemlaFlow | 93.3 | 38 | 36 | 5.96 | -0.72 | 87.2 | 7.21 |
| JODO | 91.1 | 38 | 31 | 4.72 | -0.74 | 84.1 | 4.22 |
| EQGAT-diff | 85.9 | 37 | 24 | 4.89 | -0.73 | 78.9 | 6.75 |
| MiDi | 74.4 | 33 | 31 | 4.90 | -0.74 | 63.0 | 6.00 |

SynCoGen 在合成可行性（AiZyn +12, Synth +36 vs SemlaFlow）和构象能量质量上显著领先。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 去掉化学约束 | Valid. 大幅下降 | 化学敏感的图约束是最大性能贡献者 |
| 去掉自条件(self-conditioning) | Valid./FCD 下降 | 自条件对质量有重要贡献 |
| SemlaFlow重训于SynSpace | Valid. 72.0 | 证明性能提升来自训练策略而非数据/架构 |
| SynSpace-L (更大搜索空间) | 保持高质量 | 可扩展至更大构建块词汇 |

### 关键发现

- **片段链接(Fragment Linking)**: 在3个FDA批准药物靶标上，SynCoGen 生成的链接分子 docking 得分与天然配体相当或更优，且逆合成解析率达58-79%（DiffLinker为0%）
- **药效团条件生成**: 在10个PDB/LIT-PCBA靶标中，SynCoGen 在8/10上获得最佳 docking 得分，逆合成解析率比所有基线高15-65%
- **零样本构象生成**: 给定随机反应图时，可达到接近 ETKDG（RDKit）的构象生成质量

## 亮点与洞察

1. **"合成即生成"的范式突破**: 将合成约束直接编码进生成过程，而非后处理筛选，从根本上保证可合成性
2. **多分辨率多模态**: 在构建块级做离散扩散，在原子级做连续流匹配——"两分辨率统一时间"的设计非常优雅
3. **零样本条件设计**: 不需要针对特定靶标重训练，单一摊销(amortized)模型即可完成多种药物设计任务
4. **数据贡献突出**: SynSpace 包含120万+可合成分子和750万+构象，是该领域重要的公开资源

## 局限与展望

1. **构建块词汇有限**: 当前最大词汇仅378个构建块，化学多样性受限
2. **不支持大环(macrocycle)**: 训练时约束限制了边数，无法生成大环分子
3. **缺乏湿实验验证**: 尚未对生成的分子进行实际合成和生物活性测试
4. **合成解析率上限**: AiZynthFinder 本身只能解析50-70%的已知药物分子，评估可能低估真实合成可行性

## 相关工作与启发

- 与 **CGFlow** (GFlowNet + 流匹配) 相比，SynCoGen 无需对每个靶标重训、无需外部奖励函数
- **SemlaFlow** 提供了等变骨干架构基础，但缺乏合成约束
- **DiffLinker** 是专用的片段链接模型，但生成分子完全不可合成
- 对计算化学和AI制药领域都有重要启发：合成约束可以被"内嵌"到生成模型中

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个同时建模合成路径和3D结构的联合生成框架，范式创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 无条件生成、片段链接、药效团条件生成三大任务，消融充分
- **写作质量**: ⭐⭐⭐⭐ 整体清晰，但符号系统较复杂
- **价值**: ⭐⭐⭐⭐⭐ 对AI驱动药物发现具有直接应用价值，数据集和代码均开源

<!-- RELATED:START -->

## 相关论文

- [Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](../../ICML2025/medical_imaging/scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](../../AAAI2026/medical_imaging/apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [AntigenLM: Structure-Aware DNA Language Modeling for Influenza](antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/medical_imaging/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [Geometric Representation Condition Improves Equivariant Molecule Generation](../../ICML2025/medical_imaging/geometric_representation_condition_improves_equivariant_molecule_generation.md)

<!-- RELATED:END -->
