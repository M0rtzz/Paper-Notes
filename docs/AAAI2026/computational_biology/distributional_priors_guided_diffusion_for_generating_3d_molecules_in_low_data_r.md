---
title: >-
  [论文解读] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes
description: >-
  [AAAI 2026][计算生物][3D分子生成] 本文提出 GODD（Geometric OOD Diffusion Model），通过等变非对称自编码器捕捉分布结构先验来引导扩散模型的生成过程，使得在数据丰富的分子分布上训练的模型能够泛化到数据稀缺的分布，在 OOD 结构偏移基准上成功率提升 12.6%。
tags:
  - "AAAI 2026"
  - "计算生物"
  - "3D分子生成"
  - "扩散模型"
  - "分布外泛化"
  - "结构先验"
  - "药物设计"
---

# Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes

**会议**: AAAI 2026  
**arXiv**: [2404.00962](https://arxiv.org/abs/2404.00962)  
**代码**: 无  
**领域**: 医学图像 / 分子生成  
**关键词**: 3D分子生成, 扩散模型, 分布外泛化, 结构先验, 药物设计

## 一句话总结

本文提出 GODD（Geometric OOD Diffusion Model），通过等变非对称自编码器捕捉分布结构先验来引导扩散模型的生成过程，使得在数据丰富的分子分布上训练的模型能够泛化到数据稀缺的分布，在 OOD 结构偏移基准上成功率提升 12.6%。

## 研究背景与动机

**领域现状**：基于深度学习的 3D 分子生成是药物发现的前沿方向。扩散模型（如 EDM、GeoDiff）在 3D 分子生成中展现了强大的能力，可以生成符合化学规则的分子几何构型。

**现有痛点**：（1）3D 分子生成模型通常假设训练数据和测试数据来自相同分布，但在药物设计中，目标分子往往与训练数据分布存在显著差异——例如需要生成具有新骨架（scaffold）或新官能团（functional group）的分子；（2）这种分布外（OOD）生成问题在分子设计中尤其突出，因为有价值的新药分子几乎定义上就是"前所未见的"；（3）现有 OOD 分子生成研究主要关注属性偏移（如生成更高活性的分子），而忽略了结构偏移（如生成不同骨架的分子），后者是更基本也更具挑战性的问题。

**核心矛盾**：数据稀缺区域恰恰是最需要生成模型帮助探索的区域，但模型因缺乏这些区域的训练数据而无法有效生成。这是一个类似"鸡生蛋、蛋生鸡"的困境。

**本文目标**：（1）在数据丰富的分子分布上训练生成模型；（2）使模型能够泛化到数据稀缺的分子分布；（3）处理结构层面的分布偏移（而非仅属性偏移）。

**切入角度**：作者观察到，虽然不同分子分布在具体结构上差异很大，但它们的分布先验（distributional priors）可以被编码为结构粒度的（structural-grained）表示。通过学习区分不同分布的先验表示，可以引导扩散模型向目标分布生成，即使该分布几乎没有训练数据。

**核心 idea**：用等变非对称自编码器提取"分布结构先验"，用少量目标分布的样本编码其先验特征，然后将该先验注入扩散去噪过程，引导生成向目标分布的稀疏区域探索。

## 方法详解

### 整体框架

GODD 的输入是 3D 分子的原子坐标和类型。整体框架包含两个主要组件：（1）等变非对称自编码器——从分子数据中学习分布结构先验；（2）先验引导的扩散模型——在去噪过程中利用结构先验引导生成方向。训练时在数据丰富的分布上学习，推理时通过注入目标分布的先验编码来实现 OOD 生成。

### 关键设计

1. **等变非对称自编码器（Equivariant Asymmetric Autoencoder）**:

    - 功能：从分子数据中提取区分不同分布的结构先验表示。
    - 核心思路："非对称"体现在编码器和解码器的容量不同——编码器较大，解码器较小。编码器将分子映射到潜在空间中的分布先验表示，解码器从该表示重建分子。非对称设计迫使潜在表示捕捉分布层面的高级结构信息（如骨架模式、环结构特征），而非个体分子的细节。整个架构满足 SE(3) 等变性，确保先验表示对旋转和平移不变。
    - 设计动机：对称自编码器倾向于学习逐分子的重建，而非分布层面的特征。非对称设计通过信息瓶颈迫使模型提取更抽象的分布先验，这是实现跨分布泛化的关键。

2. **先验引导的扩散去噪**:

    - 功能：在扩散模型的去噪过程中注入结构先验，引导生成方向。
    - 核心思路：标准扩散模型的去噪网络预测噪声 $\epsilon_\theta(x_t, t)$，GODD 将其扩展为 $\epsilon_\theta(x_t, t, z)$，其中 $z$ 是从目标分布的少量样本编码得到的先验表示。先验表示通过交叉注意力或加法注入到去噪网络的中间层，使每一步去噪都朝着满足目标分布结构特征的方向偏移。
    - 设计动机：无条件扩散模型的生成方向由训练数据分布决定，无法突破训练分布的范围。通过注入目标分布的先验编码，即使该分布只有极少量数据，也能有效引导生成。

3. **分布偏移下的训练策略**:

    - 功能：确保模型学到的先验表示对未见分布也有效。
    - 核心思路：训练时模拟分布偏移场景——将可用数据划分为多个子分布，交替作为"源分布"和"目标分布"进行训练。这种课程学习策略使模型学会从少量样本中提取有用的分布先验，而非记忆特定分布。
    - 设计动机：如果只在单一分布上训练，模型可能过拟合到该分布的先验特征。多分布模拟训练增强了先验提取的泛化能力。

### 损失函数 / 训练策略

总损失包含：（1）扩散模型的标准去噪损失 $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, z)\|^2]$；（2）自编码器的重建损失；（3）先验表示的对比学习损失——相同分布的分子先验应该相近，不同分布的应该远离。

## 实验关键数据

### 主实验

在标准 OOD 结构偏移基准上评估，衡量分子有效性、唯一性和新颖性的综合成功率。

| 评估场景 | 指标 | GODD | 之前SOTA | 提升 |
|----------|------|------|----------|------|
| 骨架偏移 | 成功率 | 最佳 | -- | +12.6% |
| 环结构偏移 | 成功率 | 最佳 | -- | 显著 |
| 片段药物设计 | 综合指标 | 有竞争力 | -- | 良好泛化 |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| GODD (Full) | 最佳 | 完整模型 |
| w/o 非对称设计 | 下降 | 对称AE学不到分布先验 |
| w/o 先验引导 | 显著下降 | 无条件扩散无法做OOD生成 |
| w/o 等变性 | 下降 | 破坏了分子的几何对称性 |
| w/o 多分布训练 | 下降 | 先验提取的泛化能力减弱 |

### 关键发现

- 非对称自编码器的设计是学习有效分布先验的关键——对称版本倾向于记忆个体分子而非分布特征。
- 先验引导对 OOD 生成至关重要——无引导的扩散模型在 OOD 场景下成功率大幅下降。
- 等变性保证是必要的——破坏等变性会导致生成的分子几何结构不合理。
- 模型在片段药物设计任务上也展现了良好性能，证明方法的实用价值。

## 亮点与洞察

- **非对称自编码器捕捉分布先验**的设计非常精巧——通过信息瓶颈迫使模型学习分布层面而非实例层面的表示，是解决数据稀缺分布生成问题的关键洞察。
- **将 OOD 生成问题转化为条件生成问题**的思路具有广泛的迁移价值——其他领域（如蛋白质设计、材料设计）的 OOD 生成也可以借鉴此范式。
- 12.6% 的成功率提升在分子生成领域是显著的进步，具有实际药物发现的应用潜力。

## 局限与展望

- 实验主要在相对小规模的分子数据集上进行，在大规模药物数据库上的可扩展性有待验证。
- "少量目标分布样本"的定义比较模糊——需要多少样本才能有效编码分布先验？
- 未考虑分子的药理学性质约束（如 ADMET 性质），生成的分子可能在结构上新颖但在药性上不可行。
- 可以与虚拟筛选 pipeline 集成，在生成后过滤不满足药性约束的分子。

## 相关工作与启发

- **vs EDM/GeoDiff**: 标准 3D 分子扩散模型在同分布（in-distribution）生成上表现优异，但无法处理结构偏移。GODD 通过先验引导扩展了扩散模型的适用范围。
- **vs 属性导向生成方法**: 过去的 OOD 分子生成主要关注属性偏移（如 MOOD），GODD 首次系统性地处理结构偏移，覆盖了更基础的挑战。
- **vs VAE 系列方法**: VAE 通过潜在空间插值实现一定程度的 OOD 生成，但缺乏对结构先验的显式建模。GODD 的先验编码更加可控和可解释。

## 评分

- 新颖性: ⭐⭐⭐⭐ 非对称AE + 先验引导扩散 的组合方案新颖，OOD结构偏移的问题定义有创新
- 实验充分度: ⭐⭐⭐⭐ 多种偏移类型的评估 + 药物设计应用验证 + 消融实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 对AI驱动药物发现有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](../../ICLR2026/computational_biology/contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)
- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](../../ICLR2026/computational_biology/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [\[ICLR 2026\] Protein Counterfactuals via Diffusion-Guided Latent Optimization](../../ICLR2026/computational_biology/protein_counterfactuals_via_diffusion-guided_latent_optimization.md)

</div>

<!-- RELATED:END -->
