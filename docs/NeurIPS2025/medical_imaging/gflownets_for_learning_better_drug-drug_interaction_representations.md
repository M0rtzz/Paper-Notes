---
title: >-
  [论文解读] GFlowNets for Learning Better Drug-Drug Interaction Representations
description: >-
  [NeurIPS 2025][医学图像][药物相互作用] 针对药物-药物相互作用（DDI）预测中严重的类别不平衡问题，本文提出将 GFlowNet 与变分图自编码器（VGAE）结合，通过奖励引导的生成采样为稀有交互类型生成合成样本，从而增强模型在罕见但临床关键的交互类型上的预测能力。
tags:
  - NeurIPS 2025
  - 医学图像
  - 药物相互作用
  - GFlowNet
  - 变分图自编码器
  - 类别不平衡
  - 图生成
---

# GFlowNets for Learning Better Drug-Drug Interaction Representations

**会议**: NeurIPS 2025  
**arXiv**: [2508.06576](https://arxiv.org/abs/2508.06576)  
**代码**: 无  
**领域**: 医学AI / 药物发现  
**关键词**: 药物相互作用, GFlowNet, 变分图自编码器, 类别不平衡, 图生成

## 一句话总结

针对药物-药物相互作用（DDI）预测中严重的类别不平衡问题，本文提出将 GFlowNet 与变分图自编码器（VGAE）结合，通过奖励引导的生成采样为稀有交互类型生成合成样本，从而增强模型在罕见但临床关键的交互类型上的预测能力。

## 研究背景与动机

**领域现状**: DDI 预测是药物安全的关键任务，现有方法利用化学结构和生物网络等多种特征构建预测模型。

**现有痛点**: DDI 数据集中存在严重的类别不平衡——常见交互类型（如协同效应）主导数据集，而罕见但临床上重要的交互类型严重不足，导致模型在低频类别上表现差。

**核心矛盾**: 现有SOTA方法多将 DDI 预测视为二分类问题（有/无交互），忽略了不同交互类型的语义异质性，加剧了对频繁类型的偏向。

**本文目标**: 如何在不丢失频繁类别性能的前提下，提升模型对稀有交互类型的覆盖率和预测精度。

**切入角度**: 利用 GFlowNet 的奖励比例采样特性，有选择性地为低频类别生成合成 DDI 样本。

**核心 idea**: 用 GFlowNet 按"稀有度×可信度"的奖励函数生成合成 DDI 样本来增强训练数据的类别平衡性。

## 方法详解

### 整体框架

三阶段流水线：(1) 在原始不平衡数据上预训练 VGAE 学习药物嵌入；(2) 训练 GFlowNet 学习生成合成 DDI 的策略；(3) 用合成数据增强原始数据，重训 VGAE 得到最终模型。

### 关键设计

1. **变分图自编码器（VGAE）**:

    - **功能**: 学习药物的图结构化潜在表示，并预测 DDI 类型
    - **为什么**: 图结构能自然建模药物间的多关系交互网络
    - **怎么做**: 编码器为 R-GCN（关系图卷积网络），输出每个药物的变分后验 $q_\phi(\mathbf{z}_i | \mathcal{G}) = \mathcal{N}(\mathbf{z}_i | \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$；解码器用 DistMult 或 MLP 预测交互类型概率
    - **训练目标**: 最大化 ELBO，包含重构项和 KL 散度正则化

2. **GFlowNet 合成 DDI 生成**:

    - **功能**: 按奖励比例采样生成合成 DDI 三元组 $(d_i, d_j, t)$
    - **为什么**: GFlowNet 能学习使生成概率正比于奖励的策略，天然适合偏向稀有类别的采样
    - **怎么做**: 定义三步轨迹——选择交互类型 $t$ → 选择第一个药物 $d_i$ → 从 $d_i$ 的 $K$-近邻中选择第二个药物 $d_j$。奖励函数为：
    $R(t, d_i, d_j) = \underbrace{\left(\frac{1}{n_t + 1}\right)^\alpha}_{\text{稀有度}} \times \underbrace{p_\theta(t | \mathbf{z}_i, \mathbf{z}_j)}_{\text{可信度}}$
   其中 $n_t$ 为类型 $t$ 的频率，$\alpha$ 控制对稀有类的偏好强度
    - **区别**: 相比简单过采样/SMOTE，GFlowNet 生成的样本受制于 VGAE 的可信度评分，避免生成不合理的药物对

3. **轨迹平衡（TB）损失训练**:

    - **功能**: 训练 GFlowNet 的前向策略网络
    - **为什么**: TB 损失在完整轨迹上强制流匹配条件，确保采样分布收敛到与奖励成正比
    - **怎么做**: 
    $\mathcal{L}_{\text{TB}}(\psi) = \left(\log \frac{Z_\psi \prod_{s \to s' \in \tau} P_F(s'|s;\psi)}{R(s_f)}\right)^2$
   其中 $Z_\psi$ 为可学习的配分函数（总流）
    - **区别**: 相比 DB（详细平衡）损失，TB 作用于完整轨迹更稳定

### 训练策略

- **Stage 1**: 在原始不平衡数据上预训练 VGAE，得到药物嵌入 $\mathbf{Z}$ 和解码器 $p_\theta$
- **Stage 2**: 冻结 VGAE，用其嵌入和解码器计算奖励，训练 GFlowNet 策略
- **Stage 3**: 用训练好的 GFlowNet 采样 $N$ 个合成 DDI，与原始数据合并后重训 VGAE

## 实验关键数据

### 主实验

数据集：DrugBank（1,703 种药物，191,870 对药物对，86 种 DDI 类型）

| 指标 | 无 GFlowNet | 有 GFlowNet |
|------|:---:|:---:|
| AUROC | 0.99081 | 0.99071 |
| Accuracy | 0.96859 | 0.96792 |
| AUPRC | 0.98861 | 0.98922 |
| F1 Score | 0.98982 | 0.99914 |
| Shannon Entropy (SE) ↑ | 1.23 | **1.69** |
| Jensen-Shannon Divergence (JSD) ↓ | 0.35 | **0.12** |
| Coverage ↑ | 0.2441 | **0.7709** |

### 消融实验

本文未提供消融实验表格，但通过对比分类指标和多样性指标得出关键结论：

| 评价维度 | 观察 |
|---------|------|
| 分类性能 | AUROC/Accuracy 基本不变（~0.99），说明增加合成样本不损害主流类别 |
| 多样性 | SE 从 1.23 增至 1.69（+37%），分布更均匀 |
| 分布对齐 | JSD 从 0.35 降至 0.12（-66%），合成分布更接近真实分布 |
| 覆盖率 | 从 0.2441 增至 0.7709（+216%），大幅提升对稀有类的覆盖 |

### 关键发现

- 传统分类指标（AUROC、Accuracy）几乎未变，因为这些指标被高频类别主导
- 真正改善体现在多样性指标上：Coverage 从 24.4% 提升到 77.1%，意味着模型能覆盖绝大多数交互类型
- GFlowNet 的奖励设计确保生成样本既偏向稀有类（通过稀有度项）又保持可信（通过 VGAE 解码器评分）

## 亮点与洞察

- **问题切入精准**: 聚焦 DDI 预测中被忽视的类别不平衡问题，而非单纯追求整体分类精度
- **框架设计优雅**: GFlowNet 的奖励比例采样与数据增强需求天然契合，稀有度×可信度的复合奖励简洁有效
- **度量选择恰当**: 使用 Shannon Entropy 和 JSD 而非仅依赖分类指标，更能揭示类别分布的改善
- **模块化设计**: VGAE 和 GFlowNet 的解耦设计使框架可推广到其他不平衡图分类问题

## 局限与展望

- 仅在 DrugBank 单一数据集上验证，缺乏跨数据集泛化性实验
- 分类指标几乎无变化，可能需要按类别分别报告 F1 来更好展示稀有类的改善
- 未与其他数据增强方法（如 SMOTE、GANs、mixup）进行对比
- GFlowNet 的超参数（$\alpha$、候选集大小 $K$、合成样本数 $N$）的敏感性分析缺失
- 实验部分较薄弱，仅一个主表格，缺乏深入的消融和分析

## 相关工作与启发

- **GFlowNet (Bengio et al., 2023)**: 提供了奖励比例采样的理论基础
- **VGAE (Kipf & Welling)**: 变分图自编码器作为药物表示学习的骨干
- **DDI 预测**: 现有方法如 MFConv、GraphDTA 侧重二分类，本文补充了多类别视角
- **启发**: GFlowNet + 领域特定图模型的组合可推广到其他生物医学不平衡问题，如罕见疾病建模、不良反应预测

## 评分

- 新颖性: ⭐⭐⭐⭐ GFlowNet 用于 DDI 数据增强是新颖组合，奖励函数设计巧妙
- 实验充分度: ⭐⭐ 单数据集、缺乏对比方法和消融实验
- 写作质量: ⭐⭐⭐ 方法描述清晰，但实验部分过于简略
- 价值: ⭐⭐⭐ 思路有价值，但实验验证不够充分限制了说服力

<!-- RELATED:START -->

## 相关论文

- [Interpreting GFlowNets for Drug Discovery: Extracting Actionable Insights for Medicinal Chemistry](interpreting_gflownets_for_drug_discovery_extracting_actionable_insights_for_med.md)
- [Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [Compressing Biology: Evaluating the Stable Diffusion VAE for Phenotypic Drug Discovery](compressing_biology_evaluating_the_stable_diffusion_vae_for_phenotypic_drug_disc.md)
- [Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions](robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)
- [GenMol: A Drug Discovery Generalist with Discrete Diffusion](../../ICML2025/medical_imaging/genmol_a_drug_discovery_generalist_with_discrete_diffusion.md)

<!-- RELATED:END -->
