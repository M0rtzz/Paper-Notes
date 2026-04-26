---
title: >-
  [论文解读] Protein Counterfactuals via Diffusion-Guided Latent Optimization
description: >-
  [ICLR 2026][医学图像][反事实解释] 提出MCCOP框架，在蛋白质的连续序列-结构联合潜空间中，利用预训练扩散模型作为流形先验进行梯度引导的反事实优化，以最少突变（2-3个）生成生物学可信的蛋白质变体来翻转预测器输出，同时实现模型解释和蛋白质设计假说生成。
tags:
  - ICLR 2026
  - 医学图像
  - 反事实解释
  - 蛋白质工程
  - 扩散模型
  - 流形约束
  - 序列-结构嵌入
---

# Protein Counterfactuals via Diffusion-Guided Latent Optimization

**会议**: ICLR 2026  
**arXiv**: [2603.10811](https://arxiv.org/abs/2603.10811)  
**代码**: [GitHub](https://github.com/weroks/mccop)  
**领域**: 蛋白质工程 / 可解释AI  
**关键词**: 反事实解释, 蛋白质工程, 扩散模型, 流形约束, 序列-结构嵌入

## 一句话总结
提出MCCOP框架，在蛋白质的连续序列-结构联合潜空间中，利用预训练扩散模型作为流形先验进行梯度引导的反事实优化，以最少突变（2-3个）生成生物学可信的蛋白质变体来翻转预测器输出，同时实现模型解释和蛋白质设计假说生成。

## 研究背景与动机

**领域现状**：深度学习模型能高精度预测蛋白质性质（稳定性、荧光等），但仅作为"黑箱预言机"——当模型预测一个抗体不稳定时，工程师不知道做什么突变可以挽救。

**现有痛点**：(1) 朴素梯度优化在蛋白质上会产生对抗样本（满足预测器但无法折叠的序列）；(2) 离散搜索方法（遗传算法、爬山法）需要大量突变（8-11个），效率低且难以保证结构合理性；(3) 现有可解释方法（注意力可视化、特征归因）只回答"为什么"，不提供"怎么修"的可操作指导。

**核心矛盾**：蛋白质的离散序列与连续3D结构之间存在张力——梯度方法需要连续松弛，但直接在序列空间操作忽略了空间关系；同时需要保证优化轨迹不偏离生物学可行的流形。

**本文目标** 给定预测为"不具备目标性质"的蛋白质，如何找到最小且生物学合理的序列编辑使预测翻转？如何在稀疏性、有效性和生物学可信性之间取得平衡？

**切入角度**：在CHEAP编码器提供的连续序列-结构联合嵌入空间中操作，利用预训练扩散模型DiMA作为流形先验——扩散去噪步骤充当流形投影，与分类器引导的梯度优化交替执行。

**核心 idea**：在蛋白质的联合嵌入空间中，交替执行稀疏梯度下降和扩散模型流形投影，生成最少突变、结构合理的反事实蛋白质序列。

## 方法详解

### 整体框架
输入蛋白质序列 → CHEAP编码器映射到序列-结构联合潜空间 $z \in \mathbb{R}^{L' \times D}$ → 在潜空间中交替执行：(1) 稀疏梯度步骤（仅在top-$k$敏感位置更新）+ (2) DiMA扩散模型流形投影 → 直到预测器翻转 → CHEAP解码器映射回序列和结构。

### 关键设计

1. **预测器平滑化**:

    - 做什么：使分类器的梯度场更平滑，避免高频梯度引导优化走向对抗样本
    - 核心思路：四种互补机制——(a) 谱归一化约束所有线性层的Lipschitz常数；(b) Jacobian正则化惩罚 $\|\nabla_z f_\theta(z)\|_F^2$；(c) Softplus激活（$\beta=1$）替代ReLU；(d) 嵌入空间FGSM对抗增强——解码回原始序列的扰动以原始标签加入训练，教导对语义无关扰动的不变性。平滑后梯度范数降低最高4倍，AUROC维持或提升（稳定性: 0.94→0.98）。
    - 设计动机：非平滑分类器的高频梯度产生的扰动是"对抗性"的——无约束梯度下降100%生成对抗样本（解码回原始序列），平滑化是框架可行的前提。

2. **梯度敏感度稀疏掩码**:

    - 做什么：将梯度更新限制在最敏感的少数位置，实现序列级别的稀疏突变
    - 核心思路：计算每个位置的敏感度 $s_i = \|\nabla_{z_i} \mathcal{L}_{\text{CF}}\|_2$，构建二值掩码选择top-$k$位置 $M_i = \mathbf{1}[s_i \geq s_{(k)}]$。梯度仅在掩码位置应用，非掩码位置硬重置为原始嵌入。由于CHEAP解码器是逐位MLP（$\hat{S}_i$仅依赖$z_i$），潜空间的行级掩码直接等价于序列空间的稀疏性。
    - 设计动机：蛋白质工程的核心约束是最小突变——每个额外突变都增加实验成本和失败风险。梯度敏感度自然指示哪些位置对翻转预测最关键。

3. **扩散模型流形投影**:

    - 做什么：利用预训练的DiMA扩散模型将优化轨迹拉回蛋白质嵌入的合法流形
    - 核心思路：每步优化后，将当前嵌入部分扩散到噪声水平 $t_{\text{diff}}$，然后去噪获得流形投影 $\Pi_\phi(z'_t)$，以混合系数 $\alpha=0.3$ 融合：$z_{t+1} = (1-\alpha)z'_t + \alpha \Pi_\phi(z'_t)$。这是分类器引导的"反转"——用扩散不是为了生成，而是作为优化循环中的正则器。
    - 设计动机：无流形约束的梯度优化会脱离可行蛋白质空间，产生无法折叠的序列。扩散模型的去噪步骤天然具有将样本拉回数据流形的能力。

### 损失函数 / 训练策略
反事实优化目标：$\mathcal{L}_{\text{CF}}(z_t) = \log(1+\exp(m - \tilde{y} \cdot f_\theta(z_t))) + \lambda_{\text{dist}} \|z_t - z_{\text{orig}}\|_2^2$，其中 $m$ 是置信度裕量，$\lambda_{\text{dist}}$ 控制近端-有效性权衡。早停条件：$\sigma(\tilde{y} \cdot f_\theta(z_{t+1})) \geq \tau$ 且解码序列与原始不同。预测器训练使用shallow MLP + 上述四种平滑化。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 成功率↑ | 对抗率↓ | 编辑距离↓ |
|--------|------|------|----------|------|
| Stability | MCCOP | **1.00** | 0.03 | **2.32** |
| Stability | 遗传算法 | 0.55 | 0.00 | 7.76 |
| Stability | 爬山法 | 0.23 | 0.00 | 9.46 |
| Stability | 梯度下降（无约束） | 1.00 | **1.00** | - |
| Fluorescence | MCCOP | 0.19 | 0.01 | **1.37** |
| Activity | MCCOP | **1.00** | 0.02 | **2.46** |
| Activity | 遗传算法 | 0.17 | 0.00 | 6.24 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 无平滑化 | 100%对抗率，梯度下降完全失败 |
| 平滑后梯度范数 | 降低2-4倍，AUROC不降反升（Stability: +0.04） |
| $\alpha=0$（无流形投影） | 退化为对抗优化 |
| $\alpha=1$（完全投影） | 优化不稳定 |
| $\alpha=0.3$（默认） | 最佳平衡 |

### 关键发现
- 无约束梯度下降的对抗率为100%，验证了流形约束和平滑化的必要性
- MCCOP以2-3个突变实现近完美成功率，而离散方法需要8-11个突变
- 突变集中在功能相关区域：GFP荧光的发色团附近（残基63-69）、Ube4b活性的E2结合界面（残基66-71）
- MCCOP能精确恢复数据集中已知的反事实序列（荧光16例、活性18例、稳定性4例），部分来自held-out测试集

## 亮点与洞察
- 将反事实解释从图像/表格领域首次扩展到蛋白质，巧妙利用CHEAP的逐位解码特性实现潜空间稀疏性到序列空间稀疏性的直接映射。发现的突变与已知生物物理机制吻合（发色团堆积、疏水核心固化），说明预测器确实学到了有意义的序列-功能关系。
- 扩散模型作为"正则器"而非"生成器"的使用方式新颖，提供了一种通用的流形约束优化范式。

## 局限与展望
- 生物学可信性评估依赖计算代理（ESM3 pLDDT、回旋半径等），缺少湿实验验证
- CHEAP编码器-解码器引入的重建误差对远离ESMFold训练分布的蛋白质可能产生伪影
- 仅评估了二分类任务，连续回归目标（如精确的Tm值预测）需要替换损失函数

## 相关工作与启发
- **vs DVCE/DIME (图像反事实)**: 这些方法通过引导去噪生成反事实图像，MCCOP反转了这一思路——用扩散作为优化循环中的正则投影而非生成器，更适合蛋白质的离散-连续混合特性
- **vs 潜空间适应度优化 (ReLSO等)**: 这些方法寻找全局最优而非最小编辑，且需要任务特定的生成模型训练，MCCOP无需重训练且聚焦最小修改

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散引导的反事实解释引入蛋白质领域，框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种基线、物理化学可信性全面评估，但缺少湿实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，算法描述严谨，讨论部分坦诚
- 价值: ⭐⭐⭐⭐ 同时服务于模型可解释性和蛋白质工程，具有实际应用潜力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2026/medical_imaging/multiscale_structureguided_latent_diffusion_for_mu.md)
- [\[ICLR 2026\] Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/medical_imaging/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)

<!-- RELATED:END -->
