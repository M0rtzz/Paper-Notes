---
title: >-
  [论文解读] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions
description: >-
  [ICML 2025][图像生成][退火流] 提出 Annealing Flow (AF)——基于连续归一化流（CNF）的高维多模态分布采样方法，用动态最优传输（OT）目标配合 Wasserstein 正则化训练，通过退火过程引导模式探索，在高维多模态设置中大幅优于现有 NF 和 MCMC 方法。
tags:
  - "ICML 2025"
  - "图像生成"
  - "退火流"
  - "连续归一化流"
  - "最优传输"
  - "多模态分布"
  - "高维采样"
---

# Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions

**会议**: ICML 2025  
**arXiv**: [2409.20547](https://arxiv.org/abs/2409.20547)  
**代码**: 无  
**领域**: 图像生成/采样  
**关键词**: 退火流, 连续归一化流, 最优传输, 多模态分布, 高维采样

## 一句话总结
提出 Annealing Flow (AF)——基于连续归一化流（CNF）的高维多模态分布采样方法，用动态最优传输（OT）目标配合 Wasserstein 正则化训练，通过退火过程引导模式探索，在高维多模态设置中大幅优于现有 NF 和 MCMC 方法。

## 研究背景与动机

**领域现状**：从高维多模态分布采样是统计贝叶斯推理和物理机器学习中的基础挑战。MCMC 方法在高维中混合慢，容易困在局部模式中；离散 NF 有模式坍缩风险。

**现有痛点**：
   - MCMC（包括 HMC）在仅两个模式的分布上就需要指数级步数来混合
   - 现有 NF 退火方法（如 Annealed Importance Sampling、Path-guided NFs）仍依赖 MCMC 辅助采样或大量离散化步骤的分数估计
   - 粒子优化方法（如 SVGD）对核选择和超参数敏感，计算复杂度随样本量多项式增长

**核心矛盾**：多模态分布的模式间距可能很大（在高维中尤甚），要求采样方法能"跳过"低概率区域到达不同模式，但现有方法在这方面能力有限。

**本文目标**：设计一种无需 MCMC 辅助的、训练高效且稳定的高维多模态采样方法。

**切入角度**：将连续归一化流的训练目标设计为动态最优传输 + 退火，使流学会逐步从简单分布迁移到目标分布，通过中间退火分布平滑地导航多模态景观。

**核心 idea**：退火 + OT 的组合——退火构建从简单到目标的中间分布桥梁，OT 目标确保流路径平滑且高效，Wasserstein 正则化约束流的复杂度。

## 方法详解

### 整体框架
Annealing Flow 的核心流程：
1. 构建退火中间分布序列 $\{f_k(x)\}$：$f_k(x) \propto \pi_0(x)^{1-\beta_k} \tilde{q}(x)^{\beta_k}$，其中 $\beta_0=0$（初始高斯）到 $\beta_K=1$（目标分布）
2. 将时间区间 $[0,1]$ 划分为 $K$ 段，每段学习一个从 $f_{k-1}$ 到 $f_k$ 的最优传输映射 $T_k$
3. 全部映射组合为完整的连续归一化流：$\pi_0 \to f_1 \to \cdots \to f_K = q$
4. 训练时不需要归一化常数 $Z_k$——所有计算仅依赖未归一化密度 $\tilde{f}_k$

### 关键设计

1. **动态最优传输目标 + Wasserstein 正则化**:

    - 功能：学习每段的最优传输映射 $T_k$
    - 核心思路：目标函数 = KL 散度（确保密度匹配） + $\gamma \int \|v_k\|^2 dt$（Wasserstein 正则化约束速度场平滑）
    - 数学形式：$T_k = \arg\min_T \{KL(T_{\#} f_{k-1} \| f_k) + \gamma \int \mathbb{E}[\|v_k\|^2] dt\}$
    - 设计动机：纯 KL 目标可能导致不规则的传输路径（速度场剧烈变化），Wasserstein 正则化迫使流走"直线"，大幅提升训练稳定性
    - 与现有方法的区别：不需要在训练中做分数估计和匹配（如 Path-guided NFs），也不需要 MCMC 辅助

2. **退火温度调度**:

    - 功能：构建从简单到复杂的中间分布桥梁
    - 核心思路：几何退火——$\tilde{f}_k(x) = \pi_0(x)^{1-\beta_k} \tilde{q}(x)^{\beta_k}$，低温时分布平坦（容易采样），高温时逐渐逼近目标
    - 设计动机：退火使流在早期"看到"所有模式（平坦分布下模式间障碍低），然后逐步集中到各模式
    - 关键优势：AF 需要的退火步数远少于传统方法——OT 目标使每步传输更高效

3. **理论保证（Theorem 1）**:

    - 功能：证明最优速度场的解析形式
    - 核心思路：无穷小最优速度场等于相邻退火密度的分数差 $v^* \propto \nabla \log f_k - \nabla \log f_{k-1}$
    - 设计动机：这个性质是 AF 的动态 OT 目标独有的——意味着流自然学会沿分数梯度方向传输，无需显式分数估计
    - 意义：连接了 OT 和分数匹配的理论，为 AF 的有效性提供了理论基础

### 损失函数 / 训练策略
- 每段损失：KL 散度 + Wasserstein 正则化（$\gamma$ 控制权衡）
- KL 项不需要归一化常数（通过 change-of-variables formula 计算流密度的对数）
- 速度场 $v_k(x(t), t)$ 由神经网络参数化
- 训练时仅需从当前流采样，不需要 MCMC
- 退火步数 $K$ 通常只需 3-5 步（远少于传统退火方法的数百步）

## 实验关键数据

### 主实验
多模态合成分布（高斯混合，最高 128 维）：

| 方法 | 2D GMM (W₂↓) | 32D GMM (W₂↓) | 128D GMM (W₂↓) | 退火步数 |
|------|-------------|---------------|----------------|---------|
| HMC | 0.012 | 失败 | 失败 | - |
| SVGD | 0.008 | 0.45 | 失败 | - |
| Path-guided NF | 0.005 | 0.12 | 0.89 | >50 |
| **AF (本文)** | **0.003** | **0.05** | **0.21** | **3-5** |

### Boltzmann 分布采样（物理应用）

| 目标分布 | AF (KL↓) | 最佳基线 (KL↓) | 提升 |
|---------|---------|-------------|------|
| 双阱势 (2D) | 0.02 | 0.08 | 75% |
| Lennard-Jones (30D) | 0.15 | 0.42 | 64% |
| 最不利分布 | 0.08 | 0.31 | 74% |

### 消融实验

| 配置 | 128D GMM (W₂) | 说明 |
|------|--------------|------|
| 无退火（直接 OT） | 1.52 | 模式坍缩 |
| 无 Wasserstein 正则化 | 0.78 | 训练不稳定 |
| K=1 退火步 | 0.89 | 退火不足 |
| K=3 退火步 | 0.24 | 接近最优 |
| **K=5 退火步** | **0.21** | **最优** |
| K=20 退火步 | 0.22 | 边际收益递减 |

### 关键发现
- AF 在 128D 多模态分布上仍然有效，而 MCMC 和 SVGD 完全失败
- 退火步数仅需 3-5 步即达到接近最优效果——远少于 Path-guided NF 的 50+ 步
- Wasserstein 正则化对训练稳定性至关重要——没有它，损失曲线剧烈波动
- Theorem 1 的理论预测（最优速度场 ∝ 分数差）与实验中学到的速度场高度一致
- 在"最不利分布"（deliberately adversarial distribution）上也表现良好，显示鲁棒性

## 亮点与洞察
- **退火 + OT 的组合**极其自然且强大——退火解决了 NF 的模式坍缩问题，OT 确保传输效率，两者互补而非冗余
- 仅需 3-5 步退火就超越需要 50+ 步的方法，说明动态 OT 目标的每一步传输质量远高于传统分数匹配
- Theorem 1 的理论洞察（最优速度场 = 分数差）具有独立学术价值——连接了最优传输和分数匹配两个看似不同的领域
- 对 Boltzmann 分布的采样能力使其可以直接应用于分子动力学和统计物理的实际问题
- 不需要 MCMC 辅助是重大实际优势——MCMC 的混合诊断是困难且不可靠的

## 局限与展望
- 需要知道目标分布的未归一化密度 $\tilde{q}(x)$——纯样本场景不适用
- 在超高维（>1000D）场景下的效果未验证
- 退火温度调度 $\{\beta_k\}$ 目前为手动设定，自适应调度可能进一步改进
- 速度场的神经网络参数化对网络架构选择有一定敏感性
- 连续归一化流的推理需要求解 ODE，比离散流略慢（但远快于 MCMC）

## 相关工作与启发
- **vs MCMC (HMC/PT)**: 无需混合，避免了高维中指数级混合时间的问题；无法像 MCMC 一样提供渐近精确保证
- **vs SVGD**: AF 不依赖核计算，避免了多项式复杂度；SVGD 在高维中粒子坍缩
- **vs Path-guided NF (Tian et al.)**: 需要 50+ 退火步和逐步分数估计，AF 仅需 3-5 步
- **vs Score-based Diffusion**: 扩散模型从数据学习分数，AF 从退火密度直接计算，适用场景不同（AF 用于已知密度的采样，Diffusion 用于数据驱动的生成）
- **启发**：退火 + 连续动力学的思路可推广到其他需要跨越能量障碍的采样/优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 动态OT+退火的组合及理论分析极具原创性
- 实验充分度: ⭐⭐⭐⭐ 合成+物理分布，高维验证，完整消融
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 高维多模态采样的重要突破，对统计物理和贝叶斯推理有广泛应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](../../NeurIPS2025/image_generation/physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](../../CVPR2025/image_generation/generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[ICML 2025\] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](distillation_of_discrete_diffusion_through_dimensional_correlations.md)

</div>

<!-- RELATED:END -->
