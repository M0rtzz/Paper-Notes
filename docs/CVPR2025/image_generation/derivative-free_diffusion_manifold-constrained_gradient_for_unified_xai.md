---
title: >-
  [论文解读] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI
description: >-
  [CVPR 2025][图像生成][无梯度梯度估计] 本文提出 FreeMCG，利用扩散模型生成流形上的粒子集合并结合集成卡尔曼滤波近似模型梯度在数据流形上的投影，首次统一了特征归因和反事实解释两大 XAI 任务，且仅需黑盒模型访问。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "无梯度梯度估计"
  - "扩散模型"
  - "集成卡尔曼滤波"
  - "反事实解释"
  - "特征归因"
---

# Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI

**会议**: CVPR 2025  
**arXiv**: [2411.15265](https://arxiv.org/abs/2411.15265)  
**代码**: 无  
**领域**: 可解释AI / XAI  
**关键词**: 无梯度梯度估计、扩散模型、集成卡尔曼滤波、反事实解释、特征归因

## 一句话总结
本文提出 FreeMCG，利用扩散模型生成流形上的粒子集合并结合集成卡尔曼滤波近似模型梯度在数据流形上的投影，首次统一了特征归因和反事实解释两大 XAI 任务，且仅需黑盒模型访问。

## 研究背景与动机

**领域现状**：可解释AI中最基础的工具是模型梯度——对输入的梯度用于特征归因（哪些像素重要）和反事实解释（改变输入使预测翻转）。但这两类任务在文献中一直独立处理。

**现有痛点**：梯度方法有三大局限：(1) 需要白盒访问（获取梯度必须知道模型权重），实际部署中很多模型是黑盒的；(2) 梯度解释不可靠——与对抗样本密切相关，产生的扰动远离数据流形；(3) 原始梯度无法统一用于特征归因和反事实解释。

**核心矛盾**：高维图像空间中梯度沿所有方向移动，但有意义的语义变化只在低维数据流形上。梯度中大量信息是"离开流形"的方向。

**本文目标**：设计不需模型梯度、产生流形上梯度近似、能同时用于特征归因和反事实解释的方法。

**切入角度**：梯度乘以粒子协方差 $C_{xx}$ 后自然投影到数据流形切空间（Theorem 1）。EnKF 理论表明 $C_{xx} \nabla_x f$ 可用 $C_{xf}$ 近似（Theorem 2），完全不需计算梯度。

**核心 idea**：利用扩散模型生成输入附近流形上粒子，计算输入-输出交叉协方差近似流形约束梯度，实现无需梯度的统一 XAI。

## 方法详解

### 整体框架
(1) 对输入 $x$ 添加噪声得 $x_t^{(k)}$；(2) 用 Tweedie 去噪得流形粒子 $x_{0|t}^{(k)}$；(3) 查询黑盒分类器得 $f(x_{0|t}^{(k)})$；(4) 计算 $C_{xf}$ 乘以 $(e_c - p)$ 得 FreeMCG 梯度；(5) 用于特征归因或反事实生成。

### 关键设计

1. **流形约束梯度的理论保证**:

    - 功能：证明协方差预条件梯度落在数据流形上
    - 核心思路：**Theorem 1** 证明粒子 $x^{(k)}$ 在流形上时，经验协方差 $C_{xx}$ 沿切空间扩展、沿法向收缩，因此 $C_{xx} \nabla_x \log p$ 被约束在切空间内，避免了对抗攻击方向。
    - 设计动机：从理论上保证 FreeMCG 产生有意义的、人类可理解的解释。

2. **EnKF 无梯度近似**:

    - 功能：在不计算模型梯度的前提下估计流形约束梯度
    - 核心思路：**Theorem 2** 证明 $C_{xf} = C_{xx} \nabla_x f^T + O(\delta^3)$，其中 $C_{xf} = \frac{1}{K}\sum_k (x^{(k)} - \bar{x})(f(x^{(k)}) - \bar{f})^T$ 只需粒子位置和模型前向输出。误差阶 $O(\delta^3)$ 在粒子局部集中时可忽略。
    - 设计动机：黑盒兼容性极大扩展适用范围。

3. **扩散模型生成流形粒子**:

    - 功能：高效生成满足近原始输入、在流形上、快速生成三个条件的粒子
    - 核心思路：对 $x$ 添加适量噪声得 $x_t$，用 Tweedie 公式 $D_\theta(x_t) \approx \mathbb{E}[x_0|x_t]$ 一步投影到流形上，比完整逆向采样快得多。
    - 设计动机：粒子质量直接决定近似精度和解释质量，扩散模型是当前最强数据流形建模器。

### 损失函数 / 训练策略
FreeMCG 无需训练。使用预训练扩散模型和分类器，仅涉及推理时前向计算。关键超参数为噪声水平 $\sigma_t$ 和粒子数 $K$。

## 实验关键数据

### 特征归因（ImageNet，Deletion/Insertion 指标）

| 方法 | 模型访问 | 性能 |
|------|---------|------|
| Vanilla Gradient | 白盒 | 基线 |
| Integrated Gradients | 白盒 | 较好 |
| FreeMCG (本文) | 黑盒 | SOTA |

### 反事实解释（ImageNet，FID/LPIPS/翻转率）

| 方法 | 模型访问 | 性能 |
|------|---------|------|
| DiME | 白盒 | 基线 |
| ACE | 白盒 | 中等 |
| FreeMCG (本文) | 黑盒 | SOTA |

### 关键发现
- FreeMCG 作为黑盒方法在两个任务上均超越白盒方法——流形约束比原始梯度更有效
- 反事实解释产生语义有意义的变化而非像素级对抗扰动
- 噪声水平 $\sigma_t$ 控制粒子分散程度与流形投影质量的 trade-off
- 粒子数 $K$ 增大改善精度但有递减效应

## 亮点与洞察
- **统一 XAI 框架**：首次将特征归因和反事实解释用同一数学框架统一，核心洞察"好的梯度应在流形上"非常优雅。
- **理论驱动设计**：Theorem 1 + Theorem 2 提供了完整理论链，方法有坚实数学基础。
- **黑盒胜白盒**：白盒梯度问题不在信息不够而在信息"太多"（含流形外方向），约束反而是优势。

## 局限与展望
- 需要预训练扩散模型，对小众数据域需额外训练
- 每次推理需 K 个粒子分别前向分类器，计算成本较高
- 理论假设流形局部线性，高曲率区域可能退化
- 目前仅在分类任务验证

## 相关工作与启发
- **vs Integrated Gradients**: IG 需白盒且沿直线积分，FreeMCG 黑盒且投影到流形
- **vs DiME/ACE**: 现有反事实方法需模型梯度或对抗鲁棒训练
- **vs EnKF 扩散引导**: Zheng et al. 用于图像逆问题，本文拓展到 XAI

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论优雅，统一框架首创，黑盒胜白盒令人意外
- 实验充分度: ⭐⭐⭐⭐ 两大任务均全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导精炼，动机层层递进
- 价值: ⭐⭐⭐⭐⭐ 对 XAI 领域有重要影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Decoupling Training-Free Guided Diffusion by ADMM](decoupling_training-free_guided_diffusion_by_admm.md)
- [\[CVPR 2025\] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](../../ICML2025/image_generation/local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)

</div>

<!-- RELATED:END -->
