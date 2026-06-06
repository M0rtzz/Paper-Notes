---
title: >-
  [论文解读] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model
description: >-
  [ICLR 2026][图像生成][测试时缩放] VFScale提出无需外部验证器的测试时可缩放扩散模型，通过MRNCL损失和KL正则化改善能量景观使其内在能量函数可作为验证器，结合混合MCTS去噪实现高效搜索，在6×6训练的迷宫模型能解决88%的15×15迷宫，而标准扩散模型完全失败。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "测试时缩放"
  - "无验证器"
  - "能量函数"
  - "蒙特卡洛树搜索"
  - "扩散模型推理"
---

# VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model

**会议**: ICLR 2026  
**arXiv**: [2502.01989](https://arxiv.org/abs/2502.01989)  
**代码**: [https://github.com/AI4Science-WestlakeU/VFScale](https://github.com/AI4Science-WestlakeU/VFScale)  
**领域**: 扩散模型/推理  
**关键词**: 测试时缩放, 无验证器, 能量函数, 蒙特卡洛树搜索, 扩散模型推理

## 一句话总结
VFScale提出无需外部验证器的测试时可缩放扩散模型，通过MRNCL损失和KL正则化改善能量景观使其内在能量函数可作为验证器，结合混合MCTS去噪实现高效搜索，在6×6训练的迷宫模型能解决88%的15×15迷宫，而标准扩散模型完全失败。

## 研究背景与动机

**领域现状**：受人类System 2思维启发，LLM通过Chain-of-Thought在复杂推理中表现优秀。扩散模型通过迭代细化也适合推理任务，但在问题难度超出训练分布时性能急剧下降。

**现有痛点**：(1) 简单增加采样步数很快饱和（Du et al. 2024）；(2) 通过增加样本数量的测试时缩放依赖外部验证器提供密集评分信号，但推理任务的验证器难以获取；(3) 人类能进行无外部反馈的内省推理，现有方法与此有明显差距。

**核心矛盾**：扩散模型的能量函数本身可以作为验证器（因为score function是能量梯度的负数），但现有能量景观质量不足，低能量不一定对应高质量解（performance-energy consistency差）。

**本文目标**：如何利用扩散模型的内在能量函数替代外部验证器，实现无验证器的测试时缩放？

**切入角度**：双管齐下——训练侧改善能量景观，推理侧改善搜索效率。

**核心 idea**：通过MRNCL损失对齐能量值与样本质量的单调关系，通过hMCTS在去噪过程中平衡探索与利用。

## 方法详解

### 整体框架
训练侧：在标准MSE+Contrastive损失基础上，增加MRNCL损失（对齐能量与质量的单调关系）和KL正则化（平滑能量景观）。推理侧：混合MCTS去噪——早期用BoN广泛探索，后期用MCTS深度利用。

### 关键设计

1. **MRNCL损失（Monotonic-Regression Negative Contrastive Learning）**:

    - 功能：确保离ground truth越远的样本能量越高（performance-energy consistency）
    - 核心思路：对每个正样本 $x_0$，生成两个负样本 $x_0^-$ 和 $x_0^{--}$（后者距正样本更远）。在加噪后获取三点能量值 $(0, E_t^+), (l_{2,0}^-, E_t^-), (l_{2,0}^{--}, E_t^{--})$，做线性回归求斜率 $k_t$ 和截距 $b_t$
    - 损失：$\mathcal{L}_{\text{MRNCL}} = \mathbb{E}[\max(0, \gamma - k_t) + \sum \|E - \hat{E}\|_2^2]$
    - 设计动机：原始对比损失仅要求正样本为局部能量最小值，不约束负样本间的能量序关系

2. **KL正则化**:

    - $\mathcal{L}_{\text{KL}} = \mathbb{E}_{t, p_{\theta,t}}[E_{\text{stop-grad}(\theta)}(x)] + \mathbb{E}_{t, p_{\theta,t}}[\log p_{\theta,t}(x)]$
    - 第一项鼓励样本低能量，第二项最大化采样多样性（熵最大化）
    - 在每个去噪步 $t$ 上应用（区别于Du et al. 2021仅在终端）

3. **混合MCTS去噪（hMCTS）**:

    - 早期（噪声大时）用BoN：$L$ 个初始噪声并行去噪，防止过早淘汰有前景的路径
    - 后期（噪声小时）用MCTS：
        - Selection：UCB公式 $\text{UCB}(x_t, a_t) = Q(x_t, a_t) + c\sqrt{\frac{\ln N_i}{n_i}}$
        - Expansion：单步去噪+不同高斯噪声→$K$个分支
        - Simulation：用DDIM快速采样到 $x_0$，用 $E_\theta(\hat{x}_0)$ 作为reward（无需外部验证器）
        - Backpropagation：更新路径上所有节点的值
    - DDIM的子序列采样特性使simulation高效

### 完整训练目标
$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Contrast}} + \mathcal{L}_{\text{MRNCL}} + \mathcal{L}_{\text{KL}}$

## 实验关键数据

### 基础泛化能力（N=1推理）

| 方法 | Maze 6×6 | Maze 10×10 | Maze 15×15 | Sudoku D=33 | Sudoku D=25 |
|------|----------|------------|------------|-------------|-------------|
| Original | 1.000 | 0.578 | 0.063 | 0.320 | 0.023 |
| VFScale tr. | 1.000 | 0.775 | 0.281 | 0.195 | 0.008 |

### 测试时缩放（Maze 15×15）

| 方法 | N=1 | N=11 | N=41 | N=161 |
|------|-----|------|------|-------|
| Original BoN (Energy) | 0.063 | 0.047 | 0.078 | 0.109 |
| Original BoN (GT) | 0.063 | 0.125 | 0.133 | 0.172 |
| VFScale tr. BoN (GT) | 0.250 | 0.508 | 0.656 | 0.742 |
| **VFScale tr. hMCTS** | **0.281** | — | — | **0.880** |

### 关键发现
1. **原始训练方法的测试时缩放完全失败**：即使用ground truth验证器引导BoN，Maze 15×15成功率仅从6%提到17%
2. **能量景观质量是瓶颈**：原始模型performance-energy consistency仅约70%
3. **VFScale训练显著提升可缩放性**：同等BoN预算下，GT引导的成功率从17%提升到74%
4. **hMCTS进一步释放缩放潜力**：最终达到88%成功率（6×6训练→15×15测试）
5. **MRNCL和KL正则化互补**：去掉任一都会降低性能

## 亮点与洞察
- **范式创新**：将扩散模型的内在能量函数用作验证器，真正实现"无外部反馈的内省推理"
- **MRNCL的洞察深刻**：对比学习约束正负样本关系但忽略负样本间序关系，这是能量景观质量差的根本原因
- **hMCTS的设计精巧**：早期BoN宽搜+后期MCTS深搜，完美匹配去噪过程中噪声从大到小的特性
- **惊人的泛化能力**：6×6训练→88% 15×15测试，展示了测试时缩放的真正潜力

## 局限与展望
- MCTS的计算开销随分支数 $K$ 和回滚次数 $N_r$ 增长，需要仔细平衡
- 目前仅在网格/数独等结构化推理任务上验证，语言推理等更复杂场景待探索
- MRNCL中线性回归的选择可能不是最优的单调约束
- 可以探索自适应的BoN→MCTS切换点

## 相关工作与启发
- **vs Du et al. 2024**：他们的能量扩散模型在测试时缩放上饱和，VFScale解决了根本原因
- **vs Ma et al. 2025**：他们依赖外部验证器进行样本数缩放，VFScale完全内在化
- **vs AlphaGo/AlphaZero**：借鉴MCTS的核心思想但适配扩散去噪过程

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 无验证器测试时缩放的概念、MRNCL、hMCTS均为创新
- 实验充分度: ⭐⭐⭐⭐ Maze和Sudoku充分验证，但任务类型较单一
- 写作质量: ⭐⭐⭐⭐⭐ 动机→分析→解决方案的展开逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型的推理能力和测试时缩放开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICML 2026\] Linearizing Vision Transformer with Test-Time Training](../../ICML2026/image_generation/linearizing_vision_transformer_with_test-time_training.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)

</div>

<!-- RELATED:END -->
