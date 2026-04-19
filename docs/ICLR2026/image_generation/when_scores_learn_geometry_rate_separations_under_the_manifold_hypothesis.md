---
title: >-
  [论文解读] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis
description: >-
  [ICLR 2026][图像生成][Score Learning] 在流形假设下揭示score学习中几何信息与分布信息的尺度分离现象——流形几何信息强度为 $\Theta(\sigma^{-2})$，比分布信息强 $O(\sigma^{-2})$ 倍，由此证明扩散模型的成功主要来自学习数据流形而非完整分布，并提出一行代码修改即可生成流形上的均匀分布。
tags:
  - ICLR 2026
  - 图像生成
  - Score Learning
  - 流形假设
  - 几何学习
  - 分布学习
  - 率分离
  - 均匀采样
---

# When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

**会议**: ICLR 2026  
**arXiv**: [2509.24912](https://arxiv.org/abs/2509.24912)  
**代码**: 无  
**领域**: 扩散模型理论  
**关键词**: Score Learning, 流形假设, 几何学习, 分布学习, 率分离, 均匀采样

## 一句话总结
在流形假设下揭示score学习中几何信息与分布信息的尺度分离现象——流形几何信息强度为 $\Theta(\sigma^{-2})$，比分布信息强 $O(\sigma^{-2})$ 倍，由此证明扩散模型的成功主要来自学习数据流形而非完整分布，并提出一行代码修改即可生成流形上的均匀分布。

## 研究背景与动机

**领域现状**：Score-based方法（扩散模型、贝叶斯逆问题等）通常被解释为在低噪声极限 $\sigma \to 0$ 时学习数据分布。流形假设——数据支持在高维空间的低维流形上——被广泛采用。

**现有痛点**：(1) 低温极限下score估计极其困难，实践中需要大量post-training工程来稳定；(2) 现有理论分析将几何和分布信息混合在误差界中，未揭示尺度分离；(3) 即使score近似很好，恢复的分布可能在流形上是任意的。

**核心矛盾**：分布学习（完全恢复 $\mu_{\text{data}}$）要求 $o(1)$ 的score精度，但实际score误差通常远大于此。然而扩散模型仍能生成逼真样本——这说明它们可能并非真正学会了分布。

**本文目标**：(1) 为什么扩散模型在score不完美时仍然有效？(2) 能否利用这种不完美的score做有用的事情（如均匀采样）？

**切入角度**：在 $\sigma \to 0$ 时对score函数进行渐近展开，分离几何项（leading order, $\Theta(\sigma^{-2})$）和分布项（sub-leading order, $\Theta(1)$）。

**核心 idea**：score函数中流形几何信息比分布信息强 $\sigma^{-2}$ 倍，因此几何学习允许 $o(\sigma^{-2})$ 大的误差，远比分布学习所需的 $o(1)$ 宽松。

## 方法详解

### 核心理论

1. **Score的渐近展开（Theorem 3.1的核心思想）**:
    - 在 $\sigma \to 0$ 时，score function可分解为：
    - **Leading order** $\Theta(\sigma^{-2})$：几何项，本质上是到流形的投影算子（将点拉向最近流形点）
    - **Sub-leading order** $\Theta(1)$：分布项，编码流形上的密度信息 $p_{\text{data}}$
    - 两项之间有 $\sigma^{-2}$ 的间隔——这就是"率分离"

2. **三个推论**:

   **(i) 流形集中 vs 分布恢复（Theorem 4.1）**:
   - 流形集中（生成样本落在流形附近）：score误差 $o(\sigma^{-2})$ 即可
   - 分布恢复（精确恢复 $\mu_{\text{data}}$）：score误差必须 $o(1)$
   - 差距 $O(\sigma^{-2})$，在小 $\sigma$ 下巨大

   **(ii) 均匀分布生成（Theorems 5.1-5.2）**:
   - 仅需一行代码修改：将采样算法中的score替换为 $\tilde{s}_\sigma(x) = \nabla \log p_\sigma(x) + \frac{1}{\sigma^2} \nabla d_\mathcal{M}(x)$
   - 所需score精度：$o(\sigma^{-2})$，与几何学习相同
   - 均匀分布是流形几何的最大熵表示，在推理时不需要微调

   **(iii) 贝叶斯逆问题（Theorem 6.1）**:
   - 当先验为均匀分布时，后验采样只需 $o(\sigma^{-2})$ score精度
   - 当先验为 $\mu_{\text{data}}$ 时，需要更严格的 $o(1)$ 精度
   - 这意味着最大熵先验对score误差更鲁棒

### 关键数学工具

- **Gaussian Smoothed Measure**：$\mu_\sigma = \text{law}(X + \sigma Z)$（VE）或 $\text{law}(\sqrt{1-\sigma^2}X + \sigma Z)$（VP）
- **距离函数**：$d_\mathcal{M}(x) = \frac{1}{2}\text{dist}^2(x, \mathcal{M})$
- **Riemannian体积测度**：$d\mathcal{M}(x)$ 和度量张量 $g(u)$
- **非可逆动力学的稳态分布**：当score不是梯度场时（如参数化模型），Langevin动力学的稳态分布无闭式解

### 均匀采样算法
修改Langevin动力学：$dX_t = [\nabla \log p_\sigma(X_t) + \frac{1}{\sigma^2}\nabla d_\mathcal{M}(X_t)] dt + \sqrt{2} dW_t$

其中 $\frac{1}{\sigma^2}\nabla d_\mathcal{M}$ 项精确抵消score中的几何（投影）分量，留下仅包含分布信息的项——但此时分布信息被忽略（或被噪声淹没），最终收敛到均匀分布。

## 实验关键数据

### 合成实验
- 1D圆嵌入2D空间：验证几何恢复和均匀采样的可行性
- 不同score误差水平：$o(\sigma^{-2})$ 足以恢复流形支持，但分布任意
- 一行修改后：生成流形上的均匀分布

### Stable Diffusion 1.5实验
- 在大规模预训练模型上验证理论预测
- 修改采样算法后生成更均匀覆盖流形的样本
- 多样性增加，偏差减少

### 关键发现
- 实验验证了 $\Theta(\sigma^{-2})$ 的率分离在实际模型中成立
- 均匀采样算法在不需要微调的情况下有效工作
- 标准扩散模型确实主要学习了流形几何而非精确分布

## 亮点与洞察
- **范式转换的主张**：从分布学习→几何学习，这不仅是理论发现，更是实践指导——不必追求完美的分布恢复
- **一行代码的力量**：均匀采样仅需修改一行（添加距离梯度项），实用性极强
- **对扩散模型成功的新解释**：是因为它们学会了"什么看起来像真实数据"（流形），而非"真实数据的频率分布"
- **贝叶斯逆问题的启示**：均匀先验（最大熵）比数据先验对误差更鲁棒，这对下游应用有重要指导

## 局限与展望
- 理论分析依赖流形假设（$C^4$紧致无边界流形），实际数据可能不严格满足
- $\nabla d_\mathcal{M}$ 在实际中需要估计，估计精度影响均匀采样质量
- 初步实验在Stable Diffusion 1.5上进行，更大模型和更多任务待验证
- 率分离的精确常数（前因子）未讨论，实际gap可能小于理论预测

## 相关工作与启发
- **vs Stanczuk et al. 2024**：他们估计内在维度，不涉及率分离
- **vs Ventura et al. 2024**：仅分析线性流形（子空间），本文适用于一般光滑流形
- **vs De Santi et al. 2025**：他们通过微调实现均匀采样，本文在推理时实现（无需微调）
- **vs 现有score learning理论**：它们将几何和分布误差混合，本文首次分离两者

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 率分离现象的发现和几何学习范式的主张具有突破性
- 实验充分度: ⭐⭐⭐⭐ 合成+Stable Diffusion验证，但缺少更多定量评估
- 写作质量: ⭐⭐⭐⭐⭐ 理论展开清晰，图示直观，主要贡献突出
- 价值: ⭐⭐⭐⭐⭐ 对理解扩散模型原理和改进采样有深远影响

<!-- RELATED:START -->

## 相关论文

- [Generative Model Inversion Through the Lens of the Manifold Hypothesis](../../NeurIPS2025/image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [The Spacetime of Diffusion Models: An Information Geometry Perspective](the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)

<!-- RELATED:END -->
