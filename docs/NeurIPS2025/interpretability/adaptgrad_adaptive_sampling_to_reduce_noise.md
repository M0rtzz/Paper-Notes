---
title: >-
  [论文解读] AdaptGrad: Adaptive Sampling to Reduce Noise
description: >-
  [NeurIPS 2025][可解释性][梯度平滑] AdaptGrad通过分析SmoothGrad噪声的理论来源——超范围采样行为，提出自适应调整每个输入维度的高斯采样方差以控制额外噪声上限的方法，在几乎消除梯度噪声的同时揭示更丰富的细节特征，方法极简且可与任意梯度解释方法组合。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "梯度平滑"
  - "显著性图"
  - "自适应采样"
  - "SmoothGrad"
---

# AdaptGrad: Adaptive Sampling to Reduce Noise

**会议**: NeurIPS 2025  
**arXiv**: [2410.07711](https://arxiv.org/abs/2410.07711)  
**代码**: [GitHub](https://github.com/AiShare-WHU/AdaptGrad)  
**领域**: 可解释性  
**关键词**: 梯度平滑, 显著性图, 可解释性, 自适应采样, SmoothGrad

## 一句话总结
AdaptGrad通过分析SmoothGrad噪声的理论来源——超范围采样行为，提出自适应调整每个输入维度的高斯采样方差以控制额外噪声上限的方法，在几乎消除梯度噪声的同时揭示更丰富的细节特征，方法极简且可与任意梯度解释方法组合。

## 研究背景与动机

**领域现状** 梯度是深度模型可解释性的关键信息来源，但原始梯度含大量噪声。SmoothGrad通过添加高斯噪声并平均梯度来降噪，是最广泛使用的梯度平滑方法。

**现有痛点** SmoothGrad的关键超参数 $\sigma$（高斯噪声方差）通常经验设定为 $\sigma = \alpha(x_{max} - x_{min})$，$\alpha=0.2$。这种设定导致大量采样点落在输入数据有效范围 $\Omega=[x_{min}, x_{max}]$ 之外，产生无意义样本并引入额外噪声。

**核心矛盾** SmoothGrad本是用于降噪的方法，但其超参数设定反而引入了新的噪声——采样分布域（$\mathbb{R}^D$）与数据域（$\Omega$）的不一致是根源。

**本文目标** 从理论上分析并最小化SmoothGrad的额外噪声，实现几乎无噪的梯度平滑。

**切入角度** 将SmoothGrad重新理解为蒙特卡洛近似的卷积操作，推导出额外噪声的解析表达式，据此设计自适应的采样方差。

**核心 idea** 根据每个输入像素到数据范围边界的距离自适应计算采样方差，确保超范围采样概率不超过预设上限 $1-c$。

## 方法详解

### 整体框架
AdaptGrad保持SmoothGrad的基本框架（采样邻域梯度并平均），但将各维度共享的固定方差 $\sigma^2$ 替换为各维度独立的自适应方差 $\sigma_i^2$，使得每个维度的超范围采样概率被控制在 $1-c$ 以内。

### 关键设计

1. **SmoothGrad的卷积理论分析**:
    - 功能：揭示SmoothGrad噪声的理论来源
    - 核心思路：SmoothGrad本质是梯度函数 $G(\mathbf{x})$ 与高斯核 $p(\cdot)$ 的卷积：$G_{sg}(\mathbf{x}) \simeq (G * p)(\mathbf{x})$。卷积域应为 $\Omega$，但采样分布定义在 $\mathbb{R}^D$ 上。超范围采样概率量化为 $A^i_{sg} = 1 - \frac{1}{2}\text{erf}(\frac{x_{max}-x_i}{\sqrt{2}\sigma}) + \frac{1}{2}\text{erf}(\frac{x_{min}-x_i}{\sqrt{2}\sigma})$
    - 设计动机：通过Spearman相关检验证实超范围比例(OBA)和超范围值(OBV)与噪声指标(Sparseness)呈显著负相关

2. **自适应方差计算**:
    - 功能：为每个输入维度计算最优采样方差
    - 核心思路：给定额外噪声水平 $c$（如0.95），对每个维度求解 $\sigma_i$：$\sigma_i = \frac{\min(|x_i - x_{min}|, |x_i - x_{max}|)}{\sqrt{2}\text{erfinv}(\frac{1+c}{2})}$。当 $x_i = x_{min}$ 或 $x_i = x_{max}$ 时 $\sigma_i = 0$（边界像素不扰动）。协方差矩阵为对角阵 $\Sigma_{ag} = \text{diag}(\sigma_1^2, ..., \sigma_D^2)$
    - 设计动机：像素越靠近边界，可用的安全采样空间越小，方差应越小。这是置信区间参数估计思想的自然应用

3. **与其他方法的组合**:
    - 功能：AdaptGrad可替代SmoothGrad与任意梯度方法组合
    - 核心思路：对GI（Gradient×Input）、IG（积分梯度）、NoiseGrad等方法，直接用AdaptGrad替代其中的SmoothGrad组件（用A-前缀表示），形成A-GI、A-IG、A-NG等变体
    - 设计动机：AdaptGrad与SmoothGrad接口完全兼容，是通用的梯度平滑改进

## 实验关键数据

### 定量评估——VGG16上的Sparseness指标

| 方法 | Grad | SG | **AG** | S-GI | **A-GI** | S-IG(W) | **A-IG(W)** |
|------|------|-----|--------|------|----------|---------|-------------|
| Sparseness ↑ | 0.558 | 0.529 | **0.574** | - | **更高** | - | **更高** |

### Consistency和Invariance检验（MNIST+MLP）

| 方法 | Consistency ↓ | Invariance ↓ |
|------|-------------|-------------|
| Grad | 0.02076 | 0.3483 |
| SmoothGrad | 0.01911 | 0.3613 |
| **AdaptGrad** | **0.02024** | **0.3484** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| c=0.95 | 最佳视觉质量 | 推荐默认值 |
| c=0.99 | 接近最优 | 略保守 |
| c=0.999 | 轻微退化 | 方差过小 |
| 直接clip采样 | 弱于AdaptGrad | 截断改变分布形状 |

### 关键发现
- AdaptGrad在低采样数（N=10）下已展现清晰降噪能力，SmoothGrad需N=50-70才收敛
- Invariance检验证实AdaptGrad维持了梯度在常数偏移下的稳定性（0.3484 vs SG的0.3613）
- 与GI和IG组合时改善尤为显著，揭示了更精细的面部等语义特征
- 在VGG16/ResNet50/InceptionV3三种架构上表现一致

## 亮点与洞察
- 理论分析简洁有力：从卷积视角揭示"降噪方法引入噪声"的悖论，推导出噪声的解析表达式
- 方法极简——仅改变采样方差的计算方式，不增加任何额外计算
- 对置信区间思想的巧妙借用使超参数有概率论基础的明确含义

## 局限与展望
- 理论分析假设数据域为简单的超矩形 $[x_{min}, x_{max}]^D$，实际数据流形可能更复杂
- Sparseness等指标并非完美的噪声度量，与人类感知的对齐度有限
- 对边界像素设 $\sigma=0$ 意味着完全不平滑，可能过于保守

## 相关工作与启发
- **vs SmoothGrad**: AdaptGrad解决了SmoothGrad的根本缺陷（超范围采样），理论和实验全面超越
- **vs NoiseGrad**: NoiseGrad扰动模型参数，AdaptGrad扰动输入；两者可叠加使用（A-NG）
- **vs Integrated Gradients**: IG解决梯度饱和，AdaptGrad解决噪声——两者互补（A-IG效果最佳）
- **vs FusionGrad**: FusionGrad是NoiseGrad和SmoothGrad的混合，AdaptGrad可替代其中的SmoothGrad部分

## 补充说明
- AdaptGrad的计算开销与SmoothGrad完全相同——只是方差计算方式不同，不增加采样次数
- 推荐设定 $c=0.95$，采样次数 $N=50$ 在大多数场景下足够

## 评分
- 新颖性: ⭐⭐⭐⭐ 对SmoothGrad噪声来源的理论分析新颖深刻
- 实验充分度: ⭐⭐⭐⭐ 4个指标、3种模型架构、5种方法组合，但数据集偏少
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，视觉对比直观
- 价值: ⭐⭐⭐ 解决了实际问题但应用范围局限于梯度可视化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[ICML 2025\] LANTERN: Modeling User Behavior from Adaptive Surveys with Supplemental Context](../../ICML2025/interpretability/modeling_user_behavior_from_adaptive_surveys_with_supplemental_context.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](../../ICML2026/interpretability/adaptive_querying_with_ai_persona_priors.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)

</div>

<!-- RELATED:END -->
