---
title: >-
  [论文解读] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys
description: >-
  [CVPR 2026][图像生成][天文巡天] 提出 AS-Bridge，一个基于 Brownian Bridge 扩散过程的双向生成框架，在地基 LSST 与空基 Euclid 天文巡天之间建模概率条件分布，实现跨巡天图像翻译和罕见事件检测（引力透镜）…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "天文巡天"
  - "Brownian Bridge"
  - "跨模态翻译"
  - "异常检测"
  - "概率生成"
---

# AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys

**会议**: CVPR 2026  
**arXiv**: [2603.11928](https://arxiv.org/abs/2603.11928)  
**代码**: [有](https://github.com/ZHANG7DC/AS-Bridge)  
**领域**: 扩散模型/图像生成  
**关键词**: 天文巡天, Brownian Bridge, 跨模态翻译, 异常检测, 概率生成

## 一句话总结
提出 AS-Bridge，一个基于 Brownian Bridge 扩散过程的双向生成框架，在地基 LSST 与空基 Euclid 天文巡天之间建模概率条件分布，实现跨巡天图像翻译和罕见事件检测（引力透镜），并通过 $\epsilon$-prediction 训练目标改进了标准 Brownian Bridge 的似然估计。

## 研究背景与动机
未来十年观测宇宙学将由大型巡天驱动：地基 LSST（Vera C. Rubin 天文台）提供深度多波段光学图像但受大气湍流影响导致分辨率受限、源混合；空基 Euclid 提供高分辨率近红外成像但波段更少、光谱信息不完整。两个巡天有约 7,000-9,000 deg² 的重叠天区，观测同一天体但产生根本不同的数据。

跨巡天推断在两个方向上都是病态问题：从 LSST 恢复 Euclid 级别的形态需要解决大气模糊和背景噪声带来的歧义；从 Euclid 映射回 LSST 需要从更少的波段推断光谱信息。因此，跨巡天翻译应被视为概率过程，能够采样与已有观测一致的多个有效实现。

现有的跨模态方法（GAN-based、条件扩散）通常在单方向确定性范式下开发和评估，无法忠实表示观测模态之间的完整条件分布。科学应用需要带有不确定性量化的概率生成。

## 方法详解

### 整体框架
AS-Bridge 将跨巡天翻译建模为双向 Brownian Bridge 过程。利用重叠天区的配对观测作为锚点训练，学习 LSST 和 Euclid 数据分布之间的随机路径。训练完成后可在非重叠区域生成互补观测，同时用于罕见事件检测。

### 关键设计

**1. 把跨巡天翻译写成条件分布，而不是确定性映射**

LSST 和 Euclid 看的是同一片天空，但成像物理完全不同，所以两者并不是一对一对应。AS-Bridge 把它们都看成同一个潜在天体物理过程 $\Phi$ 的两次带噪投影：$x_{\text{Euclid}} = \mathcal{O}_{\text{Euclid}}(\Phi) + \epsilon_{\text{Euclid}}$，$x_{\text{LSST}} = \mathcal{O}_{\text{LSST}}(\Phi) + \epsilon_{\text{LSST}}$。$\Phi$ 本身观测不到，于是作者把它边缘化掉，直接去学两个方向的条件分布 $p(x_{\text{Euclid}} \mid x_{\text{LSST}})$ 和 $p(x_{\text{LSST}} \mid x_{\text{Euclid}})$。这一步把问题从"GAN/确定性 I2I 给一个最像的答案"改成"给出一族与已有观测都自洽的合理实现"——因为从被大气糊掉的 LSST 反推 Euclid 形态、或从 Euclid 单波段反推 LSST 多波段颜色，本来就是有多解的病态问题，只有概率建模才能把不确定性表达出来。

**2. 用 $\epsilon$-prediction 重写 Brownian Bridge 的训练目标，稳住桥端点的梯度**

标准 Brownian Bridge 的前向过程把两端的真实观测当锚点，中间插值加噪：

$$x_t \mid (x_0, x_T) \sim \mathcal{N}\big((1-m_t)x_0 + m_t x_T,\ \delta_t I\big),\qquad \delta_t = m_t(1-m_t)$$

原始损失直接回归漂移加去噪项，但方差 $\delta_t$ 在桥的两端 $t\to0$、$t\to T$ 处趋近于 0，按 $\delta_t$ 加权会让端点附近梯度消失。作者证明改成预测噪声的 $\epsilon$-prediction 目标 $\mathcal{L} = \|\epsilon_\theta - \epsilon\|_2^2$ 等价于在标准损失上乘了一个 $\sqrt{\delta_t}$ 的权重——这是个更温和的权重，既保留了似然启发的"高噪声时间步多关注"，又不会在端点把梯度压没。对应的重建公式为

$$\hat{x}_0 = \frac{x_t - m_t x_T - \sqrt{\delta_t}\,\epsilon_\theta(x_t, x_T, t)}{1-m_t}$$

消融里直接拿 $\delta_t$ 当权重反而把 CRPS 做差了（见下文表），印证了"$\sqrt{\delta_t}$ 的温和加权"才是这套训练目标好用的关键。

**3. 把重建不一致性当成异常信号，做无监督罕见事件检测**

模型只在普通星系上训练，没见过强引力透镜这类罕见结构，于是它对这些样本的重建会"失手"——这恰好可以反过来当检测信号。具体做法是把一对配对观测通过前向过程融合到中间变量 $x_t$，再反向重建回 Euclid 域，并采样 $N$ 个随机重建 $\{\hat{x}_0^{(i)}\}_{i=1}^N$。像素级异常分数取这 $N$ 次里的最小重建误差：

$$\mathcal{A}(p) = \min_{i \in \{1,\dots,N\}} \|\hat{x}_0^{(i)}(p) - x_0(p)\|_2^2$$

取最小而不是平均，是为了把"采样噪声偶尔造成的虚高误差"压下去，只保留模型怎么采样都重建不出来的真正异常。图像级分数再按通量归一化聚合，让亮源和暗源可比：

$$\mathcal{A}(x_0) = \frac{\sum_p \mathcal{A}(p)}{\sum_p x_0(p)}$$

这等于把生成模型的"认知边界"变成了发现新现象的探针——它建模不好的地方，往往就是训练分布里被低估的稀有天体。

### 损失函数 / 训练策略
- 训练数据：使用 SLSim 模拟生成 115,000 普通星系 + 5,000 强引力透镜系统
- LSST 图像：g/r/i 三波段，64×64 像素，~0.7" seeing
- Euclid 图像：VIS 波段，0.1" 像素尺度，64×64 像素
- 110,000 普通星系训练，其余用于评估

## 实验关键数据

### 主实验（概率重建质量 CRPS↓）

| 方法 | LSST→Euclid | Euclid→LSST |
|------|-------------|-------------|
| SPADE | 3.39 | 16.52 |
| OASIS | 4.65 | 13.33 |
| Pix2Pix | 4.35 | 73.03 |
| Palette | 2.43 | 7.98 |
| Joint Diffusion | 3.14 | 15.15 |
| BB 标准损失 | 2.55 | 7.90 |
| **AS-Bridge ($\epsilon$-pred)** | **2.38** | **7.90** |

### 消融实验

| 训练目标 | CRPS (LSST→Euclid) | CRPS (Euclid→LSST) | 说明 |
|---------|---------------------|---------------------|------|
| 标准损失 | 2.55 | 7.90 | 原始 BB 目标 |
| $\sqrt{\delta_t}$ 权重 | 3.59 | 11.24 | 直接加权反而差 |
| **$\epsilon$-pred** | **2.38** | **7.90** | 温和权重最优 |

### 异常检测（强引力透镜检测）

| 方法 | FPR@1%TPR↓ | FPR@5%TPR↓ | AUPR↑ |
|------|------------|------------|-------|
| **AS-Bridge** | **0.00%** | **0.18%** | **0.80** |
| Deco-Diff | 1.1% | 5.0% | 0.61 |
| CFM | 0.24% | 1.2% | 0.75 |

### 关键发现
- 扩散/Bridge 方法全面优于非扩散方法（GAN-based），验证了基于 score 的生成建模在恢复真实条件分布方面的优势
- Euclid→LSST（从单波段推断多波段颜色）是极度病态问题，但模型仍能生成形态一致且颜色合理的多样化重建
- LSST→Euclid 翻译能正确恢复被大气 seeing 混合的多源系统中的星系数量和位置
- 单模态方法 Deco-Diff 完全无法检测结构异常，跨模态信息对罕见事件检测至关重要

## 亮点与洞察
- 首次将跨巡天翻译形式化为概率推断问题，而非简单的 I2I 翻译
- $\epsilon$-prediction 等价性的形式化证明优雅且实用，为 Brownian Bridge 训练提供了理论指导
- 将重建不一致性用于无监督异常检测是巧妙的科学应用——利用生成模型的"认知边界"来发现新现象
- 评估指标的设计（CRPS 用于概率重建质量、FPR@low TPR 用于科学发现场景）体现了对领域需求的深入理解

## 局限与展望
- 目前仅在模拟数据上训练和评估，模拟到真实数据的域差距是已知限制
- Euclid→LSST 方向的 CRPS 仍然较高（7.90），多波段颜色推断的不确定性很大
- 仅用强引力透镜作为异常事件的代表，需要更多种类的罕见天体进行验证
- 图像尺寸固定为 64×64，对大尺度结构的建模可能不足

## 相关工作与启发
- 与 Palette（条件扩散 I2I）的核心区别：Palette 从纯噪声开始反向，源图像仅作为条件信号；BB 直接在两个分布间建模随机路径
- 跨模态异常检测思路可推广到其他多传感器天文数据（如 SKA 射电 + 光学）
- $\epsilon$-prediction 的 $\sqrt{\delta_t}$ 等价权重分析对所有使用 Brownian Bridge 的工作都有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将天文巡天间的概率翻译问题形式化，跨领域创新显著
- 实验充分度: ⭐⭐⭐⭐ 双向翻译 + 异常检测 + 消融完整，但仅在模拟数据上评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰、数学推导严谨、评估指标设计周到
- 价值: ⭐⭐⭐⭐ 为即将到来的 LSST-Euclid 联合分析提供了概念验证和基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multiagent_dialogue_framework_for_c.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_the_score_function_for_generalization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
