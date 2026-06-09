---
title: >-
  [论文解读] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation
description: >-
  [ICLR 2026][图像生成][mean flow] 提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "mean flow"
  - "noise injection refinement"
  - "1-NFE"
  - "likelihood maximization"
  - "多模态"
---

# RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation

**会议**: ICLR 2026  
**arXiv**: [2602.00849](https://arxiv.org/abs/2602.00849)  
**代码**: 无  
**领域**: 扩散模型 / 单步生成 / Mean Flow 改进  
**关键词**: mean flow, noise injection refinement, 1-NFE, likelihood maximization, multimodal generation  

## 一句话总结
提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。

## 研究背景与动机

### 领域现状

**领域现状**：MeanFlow 通过学习平均速度场实现少步生成，无需预训练或蒸馏。但 1-NFE 时性能显著下降——单步传输不够精确，生成样本偏离目标分布。

**现有痛点**：1-NFE MeanFlow 在高斯混合分布上偏差大、在分子生成上生成无效结构（断裂分子）。多步（8/32 NFE）效果好但失去了效率优势。

**核心矛盾**：1-NFE 传输的确定性输出偏离真实分布（因为平均速度近似带误差），但不能增加 NFE。

**本文目标** 在保持 1-NFE 的前提下改善 MeanFlow 的生成质量。

**切入角度**：将 1-NFE 看作"粗传输"，然后加一步噪声注入来"精炼"——本质上是将 MeanFlow 的确定性输出转化为概率性输出，用噪声来补偿传输误差。训练时额外加入最大似然目标以最小化 KL 散度。

**核心 idea**：1-NFE MeanFlow 的确定性输出 + 高斯噪声注入 ≈ 更好的目标分布近似。

## 方法详解

### 整体框架

RMFlow 把 1-NFE MeanFlow 的单步传输看成一次"粗搬运"，再补一步噪声注入做精炼，生成过程写成 $x_{\text{gen}} = x_0 + \hat{u}_{0,1}(x_0; \theta) + \sigma \epsilon$，其中前两项是 MeanFlow 学到的平均速度场把噪声 $x_0$ 一步推到数据空间，$\sigma\epsilon$ 是补偿项。训练同步优化两件事：保持平均速度场本身准确，以及让加噪后的终端分布真正贴近目标分布。

### 关键设计

**1. 噪声注入精炼：把确定性点估计变成有方差的分布。** MeanFlow 在 1-NFE 下的输出 $x_0 + \hat{u}_{0,1}$ 是一个确定的点，而平均速度只是真实速度场沿路径的近似，单步传输必然带误差，于是这个点系统性地偏离目标——在高斯混合上表现为 TV 偏大，在分子生成上则直接搬出断裂的无效结构。RMFlow 的做法是在传输结果上再加一项 $\sigma\epsilon$（$\epsilon\sim\mathcal{N}(0,I)$），让原本的点估计摊开成一个以传输结果为中心、方差为 $\sigma^2$ 的小邻域分布。这一步几乎零计算开销，却把"模型把样本放偏了"的硬误差，软化成可以被概率覆盖的散布，从而让生成分布有机会盖住真实模式而不是卡在一个偏点上。

**2. 最大似然训练目标：用 NLL 把终端分布拉向目标。** 光加噪声还不够，得让加噪前的传输落点落在对的位置。RMFlow 额外引入负对数似然损失 $\mathcal{L}_{\text{NLL}} = \mathbb{E}\big[\|x_{\text{tgt}} - (x_0 + \hat{u}_{0,1})\|^2\big]$，在高斯噪声模型下，最小化这个平方误差等价于最大化目标样本在"传输落点 + $\sigma\epsilon$"这一分布下的期望对数似然。文中据此证明该损失是目标分布期望对数似然的下界，优化它就是在压低学习分布与目标分布之间的 KL 散度，给噪声注入提供了原理性的而非纯启发式的依据。

**3. 联合损失：两个目标各管一段。** 最终训练把 MeanFlow 损失与 NLL 损失加权合并，外加一项 guidance 正则化，总损失为 MeanFlow 损失加 $\lambda_1\mathcal{L}_{\text{NLL}}$ 加 $\lambda_2$ guidance 项。两者分工明确：MeanFlow 损失约束整条概率路径、对应 Wasserstein 距离意义下的传输质量，保证平均速度场把路径学对；NLL 损失只盯终端、对应 KL 散度意义下的分布贴合，保证最后落点加噪后接近目标。前者管"路走对"，后者管"终点准"，两端同时收紧才让 1-NFE 的输出既不跑偏又能覆盖真实分布。

### 损失函数 / 训练策略

大规模任务采用两阶段：先按常规训好 MeanFlow，再用 LoRA 微调阶段引入 NLL 损失做精炼，避免从头联合训练的开销。具体设置上，T2I 用 COCO 数据集、480M U-Net、在 SD-VAE 潜空间上生成；分子生成则在 QM9 与 Geom-Drugs 上验证。

## 实验关键数据

### 主实验

| 方法 | NFE | Mixture Gaussian TV ↓ | QM9 分子稳定性 ↑ |
|------|-----|---------------------|----------------|
| MeanFlow | 1 | 1.44 | 低（生成断裂） |
| MeanFlow | 8 | 0.80 | 中等 |
| MeanFlow | 32 | 0.67 | 高 |
| **RMFlow** | **1** | **0.76** | **接近 32-NFE** |

T2I (COCO FID-30K): RMFlow 达到与 Distilled SD、StyleGAN-T 可比的 FID，且不需要辅助模型。

### 关键发现
- 1-NFE RMFlow 超越 8-NFE MeanFlow（TV 0.76 vs 0.80），接近 32-NFE
- 分子生成中噪声注入有效避免了结构断裂
- 训练成本与 MeanFlow 相当（噪声注入几乎零开销）

## 亮点与洞察
- **极简改进**：只加一步 $\sigma\epsilon$ 就显著改善 1-NFE 质量——本质上是将点估计问题转为分布估计，用噪声补偿模型误差。
- **多模态通用**：同一框架处理图像、分子、时间序列，说明噪声注入精炼是模态无关的通用技巧。
- 理论上证明了 NLL 损失与 KL 散度的联系，为噪声注入提供了原理性解释。

## 局限与展望
- 噪声注入的 $\sigma$ 是超参数需要调优
- T2I 实验用 COCO + 小 U-Net，缺少 ImageNet/大模型验证
- 与 SoFlow、TwinFlow 等最新单步方法缺少直接对比
- 噪声注入是否在所有任务上都有益？高维图像中可能引入模糊

## 相关工作与启发
- **vs MeanFlow**: 1-NFE RMFlow > 8-NFE MeanFlow，核心改进是噪声注入+NLL 损失
- **vs SoFlow**: 不同的改进思路——SoFlow 学解函数，RMFlow 学平均速度+精炼

## 评分
- 新颖性: ⭐⭐⭐ 噪声注入思路简单，但在 MeanFlow 上是首次
- 实验充分度: ⭐⭐⭐⭐ 多模态验证（图像/分子/时间序列），但图像实验规模小
- 写作质量: ⭐⭐⭐⭐ 清晰有条理，理论推导严谨
- 价值: ⭐⭐⭐⭐ 提供了一个简单有效的 MeanFlow 改善方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICML 2026\] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](../../ICML2026/image_generation/watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)

</div>

<!-- RELATED:END -->
