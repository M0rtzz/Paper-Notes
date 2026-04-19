---
title: >-
  [论文解读] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning
description: >-
  [AAAI 2026][图像生成][扩散模型] 提出 Self-NPO，一种无需外部数据标注或奖励模型的负偏好优化方法，通过截断扩散微调(TDFT)让扩散模型从自身生成的低质量数据中学习"什么是不好的"，配合 CFG 引导远离不良输出，仅需不到 Diffusion-NPO 1%的训练成本即可达到可比性能。
tags:
  - AAAI 2026
  - 图像生成
  - 扩散模型
  - 偏好优化
  - 负偏好优化
  - 自学习
  - 训练效率
---

# Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning

**会议**: AAAI 2026  
**arXiv**: [2505.11777](https://arxiv.org/abs/2505.11777)  
**代码**: [https://github.com/G-U-N/Diffusion-NPO](https://github.com/G-U-N/Diffusion-NPO)  
**领域**: 图像生成  
**关键词**: 扩散模型, 偏好优化, 负偏好优化, 自学习, 训练效率

## 一句话总结

提出 Self-NPO，一种无需外部数据标注或奖励模型的负偏好优化方法，通过截断扩散微调(TDFT)让扩散模型从自身生成的低质量数据中学习"什么是不好的"，配合 CFG 引导远离不良输出，仅需不到 Diffusion-NPO 1%的训练成本即可达到可比性能。

## 研究背景与动机

### 领域现状

扩散模型在图像、视频、3D生成中取得显著成功，但在海量未过滤数据上训练的模型常生成不符合人类偏好的结果。当前偏好优化(PO)方法主要有四类：
- **可微奖励(DR)**：反向传播优化奖励模型
- **强化学习(RL)**：将去噪过程建模为MDP，用PPO优化
- **直接偏好优化(DPO)**：用偏好对数据集直接微调
- **负偏好优化(NPO)**：训练模型生成与人类偏好相反的输出，通过CFG引导远离不良结果

### 核心矛盾

所有现有方法都严重依赖**显式偏好标注**——要么需要昂贵的人工标注偏好对(如Pick-a-Pic)，要么需要训练脆弱的奖励模型(如HPSv2、ImageReward)。在偏好数据稀缺或标注困难的领域（如医学图像、专业设计），这些方法受到严重限制。

### 本文切入角度

关键洞察：扩散模型本身就能自然产生低质量输出（模糊细节、结构错误、幻觉、模式坍塌），这些输出在任何奖励模型下本质上都会获得低分。因此，**从模型自身生成的数据中学习就等同于分布正则化的负偏好优化**。

### 核心 Idea

三个关键观察支撑了Self-NPO：

**受控弱化生成能力**等价于NPO，但不能任意弱化——需保持与正模型的一定相关性以避免CFG方差爆炸

**自生成数据学习**天然满足分布保持条件

**无需完整扩散生成**——提出截断扩散微调(TDFT)，利用Tweedie公式在中间时步获取 $\mathbf{x}_0$ 估计，避免巨大的生成成本

## 方法详解

### 整体框架

Self-NPO的流程：
1. 利用参考模型进行部分截断去噪（从 $\mathbf{x}_T$ 到 $\mathbf{x}_t$）
2. 通过 Tweedie 公式获取 $\mathbf{x}_{t\to0}^{ref}$（$\mathbf{x}_0$ 的期望估计）
3. 用 $\mathbf{x}_{t\to0}^{ref}$ 作为目标微调模型（标准扩散损失）
4. 微调后的模型作为CFG中的负偏好模型使用

推理时的CFG公式：
$$\boldsymbol{\epsilon}^\omega = (\omega+1)\boldsymbol{\epsilon}_{\boldsymbol{\theta}_{pos}}(\mathbf{x}_t, t, \boldsymbol{c}) - \omega\boldsymbol{\epsilon}_{\boldsymbol{\theta}_{neg}}(\mathbf{x}_t, t, \boldsymbol{c}')$$

### 关键设计

#### 1. **NPO via 自生成数据**

从RLHF框架出发，NPO的优化目标为：
$$\max_{\mathbb{P}_\theta} \mathbb{E}[R_{\text{NPO}}(\mathbf{x}_0, \boldsymbol{c})] - \beta D_{\text{KL}}[\mathbb{P}_\theta(\mathbf{x}_0|\boldsymbol{c}) \| \mathbb{P}_{\text{ref}}(\mathbf{x}_0|\boldsymbol{c})]$$

其中 $R_{\text{NPO}} = 1 - R$ 为反转奖励。利用自生成数据的合理性：
- **分布保持**：从自身数据学习保持原始分布特性
- **奖励降低**：自生成样本因幻觉、模式坍塌和优化不完美等原因本质上具有低奖励分

设计动机：自生成数据训练的模型会"变弱"，而弱化的模型在CFG中作为负项使用，能有效引导生成远离低质量模式。同时由于分布相关性，不会导致CFG输出方差爆炸（完全独立的高斯噪声会使方差变为 $2\omega^2 + 2\omega + 1$）。

#### 2. **截断扩散微调(TDFT)**

**基线方法**（全仿真）：从 $\mathbf{x}_T$ 完整去噪到 $\mathbf{x}_0^{ref}$，然后用标准扩散损失训练：
$$\min_{\boldsymbol{\theta}} \mathbb{E}_{t,\boldsymbol{\epsilon}} \|\mathbf{x}_0^{ref} - \boldsymbol{f}_{\boldsymbol{\theta}}((\mathbf{x}_0^{ref})_t, t, \boldsymbol{c})\|_2^2$$

**问题**：完整去噪过程计算成本极高。

**TDFT的核心**：利用 Tweedie 公式，在任意中间时步 $t$ 可直接获得 $\mathbf{x}_0$ 的条件期望：
$$\mathbf{x}_{t\to0}^{ref} = \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t^{ref}] = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log \mathbb{P}_{\text{ref}}(\mathbf{x}_t^{ref}|\boldsymbol{c})$$

截断去噪过程为：
$$\mathbf{x}_T^{ref} \to \mathbf{x}_{T-1}^{ref} \to \dots \mathbf{x}_t^{ref} \xrightarrow{\text{Tweedie}} \mathbf{x}_{t\to0}^{ref}$$

#### 3. **解决分布差异**

**问题**：$\mathbf{x}_{t\to0}^{ref}$ 是多个可能 $\mathbf{x}_0$ 的加权平均，其分布与 $\mathbf{x}_0^{ref} \sim \mathbb{P}_{\text{ref}}(\mathbf{x}_0|\boldsymbol{c})$ 不同，直接在 $\mathbf{x}_{t\to0}^{ref}$ 上加噪会引入分布错配。

**解决方案**：不从 $\mathbf{x}_{t\to0}^{ref}$ 直接加噪，而是从 $\mathbf{x}_t^{ref}$ 加噪到 $\mathbf{x}_s$：
$$\mathbf{x}_s = \frac{\alpha_s}{\alpha_t}\mathbf{x}_t^{ref} + \sqrt{\sigma_s^2 - \sigma_t^2\frac{\alpha_s^2}{\alpha_t^2}}\boldsymbol{\epsilon}$$

然后用 $\mathbf{x}_{t\to0}^{ref}$ 作为 $\mathbf{x}_0$ 预测目标：
$$\min_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x}_s \sim \mathbb{P}(\mathbf{x}_s|\mathbf{x}_t^{ref})} \|\mathbf{x}_{t\to0}^{ref} - \boldsymbol{f}_{\boldsymbol{\theta}}(\mathbf{x}_s, s, \boldsymbol{c})\|_2^2$$

设计动机：这保证了 $\mathbf{x}_s$ 的分布等价于参考模型 $\mathbb{P}_{\text{ref}}(\mathbf{x}_0^{ref}|\boldsymbol{c})$ 的噪声扰动分布。

### 损失函数 / 训练策略

核心为标准扩散训练损失（$\mathbf{x}_0$-prediction形式），无需额外奖励模型或偏好数据。

推理时的权重组合策略（继承自Diffusion-NPO）：
$$\boldsymbol{\theta}_{neg} = \boldsymbol{\theta} + \alpha\boldsymbol{\eta} + \beta\boldsymbol{\delta}$$

其中 $\boldsymbol{\eta}$ 是正偏好优化偏移，$\boldsymbol{\delta}$ 是负偏好优化偏移，$\alpha, \beta \in [0,1]$ 控制混合度，确保输出稳定且相关。

### 理论基础

作者从三方面证明TDFT的合理性：
1. **分布等价**：$\mathbf{x}_s$ 的分布等价于参考模型的噪声扰动分布（定理1）
2. **梯度等价**：优化目标Eq.17有梯度等价的学习目标（定理2）
3. **标准扩散等价**：梯度等价目标等价于标准扩散模型的学习目标（定理3）

## 实验关键数据

### 主实验

在SD1.5上的定量比较（Pick-a-pic test_unique split）：

| 方法 | PickScore↑ | HPSv2.1↑ | ImageReward↑ | Aesthetic↑ |
|------|-----------|----------|-------------|-----------|
| SD-1.5 | 20.75 | 26.84 | 0.1064 | 5.539 |
| SD-1.5 + NPO | 21.26 | 27.36 | 0.2028 | 5.667 |
| **SD-1.5 + Self-NPO** | 21.00 | 27.04 | **0.2816** | 5.609 |
| DreamShaper | 21.85 | 28.85 | 0.6819 | 6.143 |
| DreamShaper + NPO (α=1.0) | 22.30 | 30.13 | 0.7258 | 6.234 |
| **DreamShaper + Self-NPO (α=1.0)** | 22.20 | **30.40** | **0.8038** | 6.196 |
| SePPO | 21.51 | 28.45 | 0.5981 | 5.892 |
| **SePPO + Self-NPO** | **21.73** | **30.28** | **0.6744** | **6.014** |

在SDXL上的比较：

| 方法 | PickScore↑ | HPSv2.1↑ | ImageReward↑ | Aesthetic↑ |
|------|-----------|----------|-------------|-----------|
| SDXL | 22.06 | 28.04 | 0.6246 | 6.114 |
| SDXL + Self-NPO | 22.26 | 28.24 | 0.6697 | **6.226** |
| Diff.-DPO | 22.57 | 29.76 | 0.8624 | 6.099 |
| Diff.-DPO + Self-NPO | 22.67 | 29.83 | 0.8784 | **6.179** |
| Juggernaut + Self-NPO | **22.77** | **30.56** | **0.9921** | 6.031 |

### 消融实验

训练成本对比：

| 方法 | 训练方式 | GPU小时 | GPU类型 |
|------|---------|--------|--------|
| Diffusion-NPO | 全权重 | 384 | A100 |
| Diffusion-NPO | LoRA | 153.6 | A800 |
| Baseline（全仿真） | 全权重 | 10.4 | A800 |
| **Self-NPO（默认K=5）** | **全权重** | **2** | **A800** |

Self-NPO仅需Diffusion-NPO训练成本的**不到1%**（2 vs 384 GPU小时），比全仿真基线快5倍。

### 关键发现

1. **即插即用特性**：Self-NPO可无缝集成到SD1.5、SDXL、CogVideoX，以及已经过偏好优化的模型上，始终带来收益
2. **ImageReward提升最显著**：在多数设置下，Self-NPO对ImageReward指标的提升最为明显，说明对人类偏好对齐的改善最大
3. **用户研究**：在色彩与光影、高频细节、低频构图、文图对齐四个维度都有改善，尤其是高频细节
4. **可控生成兼容**：与T2I-Adapter等控制插件结合使用时同样有效
5. **无条件生成提升**：Self-NPO增强的模型甚至在无条件生成下也能产出较高质量图像

## 亮点与洞察

1. **极致的训练效率**是最大亮点——2 GPU小时 vs 384 GPU小时，近200倍的效率提升，使得NPO从昂贵的大规模训练转变为轻量级的自学习
2. **无需任何外部数据**的设计极具实用价值，特别适合数据稀缺或标注困难的领域
3. **Tweedie公式的创造性应用**巧妙解决了生成成本问题——从中间时步直接跳到 $\mathbf{x}_0$ 估计
4. **分布差异的解决方案**很优雅——不从估计的 $\mathbf{x}_0$ 加噪，而是从中间状态 $\mathbf{x}_t$ 加噪，保持了真实的噪声扰动分布
5. "弱化即NPO"的直觉虽简单，但理论证明完整，从分布、梯度、损失函数三层面证明了等价性

## 局限与展望

1. **性能上界**受限：不依赖显式偏好标注意味着Self-NPO的性能上界可能低于有高质量标注的NPO
2. **奖励降低的不可控性**：自生成数据的低质量程度是隐式的，缺乏对弱化程度的精确控制
3. **仅在视觉生成上验证**：未在文本生成等其他扩散模型应用场景测试
4. **CFG依赖性**：方法的有效性完全依赖CFG推理范式，对不使用CFG的生成管线不适用
5. **可能进一步结合显式偏好**：将Self-NPO与少量高质量偏好数据结合的混合策略值得探索

## 相关工作与启发

- **Diffusion-NPO** (Wang et al., 2025)：负偏好优化的原始工作，本文是其data-free扩展
- **Diffusion-DPO** (Wallace et al., 2024)：直接偏好优化在扩散模型的应用
- **SePPO** (Zhang et al., 2024)：自评分的偏好对优化
- **RLHF for Diffusion**：从RLHF视角理解Self-NPO——KL正则化保持与参考模型的接近度
- **Key Takeaway**：Self-NPO表明扩散模型的"自我缺陷"可以被反向利用——模型知道什么生成得不好，利用这一信息比寻找外部偏好信号更高效

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "自生成数据 = 负偏好数据"的洞察新颖，TDFT是有效的技术创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖SD1.5/SDXL/CogVideoX，与NPO/DPO等全面对比，含用户研究和效率分析
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，数学推导严谨，理论证明完整
- **价值**: ⭐⭐⭐⭐⭐ — 极高的实用价值，将偏好优化的门槛降低了两个数量级

<!-- RELATED:START -->

## 相关论文

- [FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting](freeinpaint_tuning-free_prompt_alignment_and_visual_rationality_enhancement_in_i.md)
- [Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)

<!-- RELATED:END -->
