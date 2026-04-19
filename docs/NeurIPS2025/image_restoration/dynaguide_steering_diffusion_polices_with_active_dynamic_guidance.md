---
title: >-
  [论文解读] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance
description: >-
  [NeurIPS 2025][图像恢复][扩散模型] 提出 DynaGuide，在推理时通过外部潜在动力学模型对预训练扩散策略施加 classifier guidance，无需修改策略权重即可引导机器人朝向任意正/负目标，在 CALVIN 仿真上平均成功率 70%，真实机器人达 80%。
tags:
  - NeurIPS 2025
  - 图像恢复
  - 扩散模型
  - Classifier Guidance
  - Latent Dynamics Model
  - DinoV2
  - Robot Manipulation
---

# DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance

**会议**: NeurIPS 2025  
**arXiv**: [2506.13922](https://arxiv.org/abs/2506.13922)  
**代码**: [dynaguide.github.io](https://dynaguide.github.io)  
**领域**: 机器人学习 / 扩散策略引导  
**关键词**: Diffusion Policy, Classifier Guidance, Latent Dynamics Model, DinoV2, Robot Manipulation

## 一句话总结

提出 DynaGuide，在推理时通过外部潜在动力学模型对预训练扩散策略施加 classifier guidance，无需修改策略权重即可引导机器人朝向任意正/负目标，在 CALVIN 仿真上平均成功率 70%，真实机器人达 80%。

## 研究背景与动机

**领域现状**：扩散策略（Diffusion Policy）已成为机器人操控的主流范式，能学习复杂多模态行为。然而训练完成后，如何在部署时针对特定场景灵活调整行为（即"策略导向"）仍是开放问题。

**现有痛点**：
   - 目标条件策略（Goal-Conditioned Policy）需要在训练时预见所有可能的引导分布，推理时遇到分布外目标会严重退化；
   - 采样方法（如 GPC-Rank）从策略中多次采样并选最优，但依赖策略本身能生成满足目标的动作——对低概率行为无能为力；
   - 微调策略代价高且可能破坏已学技能。

**核心矛盾**：如何在**不修改**预训练扩散策略权重的前提下，灵活地将其引导至任意目标（包括多目标和负目标）？

**切入角度**：借鉴图像生成领域的 classifier guidance——训练一个外部动力学模型充当"分类器"，预测动作序列的未来视觉结果，并在去噪过程中用梯度信号直接修改动作。

**核心 idea**：外部动力学模型回答"执行此动作序列后未来会看到什么"，然后通过梯度将预测未来拉近期望目标、远离负面目标，整个过程只修改推理时的去噪方向，策略权重完全不变。

## 方法详解

### 整体框架

DynaGuide 由两个独立模块组成，在推理时协作：

- **基础扩散策略** $\pi_\theta(\mathbf{a}|o_t)$：预训练的 Diffusion Policy，通过 DDIM 去噪从高斯噪声生成动作序列，权重冻结不变；
- **引导模块**：包含潜在动力学模型 $h_\theta$ 和引导度量 $\mathbf{d}$，在每个去噪步计算梯度 $\nabla_{\mathbf{a}^k}\mathbf{d}$ 叠加到去噪信号上。

两者完全解耦，可随时更换引导模块而不影响基础策略。

### 关键设计

**1. 潜在动力学模型**

- **目标**：给定当前观测 $o_t$ 和动作序列 $\mathbf{a}$，预测 $H$ 步后的视觉状态 $\hat{z}_{t+H}$
- **编码器**：使用冻结的 DinoV2 提取 patch embedding 作为视觉潜表示 $z_t = \phi(o_t)$，语义丰富且训练稳定
- **预测器**：Transformer 架构，输入 $(z_t, \mathbf{a})$，输出 $\hat{z}_{t+H}$
- **训练目标**：简单的 MSE 回归 $\mathcal{L} = \|\phi(o_{t+H}) - h_\theta(\phi(o_t), \mathbf{a})\|_2^2$
- **数据增强**：向训练动作添加与推理时相同调度器的高斯噪声，使模型对去噪过程中的含噪动作鲁棒
- **训练数据**：仿真实验用 CALVIN 的 play data，真实实验用 UMI 开源数据+少量实验环境演示

**2. 引导度量设计（正/负多目标）**

引导条件 $\mathcal{G} = \mathbf{g}^+ \cup \mathbf{g}^-$，其中 $\mathbf{g}^+$ 是期望结果图像集合，$\mathbf{g}^-$ 是需回避的结果。将所有引导条件投影到同一 DinoV2 空间计算距离：

$$\mathbf{d} = \log\!\left[\sum_i \exp\frac{-\|\phi(g_i^+) - \hat{z}_{t+H}\|_2^2}{\sigma}\right] - \log\!\left[\sum_j \exp\frac{-\|\phi(g_j^-) - \hat{z}_{t+H}\|_2^2}{\sigma}\right]$$

- **Log-Sum-Exp 聚合**：作为 soft maximum，对多个引导条件取平滑最大值。当部分引导图像质量低（如机器人位置不对、场景不匹配）时，有用的信号不会被淹没。
- **正负分离**：第一项拉近目标，第二项推远负目标，天然支持多目标和回避行为。
- **超参数** $\sigma$：控制聚合的锐度，$\sigma$ 越小越聚焦于最近的引导条件。

**3. Classifier Guidance 注入去噪过程**

在 DDIM 的每个去噪步 $k$，将引导梯度叠加到噪声预测上：

$$\hat{\epsilon}(\mathbf{a}^k, o_t) = \epsilon(\mathbf{a}^k, o_t) - s\sqrt{1-\bar{\alpha}_k}\,\nabla_{\mathbf{a}^k}\mathbf{d}$$

- $s$ 是引导强度——越大越严格遵循目标，但过大会使轨迹不平滑
- **Stochastic Sampling 稳定化**：每个去噪步重复 $M$ 次 MCMC 采样，防止引导梯度将动作推出有效分布，允许使用更高的 $s$ 值
- 梯度通过动力学模型 $h_\theta$ 反传到动作空间，整个过程可微

### 训练策略

- 动力学模型：MSE 回归 + 动作噪声增强，用非结构化的机器人交互数据
- 引导过程：纯推理时计算，无需额外训练
- 基础策略：预训练冻结，DynaGuide 对任何 DDIM 扩散策略即插即用

## 实验关键数据

### 实验设置

在 CALVIN 仿真环境中测试 4 类场景，另有真实机器人实验。对比方法：

| 方法 | 说明 |
|------|------|
| Base Policy | 未引导的扩散策略 |
| Goal Conditioning (GC) | 以目标图像为条件训练的策略 |
| DynaGuide-Sampling (GPC) | 用同一动力学模型采样选最优动作 |
| Position Guidance (ITPS) | 用 3D 坐标引导扩散策略 |

### 主要结果

| 实验 | DynaGuide | GC | GPC | 说明 |
|------|-----------|-----|-----|------|
| ArticulatedParts（固定物体） | **70%** | ~95% | 较低 | GC 在分布内表现最佳，DynaGuide 比 base policy 提升 8.7× |
| MovableObjects（随机物体） | **显著优于 GPC** | 大幅下降 | ≈ base policy | 物体随机化导致 GC 分布外失效，GPC 采样方差过大 |
| UnderspecifiedObjectives（低质量引导） | 比 GC 高 **5.4×** | <10% | 中等 | 机器人状态随机化+场景不匹配时 DynaGuide 最鲁棒 |
| MultiObjectives（多目标+负目标） | 正目标 **80%** 成功 | 不适用 | 较低 | GPC 在回避负目标时失败率更高 |
| UnderrepresentedBehaviors（1% 数据） | **40%** | - | 更低 | 仅 1% 训练数据的行为仍可通过引导激活 |

### 真实机器人实验

使用公开预训练 UMI 策略（杯子放置任务），无需修改策略权重：

| 场景 | 成功率 | 说明 |
|------|--------|------|
| CupPreference（颜色偏好） | **72.5%** | 引导选择特定颜色杯子 |
| HiddenCup（遮挡物体） | **80%** | 引导找到被遮挡的红色杯子 |
| NovelBehavior（新行为） | 交互翻倍 | 引导触碰鼠标——训练数据中不存在的行为 |

### 关键发现

- **即插即用验证**：在 off-the-shelf 真实策略上直接使用，无需任何微调
- **对低质量引导极其鲁棒**：引导条件中机器人位姿、无关物体不匹配时，LSE 聚合仍能提取有效信号
- **主动引导 vs 被动采样**：DynaGuide 直接修改去噪方向，能激活策略中低概率模式；GPC 只能从策略已有的采样中选择
- **数据效率**：仅 1% 的目标行为训练数据，DynaGuide 仍达 40% 成功率

## 亮点与洞察

- **模块化分离是核心优势**：策略和引导完全解耦，同一策略可搭配不同引导模块完成不同任务，实际部署效率极高
- **DinoV2 作为通用状态空间**：冻结视觉 backbone 提供稳定的语义比较空间，避免了端到端训练中表示漂移的问题
- **Classifier Guidance 从图像生成迁移到机器人**：证明扩散模型的引导理论在动作空间同样有效，为机器人策略的推理时定制开辟新路径
- **负目标引导的实用价值**：真实部署中"不要做什么"和"要做什么"同样重要，DynaGuide 天然支持

## 局限与展望

- 需要额外训练动力学模型，增加了系统复杂度
- 引导条件目前仅支持视觉观测图像，不支持语言或运动学演示等更丰富的模态
- Stochastic Sampling（每步重复 $M$ 次）增加推理延迟
- 只能指定"期望/不期望的结果"，无法精细控制"如何"达到目标的过程
- 动力学模型的预测质量上限决定引导效果的天花板

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 classifier guidance 从图像生成迁移到机器人扩散策略，外部动力学模型充当分类器的设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 5 组仿真实验 + 3 组真实机器人实验，消融全面
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，图表信息量大
- 实用价值: ⭐⭐⭐⭐ 即插即用特性对实际机器人部署有直接意义

<!-- RELATED:START -->

## 相关论文

- [Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [Blind Noisy Image Deblurring Using Residual Guidance Strategy](../../ICCV2025/image_restoration/blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](../../ECCV2024/image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion](../../ICCV2025/image_restoration/decouple_to_reconstruct_high_quality_uhd_restoration_via_active_feature_disentan.md)

<!-- RELATED:END -->
