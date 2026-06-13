---
title: >-
  [论文解读] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models
description: >-
  [CVPR 2025][视频生成][交互动力学] InterDyn 提出将视频扩散模型作为隐式物理引擎，通过在 Stable Video Diffusion 上引入交互控制分支（ControlNet-like），从单帧图像和驱动运动信号生成物理上合理的交互动力学视频，在 Something-Something-v2 数据集上 FVD 指标超过基线 CosHand 达 77%。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "交互动力学"
  - "视频扩散模型"
  - "可控生成"
  - "隐式物理模拟"
  - "人物交互"
---

# InterDyn: Controllable Interactive Dynamics with Video Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.11785](https://arxiv.org/abs/2412.11785)  
**代码**: [https://interdyn.is.tue.mpg.de/](https://interdyn.is.tue.mpg.de/)  
**领域**: 扩散模型  
**关键词**: 交互动力学, 视频扩散模型, 可控生成, 隐式物理模拟, 人物交互

## 一句话总结
InterDyn 提出将视频扩散模型作为隐式物理引擎，通过在 Stable Video Diffusion 上引入交互控制分支（ControlNet-like），从单帧图像和驱动运动信号生成物理上合理的交互动力学视频，在 Something-Something-v2 数据集上 FVD 指标超过基线 CosHand 达 77%。

## 研究背景与动机

1. **领域现状**：预测交互对象的动力学是智能系统的核心能力。现有方法可分为：(1) 基于显式物理模拟的方法，需要3D重建和物理引擎，计算昂贵且泛化性差；(2) 基于关键点/图神经网络的方法，仅在简化合成环境中验证。

2. **现有痛点**：
    - 显式物理模拟依赖准确的3D重建，误差累积，且难以应对复杂真实场景
    - 最近的生成式方法（如 CosHand）只能预测单个未来状态（图像到图像），无法捕捉交互后持续的连续动力学
    - 静态状态转移无法表达交互过程中的连续动态，如倒水后水位持续上升

3. **核心矛盾**：交互动力学是连续的时间过程，但现有方法要么需要完整的物理模拟管线，要么只做离散状态预测，两者都无法兼顾真实性和实用性。

4. **本文目标** 在不需要3D重建和物理引擎的前提下，从单帧图像和控制信号生成物理合理的交互动力学视频。

5. **切入角度**：大规模视频模型在海量视频数据上训练后，隐式地学习到了复杂的物理交互知识，只需要一种有效的控制机制来引导这些知识。

6. **核心 idea**：冻结预训练SVD的权重，仅训练一个ControlNet分支来注入驱动运动信号，将视频扩散模型当作隐式物理引擎使用。

## 方法详解

InterDyn 的方法核心很直观：将 Stable Video Diffusion (SVD) 视为一个已经"学会了物理"的模型，通过添加控制信号（如手部掩码序列）来精确引导它生成物体的交互动力学。关键洞察是——视频模型不仅是渲染器，还是隐式的物理模拟器。

### 整体框架

输入：一张初始图像 $\boldsymbol{x} \in \mathbb{R}^{1 \times H \times W \times 3}$ + 控制信号序列 $\boldsymbol{c} \in \mathbb{R}^{N \times H \times W \times 3}$（如手部二值掩码序列）。
输出：N帧视频 $\boldsymbol{y} \in \mathbb{R}^{N \times H \times W \times 3}$，展示手部运动及其引发的物体动力学。

架构基于 SVD（14帧 image-to-video），主干冻结，添加可训练的 ControlNet 编码器分支。

### 关键设计

1. **ControlNet式控制分支**:
    - 功能：将驱动运动控制信号注入视频生成过程
    - 核心思路：复制SVD编码器 $E$ 作为可训练副本，通过零初始化卷积层的skip connection连接到SVD冻结的解码器。用小型CNN $\mathcal{E}(\cdot)$ 将控制信号编码到latent空间，加到 ControlNet 编码器的输入噪声latent上。控制分支同样包含卷积、空间和时序块的交错结构，使其能以时序感知方式处理控制信号
    - 设计动机：冻结SVD权重保留其学到的动力学先验，避免灾难性遗忘；ControlNet架构允许精确控制同时保持生成质量；时序感知设计使模型对噪声控制信号（如SAM2输出的粗糙手部掩码）具有鲁棒性

2. **二值掩码驱动信号**:
    - 功能：以最简化的形式编码驱动实体（如手）的运动轨迹
    - 核心思路：使用SAM2从手部边界框生成逐帧二值掩码序列作为控制信号。该掩码仅编码"驱动者"的运动，不提供任何关于被操纵物体的信号——物体的动力学完全由模型隐式推理
    - 设计动机：二值掩码最易获取且与任务无关，实验表明控制信号类型对生成质量影响不大（见附录消融）

3. **训练与推理策略**:
    - 功能：高效微调并应用classifier-free guidance
    - 核心思路：使用EDM框架，噪声分布 $\log\sigma \sim \mathcal{N}(0.7, 1.6^2)$，Adam优化器 $lr=10^{-5}$，视频降采样至7FPS以平衡短程和长程事件。5%概率随机丢弃输入图像用于classifier-free guidance。推理时使用Euler调度器，50步去噪
    - 设计动机：7FPS的降采样让14帧视频覆盖约2秒时间，足以展示大多数交互动力学；motion ID设为40以匹配SVD先验

### 损失函数 / 训练策略

使用标准的扩散训练目标：去噪损失。在2张80GB H100上训练，每GPU batch size 4。两个版本：256×256（匹配CosHand）和256×384（匹配SVD先验宽高比）。

## 实验关键数据

### 主实验

**Something-Something-v2 (SSV2) 定量对比**

| 方法 | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | KVD↓ | Motion Fidelity↑ |
|------|-------|-------|--------|------|------|-------------------|
| Seer | 0.418 | 10.71 | 0.588 | 287.46 | 81.31 | — |
| DynamiCrafter | — | — | — | 204.11 | 31.81 | — |
| CosHand-Independent | 0.615 | 16.87 | 0.313 | 91.18 | 19.24 | 0.432 |
| CosHand-Autoregressive | 0.531 | 14.92 | 0.408 | 90.30 | 13.68 | 0.570 |
| **InterDyn 256×256** | **0.664** | **18.60** | **0.260** | **19.27** | **1.99** | **0.633** |
| **InterDyn 256×384** | **0.680** | **19.04** | **0.252** | **22.22** | **2.09** | **0.641** |

InterDyn在LPIPS上超越CosHand 37.5%，FVD上超越77%。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| CLEVRER力传播 | 可生成多物体碰撞链式反应 | 隐式理解力的传播 |
| CLEVRER反事实推理 | 相同图像+不同控制信号→不同合理结果 | 模型具有反事实推理能力 |
| 控制信号类型 | 二值掩码vs语义掩码差异不大 | 模型对控制信号类型不敏感 |
| 噪声掩码鲁棒性 | SAM2粗糙掩码仍生成精细手部细节 | 时序感知分支有效 |

### 关键发现
- InterDyn 能生成多种复杂物理现象：铰接物体运动、倒水（水位上升）、物体掉落弹跳、挤压变形/恢复、反射等
- 在CLEVRER合成数据上验证了模型具备力传播推理和反事实推理能力
- CosHand的帧间独立方法图像质量高但时序不一致；自回归方法运动更好但图像退化
- 模型甚至能生成仅有运动模糊的帧中仍然合理的手部细节

## 亮点与洞察
- **视频模型即物理引擎**：这是全文最重要的洞察。大规模视频数据训练后的模型隐式包含了物理交互知识，这个观点可以启发在更多物理推理任务中使用视频生成模型
- **控制与生成的优雅分离**：通过冻结SVD+可训练ControlNet，实现了"物理知识保留+精确控制注入"的双赢，这种设计模式可迁移到其他条件视频生成任务
- **反事实推理能力**：在CLEVRER上的实验显示模型能对同一场景做不同控制信号下的合理推理，暗示了视频模型作为世界模型的潜力

## 局限与展望
- 物体动力学的生成是隐式的、概率性的，无法保证物理准确性（例如精确的碰撞角度和速度）
- 目前仅在14帧上训练，长时间交互动力学的生成未探索
- 主要验证了人手-物体交互，全身交互和多人场景尚未涉及
- 图像质量指标（FID/KID）上偶尔不如CosHand，可能因SVD多阶段训练退化了空间先验
- 生成的手部细节虽然合理但不总是一致，手指运动有时不稳定

## 相关工作与启发
- **vs CosHand**: CosHand是图像到图像的状态转移方法，无法捕捉交互后的持续动力学；InterDyn直接生成连续视频，可以展示物体在力作用后的后续运动
- **vs PhysGen**: PhysGen依赖显式物理引擎计算运动并仅适用于刚体；InterDyn完全隐式，可处理软体、液体等复杂物理
- **vs Seer/DynamiCrafter**: 这些基于文本控制的方法缺乏精细的空间控制，FVD比InterDyn差一个数量级

## 评分
- 新颖性: ⭐⭐⭐⭐ "视频模型即物理引擎"的观点有深度启发性，方法设计清晰
- 实验充分度: ⭐⭐⭐⭐ 从合成到真实、从简单到复杂的渐进验证，定量定性都有
- 写作质量: ⭐⭐⭐⭐ 叙事清晰，从问题定义到实验设计逻辑通顺
- 价值: ⭐⭐⭐⭐ 开辟了视频模型作为隐式物理模拟器的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] VidTwin: Video VAE with Decoupled Structure and Dynamics](vidtwin_video_vae_with_decoupled_structure_and_dynamics.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)

</div>

<!-- RELATED:END -->
