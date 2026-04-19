---
title: >-
  [论文解读] Phased One-Step Adversarial Equilibrium for Video Diffusion Models
description: >-
  [AAAI 2026][视频扩散蒸馏] 提出 V-PAE（Video Phased Adversarial Equilibrium），通过**稳定性预热 + 统一对抗均衡**两阶段蒸馏框架，将大规模视频扩散模型（如 Wan2.1-I2V-14B）压缩至单步生成，实现 100 倍加速，在 VBench-I2V 上平均质量超越已有加速方法 5.8%。
tags:
  - AAAI 2026
  - 视频扩散蒸馏
  - 单步生成
  - 对抗训练
  - 图像到视频
  - 采样加速
---

# Phased One-Step Adversarial Equilibrium for Video Diffusion Models

**会议**: AAAI 2026  
**arXiv**: [2508.21019](https://arxiv.org/abs/2508.21019)  
**代码**: https://v-pae.github.io/  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散蒸馏, 单步生成, 对抗训练, 图像到视频, 采样加速

## 一句话总结

提出 V-PAE（Video Phased Adversarial Equilibrium），通过**稳定性预热 + 统一对抗均衡**两阶段蒸馏框架，将大规模视频扩散模型（如 Wan2.1-I2V-14B）压缩至单步生成，实现 100 倍加速，在 VBench-I2V 上平均质量超越已有加速方法 5.8%。

## 研究背景与动机

1. **领域现状**：视频扩散模型（如 Wan2.1-I2V-14B）在保真度上取得巨大进步，但生成 5 秒视频需 50 步迭代、约 15 分钟（8×H20 GPU），计算代价极高。
2. **现有痛点**：现有视频蒸馏方法多是从图像蒸馏技术直接迁移（如 LCM、DMD2、ADD），存在两个根本局限：(a) 无法将大规模（>10B）视频模型蒸馏到单步；(b) 缺乏条件任务的泛化能力，在 I2V 生成中导致语义退化和条件帧崩塌。
3. **核心矛盾**：单步对抗蒸馏要求从高斯噪声直接生成视频，但生成视频与真实视频的分布差距巨大，使鉴别器判别过于简单，梯度微弱导致训练不稳定。现有方法（如 DMD2、ADD）只能在中高信噪比区间训练，与低信噪比采样分布不匹配。
4. **本文目标**：如何在大规模视频模型上实现高质量单步蒸馏？如何在 I2V 任务中保持视频-图像主题一致性？
5. **切入角度**：将蒸馏过程分为两阶段——先用 VSD 缩小分布差距（预热），再在更接近的分布上进行对抗蒸馏（均衡），并复用生成器参数作为鉴别器骨干以节省显存。
6. **核心 idea**：分阶段优化策略确保对抗蒸馏在分布对齐后进行，避免了同时混合多种损失导致的次优优化方向问题。

## 方法详解

### 整体框架

V-PAE 包含两个顺序阶段：(a) **稳定性预热**（Stability Priming）——利用变分分数蒸馏（VSD）缩小生成视频与真实视频的分布距离；(b) **统一对抗均衡**（Unified Adversarial Equilibrium）——复用生成器参数作为鉴别器骨干，在高斯噪声空间实现共演化对抗均衡。对于条件 I2V 任务，额外引入语义鉴别器头和条件 SDS 损失来保持视频-图像主题一致性。

### 关键设计

1. **稳定性预热（Phase I）**

    - 功能：缩小单步生成视频 $\hat{x}_0^\theta$ 与真实视频 $x \sim p_{\text{data}}$ 之间的分布距离，为后续对抗蒸馏提供稳定起点。
    - 核心思路：定义三个模型——预热生成器 $\mu^\theta$、真实模型 $\mu_{\text{real}}$（固定）和伪模型 $\mu_{\text{fake}}^\phi$（用 LoRA 轻量追踪生成分布）。生成器在低信噪比区间 $t \in [0.9T, T]$ 直接生成 $\hat{x}_0^\theta$，然后通过加噪后输入真实/伪模型，计算分数梯度差异作为分布匹配损失。伪模型采用 LoRA 适配和零初始化以提高大规模模型的追踪稳定性。
    - 设计动机：如果不做预热直接对抗蒸馏，生成视频质量太差导致鉴别器提供信息量不足的梯度，训练极其不稳定。与 DMD2 同时优化 VSD+对抗损失不同，V-PAE 通过分阶段避免了次优优化方向。

2. **统一对抗均衡（Phase II）**

    - 功能：在预热基础上，通过自鉴别对抗训练进一步提升单步视频生成质量。
    - 核心思路：复用生成器 $\mu^\theta$ 作为鉴别器骨干（自鉴别器），配合轻量鉴别器头 $d^\psi$ 计算判别 logits。生成器从端点 $\epsilon$ 直接采样 $\hat{x}_0^\theta = f^\theta(\epsilon, T)$，加噪后送入自鉴别器抽取多层特征用于判别。采用 Hinge loss 进行对抗训练，并使用 EMA 权重 $\theta^-$ 保证均衡稳定性。此外引入时空差分 R1 正则化 $\mathcal{L}_{\text{STR1}}$ 防止梯度爆炸，其中空间扰动 $\sigma_s=0.01$、时间扰动 $\sigma_t=0.1$。
    - 设计动机：传统方式需要单独的鉴别器骨干（冻结导致参数不对称、全参训练导致 OOM）。复用生成器作为骨干在有限显存下实现高效共演化，是唯一能在 14B 模型上可行的方案。

3. **视频-图像主题一致性保持**

    - 功能：解决 I2V 蒸馏中的语义退化和条件帧崩塌问题。
    - 核心思路：(a) **语义鉴别器头**：通过可学习查询 $q$ 与条件图像嵌入、文本嵌入拼接后做自注意力，再与骨干多层特征交叉注意力，增强语义感知能力。(b) **条件 SDS 损失**：利用预训练模型 $\mu_{\text{real}}$ 的分布稳定性，通过 $\mathcal{L}_{\text{C-SDS}} = \mathbb{E}[\|\hat{x}_0^\theta - f_{\text{real}}(\text{sg}(\hat{x}_{t'}^\theta), t')\|^2]$ 减小条件帧与生成帧之间的差异，防止条件帧崩塌。
    - 设计动机：I2V 是视频生成的主要应用，但单步蒸馏严重破坏视频与输入图像的语义一致性。多模态语义鉴别器头弥补了纯对抗训练对语义感知的不足。

### 损失函数 / 训练策略

- **Phase I**（500 步）：学习率 $1 \times 10^{-6}$，用 VSD 分布匹配损失训练，伪模型用 LoRA 追踪。
- **Phase II**（1000 步）：学习率 $2 \times 10^{-6}$，生成器损失 $\mathcal{L}_G = \mathcal{L}_{\text{UAE-G}} + 10 \cdot \mathcal{L}_{\text{C-SDS}}$，鉴别器头损失 $\mathcal{L}_D = \mathcal{L}_{\text{UAE-D}} + \mathcal{L}_{\text{STR1}}$，EMA 衰减率 0.995。
- 数据：合成数据（Wan2.1-T2V-14B 生成）+ 开源数据（Koala-36M、Intern4K）。

## 实验关键数据

### 主实验

在 VBench-I2V 上对比各蒸馏方法（基线模型 Wan2.1-I2V-14B）：

| 方法 | 类型 | NFE | 语义对齐 (SA) | 时序连贯 (TC) | 帧质量 (FQ) | 总分 | 延迟(s) |
|------|------|-----|-------------|-------------|-----------|------|---------|
| Baseline | Euler | 100 | 92.90 | 80.82 | 70.44 | - | 890 |
| APT | AD | 1 | 84.87 | 75.21 | 64.69 | - | - |
| DMD2 | VSD | 1 | 83.15 | 71.67 | 62.47 | - | - |
| MD | VSD | 1 | 84.02 | 74.25 | 64.76 | - | - |
| **V-PAE** | AD | **1** | **91.54** | **79.56** | **68.66** | - | **9.37** |
| **V-PAE** | AD | 4 | **94.93** | **82.24** | **70.76** | - | - |

### 消融实验

| 配置 | SA | TC | FQ | 说明 |
|------|------|------|------|------|
| 无预热 + 对抗 | - | - | - | 对抗训练不稳定，大分布偏差 |
| 一致性蒸馏预热 (APT式) | 80.05 | 67.96 | 56.55 | Phase I 质量中等 |
| V-PAE Phase I (VSD) | 84.92 | 72.34 | 59.75 | VSD 预热更好 |
| 冻结骨干鉴别器 | 较低 | 较低 | 较低 | 参数不对称导致质量下降 |
| 全参鉴别器 | OOM | OOM | OOM | 14B 模型显存不足 |
| **自鉴别器（V-PAE）** | **最优** | **最优** | **最优** | 共演化均衡 |

### 关键发现

- 稳定性预热至关重要：去掉 Phase I 导致对抗训练严重不稳定。
- V-PAE 在 1-NFE 下接近 100-NFE 基线（差距仅 1.5%），4-NFE 下反超 3.3%。
- 自鉴别器是唯一在 14B 参数量级可行且效果最优的鉴别器方案。
- 条件 SDS 损失的注入强度 $\lambda=10$ 在 I2V 中对防止条件帧崩塌效果显著，$\lambda$ 过小会导致首帧与后续帧不一致。

## 亮点与洞察

- **分阶段蒸馏思想**：先缩小分布差距再做对抗是非常直觉且有效的策略，避免了 DMD2 那种同时优化多损失导致的次优方向问题，这个思路可以迁移到其他分布差距大的蒸馏任务。
- **自鉴别器设计**：复用生成器作为鉴别器骨干，既省显存又实现共演化，是在超大模型上做对抗蒸馏的巧妙工程方案。
- **从 15 分钟到 10 秒的 100x 加速**，对实时视频生成应用有重大意义。

## 局限与展望

- 仅在 I2V 任务上验证，T2V 的泛化能力未充分展示。
- 训练仍需大量高质量视频数据（合成+开源），数据准备成本不低。
- 单步生成与 100 步模型仍有 1.5% 的质量差距，极端场景（复杂运动、多物体交互）可能更明显。
- 未与最新的 Flow Matching 加速方法进行对比。

## 相关工作与启发

- **vs DMD2**：DMD2 同时优化 VSD + 对抗损失导致次优方向，V-PAE 通过分阶段避免了这个问题。
- **vs APT**：APT 只对小规模模型和短视频有效，V-PAE 首次实现 14B 模型的单步蒸馏。
- **vs ADD/SDXL-Turbo**：ADD 限制在 4 步均匀时间步，无法做单步；V-PAE 的自鉴别器突破了这个限制。

## 评分

- 新颖性: ⭐⭐⭐⭐ 分阶段蒸馏和自鉴别器思路有新意，但核心组件（VSD、对抗蒸馏）都是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ VBench-I2V 评测全面，消融充分，但缺少 T2V 和其他模型的验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机推导合理
- 价值: ⭐⭐⭐⭐⭐ 100x 加速对视频生成实时化有重大实用价值

<!-- RELATED:START -->

## 相关论文

- [Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)
- [FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)

<!-- RELATED:END -->
