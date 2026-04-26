---
title: >-
  [论文解读] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][视频生成加速] 提出 DOLLAR，结合变分分数蒸馏（VSD）和一致性蒸馏（CD）实现少步视频生成，并引入潜在奖励模型微调策略进一步提升质量，4 步学生模型在 VBench 上达到 82.57 分超越教师模型和 Gen-3、Kling 等基线，单步蒸馏实现 278.6 倍加速。
tags:
  - ICCV 2025
  - 视频生成加速
  - 蒸馏
  - 一致性蒸馏
  - 变分分数蒸馏
  - 潜在奖励优化
---

# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**会议**: ICCV 2025  
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)  
**代码**: 无  
**领域**: 视频生成 / 扩散模型  
**关键词**: 视频生成加速, 蒸馏, 一致性蒸馏, 变分分数蒸馏, 潜在奖励优化

## 一句话总结

提出 DOLLAR，结合变分分数蒸馏（VSD）和一致性蒸馏（CD）实现少步视频生成，并引入潜在奖励模型微调策略进一步提升质量，4 步学生模型在 VBench 上达到 82.57 分超越教师模型和 Gen-3、Kling 等基线，单步蒸馏实现 278.6 倍加速。

## 研究背景与动机

**领域现状**：扩散概率模型在视频生成领域取得了显著进展，能够生成高质量的长视频（如 10 秒、128 帧、12 FPS）。但推理时需要大量采样步骤（通常 50 步 DDIM），每一步都涉及大规模 3D UNet 或 DiT 的前向传播，导致生成一段视频可能需要数分钟。

**现有痛点**：直接减少采样步数会严重损害视频质量和多样性。现有加速方法主要有两条路线：一致性蒸馏（Consistency Distillation）能实现少步采样但容易丢失细节和多样性；分数蒸馏（Score Distillation）能保持质量但需要更多步数。两者各有缺陷，尚无统一方案同时兼顾质量、多样性和速度。

**核心矛盾**：一致性蒸馏强制模型在单步/少步内直接预测最终输出，容易产生模式坍缩（mode collapse）和细节模糊；变分分数蒸馏保持了分布匹配但收敛到少步时效果不够稳定。视频相比图像维度更高、时间一致性要求更严格，使得这些问题更加突出。

**本文目标**：设计一种两阶段蒸馏方案，让学生模型在 1-4 步内生成高质量、高多样性的视频，并提供一种通用的微调机制来根据任意奖励指标进一步提升性能。

**切入角度**：作者观察到 VSD 和 CD 具有互补性——VSD 擅长保持分布多样性但单步效果差，CD 擅长少步生成但容易过拟合。如果先用 VSD 预热再用 CD 精化，可以同时获得两者的优点。此外，引入潜在空间的奖励模型可以绕过解码瓶颈，高效利用任何质量指标来微调。

**核心 idea**：两阶段蒸馏（VSD→CD）获得高质量少步基础模型，再用潜在奖励模型优化（Latent Reward Optimization）根据指定质量指标微调，实现质量-速度-多样性的最优平衡。

## 方法详解

### 整体框架

DOLLAR 的训练流程分为三个阶段：（1）变分分数蒸馏（VSD）阶段，让学生模型学习教师模型的分数函数，建立初始的少步生成能力；（2）一致性蒸馏（CD）阶段，在 VSD 基础上进一步训练，使模型在 1-4 步内输出高质量视频；（3）潜在奖励优化（LRO）阶段，在潜在空间训练轻量奖励模型，用 REINFORCE 算法微调学生模型以最大化任意奖励指标。最终推理时只需 1-4 步去噪即可生成 10 秒 128 帧视频。

### 关键设计

1. **两阶段蒸馏：VSD + CD（Variational Score Distillation + Consistency Distillation）**:

    - 功能：渐进式压缩教师模型的采样步数，同时保持质量和多样性
    - 核心思路：第一阶段（VSD）训练学生模型匹配教师的分数函数 $\nabla_{x_t} \log p_\text{teacher}(x_t)$，使学生在各噪声水平的预测与教师一致。VSD 的损失为 $\mathcal{L}_\text{VSD} = \mathbb{E}_{t, x_t}[\|\epsilon_\theta(x_t, t) - \epsilon_\text{teacher}(x_t, t)\|^2]$，但加入了一个辅助模型估计学生自身的分数（避免模式坍缩）。第二阶段（CD）在 VSD 预训练的基础上，强制学生在相邻时间步的输出保持一致：$\mathcal{L}_\text{CD} = \|f_\theta(x_t, t) - f_{\theta^-}(x_{t'}, t')\|$，其中 $\theta^-$ 是 EMA 参数
    - 设计动机：仅用 VSD 蒸馏到 1-4 步效果不够好，仅用 CD 从随机初始化开始容易模式坍缩。VSD→CD 的渐进策略让 CD 从一个好的起点出发，更容易收敛到高质量解

2. **潜在奖励模型优化（Latent Reward Optimization, LRO）**:

    - 功能：利用任意质量指标（如 VBench 分数、美学评分等）进一步微调蒸馏后的模型
    - 核心思路：传统的奖励优化需要将潜在变量解码为像素空间再计算奖励，显存消耗巨大（尤其是视频）。LRO 的核心创新是在潜在空间训练一个轻量级奖励代理模型 $R_\phi(z)$，用它来近似像素空间的奖励。训练时先收集 (潜在表示 $z$, 对应奖励值 $r$) 的数据对，然后训练 $R_\phi$ 做回归。微调学生模型时使用 REINFORCE 策略梯度：$\nabla_\theta \mathbb{E}[R_\phi(z)] \approx \mathbb{E}[\nabla_\theta \log p_\theta(z) \cdot R_\phi(z)]$
    - 设计动机：在潜在空间操作避免了解码开销（视频解码非常昂贵），且 REINFORCE 不要求奖励函数可微，使得任何黑盒质量指标都可以作为优化目标

3. **视频时间一致性保持（Temporal Consistency Preservation）**:

    - 功能：确保少步生成的视频在时间维度上保持流畅连贯
    - 核心思路：在 VSD 和 CD 阶段的损失中加入时间维度的约束。具体来说，在计算蒸馏损失时不仅比较单帧质量，还比较相邻帧之间的光流一致性和特征空间相似度。CD 阶段使用 3D 一致性约束，确保模型在时间步 $t$ 和 $t'$ 的预测在时间维度上也是一致的
    - 设计动机：视频蒸馏与图像蒸馏的关键区别在于时间一致性。如果不加约束，少步采样可能在每帧质量尚可的情况下产生闪烁（flickering）或时间不连贯

### 损失函数 / 训练策略

三阶段训练，各阶段损失：（1）VSD 阶段：$\mathcal{L} = \mathcal{L}_\text{VSD} + \lambda_\text{temp} \mathcal{L}_\text{temporal}$；（2）CD 阶段：$\mathcal{L} = \mathcal{L}_\text{CD} + \lambda_\text{temp} \mathcal{L}_\text{temporal}$；（3）LRO 阶段：REINFORCE 策略梯度 + KL 正则化防止偏离蒸馏结果。三个阶段依次训练，总训练量约为教师模型训练量的 10%。

## 实验关键数据

### 主实验

| 方法 | 步数 | VBench↑ | 质量分↑ | 多样性↑ | 加速比 |
|------|------|---------|---------|---------|--------|
| DOLLAR (4步) | 4 | **82.57** | **84.1** | **78.3** | 12.5x |
| DOLLAR (1步) | 1 | 80.12 | 81.5 | 76.1 | **278.6x** |
| Teacher (50步) | 50 | 81.23 | 83.2 | 77.5 | 1x |
| Gen-3 | — | 80.45 | 82.1 | 76.8 | — |
| T2V-Turbo | 4 | 78.92 | 80.3 | 74.5 | 12.5x |
| Kling | — | 79.88 | 81.0 | 76.2 | — |

### 消融实验

| 配置 | VBench↑ | 说明 |
|------|---------|------|
| Full DOLLAR (VSD+CD+LRO) | **82.57** | 完整三阶段 |
| VSD+CD (无LRO) | 81.45 | 无奖励优化，已超越教师 |
| 仅CD (无VSD预热) | 79.23 | 无VSD预热，模式坍缩，多样性差 |
| 仅VSD (无CD精化) | 80.18 | 4步效果尚可但低于CD精化后 |
| VSD+CD+像素空间奖励 | OOM | 视频解码后显存溢出 |
| LRO w/ 可微奖励 | 82.31 | 用可微奖励替代REINFORCE，差异不大 |

### 关键发现

- VSD→CD 两阶段蒸馏是关键设计，仅 CD 比完整方案低 3.34 分（模式坍缩），仅 VSD 低 2.39 分（少步不够精确）
- LRO 在 VSD+CD 基础上额外提升 1.12 分，验证了潜在奖励优化的有效性
- DOLLAR 的 4 步模型以 82.57 分超越了 50 步教师模型（81.23），再次验证了"蒸馏可以超越教师"的可能性
- 1 步模型实现 278.6 倍加速，接近实时生成 10 秒视频
- 人类评估进一步验证了 4 步学生模型优于 50 步教师模型

## 亮点与洞察

- **VSD+CD 互补蒸馏策略**：用 VSD 建立好的初始分布再用 CD 压缩步数，这种"先粗后细"的蒸馏范式非常优雅，且思路可以推广到其他模态（如音频、3D）的扩散模型加速
- **潜在奖励模型的巧妙设计**：在潜在空间而非像素空间计算奖励，完美解决了视频解码的显存瓶颈。且不要求奖励可微（用 REINFORCE），使得任何黑盒评估指标都可以作为优化目标——这个思路对所有涉及高维输出的生成模型微调都有价值
- **超越教师模型**：蒸馏后的学生模型不仅更快还更好，这得益于一致性约束和奖励优化带来的额外正则化。这一现象表明蒸馏过程中可以"注入"新的归纳偏置来改善教师的缺陷

## 局限与展望

- 三阶段训练流程较复杂，需要仔细调节各阶段的超参数和训练时长
- 潜在奖励模型的质量取决于训练数据的代表性，可能在分布外的提示词上失效
- 目前聚焦于 10 秒视频，扩展到更长视频（>1 分钟）的效果尚未验证
- VBench 指标虽然常用，但与人类感知的相关性仍有争议
- 未来方向：将 LRO 扩展为多奖励联合优化；探索 0 步生成（如 consistency model 风格）；将方法应用到 image-to-video 和 video editing 任务

## 相关工作与启发

- **vs T2V-Turbo**: T2V-Turbo 使用一致性蒸馏进行视频加速，但缺少 VSD 预热阶段，容易模式坍缩。DOLLAR 的 VSD+CD 两阶段策略在质量和多样性上都更优
- **vs AnimateLCM**: AnimateLCM 基于 LCM（Latent Consistency Model）做视频蒸馏，主要关注图像质量但忽略了时间一致性。DOLLAR 显式建模了时间维度的约束
- **vs Progressive Distillation（图像领域）**: 图像蒸馏的渐进方案（如 SDXL-Turbo）不需要考虑时间一致性。DOLLAR 将渐进蒸馏思想扩展到视频领域并解决了时间维度的新挑战
- **vs RLHF for LLMs**: LRO 的思路与 LLM 的 RLHF 异曲同工，都是用奖励模型微调生成模型，但 DOLLAR 在潜在空间操作解决了视频的显存问题

## 评分

- 新颖性: ⭐⭐⭐⭐ VSD+CD 两阶段组合和 LRO 都是有意义的创新，但各组件单独看并不全新
- 实验充分度: ⭐⭐⭐⭐⭐ VBench 全面评测、多基线对比、人类评估、详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，三阶段流程容易理解
- 价值: ⭐⭐⭐⭐⭐ 278.6x 加速 + 超越教师质量，对视频生成的实际部署有重大意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](long_context_tuning_for_video_generation.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)

<!-- RELATED:END -->
