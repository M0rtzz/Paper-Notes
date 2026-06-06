---
title: >-
  [论文解读] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency
description: >-
  [ICLR 2026][图像生成][连续时间一致性模型] 提出 rCM（score-regularized continuous-time consistency model），首次将连续时间一致性蒸馏扩展到 14B 参数的文生图/视频模型，通过结合前向散度（一致性）和反向散度（score蒸馏）…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "连续时间一致性模型"
  - "Score蒸馏"
  - "大规模蒸馏"
  - "JVP"
  - "少步生成"
---

# Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

**会议**: ICLR 2026  
**arXiv**: [2510.08431](https://arxiv.org/abs/2510.08431)  
**代码**: [项目页](https://research.nvidia.com/labs/dir/rcm)  
**领域**: 图像生成  
**关键词**: 连续时间一致性模型, Score蒸馏, 大规模蒸馏, JVP, 少步生成

## 一句话总结
提出 rCM（score-regularized continuous-time consistency model），首次将连续时间一致性蒸馏扩展到 14B 参数的文生图/视频模型，通过结合前向散度（一致性）和反向散度（score蒸馏），在保持多样性的同时匹配 DMD2 的质量，实现 15-50× 加速。

## 研究背景与动机
- sCM（连续时间一致性模型）理论优雅，但在大规模文生图/视频模型上的适用性不明——JVP 计算与 FlashAttention、并行训练不兼容
- sCM 在细节生成上存在质量问题（误差累积 + 前向散度的 mode-covering 特性导致质量扩散）
- Score/对抗蒸馏方法（如 DMD2）在质量上领先，但存在模态坍塌和多样性不足
- 前向散度（一致性模型）与反向散度（score蒸馏）具有互补性

## 方法详解

### 整体框架
rCM = sCM（前向散度一致性蒸馏）+ DMD（反向散度score蒸馏）+ 基础设施优化。训练交替优化学生模型（rCM loss）和 fake score 网络（flow matching loss）。

### 关键设计

1. **FlashAttention-2 JVP 内核**: 开发 Triton 内核，将 JVP 集成到 FlashAttention-2 前向传播中，支持自注意力和交叉注意力，兼容 FSDP 和 Context Parallelism，使 sCM 训练可扩展到 10B+ 参数模型。

2. **Score 正则化**: 将 DMD loss 作为长跳正则器补充 sCM。最终目标：
$$\mathcal{L}_{\text{rCM}}(\theta) = \mathcal{L}_{\text{sCM}}(\theta) + \lambda \mathcal{L}_{\text{DMD}}(\theta)$$
   $\lambda=0.01$ 跨模型和任务通用。sCM 提供 mode-covering（高多样性），DMD 提供 mode-seeking（高质量）。

3. **稳定时间导数计算**: 针对大模型 JVP 训练不稳定问题，提出两种方案：

    - 半连续时间：空间部分用 JVP，时间部分用有限差分近似（$\Delta t = 10^{-4}$）
    - 高精度时间：对时间嵌入层强制 FP32 精度

4. **Rollout 策略**: 学生可做任意步采样，随机选择步数 $N \in [1, N_{\max}]$，仅对最后一步反传 DMD loss，使用随机时间步确保覆盖整个时间范围。

### 损失函数 / 训练策略
- sCM loss（切线归一化）：$\mathcal{L}_{\text{sCM}} = \mathbb{E}\left[\left\|\mathbf{F}_\theta - \mathbf{F}_{\theta^-} - \frac{\mathbf{g}}{\|\mathbf{g}\|_2^2 + c}\right\|_2^2\right]$
- DMD loss：基于 fake score 和 teacher score 的差异引导学生
- Fake score 网络用 flow matching loss 在学生生成的数据上训练

## 实验关键数据

### 主实验（GenEval T2I）

| 模型 | 参数 | NFE | Overall | Counting | Position |
|------|------|-----|---------|----------|----------|
| FLUX.1-dev | 12B | 50 | 0.66 | 0.74 | 0.22 |
| Cosmos-Predict2 14B (teacher) | 14B | 70 | 0.84 | 0.79 | 0.64 |
| Cosmos-Predict2 + DMD2 | 2B | 4 | 0.80 | 0.70 | 0.57 |
| **Cosmos-Predict2 + rCM** | **2B** | **4** | **0.81** | **0.73** | **0.58** |
| **Cosmos-Predict2 + rCM** | **14B** | **4** | **0.83** | **0.80** | **0.59** |
| **Cosmos-Predict2 + rCM** | **14B** | **1** | **0.82** | **0.84** | **0.49** |

### VBench 视频实验

| 模型 | 参数 | NFE | Total Score | Throughput(FPS) |
|------|------|-----|-------------|-----------------|
| Wan2.1 14B (teacher) | 14B | 100 | 83.58 | 0.18 |
| Wan2.1 + DMD2 | 1.3B | 4 | 84.56 | 14.6 |
| **Wan2.1 + rCM** | **1.3B** | **4** | **84.43** | **14.6** |
| **Wan2.1 + rCM** | **14B** | **2** | **85.05** | **8.3** |

### 关键发现
- rCM 在质量上匹配或超过 DMD2，同时在多样性上明显优于 DMD2（Figure 1 显示 DMD2 生成物体位置/姿态趋同）
- 14B rCM 4步 GenEval 0.83，接近 teacher 70步的 0.84
- 视频任务中 rCM 2步即可达到接近 teacher 的 VBench 分数
- $\lambda=0.01$ 在质量和多样性之间取得最佳平衡
- 纯 sCM 在文字渲染等精细场景存在明显质量缺陷，rCM 成功修复

## 亮点与洞察
- 首次将 JVP-based 连续时间一致性扩展到 14B 参数和 5 秒视频
- 从前向/反向散度互补性的角度理解蒸馏方法的统一框架
- 无需 GAN 调优或大量超参搜索，$\lambda=0.01$ 跨任务通用
- rCM 的多样性优势对交互式 world model 等需要多样响应的场景尤为重要

## 局限与展望
- 需要额外的 fake score 网络（内存开销）
- JVP 计算仍比标准前向传播慢，训练成本高
- 1步视频生成质量仍有明显下降（VBench 从 85.05 降至 83.02）
- 对 autoregressive video diffusion 的扩展仅有展望

## 相关工作与启发
- sCM 和 MeanFlow 提供了理论基础
- DMD/DMD2 提供了反向散度蒸馏的实践方案
- DDO 和 DDRL 的前向+反向散度联合思想是 rCM 的哲学基础
- 为大规模视觉生成模型的部署提供了实用加速方案

## 技术细节补充
- TrigFlow 噪声调度：$\alpha_t = \cos(t), \sigma_t = \sin(t)$，与 rectified flow 通过 SNR 匹配互转
- Fake score 网络用 flow matching loss 在学生生成数据上训练，交替优化
- Selective Activation Checkpointing (SAC) 用于减少内存消耗
- Teacher 使用 CFG，CFG 同时蒸馏到学生中
- 全参数微调（不用 LoRA），强调 rCM 的稳定性
- 实验涵盖 Cosmos-Predict2（0.6B/2B/14B T2I）和 Wan2.1（1.3B/14B T2V）
- Wan2.1 14B 2步加速达 8.3 FPS vs teacher 的 0.18 FPS（约 46× 加速）

## 评分
- 新颖性: ⭐⭐⭐⭐ 前向+反向散度结合的理论洞察有价值，但各组件已知
- 实验充分度: ⭐⭐⭐⭐⭐ 验证规模前所未有（14B参数、T2I+T2V、多步消融）
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析清晰，工程细节详尽
- 价值: ⭐⭐⭐⭐⭐ 解决了大规模扩散模型加速的核心问题，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[ICLR 2026\] Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)
- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
