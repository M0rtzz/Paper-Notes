---
title: >-
  [论文解读] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation
description: >-
  [图像生成] SANA-Sprint 提出混合蒸馏框架（连续时间一致性模型 + 潜空间对抗蒸馏），将预训练 Flow Matching 模型无损转换为 TrigFlow 并通过 sCM+LADD 联合训练，实现 1-4 步统一自适应高质量文本到图像生成，H100 上单步仅需 0.1 秒。
tags:
  - 图像生成
---

# SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.09641](https://arxiv.org/abs/2503.09641)
- **代码**: [GitHub](https://github.com/NVlabs/Sana) | [HuggingFace Models](https://huggingface.co/Efficient-Large-Model/SANA-Sprint)
- **机构**: NVIDIA, MIT, 清华大学, HuggingFace
- **领域**: 图像生成 / 扩散模型蒸馏
- **关键词**: 一步生成, 连续时间一致性模型 (sCM), 潜空间对抗蒸馏 (LADD), Flow Matching → TrigFlow 转换, 实时交互生成

## 一句话总结

SANA-Sprint 提出混合蒸馏框架（连续时间一致性模型 + 潜空间对抗蒸馏），将预训练 Flow Matching 模型无损转换为 TrigFlow 并通过 sCM+LADD 联合训练，实现 1-4 步统一自适应高质量文本到图像生成，H100 上单步仅需 0.1 秒。

## 研究背景与动机

扩散模型通常需要 50-100 步迭代去噪，计算代价高昂。现有步数蒸馏方法存在明显局限：

**GAN-based 方法**（如 ADD、LADD）：训练不稳定，存在模式崩溃和对抗动态振荡问题，超参调优困难

**VSD-based 方法**：需要额外联合训练扩散模型，GPU 显存压力大

**一致性模型（CM）**：在超少步数（<4 步）质量下降明显，特别是在文本-图像任务中截断误差导致语义对齐退化

这些挑战促使作者设计一个兼顾效率、灵活性和质量的统一蒸馏框架。SANA-Sprint 基于预训练 SANA 模型，结合近期连续时间一致性模型 (sCM) 的进展，消除离散时间一致性模型的离散化误差。

## 方法详解

### 整体框架

SANA-Sprint 采用三阶段设计：
1. **无训练转换**：将 Flow Matching 模型数学等价转换为 TrigFlow 模型
2. **稳定性优化**：通过 QK 归一化和密集时间嵌入解决训练不稳定性
3. **混合蒸馏**：sCM 损失提供教师对齐和多样性保持，LADD 对抗损失增强单步保真度

### 关键设计 1：Flow Matching → TrigFlow 无损转换

Flow Matching 和 TrigFlow 在三个方面存在不匹配：时间域（[0,1] vs [0,π/2]）、噪声调度（数据尺度不一致）、预测目标（静态系数 vs 时变系数）。

作者推导出无损数学转换：

$$t_{\text{FM}} = \frac{\sin(t_{\text{Trig}})}{\sin(t_{\text{Trig}}) + \cos(t_{\text{Trig}})}$$

$$x_{t,\text{FM}} = \frac{x_{t,\text{Trig}}}{\sigma_d} \cdot \sqrt{t_{\text{FM}}^2 + (1-t_{\text{FM}})^2}$$

TrigFlow 模型输出通过线性组合得到：

$$\hat{F_\theta} = \frac{1}{\sqrt{t_{\text{FM}}^2 + (1-t_{\text{FM}})^2}} \left[ (1-2t_{\text{FM}}) x_{t,\text{FM}} + (1-2t_{\text{FM}}+2t_{\text{FM}}^2) v_\theta \right]$$

实验验证转换前后 FID 几乎不变（5.81 vs 5.73），证明理论和实践上均无损。

### 关键设计 2：稳定连续时间蒸馏

- **密集时间嵌入**：将噪声系数 $c_{\text{noise}}(t)$ 从 $1000t$ 改为 $t$，避免时间导数 $\partial_t F_{\theta^{-}}$ 被放大 1000 倍导致梯度爆炸。PCA 可视化显示 0~1 范围的嵌入更加密集相似
- **QK 归一化**：在自注意力和交叉注意力的 Query/Key 上引入 RMS Normalization，解决模型从 0.6B 扩展到 1.6B 时梯度范数超过 $10^3$ 导致训练崩溃的问题。仅需 5000 次迭代微调教师模型

### 关键设计 3：sCM + LADD 混合损失

sCM 以局部方式蒸馏教师信息（学习相邻时间步的一致性），收敛较慢。引入 LADD 对抗损失提供跨时间步的全局监督：

$$\mathcal{L} = \mathcal{L}_{\text{sCM}} + \lambda \mathcal{L}_{\text{adv}}, \quad \lambda = 0.5$$

- LADD 使用冻结的教师模型作为特征提取器，在潜空间训练多个判别器头
- 判别器使用 hinge loss 区分真实噪声样本和生成噪声样本

### 额外最大时间加权

以概率 $p$ 将训练时间步设为 $t = \pi/2$（纯噪声），强化单步生成能力。实验发现 50% 概率最优。

### 损失函数

sCM 损失（连续时间一致性）：
$$\mathcal{L}_{\text{sCM}} = \mathbb{E}_{x_t, t} \left[ \frac{e^{w_\phi(t)}}{D} \left\| \hat{F_\theta} - \hat{F_{\theta^{-}}} - \cos(t) \frac{d\hat{f_{\theta^{-}}}}{dt} \right\|_2^2 - w_\phi(t) \right]$$

其中 $w_\phi(t)$ 是自适应加权函数，用于最小化不同时间步的方差。

## 实验关键数据

### 主实验：与 SOTA 方法的效率和性能对比

| 方法 | 步数 | 吞吐量 (samples/s) | 延迟 (s) | 参数量 (B) | FID ↓ | CLIP ↑ | GenEval ↑ |
|---|---|---|---|---|---|---|---|
| FLUX-schnell | 4 | 0.5 | 2.10 | 12.0 | 7.94 | 28.14 | 0.71 |
| SDXL-DMD2† | 4 | 2.27 | 0.54 | 0.9 | 6.82 | 28.84 | 0.60 |
| SD3.5-Turbo | 4 | 0.94 | 1.15 | 8.0 | 11.97 | 27.35 | 0.72 |
| **SANA-Sprint 0.6B** | **4** | **5.34** | **0.32** | **0.6** | **6.48** | **28.45** | **0.76** |
| **SANA-Sprint 1.6B** | **4** | **5.20** | **0.31** | **1.6** | **6.54** | **28.45** | **0.77** |
| FLUX-schnell | 1 | 1.58 | 0.68 | 12.0 | 7.26 | 28.49 | 0.69 |
| SDXL-DMD2† | 1 | 3.36 | 0.32 | 0.9 | 7.10 | 28.93 | 0.59 |
| **SANA-Sprint 0.6B** | **1** | **7.22** | **0.21** | **0.6** | **7.04** | **28.04** | **0.72** |
| **SANA-Sprint 1.6B** | **1** | **6.71** | **0.21** | **1.6** | **7.69** | **28.27** | **0.76** |

关键发现：
- 4 步 SANA-Sprint 0.6B 吞吐量是 FLUX-schnell 的 **10.7×**，FID 更优 (6.48 vs 7.94)
- 单步生成 SANA-Sprint 0.6B 延迟仅 0.21s，GenEval 达 0.72，超越同步数 FLUX-schnell

### 消融实验：损失组合与训练策略

| 实验项 | FID ↓ | CLIP ↑ |
|---|---|---|
| 仅 sCM | 8.93 | 27.51 |
| 仅 LADD | 12.20 | 27.00 |
| **sCM + LADD** | **8.11** | **28.02** |
| 无 CFG Embed | 9.23 | 27.15 |
| 有 CFG Embed | 8.72 | 28.09 |
| 0% maxT | 9.44 | 27.65 |
| 50% maxT | 8.32 | 27.94 |
| 70% maxT | 8.11 | 28.02 |
| sCM:LADD = 1.0:1.0 | 8.81 | 27.93 |
| sCM:LADD = 1.0:0.5 | 8.43 | 27.85 |
| sCM:LADD = 1.0:0.1 | 8.90 | 27.76 |

关键发现：
- sCM + LADD 互补效果显著：FID 从单独 LADD 的 12.20 降至 8.11
- CFG 嵌入提升 CLIP 得分 0.94
- 最大时间加权从 0% 到 50% 使 FID 从 9.44 降至 8.32

### 实时交互生成

- ControlNet 集成后实现 1024×1024 图像 250ms 生成延迟（H100）
- RTX 4090 消费级 GPU 上单步 0.31s

## 亮点与洞察

1. **无训练转换的优雅设计**：Flow Matching → TrigFlow 的数学等价变换避免了重新预训练教师模型的巨大开销，且通过可微变换兼容自动微分
2. **统一步自适应模型**：1-4 步共用同一模型，无需针对不同步数分别训练（许多竞品如 SDXL-DMD2 和 PCM 需要步数特定模型）
3. **混合蒸馏的互补性**：sCM 保证与教师分布对齐+多样性保持，LADD 提升单步保真度，二者正交互补
4. **工程发现**：密集时间嵌入（$c_{\text{noise}}=t$ 而非 $1000t$）和 QK-Norm 是训练大规模连续时间一致性模型的关键稳定化技巧

## 局限性

1. 验证主要基于 SANA 架构，虽然声称可推广到 FLUX/SD3 等 Flow Matching 模型，但未提供实验验证
2. 生成质量仍未完全达到 20 步教师模型水平（特别是复杂语义场景）
3. LADD 判别器增加了训练复杂度，虽然相比 VSD 更轻量但仍有优化空间
4. ControlNet 仅验证了 HED 边缘条件，其他条件类型（深度、分割等）效果未知

## 相关工作与启发

- **sCM** [Lu et al., 2024]：连续时间一致性模型的理论基础，SANA-Sprint 直接沿用其训练框架
- **LADD** [Sauer et al., 2024]：潜空间对抗蒸馏的关键参考，使用教师特征空间训练判别器
- **SANA** [Xie et al., 2024]：基础模型架构，线性注意力变压器实现高效图像生成
- **启发**：对于其他需要少步推理的生成任务（视频、3D），Flow→TrigFlow 转换 + 混合蒸馏的范式具有通用价值

## 评分 ⭐⭐⭐⭐⭐

工程价值极高的工作。Flow→TrigFlow 无损转换消除了 sCM 对专用预训练的依赖，降低了连续时间一致性蒸馏的门槛；混合蒸馏的创新组合在速度-质量帕累托前沿取得了显著突破。0.6B 参数量下 1 步 FID 7.04 / GenEval 0.72 的成绩令人印象深刻，对消费级 GPU 部署有直接推动作用。

<!-- RELATED:START -->

## 相关论文

- [Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)

<!-- RELATED:END -->
