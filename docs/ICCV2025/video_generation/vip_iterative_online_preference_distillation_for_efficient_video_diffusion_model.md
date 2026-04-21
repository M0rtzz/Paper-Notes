---
title: >-
  [论文解读] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models
description: >-
   > 提出 ReDPO 损失函数和 V.I.P. 迭代在线偏好蒸馏框架，将偏好学习 (DPO) 与 SFT 正则化相结合用于剪枝后视频扩散模型的蒸馏，在参数减少 36.2%-67.5% 的情况下匹配甚至超越完整模型性能。
tags:

---

# V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models

| 信息 | 内容 |
|------|------|
| 会议 | ICCV 2025 |
| arXiv | [2508.03254](https://arxiv.org/abs/2508.03254) |
| 代码 | [项目页面](https://jiiiisoo.github.io/VIP.github.io/) |
| 领域 | 视频生成 · 扩散模型蒸馏 · 偏好学习 |
| 关键词 | knowledge distillation, DPO, pruning, video diffusion, preference learning |

## 一句话总结

> 提出 ReDPO 损失函数和 V.I.P. 迭代在线偏好蒸馏框架，将偏好学习 (DPO) 与 SFT 正则化相结合用于剪枝后视频扩散模型的蒸馏，在参数减少 36.2%-67.5% 的情况下匹配甚至超越完整模型性能。

## 研究背景与动机

### 视频扩散模型的效率挑战

文本到视频 (T2V) 模型计算成本极高，在边缘设备部署困难。剪枝可以减少参数但会导致性能下降，需要知识蒸馏来恢复。

### 现有蒸馏方法的局限性

**SFT 的盲目模仿**：监督微调 (SFT) 最小化学生与教师预测的 L2 距离，但容量不足的学生模型无法完全复制教师输出，导致**分布平均化**——学生产生教师分布中不存在的"平滑"样本

**模式坍缩问题**：容量受限的学生在 SFT 下倾向于产生模糊输出和弱运动动态

**剪枝的选择性退化**：剪枝后某些属性退化而其他属性可能保持甚至提升，SFT 无法利用这一特点

### 核心洞察

作者观察到：**剪枝后的模型在某些维度上甚至优于完整模型**。因此，蒸馏不应盲目地全面模仿教师，而应**有选择地修复退化的属性**，这正是偏好学习 (DPO) 的优势所在。

## 方法详解

### 整体框架

V.I.P. = **V**ideo diffusion distillation via **I**terative **P**reference learning

工作流：$M_0$ (完整模型) → 剪枝 → $M_1$ → 评估+数据筛选+ReDPO训练 → $M_1'$ → 再次剪枝 → $M_2$ → 迭代...

### 1. 剪枝策略

逐步移除影响最小的模块（而非一次性大幅剪枝）：
- 单独移除每个 block，用 VideoScore 评估影响
- 选择对总分影响最小的 block 移除
- 识别因移除导致性能下降的属性作为恢复目标

### 2. 数据筛选（Data Curation）

**Prompt 筛选**：保留 5-25 词的高质量提示词，通过 LLM 过滤确保与目标属性相关。

**视频筛选**：用完整模型和剪枝模型分别生成视频，VideoScore 充当奖励模型，构建偏好对：

$$S(v_{\text{full}}) > S(v_{\text{pruned}}) > \tau_p$$

确保"赢"样本质量高，同时捕捉剪枝模型的实际弱点。

### 3. ReDPO：正则化扩散偏好优化

核心创新：将 DPO 与 SFT 结合，解决 DPO 的过度优化问题。

**扩散 DPO 损失**：

$$L_{\text{diff-dpo}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( -\beta T \omega(\lambda_t) \left( \|\epsilon^w - \epsilon_\theta(x_t^w, t)\|^2 - \|\epsilon^w - \epsilon_{\text{ref}}(x_t^w, t)\|^2 - (\|\epsilon^l - \epsilon_\theta(x_t^l, t)\|^2 - \|\epsilon^l - \epsilon_{\text{ref}}(x_t^l, t)\|^2) \right) \right) \right]$$

**SFT 正则化项**（防止过度优化）：

$$L_{\text{SFT}}(\theta) = \|\epsilon_\theta(x_t^w, t) - \epsilon_{\text{ref}}(x_t^w, t)\|^2$$

**最终 ReDPO 损失**：

$$L_{\text{ReDPO}}(\theta) = L_{\text{diff-dpo}}(\theta) + w_{\text{SFT}} \cdot L_{\text{SFT}}(\theta)$$

- DPO 引导学生**有选择地**修复退化属性
- SFT 约束偏好概率，防止 DPO 过度优化导致绝对质量下降

### 4. V.I.P. 迭代在线蒸馏

关键设计：
- **教师不变**：始终是初始完整模型 $M_0$
- **学生迭代更新**：每轮剪枝后重新评估弱点 → 重新筛选数据 → 重新训练
- **在线优势**：与离线 DPO 不同，每个阶段用最新的剪枝模型生成 losing 样本，确保训练数据与当前策略对齐

## 实验

### 主实验：V.I.P. 在两个基线上的表现（VideoScore 评估）

| 模型 | 阶段 | Visual Quality | Temporal Consist. | Dynamic Degree | Text Align. | Average | 参数 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| VideoCrafter2 Full | - | 2.627 | 2.602 | 2.728 | 2.491 | 2.613 | 1.413B |
| VC2 Stage 2 Pruned | - | 2.627 | 2.595 | 2.725 | 2.486 | 2.608 | 0.902B |
| **VC2 + ReDPO** | Stage 2 | **2.629** | **2.617** | 2.728 | **2.518** | **2.623 (+0.010)** | 0.902B |
| AnimateDiff Full | - | 2.575 | 2.505 | 2.684 | 2.486 | 2.563 | 0.453B |
| AD Stage 3 Pruned | - | 2.552 | 2.469 | 2.736 | 2.505 | 2.566 | 0.147B |
| **AD + ReDPO** | Stage 3 | **2.569** | **2.513** | 2.695 | **2.496** | **2.568 (+0.005)** | 0.147B |

关键发现：
- VideoCrafter2 减少 **36.2% 参数**（1.413B→0.902B），FLOPs 减少 21%，**所有指标匹配或超越完整模型**
- AnimateDiff 减少 **67.5% 参数**（0.453B→0.147B），仍保持与完整模型相当的性能

### ReDPO vs. SFT 对比

| 模型 | 方法 | Visual Quality | Temporal Consist. | Dynamic Degree | Text Align. |
|------|------|:---:|:---:|:---:|:---:|
| VC2 | SFT | 2.628 | 2.613 | 2.724 | 2.505 |
| VC2 | **ReDPO** | **2.629** | **2.617** | **2.728** | **2.518** |
| AD | SFT | 2.564 | 2.515 | 2.679 | 2.477 |
| AD | **ReDPO** | **2.569** | **2.513** | **2.695** | **2.496** |

ReDPO 在几乎所有指标上一致优于 SFT。SFT 的分布平均化导致模糊输出和弱文本对齐。

### 消融实验

| 消融项 | Visual Quality | Temporal Consist. | Dynamic Degree | Text Align. |
|--------|:---:|:---:|:---:|:---:|
| w/o SFT (仅 DPO) | 2.625 | 2.583 | **2.729** | 2.471 |
| w/o online (离线) | 2.626 | 2.603 | 2.719 | 2.483 |
| **V.I.P. (完整)** | **2.629** | **2.617** | 2.728 | **2.518** |

- 移除 SFT：时间一致性显著下降（DPO 过度优化）
- 移除在线迭代：性能全面下降（一次性剪枝损失太大）

### 用户研究

V.I.P. 在总体偏好上显著优于 SFT 和完整模型，表明 ReDPO 有效对齐了人类偏好。

## 亮点与洞察

1. **首次将偏好学习用于扩散模型蒸馏**：跳出 SFT 的框架，利用 DPO 的对比学习让学生模型有选择性地分配有限容量
2. **SFT 作为正则化器的妙用**：既不纯 SFT（会平均化）也不纯 DPO（会过度优化），两者互补
3. **在线迭代 vs. 一次性剪枝**：渐进式剪枝让模型在每步都有机会适应和恢复
4. **"剪枝后某些属性反而变好"的观察**是该方法的理论基础——不应浪费容量去模仿已经很好的属性

## 局限性

- 每个阶段都需要生成视频并用 VideoScore 评估，训练流程较为复杂
- 依赖单一奖励模型 (VideoScore) 的评分可能引入偏差
- 动态度分数有时会在提升一致性的同时下降，反映了质量-运动的固有权衡
- 实验仅在 AnimateDiff 和 VideoCrafter2 上验证，未涉及更大规模的视频生成模型

## 相关工作

- **视频扩散蒸馏**：BK-SDM 特征蒸馏、对抗损失蒸馏等
- **偏好对齐**：DPO、PPO、VideoDPO 等
- **模型剪枝**：通道剪枝、块剪枝等结构化剪枝方法
- **在线学习**：on-policy DPO、自我对弈等

## 评分

| 维度 | 分数 |
|------|:----:|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [Articulated Kinematics Distillation from Video Diffusion Models](../../CVPR2025/video_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)
- [EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)

<!-- RELATED:END -->
