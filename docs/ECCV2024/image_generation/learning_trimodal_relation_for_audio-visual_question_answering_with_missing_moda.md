---
title: >-
  [论文解读] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality
description: >-
  [ECCV 2024][图像生成] 提出基于三模态关系的缺失模态 AVQA 框架，通过 RMM 生成器召回缺失模态特征并用 AVR 扩散模型跨模态增强，即使音频或视觉缺失也能准确回答问题。 现有音频-视觉问答（AVQA）方法依赖完整的视觉和音频输入，但在真实场景中设备故障、数据传输错误等常导致某一模态缺失…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality

**会议**: ECCV 2024  
**arXiv**: [2407.16171](https://arxiv.org/abs/2407.16171)  
**领域**: 图像生成

## 一句话总结

提出基于三模态关系的缺失模态 AVQA 框架，通过 RMM 生成器召回缺失模态特征并用 AVR 扩散模型跨模态增强，即使音频或视觉缺失也能准确回答问题。

## 研究背景与动机

现有音频-视觉问答（AVQA）方法依赖完整的视觉和音频输入，但在真实场景中设备故障、数据传输错误等常导致某一模态缺失。此时现有方法性能严重下降。已有缺失模态方法多处理一对一的模态对，忽视了不同模态之间的相互依赖关系，尤其无法灵活地根据问题上下文生成伪特征。

本文受人类认知心理学启发——人类可以通过音视觉整合来回忆缺失信息——提出了一种新的 AVQA 框架来应对缺失模态问题。

## 方法详解

### 整体框架

系统由三个核心组件构成：
1. **Relation-aware Missing Modal (RMM) 生成器**：利用可用的两种模态（如视觉+文本）来召回缺失模态（如音频）的伪特征
2. **Audio-Visual Relation-aware (AVR) 扩散模型**：将伪特征与真实特征拼接，通过扩散过程跨模态增强特征表示
3. **AVQA 骨干网络**：接收增强后的特征进行问答预测

### 关键设计

**RMM 生成器**采用基于 slot 的架构，每种模态由 L 个可学习参数向量表示。通过 addressing vector 机制，计算可用模态特征与 slot 之间的相关性，利用 element-wise 乘法融合视觉-文本 addressing vector，再对缺失模态的 slot 加权求和得到伪特征。三种模态的生成器共享权重。

**AVR 扩散模型**将音频特征和视觉特征拼接为联合特征，经过前向加噪和反向去噪过程，学习利用跨模态互补信息来增强特征表示。训练时使用真实特征对，推理时使用伪特征与真实特征的组合。

### 损失函数

总损失函数包含三部分：

$$\mathcal{L}_{Total} = \mathcal{L}_{avqa} + \lambda_1 \mathcal{L}_{rmmr} + \lambda_2 \mathcal{L}_{ave}$$

- $\mathcal{L}_{rmmr}$：关系感知缺失模态召回损失，L2 约束伪特征逼近真实特征
- $\mathcal{L}_{ave}$：音视觉增强损失，标准扩散去噪损失
- $\mathcal{L}_{avqa}$：三组交叉熵损失（完整输入、音频缺失、视觉缺失）

超参数设置 $\lambda_1 = \lambda_2 = 1$，RMM slot 数 $L=75$，扩散步数 $T=10$。

## 实验关键数据

### 主实验

在 MUSIC-AVQA 数据集上，各 AVQA 网络 + Ours 的结果（All Avg 准确率 %）：

| 方法 | 视觉缺失 (原始) | 视觉缺失 (+Ours) | 音频缺失 (原始) | 音频缺失 (+Ours) |
|------|:-:|:-:|:-:|:-:|
| AVSD | 59.25 | **68.91** | 41.08 | **69.90** |
| Pano-AVQA | 51.14 | **67.87** | 42.91 | **69.90** |
| AVST | 59.14 | **67.98** | 36.60 | **69.71** |
| PSTP-Net* | 59.27 | **66.39** | 67.74 | **71.55** |

音频缺失场景提升尤为显著，AVST 从 36.60% 提升至 69.71%（+33.11%）。

### 消融实验

与其他缺失模态处理方法在 MUSIC-AVQA 上的对比（基于 AVST 骨干，All Avg %）：

| 方法 | 视觉缺失 | 音频缺失 |
|------|:-:|:-:|
| ActionMAE | 64.12 | 65.38 |
| ShaSpec | 63.87 | 66.21 |
| Missing-aware Prompting | 65.44 | 67.13 |
| **Ours** | **67.98** | **69.71** |

### 关键发现

- 音频缺失场景下改进更大，说明视觉模态包含更多可用于跨模态召回的信息
- RMM 生成器通过三模态关系比单一模态对的方法更有效
- AVR 扩散模型能进一步增强伪特征和真实特征的表示质量
- 方法可灵活集成到多种 AVQA 骨干网络中

## 亮点与洞察

- 首次在 AVQA 任务中同时解决双向缺失模态（音频缺失和视觉缺失）
- slot-based addressing 机制优雅地利用三模态关系生成伪特征
- 扩散模型用于特征增强而非图像生成是有创意的应用
- 通用框架可插拔到现有 AVQA 网络，实用性强

## 局限性

- 仅处理单一模态缺失，未考虑多模态同时缺失的极端场景
- 扩散过程增加了推理时间开销
- 伪特征质量受 slot 数量 L 和训练数据分布影响

## 评分

⭐⭐⭐⭐ 创新性强，实验充分，缺失模态场景实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

</div>

<!-- RELATED:END -->
