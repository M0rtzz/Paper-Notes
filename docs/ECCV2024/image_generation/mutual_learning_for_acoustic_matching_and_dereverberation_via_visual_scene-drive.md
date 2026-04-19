---
title: >-
  [论文解读] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion
description: >-
  [ECCV 2024][图像生成] 提出 MVSD，一个基于扩散模型的互学习框架，将视觉声学匹配（VAM）和去混响作为对称互逆任务联合训练，利用两者的互惠关系克服配对数据稀缺问题，并首次将扩散模型用于视觉引导的混响风格迁移。
tags:
  - ECCV 2024
  - 图像生成
---

# Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion

**会议**: ECCV 2024  
**arXiv**: [2407.10373](https://arxiv.org/abs/2407.10373)  
**领域**: 图像生成

## 一句话总结

提出 MVSD，一个基于扩散模型的互学习框架，将视觉声学匹配（VAM）和去混响作为对称互逆任务联合训练，利用两者的互惠关系克服配对数据稀缺问题，并首次将扩散模型用于视觉引导的混响风格迁移。

## 研究背景与动机

- **视觉声学匹配（VAM）的重要性**：在增强现实/虚拟现实中，真实的混响音频对沉浸式体验至关重要，需要根据视觉场景将干净音频转换为相应混响版本
- **去混响的价值**：混响会降低语音可理解性和 ASR 系统精度，去混响对电话会议、助听器、语音助手等应用有重要价值
- **现有方法的局限**：
    - VAM 和去混响被独立研究，忽略了两者之间的固有互惠关系
    - 训练需要大量配对数据（干净音频+混响音频），但实际获取困难
    - 基于 GAN 的条件生成存在训练不稳定和过度平滑问题
- **核心洞察**：去混响是 VAM 的逆任务，两者可以互为评估器提供反馈信号，从而减少对配对数据的依赖

## 方法详解

### 整体框架

MVSD 包含两个基于视觉场景驱动扩散模型的转换器：
- **Reverberator** $f_\theta$：将干净音频 $\mathbf{a}_c$ 在视觉场景 $\mathbf{v}$ 条件下转为混响音频 $\hat{\mathbf{a}}_r$
- **Dereverberator** $g_\phi$：将混响音频 $\mathbf{a}_r$ 在视觉场景条件下还原为干净音频 $\hat{\mathbf{a}}_c$

两者形成闭环：一个的输出作为另一个的输入，互相提供反馈信号。

### 关键设计

**互学习机制**：
- 正向环路（VAM）：$\mathbf{a}_c \xrightarrow{f_\theta} \hat{\mathbf{a}}_r \xrightarrow{g_\phi} \tilde{\mathbf{a}}_c$，计算 $\tilde{\mathbf{a}}_c$ 与 $\mathbf{a}_c$ 的误差反馈给 $f_\theta$
- 反向环路（去混响）：$\mathbf{a}_r \xrightarrow{g_\phi} \hat{\mathbf{a}}_c \xrightarrow{f_\theta} \tilde{\mathbf{a}}_r$，计算 $\tilde{\mathbf{a}}_r$ 与 $\mathbf{a}_r$ 的误差反馈给 $g_\phi$
- **关键优势**：这种循环一致性误差不需要 $\mathbf{a}_c$ 和 $\mathbf{a}_r$ 对齐，因此可以利用非配对数据

**非配对数据利用**：
- 非配对自然混响音频 $\mathcal{U} = \{(\mathbf{v}', \mathbf{a}_r')\}$：先去混响再重建，计算循环误差
- 非配对干净音频 $\mathcal{C} = \{\mathbf{a}_c''\}$：随机采样视觉场景进行混响再去混响，计算循环误差

**视觉场景驱动扩散模型（VSD）**：
- **视觉场景编码器**：ResNet-18 提取 256 维视觉嵌入作为条件控制
- **可控 Unet**：编码器-解码器对称设计，各 3 个注意力块
    - 自注意力块使用 stride=4 的空洞卷积快速降低特征图尺寸
    - **选择性跨模态注意力**：仅在第 3 个编码器块和第 1 个解码器块使用跨模态注意力（视觉↔音频），减少计算开销
    - 源频谱图与噪声拼接作为内容输入，保留语言信息

**训练策略**：
- 先用配对数据进行监督训练建立基准
- 训练超过 100 个 epoch 后逐步引入非配对数据
- 在每个 mini-batch 中计算两个转换器的预测并基于对称模型的反馈更新参数

### 损失函数

总损失由三部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_d + \mathcal{L}_m + \mathcal{L}_{sty}$$

- $\mathcal{L}_d$：扩散模型噪声预测损失
- $\mathcal{L}_m$：互学习循环一致性损失（覆盖配对+非配对数据），使用 L1 距离
- $\mathcal{L}_{sty}$：风格损失，直接 L1 约束预测频谱图与目标频谱图的一致性

## 实验关键数据

### 主实验

**VAM 任务对比（SoundSpaces-Speech）**：

| 方法 | STFT↓ (Seen) | RTE(s)↓ (Seen) | MOSE↓ (Seen) | STFT↓ (Unseen) | RTE(s)↓ (Unseen) |
|------|-------------|----------------|-------------|----------------|-----------------|
| AV U-Net | 0.638 | 0.095 | 0.353 | 0.658 | 0.118 |
| AViTAR | 0.665 | 0.034 | 0.161 | 0.822 | 0.062 |
| MVSD w/o unpaired | 0.573 | 0.033 | 0.148 | 0.736 | 0.055 |
| **MVSD** | **0.508** | **0.030** | **0.142** | **0.637** | **0.051** |

**去混响任务对比（SoundSpaces-Speech）**：

| 方法 | PESQ↑ | WER(%)↓ | EER(%)↓ |
|------|-------|---------|---------|
| Reverberant（无处理） | 1.54 | 8.86 | 5.23 |
| MetricGAN+ | 2.33 | 7.49 | 5.16 |
| VIDA | 2.37 | 4.44 | 4.58 |
| **MVSD** | **2.53** | **4.27** | **4.46** |

### 消融实验

**用户研究（VAM 偏好率）**：

| 对比方法 | SoundSpaces (对方/MVSD) | AVSpeech (对方/MVSD) |
|----------|------------------------|---------------------|
| Input Speech | 39.3% / 60.7% | 38.2% / 61.8% |
| Image2Reverb | 20.8% / 79.2% | - |
| AV U-Net | 23.4% / 76.6% | 21.9% / 78.1% |
| AViTAR | 34.7% / 65.3% | 44.1% / 55.9% |

**非配对数据和组件消融**（从论文表 1 中提取）：

| 变体 | STFT↓ (Seen) | RTE(s)↓ | MOSE↓ |
|------|-------------|---------|-------|
| w/o visual scene | 0.691 | 0.188 | 0.156 |
| w/o unpaired data | 0.573 | 0.033 | 0.148 |
| **MVSD (完整)** | **0.508** | **0.030** | **0.142** |

### 关键发现

1. MVSD 在 SoundSpaces-Speech Seen 集上实现 **23.6% 的 STFT 相对提升**（0.665→0.508）和 **11.8% 的 RTE 相对提升**
2. **非配对数据（仅占训练数据 17.3%）可进一步提升 9.1% 的 RTE 性能**，证明互学习框架对非配对数据的有效利用
3. 去掉视觉场景信息后 RTE 从 0.030 升至 0.188（6.3 倍退化），证明视觉条件对混响风格控制的关键作用
4. MVSD 在用户研究中一致优于所有基线，包括 SOTA 方法 AViTAR（65.3% vs 34.7%）

## 亮点与洞察

- **互学习的对偶视角**：首次揭示 VAM 和去混响的互惠关系，两个任务互为评估器的设计优雅且高效
- **非配对数据利用**：通过循环一致性损失巧妙地利用易获取的非配对数据，实质上是音频领域的 CycleGAN 思想
- **扩散模型替代 GAN**：用扩散模型作为条件转换器，解决了 GAN 在声学匹配中的训练不稳定和过度平滑问题
- **选择性跨模态注意力**：仅在关键层使用跨模态注意力，在效果和效率间取得平衡

## 局限性

- 推理时间约 1.09 秒，尚无法满足实时应用需求
- 非配对数据的引入依赖训练先收敛到一定程度（>100 epoch），否则可能导致域偏移
- 频谱图级的 L1 风格损失可能不完全捕获感知上的混响特征

## 评分

⭐⭐⭐⭐ (4/5) — 互学习框架设计巧妙，非配对数据利用策略有效，在 VAM 和去混响两个任务上均取得显著提升，但实时性和感知评估方面仍有提升空间

<!-- RELATED:START -->

## 相关论文

- [Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_visual_instruction_tuning.md)
- [Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)

<!-- RELATED:END -->
