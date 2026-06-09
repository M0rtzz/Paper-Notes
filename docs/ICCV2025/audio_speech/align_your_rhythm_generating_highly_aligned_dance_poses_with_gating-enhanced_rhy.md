---
title: >-
  [论文解读] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation
description: >-
  [ICCV 2025][音频/语音][音乐驱动舞蹈生成] 提出Danceba框架，通过基于相位的节奏提取（PRE）、时序门控因果注意力（TGCA）和并行Mamba运动建模（PMMM）三个核心模块，实现音乐驱动的高节奏对齐、高多样性舞蹈生成，在AIST++数据集上FIDk提升48.68%、BAS提升12%。
tags:
  - "ICCV 2025"
  - "音频/语音"
  - "音乐驱动舞蹈生成"
  - "节奏对齐"
  - "门控注意力"
  - "Mamba"
  - "人体动作生成"
---

# Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation

**会议**: ICCV 2025  
**arXiv**: [2503.17340](https://arxiv.org/abs/2503.17340)  
**代码**: [https://danceba.github.io/](https://danceba.github.io/)  
**领域**: 人体理解 / 音乐驱动舞蹈生成  
**关键词**: 音乐驱动舞蹈生成, 节奏对齐, 门控注意力, Mamba, 人体动作生成

## 一句话总结
提出Danceba框架，通过基于相位的节奏提取（PRE）、时序门控因果注意力（TGCA）和并行Mamba运动建模（PMMM）三个核心模块，实现音乐驱动的高节奏对齐、高多样性舞蹈生成，在AIST++数据集上FIDk提升48.68%、BAS提升12%。

## 研究背景与动机

音乐驱动的舞蹈生成旨在根据音乐自动合成与之同步的3D舞蹈动作，在游戏、VR、影视制作等领域有广泛应用。现有方法面临的核心挑战是生成既能准确跟随音乐节奏、又具有自然性和多样性的舞蹈。

Bailando等代表性方法虽然引入了交叉条件因果注意力来对齐音乐和舞蹈姿态，但该注意力机制主要捕获局部对应关系，未能充分利用音乐中的全局节奏特征。后续方法如Enhancing-Bailando使用了MERT预训练音频编码器，但该编码器主要捕获语义信息而非显式的节奏特征。Beat-It虽然显式引入了节拍同步，但其刚性控制约束和强正则化显著限制了生成舞蹈的多样性和表现力。

本文的切入点是：现有方法要么忽略了音乐中显式的节奏结构，要么以牺牲多样性为代价来强制节拍对齐。核心idea是构建门控增强的节奏感知特征表示，在不牺牲动作自然性和多样性的前提下实现精确的节奏对齐。

## 方法详解

### 整体框架
Danceba沿用Bailando的VQ-VAE + 自回归框架：先用预训练的Pose VQ-VAE将上/下半身舞蹈编码为离散码本序列，再用自回归模型预测下一步的姿态码。核心改进在于将原始注意力替换为三个新模块的级联：PRE→TGCA→PMMM→TGCA。

### 关键设计

1. **基于相位的节奏提取（PRE）**:

    - 功能：从音乐输入中显式提取节奏特征，与语义特征解耦
    - 核心思路：对输入音乐特征 $\mathbf{m}$ 做短时傅里叶变换（STFT），提取相位角 $\boldsymbol{\varphi} = \text{Angle}(\text{STFT}(\mathbf{m}))$。相位角天然反映时间偏移和周期结构——低频相位变化对应宏观节奏模式，高频变化捕获细粒度节奏细节。再经中心裁剪对齐时间维度，通过 Linear + BN + ReLU 变换为嵌入空间特征 $\mathbf{X}_\varphi$，与原始输入特征相加融合
    - 设计动机：原始音乐特征中节奏信息稀疏（"rhythm-poor"），而STFT相位特征丰富（"rhythm-rich"）。使用STFT而非标准FT是因为音乐的非平稳性需要滑动窗口捕获突变节拍

2. **时序门控因果注意力（TGCA）**:

    - 功能：加强全局节奏特征对舞蹈生成的指导，解决交叉条件因果注意力中下一token缺乏历史token控制信号的问题
    - 核心思路：在交叉条件因果注意力 $C^3\text{Attention}$ 的基础上引入门控机制：
    $\mathbf{X}_{attn} = C^3\text{Attention}(\mathbf{X}) \odot \text{SiLU}(\text{Linear}(\mathbf{X}))$
      通过逐元素乘法将门控信号与注意力输出结合
    - 设计动机：可视化分析显示原始因果注意力中预测token缺乏来自历史token的明确控制信号，导致误差累积和不自然动作。门控机制让历史token更好地指导下一token预测，建立稳定的全局节拍注意力

3. **并行Mamba运动建模（PMMM）**:

    - 功能：分别建模上半身和下半身的运动序列，捕获各自独特的运动动力学
    - 核心思路：三个并行Mamba分支分别处理音乐、上半身、下半身的TGCA增强特征。每个分支包含 Mamba(RMSNorm(·)) + 残差连接 + GateMlp(RMSNorm(·)) + 残差连接。GateMlp通过门控机制选择性保留关键运动特征
    - 设计动机：上下半身运动模式和速度差异显著，需要分别处理以保持连贯性和多样性。Mamba在人体运动生成中已展现超越Transformer的序列建模能力，且更适合长序列生成

### 损失函数 / 训练策略
- 使用交叉熵损失对每个时间步的上下半身姿态码预测进行监督：
  $$\mathcal{L}_{CE} = \frac{1}{T'} \sum_{t=0}^{T'-1} \sum_{h=u,l} \text{CrossEntropy}(a_t^h, p_{t+1}^h)$$
- 推理时根据初始姿态码和完整音乐自回归预测，再用Pose VQ-VAE解码器还原舞蹈

## 实验关键数据

### 主实验：AIST++数据集

| 方法 | FIDk↓ | FIDg↓ | Divk↑ | Divg↑ | BAS↑ |
|------|-------|-------|-------|-------|------|
| Ground Truth | 17.10 | 10.60 | 8.19 | 7.45 | 0.2374 |
| Bailando | 28.16 | 9.62 | 7.83 | 6.34 | 0.2332 |
| DiffDance | 24.09 | 20.68 | 6.02 | 2.89 | 0.2418 |
| Bailando++ | 22.74 | 11.58 | 7.94 | 6.34 | 0.2263 |
| E3D2 | 26.25 | 8.94 | 7.96 | 6.49 | 0.2232 |
| Lodge | 37.09 | 18.79 | 5.58 | 4.85 | 0.2423 |
| **Danceba (Best)** | **11.67** | 11.90 | **8.52** | **7.55** | **0.2714** |

### 消融实验

| 配置 | FID↓ | Div↑ | BAS↑ |
|------|------|------|------|
| Ground Truth | 13.85 | 7.82 | 0.2374 |
| w/o TGCA | 33.22 | 6.74 | – |
| w/o PMMM | 25.24 | 5.78 | – |
| w/o PRE | 20.62 | 6.53 | 0.2474 |
| **Danceba** | **14.53** | **7.67** | **0.2779** |
| Danceba-Single | FIDk=82.50 | – | – |
| Danceba-Parallel | FIDk=15.48 | – | – |

### 关键发现
- FIDk下降48.68%（11.67 vs 22.74），BAS提升12%（0.2714 vs 0.2423），Divk和Divg均超过Ground Truth
- 去掉PRE模块后FID恶化41.9%、BAS下降11%，说明基于相位的节奏提取至关重要
- 并行Mamba vs 单一Mamba：FIDk从82.50降至15.48（改善67），证明分别建模上下半身的必要性
- TGCA的注意力热力图可视化表明门控机制有效建立了历史token到预测token的控制信号

## 亮点与洞察
- 利用STFT相位信息作为节奏特征的思路简洁有效，是一个领域通用的trick——相位天然编码时间偏移和周期信息
- 门控机制（SiLU gate）与因果注意力的结合是一种低成本的全局上下文增强策略，可迁移至其他自回归生成任务
- 并行Mamba的设计思路提示：对于具有明确结构划分的生成对象（如上下半身），分别建模后融合优于统一建模

## 局限与展望
- 音乐特征编码仅用简单线性层，未利用预训练音频模型（Jukebox、MERT、CLAP），可能遗漏细微音乐结构
- 沿用Bailando的Pose VQ-VAE，编码空间有限，可能限制精细运动细节的表达
- 仅在AIST++一个数据集上评估，未验证跨数据集泛化能力
- 未考虑风格条件输入（如舞蹈类型/风格控制），限制了灵活性
- 训练中随机采样带来方差，需5次独立训练取平均才有稳定结果

## 相关工作与启发
- **vs Bailando**: Danceba继承了Bailando的VQ-VAE + 自回归框架，但通过PRE-TGCA-PMMM三模块从特征提取、注意力机制和运动建模三个维度系统性改进
- **vs Beat-It**: Beat-It用显式节拍预测器强制同步，牺牲多样性；Danceba通过节奏感知特征隐式对齐，在不牺牲Div的前提下大幅提升BAS
- **vs EDGE**: EDGE使用条件扩散模型生成舞蹈，在FIDk和Div上均弱于Danceba，说明自回归+VQ-VAE框架配合节奏增强仍具竞争力
- **vs Motion Mamba系列**: Danceba验证了Mamba在舞蹈生成中的有效性，且并行结构比单一Mamba显著更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 基于STFT相位的节奏提取和门控因果注意力设计新颖
- 实验充分度: ⭐⭐⭐⭐ 消融全面（含5次独立训练均值），但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ 动机清晰，热力图可视化直观，模块间关系说明充分
- 价值: ⭐⭐⭐⭐ 在音乐驱动舞蹈生成方向贡献显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICML 2025\] OmniAudio: Generating Spatial Audio from 360-Degree Video](../../ICML2025/audio_speech/omniaudio_generating_spatial_audio_from_360-degree_video.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](../../CVPR2025/audio_speech/enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](../../ACL2026/audio_speech/immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)

</div>

<!-- RELATED:END -->
