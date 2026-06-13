---
title: >-
  [论文解读] Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation
description: >-
  [ECCV 2024][音频/语音][Dance Generation] 提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。 舞蹈编排的核心需求：真实舞蹈创作要求动作与音乐节拍精确同步，同时能够将特定关键姿态分…
tags:
  - "ECCV 2024"
  - "音频/语音"
  - "Dance Generation"
  - "Beat Synchronization"
  - "扩散模型"
  - "Keyframe Control"
  - "3D Motion"
---

# Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation

**会议**: ECCV 2024  
**arXiv**: [2407.07554](https://arxiv.org/abs/2407.07554)  
**代码**: [项目主页](https://zikaihuangscut.github.io/Beat-It/)  
**领域**: 音频语音  
**关键词**: Dance Generation, Beat Synchronization, Multi-Condition Diffusion, Keyframe Control, 3D Motion

## 一句话总结

提出 Beat-It 框架，通过将节拍条件从音乐中解耦并设计层次化多条件融合机制，实现了节拍同步且关键帧可控的 3D 舞蹈生成，在 AIST++ 上大幅领先现有方法。

## 研究背景与动机

**舞蹈编排的核心需求**：真实舞蹈创作要求动作与音乐节拍精确同步，同时能够将特定关键姿态分配到指定节拍上，但传统人工编舞耗时耗力

**现有方法节拍对齐差**：已有的音乐驱动舞蹈生成方法（如 EDGE、Bailando、FACT）主要学习音乐到舞蹈的直接映射，缺乏对节拍的显式建模，导致生成动作与音乐节拍不同步

**关键帧控制能力弱**：现有支持关键帧的方法（如 EDGE）通过简单的时间/空间混合来引入关键帧约束，但由于关键帧的稀疏性，密集的音乐特征会压制关键帧条件，导致可控性不足

**多条件融合困难**：关键帧条件极为稀疏而音乐/节拍条件密集，简单拼接会引入大量 padding 噪声，造成条件冲突，严重影响生成质量

**缺乏节拍可控性**：目前没有方法能够同时实现节拍指定和关键帧引导的舞蹈生成，无法满足实际编舞中"指定关键姿态出现在特定节拍"的需求

**节拍表示不充分**：直接使用二值掩码（0/1）表示节拍过于稀疏，难以被模型有效利用，容易被当作噪声忽略

## 方法详解

### 整体框架

Beat-It 基于扩散模型，接受三个输入条件：音乐条件 $\mathcal{C}$、稀疏关键帧条件 $\mathcal{X}^{ref}$、节拍条件 $\mathcal{B}$。三者分别经过对应编码器得到嵌入后，送入层次化多条件融合模块生成综合条件特征，最终由条件扩散去噪模块生成舞蹈动作序列。扩散模块采用与 EDGE 类似的 Transformer 架构，直接预测干净样本 $\mathbf{x}_0$。

### 关键设计 1：最近节拍距离表示（Nearest Beat Distance）

摒弃传统的二值节拍掩码，将节拍条件表示为向量 $b$，每个分量 $b^i$ 表示当前帧到最近节拍帧的距离。这一表示方式不仅缓解了稀疏性问题，还为模型提供了局部时间上下文信息，有助于更精准地捕获节奏特征。节拍嵌入通过独立的嵌入层和 Transformer 编码器 $\mathcal{E}_b$ 获得。

### 关键设计 2：层次化多条件融合机制（Hierarchical Multi-Condition Fusion）

分两阶段融合不同稀疏度的条件：

- **第一阶段（稀疏-密集融合）**：通过 Beat-Aware Dilated Cross-Attention 将稀疏的关键帧条件注入密集的音乐和节拍条件。核心技巧是 **节拍感知掩码膨胀**——距离节拍帧越近的关键帧获得越大的膨胀步长 $n = \lceil s \cdot e^{-2b^i/d^i} \rceil$，使得节拍附近的关键帧能够影响更多周围帧，实现节拍感知的条件传播
- **第二阶段（密集-密集融合）**：将经关键帧增强后的音乐嵌入和节拍嵌入拼接，送入 Transformer 融合模块生成最终的多条件特征

实现上使用 6 个稀疏-密集融合块（膨胀步长 $s$ = 4, 8, 12, 16, 20, 24）和 2 个密集-密集融合块。

### 关键设计 3：节拍对齐损失（Beat Alignment Loss）

预训练一个节拍距离估计器，在训练过程中对生成动作的节拍进行显式监督：

$$\mathcal{L}_{beat} = \sum_{i=1}^{L} w_s^i \cdot w_b^i \cdot \text{MSE}(b^i, \hat{b}^i)$$

其中 $w_s^i$ 是自适应权重，在节拍对齐差的区域加大惩罚；$w_b^i = e^{-2b^i/d^i}$ 使靠近节拍的帧获得更强监督。这确保了生成动作的运动节拍与给定节拍条件精确对齐。

## 损失函数与训练

总损失为 $\mathcal{L} = \mathcal{L}_{simple} + \lambda_{kin}\mathcal{L}_{kin} + \lambda_{beat}\mathcal{L}_{beat}$，其中：

- $\mathcal{L}_{simple}$：基础扩散重建损失
- $\mathcal{L}_{kin}$：运动学损失（关节位置 + 速度 + 脚部接触一致性 + 加速度），权重 $\lambda_{kin}=1$
- $\mathcal{L}_{beat}$：节拍对齐损失，权重 $\lambda_{beat}=0.5$

训练时随机采样 1%-30% 的 GT 帧作为关键帧条件，使用 GT 运动节拍作为节拍条件。Cosine 噪声调度，1000 步扩散，Adam 优化器（lr=2e-4），batch size=64，4 块 RTX3090。

## 实验

数据集为 AIST++（1408 段舞蹈，10 种街舞风格），裁剪为 5 秒片段（30fps），2.5 秒重叠。

### 表1：AIST++ 上与现有方法的定量比较

| 方法 | PFC ↓ | BAS ↑ | Div_k → | Div_m → | KPD ↓ | BAP ↑ |
|------|-------|-------|---------|---------|-------|-------|
| Ground Truth | 1.338 | 0.384 | 9.773 | 7.212 | - | - |
| FACT | 2.698 | 0.202 | 9.704 | 7.342 | - | - |
| Bailando | 1.578 | 0.215 | 9.622 | 7.175 | - | - |
| EDGE (keyframes) | 1.084 | 0.235 | 9.743 | 7.274 | 0.859 | - |
| **Beat-It (ours)** | **0.966** | **0.661** | **9.660** | **7.248** | **0.306** | **0.793** |

Beat-It 在 BAS 上比 EDGE 提升 0.426，在 KPD 上改善 0.553，节拍同步和关键帧可控性均大幅领先。PFC 也达到最优，运动学质量优于所有方法（甚至优于 GT）。

### 表2：消融实验

| 变体 | PFC ↓ | BAS ↑ | KPD ↓ | BAP ↑ |
|------|-------|-------|-------|-------|
| w/o HF（去掉层次融合） | 25.626 | 0.322 | 0.477 | 0.323 |
| w/o BD（去掉节拍感知膨胀） | 1.632 | 0.358 | 0.389 | 0.371 |
| w/o $\mathcal{L}_{beat}$ | 1.342 | 0.397 | 0.343 | 0.411 |
| **完整模型** | **0.966** | **0.661** | **0.306** | **0.793** |

三个核心组件均不可或缺。去掉层次融合导致 PFC 暴涨至 25.6，质量崩塌；去掉节拍对齐损失使 BAP 从 0.793 降至 0.411。

用户研究（18 人评估 20 段舞蹈）：Beat-It 相对 FACT 胜率 92.2%，相对 Bailando 胜率 78.8%，相对 EDGE 在质量/可控性上分别为 60.3%/86.9%。

## 亮点

- **首次实现节拍可控 + 关键帧引导的舞蹈生成**，填补了该方向的空白
- **最近节拍距离表示**设计简洁有效，将稀疏的二值节拍信号转化为密集且包含时间上下文的连续表示
- **层次化多条件融合**巧妙解决了不同稀疏度条件的冲突问题，节拍感知膨胀是亮点设计
- **节拍对齐损失**通过预训练估计器提供显式监督，BAS 提升显著（0.235→0.661）
- 支持任意节拍指定和灵活的关键帧分配，实际编舞应用价值高

## 局限性

- 仅在 AIST++ 数据集上验证，该数据集以街舞为主，泛化到其他舞蹈风格（如芭蕾、民族舞）未知
- 节拍距离估计器需要额外预训练，增加了流程复杂度
- 论文未讨论生成舞蹈的长期一致性和过渡自然度
- 未与 DanceFormer 等非开源方法进行直接比较
- 关键帧的自动选取策略未深入探讨，测试时随机采样 10% 可能不是最优策略

## 相关工作

- **单条件舞蹈生成**：FACT（Full-Attention Cross-Modal Transformer）、Bailando（VQ-VAE + Motion GPT）、DanceFormer（两阶段确定性框架）
- **多条件舞蹈生成**：EDGE（扩散模型 + 时间/空间混合关键帧控制）、LDA（风格标签辅助）、TM2D（音乐+文本双模态）
- **多条件扩散生成**：ControlNet、T2I-Adapter、Uni-ControlNet 等图像领域工作，Beat-It 将多条件融合思路引入舞蹈生成
- **关键帧控制**：Yang et al.（归一化流 + 时间嵌入）、DiffKFC（膨胀注意力，Beat-It 在其基础上加入节拍感知）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 节拍解耦 + 层次融合 + 节拍感知膨胀的组合设计新颖，首次解决节拍可控舞蹈生成问题
- 实验充分度: ⭐⭐⭐⭐ — 定量比较 + 消融 + 用户研究完整，但仅单数据集且缺少与部分方法的比较
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述详实，图示直观
- 价值: ⭐⭐⭐⭐ — 对舞蹈生成的可控性有实质推进，节拍距离表示和层次融合思路可推广到其他时序生成任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](../../CVPR2025/audio_speech/synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](../../CVPR2025/audio_speech/enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)

</div>

<!-- RELATED:END -->
