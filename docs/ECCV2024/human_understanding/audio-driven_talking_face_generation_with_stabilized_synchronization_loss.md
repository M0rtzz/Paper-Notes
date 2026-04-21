---
title: >-
  [论文解读] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss
description: >-
  [ECCV2024][人体理解][talking face generation] 提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。
tags:
  - ECCV2024
  - 人体理解
  - talking face generation
  - lip synchronization
  - SyncNet
  - lip leaking
  - GAN
---

# Audio-Driven Talking Face Generation with Stabilized Synchronization Loss

**会议**: ECCV2024  
**arXiv**: [2307.09368](https://arxiv.org/abs/2307.09368)  
**代码**: [yamand16/TalkingFaceGeneration](https://yamand16.github.io/TalkingFaceGeneration/)  
**领域**: human_understanding  
**关键词**: talking face generation, lip synchronization, SyncNet, lip leaking, GAN

## 一句话总结

提出 AVSyncNet、stabilized synchronization loss 和 silent-lip generator 三项改进，系统性地解决音频驱动说话人脸生成中 SyncNet 不稳定和嘴唇泄漏两大核心问题，在唇形同步和视觉质量上均达到 SOTA。

## 背景与动机

音频驱动的说话人脸生成（audio-driven talking face generation）旨在根据给定的音频和参考视频生成具有准确唇形同步和高视觉质量的真实视频，同时保持身份和视觉特征。该任务在电影配音、在线教育、视频会议增强等领域有广泛应用。

当前方法面临两个根本性挑战：

1. **SyncNet 的不稳定性**：Wav2Lip 引入的 SyncNet 被广泛用于计算 lip-sync loss，但它在真实数据上的余弦相似度表现极不稳定——即使是 GT（ground truth）的音频-唇形对也常得到低分。这种不稳定的训练信号会导致训练不稳定、唇形同步差、视觉质量退化，尤其在高分辨率场景下问题更严重。
2. **嘴唇泄漏（lip leaking）**：当前标准做法将人脸下半部遮罩作为 pose reference，并随机选取同一视频的另一帧作为 identity reference 来保持身份信息。然而随机选取的 identity reference 嘴唇形态可能与目标帧相似，导致模型为了更快收敛而直接复制 identity reference 的嘴唇动作，而非从音频学习正确唇形。

## 核心问题

- SyncNet 对 GT 音频-唇形对给出波动剧烈的余弦相似度，提供错误的训练梯度
- lip-sync loss 与 reconstruction loss 相互冲突，使模型陷入唇形同步差或视觉质量差的困境
- identity reference 中随机出现的唇形与目标相似时，模型倾向于"抄袭"而非学习正确映射
- SyncNet 平移不变性差（shift-invariance），轻微位移即影响预测结果

## 方法详解

### 1. Silent-Lip Generator ($G_S$)

为解决嘴唇泄漏问题，提出一个预处理模块 $G_S$，在将 identity reference 送入主生成器之前先将其嘴唇修改为闭合状态（"静默唇形"）。

- **架构**：与主生成器 $G_L$ 相同的 U-Net 架构
- **训练策略**：在 LRS2 上单独训练，**不使用任何 synchronization loss**，仅用 GAN loss + 像素 loss + 感知 loss
- **关键洞察**：没有 sync loss 的监督下，模型在输入静默音频时隐式地学会生成闭合唇形
- **推理时**：仅输入静默音频给 $G_S$，使其将任意 identity reference 的嘴唇变为闭合状态
- **效果**：消除了 identity reference 中嘴唇运动的多样性，从根源上切断了 lip leaking 路径

### 2. AVSyncNet

基于 ResNet-50 图像编码器 + ResNetSE-34 音频编码器重新设计 SyncNet，替代原始的简单 CNN 架构：

- **图像编码器**：使用 ResNet-50，输入为人脸下半部（112×224），具有更强的 shift-invariance
- **音频编码器**：使用 ResNetSE-34（专为频谱图设计的 ResNet-34 变体）
- **训练**：在 LRS2 上计算音频-唇形特征的余弦相似度 + BCE loss
- 每步输入 5 帧连续图像与对应音频，负样本从视频非重叠部分随机选取
- 实验表明 AVSyncNet 在 GT 数据上余弦相似度波动明显减小，且平移不变性大幅提升

### 3. Stabilized Synchronization Loss ($L_{ss}$)

即使 AVSyncNet 更稳定，仍无法完全消除不稳定性，因此提出稳定化同步损失：

$$L_{ss} = -\log\left(1 - \frac{|x - y| + \epsilon}{|x - y| + |y - d| + \epsilon}\right)$$

其中 $x = \text{AVsim}(I', A)$（生成图-音频相似度），$y = \text{AVsim}(I^{GT}, A)$（GT-音频相似度），$d = \text{AVsim}(I^R, A)$（identity reference-音频相似度）。

- **核心思想**：不直接使用生成唇形与音频的绝对相似度，而是计算 GT 对与生成对之间的**相对差距**——引导模型生成与 GT 具有相似同步分数的唇形
- **identity reference 惩罚项**：当 $d$ 较高（identity reference 与音频相似度高）时加大惩罚，进一步抑制嘴唇泄漏
- 类似蒸馏思想：忽略绝对分数，仅利用分数差提供梯度

### 4. 主生成器架构与训练

- **架构**：U-Net 设计，包含独立的 identity encoder 和 pose encoder（各为 CNN + 残差连接），face decoder 使用转置卷积 + skip connection
- **音频编码器**：使用 AVSyncNet 的预训练冻结音频编码器，获取更好的音频嵌入
- **Adaptive Triplet Loss ($L_{at}$)**：最小化生成图与 GT 距离、最大化生成图与 identity reference 距离，并根据 GT 与 identity reference 相似度自适应调节
- **总损失**：$L = L_{GAN} + 10 L_{pixel} + L_{per} + 2 L_{ss} + 0.5 L_{at}$
- **后处理**：使用 VQFR 提升输出视觉质量和分辨率

## 实验关键数据

在 LRS2 测试集上的主要指标（无后处理 / 有 VQFR 后处理 vs 之前 SOTA）：

| 指标 | 本文 (w/o FR) | 本文 (w/ VQFR) | 之前最佳 |
|------|:---:|:---:|:---:|
| SSIM ↑ | **0.952** | 0.905 | 0.87 (IPLAP) |
| PSNR ↑ | **32.64** | 31.80 | 29.67 (IPLAP) |
| FID ↓ | **3.83** | 5.23 | 4.10 (IPLAP) |
| LMD ↓ | **1.13** | 1.36 | 2.11 (IPLAP) |
| LSE-C ↑ | 8.41 | **8.52** | 8.53 (TalkLip) |
| LSE-D ↓ | 6.03 | **5.83** | 6.08 (TalkLip) |
| IFC ↓ | **0.16** | 0.27 | 0.20 (IPLAP) |

- LMD 指标大幅领先（1.13 vs 2.11），说明唇形准确度显著提升
- 视觉质量（SSIM/PSNR/FID）全面超越所有方法
- 在未见过的 LRW 和 HDTF 数据集上同样达到 SOTA

消融实验关键发现：
- Silent-lip generator 将 LMD 从 2.325 降至 1.741，LSE-C 从 7.271 升至 7.752
- Stabilized sync loss 将 PSNR 从 27.18 提升至 31.17，SSIM 从 0.872 升至 0.925
- Adaptive triplet loss 进一步将 PSNR 提升至 32.75，FID 降至 4.02
- AVSyncNet 替换 SyncNet 使 LSE-C 方差从 1.16 降至 0.97

## 亮点

- **问题分析透彻**：系统识别了 SyncNet 不稳定性和 lip leaking 两大根本问题，并提供了可视化和定量分析支撑
- **Silent-lip generator 设计巧妙**：通过无 sync loss 训练 + 静默音频输入的方式隐式学习闭唇生成，无需额外标注数据
- **Stabilized sync loss 原理优雅**：用相对差距替代绝对分数，同时融入 identity reference 惩罚，一个公式解决训练不稳定和 lip leaking 两个问题
- **消融实验充分**：逐步验证每个组件的贡献，包括不同后处理方法、不同 sync loss 变体、silent face 生成策略等
- 无后处理版本在多数指标上已大幅超越 SOTA，说明核心方法的有效性

## 局限与展望

- 输入分辨率仅 96×96，依赖 VQFR 后处理提升分辨率，后处理引入额外伪影且降低部分指标
- Silent-lip generator 需要单独预训练，增加了模型复杂度和训练成本
- 仅在 LRS2 上训练，未探索更大规模数据集的效果
- 未考虑 3D 面部先验（如 3DMM），可能限制了极端姿态下的表现
- 推理需要两次前向传播（$G_S$ + $G_L$），在实时应用中可能有性能瓶颈
- 未探索扩散模型等更新范式的潜力

## 与相关工作的对比

| 方法 | 核心策略 | 局限 |
|------|----------|------|
| Wav2Lip | SyncNet + lip-sync loss | SyncNet 不稳定、高分辨率下退化 |
| TalkLip | 全局音频编码器 + 唇读损失 | LSE-C 略优但 LSE-D 较差，泛化有伪影 |
| IPLAP | 中间 landmark + motion field | 视觉质量好但唇形同步不足 |
| DINet | 变形模块对齐特征 | FID 低但 LSE 指标差 |
| VideoReTalking | canonical expression 预处理 | 类似思路但身份保持和效率不如本文 |
| SIDGAN | shift-invariant APS-SyncNet | 分析问题类似，但本文方案更简洁有效 |

本文与 SIDGAN 问题意识最接近，但提出了不同的解决路径（稳定化损失 vs 金字塔模型），且本文的 silent-lip generator 是独特贡献。

## 启发与关联

- **相对损失设计**的思想可推广到其他存在"不可靠 teacher"的场景——当预训练评估模型本身有噪声时，用相对差距替代绝对分数是通用策略
- Silent-lip generator 的"弱条件训练→推理时利用退化行为"的思路可启发其他需要去除条件信息的任务
- lip leaking 问题在所有条件生成任务中普遍存在（如图像编辑、风格迁移中的信息泄漏），本文的分析和解决方案具有借鉴价值
- 独立的 identity encoder 和 pose encoder 设计支持了更好的任务解耦，这一模式在多条件生成中值得推广

## 评分
- 新颖性: ⭐⭐⭐⭐ (问题分析深入，三项贡献互补且各有新意)
- 实验充分度: ⭐⭐⭐⭐⭐ (多数据集、全面消融、定性定量分析完整)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，公式推导完整)
- 价值: ⭐⭐⭐⭐ (对 talking face generation 领域的核心痛点给出了系统性解决方案)

<!-- RELATED:START -->

## 相关论文

- [MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)
- [MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](../../CVPR2025/human_understanding/moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)
- [Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)
- [UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](../../CVPR2026/human_understanding/unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](../../AAAI2026/human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)

<!-- RELATED:END -->
