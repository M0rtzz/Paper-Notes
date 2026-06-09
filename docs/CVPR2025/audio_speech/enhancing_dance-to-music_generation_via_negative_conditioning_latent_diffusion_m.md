---
title: >-
  [论文解读] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model
description: >-
  [CVPR 2025][音频/语音][舞蹈生成音乐] 提出 PN-Diffusion，利用正向播放和反向播放的舞蹈视频分别提取正负节奏条件，设计双向扩散与反向过程来联合训练 U-Net，增强生成音乐与舞蹈动作的节奏一致性和音乐质量…
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "舞蹈生成音乐"
  - "负条件扩散"
  - "双向扩散过程"
  - "节奏对齐"
  - "潜在扩散模型"
---

# Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.22138](https://arxiv.org/abs/2503.22138)  
**代码**: [https://github.com/Changchangsun/PN-Diffusion](https://github.com/Changchangsun/PN-Diffusion)  
**领域**: 扩散模型  
**关键词**: 舞蹈生成音乐, 负条件扩散, 双向扩散过程, 节奏对齐, 潜在扩散模型

## 一句话总结
提出 PN-Diffusion，利用正向播放和反向播放的舞蹈视频分别提取正负节奏条件，设计双向扩散与反向过程来联合训练 U-Net，增强生成音乐与舞蹈动作的节奏一致性和音乐质量，在 AIST++ 和 TikTok 数据集上 BCS 提升 1.80/3.85、BHS 提升 4.22/5.90。

## 研究背景与动机

**领域现状**：条件扩散模型在跨模态生成（T2I、T2A、T2V）上取得了显著成功。舞蹈生成音乐（D2M）任务要求从舞蹈视频生成节奏协调的伴奏音乐，是视频分享平台（TikTok、YouTube）的重要应用。现有 D2M 扩散方法如 CDCD 和 LORIS 提取舞蹈视频的视觉节奏和动作信息作为条件输入 U-Net。

**现有痛点**：现有方法只利用正向播放的舞蹈视频提取正向节奏线索和动作信息作为条件，忽略了反向播放视频携带的"负向"节奏信息。类比机器学习中正负样本的双向引导——正样本告诉模型"该做什么"，负样本告诉模型"该避免什么"——只有正向条件的训练是不完整的。

**核心矛盾**：如何在条件扩散模型中有效定义和利用"负条件"？将负条件直接影响正向过程的噪声预测，还是引入独立的负向噪声过程？这在序列化多模态 U-Net 架构中如何无缝集成？

**本文目标** (1) 如何定义 D2M 任务中的负条件？(2) 如何在 LDM 框架中集成正负条件进行双向训练？(3) 如何让负条件真正帮助提升生成音乐的节奏对齐和质量？

**切入角度**：巧妙利用舞蹈视频的时间特性——正向播放和反向播放提供相反的时间节奏信息。反向播放保留相同的姿态和过渡但方向相反，构成最忠实的负样本配对。

**核心 idea**：用反向播放的舞蹈视频作为负条件，设计双向扩散（加正/负噪声）和双向反向过程来联合训练 U-Net，让正条件指导恢复正噪声、负条件指导预测负噪声，双向强化节奏学习。

## 方法详解

### 整体框架
输入为舞蹈视频，输出为 Mel 频谱图（可转回音频）。流程：(1) 音频转为 256×256 的 Mel 频谱图，用 VAE 压缩到 32×32 的潜在空间；(2) 从正向/反向播放的视频中分别提取视觉嵌入（I3D）和动作信息（BlazePose + ST-GCN），拼接为正条件 $c^+$ 和负条件 $c^-$；(3) 在潜在空间上执行双向扩散和双向反向过程训练 U-Net；(4) 推理时只用正条件采样生成。

### 关键设计

1. **正负条件定义与提取（Positive & Negative Conditioning）**:

    - 功能：从舞蹈视频中提取双向节奏和动作条件
    - 核心思路：正条件 $c^+$ = 正向播放视频的 I3D 视觉嵌入 $p \in \mathbb{R}^{2048}$ 拼接 ST-GCN 动作嵌入 $q \in \mathbb{R}^{1024}$。负条件 $c^-$ = 相同处理但对反向播放的视频。I3D 捕捉视频帧的视觉节奏，BlazePose 提取 33 个人体关键点，ST-GCN 编码时空图序列中的运动模式。反向播放改变了关键点的时序方向，从而反转运动节奏。
    - 设计动机：直接定义"完全相反"的负样本很难，但反向播放自然提供了时间维度上的对立——相同姿态和过渡但时序方向相反。这比随机选择不相关视频作为负样本更有针对性，形成更忠实的正负配对。

2. **双向扩散过程（Dual Diffusion Process）**:

    - 功能：构建两个平行的前向扩散过程，分别添加方向相反的噪声
    - 核心思路：从同一起点 $z_0$ 出发，正向扩散添加噪声 $\epsilon$：$z_t^+ = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$；负向扩散添加相反噪声 $-\epsilon$：$z_t^- = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} (-\epsilon)$。两个过程共享相同的 noise schedule 和起点，只是添加噪声的方向相反。
    - 设计动机：传统扩散模型只有一个前向过程，U-Net 只需学习预测一个方向的噪声。引入对称的负向扩散，使 U-Net 不仅要恢复正噪声的能力，还要通过负条件判断负噪声的方向。这种双向对比训练强化了 U-Net 对时间节奏方向的敏感性。

3. **双向去噪目标函数（Bi-directional Denoising Objective）**:

    - 功能：将正负条件下的噪声预测目标统一为训练损失
    - 核心思路：U-Net 用正条件预测正噪声 $\epsilon_\theta^+(z_t^+, t, c^+)$，用负条件预测负噪声 $\epsilon_\theta^-(z_t^-, t, c^-)$。总损失为 $L_\epsilon = \alpha \|\epsilon - \epsilon_\theta^+(z_t^+, t, c^+)\|_2^2 + (1-\alpha) \|-\epsilon - \epsilon_\theta^-(z_t^-, t, c^-)\|_2^2$，其中 $\alpha$ 控制正负任务的权重平衡。推理时只用正条件进行采样。
    - 设计动机：正条件指导"生成什么样的节奏"，负条件指导"避免生成什么样的节奏"，双向目标让同一个 U-Net 同时学习这两种能力。正噪声的预测能力因为有负噪声的对比而增强，类似对比学习的效果。

### 损失函数 / 训练策略
- 第一阶段：训练 VAE 编码器/解码器实现 Mel 频谱图的感知压缩（perceptual loss + patch-based adversarial loss）
- 第二阶段：在潜在空间训练条件 DDPM，使用双向去噪目标 $L_\epsilon$
- $\alpha$ 通过网格搜索确定最优值
- 采样率 22,050 Hz，5 秒音乐片段，扩散步数 1000，batch size 32
- 推理时只用正条件（正向播放的舞蹈视频），采样 1000 步

## 实验关键数据

### 主实验

| 数据集 | 指标 | PN-Diffusion | 之前SOTA (LORIS) | 提升 |
|--------|------|-------------|-----------------|------|
| AIST++ | BCS ↑ | **97.72** | 95.92 | +1.80 |
| AIST++ | BHS ↑ | **93.73** | 89.51 | +4.22 |
| AIST++ | F1 ↑ | **95.69** | 92.67 (CMT) | +2.01* |
| AIST++ | FAD_v ↓ | **5.62** | 6.27 | -0.65 |
| AIST++ | FAD_p ↓ | **13.47** | 16.49 | -3.02 |
| TikTok | BCS ↑ | **92.68** | 88.83 | +3.85 |
| TikTok | BHS ↑ | **88.56** | 82.66 | +5.90 |
| TikTok | F1 ↑ | **90.56** | 85.71 | +4.85 |

### 消融实验

| 配置 | BCS | BHS | F1 | 说明 |
|------|-----|-----|-----|------|
| Only positive (baseline) | ~95.9 | ~89.5 | ~92.7 | 只用正条件 |
| + Negative conditioning | **97.72** | **93.73** | **95.69** | 加入负条件双向训练 |
| $\alpha=1.0$ (仅正) | 较低 | 较低 | 较低 | 退化为标准扩散 |
| $\alpha=0.5$ | 最优附近 | 最优附近 | 最优附近 | 正负均衡 |
| $\alpha=0.0$ (仅负) | 最低 | 最低 | 最低 | 无正向信号 |

### 关键发现
- 在 BCS 和 BHS 两个核心节奏对齐指标上，PN-Diffusion 显著超越所有基线，说明负条件确实增强了节奏学习
- FAD 指标（衡量生成音乐质量）也全面领先，表明负条件训练不仅改善对齐还提升了音乐质量
- TikTok 数据集上提升比 AIST++ 更大（BHS +5.90 vs +4.22），可能因为 TikTok 视频动作更多样，负条件的对比效果更明显
- $\alpha$ 参数需要仔细调节，过度偏向正或负都会降低性能，0.5 附近最优
- 主观评估（MOS 和 Turing Test）也confirm了客观指标的趋势

## 亮点与洞察
- **反向播放作为负条件的巧妙定义**：在 D2M 任务中定义"负样本"一直是难题，作者巧妙利用视频反向播放来反转节奏，这是一个零成本、零标注且语义上合理的负样本构造方案。可迁移到其他视频条件生成任务。
- **双向扩散过程的对称设计**：正负噪声方向相反但共享起点和 schedule，在不增加模型参数的前提下（共享同一 U-Net）实现了对比训练。这种双向过程可以推广到其他需要"引导方向"的条件生成任务。
- **推理时零开销**：负条件只参与训练，推理时完全不引入额外计算，实际部署与标准 LDM 完全一致。

## 局限与展望
- 仅在 5 秒片段上实验，生成更长音乐（如完整歌曲）的质量和连贯性未知
- "反向播放=负样本"的假设在某些对称性强的舞蹈（如华尔兹来回旋转）中可能不成立
- 未探索更复杂的负样本构造策略（如速度变化、乱序帧等）
- 音乐多样性评估不足——同一舞蹈能否生成不同风格的合适音乐？
- 仅关注节奏对齐，未考虑音乐的旋律结构、和声等更高级音乐特征
- FAD 指标依赖预训练特征提取器，可能存在评估偏差

## 相关工作与启发
- **vs CDCD**: CDCD 和 LORIS 只用正向视频条件的标准 LDM，PN-Diffusion 在此基础上引入负条件双向训练，在完全兼容的框架下取得显著提升。
- **vs Classifier-Free Guidance**: CFG 通过 unconditional 和 conditional 的差值做引导，PN-Diffusion 的思想类似但将"来自方向"拓展为"正负条件"的对比。
- **vs D2M-GAN**: GAN-based 方法难以建模长程时间依赖，PN-Diffusion 基于扩散模型天然处理序列生成，效果全面更优。

## 评分
- 新颖性: ⭐⭐⭐⭐ 负条件+双向扩散的思路在 D2M 中是首次，反向播放做负样本设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、8 个指标、主观+客观评估
- 写作质量: ⭐⭐⭐⭐ 动机和方法描述清晰，数学形式化完整
- 价值: ⭐⭐⭐⭐ 负条件训练的通用思路有启发性，但 D2M 本身应用范围较窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction](../../NeurIPS2025/audio_speech/mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/audio_speech/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
