---
title: >-
  [论文解读] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing
description: >-
  [CVPR 2025][图像生成][电影配音] 本文提出 EmoDubber，一个情感可控的电影配音架构，通过时长级对比学习对齐唇动与韵律、发音增强策略提升清晰度、基于流匹配的正负引导机制控制情感类型和强度，在唇形同步和发音清晰度上全面超越现有方法。
tags:
  - CVPR 2025
  - 图像生成
  - 电影配音
  - 情感可控
  - 唇形同步
  - 流匹配
  - 正负引导
---

# EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing

**会议**: CVPR 2025  
**arXiv**: [2412.08988](https://arxiv.org/abs/2412.08988)  
**代码**: https://github.com/GalaxyCong/DubFlow (有)  
**领域**: 扩散模型  
**关键词**: 电影配音, 情感可控, 唇形同步, 流匹配, 正负引导

## 一句话总结

本文提出 EmoDubber，一个情感可控的电影配音架构，通过时长级对比学习对齐唇动与韵律、发音增强策略提升清晰度、基于流匹配的正负引导机制控制情感类型和强度，在唇形同步和发音清晰度上全面超越现有方法。

## 研究背景与动机

**领域现状**：电影配音（Visual Voice Cloning, V2C）旨在将文本转化为与视频唇动同步且带有指定说话人声纹的语音。现有方法分为两类：一类聚焦说话人风格表示（V2C-Net、StyleDubber），一类利用视频信息建模韵律（HPMDubbing、MCDubber）。

**现有痛点**：(1) 音频-视觉同步和清晰发音难以同时保证——现有方法在视频帧和梅尔频谱层面工作，忽略了音素级别的发音信息，导致生成语音含混不清；(2) 情感表达僵硬且缺乏可控性——用户无法指定情感类型和强度，这在电影后期制作中是关键需求。

**核心矛盾**：配音任务需要在四个维度同时优化——唇形同步、发音清晰、说话人克隆、情感控制——而现有方法只解决了前三个中的部分。

**本文目标** (1) 如何实现精确的唇动-韵律对齐？(2) 如何提升发音清晰度？(3) 如何让用户灵活控制情感类型和强度？

**切入角度**：作者观察到人类语音中情感往往是混合的而非单一的，因此设计了正负引导机制——增强目标情感的同时抑制其他情感，实现更精确的情感控制。

**核心 idea**：通过时长级对比学习对齐唇动和韵律，用发音增强融合音素序列，再用流匹配的正负分类器引导实现情感强度的灵活控制。

## 方法详解

### 整体框架

EmoDubber 接收四个输入：静默视频 $V_l$、参考音频 $R_a$、文本 $T_p$、用户情感指令 $E = \{c, \alpha, \beta\}$。输出是情感可控的配音音频 $\hat{Y}$。流程分四个模块依次处理：(1) LPA 对齐唇动与韵律；(2) PE 增强发音信息；(3) SIA 注入说话人风格生成声学先验；(4) FUEC 用流匹配生成梅尔频谱并注入情感。最终由声码器转换为波形。

### 关键设计

1. **唇动相关韵律对齐（Lip-related Prosody Aligning, LPA）**:

    - 功能：学习唇动与语音韵律的内在一致性，建立正确的时序对齐
    - 核心思路：将唇动嵌入 $\mathcal{E}$ 作为 Query，音素韵律嵌入 $\mathcal{O}_p$（包含风格音素 + 音高 + 能量）作为 Key/Value，通过多头注意力得到唇-韵律上下文序列 $C_{pho}$。关键创新是时长级对比学习（DLCL）：用 MFA 强制对齐生成的"0-1"时长矩阵 $M^{gt}_{lip,pho}$ 作为正样本对，鼓励正确时序位置的注意力权重高于其他位置。损失为 $\mathcal{L}_{cl} = -\log \frac{\sum\exp(\text{sim}^+/\tau)}{\sum\exp(\text{sim})}$，保证单调性和满射性
    - 设计动机：之前方法用简单 MSE 或无约束注意力，无法保证唇动和音素的正确时序对应。DLCL 通过正负配对的对比学习显式强制单调对齐，相比对角约束更灵活且更精确

2. **发音增强策略（Pronunciation Enhancing, PE）**:

    - 功能：将音素级信息扩展到视频帧级别并与唇韵律序列融合，提升语音清晰度
    - 核心思路：用单调对齐搜索（MAS）从注意力矩阵中提取每个音素的显式时长 $D_p$，通过长度调节器将音素嵌入 $\mathcal{O}_s \in \mathbb{R}^{P \times d_m}$ 扩展到视频级 $\mathcal{O}^v_s \in \mathbb{R}^{F \times d_m}$。然后用音视频高效 Conformer（AVEC，5 个 Conformer Block + CTC 层）融合两种特征：唇-韵律上下文 $C_{pho}$ 和发音增强序列 $\mathcal{O}^v_s$。CTC 层通过最大化正确音素概率保证发音
    - 设计动机：现有配音方法在帧级或梅尔频谱级工作，忽略了音素级发音细节。通过显式扩展音素序列并用 Conformer 融合，为生成过程提供音素级监督，避免"含混不清"的问题

3. **基于流匹配的用户情感控制（Flow-based User Emotion Controlling, FUEC）**:

    - 功能：在流匹配生成过程中注入用户指定的情感类型和强度
    - 核心思路：用 OT-CFM 训练流匹配预测网络（FMPN）生成梅尔频谱。推理时引入正负引导机制（PNGM）：用预训练情感分类器 $\psi$ 预测当前中间状态 $\phi_t(x)$ 的情感分布，修改速度场为 $\tilde{v}_{t,i} = v_t + \gamma(\alpha \nabla\log p_\psi(c_i|\phi_t) - \beta \nabla\log p_\psi(\sum_{j\neq i} l_j c_j|\phi_t))$。$\alpha$ 正引导增强目标情感 $c_i$，$\beta$ 负引导抑制其他情感的加权混合。用户通过调节 $\alpha \in [0,9]$、$\beta \in [0,2]$ 灵活控制
    - 设计动机：传统分类器引导只增强目标情感，但人类语音中情感是混合的。PNGM 通过同时增强和抑制实现更精确的情感分离——增大 $\alpha$ 让目标情感更突出，增大 $\beta$ 让其他情感更弱，两者可独立调节

### 损失函数 / 训练策略

总训练损失包含：流匹配损失 $\mathcal{L}_\theta$（预测向量场 vs 目标向量场的 MSE）、对比学习损失 $\mathcal{L}_{cl}$（LPA 的 DLCL）、CTC 损失（PE 中确保发音正确）。情感分类器 $\psi$ 在 Emobox 的 13 个情感数据集（50,000+ 录音）上预训练。音素编码器、USL 和流解码器在 LibriSpeech 上预训练。推理时 $\gamma=15$。

## 实验关键数据

### 主实验

Chem 基准（Setting 1.0 + 2.0）：

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | SECS↑ |
|------|--------|--------|------|-------|
| GT | 8.12 | 6.59 | 3.85 | 100.0 |
| HPMDubbing | 7.85 | 7.19 | 16.05 | 85.09 |
| StyleDubber | 3.87 | 10.92 | 13.14 | 87.72 |
| Speaker2Dub | 3.76 | 10.56 | 16.98 | 74.73 |
| **EmoDubber** | **8.11** | **6.92** | **11.72** | **90.62** |

零样本设置（Dub 3.0）：

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | MOS-S↑ | MOS-N↑ |
|------|--------|--------|------|--------|--------|
| StyleDubber | 6.17 | 9.11 | 15.10 | 4.03 | 3.85 |
| Speaker2Dub | 4.83 | 10.39 | 15.91 | 3.98 | 4.01 |
| **EmoDubber** | **7.40** | **6.65** | **14.03** | **4.07** | **4.05** |

### 消融实验

| 配置 | LSE-C↑ | LSE-D↓ | WER↓ | 说明 |
|------|--------|--------|------|------|
| Full EmoDubber | 8.11 | 6.92 | 11.72 | 完整模型 |
| w/o DLCL | ~5.5 | ~9.0 | ~14.0 | 去掉对比学习，唇同步显著下降 |
| w/o PE | ~7.0 | ~7.5 | ~15.0 | 去掉发音增强，WER 上升 |
| w/o PNGM | - | - | - | 情感强度不可控 |

### 关键发现

- EmoDubber 在唇同步指标（LSE-C: 8.11 vs GT: 8.12）上几乎匹配真值，远超所有对比方法
- WER 从 HPMDubbing 的 16.05% 和 StyleDubber 的 13.14% 降至 11.72%，发音清晰度显著提升
- 说话人相似度 SECS 达 90.62%，超越所有方法包括 StyleDubber 的 87.72%
- 在零样本多说话人设置下仍保持优势，说明泛化能力强
- PNGM 中 $\alpha$ 和 $\beta$ 可独立控制：增大 $\alpha$ 增强目标情感的 Intensity Score，增大 $\beta$ 抑制其他情感

## 亮点与洞察

- **唇同步近真值的突破**：LSE-C 8.11 vs GT 8.12，说明 DLCL 的显式时长级对齐极为有效。这比之前 HPMDubbing 的层级建模更直接
- **正负引导机制（PNGM）概念新颖**：不只增强目标情感，还主动抑制其他情感的混合成分，更符合人类情感的混合特性。这个设计可迁移到任何需要属性控制的流匹配/扩散生成任务
- **端到端的音素到波形管线**：整合了对齐、增强、风格注入和情感控制，是目前最完整的配音系统

## 局限与展望

- 情感分类器是预训练固定的，限制了能处理的情感类别数量
- 仅在英文数据集上验证，跨语言适应性未知
- 训练需要额外标注的 MFA 对齐和情感标签
- PNGM 的 $\gamma$、$\alpha$、$\beta$ 需要用户手动设定，交互成本较高
- 未与最新的大规模语音生成模型（如 VALL-E、VoiceCraft）对比

## 相关工作与启发

- **vs HPMDubbing**: HPM 通过唇、面部、场景三层级建模视觉信息到韵律。EmoDubber 用 DLCL 在时长级显式对齐，唇同步大幅领先（LSE-C 8.11 vs 7.85），且增加了情感控制
- **vs StyleDubber**: StyleDubber 用多尺度风格适配器增强说话人特征。EmoDubber 的 SIA 模块功能类似，但通过 FUEC 额外增加了情感控制维度
- **vs 情感 TTS（如 EmoSphere）**: 传统情感 TTS 只用正引导增强情感。EmoDubber 的 PNGM 加入负引导抑制非目标情感，控制更精确

## 评分

- 新颖性: ⭐⭐⭐⭐ 正负引导机制和时长级对比学习都有新意，但整体是多模块组合
- 实验充分度: ⭐⭐⭐⭐ 多个配音数据集、零样本设置、MOS 评测覆盖全面
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰，公式与图的配合好
- 价值: ⭐⭐⭐⭐ 统一了唇同步+发音+情感控制的配音系统，对电影后期制作有明确应用

<!-- RELATED:START -->

## 相关论文

- [ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)
- [OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)
- [3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)

<!-- RELATED:END -->
