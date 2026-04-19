---
title: "INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations"
description: "INFP提出统一的音频驱动交互式头部生成框架，通过双轨音频驱动agent在说话和倾听状态间自然切换，无需手动角色分配"
tags: ["头部生成", "对话交互", "音频驱动", "扩散模型", "人脸动画"]
---

# INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations

**会议**: CVPR 2025  
**arXiv**: [2412.04037](https://arxiv.org/abs/2412.04037)  
**代码**: https://grisoon.github.io/INFP/ (项目页)  
**领域**: 图像生成 / 人脸动画  
**关键词**: 交互式头部生成, 双人对话, 音频驱动, 运动潜空间, 扩散Transformer

## 一句话总结

INFP 提出了一个统一的音频驱动交互式头部生成框架，通过双轨音频（agent + 对话伙伴）驱动 agent 在说话和倾听状态间自然切换，无需手动角色分配或显式角色切换，同时引入大规模 DyConv 数据集支持研究。

## 研究背景与动机

**领域现状**：音频驱动头部生成（audio-driven head generation）是构建对话智能体的核心技术。目前分为两个独立方向：talking-head generation（说话者头部生成，强调唇部同步）和 listening-head generation（倾听者头部生成，关注非语言反馈如点头、表情变化）。

**现有痛点**：现有方法只关注单侧通信，要么只能说，要么只能听。少数探索双人对话的工作（如 ViCo-X、DIM）将模型拆分为 Speaker Generator 和 Listener Generator，需要手动分配角色和显式切换。这种设计导致状态转换不自然，且无法覆盖双方同时说话等真实场景。

**核心矛盾**：将"说"和"听"分成两个独立模型是一种过度简化——真实对话中角色切换频繁且渐进，不存在明确的切换点。手动分割音频为说话片段和倾听片段再分别处理，必然导致过渡不连贯。

**本文目标** 如何让一个统一的模型根据双轨对话音频的内容，自动驱动 agent 在说话、倾听和交互状态间自然过渡。

**切入角度**：作者观察到，在双人对话中，agent 的状态完全由双轨音频决定——当 agent 自己的音频活跃时应表现为说话状态，当对方音频活跃时应表现为倾听状态。因此可以用音频信号强度隐式控制状态切换，而非显式判断。

**核心 idea**：用双轨音频同时检索语言和非语言运动记忆库，通过信号强度自动实现说/听状态的连续切换。

## 方法详解

### 整体框架

INFP 分为两个阶段：(1) Motion-Based Head Imitation（运动基头部模仿）：从真实对话视频中学习面部交际行为并编码到低维运动潜空间，用运动潜编码驱动静态肖像图动起来；(2) Audio-Guided Motion Generation（音频引导运动生成）：学习从双轨对话音频到运动潜编码的映射，通过扩散去噪实现音频驱动的交互式头部生成。

### 关键设计

1. **运动编码与头部模仿**:

    - 功能：建立解耦的运动潜空间，将面部动作从外观中分离
    - 核心思路：运动编码器 $E_m$ 将面部图像编码为低维 1D 运动潜编码。为实现解耦，设计了混合面部表征：(a) 遮蔽面部大部分像素，只保留眼部和嘴唇区域（最具表现力的部分），阻止头发、背景等无关信息干扰；(b) 用面部估计模型获取面部顶点，将轮廓点投影到遮蔽图像上提供朝向信息。然后用运动流估计模型 F 从源和驱动运动编码预测光流，warping 特征体积后经 face decoder 合成最终视频
    - 设计动机：使用隐式表征替代显式 3DMM 系数，因为 3DMM 表达力有限且表情系数与脸型存在纠缠

2. **Interactive Motion Guider（交互运动引导器）**:

    - 功能：从双轨音频中自适应提取交互式运动特征
    - 核心思路：设计两个可学习 memory bank——$M_v$（语言运动库，64 个 512 维可学习嵌入）和 $M_{nv}$（非语言运动库）。agent 音频 $A_{self}$ 作为 Query 通过 cross-attention 从 $M_v$ 中检索语言运动特征；对方音频 $A_{other}$ 作为 Query 从 $M_{nv}$ 中检索非语言运动特征。当 agent 说话时 $A_{self}$ 信号强，语言运动特征主导；当对方说话时 $A_{other}$ 信号强，非语言运动特征主导。两者通过逐元素相加 + MLP 融合为交互运动特征 $f_m$。还引入运动风格向量 $s_m$ 通过 StyleGAN2 的 style modulation 层编辑 memory bank 嵌入，注入全局情感/态度信息
    - 设计动机：双轨音频的信号强度自然对应了角色状态，无需任何显式角色判断逻辑。memory bank 存储了典型的运动模式，cross-attention 根据音频内容灵活组合

3. **Conditional Diffusion Transformer**:

    - 功能：将交互运动特征映射到预训练的运动潜空间
    - 核心思路：仅 4 个 block 的轻量 Transformer，每个 block 包含 self-attention → motion attention → temporal attention。motion attention 中以潜特征为 Query、交互运动特征 $f_m$ 为 Key-Value 进行 cross-attention。temporal attention 利用前一窗口最后 10 帧的运动潜编码作为条件，确保相邻窗口平滑过渡。使用 DDIM 采样器 20 步去噪
    - 设计动机：4-block 的轻量设计支持实时交互。temporal layer 参考 AnimateDiff 的做法确保时序连贯性

### 损失函数 / 训练策略

阶段 1 使用标准的图像重建和感知损失。阶段 2 使用扩散去噪损失（预测噪声），AdamW 优化器（lr=1e-4, wd=1e-2, bs=32）。训练策略：style vector 以 0.3 概率置零，运动特征和前序潜编码以 0.5 概率 drop 实现 classifier-free guidance。warm-up 阶段先用单侧对话片段训练初始化 memory bank，再用多轮对话数据完成全部训练。

## 实验关键数据

### 主实验

| 方法 | SSIM↑ | PSNR↑ | FID↓ | SyncScore↑ | SID↑ | Var↑ |
|------|-------|-------|------|------------|------|------|
| DIM | 0.651 | 20.42 | 34.36 | 4.778 | 0.766 | 0.825 |
| INFP | **0.834** | **31.56** | **15.73** | **7.188** | **2.613** | **2.386** |
| GT | 1.000 | - | 0.000 | 7.261 | 2.891 | 2.435 |

INFP 在所有指标上大幅领先 DIM，SyncScore (7.188 vs 4.778) 接近于 GT (7.261)，SID 和 Var 接近 GT 表明运动多样性优异。

### 消融实验

| 配置 | SSIM | FID | SyncScore | SID |
|------|------|-----|-----------|-----|
| INFP (完整) | 0.834 | 15.73 | 7.188 | 2.613 |
| w/o Motion Memory | 0.830 | 18.33 | 6.103 | 2.153 |
| w/o Style Modulation | 0.831 | 16.03 | 7.062 | 2.551 |
| w/ Intact Image (不做遮蔽) | 0.802 | 16.99 | 6.812 | 2.470 |
| w/ Landmarks Map (替代混合表征) | 0.821 | 16.33 | 6.833 | 2.601 |

### 关键发现

- Motion Memory 贡献最大（去掉后 SyncScore 降 1.085，FID 升 2.6），验证了 memory bank 对提取交互信息的关键作用
- 混合面部表征（遮蔽+轮廓点）优于完整图像（SSIM 0.834 vs 0.802）和纯 landmarks（0.821），说明去除无关信息对运动编码解耦至关重要
- 在 ViCo 倾听头部生成 benchmark 上，INFP 在 FD（18.63 vs DIM 23.88）和 SID（4.78 vs 3.71）上均超越 SOTA
- 用户研究（20 人 MOS 评分）中，INFP 在自然度（4.38 vs 2.71）、运动多样性（4.49 vs 2.14）上大幅领先 DIM

## 亮点与洞察

- **用双轨音频信号强度隐式切换角色是一个非常自然的设计**：当 agent 说话时 $A_{self}$ 信号强自然驱动说话运动，不需要任何显式角色判断。这比传统的"先判断谁在说话再分发到不同模型"简洁且鲁棒得多
- **Memory bank 作为运动模式库的思路可迁移**：可学习嵌入存储语言/非语言运动原型，用音频 cross-attention 检索组合的设计，可以应用到手势生成、全身动作合成等任务
- **DyConv 数据集的贡献不可忽视**：200+ 小时、高面部分辨率（>400×400）、音频分离+说话人检测的双人对话数据集，填补了该领域数据的空白

## 局限与展望

- 仅生成头部区域，未涉及手势、上半身等更丰富的非语言行为
- 数据集 DyConv 主要来自网络视频，场景以面对面对话为主，对电话、群聊等场景的泛化能力未验证
- 4-block diffusion transformer 虽然轻量，但实时推理性能（FPS）的具体数据论文中未明确给出
- 音频分离的准确度会直接影响 agent 和对方音频的质量，在嘈杂环境下的鲁棒性值得进一步研究

## 相关工作与启发

- **vs DIM**: DIM 将模型拆分为 Speaker/Listener Generator 需要手动角色分配和预训练后微调；INFP 是统一模型，性能在所有指标上大幅领先
- **vs ViCo-X**: ViCo-X 设计了显式 Role Switcher 桥接两个生成器，导致状态切换不自然；INFP 的隐式切换更流畅
- **vs Wav2Lip/VASA-1**: 这些 talking head 方法只处理说话状态，无法处理倾听或交互场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 提出了交互式头部生成的新范式，隐式角色切换和 motion memory bank 设计新颖
- 实验充分度: ⭐⭐⭐⭐ 交互/倾听/说话三个场景都有评估，但缺少与更多最新方法的对比（部分未开源）
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，但部分实现细节需要看补充材料
- 价值: ⭐⭐⭐⭐ 对构建更自然的对话 AI agent 有重要参考价值，DyConv 数据集也是有价值的贡献

<!-- RELATED:START -->

## 相关论文

- [PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](../../CVPR2026/image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation](echomimicv2_towards_striking_simplified_and_semi-body_human_animation.md)
- [InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions](../../ICLR2026/image_restoration/interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions.md)

<!-- RELATED:END -->
