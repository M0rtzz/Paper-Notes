---
title: >-
  [论文解读] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance
description: >-
  [CVPR 2026][视频理解][轨迹可控视频生成] 提出 FlashMotion，首个实现少步（4步）轨迹可控视频生成的三阶段训练框架，通过训练轨迹适配器→蒸馏快速生成器→混合对抗+扩散微调适配器的策略，在 4 步推理下同时超越现有多步方法的视觉质量和轨迹精度，实现 47 倍加速。
tags:
  - CVPR 2026
  - 视频理解
  - 轨迹可控视频生成
  - 少步蒸馏
  - 对抗训练
  - 扩散判别器
  - 视频加速
---

# FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance

**会议**: CVPR 2026  
**arXiv**: [2603.12146](https://arxiv.org/abs/2603.12146)  
**代码**: https://github.com/quanhaol/FlashMotion  
**领域**: 视频理解 / 视频生成  
**关键词**: 轨迹可控视频生成, 少步蒸馏, 对抗训练, 扩散判别器, 视频加速

## 一句话总结

提出 FlashMotion，首个实现少步（4步）轨迹可控视频生成的三阶段训练框架，通过训练轨迹适配器→蒸馏快速生成器→混合对抗+扩散微调适配器的策略，在 4 步推理下同时超越现有多步方法的视觉质量和轨迹精度，实现 47 倍加速。

## 研究背景与动机

**领域现状**：扩散模型驱动的视频生成取得了显著进展，尤其是轨迹可控视频生成——用户指定前景物体的运动轨迹（bbox 或分割 mask），模型沿预定轨迹生成视频。MagicMotion、Tora、LeviTor 等方法通过在基础视频生成模型上添加轨迹适配器（adapter）实现了精确的运动控制。

**现有痛点**：所有现有的轨迹可控方法都依赖多步去噪推理（50步以上），导致生成一段 121 帧的视频需要约 1160 秒（>19分钟）。虽然视频蒸馏方法（如 DMD、LCM、CausVid）可以将通用视频生成模型压缩为少步版本，但直接将这些蒸馏方法应用于轨迹可控生成会导致视觉质量和轨迹精度的显著退化。

**核心矛盾**：多步适配器（SlowAdapter）是在多步生成器（SlowGenerator）的渐进去噪路径上训练的，其轨迹条件通过逐步细化引导噪声。而少步生成器（FastGenerator）的去噪路径完全不同——仅用 4 步就要完成全部生成。因此 SlowAdapter 与 FastGenerator 天然不兼容。直接组合会导致色彩偏移、模糊和轨迹失控。

**本文目标**：设计一个训练框架，使轨迹适配器能在少步生成器上正常工作，在 4 步推理内同时保证视觉质量和轨迹精度。

**切入角度**：作者发现用标准扩散损失微调适配器可以恢复轨迹精度但会产生严重模糊（因为像素级监督无法保证分布一致性），引入对抗训练可以消除模糊但会损失轨迹精度。因此需要同时使用两种损失并动态平衡。

**核心 idea**：三阶段训练——先训练多步适配器，再蒸馏快速生成器，最后用扩散损失+对抗损失的混合策略微调适配器以适配快速生成器。

## 方法详解

### 整体框架

FlashMotion 的训练流程分为三个阶段：**Stage 1** 在多步视频生成器（Wan2.2-TI2V-5B）上训练 SlowAdapter 来学习轨迹控制；**Stage 2** 通过 DMD 蒸馏将多步生成器压缩为 4 步 FastGenerator；**Stage 3** 用混合扩散+对抗策略将 SlowAdapter 微调为适配 FastGenerator 的 FastAdapter。推理时，FastGenerator + FastAdapter 仅需 4 步去噪即可生成轨迹精确的高质量视频。

### 关键设计

1. **轨迹适配器架构（Trajectory Adapter）**:

    - 功能：将用户指定的运动轨迹（bbox/mask）注入视频生成过程
    - 核心思路：设计了两种适配器架构——ControlNet 和轻量级 ResNet。适配器块数与基础 DiT 的 block 数一致。3D VAE 编码器将轨迹图编码为潜空间表示 $Z_{trajectory} \in R^{T/4 \times H/16 \times W/16 \times 48}$，每个适配器 block 的输出通过零初始化卷积层添加到对应的 DiT block，提供轨迹引导。训练采用由密到疏的渐进策略——先用分割 mask 训练 4.6K 步，再用 bbox 微调 5.4K 步
    - 设计动机：ControlNet 参数量大（10.28B）但轨迹控制更精确；ResNet 参数少（5.02B）但推理更快。两种架构的设计验证了 FlashMotion 框架的通用性

2. **扩散判别器（Diffusion Discriminator）**:

    - 功能：在对抗训练中区分真实视频和生成视频，消除因纯扩散损失导致的模糊伪影
    - 核心思路：判别器骨干从 Wan2.2-TI2V-5B 克隆，冻结骨干参数，仅训练新加的注意力分类器。分类器接收 DiT 中间层特征，通过三层注意力处理一个可学习的 query token——**语义自注意力**（整合首帧图像和文本信息）、**轨迹交叉注意力**（关注轨迹 token）、**视频交叉注意力**（关注视频 token），最终输出真/假 logit
    - 设计动机：纯扩散损失 $\mathcal{L}_{diffusion} = \|G_\theta(x_t, t) - x_0^{real}\|^2$ 只提供像素级监督，无法保证生成分布与真实分布的一致性，导致模糊。判别器通过分布级约束消除模糊，而三层注意力设计使其能同时考虑语义、轨迹和视频信息

3. **动态扩散损失缩放（Dynamic Diffusion Loss Scale）**:

    - 功能：平衡扩散损失和对抗损失的梯度量级
    - 核心思路：总损失为 $\mathcal{L} = \mathcal{L}_{\mathcal{G}} + \lambda \mathcal{L}_{diffusion}$，其中 $\lambda = \frac{1}{4 \times 10^{-3} \times step + 0.1}$。训练初期 $\lambda$ 较大，以扩散损失为主保证轨迹对齐；随训练进行 $\lambda$ 逐渐减小，让对抗损失主导以提升视觉质量
    - 设计动机：实验发现训练早期扩散损失的梯度远大于对抗损失，直接等权组合仍会导致模糊。动态缩放使两种损失在不同训练阶段发挥各自优势——先"画对"再"画好"

### 损失函数 / 训练策略

- **Stage 1**（SlowAdapter）：标准扩散去噪损失，16 GPU × 10K 步
- **Stage 2**（FastGenerator）：DMD 分布匹配损失 $\nabla\mathcal{L}_{DMD} = \mathbb{E}[-(s_{real} - s_{fake})\frac{dG_\theta}{d\theta}]$，16 GPU × 5.5K 步
- **Stage 3**（FastAdapter）：混合损失 $\mathcal{L} = \mathcal{L}_{\mathcal{G}} + \lambda\mathcal{L}_{diffusion}$，判别器和适配器交替优化（1:5 更新比），仅需 **4 GPU × 1K 步**，极轻量

## 实验关键数据

### 主实验（FlashBench）

| 方法 | 步数 | FID↓ | FVD↓ | M IoU↑ | B IoU↑ | 去噪时间(s) |
|--------|------|------|----------|------|------|------|
| MagicMotion | 50 | 20.03 | 138.83 | 68.10 | 73.68 | 1158.63 |
| Wan+ResNet | 50 | 19.03 | 139.61 | 52.19 | 57.76 | 333.00 |
| DMD (ResNet) | 4 | 24.38 | 228.33 | 43.24 | 52.61 | 11.72 |
| LCM (ResNet) | 4 | 26.79 | 462.09 | 55.31 | 60.80 | 11.72 |
| **FlashMotion (ResNet)** | **4** | **15.81** | **108.96** | **63.96** | **70.01** | **11.72** |
| **FlashMotion (ControlNet)** | **4** | **14.35** | **96.08** | **69.15** | **75.38** | **24.44** |

### 消融实验（FlashBench, ResNet）

| 配置 | FID↓ | FVD↓ | M IoU↑ | B IoU↑ |
|------|---------|------|------|------|
| Slow Adapter 直接用 | 22.75 | 168.46 | 49.79 | 56.62 |
| w/o 扩散损失 | 18.87 | 161.07 | 52.04 | 58.04 |
| w/o GAN 损失 | 22.74 | 206.75 | 65.82 | 70.60 |
| w/o 动态缩放 | 26.32 | 210.93 | 65.54 | 69.77 |
| **FlashMotion (完整)** | **15.81** | **108.96** | **63.96** | **70.01** |

### 关键发现

- **FlashMotion 4 步超越 MagicMotion 50 步**：ControlNet 版本的 FID 14.35 vs 20.03，FVD 96.08 vs 138.83，且实现 **47× 加速**（24s vs 1158s）
- **GAN 损失是消除模糊的关键**：移除 GAN 损失后 FVD 从 108.96 暴涨至 206.75（约 90% 退化），但轨迹精度反而略好，说明 GAN 损失主要影响视觉质量
- **扩散损失是轨迹精度的关键**：移除扩散损失后 M IoU 从 63.96 降至 52.04，生成物体明显偏离预定轨迹
- **动态缩放不可或缺**：固定 $\lambda=1$ 导致 FID 从 15.81 升至 26.32（模糊加重），证明训练初期需抑制扩散损失的过大梯度
- ControlNet 适配器在轨迹精度和视觉质量上全面优于 ResNet，但推理时间约为后者的 2 倍

## 亮点与洞察

- **三阶段拆解的优雅设计**：将"快速+可控"分解为先学控制、再学快速、最后对齐两者，每个阶段目标清晰。Stage 3 仅需 1K 步训练即可完成，体现了 SlowAdapter 提供的强先验的价值
- **扩散判别器的三层注意力设计**：语义、轨迹、视频三路信息的分离式注入使判别器能同时感知多维度信息，比简单的 CNN 判别器更适合条件生成任务。这一设计可迁移到其他条件视频生成的加速中
- **动态损失缩放的实用价值**：简单但有效地解决了扩散损失和对抗损失的梯度不平衡问题，写法简洁（一行公式），但对最终效果影响巨大

## 局限与展望

- FlashBench 虽然支持长视频评测，但当前仅基于 MagicData 的扩展，数据多样性有限
- Stage 2 的蒸馏依赖特定的 DMD 方法，换用其他蒸馏方法（如 CausVid）能否进一步提升尚未探索
- 当前仅支持 bbox/mask 轨迹条件，对更灵活的轨迹表示（如点轨迹、语义描述）的支持待研究
- ControlNet 版本虽然效果更好但仍有 10.28B 参数，在端侧部署仍面临挑战
- DMD 和 GAN 在 ControlNet 下直接 OOM，限制了更多蒸馏方法的比较

## 相关工作与启发

- **vs MagicMotion**：MagicMotion 在 CogVideoX 上实现精确轨迹控制但需 50 步推理；FlashMotion 在 Wan2.2 上实现 4 步推理且效果更好
- **vs APT/APT2**：APT 系列的一步对抗蒸馏在通用视频生成中有效，但不支持轨迹条件；FlashMotion 的扩散判别器设计借鉴了 APT 的思路但加入了轨迹感知
- **vs Tora/LeviTor**：Tora 和 LeviTor 分别基于 CogVideoX 和 SVD，效果均不及 FlashMotion 且推理更慢

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个少步轨迹可控视频生成框架，三阶段拆解合理，判别器设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 benchmark + 两种适配器架构 + 详尽消融，包括按物体数量分组的分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，图示到位
- 价值: ⭐⭐⭐⭐⭐ 47倍加速且质量更优，对可控视频生成的实用化有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)

<!-- RELATED:END -->
