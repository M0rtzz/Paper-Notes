---
title: >-
  [论文解读] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video
description: >-
  [CVPR 2025][模型压缩][3D talking head] 提出 InsTaG，通过 Identity-Free Pre-training 从多人长视频中提取通用运动先验，再通过 Motion-Aligned Adaptation 仅用 5 秒视频即可快速学习高保真个性化 3D 说话人头像，实现 82.5 FPS 实时推理。
tags:
  - CVPR 2025
  - 模型压缩
  - 3D talking head
  - 3D Gaussian Splatting
  - few-shot adaptation
  - identity-free pre-training
  - motion alignment
  - real-time rendering
---

# InsTaG: Learning Personalized 3D Talking Head from Few-Second Video

**会议**: CVPR 2025  
**arXiv**: [2502.20387](https://arxiv.org/abs/2502.20387)  
**代码**: [项目页面](https://fictionarry.github.io/InsTaG/)  
**领域**: 3d_vision  
**关键词**: 3D talking head, 3D Gaussian Splatting, few-shot adaptation, identity-free pre-training, motion alignment, real-time rendering

## 一句话总结

提出 InsTaG，通过 Identity-Free Pre-training 从多人长视频中提取通用运动先验，再通过 Motion-Aligned Adaptation 仅用 5 秒视频即可快速学习高保真个性化 3D 说话人头像，实现 82.5 FPS 实时推理。

## 研究背景与动机

**领域现状**: 基于辐射场（NeRF/3DGS）的 3D 说话人头像合成已能生成高保真个性化视频，但 person-specific 方法需要大量高质量视频数据和长时间训练才能适配每个新身份。

**现有痛点**:
1. **数据需求高**: 现有 person-specific 方法（ER-NeRF、TalkingGaussian 等）需要数分钟的长视频训练
2. **适配时间长**: 从零训练一个新身份需要数小时
3. **少样本方法质量差**: 现有 one-shot/few-shot 方法（Real3DPortrait、MimicTalk）为获得泛化性牺牲了个性化质量和推理效率

**核心矛盾**: 要实现快速适配新身份，需要通用运动先验；但 person-specific 架构无法直接在多人数据上预训练，因为不同身份的外观和个性化运动会互相冲突。

**切入角度**: 完全解耦通用运动学习和个性化适配——预训练阶段用临时的个性化场来"过滤"身份信息；适配阶段用运动对齐器将预训练运动适配到新身份。

## 方法详解

### 整体框架

InsTaG 基于 3DGS 的 person-specific 合成器，采用 face-mouth 分解架构（脸部+口腔内部两个分支），每个分支包含结构场（静态 Gaussian）和运动场（预测变形）。核心流程：
1. **预训练阶段**: 从多人长视频语料中学习 Universal Motion Field (UMF)
2. **适配阶段**: 用少量视频初始化新身份的 person-specific 模型，通过 Motion Aligner 和 Face-Mouth Hook 快速对齐

### 关键设计

**1. Identity-Free Pre-training**
- **功能**: 从多人长视频中提取通用运动先验，存入共享的 Universal Motion Field (UMF)。
- **核心思路**: 为每个训练视频分配临时的 Personalized Field（私有结构场 $\theta_P^i$ + 私有小运动场 $\mathcal{D}_P^i$），存储身份外观和个性化运动。UMF 预测身份无关的通用变形 $\delta_U$，Personalized Field 预测残差运动 $\delta_P^i$，两者叠加后渲染：$\tilde{\theta}_P^i = \theta_P^i + \delta_U + \delta_P^i$。
- **Negative Contrast Loss**: 截断点积损失 $\mathcal{L}_C = \mathbb{I}_{trunc}(\Delta\mu_P^i \cdot \Delta\mu_P^j)$，鼓励不同身份的个性化运动尽量不同，从而最大化过滤出通用运动到 UMF。
- **设计动机**: 解决 person-specific 架构在多人数据上的身份冲突问题；对比损失的截断设计避免过度推离已经差异足够大的运动。

**2. Motion-Aligned Adaptation**
- **Motion Aligner**: 学习坐标偏移 $\Delta\mu_A$ 和运动缩放因子 $\tau_A$，解决新身份与 UMF 隐含面部结构的偏差和运动尺度差异。查询 UMF 前先加偏移：$\Delta\mu' = \Delta\mu \times \tau_A$，其中 $\Delta\mu \in \mathcal{U}(\mu + \Delta\mu_A, \mathbf{C})$。
- **Face-Mouth Hook**: 从脸部分支的变形中取最大/最小 $\Delta\mu$ 作为唇部运动线索，输入口腔运动场指导口腔内部运动对齐。缩放因子 $\tau_M$ 由唇部运动距离 $\mu_{dist}$ 控制，确保口腔开合度与唇部同步。
- **设计动机**: 直接用预训练 UMF 驱动 Gaussian 变形（而非引导图像生成），需要精确的几何对齐；Face-Mouth Hook 解决少数据下脸部-口腔运动不协调问题。

**3. Geometry Prior Regularizer**
- **功能**: 利用几何估计器提供深度和法线先验，正则化 3DGS 在少视角下的几何退化。
- **核心思路**: $\mathcal{L}_{Geo} = \lambda_D L_D(D, \check{D}) + \lambda_N \sum(1 - N_{i,j} \cdot \check{N}_{i,j})$
- **设计动机**: 少量训练数据的视角覆盖不足导致几何歧义，单目深度和法线估计提供额外约束。

### 损失函数 / 训练策略

- **预训练**: $\mathcal{L}_{pre}^i = \mathcal{L}_I(I^i, I_{GT}^i) + \lambda_C \sum_{j \neq i} \mathcal{L}_C$，$\lambda_C = 1$，5 个长视频，150K 迭代
- **适配**: $\mathcal{L}_{ada} = \mathcal{L}_I + \mathcal{L}_{Geo}$，先 warm-up（无 Motion Aligner），再全损失训练，10K 迭代
- 图像损失 $\mathcal{L}_I$ = L1 + D-SSIM，AdamW 优化器

## 实验关键数据

### 主实验（5 秒视频自重建）

| 方法 | 类型 | PSNR(A/F)↑ | LPIPS(A/F)↓ | LMD↓ | Sync-C↑ | 训练时间↓ | FPS↑ | 实时 |
|---|---|---|---|---|---|---|---|---|
| ER-NeRF | 从零 | 28.23/25.63 | 0.040/0.031 | 3.541 | 3.074 | 2h | 30.8 | ✓ |
| TalkingGaussian | 从零 | 28.32/26.01 | 0.041/0.028 | 3.588 | 3.556 | 31min | 114.2 | ✓ |
| MimicTalk | 适配 | 24.69/26.27 | 0.075/0.031 | 3.489 | 6.926 | 16min | 8.2 | ✗ |
| **InsTaG** | **适配** | **28.86/26.32** | **0.039/0.026** | **3.167** | 5.318 | 13min | **82.5** | ✓ |

### 不同数据量对比

| 方法 | 5s PSNR↑ | 10s PSNR↑ | 20s PSNR↑ |
|---|---|---|---|
| TalkingGaussian | 28.321 | 29.130 | 29.536 |
| MimicTalk | 24.69 | - | - |
| **InsTaG** | **28.86** | **29.45** | **29.82** |

### 关键发现

1. **5 秒视频即可超越从零训练的方法**: InsTaG 用 5 秒视频 + 13 分钟适配即可达到甚至超越 ER-NeRF/TalkingGaussian 等用完整视频从零训练数小时的结果。
2. **唇同步与渲染质量兼得**: LMD（唇部运动距离）达到最优 3.167，同时 PSNR 最高，证明个性化运动先验的有效性。
3. **实时推理**: 82.5 FPS，远优于 MimicTalk（8.2 FPS），与从零训练的 TalkingGaussian（114.2 FPS）接近但适配更快。
4. **跨身份/性别/语言泛化**: 实验验证了在不同身份、性别和语言场景下的有效性。

## 亮点与洞察

- Identity-Free Pre-training 巧妙地在 person-specific 架构上实现了多人预训练，核心洞察是用临时个性化场"吸收"身份信息
- Negative Contrast Loss 的截断设计精妙——只惩罚方向相同的个性化运动，对已经足够不同的不再推离
- Face-Mouth Hook 简洁有效地解决了 face-mouth 分解架构在少数据下的运动不协调问题
- 整体架构紧凑一致，预训练和适配无缝衔接，无需额外 2D-to-3D 模块

## 局限与展望

- 预训练仅用 5 个长视频，扩大预训练语料可能进一步提升通用性
- 适配阶段仍需约 13 分钟，距离"即时"还有距离
- 未涉及说话人头部以外身体部分的建模
- Sync-C（音频-视觉同步）指标低于 MimicTalk，可能因为 MimicTalk 使用了更大的骨干网络
- face-mouth 分解假设了固定的结构，对张嘴极大等极端表情可能受限

## 相关工作与启发

- TalkingGaussian 的 face-mouth 分解被本文继承并通过 Hook 机制强化
- 不同于 MimicTalk 用 LoRA 注入预训练语音模型，InsTaG 直接预训练运动场，更紧凑高效
- 运动先验解耦+对齐的思路可推广到其他 person-specific 3D 重建任务

## 评分

⭐⭐⭐⭐ — 技术路线清晰，Identity-Free Pre-training 的设计新颖优雅，实验充分展示了在极少数据下的卓越性能和效率。

<!-- RELATED:START -->

## 相关论文

- [Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)
- [Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/model_compression/syncity_training-free_generation_of_3d_worlds.md)
- [MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation](mutri_multi-view_tri-alignment_for_oct_to_octa_3d_image_translation.md)
- [Plug-and-Play Versatile Compressed Video Enhancement](plug-and-play_versatile_compressed_video_enhancement.md)

<!-- RELATED:END -->
