---
title: >-
  [论文解读] GenFusion: Closing the Loop between Reconstruction and Generation via Videos
description: >-
  [CVPR 2025][3D视觉][3D重建] 提出 GenFusion，通过重建驱动的视频扩散模型修复 3D 重建伪影并生成不可见区域内容，设计循环融合管线迭代地将生成结果加入训练集，实现稀疏视图下高质量 3D 场景重建和内容扩展。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D重建"
  - "视频扩散模型"
  - "稀疏视图重建"
  - "高斯泼溅"
  - "循环融合"
---

# GenFusion: Closing the Loop between Reconstruction and Generation via Videos

**会议**: CVPR 2025  
**arXiv**: [2503.21219](https://arxiv.org/abs/2503.21219)  
**代码**: [https://genfusion.sibowu.com](https://genfusion.sibowu.com)  
**领域**: 3D视觉  
**关键词**: 3D重建, 视频扩散模型, 稀疏视图重建, 高斯泼溅, 循环融合

## 一句话总结

提出 GenFusion，通过重建驱动的视频扩散模型修复 3D 重建伪影并生成不可见区域内容，设计循环融合管线迭代地将生成结果加入训练集，实现稀疏视图下高质量 3D 场景重建和内容扩展。

## 研究背景与动机

**领域现状**：3D 重建（NeRF/3DGS）和 3D 生成是两个高速发展的领域，但它们之间存在显著的"条件差距"——可扩展的 3D 场景重建通常需要密集视图输入，而 3D 生成通常只需要单张或无输入图像。前者在欠观察区域产生 floaters 和背景坍塌伪影，后者虽然能凭空生成内容但在场景级别的重建质量和视图覆盖率远不如密集捕获的重建。

**现有痛点**：稀疏视图 3D 重建面临本质上的欠约束问题，无穷多个 photo-consistent 解释都可能匹配输入图像。已有正则化方法（稀疏性、平滑性、单目深度引导）虽有改善但在远离训练视图的轨迹上仍力不从心。前馈重建方法（pixelSplat、MVSplat 等）通常限于 4-8 张输入图像就出现性能饱和。ReconFusion 等方法在视图插值上效果好，但在显著偏离输入视图的轨迹上仍然挣扎。

**核心矛盾**：3D 约束和生成先验之间的错位——重建需要密集视角覆盖来消除歧义，生成模型有丰富先验但缺乏 3D 一致性约束。如何让两者互补互利？

**本文目标** 探索 3D 重建和生成如何以可扩展的方式互相补充，放松对输入视图数量的限制，实现从稀疏视图甚至遮挡输入的高质量新视图合成和场景扩展。

**切入角度**：将重建和生成连接在视频空间——从伪影丰富的 RGB-D 渲染出发训练视频扩散模型学习修复能力，再通过循环迭代将修复结果加入 3D 重建的训练集。关键洞察是通过遮挡 3D 重建（masking 75% 像素）可以系统性地制造训练数据。

**核心 idea**：用遮挡重建模拟远视角伪影来训练视频扩散模型修复能力，再通过循环融合迭代地将生成视频加入重建训练集，实现重建和生成的正反馈循环。

## 方法详解

### 整体框架

分为两个阶段：预训练阶段在大规模数据集（DL3DV-10K）上微调视频扩散模型（基于 DynamiCrafter），学习从伪影 RGB-D 渲染恢复真实视频的能力；零样本泛化阶段在新场景上执行循环融合——重建→渲染伪影视频→扩散模型修复→修复结果加入训练集→继续重建，迭代进行。

### 关键设计

1. **遮挡 3D 重建训练数据生成**:

    - 功能：为视频扩散模型生成"伪影输入-真实输出"训练对
    - 核心思路：将输入视频帧划分为 4 个不重叠的 patch（左上、右上、左下、右下），每个场景只保留其中一个 patch（如左上或右下）用于 3D 重建（2DGS），mask 掉其余 75% 像素。然后沿原始相机轨迹渲染完整帧的 RGB-D 视频作为伪影输入，原始视频作为真实输出
    - 设计动机：直接均匀下采样帧只能模拟视图插值，中点分割只留下大部分内容不可见。遮挡重建巧妙地模拟了窄视场相机的效果，产生的伪影模式（floaters、空洞、黑色区域）与真实远视角渲染高度一致，同时保留了足够上下文促进外推

2. **重建驱动的视频扩散模型**:

    - 功能：从伪影 RGB-D 视频生成真实的、无伪影的 RGB-D 视频
    - 核心思路：基于 DynamiCrafter 微调，将 RGB VAE 替换为预训练的 RGB-D VAE（LDM3D）以引入几何信息而不改变扩散架构。伪影 RGB-D 视频经编码后与逐帧噪声拼接作为序列条件输入。同时用最近输入视图的 CLIP 特征提供全局场景信息。两阶段训练：coarse 阶段 320×512 分辨率 30K 步，fine 阶段 512×960 分辨率 34K 步
    - 设计动机：RGB-D VAE 替代 RGB VAE 虽然改变了预训练的 latent 空间，但实验证明 FID 反而更好（25.40 vs 26.16），因为深度信息为帧间一致性提供了几何约束。序列条件输入允许渲染视频的丰富视觉细节直接引导生成

3. **循环融合优化管线**:

    - 功能：迭代地将修复视频加入 3D 重建训练集，实现渐进式场景扩展和伪影修复
    - 核心思路：基于 2DGS 表示，每 K 次迭代执行一轮：采样新轨迹→渲染 RGB-D 视频→扩散模型修复→修复结果加入监督集。采样两类轨迹——相邻输入视图间插值和跨所有相机位姿的螺旋/球面路径。对于大面积不可见区域，通过不可靠深度检测（累积透明度 $T < \tau_T$ 或深度差 $|D-\hat{D}| > \tau_D$）自动添加新高斯点。生成损失权重使用正弦退火策略 $\lambda(k) = \sin(\frac{k-K_{start}}{K_{end}-K_{start}} \cdot \pi)$
    - 设计动机：一次性全局优化容易让生成先验压倒重建约束或反之，循环迭代让两者渐进地互相增强形成正反馈。正弦退火避免生成损失过早或过晚主导，新轨迹采样确保视角和角度的全面覆盖

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{gen}$，其中 $\mathcal{L}_{recon} = \lambda_{l_1} \mathcal{L}_{l1} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{mono} \mathcal{L}_{mono}$。$\mathcal{L}_{mono}$ 是尺度不变深度损失确保渲染深度与扩散模型预测深度一致。推理使用 DDIM 25 步采样，CFG 尺度 3.2。

## 实验关键数据

### 主实验（Mip-NeRF360 稀疏视图）

| 方法 | 3-view PSNR | 6-view PSNR | 9-view PSNR | 平均 PSNR |
|------|------------|------------|------------|----------|
| **GenFusion** | 15.29 | **17.16** | **18.36** | **16.93** |
| ReconFusion | **15.50** | 16.93 | 18.19 | 16.87 |
| 3DGS | 13.06 | 14.96 | 16.79 | 14.94 |
| 2DGS | 13.07 | 15.02 | 16.67 | 14.92 |
| FSGS | 14.17 | 16.12 | 17.94 | 16.08 |

### 消融实验（视频扩散模型设计）

| 配置 | FID ↓ | 说明 |
|------|------|------|
| RGB VAE, 512×320 | 26.16 | 基线 |
| RGB-D VAE, 512×320 | 25.40 | 替换为 RGB-D VAE，FID 反而更好 |
| RGB-D VAE, 48帧 | 29.35 | 增加帧数降低质量 |
| RGB-D VAE, 960×512 | **22.55** | 提高分辨率显著改善 |

### 关键发现
- GenFusion 首次证明高斯泼溅在稀疏视图设置下能达到与 SOTA NeRF 方法（ReconFusion）相当的性能，此前 GS 在稀疏视图下远落后于 NeRF
- RGB-D VAE 替代 RGB VAE 不仅不降低质量反而提升了 FID，说明深度信息对视频一致性有正面作用
- 分辨率从 512×320 提升到 960×512 带来了显著的 FID 改善（25.40→22.55）
- 在 DL3DV 和 TnT 数据集的远视角渲染上，GenFusion 相比 3DGS/2DGS/FSGS 有显著优势，验证了循环融合管线的有效性

## 亮点与洞察
- **遮挡重建生成训练数据的思路极其巧妙**：简单地 mask 75% 像素就能模拟远视角伪影，无需任何额外标注就能从大规模视频数据集生成海量训练对。这个思路可以推广到任何需要退化-真实对训练数据的场景
- **循环融合形成正反馈**：重建和生成不是二选一而是理论上能互相增强的正反馈环，这一哲学思想对 3D vision 领域有启发意义
- **用视频作为重建和生成之间的桥梁**：不直接在 3D 空间连接两者，而是通过视频渲染这一自然中间表示传递信息，保持了方法的模块性和简洁性

## 局限与展望
- 视频扩散模型的推理速度较慢，循环融合需要多轮扩散采样，总体时间成本较高
- 16 帧的视频长度限制了单次生成的视角覆盖范围
- 当场景完全不可见时（如室内场景的反面），生成内容的 3D 一致性无法保证
- 目前只在物体和室内/户外场景上验证，未涉及动态场景
- 正弦退火策略中 $K_{start}$ 和 $K_{end}$ 的选择可能需要针对不同场景调整

## 相关工作与启发
- **vs ReconFusion**：同样利用生成先验引导 3D 重建，但 ReconFusion 基于 NeRF+图像扩散使用 SDS loss，GenFusion 基于 GS+视频扩散使用直接光度损失更高效稳定。GenFusion 还支持场景扩展而非仅视图插值
- **vs ViewCrafter**：ViewCrafter 也利用 3D 信息（点云）进行视频生成，但不涉及循环融合和渐进式场景扩展
- **vs 前馈重建方法（pixelSplat/MVSplat/DepthSplat）**：前馈方法受限于少量输入视图（<10），GenFusion 的循环融合理论上可扩展到任意数量视图

## 评分
- 新颖性: ⭐⭐⭐⭐ 遮挡重建+循环融合的思路新颖实用，但整体方法由多个已有组件拼接
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种设置（3/6/9 views、masked input），但缺少一些常见基线和运行时间对比
- 写作质量: ⭐⭐⭐⭐ 叙事逻辑清晰，方法描述到位，部分细节略有冗余
- 价值: ⭐⭐⭐⭐ 首次证明 GS 可在稀疏视图下匹敌 NeRF SOTA，循环融合思路有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](slam3r_real-time_dense_scene_reconstruction_from_monocular_rgb_videos.md)
- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] SemAlign3D: Semantic Correspondence Between RGB-Images Through Aligning 3D Object-Class Representations](semalign3d_semantic_correspondence_between_rgb-images_through_aligning_3d_object.md)

</div>

<!-- RELATED:END -->
