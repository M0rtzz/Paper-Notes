---
title: >-
  [论文解读] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation
description: >-
  [CVPR 2025][视频生成][运动控制] 提出 Tora，首个面向轨迹控制的 Diffusion Transformer（DiT）视频生成框架，通过轨迹提取器（3D VAE 编码运动轨迹为时空 patch）和运动引导融合器（自适应归一化注入 DiT 块），实现了可扩展的、支持多分辨率/多时长/多宽高比的轨迹控制视频生成，在 128 帧测试中轨迹精度比 UNet 方法高 3-5 倍。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "运动控制"
  - "Transformer"
  - "轨迹引导"
  - "光流编码"
---

# Tora: Trajectory-Oriented Diffusion Transformer for Video Generation

**会议**: CVPR 2025  
**arXiv**: [2407.21705](https://arxiv.org/abs/2407.21705)  
**代码**: [https://github.com/alibaba/Tora](https://github.com/alibaba/Tora)  
**领域**: 图像生成 / 视频生成  
**关键词**: 视频生成, 运动控制, Diffusion Transformer, 轨迹引导, 光流编码

## 一句话总结
提出 Tora，首个面向轨迹控制的 Diffusion Transformer（DiT）视频生成框架，通过轨迹提取器（3D VAE 编码运动轨迹为时空 patch）和运动引导融合器（自适应归一化注入 DiT 块），实现了可扩展的、支持多分辨率/多时长/多宽高比的轨迹控制视频生成，在 128 帧测试中轨迹精度比 UNet 方法高 3-5 倍。

## 研究背景与动机

1. **领域现状**：扩散模型在视频生成领域取得了显著进展。以 Sora 为代表的 Diffusion Transformer（DiT）架构展示了生成 10-60 秒高质量视频的能力，远超基于 UNet 的方法。视频生成的关键挑战之一是运动控制——让生成的视频中的物体按照指定方式运动。

2. **现有痛点**：（a）之前的运动控制方法（DragNUWA、MotionCtrl 等）都基于 UNet 架构，受限于 16 帧/固定分辨率，在较长序列上出现运动模糊、物体变形和不自然的平移漂移；（b）DiT 架构虽然可扩展性强，但尚未有工作将轨迹控制适配到 DiT 框架中；（c）DiT 使用 Video Autoencoder 和 patchification 将视频转为 patch 序列，帧间位移的运动表示方式与 patch 空间不兼容。

3. **核心矛盾**：如何将用户指定的轨迹条件与 DiT 的可扩展架构无缝对齐？UNet 方法的注入策略（如简单拼接或线性投影）无法兼容 DiT 的 patch 化潜空间和交替的时空注意力机制。

4. **本文目标**：设计一套与 DiT 扩展性兼容的轨迹编码和融合机制，实现对视频内容运动的精确控制，同时支持可变时长、分辨率和宽高比。

5. **切入角度**：将轨迹位移转换为 RGB 域的光流可视化 → 用3D VAE 压缩到与视频 patch 相同的潜空间 → 通过堆叠卷积层提取多层级运动条件 → 用自适应归一化层注入到 DiT 块中。

6. **核心 idea**：用 3D Motion VAE 将轨迹编码为与视频 patch 同一潜空间的运动表示，再通过自适应归一化将多层级运动条件逐层融入 DiT，保持 DiT 的扩展性的同时实现精确轨迹跟踪。

## 方法详解

### 整体框架
Tora 基于 OpenSora（开源 Sora 实现）构建，包含三个核心组件：（1）Spatial-Temporal DiT（ST-DiT）——交替使用空间自注意力和时序自注意力的基础生成模型；（2）Trajectory Extractor（TE）——将用户轨迹编码为分层时空运动 patch；（3）Motion-guidance Fuser（MGF）——将运动 patch 注入 DiT 块。输入为文本描述 + 可选图像 + 轨迹坐标序列，输出为跟随指定轨迹的生成视频（最长 204 帧，最高 720p）。

### 关键设计

1. **轨迹提取器（Trajectory Extractor, TE）**:
    - 功能：将任意用户轨迹编码为与视频 patch 同一潜空间的分层运动条件
    - 核心思路：分三步处理（a）将轨迹坐标转为帧间位移 $(u, v)$，生成轨迹图 $g \in \mathbb{R}^{L \times H \times W \times 2}$，并施加高斯滤波减轻稀疏性；（b）将位移图转换为 RGB 色彩空间得到 $g_{vis}$，然后通过自训练的 3D VAE（基于 MAGVIT-v2 架构简化，空间 8x + 时序 4x 压缩）编码为运动潜表示 $g_m$；（c）对 $g_m$ 进行 patchification 后通过堆叠残差卷积层提取多级运动特征 $f_i$，每级对应一个 DiT 块
    - 设计动机：DiT 中视频通过 autoencoder + patchification 转为 patch 序列，每个 patch 跨越多帧。直接使用帧间位移与 patch 空间不匹配。3D VAE 将轨迹压缩到与视频 patch 同一潜空间，信息密度和维度完美对齐

2. **运动引导融合器（Motion-guidance Fuser, MGF）**:
    - 功能：将多层级运动条件注入到对应的 DiT 块中
    - 核心思路：作者探索了三种融合架构——额外通道拼接（MLP）、交叉注意力、自适应归一化（Adaptive Norm）。最终选择 Adaptive Norm：将运动特征 $f_i$ 通过两个零初始化卷积层分别生成 scale $\gamma_i$ 和 shift $\beta_i$，然后对 DiT 块的隐状态做线性调制：$h_i = \gamma_i \cdot h_{i-1} + \beta_i + h_{i-1}$
    - 设计动机：Adaptive Norm 不需要严格对齐（交叉注意力的难题），能动态适应不同条件，且零初始化保证训练初期不破坏基础模型的生成能力。实验证明它在 FVD、CLIPSIM、TrajError 三个指标上全面优于通道拼接和交叉注意力

3. **两阶段训练策略**:
    - 功能：从密集光流逐步过渡到稀疏轨迹，提升运动学习效果
    - 核心思路：第一阶段用密集光流（由 RAFT 提取）作为轨迹训练 2 epoch，提供丰富的运动信息帮助模型理解运动模式；第二阶段切换到稀疏轨迹（随机选 1-N 条物体轨迹 + 高斯平滑），微调 1 epoch 使模型适应用户友好的交互方式
    - 设计动机：直接用稀疏轨迹训练信息量不足，模型难以学习；直接用密集光流则无法适配推理时的稀疏输入。两阶段策略兼顾了信息丰富度和使用灵活性

### 损失函数 / 训练策略
- 使用标准的扩散模型噪声预测损失 $\ell_\epsilon = \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2$
- 采用 Adapter-like 策略：仅训练时序块 + TE + MGF，冻结空间块，保留基础模型的生成知识
- 3D Motion VAE 预先在光流数据集上训练后冻结
- 数据处理：场景检测切分 → 无效视频去除 → 美学/光流评分过滤 → 相机运动过滤 → PLLaVA 生成标注

## 实验关键数据

### 主实验

不同帧数下的视频生成对比:

| 方法 | FVD↓ (128帧) | CLIPSIM↑ (128帧) | TrajError↓ (128帧) |
|------|-------------|-------------------|---------------------|
| VideoComposer | 856 | 0.2236 | 58.76 |
| DragNUWA | 784 | 0.2305 | 41.25 |
| MotionCtrl | 731 | 0.2331 | 38.39 |
| OpenSora (无轨迹) | 533 | 0.2411 | 373.17 |
| OpenSora-DragNUWA* | 565 | 0.2393 | 21.75 |
| **Tora** | **494** | **0.2418** | **11.72** |

Tora 的轨迹误差仅为 UNet 方法的 1/3 到 1/5，FVD 也比 UNet 方法好 30-40%。同时 Tora 还优于基础 OpenSora 的视觉质量（FVD 494 vs 533），说明轨迹控制反过来也改善了生成稳定性。

### 消融实验

轨迹压缩方式:

| 配置 | FVD↓ | TrajError↓ |
|------|------|------------|
| 关键帧采样 | 581 | 27.61 |
| 平均池化 | 558 | 20.97 |
| **3D VAE（本文）** | **513** | **14.25** |

融合架构:

| 配置 | FVD↓ | TrajError↓ |
|------|------|------------|
| 额外通道拼接 | 542 | 21.07 |
| 交叉注意力 | 526 | 18.36 |
| **自适应归一化** | **513** | **14.25** |

训练策略:

| 配置 | FVD↓ | TrajError↓ |
|------|------|------------|
| 仅密集光流 | 601 | 39.34 |
| 仅稀疏轨迹 | 556 | 24.73 |
| **两阶段混合** | **513** | **14.25** |

### 关键发现
- 3D VAE 压缩轨迹到视频潜空间是关键，比简单的帧采样和平均池化分别好 48% 和 32%（TrajError）
- 自适应归一化在效果和效率上全面优于交叉注意力和通道拼接
- MGF 放在 Temporal DiT 块中比放在 Spatial 块中效果好很多（TrajError 23.39→14.25）
- 两阶段训练策略必不可少，单独使用密集/稀疏轨迹效果都远逊于混合策略
- Tora 可无缝迁移到 CogVideoX（2B/5B）上，验证了模块的通用性和 scaling 能力
- 加入轨迹控制后不仅提升了运动精度，还改善了基础模型的视觉质量（抑制时序伪影）

## 亮点与洞察
- **3D Motion VAE 对齐潜空间**：将轨迹编码到与视频 patch 相同的潜空间是本文最核心的创新。这种"条件信号与主信号共享潜空间"的思路可推广到其他条件控制任务（如音频到视频等）
- **Adaptive Norm 的零初始化技巧**：零初始化确保训练初期运动条件注入为零，不破坏预训练的基础生成能力，之后逐渐学习运动引导。这种"渐进式激活"策略非常优雅
- **运动控制改善视觉质量**：出乎意料的是，加入轨迹控制后 FVD 反而优于无控制的 OpenSora。这说明合理的运动先验可以帮助模型克服长视频中的时序不一致问题
- **User Study 竞争力**：与闭源的 Vidu 基本持平，仅略逊于 Kling

## 局限与展望
- 轨迹控制精度与视频时长成正比增加（更长视频误差更大），长程一致性仍有提升空间
- 目前仅支持物体轨迹控制，相机运动控制需要额外设计
- 训练数据处理流程复杂（场景检测→美学过滤→运动分割→相机检测），数据工程成本不低
- 对于快速运动或遮挡场景，光流估计本身的误差会传导到轨迹编码中
- 未来可结合 3D 感知（深度、场景结构）实现物理更合理的运动控制

## 相关工作与启发
- **vs DragNUWA**: DragNUWA 是首个在 VDM 中引入轨迹控制的工作，但基于 UNet 仅支持 16 帧低分辨率。Tora 将轨迹控制扩展到 DiT 架构，支持 204 帧/720p，轨迹精度提升 3-5 倍
- **vs MotionCtrl**: MotionCtrl 分别控制相机和物体运动、灵活性好，但同样受限于 UNet 的扩展性。Tora 在 128 帧测试中 TrajError 仅 11.72 vs MotionCtrl 的 38.39
- **vs OpenSora**: OpenSora 视觉质量高但无运动控制能力。Tora 在保持/提升视觉质量的同时增加了精确的轨迹控制
- **vs CogVideoX**: Tora 的运动模块可无缝迁移到 CogVideoX 上，说明方法具有良好的通用性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个在 DiT 架构上实现轨迹控制的工作，3D VAE 潜空间对齐思路巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 16/64/128 帧多配置对比、三类消融（压缩/融合/训练）、多模型 scaling、user study
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法图直观，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 首开 DiT 轨迹控制先河，Alibaba 开源代码，对视频生成领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)

</div>

<!-- RELATED:END -->
