---
title: >-
  [论文解读] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos
description: >-
  [ICCV 2025][3D视觉][视频物体插入] 提出一种混合管线将 3D 手镯逼真插入视频：利用 3D 高斯泼溅（3DGS）保证时序一致性，用 2D 扩散模型增强光照真实感，并通过光照驱动（Shading-Driven）管线分离 albedo/shading/反射残差分别优化，在用户研究中以 81.7% 的真实感偏好率大幅超越现有方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 视频物体插入
  - 3D高斯泼溅
  - 扩散模型
  - 光照增强
  - 时序一致性
  - 虚拟试戴
---

# From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos

**会议**: ICCV 2025  
**arXiv**: [2507.20331](https://arxiv.org/abs/2507.20331)  
**代码**: https://cjeen.github.io/BraceletPaper/  
**领域**: 3d_vision / 视频物体插入  
**关键词**: 视频物体插入, 3D高斯泼溅, 扩散模型, 光照增强, 时序一致性, 虚拟试戴

## 一句话总结

提出一种混合管线将 3D 手镯逼真插入视频：利用 3D 高斯泼溅（3DGS）保证时序一致性，用 2D 扩散模型增强光照真实感，并通过光照驱动（Shading-Driven）管线分离 albedo/shading/反射残差分别优化，在用户研究中以 81.7% 的真实感偏好率大幅超越现有方法。

## 研究背景与动机

**视频物体插入的核心矛盾是什么？** 将 3D 物体插入视频需要同时满足两个相互矛盾的要求：

**时序一致性**：插入物体在帧间不应闪烁或突变，需要正确处理运动、视角变化和遮挡

**光照真实感**：插入物体必须与场景的光照条件自然融合，包括阴影、高光和反射

**现有方法为何无法兼顾？**

- **2D 扩散模型方案**（AnyV2V、ReVideo、ConsistI2V）：利用 Anydoor 等方法在首帧编辑后传播，可以生成高视觉质量的结果，但 2D 参考缺乏生成一致新视角的信息，导致物体移动或视角变化时产生闪烁和身份不一致
- **传统 3D 渲染方案**：AR 管线可以天然维持时序一致性，但难以精确建模复杂的环境光照交互，导致渲染结果不真实（缺少阴影、高光不自然、与背景不融合）

**为什么选择手镯作为研究场景？** 手镯是一个理想的研究对象——包含复杂动态高光（金属/宝石反射），手腕运动节奏快且多变，遮挡关系频繁变化（手指遮住手镯等），对时序一致性和光照要求都极高。

## 方法详解

### 整体框架

管线由三个模块串联：

1. **运动和遮挡感知计算**（Motion & Occlusion-Aware Calculation）
2. **真实感增强**（Realism Enhancement）
3. **时序平滑**（Temporal Smoothing）

### 模块 1：运动与遮挡感知插入

遵循传统 AR 管线：

- **交互式 GUI**：用户通过轻量 GUI 放置手镯、调整位姿（位置/朝向/缩放），选择跟踪锚点
- **2D 关键点跟踪**：使用 CoTracker 在皮肤上跟踪 2D 关键点，估计 3D 位姿跨帧变化
- **遮挡处理**：通过比较手镯深度和 UniDepth 估计的场景深度生成逐帧遮挡图
- **初始预览**：将 3DGS 渲染的手镯叠加到视频帧上

### 模块 2：光照驱动真实感增强

核心创新——不直接在 RGB 空间增强，而是先在 shading 空间操作：

**图像分解**：将线性 RGB 图像分解为三个固有成分：

$$\tilde{\mathbf{I}}_t = \mathbf{A}_t \cdot \mathbf{S}_t + \mathbf{R}_t$$

其中 $\mathbf{A}_t$ 为 albedo（反射属性），$\mathbf{S}_t$ 为 shading（光照效果），$\mathbf{R}_t$ 为残差（高光等）。

**区域分割**：将场景分为三个区域——手镯区域 $\mathbf{M}$、背景区域 $\mathbf{M}_{bg}$（扩展 bounding box 外部）、周围区域 $\mathbf{M}_{surr}$（手镯附近的阴影投射区）。增强模型不改变背景 shading。

**两阶段颜色管线**：

1. **手镯重光照网络 $f_{br}$**：输入手镯 shading、背景 shading 和法线图，输出重光照后的手镯 shading
2. **阴影生成网络 $f_{sh}$**：输入重光照 shading、背景和周围 shading，生成包含自然阴影的完整 shading
3. **sRGB 增强网络 $f_{sRGB}$**：在 sRGB 空间精炼细节

**网络架构**：基于 Stable Diffusion 微调，采用单步扩散公式（zero latent → clean output）保持对输入的高保真度。增强解码器通过额外编码器 $\mathcal{E}_f$ 提取多尺度特征注入 VAE 解码器，防止解码引入偏差。

**训练数据生成**：收集约 13,000 张佩戴手镯的图像，通过数据增强构建训练对：
- 手镯重光照：合成多光源 shading 图 + 灰度混合 + 亮度翻转
- 阴影生成：高斯模糊 + 随机高斯块添加

### 模块 3：时序平滑

**手镯平滑**：逐帧优化 3DGS 颜色属性，使用滑动窗口和高斯权重确保相邻帧过渡平滑：

$$\mathcal{G}_t^* = \arg\min_{\mathcal{G}} \sum_{k=t-W/2}^{t+W/2} w(k-t) \cdot \|\mathcal{R}(\mathbf{K}, \mathbf{P}_k, \mathcal{G}) - \mathbf{I}_t^{\text{refined}}\|^2$$

仅优化颜色属性以保持几何结构和身份不变。

**阴影平滑**：均匀选关键帧，使用 EbSynth 插值中间帧。最终用手镯掩码混合两者。

## 实验关键数据

### 主实验：用户研究

| 方法 | 真实感↑ | 一致性↑ | 保真度↑ |
|------|---------|--------|---------|
| AnyV2V | 8.7% | 8.7% | 6.7% |
| ReVideo | 4.0% | 3.6% | 4.4% |
| ConsistI2V | 5.6% | 3.6% | 4.8% |
| **Ours** | **81.7%** | **84.1%** | **84.1%** |

36 名参与者评估，方法在三个维度上获得压倒性偏好（81-84%）。

### 定量对比

| 方法 | DeQA Score↑ | Temporal Consistency↑ | CLIP Score↑ |
|------|-------------|----------------------|-------------|
| AnyV2V | 3.41 | 0.971 | 0.659 |
| ReVideo | 3.17 | 0.977 | 0.661 |
| ConsistI2V | 2.44 | 0.976 | 0.638 |
| **Ours** | **3.66** | **0.984** | **0.662** |

全部三个指标最优，时序一致性 0.984 特别突出（得益于 3DGS 中间表示）。

### 消融实验

**Shading-Driven 增强**：与直接在 sRGB 空间预测增强的基线相比，光照驱动管线生成的手镯细节更丰富、光照交互更自然，基线结果显得扁平且融入感差。

**时序平滑**：
- 无平滑：明显闪烁和运动不一致
- 仅 EbSynth：过渡更平滑但细节损失
- **3DGS 平滑**（本方法）：兼顾平滑过渡和细节保留

## 亮点与洞察

1. **3D+2D 混合范式的首次成功**：3DGS 负责几何一致性，2D 扩散负责光照真实感，二者优势互补，避免各自缺陷
2. **Shading-Driven 分解的精妙设计**：在 shading 空间操作 avoids 颜色/纹理信息的意外修改，阴影作为独立模块处理增加了灵活性
3. **单步扩散的合理选择**：多步扩散会引入随机性破坏保真度，单步扩散既利用了 SD 先验又保证了确定性输出
4. **交互式 GUI 的工程完善度**：用户可直接放置、调整手镯，选择跟踪锚点，降低了实际使用门槛

## 局限性

- 对低质量背景视频（运动模糊、严重伪影）的处理能力有限
- 仅在手镯场景验证，尚未推广到一般物体插入
- 依赖 CoTracker 的 2D 跟踪质量，快速手部运动时可能失效
- 训练数据量（13K 张）相对有限，可能影响对极端光照条件的泛化

## 相关工作与启发

- **AnyDoor**：零样本物体级图像定制，本文用其编辑首帧作为基线的输入
- **3D Gaussian Splatting**：作为高效 3D 表示用于保持跨视角一致性
- **图像和谐化**（Intrinsic Harmonization）：在 shading 域做和谐化的思路启发了本文，但现有方法无法处理投射阴影
- **启发**：3DGS 作为中间表示实现时序平滑的思路非常优雅，可推广到任何需要在视频中保持物体外观一致性的场景

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 3D+2D 混合管线 + Shading-Driven 增强的组合设计新颖实用
- 实验充分度：⭐⭐⭐⭐ — 用户研究 + 定量指标 + 消融完整，但数据集较小（56个案例）
- 实用价值：⭐⭐⭐⭐⭐ — 直接可用于珠宝虚拟试戴/AR 场景，交互 GUI 提升体验
- 写作质量：⭐⭐⭐⭐ — 管线叙述清晰，图示丰富

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Towards Realistic Example-Based Modeling via 3D Gaussian Stitching](../../CVPR2025/3d_vision/towards_realistic_example-based_modeling_via_3d_gaussian_stitching.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos](trace_learning_3d_gaussian_physical_dynamics_from_multi-view_videos.md)
- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](../../CVPR2025/3d_vision/vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

<!-- RELATED:END -->
