---
title: >-
  [论文解读] DGH: Dynamic Gaussian Hair
description: >-
  [NeurIPS 2025][3D视觉][动态头发建模] 提出 Dynamic Gaussian Hair (DGH)，一个数据驱动的 coarse-to-fine 框架，通过体素隐式变形模型学习头发动力学，并结合柱状 Gaussian 表示与曲率混合策略实现动态头发的逼真新视角渲染。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 动态头发建模
  - 3D Gaussian Splatting
  - 头发动力学
  - 新视角合成
  - 数字人
---

# DGH: Dynamic Gaussian Hair

**会议**: NeurIPS 2025  
**arXiv**: [2512.17094](https://arxiv.org/abs/2512.17094)  
**代码**: [项目页面](https://junyingw.github.io/paper/dgh)  
**领域**: 3d_vision  
**关键词**: 动态头发建模, 3D Gaussian Splatting, 头发动力学, 新视角合成, 数字人  

## 一句话总结

提出 Dynamic Gaussian Hair (DGH)，一个数据驱动的 coarse-to-fine 框架，通过体素隐式变形模型学习头发动力学，并结合柱状 Gaussian 表示与曲率混合策略实现动态头发的逼真新视角渲染。

## 研究背景与动机

**动态头发建模是数字人的核心难题**：头发的复杂运动、遮挡和光散射使得逼真的动态头发生成极具挑战，对 AR/VR 虚拟形象至关重要。

**物理仿真方法不可扩展**：传统基于物理的头发仿真（如 XPBD）需要针对每种发型手动调参（刚度、密度、阻尼等），且计算开销巨大，不适合 AR/VR 设备。

**现有 NeRF/3DGS 头像方法忽视头发动态**：3DGS 头部虚拟形象中头发仅做刚性变换，缺乏重力效应和二次运动效果，尤其长发不自然。

**头发渲染代价极高**：逼真的路径追踪渲染需要考虑半透明、遮挡和光散射，通常需要 GPU 渲染农场，实时引擎只能做近似。

**静态重建方法无法处理动态外观**：GaussianHair、Gaussian Haircut 等方法重建静态头发几何后需外接物理引擎动画化，流程割裂且需额外调参。

**缺乏真实动态头发数据**：没有带精确发丝追踪的真实动态头发采集数据，需要构建合成数据集推动研究。

## 方法详解

### 整体框架

DGH 分为两个阶段：**Stage I** 学习头发动力学变形（coarse-to-fine），将规范空间的静态头发在给定头部运动下变形为动态头发点云；**Stage II** 优化动态头发外观，将变形后的头发以柱状 3D Gaussian 表示并通过可微渲染生成逼真图像。推理时以递归方式逐帧预测变形并渲染。

### 关键设计 1：Coarse Stage — 姿态驱动体素隐式变形

- **功能**：将规范头发点云刚性变换到当前头部姿态后，通过 MLP 预测每个点的位移，得到初始变形头发。
- **核心思路**：将头发和上身 mesh 分别编码为 SDF 体素，通过 3D U-Net 编码器（ℰ_hair 和 ℰ_pose）提取特征并拼接。对每个头发点从体素中插值特征，与位置编码和头部姿态一起送入 MLP 预测位移 Δp。
- **设计动机**：体素隐式表示可以 mesh-free 地处理各类发型（长发、卷发、马尾等），无需显式物理仿真。SDF 约束（ℒ_SDF）惩罚穿入身体 mesh 的点，确保物理合理性。该阶段与时间无关，为后续动力学细化提供稳定初始化。

### 关键设计 2：Fine Stage — 时序 Flow 细化与跨帧注意力

- **功能**：在 coarse 变形基础上，预测相邻帧间的 3D flow 向量来建模惯性、振荡、阻尼等动态效应。
- **核心思路**：将前两帧变形后的头发体素 V^{t-2} 和 V^{t-1} 编码后做 cross-attention（Q=V^{t-2}, K/V=V^{t-1}），融合的时序特征与当前姿态和前一帧 flow 拼接，送入 MLP 预测当前帧 flow。
- **设计动机**：仅靠 coarse stage 的单帧姿态无法捕获头发的时间依赖运动（如甩头后的惯性摆动）。通过跨帧注意力机制利用前两帧信息推断运动趋势，实现时间一致性。这种数据驱动的方式替代了传统物理仿真的多步迭代求解。

### 关键设计 3：柱状 Gaussian 表示与 Strand-Guided 外观优化

- **功能**：将每条发丝表示为一系列柱状 Gaussian 基元，通过轻量 MLP 根据运动状态优化外观参数（颜色 SH 系数、尺度、不透明度）。
- **核心思路**：MLP 𝒟 以变形头发体素特征、位置/切线/视角方向的位置编码为输入，输出修正后的 SH 颜色、尺度和不透明度。利用发丝切线向量 t 建模头发各向异性散射。通过可微 tile rasterizer 渲染最终图像。
- **设计动机**：简单地将规范帧颜色沿时间传播会因动态遮挡导致外观缺失。加入切线信息后 MLP 可学习发丝 BSDF 的各向异性效应，实现运动中视角一致的逼真渲染。

### 关键设计 4：曲率自适应 Gaussian 混合

- **功能**：根据发丝局部曲率自适应混合相邻 Gaussian 的颜色和不透明度，消除高曲率区域的着色不连续。
- **核心思路**：计算每段发丝的切线 t_i、曲率 κ_i，归一化后作为混合权重 w_i，对相邻 Gaussian 的 SH 系数和 opacity 进行加权插值：SH_blended = SH_i·(1-w_i) + SH_{i-1}·w_i。
- **设计动机**：发丝以固定段数离散化后，高曲率处相邻切线差异大，导致着色跳变。增加段密度代价高昂，而曲率混合以零额外几何成本平滑过渡，在训练中联合优化。

## 损失函数与训练

- **Stage I Coarse**：ℒ_total = λ_p · ℒ_point + λ_SDF · ℒ_SDF（λ_p=1.0, λ_SDF=0.01）。ℒ_point 为预测与 GT 位移的 MSE，ℒ_SDF 惩罚穿入身体的点。
- **Stage I Fine**：ℒ_flow = MSE(ℱ_flow^t, ℱ_GT^t)，监督预测 flow 与 GT flow 的一致性。
- **Stage II**：ℒ = λ_rgb·L_rgb + λ_ssim·L_ssim + λ_lpips·L_lpips（λ_rgb=1.0, λ_ssim=0.1, λ_lpips=0.1）。
- 在单张 A100 上用 Adam（lr=1e-4）训练，每迭代采样 200K 个头发点。

## 实验关键数据

### 动态头发外观质量对比（Table 1）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| **DGH (Ours)** | **27.01** | **0.871** | **0.127** |
| Gaussian Haircut | 23.07 | 0.847 | 0.131 |
| 3DGS | 20.91 | 0.822 | 0.164 |

DGH 在所有 5 个发型对象上 PSNR 和 SSIM 均最高，相比 3DGS 提升约 6 dB PSNR。

### 头发变形精度对比（Table 2）

| 方法 | L2 Error↓ | Chamfer↓ |
|------|-----------|----------|
| **DGH (Ours)** | **0.0832** | **0.0266** |
| Rigged hair（刚性变换） | 0.1639 | 0.0424 |

DGH 将变形误差降低了约 49%（L2）和 37%（Chamfer），有效建模重力和碰撞。

### 消融实验（Table 3 & 4）

- **动力学消融**：去掉 SDF 约束 L2 增至 0.1269，去掉 fine stage 增至 0.0964，去掉 cross-attention 增至 0.0909，完整模型 0.0832。
- **外观消融**：无切线 & 无混合 PSNR=20.89，加切线 PSNR=25.08，完整模型 PSNR=28.12，切线和曲率混合各贡献显著。

## 亮点

- **全数据驱动替代物理仿真**：无需手动调参即可泛化到多种发型和运动，端到端可微分训练。
- **Coarse-to-fine 解耦设计**：姿态驱动初始化 + 时序 flow 细化，既保证稳定性又捕获高频动态。
- **曲率混合策略简洁有效**：零额外几何成本消除离散化伪影，PSNR 提升约 3 dB。
- **与 Gaussian 头像框架无缝集成**：可直接与身体 Gaussian 基元合并实现完整虚拟形象动画。
- **构建了高质量合成动态头发数据集**：含多种发型、10K 帧几何 + 12K 张多视角图像，计划开源。

## 局限与展望

- **仅在合成数据上验证**：缺乏真实捕获数据的评估，合成数据与真实头发之间存在 domain gap。
- **每种发型需独立训练模型**：尚未实现跨发型的泛化模型，扩展到大量发型成本高。
- **递归推理可能累积误差**：长序列中 flow 预测的误差可能逐帧放大。
- **固定照明条件**：训练在单一光照下进行，对复杂光照变化的鲁棒性未验证。
- **需要预先的发丝几何**：输入需要规范发型的点云/发丝模型，无法从图像直接推断。

## 与相关工作的对比

- **vs HVH / NeuWigs**：HVH 和 NeuWigs 也用神经体素表示动态头发，但它们聚焦于捕获重建，DGH 聚焦于从静态发型出发的动画化生成，且加入了 3DGS 外观优化。
- **vs GaussianHair / Gaussian Haircut**：这两者仅重建静态头发几何并需外接物理引擎动画化，DGH 端到端学习动态变形和时变外观，在动态渲染质量上大幅领先（PSNR +4~6 dB）。
- **vs 3DGS 头像方法**（GaussianAvatars, Rig3DGS 等）：这些方法中头发仅刚性跟随头部，DGH 补充了动态非刚性变形能力。
- **vs 物理仿真**（XPBD, Quaffure）：物理方法精度高但不可微、需调参且计算量大，DGH 以数据驱动方式实现可比效果并支持梯度优化。

## 评分

- 新颖性: ⭐⭐⭐⭐ — coarse-to-fine 动力学学习 + 曲率 Gaussian 混合是有价值的新贡献
- 实验充分度: ⭐⭐⭐⭐ — 多发型对比、消融完整，但仅限合成数据
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、公式规范，附录详实
- 价值: ⭐⭐⭐⭐ — 为动态头发建模提供了可扩展的数据驱动方案，对数字人领域有实际推动

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](../../ICCV2025/3d_vision/haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](../../ICCV2025/3d_vision/beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[CVPR 2026\] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](../../CVPR2026/3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

<!-- RELATED:END -->
