---
title: >-
  [论文解读] I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM
description: >-
  [ECCV 2024][3D视觉][SLAM] 提出I²-SLAM，将物理成像过程（运动模糊建模+色调映射）集成到视觉SLAM系统中，通过HDR辐射场地图、多虚拟相机运动模糊模拟和可微分色调映射的联合优化，从手持随意拍摄的退化视频中重建出清晰的HDR 3D地图和更精确的相机轨迹。 领域现状：视觉SLAM将环境映射为3D表示…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "SLAM"
  - "运动模糊"
  - "HDR重建"
  - "3D高斯溅射"
  - "神经辐射场"
---

# I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM

**会议**: ECCV 2024  
**arXiv**: [2407.11347](https://arxiv.org/abs/2407.11347)  
**代码**: [https://3d.snu.ac.kr/publications/I2SLAM](https://3d.snu.ac.kr/publications/I2SLAM) (项目页)  
**领域**: 3D视觉  
**关键词**: SLAM, 运动模糊, HDR重建, 3D高斯溅射, 神经辐射场

## 一句话总结
提出I²-SLAM，将物理成像过程（运动模糊建模+色调映射）集成到视觉SLAM系统中，通过HDR辐射场地图、多虚拟相机运动模糊模拟和可微分色调映射的联合优化，从手持随意拍摄的退化视频中重建出清晰的HDR 3D地图和更精确的相机轨迹。

## 研究背景与动机

**领域现状**：视觉SLAM将环境映射为3D表示，广泛应用于VR/AR、机器人导航和碰撞处理。近年来基于NeRF和3D高斯溅射（3DGS）的dense visual SLAM方法能够合成逼真图像，代表方法包括iMAP、NICE-SLAM、Co-SLAM、Point-SLAM和SplaTAM等。这些方法在理想输入下能取得不错的效果。

**现有痛点**：尽管已有很多构建视觉表示的SLAM方法，绝大多数在真实场景中无法保持性能。手持随意拍摄的视频——SLAM系统最常见的输入——面临两个核心挑战：（1）相机运动导致的**运动模糊**使得图像退化；（2）自动曝光和白平衡导致帧间**外观不一致**。这些退化不仅降低地图质量，还累积性地损害位姿估计精度，是整个SLAM系统的关键瓶颈。

**核心矛盾**：现有SLAM系统假设输入图像是"理想"的单一位姿瞬时曝光，而真实图像是在一段曝光时间内沿相机轨迹积分的结果，且经过了非线性的ISP流程。这个建模假设和物理现实之间的gap是性能退化的根源。2D图像去模糊（如NAFNet）虽然可以预处理输入，但去模糊后的图像在不同视角间缺乏3D一致性，反而可能引入新问题。

**本文目标** 在SLAM框架内直接对物理成像过程建模，使系统能从退化输入中恢复清晰的HDR地图和准确轨迹。具体分解为：（1）如何建模运动模糊过程以恢复清晰表示？（2）如何处理帧间曝光和白平衡变化？（3）如何将这些模块无缝集成到已有SLAM流程中？

**切入角度**：与"先修复再建图"不同，作者选择"在建图中对退化建模"——将图像形成过程的逆过程嵌入SLAM的分析-合成优化循环中。运动模糊图像被视为曝光时间内沿轨迹多个位姿的图像积分，外观变化被分解为白平衡、曝光时间和相机响应函数三个可微分的显式变量。

**核心 idea**：通过反转物理成像过程（运动模糊积分+色调映射），将HDR线性辐射场作为SLAM的核心表示，联合优化场景、位姿和成像参数。

## 方法详解

### 整体框架
I²-SLAM是一个通用模块，可以挂载到任何基于图像输入的dense visual SLAM pipeline上。输入是可能包含运动模糊和外观变化的RGB/RGBD视频流。地图表示为HDR辐射场（线性颜色空间），支持NeRF和3DGS两种表示。系统对每帧维护：起始和结束位姿 $\mathbf{T}(t_s), \mathbf{T}(t_e)$（共同构成运动模糊内核）、曝光时间 $\Delta t$、白平衡参数 WB、相机响应函数 CRF。通过最小化合成的退化图像与实际输入之间的差异来联合优化所有变量。

### 关键设计

1. **HDR辐射场 + 运动模糊模拟**:

    - 功能：将地图表示为线性HDR辐射值，并通过多虚拟相机积分模拟运动模糊
    - 核心思路：地图的颜色输出 $\mathbf{c}(\mathbf{T}, \mathbf{p})$ 为线性HDR值。运动模糊图像被建模为曝光时间 $[t_s, t_e]$ 内沿相机轨迹的积分：$C_{\text{HDR}}(\mathbf{p}) = \Delta t \cdot \frac{1}{N_{\text{cam}}} \sum_{j=1}^{N_{\text{cam}}} \mathbf{c}(\mathbf{T}(t_j), \mathbf{p})$，其中在 $t_s$ 和 $t_e$ 之间线性插值出 $N_{\text{cam}}=5$ 个虚拟相机位姿（平移用线性插值，旋转用球面插值Slerp）。优化变量包括起始位姿、结束位姿和曝光时间
    - 设计动机：HDR线性颜色空间使运动模糊的"光强叠加"在物理上正确——在非线性sRGB空间中做加权平均是物理上错误的。同时，将模糊kernel显式建模为位姿间插值而非传统卷积核，天然兼容SLAM的位姿优化框架

2. **可微分色调映射模块**:

    - 功能：建模从HDR辐射场到LDR相机像素值的完整映射过程，处理帧间外观变化
    - 核心思路：色调映射 $\Psi_i$ 由三个可学习组件构成：逐通道白平衡 $\text{WB}_i$（3个参数的逐通道乘法）、相机响应函数 $\text{CRF}_i$（每个颜色通道用256维均匀采样网格参数化的单调递增函数）、以及动态范围裁剪。CRF通过添加leaky clipping函数（$\alpha=0.01$）来保证梯度可以通过饱和区域回传。最终 $C_{\text{LDR}} = \text{CRF}(\text{WB}(\Delta t \cdot \mathbf{c}))$。所有参数都是逐帧学习的
    - 设计动机：真实场景中ISP会自动调整曝光、白平衡和色调曲线，导致帧间颜色不一致。如SplaTAM的3DGS颜色是固定的，无法适应不同帧的亮度变化，产生严重的拼接伪影。通过显式建模ISP流程，让地图表示保持物理意义上的一致HDR值

3. **轨迹正则化与初始化策略**:

    - 功能：利用SLAM已有的全局轨迹信息约束运动模糊方向和幅度，稳定优化
    - 核心思路：设计轨迹损失 $\mathcal{L}_{\text{traj}}$ 包含两个约束：（1）曝光时间内的相机运动方向应与全局轨迹对齐——将 $\mathbf{t}(t_e^{i-1})$ 约束到前一帧和当前帧中心位姿的线性插值上；（2）运动幅度应与曝光时间和时间速度成正比。引入全局尺度参数 $a$（与视频帧率相关）并联合优化。初始化时利用前两帧的位姿外推新帧的起始和结束位姿，并设置小的初始分离距离
    - 设计动机：单纯依靠图像渲染损失优化起始/结束位姿容易陷入局部最优。SLAM本身已经维护了全局轨迹估计，利用这个免费的先验来正则化模糊运动方向是SLAM特有的优势——这是静态3D重建方法所不具备的

### 损失函数 / 训练策略
总损失为三部分加权和：$\mathcal{L} = \lambda_{\text{img}} \mathcal{L}_{\text{img}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{traj}} \mathcal{L}_{\text{traj}}$。图像损失使用L1距离比较合成LDR图像和实际输入。深度损失为在曝光期间内选择深度误差最小的位姿来计算深度渲染损失（假设深度传感器在某个瞬间采集）。RGB-SLAM使用DROID-SLAM的位姿作为初始值；RGBD-SLAM部署在SplaTAM上。

## 实验关键数据

### 主实验（RGB-SLAM，基于NeRF-SLAM†）

| 数据集 | 指标 | I²-SLAM | NeRF-SLAM† | 提升 |
|--------|------|---------|------------|------|
| 合成数据集（平均） | PSNR | 28.89 | 26.70 | +2.19dB |
| 合成数据集（平均） | SSIM | 0.887 | 0.842 | +0.045 |
| TUM-RGBD fr3/office | ATE-RMSE | 1.95cm | 7.13cm | 3.6倍 |
| 合成数据集 SP | ATE-RMSE | 1.50cm | 3.97cm | 2.6倍 |

### 消融实验（ScanNet RGBD）

| 配置 | ATE-RMSE | PSNR | SSIM | LPIPS |
|------|----------|------|------|-------|
| Full (运动模糊+HDR+轨迹正则) | **2.56** | **25.62** | **0.801** | **0.195** |
| w/o 运动模糊建模 | 2.66 | 24.80 | 0.769 | 0.203 |
| w/o HDR地图 | 2.60 | 22.39 | 0.756 | 0.226 |
| w/o HDR + w/o 运动模糊 | 2.63 | 22.36 | 0.755 | 0.228 |
| 全部去掉（baseline） | 2.71 | 23.05 | 0.793 | 0.235 |

### 关键发现
- HDR地图对渲染质量的提升最大——从22.36到25.62的PSNR提升中，HDR贡献了约3dB
- 运动模糊建模主要提升跟踪精度——ATE-RMSE从2.66降到2.56，同时也提升渲染质量
- 轨迹正则化对稳定多相机优化至关重要，影响跟踪和建图两方面
- 在外观变化剧烈的场景（ScanNet 0785-00、合成数据SP）中，I²-SLAM的优势最为显著
- 使用20%迭代次数的I²-SLAM-S在相似运行时间下仍优于全迭代的SplaTAM
- 2D去模糊预处理（NAFNet + NeRF-SLAM）效果有限，因为缺乏多视角一致性

## 亮点与洞察
- **物理正确的成像逆过程**：不是"事后修补"退化，而是在优化目标中直接建模退化产生的原因。这让运动模糊不再是噪声而是约束——模糊图像实际上提供了轨迹信息。这种"analysis-by-synthesis"的理念非常优雅
- **HDR线性颜色空间的关键作用**：线性HDR不仅让运动模糊的加性叠加物理正确，还简化了外观变化的建模——所有非线性因素都集中在可微分的色调映射模块中。这个设计把复杂问题分解得非常干净
- **通用模块化设计**：I²-SLAM作为一个可挂载模块，已在NeRF-based和3DGS-based两种SLAM上验证，展现了对不同场景表示的兼容性。这种设计可以迁移到未来新的SLAM方法

## 局限与展望
- 多虚拟相机模拟运动模糊带来了显著的计算开销——渲染时间和优化时间分别增加了约4-5倍
- 用5个虚拟相机做离散近似可能对极度模糊的场景不够准确，作者的消融显示增加相机数量有单调提升但边际递减
- 仅处理线性运动模糊（匀速假设），对于旋转主导或加速运动的模糊效果可能有限
- 当前CRF建模为单调递增函数，可能对极紫或极红色偏的自动白平衡校正不够灵活
- 深度传感器的运动模糊问题没有显式建模（假设某一瞬间采集），但TOF深度传感器在快速运动时也会有误差

## 相关工作与启发
- **vs BAD-NeRF**: 也用多位姿积分建模模糊，但BAD-NeRF是静态重建方法，不在SLAM框架内。I²-SLAM利用SLAM的全局轨迹做正则化，这是BAD-NeRF没有的优势
- **vs SplaTAM**: 基于3DGS的RGBD-SLAM，颜色表示固定，无法处理外观变化导致亮度不一致伪影。I²-SLAM通过色调映射模块解决此问题
- **vs NeRF-SLAM**: 使用DROID-SLAM做跟踪+NeRF做建图，但没有对退化建模。I²-SLAM在其基础上加入成像逆过程显著提升两方面性能
- **vs HDR-NeRF/HDR-Plenoxels**: 在静态重建中建模HDR，但没有处理运动模糊，也不在SLAM框架内

## 评分
- 新颖性: ⭐⭐⭐⭐ 将成像逆过程系统性地集成到SLAM中，每个组件设计都有物理动机
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实数据集，RGB+RGBD模式，多种基线对比，详尽的消融
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，物理建模和SLAM集成的衔接流畅
- 价值: ⭐⭐⭐⭐ 解决了实际部署中的核心痛点，模块化设计便于集成

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[ECCV 2024\] Deep Patch Visual SLAM](deep_patch_visual_slam.md)
- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](../../CVPR2026/3d_vision/unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](../../CVPR2025/3d_vision/mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)

</div>

<!-- RELATED:END -->
