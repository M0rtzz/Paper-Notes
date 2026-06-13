---
title: >-
  [论文解读] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 本文提出 Horizon-GS，通过粗到精两阶段训练策略、相机分布平衡机制和多分辨率 LOD 结构，首次实现了航空视角和街景视角的统一 3D 高斯溅射重建和实时渲染，在多个城市场景数据集上达到 SOTA 渲染质量。 1. 领域现状：3D 高斯溅射（3D-GS）以出色的视觉质…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "航空-地面统一重建"
  - "粗到精训练"
  - "多尺度LOD"
  - "大场景渲染"
---

# Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes

**会议**: CVPR 2025  
**arXiv**: [2412.01745](https://arxiv.org/abs/2412.01745)  
**代码**: [https://city-super.github.io/horizon-gs/](https://city-super.github.io/horizon-gs/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 航空-地面统一重建, 粗到精训练, 多尺度LOD, 大场景渲染

## 一句话总结

本文提出 Horizon-GS，通过粗到精两阶段训练策略、相机分布平衡机制和多分辨率 LOD 结构，首次实现了航空视角和街景视角的统一 3D 高斯溅射重建和实时渲染，在多个城市场景数据集上达到 SOTA 渲染质量。

## 研究背景与动机

1. **领域现状**：3D 高斯溅射（3D-GS）以出色的视觉质量和实时渲染速度成为神经场景重建的主流方法。在大尺度城市场景中，航空视角（VastGaussian、CityGaussian 等）和街景视角（Hierarchical-3DGS 等）已分别取得了优秀成果。

2. **现有痛点**：现有方法只能处理单一视角类型——要么航空，要么街景，无法将两者统一训练和渲染。直接在两种视角上联合训练 3DGS 效果反而比各自单独训练更差。

3. **核心矛盾**：航空和街景视角联合训练存在两大冲突：(1) **梯度积累冲突**——街景视角会导致密度化策略移除被遮挡的高斯体，而航空视角又要求在这些区域重新生长，互相干扰导致密度化过程不稳定；(2) **相机分布不均衡**——街景相机数量多且捕捉近距离细节，航空相机数量少且聚焦远距大范围，导致训练偏向街景细节而忽略航空结构。

4. **本文目标** 在一个统一的高斯模型中实现航空和街景视角的高质量重建与渲染。

5. **切入角度**：作者观察到航空视角提供全局几何框架，街景视角提供局部细节。因此采用分阶段策略——先用航空视角建立粗糙几何骨架，再用街景视角填充细节。

6. **核心 idea**：粗到精两阶段训练（航空建骨架、街景填细节）+ 相机平衡采样 + 多层 LOD = 统一航空-地面重建。

## 方法详解

### 整体框架

Horizon-GS 基于 Scaffold-GS 的锚点+神经高斯设计。对于大场景，先将场景分块（chunks），每块独立训练后合并。每块的训练分两阶段：第一阶段以航空视角为主建立粗糙几何，初始化多层 LOD 锚点结构；第二阶段冻结第一阶段的参数，以街景视角为主通过增强的密度化策略添加精细细节。框架同时支持 3D 高斯（高质量渲染）和 2D 高斯（精确几何重建）。

### 关键设计

1. **粗到精两阶段训练策略**:

    - 功能：解决航空-街景梯度冲突问题
    - 核心思路：第一阶段（60k 迭代）以航空视角为主（采样概率 $R/(R+1)$，$R=2$ 即 67% 航空），只从航空图像积累梯度做密度化，让高斯体充分发展并覆盖全局特征空间。街景视角引导初始精细高斯的放置但不参与密度化决策。第二阶段（40k 迭代）冻结第一阶段的 MLP 权重和高斯属性以保持全局骨架，减少航空采样比例（$R=1$，50%），用街景视角精细化细节。关键在于第二阶段的密度化策略做了增强——不使用 Scaffold-GS 的体素平均梯度，而是考虑单个神经高斯的最大梯度 $\nabla_g$、平均不透明度 $\sigma$ 和最大投影半径 $r$，满足 $\nabla_g \cdot r \cdot \sigma^{\tau_\sigma} > \tau_g$ 的高斯被提升为新锚点。
    - 设计动机：直接联合训练时两种视角的梯度互相抵消导致高斯密度化失败。通过分阶段避免冲突——先让航空视角无干扰地建立全局结构，再让街景视角无干扰地添加细节。第二阶段冻结第一阶段参数确保全局骨架不被破坏。

2. **多分辨率 LOD 构建**:

    - 功能：适应航空和街景视角之间巨大的细节差异，支持实时渲染
    - 核心思路：借鉴 Octree-GS 的 LOD 策略，根据航空和街景相机距离自动计算所需的 LOD 层数。航空层数 $K_{aerial} = \lfloor \log_2(D_{aerial}/d_{aerial}) \rfloor + 1$，总层数 $K = K_{aerial} + \lfloor \log_2(d_{aerial}/d_{street}) \rfloor$。第一阶段只激活 $K_{aerial}$ 层，新锚点加到同级 LOD。第二阶段开放所有层，新锚点加到下一级 LOD 以捕捉街景的更高频信息。
    - 设计动机：航空视角对应粗尺度，街景视角对应细尺度，用单一静态分辨率的高斯无法同时满足两者需求。LOD 结构允许不同尺度的细节在不同层级表示，且支持实时渲染（大场景 51.5 FPS）。

3. **大场景分块训练与合并策略**:

    - 功能：将大场景扩展到 GPU 可处理的规模
    - 核心思路：沿地面投影将场景分为 $m \times n$ 块，扩展每块边界确保充分重叠。对航空视角，增加可见相机和辅助点云；对街景视角（遮挡严重），从深度图生成点云确保训练覆盖。各块独立训练后，丢弃边界外的高斯并拼接合并。为加速渲染，将混合表示转为全显式表示（去掉视角输入的 MLP，将颜色 MLP 替换为球谐函数预测）。
    - 设计动机：VastGaussian 的投影分块方法在处理街景视角时存在投影误差。通过为航空和街景分别定制数据增补策略解决了这个问题。

### 损失函数 / 训练策略

渲染损失 $\mathcal{L}_R = \mathcal{L}_1 + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{vol}\mathcal{L}_{vol} + \lambda_d\mathcal{L}_d + \lambda_o\mathcal{L}_o$，其中深度监督权重从 1 指数衰减到 0.01。几何重建额外加法线一致性损失 $\mathcal{L}_S = \mathcal{L}_R + \lambda_n\mathcal{L}_n$。掩码正则化 $\mathcal{L}_o$ 用于消除行人、车辆和天空的影响。在 A100 80G GPU 上训练，大场景各块并行约 4 小时。

## 实验关键数据

### 主实验

**小场景渲染质量（表1，Block_Small 场景）**：

| 方法 | Aerial PSNR↑ | Aerial LPIPS↓ | Street PSNR↑ | Street LPIPS↓ |
|------|-------------|--------------|-------------|--------------|
| 3D-GS | 25.44 | 0.325 | 21.81 | 0.371 |
| Scaffold-GS | 28.44 | 0.191 | 23.84 | 0.271 |
| Hier-GS | 28.31 | 0.189 | 23.75 | 0.220 |
| **Ours** | **30.59** | **0.094** | **23.80** | **0.209** |

**大场景渲染质量（表2，Block_A 场景）**：

| 方法 | Aerial PSNR↑ | Aerial LPIPS↓ | Street PSNR↑ | Street LPIPS↓ |
|------|-------------|--------------|-------------|--------------|
| 2D-GS | 20.63 | 0.595 | 19.57 | 0.477 |
| Scaffold-GS* | 27.62 | 0.206 | 23.10 | 0.277 |
| **Ours*** | **28.89** | **0.151** | **23.66** | **0.255** |

### 消融实验

**UC-GS 数据集视角外推对比（表3）**：

| 方法 | Held-out PSNR↑ | +1m PSNR↑ | +1m 5°down PSNR↑ |
|------|---------------|-----------|-------------------|
| 3D-GS | 23.47 | 20.83 | 21.25 |
| UC-GS | 25.95 | 23.52 | 24.15 |
| **Ours** | 25.35 | **25.46** | **25.37** |

### 关键发现

- Horizon-GS 在航空视角上的 PSNR 提升非常显著（比 Scaffold-GS 高 2+ dB），说明两阶段策略有效解决了航空-街景冲突。
- 在街景视角上与 Hierarchical-3DGS（专为街景设计）基本持平，证明统一模型不牺牲单一视角质量。
- 视角外推实验（UC-GS 表3）中，当测试视角偏移训练分布时（+1m shift），Horizon-GS 的性能降幅远小于其他方法，显示出更强的泛化能力。
- 分块训练（*标记）对大场景有显著帮助：Scaffold-GS* vs Scaffold-GS 在 Block_A 航空 PSNR 提升约 5 dB。
- 大场景渲染速度达 51.5 FPS，支持实时应用。

## 亮点与洞察

- **"先骨架后细节"的两阶段训练哲学**：用航空视角建立全局几何、用街景视角补充局部细节，这种从粗到精的分阶段策略适用于所有多尺度融合场景。类似思想可迁移到卫星-无人机-地面多层级重建。
- **冻结第一阶段参数的"保护机制"**：防止精细阶段破坏已建立的全局结构，是一种简洁有效的防遗忘策略。
- **跨视角数据集贡献**：构建了 5 个合成 + 2 个真实世界的航空-街景校准数据集，填补了该领域的数据空白，对后续研究有重要价值。

## 局限与展望

- 真实场景（Real）上的 PSNR 明显低于合成场景（Synthetic），说明对真实数据的处理仍有改进空间。
- 需要预处理深度图（Depth-Anything-V2）和语义分割掩码（Grounded-SAM），pipeline 依赖较多。
- 街景视角的 PSNR 提升不如航空视角显著，可能需要更专门的街景优化策略。
- 分块训练的块边界处可能存在不一致，虽然重叠设计缓解了这个问题但未完全解决。
- 未考虑动态物体建模（行人、车辆只是用掩码去除），限制了在自动驾驶等动态场景中的应用。

## 相关工作与启发

- **vs VastGaussian/CityGaussian**: 这些方法只处理航空视角的大场景重建。Horizon-GS 在统一处理两种视角的同时，航空渲染质量也优于它们。
- **vs Hierarchical-3DGS**: Hier-GS 专门为街景设计的 LOD 层次结构在单一视角上有优势，但无法自然支持航空视角。Horizon-GS 的双阶段 LOD 构建兼顾了两种视角。
- **vs UC-GS**: UC-GS 引入跨视角不确定性但缺乏大场景可扩展性。Horizon-GS 通过分块策略解决了扩展性问题，且在视角外推上更稳定。
- **vs BungeeNeRF**: BungeeNeRF 的多尺度 NeRF 启发了本文的粗到精策略，但 Horizon-GS 基于高斯溅射实现了实时渲染。

## 评分

- 新颖性: ⭐⭐⭐⭐ 航空-街景统一重建的问题定义有价值，两阶段策略设计合理但部分组件来自已有工作
- 实验充分度: ⭐⭐⭐⭐⭐ 11个场景覆盖合成/真实、小/大规模，与多种基线全面对比，视角外推实验说服力强
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深入（图3的冲突分析），整体叙事流畅，图表质量很高
- 价值: ⭐⭐⭐⭐ 对数字孪生、VR/AR、自动驾驶等需要跨尺度场景的应用有直接价值，数据集贡献突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](../../ICCV2025/3d_vision/s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](../../ECCV2024/3d_vision/gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
