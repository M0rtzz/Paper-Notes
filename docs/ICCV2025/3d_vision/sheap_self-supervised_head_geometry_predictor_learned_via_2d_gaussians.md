---
title: >-
  [论文解读] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians
description: >-
  [ICCV 2025][3D视觉][3D head reconstruction] 提出SHeaP，利用2D Gaussian Splatting替代传统可微mesh渲染进行自监督3DMM预测训练，通过将Gaussians绑定到3DMM mesh上实现重动画，并设计graph卷积Gaussians生成器和几何一致性正则化，在NoW和Nersemble基准上超越所有自监督方法。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D head reconstruction
  - 2D Gaussian Splatting
  - 3DMM
  - 自监督学习
  - face geometry
  - rigged avatar
---

# SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians

**会议**: ICCV 2025  
**arXiv**: [2504.12292](https://arxiv.org/abs/2504.12292)  
**机构**: Woven by Toyota, Toyota Motor Europe, TU Munich, Kyoto University
**领域**: 3D视觉 / 人脸重建 / 自监督学习  
**关键词**: 3D head reconstruction, 2D Gaussian Splatting, 3DMM, self-supervised, face geometry, rigged avatar

## 一句话总结
提出SHeaP，利用2D Gaussian Splatting替代传统可微mesh渲染进行自监督3DMM预测训练，通过将Gaussians绑定到3DMM mesh上实现重动画，并设计graph卷积Gaussians生成器和几何一致性正则化，在NoW和Nersemble基准上超越所有自监督方法。

## 背景与动机
从单张2D图像实时重建3D人头模型是CV核心任务，应用于AR/VR/数字人等领域。由于大规模3D GT数据稀缺，自监督方法从2D视频学习成为主流。传统自监督方法使用可微mesh渲染（如DECA/EMOCA），但面临两大瓶颈：(1) mesh光栅化不连续导致梯度不准确；(2) mesh渲染缺乏真实感，使光度损失的监督信号质量有限。此外，mesh渲染需要精细的面部mask来排除头发/肩膀区域，增加了预处理负担。

## 核心问题
如何提升自监督3DMM预测器的训练效果？核心挑战：(1) 渲染质量——可微mesh渲染的局限性限制了光度损失的有效性；(2) 几何-外观耦合——如何确保Gaussians的外观监督能有效传导到底层3DMM的几何参数？

## 方法详解

### 整体框架
SHeaP采用source-target重动画范式进行自监督训练。给定source图像，ViT预测3DMM参数（shape β、pose θ、expression ψ）和identity features f；Gaussians Regressor根据f和DINOv2特征预测一组绑定到3DMM mesh上的2D Gaussians。然后将rigged head avatar重动画到target帧的pose/expression，渲染后与target GT图像计算光度损失进行反向传播。

### 关键设计

1. **3DMM参数估计器**：采用类似TokenFace的ViT架构，将人脸图像分割为patches输入ViT，附加5个可学习token（shape/expression/pose/lighting/features），输出经LayerNorm和MLP产生各项3DMM参数。ViT权重用FaRL初始化。

2. **Gaussians回归器（核心创新）**：由两个子网络组成——

    - **UV Map Generator**：将ViT输出的identity features f reshape为feature map，通过Lightweight GAN架构与DINOv2特征做cross-attention，生成UV空间的特征张量M。
    - **Graph卷积网络**：每个Gaussian有一个parent face和一个可学习embedding e_i；embedding与从UV map采样的region features拼接后，通过ResNet风格的graph卷积网络生成最终Gaussian属性（offset x、scale s、rotation q、albedo c、opacity σ，共14维）。邻接矩阵按mesh face的2度邻域定义。

3. **Gaussians动态密集化/剪枝**：跟踪每个Gaussian prototype的opacity均值和位置梯度。每t_densify步，删除opacity最低的n_prune个prototype，复制位置梯度最大的n_densify个并加噪。保持总数恒定，每个face最少1个最多6个Gaussian。

4. **2DGS绑定机制**：改进GaussianAvatars的绑定公式，将等向性缩放替换为各向异性缩放矩阵S_p = diag(s_u, s_v, s_n)，其中s_u/s_v为三角面片在UV方向的长度，s_n = min(s_u, s_v)。法线方向的缩放不影响2D Gaussian的最终scale，但影响中心位置μ，使Gaussian可以沿法线方向偏移出mesh表面以捕捉mesh未覆盖的细节。

5. **光照模型**：Gaussians Regressor输出反照率albedo，结合基于球谐函数(SH)的Lambertian着色模型。ViT从source图像预测光照PCA权重，通过Basel Illumination Prior变换得到SH系数，最终颜色 = albedo × SH光照。

### 损失函数
- **Landmarks Loss**：预测mesh投影landmarks与2D检测landmarks的L1损失（极小权重，避免landmarks不准确性的影响）
- **光度损失**：target图像与渲染图像间的4项损失（L1 + perceptual + SSIM等）
- **几何一致性正则化（关键）**：约束Gaussians的法线与其parent face法线一致，以及Gaussians的depth map与mesh渲染depth map一致。确保Gaussians的外观优化能有效传导到3DMM几何。

## 实验关键数据

### NoW Benchmark（中性人脸几何评估）
| 方法 | 训练数据 | Median↓ | Mean↓ | Std↓ |
|------|---------|---------|-------|------|
| DECA (3D sup.) | 2D+3D | 1.09 | 1.38 | 1.18 |
| MICA (3D sup.) | 2D+3D | 0.91 | 1.14 | 0.95 |
| TokenFace (3D sup.) | 2D+3D | **0.87** | **1.07** | **0.88** |
| DECA (self-sup.) | 2D only | 1.09 | 1.38 | 1.18 |
| SMIRK (self-sup.) | 2D only | 1.20 | 1.47 | 1.16 |
| **SHeaP** | 2D only | **0.97** | **1.22** | **1.04** |

- 纯2D自监督训练即超越所有self-sup.方法，接近3D监督方法MICA

### Nersemble Benchmark（表情人脸几何评估，新提出）
- SHeaP在非中性表情重建上同样大幅超越所有公开方法
- AffectNet情感分类准确率也达到SOTA

### 消融实验要点
- 使用2DGS vs 3DGS：2DGS整体更优（法线和depth更精确，增强几何耦合）
- 几何一致性正则化贡献最大（去掉后NoW median从0.97恶化到1.15+）
- UV Map Generator + Graph Conv的组合优于纯MLP/纯CNN方案
- 密集化/剪枝机制带来0.02 median的进一步提升

## 亮点
- **2DGS渲染突破mesh渲染瓶颈**：利用2D surfel的优势（精确depth/normal、闭合形式法线计算），本质上提升了自监督信号质量
- **Gaussians几何一致性正则化至关重要**：通过depth和normal一致性约束，将外观学习的梯度有效传导到3DMM参数——这是该方法成功的核心
- **Graph卷积Gaussians生成器设计精巧**：UV map提供全局identity信息+graph conv实现局部Gaussians协调，比直接预测更稳定
- **无需面部mask**：Gaussians的灵活性使模型可自然覆盖头发/肩膀，避免了传统方法对精细face mask的依赖
- **新benchmark**：在Nersemble上建立的表情几何评估填补了非中性表情基准的空白

## 局限与展望
- 仅在FLAME 3DMM上验证，未扩展到其他morphable model
- 推理时仅输出3DMM mesh，Gaussians仅用于训练——未探索直接输出rigged Gaussians avatar的可能性
- 训练依赖成对同identity视频帧（source-target对），数据要求较高
- 光照模型限制为Lambertian+SH，无法建模镜面反射等复杂光照效应

## 与相关工作的对比
- **vs. DECA/EMOCA**：传统diff mesh渲染的自监督方法；SHeaP用2DGS渲染大幅提升监督信号质量
- **vs. SMIRK**：同样追求表情准确性，但SMIRK使用neural renderer条件化mesh渲染做光度损失，SHeaP的2DGS方案更直接且效果更好
- **vs. GaussianAvatars**：GaussianAvatars假设3DMM tracking已给定，仅优化Gaussians；SHeaP同时预测3DMM和Gaussians，挑战更大
- **vs. TokenFace**：TokenFace使用3D监督数据；SHeaP仅用2D数据训练即接近其性能

## 启发与关联
- 2DGS在自监督3D形变模型学习中的应用为其他可形变3D重建任务（手/身体）提供了新范式
- 几何一致性正则化的思路可迁移到任何需要从外观学习几何的self-supervised pipeline
- Gaussians作为训练时的"高质量渲染代理"而非最终表示，这种设计思路值得关注
- 密集化/剪枝在latent space（prototype）而非Gaussian空间操作，为动态Gaussian数量管理提供了新思路
- 2DGS的法线闭合形式计算使得几何一致性正则化成为可能——如果用3DGS则法线定义不明确，这一设计选择的必要性值得注意

## 技术细节补充
- 训练数据：VoxCeleb2视频数据集（大规模in-the-wild人脸视频）
- 3DMM使用FLAME模型（shape 300维、expression 100维、pose 6维）
- 初始每个face分配2个Gaussians，经密集化/剪枝后收敛到平均约4个
- 推理速度：单张图像约15ms（ViT + 3DMM参数预测），满足实时应用需求
- DINOv2特征提供的语义信息帮助区分面部不同区域（眼/鼻/嘴/发际线），使Gaussians分布更合理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将2DGS引入自监督3DMM学习，graph conv + UV map的Gaussians生成器设计新颖
- 实验充分度: ⭐⭐⭐⭐ NoW + 新Nersemble benchmark + AffectNet + 详细消融，但缺少更多3DMM的对比
- 写作质量: ⭐⭐⭐⭐⭐ 管线清晰、各组件动机明确，图示和公式搭配良好
- 价值: ⭐⭐⭐⭐⭐ 为自监督人脸重建树立新SOTA，2DGS+3DMM的范式具有较大影响力

<!-- RELATED:START -->

## 相关论文

- [Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](../../NeurIPS2025/3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)

<!-- RELATED:END -->
