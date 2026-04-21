---
title: >-
  [论文解读] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis
description: >-
  [CVPR 2025][3D视觉][地空联合重建] 本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。
tags:
  - CVPR 2025
  - 3D视觉
  - 地空联合重建
  - 跨视角匹配
  - 伪合成数据
  - 多视图几何
  - 新视角合成
---

# AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis

**会议**: CVPR 2025  
**arXiv**: [2504.13157](https://arxiv.org/abs/2504.13157)  
**代码**: [https://aerial-megadepth.github.io](https://aerial-megadepth.github.io)  
**领域**: 3D视觉  
**关键词**: 地空联合重建, 跨视角匹配, 伪合成数据, 多视图几何, 新视角合成

## 一句话总结
本文提出AerialMegaDepth数据集生成框架，通过将Google Earth的伪合成航空渲染与MegaDepth的真实地面图像联合配准到统一坐标系中，构建了13.2万张混合高度图像的大规模训练数据，微调DUSt3R后将地空配对的相机旋转估计准确率从5%提升到56%，同时显著改善了新视角合成质量。

## 研究背景与动机
多视图3D重建和相机注册是计算机视觉的基石任务。近年来，基于学习的方法（如DUSt3R、MASt3R）在「野外」图像的几何估计上取得了显著进展，但面对一个关键场景几乎完全失败：**地面与航空视角之间的极端视角变化**。测试表明，预训练的DUSt3R在地空图像对上的相机旋转估计准确率（5°以内）仅有约5%。

作者的核心假设是：这一失败的根源在于缺乏联合配准的地空训练数据。现有数据集如MegaDepth主要由游客拍摄的地面图像构成，少有涵盖从无人机到地面的连续视角变化。而独立获取地面和航空图像的相机位姿虽然容易，但将它们合并到同一坐标系需要专业传感器或大量人工对齐，难以大规模化。

本文的解决方案很巧妙：利用Google Earth等地理空间平台的3D纹理网格进行多高度渲染（伪合成数据），再将大量真实地面图像通过视觉定位管道注册到同一坐标系中，形成混合数据集。

## 方法详解

### 整体框架
数据生成分两步：1）从Google Earth渲染伪合成图像并构建3D重建；2）将MegaDepth的真实地面图像配准到伪合成重建中。最终生成的AerialMegaDepth包含137个地标场景、132,137张联合配准的图像。基于此数据集微调DUSt3R和MASt3R等模型来改善地空几何估计。

### 关键设计

1. **伪合成数据生成**:

    - 选择Google Earth作为数据源，因其覆盖大量地标且纹理质量高
    - **自动生成查询视角**：利用MegaDepth图像的EXIF GPS标签将SfM重建转换到全球坐标系（ECEF），从中采样200个点作为look-at目标
    - 在每个地标生成600张不同高度（1m到350m）的渲染图像，共82,220张
    - 称为"伪合成"而非"合成"，因为3D网格是由真实照片纹理化的
    - 虽然Google Earth不提供底层3D网格的直接访问，但已知相机内外参，因此通过特征提取+匹配+三角化重建3D点云

2. **真实图像联合配准**:

    - 关键观察：尽管伪合成图像和真实图像存在域差距（缺少瞬态物体、光照简单），但SOTA特征匹配方法仍能建立可靠对应
    - 采用标准视觉定位流水线：为每张真实图像检索最相似的伪合成图像 → 2D特征匹配提升为2D-3D对应 → RANSAC PnP求解6-DoF位姿
    - 进一步用COLMAP MVS生成半密集深度图用于监督
    - 共配准49,937张MegaDepth图像与82,200张伪合成图像

3. **训练数据对构造——非对称共视矩阵**:

    - 计算N×N共视矩阵$\mathcal{C}$，其中$\mathcal{C}[i,j]$表示图像$i$中可见的3D点在图像$j$中的可见比例
    - 对地空设置，优先选择非对称高的图像对（地面图像只看到大场景的一小部分，航空图像看到很多）
    - 设计分数$s = \text{AM}/\text{HM}$（算术平均/调和平均），高分数表示视角差异大
    - 总共生成150万训练图像对

4. **下游任务微调**:

    - **多视图位姿与几何估计**：在DUSt3R和MASt3R上微调，回归图像对的3D点图（pointmap）
    - **新视角合成**：在ZeroNVS上微调，以单张航空图像为参考生成地面视角，训练时与MegaScenes按3:1比例混合防止过拟合

### 损失函数 / 训练策略
- 使用DUSt3R原始的pointmap回归损失进行微调
- 从8个公开数据集预训练的DUSt3R checkpoint开始微调
- 新视角合成使用ZeroNVS的标准扩散损失，基于MegaScenes预训练checkpoint微调

## 实验关键数据

### 主实验 — 地空相机配准

| 方法 | RRA@5° | RRA@10° | RRA@15° | RTA@5° |
|------|--------|---------|---------|--------|
| DUSt3R (baseline) | 5.20 | 7.95 | 9.48 | 2.75 |
| DUSt3R + MatrixCity | 17.85 | 37.28 | 42.80 | 11.33 |
| DUSt3R + PSynth | 31.28 | 47.63 | 51.61 | 28.78 |
| DUSt3R + Hybrid | **55.96** | **71.25** | **76.15** | **46.48** |
| MASt3R (baseline) | 3.36 | 3.36 | 4.59 | 2.45 |
| MASt3R + Hybrid | **49.54** | **66.36** | **72.48** | **42.51** |

### 多视图场景注册（1航空+N地面图像，RRA@15°）

| 方法 | N=2 | N=3 | N=4 | N=5 |
|------|-----|-----|-----|-----|
| DUSt3R-GA (baseline, 仅地面) | 12.20 | 32.21 | 38.31 | 43.98 |
| DUSt3R-GA + Hybrid (含1航空) | **56.10** | **55.28** | **57.72** | **59.27** |

### 新视角合成（航空→地面）

| 方法 | DreamSim↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|-----------|--------|-------|-------|
| ZeroNVS (MegaScenes) - 伪合成 | 0.448 | 0.413 | 10.85 | 0.416 |
| ZeroNVS (Ours) - 伪合成 | **0.377** | **0.359** | **12.38** | **0.484** |
| ZeroNVS (MegaScenes) - 真实 | 0.550 | 0.639 | 7.48 | 0.183 |
| ZeroNVS (Ours) - 真实 | **0.442** | **0.580** | **8.22** | **0.218** |

### 消融实验

| 配置 | RRA@5° | 说明 |
|------|--------|------|
| 仅伪合成数据（PSynth） | 31.28 | 相比baseline提升6倍 |
| 仅合成数据（MatrixCity） | 17.85 | 纯合成数据有域差距 |
| 混合数据（Hybrid） | **55.96** | 真实地面图像弥补域差距 |

### 关键发现
- 混合数据比纯伪合成数据提升近一倍（RRA@5°: 31%→56%），证明真实地面图像对弥合域差距至关重要
- 伪合成数据比纯合成数据（MatrixCity）更有效，因为其纹理来自真实照片
- 仅增加一张航空参考图像就能显著提升多张地面图像的位姿估计（像"鸟瞰地图"一样连接各地面视角）
- 微调后模型在同视角对（地地、空空）上的性能没有明显下降
- 3D点图精度也大幅提升（1m误差内的点比例从42%提升到62%）

## 亮点与洞察
- 数据驱动的思路值得学习：与其设计复杂的跨视角匹配算法，不如提供合适的跨视角训练数据
- 混合真实+伪合成数据的策略巧妙地结合了两者优势：伪合成数据提供航空视角覆盖，真实数据提供视觉保真度
- 一张航空图像作为"地图"来连接互相没有重叠的地面图像，这个insight对实际无人机应用很有启发
- 非对称共视矩阵的图像对选择策略（AM/HM得分）优雅地刻画了地空视角差异
- 框架具有很强的可扩展性——可以接入其他众源数据集和地理空间平台

## 局限与展望
- Google Earth的3D网格质量参差不齐，部分地标的地面纹理较差
- GPS标签精度有限，可能引入初始对齐误差
- 数据集覆盖137个地标，地理分布可能存在偏差（主要是旅游景点）
- 新视角合成的质量虽有提升但仍有明显瑕疵，极端视角变化下的生成仍是开放问题
- 目前依赖COLMAP的MVS来生成深度监督信号，其精度受限于SfM重建质量
- 未探索卫星视角与无人机视角之间的桥接

## 相关工作与启发
- MegaDepth开创了利用互联网众源图像的SfM重建来训练几何估计模型的范式，本文将其扩展到地空联合设置
- DUSt3R和MASt3R展示了端到端学习3D几何的强大能力，但受限于训练数据的视角分布
- BlendedMVS混合了渲染图像和真实图像用于MVS训练，本文将类似思想推广到跨视角设置
- 视觉定位方法（如利用3D城市网格进行定位）验证了伪合成到真实图像的特征匹配可行性
- 未来可以进一步桥接卫星→无人机→地面三级视角，迈向行星级3D重建

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据构建框架新颖（混合伪合成+真实），但方法主要是利用现有模型
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务评估（位姿估计、多视图注册、NVS），在真实数据上零样本评测
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述极其清晰，实验设计合理，图示直观
- 价值: ⭐⭐⭐⭐⭐ 数据集将推动整个地空3D重建领域发展，框架可扩展到更多场景

<!-- RELATED:START -->

## 相关论文

- [MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
