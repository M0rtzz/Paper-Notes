---
title: >-
  [论文解读] Open-Canopy: Towards Very High Resolution Forest Monitoring
description: >-
  [CVPR 2025][自动驾驶][树冠高度估计] Open-Canopy 提出了首个开放获取的国家级超高分辨率（1.5m）树冠高度估计基准数据集，覆盖法国超过 87,000 km²，结合 SPOT 卫星影像和航空 LiDAR 数据，同时还提出了树冠高度变化检测的基准任务 Open-Canopy-Δ，在系列 SOTA 模型上建立了全面的实验基准。
tags:
  - CVPR 2025
  - 自动驾驶
  - 树冠高度估计
  - 超高分辨率遥感
  - LiDAR
  - 卫星图像
  - 森林监测
---

# Open-Canopy: Towards Very High Resolution Forest Monitoring

**会议**: CVPR 2025  
**arXiv**: [2407.09392](https://arxiv.org/abs/2407.09392)  
**代码**: https://github.com/fajwel/Open-Canopy (有)  
**领域**: 遥感 / 分割  
**关键词**: 树冠高度估计, 超高分辨率遥感, LiDAR, 卫星图像, 森林监测

## 一句话总结
Open-Canopy 提出了首个开放获取的国家级超高分辨率（1.5m）树冠高度估计基准数据集，覆盖法国超过 87,000 km²，结合 SPOT 卫星影像和航空 LiDAR 数据，同时还提出了树冠高度变化检测的基准任务 Open-Canopy-Δ，在系列 SOTA 模型上建立了全面的实验基准。

## 研究背景与动机

**领域现状**：从卫星图像中估计树冠高度及其变化在环境监测中有重大应用——包括森林健康评估、伐木活动追踪、木材资源估算和碳储量计算。现有方法在中低分辨率（10-30m，如 Sentinel-2）上已取得不错进展，但米级分辨率下的高精度树冠高度估计仍具挑战性。

**现有痛点**：(1) 缺乏开放数据集：大多数现有森林遥感数据集基于商业或封闭数据源（如 Planet、WorldView），严重影响模型的可复现性和公平比较；(2) 分辨率不足：公开可用的全球数据集（如 GEDI + Sentinel-2）分辨率仅 10m，无法捕捉单棵树的信息；(3) 变化检测缺失：几乎没有公开的数据集支持多时相树冠高度变化检测任务。

**核心矛盾**：米级分辨率的树冠映射是精细化森林管理的关键需求，但数据获取和开放性之间存在严重矛盾——高分辨率卫星数据通常是商业的，而免费的公开数据分辨率又太低。

**本文目标**：(1) 构建首个开放获取、国家级、米级分辨率的树冠高度估计基准；(2) 提出多时相变化检测基准；(3) 在统一框架下评估多种 SOTA 计算机视觉模型的表现。

**切入角度**：法国拥有两个关键的公开数据资源——SPOT 6-7 卫星影像（1.5m 分辨率，开放获取）和 IGN 航空 LiDAR HD（厘米级精度的高程数据）。通过将两者结合，可以构建大规模、高质量的训练数据而无需购买商业数据。

**核心 idea**：利用法国公开的 SPOT 卫星影像和 LiDAR 数据构建首个国家级开放树冠高度估计基准，并引入时序变化检测作为新的挑战性任务。

## 方法详解

### 整体框架
Open-Canopy 不是一个新模型，而是一个数据集+基准系统。其框架包括：(1) 数据采集与处理——将 SPOT 卫星影像与 LiDAR 数据精确对齐；(2) 数据集构建——划分训练/验证/测试集，覆盖法国多种地形和植被类型；(3) 基准评测——在统一实验设置下评估多种 SOTA 架构。

### 关键设计

1. **大规模数据集构建（Open-Canopy）**:

    - 功能：为超高分辨率树冠高度估计提供标准化训练和评测数据
    - 核心思路：数据集覆盖法国超过 87,000 km² 的区域，使用 1km×1km 的瓦片网格进行组织。每个瓦片包含：(a) SPOT 6-7 多光谱卫星图像（1.5m 分辨率，4 波段 RGBNIR）作为输入；(b) 由航空 LiDAR HD 数据导出的树冠高度模型（CHM）作为真值标签。LiDAR 数据精度达到厘米级，通过对点云数据计算数字表面模型（DSM）和数字地形模型（DTM）之差获得树冠高度。训练集与测试集之间设置 1km 的缓冲区避免数据泄漏。
    - 设计动机：法国的 SPOT 数据和 LiDAR HD 数据均可免费获取（开放许可），这使得整个数据集可以被全球研究者完全复现。87,000 km² 的覆盖范围确保了植被类型（阔叶林、针叶林、混交林、灌木等）和地形条件的充分多样性。

2. **时序变化检测基准（Open-Canopy-Δ）**:

    - 功能：评估模型对树冠高度时间变化的检测能力
    - 核心思路：利用不同年份（通常间隔 2-5 年）的 SPOT 卫星影像和 LiDAR 数据构建成对的时序观测。模型接收两个时间点的卫星影像，输出树冠高度变化图。变化类型包括自然生长（高度增加）、采伐活动（高度骤降）、自然灾害影响等。这是一个极具挑战性的任务，因为模型需要同时理解单时相的高度估计和跨时相的差异检测。
    - 设计动机：在实际林业管理中，树冠高度的变化比绝对高度更有价值——它直接反映了伐木、灾害、再生等关键事件。然而，这个任务目前完全没有公开的评测基准。

3. **全面的模型基准评测**:

    - 功能：在统一设置下评估多种 SOTA 计算机视觉架构
    - 核心思路：评估的模型包括：(a) 经典 CNN 架构——UNet（ResNet 骨干）、DeepLab；(b) Transformer 架构——ViT (small/base)、PVTv2；(c) 混合架构——Swin Transformer + UNet；(d) 领域专用方法——Tolan et al. 的自监督预训练模型。所有模型使用统一的数据增强、训练超参数和评估指标（MAE、RMSE、R²等），确保公平比较。评估在高度估计（Open-Canopy）和变化检测（Open-Canopy-Δ）两个任务上分别进行。
    - 设计动机：不同架构对遥感数据的表现差异尚不明确——例如 Transformer 在自然图像上已超越 CNN，但在遥感场景下是否仍有优势？大型预训练模型 vs 从头训练的小模型在数据充足时谁更好？这些问题需要在标准化基准上回答。

### 损失函数 / 训练策略
基准实验使用 L1 损失（MAE）或 L2 损失（MSE）进行热量训练。所有模型在同一数据划分和相同训练设置（PyTorch Lightning + Hydra 配置）下训练。采用 ImageNet 预训练权重初始化，学习率通过 cosine 调度衰减。

## 实验关键数据

### 主实验（树冠高度估计）

| 模型 | 骨干网络 | MAE (m)↓ | RMSE (m)↓ | R² ↑ |
|------|---------|----------|-----------|------|
| UNet | ResNet-50 | ~2.8 | ~4.5 | ~0.72 |
| PVTv2 | PVTv2-B2 | ~2.5 | ~4.0 | ~0.78 |
| ViT-Small | DINOv2 预训练 | ~2.6 | ~4.1 | ~0.76 |
| Tolan et al. | SSL 预训练 | ~2.7 | ~4.3 | ~0.74 |
| Swin-UNet | Swin-T | ~2.5 | ~4.1 | ~0.77 |

### 变化检测（Open-Canopy-Δ）

| 方法 | 变化 MAE (m)↓ | 说明 |
|------|-------------|------|
| 单独两次预测取差 | ~3.5 | 误差累积严重 |
| 端到端双输入模型 | ~3.0 | 直接学习变化映射 |
| Siamese 网络 | ~2.9 | 双分支结构效果稍好 |

### 关键发现
- **Transformer > CNN 但优势有限**：PVTv2 和 Swin 比传统 UNet 有约 10% 的提升，但差距不如在自然图像任务上显著，表明遥感任务中局部纹理信息同样重要
- **变化检测极具挑战**：即使最好的模型，变化检测 MAE 也在 3m 左右，对于精确的伐木检测（通常高度变化 >10m）尚可接受，但对缓慢的自然生长检测（年变化 <1m）则完全不足
- **数据规模是王道**：Open-Canopy 的大规模数据（87,000 km²）显著优于小数据集上训练的模型，预训练权重在大数据量下的边际增益递减
- **空间泛化是关键瓶颈**：模型在训练区域外的表现明显下降，尤其是在植被类型差异大的区域

## 亮点与洞察
- **第一个大规模开放基准的价值**：在遥感领域，开放数据集的缺乏一直是可复现研究的最大障碍。Open-Canopy 利用法国的公开数据政策填补了这一关键空白，其数据获取和处理流程也可以被其他国家复用
- **变化检测任务的开创性**：Open-Canopy-Δ 首次将时序树冠变化检测标准化为一个计算机视觉基准任务，为将来的方法研究提供了明确的评测目标
- **360GB 的数据规模**：在遥感基准数据集中属于超大规模，足以训练和评估大型模型，减少了对预训练的依赖

## 局限与展望
- 仅覆盖法国，植被类型偏向温带森林，对热带雨林、干旱地区等场景的泛化能力未知
- SPOT 卫星影像只有 4 波段（RGBNIR），缺少短波红外（SWIR）等对植被分析有价值的光谱信息
- 变化检测任务的精度仍然较低，未来需要探索更强的时序建模方法（如 transformer with temporal attention）
- LiDAR 真值本身存在一定噪声（尤其是非分类点的处理），可能限制了模型训练的上界
- 未来可以扩展到其他国家（如使用 Pléiades Neo 的免费学术数据），构建全球级别的基准

## 相关工作与启发
- **vs Global Canopy Height (Meta AI)**: Meta 使用 GEDI + Sentinel-2 构建全球 10m 分辨率树冠图。Open-Canopy 在分辨率上提升了 ~7 倍（1.5m vs 10m），但覆盖范围仅限法国
- **vs Tolan et al.**: Tolan 等人使用商业卫星数据 + 自监督预训练做树冠高度估计。Open-Canopy 证明在足够大的开放数据上，简单的监督学习可以达到可比甚至更好的效果
- **vs FLAIR 数据集**: FLAIR 是法国的土地覆盖分类基准，Open-Canopy 是其在树冠高度回归任务上的补充，二者覆盖区域有重叠

## 评分
- 新颖性: ⭐⭐⭐ 核心贡献是数据集而非方法，但填补了重要的数据空白
- 实验充分度: ⭐⭐⭐⭐⭐ 模型基准评测非常全面，涵盖多种架构和两个任务
- 写作质量: ⭐⭐⭐⭐ 数据集描述详尽，实验设置透明可复现
- 价值: ⭐⭐⭐⭐ 作为基准数据集的长期价值很高，数据公开政策友好

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)

<!-- RELATED:END -->
