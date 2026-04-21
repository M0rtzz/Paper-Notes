---
title: >-
  [论文解读] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2026][自动驾驶][全向感知] O3N 首次提出全向开放词汇占用预测任务，设计纯视觉端到端框架：Polar-spiral Mamba (PsM) 在极坐标空间以螺旋扫描建模全景几何连续性；Occupancy Cost Aggregation (OCA) 构建 voxel-text 匹配代价体积避免直接特征对齐的过拟合；Natural Modality Alignment (NMA) 通过无梯度随机游走对齐 pixel-voxel-text 三模态嵌入。在 QuadOcc 上达 16.54 mIoU / 21.16 Novel mIoU（SOTA），大幅超越 OVO 基线。
tags:
  - CVPR 2026
  - 自动驾驶
  - 全向感知
  - 开放词汇
  - 占用预测
  - 全景图像
  - Mamba
---

# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**会议**: CVPR 2026  
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)  
**代码**: [GitHub](https://github.com/) (即将开源)  
**领域**: 自动驾驶  
**关键词**: 全向感知, 开放词汇, 占用预测, 全景图像, Mamba

## 一句话总结

O3N 首次提出全向开放词汇占用预测任务，设计纯视觉端到端框架：Polar-spiral Mamba (PsM) 在极坐标空间以螺旋扫描建模全景几何连续性；Occupancy Cost Aggregation (OCA) 构建 voxel-text 匹配代价体积避免直接特征对齐的过拟合；Natural Modality Alignment (NMA) 通过无梯度随机游走对齐 pixel-voxel-text 三模态嵌入。在 QuadOcc 上达 16.54 mIoU / 21.16 Novel mIoU（SOTA），大幅超越 OVO 基线。

## 研究背景与动机

**领域趋势**：全向图像（360° 全景）在自动驾驶和具身智能中不可或缺，提供完整空间覆盖和语义连续性。3D 语义占用预测将 2D 视觉提升到 3D 空间，是精确空间推理的基础。

**现有方法的双重局限**：
   - **视角限制**：现有占用预测方法大多依赖多视图环视相机（如 nuScenes 6 相机），不适用于使用单个全景相机的机器人和具身智能体
   - **封闭词汇**：现有方法只能识别训练时预定义的语义类别，无法泛化到开放世界的未知物体（如将箱子误分类为道路、将狗误分类为自行车）

**全景图像的特殊挑战**：等距矩形投影 (ERP) 引入严重几何畸变——远离视点的区域在图像中占比越来越小（latitude distortion + extension distortion），导致：(a) 像素-体素映射不均匀；(b) 简单的三模态特征对齐策略容易过拟合到可见语义，misalign 新类语义

**本文贡献**：首次定义**全向开放词汇占用预测**任务——输入单张全景 RGB + 任意类名文本 → 输出 3D 语义占用（含未见类别），并提出首个纯视觉端到端框架 O3N。

## 方法详解

### 整体框架

输入：等距矩形投影全景图 → CLIP 视觉编码器提取图像特征 + CLIP 文本编码器提取类名嵌入 → 2D-to-3D 视角变换（生成立方体和圆柱体双体素表示）→ 3D 解码器（集成 PsM）→ OCA + NMA 模块 → 占用预测头输出 voxel 级语义标签。

关键创新：三个模块分别解决全景几何建模（PsM）、开放词汇语义学习（OCA）、跨模态对齐（NMA）三大挑战。

### 关键设计

1. **Polar-spiral Mamba (PsM) 模块**：

    - 功能：在极坐标空间中高效建模全向 3D 体素特征，保持空间连续性
    - 核心问题：圆柱体素在极坐标角度划分处存在数据不连续性（极点附近尤为明显），标准 3D 卷积无法适应，Transformer 计算成本过高
    - 设计方案：**双分支架构**——
        - *极坐标分支*：圆柱体素 $\mathbf{V}_p \in \mathbb{R}^{C \times R \times P \times Z}$ 压缩为 BEV 特征 $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$，然后用 **P-SMamba 螺旋扫描**——从极点出发、半径逐渐增大的螺旋路径，天然匹配全景成像"近密远疏"的信息密度分布
        - *笛卡尔分支*：立方体素 $\mathbf{V}_c \in \mathbb{R}^{C \times H \times W \times D}$
        - *跨坐标聚合*：利用预计算的极坐标-笛卡尔投影关系重采样并融合：$\mathbf{V}_f^i = \mathbf{V}_c^i + \Phi_{\rho(c)}(\mathbf{V}_p^i)$
    - 优势：Spatial-Mamba 提供 Transformer 级长程建模能力，但只有线性复杂度；螺旋扫描路径保证极点区域的空间连续性

2. **Occupancy Cost Aggregation (OCA)**：

    - 功能：构建体素-文本匹配代价体积，替代直接特征对齐，缓解开放词汇下的过拟合
    - 核心思路：类比 2D 开放词汇分割中的 image-text matching cost，定义**占用代价** $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$（体素嵌入和文本嵌入的 cosine 相似度）→ 3D 卷积提取初始代价嵌入 → ASPP 空间聚合（多尺度感受野）→ Linear Transformer 类间聚合 → 残差预测
    - **Scene Affinity Loss** $\mathcal{L}_{oca}$：不用简单交叉熵（会导致孤立的体素语义映射），而用 Precision + Recall + Specificity 同时度量同类和异类体素关系，增强泛化性
    - 训练时仅在 base class 体素上计算 $\mathcal{L}_{oca}$

3. **Natural Modality Alignment (NMA)**：

    - 功能：无梯度地对齐文本嵌入和语义原型，缩小 CLIP 固有的 image-text domain gap
    - 核心问题：CLIP 尽管经过海量预训练，image 和 text 嵌入之间仍存在模态鸿沟，全景投影误差进一步加剧；直接用可学习策略会过拟合到 base class 分布
    - 设计方案：**无梯度 Random Walk 迭代对齐**——
        - EMA 更新 base class 语义原型：$\mathbf{P}_t^b = \alpha \cdot \mathbf{P}_{t-1}^b + (1-\alpha) \cdot \bar{\mathbf{f}}_{seg}$
        - 计算文本-原型 affinity $\mathcal{S} = \lambda \frac{\mathbf{T}_t^0 \cdot \mathbf{P}_t^0}{\|\mathbf{T}_t^0\| \|\mathbf{P}_t^0\|}$
        - 用 Random Walk 交替更新原型和文本嵌入至收敛：$\mathbf{T}_t^\infty = (1-\beta)(\mathbf{I} - \beta^2 \mathcal{A})^{-1}(\beta \mathcal{S} \mathbf{P}_t^0 + \mathbf{T}_t^0)$
    - 关键细节：还引入 novel class 的可学习原型（隐式捕捉未见语义）；整个过程无梯度回传，避免对训练分布过拟合

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{vox-pix} + \mathcal{L}_{oca}$
    - $\mathcal{L}_{occ}$：交叉熵 + geometric/semantic scene-class affinity loss + focal point loss（MonoScene 标准损失）
    - $\mathcal{L}_{vox-pix}$：体素-像素特征对齐损失（来自 OVO）
    - $\mathcal{L}_{oca}$：scene affinity loss（仅 base class 体素）
- **推理策略**：base class 用占用预测头直接预测；novel class 用蒸馏模块的体素嵌入 $\mathbf{V}$ 与 novel class 文本嵌入的相似度 + OCA 预测概率组合
- **训练配置**：MonoScene 为主体网络，25 epochs，4×RTX3090，batch=4

## 实验关键数据

### 主实验（QuadOcc 验证集）

| 方法 | 类型 | mIoU | Novel mIoU | Base mIoU |
|------|------|------|-----------|-----------|
| MonoScene (全监督) | Camera | 19.19 | 25.56 | 12.82 |
| OneOcc (全监督) | Camera | 20.56 | 27.53 | 13.59 |
| OVO (开放词汇) | Camera | 14.33 | 18.15 | 10.52 |
| **O3N (开放词汇)** | Camera | **16.54** | **21.16** | **11.92** |

- O3N 超越 OVO +2.21 mIoU / +3.01 Novel mIoU
- O3N 的 Novel mIoU (21.16) 超越多个全监督方法（SSCNet 20.13、OccFormer 20.04、VoxFormer-S 14.54）
- 在 SGN-S backbone 上也带来一致增益（13.81→15.52 mIoU），证明框架通用性

### 消融实验

| 配置 | Novel mIoU | Base mIoU | mIoU | FPS | 显存(GB) |
|------|-----------|-----------|------|-----|---------|
| Baseline（无三模块） | 18.06 | 10.90 | 14.48 | 10.67 | 4.28 |
| + PsM | 18.59 (+0.53) | 11.05 | 14.82 | 9.98 | 4.31 |
| + PsM + OCA | 19.78 (+1.72) | 11.02 | 15.40 | 9.71 | 4.86 |
| + PsM + OCA + NMA | **21.16 (+3.10)** | **11.92** | **16.54** | 9.41 | 4.97 |

### 关键发现

- **PsM**：极坐标螺旋扫描带来 +0.53 Novel mIoU，几乎无显存开销（+0.03GB），线性复杂度
- **OCA**：代价体积聚合是性能主力，+1.72 Novel mIoU，显著减少开放词汇下的过拟合
- **NMA**：无梯度对齐进一步释放 +1.38 Novel mIoU，证明缩小模态鸿沟的重要性
- **效率可接受**：完整 O3N 仍保持 9.41 FPS / 4.97GB 显存，支持准实时推理
- **H3O 数据集**：在人视角模拟数据集上也取得一致提升（23.39→24.25 mIoU）

## 亮点与洞察

- **任务定义的先驱性**：首次提出全向开放词汇占用预测任务，为具身智能和机器人感知提供新研究方向
- **极坐标螺旋扫描的几何洞察**：P-SMamba 的螺旋路径设计精准匹配全景成像的信息密度分布规律——近处密集、远处稀疏，这是针对 ERP 畸变的优雅解决方案
- **无梯度对齐避免过拟合**：NMA 用 Random Walk + Neumann 级数闭合解的方式对齐模态嵌入，既有理论保证（收敛性）又避免了学习过程中对 base class 的过拟合
- **模块化和通用性**：O3N 可插入 MonoScene、SGN 等不同占用网络，不依赖特定架构

## 局限与展望

- **场景规模有限**：QuadOcc 仅 6 个语义类、H3O 10 个类，开放词汇的真正挑战（几十到上百类）未测试
- **Novel class 占比极高**：QuadOcc 中 vehicle/road/building 占 ~68% 体素，H3O 中 novel class 占 ~75%——novel class 其实涵盖了大部分场景，泛化难度相对可控
- **全景 baseline 较弱**：对比的 MonoScene、SGN 等都是相对早期的架构，缺乏与更强 occupancy 方法（如 SurroundOcc、GaussianFormer）的比较
- **仅单帧输入**：未利用时序信息，多帧全景输入可能大幅提升效果
- **改进方向**：(a) 扩展到更大规模语义词汇和真实室外场景；(b) 引入时序建模；(c) 与 LLM 结合实现交互式场景理解

## 相关工作与启发

- **vs OVO**：OVO 是开放词汇占用预测的先驱，用冻结 2D 分割器 + CLIP 蒸馏；O3N 在此基础上增加 OCA（代价体积）和 NMA（无梯度对齐），分别针对过拟合和模态鸿沟
- **vs OneOcc**：OneOcc 实现了纯视觉全景占用预测但是封闭词汇；O3N 在其基础上扩展到开放词汇
- **vs CAT-Seg (2D)**：OCA 的代价聚合思想借鉴自 2D 开放词汇分割中的 image-text matching cost；O3N 将其扩展到 3D 体素空间
- **启发**：极坐标表示 + 螺旋扫描的思路可推广到其他全景任务（如全景深度估计、全景检测）；无梯度对齐策略对其他存在 domain gap 的开放词汇任务也有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义全向开放词汇占用预测任务，三个模块各有明确的设计洞察
- 实验充分度: ⭐⭐⭐⭐ 双数据集 + 双 backbone + 充分消融，但 benchmark 规模偏小
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰（NMA 的 Neumann 级数），方法图详尽
- 价值: ⭐⭐⭐⭐ 为具身智能和全景感知开辟新任务和新方法，方向正确且有实际意义

<!-- RELATED:START -->

## 相关论文

- [Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

<!-- RELATED:END -->
