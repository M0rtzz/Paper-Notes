---
title: >-
  [论文解读] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2026][自动驾驶][全向感知] O3N 首次提出全向开放词汇占用预测任务，设计纯视觉端到端框架：Polar-spiral Mamba (PsM) 在极坐标空间以螺旋扫描建模全景几何连续性…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "全向感知"
  - "开放词汇"
  - "占用预测"
  - "全景图像"
  - "Mamba"
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

O3N 想用一台全景相机就完成开放词汇的 3D 占用预测：既要在 360° 全景下把空间几何建准，又要能识别训练时没见过的类别。整条流水线先用 CLIP 的视觉编码器从等距矩形投影全景图里提图像特征、用 CLIP 文本编码器把任意类名编码成文本嵌入；接着做 2D-to-3D 视角变换，同时生成笛卡尔的立方体素和极坐标的圆柱体素两套表示，喂进集成了 PsM 的 3D 解码器；最后由 OCA、NMA 两个模块把体素特征和文本嵌入对齐，再经占用预测头输出每个 voxel 的语义标签。

三个核心模块各管一摊：PsM 负责在全景畸变下把几何建连续，OCA 负责让开放词汇语义不过拟合到可见类，NMA 负责弥合 CLIP 文本和视觉之间的模态鸿沟。

### 关键设计

**1. Polar-spiral Mamba (PsM)：在极坐标里把全景几何建得连续又省算力。**

全景成像有个绕不开的麻烦——圆柱体素在极坐标的角度划分处天然不连续，极点附近尤其严重，标准 3D 卷积没法适应这种拓扑，换成 Transformer 又算不起。PsM 用双分支架构绕开这个两难：极坐标分支把圆柱体素 $\mathbf{V}_p \in \mathbb{R}^{C \times R \times P \times Z}$ 先压成 BEV 特征 $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$，再用 P-SMamba 做螺旋扫描——扫描路径从极点出发、半径逐渐增大，沿螺旋线把体素串成一条序列；笛卡尔分支则保留立方体素 $\mathbf{V}_c \in \mathbb{R}^{C \times H \times W \times D}$。两套表示通过预计算好的极坐标-笛卡尔投影关系重采样融合：

$$\mathbf{V}_f^i = \mathbf{V}_c^i + \Phi_{\rho(c)}(\mathbf{V}_p^i)$$

螺旋扫描之所以管用，是因为它的"近密远疏"恰好对上了全景成像的信息密度分布——靠近视点的区域采样密、远处稀，扫描顺序顺着这个规律走就保住了极点区域的空间连续性；而底层用的是 Spatial-Mamba，长程建模能力接近 Transformer，复杂度却只有线性，这也是为什么消融里加 PsM 几乎不增显存（+0.03GB）。

**2. Occupancy Cost Aggregation (OCA)：用匹配代价体积代替直接特征对齐，防止过拟合到可见类。**

开放词汇占用预测最容易栽的坑是：直接把体素特征和文本特征硬对齐，模型会过拟合到训练时见过的语义，一遇到新类就 misalign。OCA 借了 2D 开放词汇分割里 image-text matching cost 的思路，不直接对齐，而是先构造一个体素-文本的匹配代价体积。对每个体素嵌入 $V_i$ 和类名文本嵌入 $T_l$，算它们的 cosine 相似度作为占用代价 $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$，再把这个代价体积依次过 3D 卷积提初始代价嵌入、ASPP 做多尺度空间聚合、Linear Transformer 做类间聚合，最后残差预测。这样模型学的是"体素和各个类名之间的相对匹配关系"而非"体素该映射到哪个固定类别"，对新类的泛化更稳。

配套的监督也避开了简单交叉熵——后者会把每个体素孤立地往某个语义上推。OCA 改用 Scene Affinity Loss $\mathcal{L}_{oca}$，同时用 Precision、Recall、Specificity 度量同类体素该靠拢、异类体素该分开的关系，把场景的结构信息也喂进监督里。训练时这项损失只在 base class 体素上算。

**3. Natural Modality Alignment (NMA)：无梯度对齐文本嵌入与语义原型，弥合 CLIP 的模态鸿沟。**

CLIP 哪怕海量预训练，image 和 text 嵌入之间仍有一道模态鸿沟，全景投影误差又把它进一步撑大；但如果用可学习的对齐策略去缩这道沟，又会过拟合到 base class 的分布。NMA 干脆把对齐做成无梯度的 Random Walk 迭代。它先用 EMA 维护 base class 的语义原型 $\mathbf{P}_t^b = \alpha \cdot \mathbf{P}_{t-1}^b + (1-\alpha) \cdot \bar{\mathbf{f}}_{seg}$，再算文本和原型之间的 affinity $\mathcal{S} = \lambda \frac{\mathbf{T}_t^0 \cdot \mathbf{P}_t^0}{\|\mathbf{T}_t^0\| \|\mathbf{P}_t^0\|}$，然后让原型和文本嵌入交替游走到收敛。收敛态有 Neumann 级数的闭式解：

$$\mathbf{T}_t^\infty = (1-\beta)(\mathbf{I} - \beta^2 \mathcal{A})^{-1}(\beta \mathcal{S} \mathbf{P}_t^0 + \mathbf{T}_t^0)$$

整个迭代不回传梯度，所以不会被训练分布带偏，这正是它能稳定弥合鸿沟又不过拟合的原因。为了照顾未见类别，NMA 还额外引入 novel class 的可学习原型，隐式地把新类语义也纳进对齐空间。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
