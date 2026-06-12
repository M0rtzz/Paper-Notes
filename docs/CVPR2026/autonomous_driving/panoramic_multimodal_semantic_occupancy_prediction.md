---
title: >-
  [论文解读] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots
description: >-
  [CVPR 2026][自动驾驶][全景占据预测] 面向四足机器人构建首个全景多模态（RGB+热成像+偏振+LiDAR）语义占据数据集PanoMMOcc，并提出VoxelHound框架，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块实现鲁棒的3D占据预测…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "全景占据预测"
  - "四足机器人"
  - "多模态融合"
  - "垂直抖动补偿"
  - "BEV感知"
---

# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**会议**: CVPR 2026  
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)  
**代码**: [https://github.com/SXDR/PanoMMOcc](https://github.com/SXDR/PanoMMOcc)  
**领域**: 自动驾驶  
**关键词**: 全景占据预测, 四足机器人, 多模态融合, 垂直抖动补偿, BEV感知  

## 一句话总结
面向四足机器人构建首个全景多模态（RGB+热成像+偏振+LiDAR）语义占据数据集PanoMMOcc，并提出VoxelHound框架，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块实现鲁棒的3D占据预测，达到23.34% mIoU（+4.16%）。

## 背景与动机

**领域现状**：3D语义占据预测是连接感知与运动规划的关键中间表示，能统一建模自由空间、占据空间和未知空间。全景相机提供360°无盲区视觉覆盖，非常适合移动机器人。然而，现有占据预测方法和数据集几乎全部面向轮式自动驾驶场景——使用多视角针孔相机和车载LiDAR。四足机器人面临三个独特挑战：(1) 传感器视点低，自遮挡严重；(2) 步态运动引起剧烈的垂直抖动，导致图像模糊和特征错位；(3) 仅依赖RGB在光照变化、低纹理区域和长距离场景下不够鲁棒。因此，需要全景成像+多模态感知的联合方案，但此前不存在这样的数据集和方法。

**本文目标**：如何在四足机器人平台上，利用全景相机和多种互补传感器（热成像、偏振、LiDAR），克服步态抖动和单一模态局限性，实现准确的3D语义占据预测？这个问题包含三个子问题：(1) 缺少面向四足机器人的全景多模态占据数据集；(2) 步态引起的垂直抖动破坏BEV变换的空间一致性；(3) 异构模态的有效融合策略。

## 方法详解

### 整体框架

VoxelHound 要解决的是四足机器人在低视点、步态抖动、单模态不鲁棒三重困境下的 3D 语义占据预测。它同时接收全景 RGB（PAL 相机，360°×70° FoV）、热成像、偏振三种图像和 LiDAR 点云：相机分支对三种图像分别用 ResNet-18 提多尺度特征、FPN 聚合，再经垂直抖动补偿（VJC）清掉步态偏移后做 2D→BEV 投影；LiDAR 分支把点云体素化后用稀疏 3D 卷积压到同一个 BEV 平面。四路 BEV 特征经多模态信息提示融合（MIPF）汇聚后过 SECOND-FPN 编码器做上下文建模，最后占据头把 BEV 的通道维 reshape 成垂直维，输出 64×64×16 的 3D 占据（12 个语义类 + 空闲类）。VJC 与 MIPF 是其中两个核心模块。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IMG["全景 RGB + 热成像 + 偏振"] --> CAM["相机分支<br/>ResNet-18 多尺度特征 + FPN 聚合"]
    CAM --> VJC["垂直抖动补偿 VJC<br/>沿宽度取均值 → 1D 卷积回归 Δh → 重采样对齐"]
    VJC --> CB["2D→BEV 投影"]
    LID["LiDAR 点云"] --> LB["LiDAR 分支<br/>体素化 + 稀疏 3D 卷积 → BEV 平面"]
    CB --> MIPF["多模态信息提示融合 MIPF<br/>LiDAR 为几何主体，图像压成语义提示做门控调制"]
    LB --> MIPF
    MIPF --> ENC["SECOND-FPN 编码器<br/>BEV 上下文建模"]
    ENC --> HEAD["占据头<br/>通道维 reshape 成垂直维"]
    HEAD --> OUT["3D 语义占据<br/>64×64×16，12 语义类 + 空闲类"]
```

### 关键设计

**1. 垂直抖动补偿模块 VJC：把步态抖动从特征里"减"出去**

四足机器人迈步时身体沿垂直轴上下振荡，采集到的图像因此带有系统性的垂直方向偏移，直接做 BEV 变换会让特征错位。VJC 插在图像编码器和 BEV 变换之间，专门估计并抵消这个偏移：先对特征图沿宽度维取均值，得到只保留垂直结构的特征 $\mathbf{F}_v \in \mathbb{R}^{C \times H}$，用两层 1D 卷积 + ReLU 编码后，再经自适应平均池化 + 线性层回归出一个全局垂直偏移量 $\Delta h$，最后用带 $\Delta h$ 的采样网格做双线性插值把特征对齐回来。

整个模块几乎不增加参数和显存（隐藏通道取 64 时仅 +0.04M），却把抖动这一物理噪声在进入 BEV 空间前就清理掉，相当于给后续所有模块喂的是"站稳了拍"的特征。

**2. 多模态信息提示融合 MIPF：让 LiDAR 当几何主体、图像只做语义补充**

拼接或相加式的融合对所有模态一视同仁，但 LiDAR 提供的是稳定的 3D 几何骨架、图像模态主要贡献语义——平等对待反而会让噪声更大的图像污染几何。MIPF 改用非对称的"几何主导 + 语义补充"：各模态先用 1×1 卷积投到共享嵌入空间，再对每个图像模态的 BEV 特征做全局平均池化 + MLP，压成一个紧凑的语义提示向量 $\mathbf{p}_m$；然后以 LiDAR BEV 特征为 query、语义提示为 key/value 做注意力交互，结果经 sigmoid 门控做残差调制，即用提示去自适应地重加权 LiDAR 特征，而不是直接覆盖几何结构。

因为提示只有 3 个 token，这比在整张 BEV 上做密集空间交叉注意力高效得多；同时残差调制的形式保证了即使图像分支在夜间或低纹理下不可靠，几何主体也不会被冲垮。

### 损失函数 / 训练策略

采用综合损失：交叉熵 $\mathcal{L}_{ce}$ + Lovász-Softmax $\mathcal{L}_{ls}$（处理类别不平衡）+ 几何亲和损失 $\mathcal{L}_{scal}^{geo}$ + 语义亲和损失 $\mathcal{L}_{scal}^{sem}$（鼓励相邻体素一致）。AdamW 优化器，学习率 4e-4，权重衰减 0.01，训练 48 epoch，4×RTX 3090。

## 实验关键数据

| 方法 | 模态 | mIoU |
|------|------|------|
| MonoScene | C | 8.94 |
| EFFOcc-C | C | 4.47 |
| EFFOcc-L | L | 18.77 |
| EFFOcc-T (C+L) | C+L | 19.18 |
| **VoxelHound** | **C+L+T+P** | **23.34** |

| 光照条件 | 模态 | mIoU |
|----------|------|------|
| 白天 | C+L | 22.56 |
| 白天 | C+L+T+P | 23.34 |
| 夜晚 | C+L | 19.17 |
| 夜晚 | C+L+T+P | 18.68 |

### 消融实验要点
- 基线（无VJC无MIPF）: 22.74 mIoU
- +VJC: 22.92（+0.18），验证了抖动补偿的有效性
- +MIPF: 23.14（+0.40），融合模块贡献更大
- 两者同时: 23.34（+0.60），两模块互补
- VJC隐藏通道维度：64最优（23.34），参数增量极小（0.04M）
- MIPF：提示通道维度8、注意力头数8时最优（23.34）

## 亮点与洞察
- **首创性**：首个面向四足机器人的全景多模态占据数据集，填补重要空白
- **VJC设计简洁有效**：用1D卷积估计全局垂直偏移量来补偿步态抖动，思路清晰、计算开销几乎为零
- **MIPF的非对称融合哲学**：将图像模态压缩为紧凑prompt而非做密集交叉注意力，既保护了LiDAR几何主体，又引入了语义增强。这个"几何主导、语义补充"的思路可以迁移到其他多模态融合场景
- **四种传感模态**：热成像在低光照下增强鲁棒性，偏振成像揭示材质和弱目标线索——这些非常规模态的引入值得关注
- **标定工具开源**：提供了LiDAR-相机标定工具

## 局限与展望
- 数据集规模有限（21.6k帧），远小于大规模自动驾驶数据集（nuScenes 40k、SemanticKITTI 43k），难以训练大模型
- 体素分辨率0.4m较粗，不适用于需要精细几何的抓取等操作任务
- 夜间+全模态（18.68 mIoU）反而低于白天+C+L配置（22.56），说明热成像和偏振在夜间的贡献需要更好的融合策略
- 只覆盖室外场景，缺少室内环境
- VJC只补偿全局垂直偏移，对旋转和局部形变未建模
- 主要在自建数据集上验证，缺乏在其他占据benchmark上的泛化性验证

## 相关工作与启发
- **vs EFFOcc**：现有最接近的baseline。VoxelHound在camera+LiDAR配置上已超越EFFOcc-T 4.16 mIoU，加入热成像和偏振后优势更明显。核心差异在于MIPF的非对称融合策略和VJC的抖动补偿。
- **vs MonoScene**：MonoScene是单目相机占据预测方法，在全景场景下只有8.94 mIoU，说明纯视觉方法在四足平台上严重不足（低视点、抖动、光照变化）。
- **vs QuadOcc**：同样面向四足机器人但只使用全景RGB，且类别更少（6类），PanoMMOcc在传感模态丰富度和标注完整度上有显著优势。

## 相关工作与启发
- MIPF中"prompt式融合"的设计可推广到其他多模态任务——用轻量prompt代替密集特征交互

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个四足机器人全景多模态占据数据集和框架，填补空白
- 实验充分度: ⭐⭐⭐ 仅在自建数据集上验证，缺乏跨数据集泛化实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建细节充分，附录详尽
- 价值: ⭐⭐⭐⭐ 数据集和标定工具开源对社区有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
