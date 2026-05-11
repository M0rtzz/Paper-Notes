---
title: >-
  [论文解读] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs
description: >-
  [CVPR 2026][自动驾驶][语义占用预测] M²-Occ 针对相机故障导致视图缺失的真实场景，提出 MMR（利用相邻相机 FoV 重叠在特征空间重建缺失视图表示）+ FMM（可学习语义原型 memory bank 精炼模糊 voxel 特征），在 SurroundOcc 基线上缺失后视摄像头 IoU +4.93%，缺失 5 个摄像头时仍维持 18.36% IoU（基线崩到 13.35%），且完整视图下性能不妥协。
tags:
  - CVPR 2026
  - 自动驾驶
  - 语义占用预测
  - 传感器故障
  - 缺失视图重建
  - 语义原型
  - 鲁棒感知
---

# M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**会议**: CVPR 2026  
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)  
**代码**: [github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ)  
**领域**: 自动驾驶 / 3D感知  
**关键词**: 语义占用预测, 传感器故障, 缺失视图重建, 语义原型, 鲁棒感知

## 一句话总结

M²-Occ 针对相机故障导致视图缺失的真实场景，提出 MMR（利用相邻相机 FoV 重叠在特征空间重建缺失视图表示）+ FMM（可学习语义原型 memory bank 精炼模糊 voxel 特征），在 SurroundOcc 基线上缺失后视摄像头 IoU +4.93%，缺失 5 个摄像头时仍维持 18.36% IoU（基线崩到 13.35%），且完整视图下性能不妥协。

## 研究背景与动机

**领域现状**：3D 语义占用预测（SOP）是自动驾驶的关键任务，用 voxel 级表示描述车辆周围的几何结构和语义信息。基于相机的方案（SurroundOcc、TPVFormer、VoxFormer）已取得不错进展，通常假设 6 个环视相机全部正常工作。

**被忽视的致命问题**：现实中相机经常出故障——镜头遮挡、硬件损坏、通信中断。即使只有一个相机失效，SurroundOcc 等模型性能就会**断崖式下降**。例如丢失后视摄像头时 IoU 从 32.38% 暴跌到 23.94%（-26%），这对安全关键系统是不可接受的。

**现有鲁棒性工作聚焦 BEV 而非 3D occupancy**：M-BEV、MetaBEV、SafeMap 等处理缺失视图的工作都针对 BEV 检测/地图构建，未涉及密集 3D 语义占用预测这一更困难的任务。

**核心思路**：模拟人类"从上下文推断未见区域 + 利用记忆补全信息"的能力：(a) MMR 利用相邻相机的 FoV 重叠区域在特征空间重建缺失视图；(b) FMM 用全局语义原型作为先验知识，精炼重建后仍模糊的 voxel 特征。

## 方法详解

### 整体框架

输入：N 个环视相机图像（部分可能缺失）→ 共享 ResNet-101 + FPN 提取 2D 特征 → MMR 模块重建缺失视图的特征表示 → 2D-to-3D 视角变换（spatial cross-attention）构建统一 3D 体积 → FMM 模块用语义原型精炼 voxel 特征 → 3D occupancy head 预测逐 voxel 语义标签。

### 关键设计

1. **MMR（Multi-view Masked Reconstruction）**:

    - 功能：在特征空间恢复缺失视图的 2D 特征表示
    - 设计动机：nuScenes 的 6 个环视相机存在显著的 FoV 重叠（如前左相机和前相机的右侧边界区域重叠）。当某个相机故障时，其覆盖区域的部分信息可以从相邻相机的边界特征中恢复
    - 核心流程：
        - **视角关系建模**：将 6 个相机建模为循环图 $\mathcal{N}(v_i) = \{v_{(i-1) \bmod N},\ v_{(i+1) \bmod N}\}$
        - **重叠区域特征提取**：从左右相邻相机的特征图中裁剪重叠边界区域（宽度 $w_{ov}$），与可学习 mask token $\mathbf{e}_{mask}$ 拼接：$\mathbf{f}_{ref} = \text{Concat}(\mathbf{f}_{left}[:,-w_{ov}:],\ \mathbf{e}_{mask},\ \mathbf{f}_{right}[:,:w_{ov}])$
        - **Transformer 解码重建**：6 层 Transformer decoder（8 头 attention）+ 可学习位置编码，将粗糙的结构先验细化为近似原始完整特征的重建结果 $\hat{\mathbf{f}}_i = \mathcal{D}(\mathbf{f}_{ref} + \mathbf{p}_{pos})$
    - 与生成方法的区别：MMR 在**特征空间**重建而非生成原始像素，避免了图像生成的高计算成本和噪声引入
    - 重建损失：$\mathcal{L}_{MMR} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i^{gt}\|_2^2$，仅在被 mask 的视图上计算，防止学到恒等映射

2. **FMM（Feature Memory Module）**:

    - 功能：用全局语义原型作为先验知识，精炼 3D voxel 特征中的语义模糊区域
    - 设计动机：MMR 恢复了几何结构，但重建特征仍可能存在模糊或语义歧义，尤其在远离重叠区域的中心盲区。FMM 类似"长期记忆"，存储了每类物体"理想"的特征表示，用于纠正偏差
    - **Single-Proto 策略**：每个语义类别维护一个全局质心原型 $\mathbf{m}_k$，通过动量滑动平均更新 $\mathbf{m}_k^{(t)} = (1-\lambda)\mathbf{m}_k^{(t-1)} + \lambda \cdot \bar{\mathbf{f}}_k$（$\lambda=0.1$），过滤 mini-batch 噪声。**实验证明 Single-Proto 比 Multi-Proto 更稳定**
    - **Multi-Proto 策略**：每个类别维护 $N_p$ 个子原型捕捉类内变异（如"卡车"包含皮卡和半挂车）。查询时计算 cosine 相似度 + softmax（带温度 $\tau$）加权检索。但在缺失视图条件下，相似度路由可能被噪声放大，导致过度碎片化
    - **Memory-Enhanced Feature**：$\mathbf{x}' = \mathbf{x} + \sum_{k=1}^{K} P(k) \sum_{j=1}^{N_p} \alpha_{k,j} \mathbf{m}_{k,j}$，用预测类别概率 $P(k)$ 作为门控，以残差方式注入语义先验

3. **训练策略——Random View Masking (RVM)**:

    - 训练时随机丢弃部分视图的图像，模拟真实故障场景
    - 测试时按特定模式 mask（单视图/多视图），评估鲁棒性
    - 这种 masking 训练策略类似 MAE 的思想，但作用在整个相机视图级别而非 patch 级别

### 损失函数

总损失 = SurroundOcc 原始占用预测损失 + $\mathcal{L}_{MMR}$（特征重建损失）

## 实验关键数据

### 主实验——单视图缺失

| 缺失视图 | 指标(IoU) | M²-Occ | SurroundOcc基线 | 提升 |
|---------|-----------|--------|----------------|------|
| 后视 (Back) | IoU | **28.87** | 23.94 | **+4.93** |
| 前视 (Front) | IoU | **30.40** | 25.03 | **+5.37** |
| 前左 (Front Left) | IoU | **31.25** | 30.74 | +0.51 |
| 前右 (Front Right) | IoU | **31.17** | 30.56 | +0.61 |
| 后左 (Back Left) | IoU | **31.08** | 30.35 | +0.73 |
| 后右 (Back Right) | IoU | **31.19** | 30.62 | +0.57 |
| 标准（无缺失） | IoU | 32.38 | 32.38 | 0（不妥协） |

- 后视和前视缺失时提升最大（+4.93/+5.37），因为这两个位置与相邻相机重叠最少，原始模型损失最严重
- 完整视图下性能不下降，说明 MMR+FMM 不引入负面干扰

### 多视图缺失扩展实验

| 缺失相机数 | 指标(IoU) | M²-Occ | 基线 | 提升 |
|-----------|-----------|--------|------|------|
| 1 个 | IoU | **30.66** | 28.42 | +2.24 |
| 3 个 | IoU | **26.06** | 20.52 | **+5.54** |
| 5 个 | IoU | **18.36** | 13.35 | **+5.01** |

- 随着缺失相机数增加，鲁棒性优势不断扩大
- 5 个相机缺失（仅剩 1 个）的极端情况下基线 IoU 崩溃到 13.35%，M²-Occ 仍维持 18.36%，保留了关键结构信息

### 消融实验

| 配置 | IoU | mIoU | 说明 |
|------|-----|------|------|
| 无缺失 baseline | 30.13 | 15.31 | 完整输入参考 |
| 有缺失 + 无恢复 | 26.76 | 13.21 | 缺失导致 -3.37 IoU |
| + MMR | 28.19 | 13.79 | 恢复几何结构 +1.43 |
| + MMR + Single-Proto | **28.38** | **13.55** | 最优组合 |
| + MMR + Multi-Proto | 27.76 | 12.15 | 多原型反而不稳定 |

### 关键发现

- **MMR 主要恢复大尺度几何结构**：drive surface、vehicle 等大物体 IoU 大幅提升（如缺失后视时 drive.surf. 从 27.51% 升至 35.02%），但小物体（行人、交通锥）反而可能下降，因为重建特征丢失了高频细节
- **Single-Proto 优于 Multi-Proto**：在视图缺失条件下，visual evidence 本就稀疏，Multi-Proto 的相似度路由反而放大噪声。简单稳定的单质心比精细但脆弱的多子原型更鲁棒
- **计算开销可控**：显存仅增加约 0.15 GB（2.5%），推理时延随缺失视图数线性增加（每个缺失视图需 MMR 重建一次）

## 亮点与洞察

- **问题定义精准**：首次系统研究 3D 语义占用预测在相机缺失条件下的鲁棒性，建立了完整的评估协议（单视图确定性故障 + 多视图随机 dropout），填补了一个重要的研究空白
- **特征空间而非像素空间重建**：MMR 选择在特征空间做重建，巧妙利用了相邻相机 FoV 重叠提供的自然冗余，避免了图像生成的高成本和不稳定性
- **全局语义原型作为 fallback**：FMM 的设计思路很实用——当局部重建特征质量不佳时，退回到全局统计先验（"这个区域应该长成车/路面的样子"），保证语义一致性

## 局限与展望

- **小物体性能下降**：MMR 恢复的特征丢失高频信息，行人/交通锥等小物体 IoU 反而可能下降（如缺失后视时 pedestrian 从 12.50% 降到 10.51%），这在安全关键场景是隐患
- **Multi-Proto 策略未奏效**：消融表明 Multi-Proto 不如 Single-Proto，但原因分析不够深入——可能需要更好的原型更新策略或噪声抑制机制
- **未考虑时序信息**：相邻帧可以提供额外的上下文弥补当前帧缺失，但 M²-Occ 是纯单帧方法
- **仅在 SurroundOcc 一个 baseline 上验证**：未展示在 TPVFormer、OccFormer 等其他主流 occupancy 方法上的泛化能力
- **推理延迟线性增加**：每个缺失视图需要单独运行 Transformer decoder 重建，5 个缺失视图时延迟从 0.50s 增到 1.25s（2.5x），可能不满足实时要求

## 相关工作与启发

- **vs M-BEV**：M-BEV 也用 masked view reconstruction 但针对 BEV 检测任务；M²-Occ 将其扩展到密集 3D occupancy 预测，并新增 FMM 做语义正则
- **vs MetaBEV**：MetaBEV 通过 LiDAR-Camera 跨模态融合处理传感器故障，需要 LiDAR 硬件；M²-Occ 纯相机方案，成本更低
- **vs MAE**：MAE 做 patch 级 masking 的自监督预训练；M²-Occ 的 MMR 做整个视图级别的 masking，且是有监督的（有完整视图的 GT 特征做监督）
- **启发**：FMM 的语义原型 memory bank 思路可以迁移到其他存在输入退化的感知任务（如雨天/雾天/夜间条件下的 3D 感知），作为通用的语义稳定化模块

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 occupancy prediction 的传感器缺失鲁棒性，MMR+FMM 组合合理
- 实验充分度: ⭐⭐⭐⭐ 系统的单/多视图缺失评估协议，消融完整，但仅一个 baseline
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法图直观，实验分析到位
- 价值: ⭐⭐⭐⭐ 解决了一个真实且被忽视的安全问题，对自动驾驶部署有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
