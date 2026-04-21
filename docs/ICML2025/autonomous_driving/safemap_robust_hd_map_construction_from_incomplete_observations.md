---
title: >-
  [论文解读] SafeMap: Robust HD Map Construction from Incomplete Observations
description: >-
  [ICML 2025][自动驾驶][HD Map Construction] SafeMap 提出了一个即插即用的鲁棒高精地图构建框架，通过高斯采样视角重建（G-PVR）和蒸馏式 BEV 校正（D-BEVC）两个模块，在相机视角缺失的不完整观测条件下仍能准确构建矢量化高精地图。
tags:
  - ICML 2025
  - 自动驾驶
  - HD Map Construction
  - BEV Perception
  - Sensor Failure Robustness
  - View Reconstruction
  - 知识蒸馏
---

# SafeMap: Robust HD Map Construction from Incomplete Observations

**会议**: ICML 2025  
**arXiv**: [2507.00861](https://arxiv.org/abs/2507.00861)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: HD Map Construction, BEV Perception, Sensor Failure Robustness, View Reconstruction, knowledge distillation

## 一句话总结

SafeMap 提出了一个即插即用的鲁棒高精地图构建框架，通过高斯采样视角重建（G-PVR）和蒸馏式 BEV 校正（D-BEVC）两个模块，在相机视角缺失的不完整观测条件下仍能准确构建矢量化高精地图。

## 研究背景与动机

在线高精地图（HD Map）构建是自动驾驶的关键任务，为车辆规划和导航提供精确的静态环境信息。当前主流方法依赖多视角相机的完整输入，但在实际驾驶场景中，相机可能因遮挡、故障或损坏而丢失某些视角的图像数据。

现有方法存在以下问题：

**脆弱性暴露**：MapBench 评测表明传感器故障会严重影响 HD Map 模型性能，威胁交通安全

**已有方案局限**：MetaBEV、UniBEV 等方法针对 3D 目标检测的传感器失效问题，仍依赖完整多视角图像；M-BEV 仅利用相邻视角局部裁剪进行恢复，需要预设裁剪比例且未充分利用所有可用视角信息

**领域空白**：面向 HD Map 构建的不完整观测鲁棒方法尚未被充分探索，而地图构建高度依赖周围相机捕获的静态环境数据

SafeMap 是**首个**专门针对不完整多视角相机数据进行 HD Map 构建的鲁棒框架。

## 方法详解

### 整体框架

SafeMap 构建在 MapTR 框架之上，由四个核心组件组成：

1. **Map Encoder**：2D 特征提取器 + PV-to-BEV 变换模块
2. **G-PVR 模块**（Gaussian-based Perspective View Reconstruction）：基于高斯采样的视角特征重建
3. **D-BEVC 模块**（Distillation-based BEV Correction）：蒸馏式 BEV 特征校正
4. **Map Decoder**：基于 MapTR 的解码器与预测头

训练时通过 **Random View Masking (RVM)** 随机屏蔽某个视角的 2D 图像特征来模拟相机故障，然后通过 G-PVR 和 D-BEVC 模块恢复缺失信息。测试时，重建模块能够预测缺失视角的特征。

### 关键设计

#### G-PVR：高斯采样视角重建模块

该模块是论文最核心的创新，解决了"如何从多个可用视角重建缺失视角"的问题。

**核心思想**：不同视角对缺失视角的重建贡献不同——相邻视角包含最相关的信息，应获得更高权重；远离的视角（如对面视角）信息较少，权重应降低。

**具体步骤**：

1. **全景视角拼接**：以缺失视角的左右相邻视角为起点，按空间距离远近依次排列所有可用帧，拼接成全景透视图 $F_{PPV} = \text{Concat}(F_{PV}^a) \in \mathbb{R}^{H \times N_a W \times C}$

2. **高斯参考点生成**：在全景透视图上生成高斯分布的参考点：

    - 水平方向：$p_x \sim \mathcal{N}(N_a W / 2, \sigma^2)$（以中心为均值，使采样点集中在相邻视角区域）
    - 垂直方向：$p_y \sim \mathcal{U}(0, H)$（均匀分布覆盖全高度）

3. **可变形注意力重建**：利用可学习 query $V$ 和偏移网络 $\theta_{\text{offset}}$ 生成采样偏移，通过可变形注意力机制在参考点位置采样特征：
    $\hat{k} = \hat{x} W_k, \quad \hat{v} = \hat{x} W_v$
    $\Delta p = \theta_{\text{offset}}(V), \quad \hat{x} = \phi(F_{PV}^a; p + \Delta p)$

4. **MAE 式 Transformer 重建**：使用类 MAE 的 Transformer blocks 重建缺失视角特征

**G-PVR 相比 Local PVR（M-BEV 方法）的优势**：
- 利用所有可用视角而非仅相邻两个视角
- 无需预设裁剪比例
- 通过高斯分布自然编码视角重要性先验
- 可灵活适应不同数量的输入视角和空间关系

#### D-BEVC：蒸馏式 BEV 校正模块

在 PV 层面重建之后，还需要在 BEV 全局特征空间进一步校正。D-BEVC 利用完整观测的全景 BEV 特征 $F_{BEV}^{com}$ 作为监督信号，通过 MSE 损失校正不完整观测的 BEV 特征 $F_{BEV}^{incom}$：

$$\mathcal{L}_{Cor} = \text{MSE}(F_{BEV}^{com}, F_{BEV}^{incom})$$

该模块使不完整观测的 BEV 特征在训练过程中隐式受益于完整观测的全景 BEV 特征。

### 损失函数 / 训练策略

**总体优化目标**由三部分组成：

$$L = L_{\text{map}} + \lambda_1 L_{\text{Rec}} + \lambda_2 L_{\text{Cor}}$$

| 损失项 | 含义 | 权重 |
|--------|------|------|
| $L_{\text{map}}$ | 地图构建损失（分类 + point2point + 边缘方向） | 1.0 |
| $L_{\text{Rec}}$ | PV 重建损失：$\|F_{PV}^{com} - F_{PV}^{incom}\|$ | $\lambda_1 = 0.05$ |
| $L_{\text{Cor}}$ | BEV 校正损失：$\text{MSE}(F_{BEV}^{com}, F_{BEV}^{incom})$ | $\lambda_2 = 5$ |

**训练策略**：
- 优化器：AdamW，学习率 $4.2 \times 10^{-4}$
- 训练：nuScenes 上微调 8 epochs，Argoverse2 上微调 2 epochs
- Batch size：nuScenes 为 4，Argoverse2 为 6
- 高斯方差：$\sigma = 3$
- 每帧最多 100 个地图元素，每个元素 20 个点
- BEV 网格大小 0.75m，Transformer 解码器 2 层
- 训练时每帧随机丢弃一个视角的 RGB 图像

## 实验关键数据

### 主实验

实验基于 nuScenes（6 视角）和 Argoverse2（7 视角）数据集，评估指标为 mAP（基于 Chamfer 距离，阈值 0.5m/1.0m/1.5m）。

| 数据集 | 基线模型 | 场景 | mAP 提升 | 说明 |
|--------|----------|------|----------|------|
| nuScenes | HIMap | 各种缺失视角 | +2.4% ~ +18.2% | 6 种单视角缺失场景 |
| Argoverse2 | MapTR | 完整观测 | +1.0% | 完整视角下也有提升 |
| Argoverse2 | MapTR | 前视角缺失 | +4.1% | 前视角影响最大 |
| nuScenes | MapTR | 传感器腐蚀 | mRR +1.9%, mCE +9.4% | 8 种真实腐蚀场景 |
| nuScenes | HIMap | 传感器腐蚀 | mRR +6.2%, mCE +16.8% | MapBench 评测 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| Baseline（无重建模块） | 基准 | 仅 MapTR 原模型 |
| + G-PVR only | 显著提升 | PV 层面重建有效 |
| + D-BEVC only | 显著提升 | BEV 层面校正有效 |
| + G-PVR + D-BEVC（完整 SafeMap） | 最佳 | 两模块互补 |
| Mean-PVR（平均所有可用视角） | 低于基准 | 简单平均不可行 |
| MAE-PVR（MAE + masked token） | 优于 Mean | 但弱于 G-PVR |
| Standard-PVR（均匀参考点） | 优于 MAE | 但弱于 G-PVR |
| Gaussian-PVR（高斯参考点） | **最优** | 先验知识有效 |
| D-BEVC w/ L1 | 次优 | Manhattan 距离 |
| D-BEVC w/ L2 | **最优** | Euclidean 距离（默认） |
| D-BEVC w/ KL | 最弱 | KL 散度效果差 |

### 关键发现

1. **前/后视角最关键**：缺失前视角（CAM_FRONT）和后视角（CAM_BACK）对性能影响最大，因为这两个视角包含最多的关键地图元素
2. **参数增加极小**：SafeMap 仅增加 0.4MB~3.6MB 参数，推理速度和 GPU 内存消耗几乎不变
3. **多视角缺失容忍度**：随着缺失视角数量增加（1→5），性能逐渐下降；SafeMap 在缺失 1~5 个视角时均显著优于 MapTR，性能衰减更缓慢
4. **超参数稳健**：$\lambda_1 \in [0.01, 0.09]$，$\lambda_2$ 和 $\sigma \in [1, 5]$ 范围内性能均表现稳定

## 亮点与洞察

1. **高斯采样编码视角先验**：将相邻视角更重要的直觉通过高斯分布优雅地编码为参考点分布，比均匀采样或局部裁剪更自然有效
2. **双层重建策略**：PV 层面重建（G-PVR）负责局部特征恢复，BEV 层面校正（D-BEVC）负责全局一致性，两者互补
3. **即插即用设计**：可直接集成到 MapTR、HIMap 等现有框架中，无需大幅修改架构
4. **完整场景也有提升**：虽然主要面向不完整观测，SafeMap 在完整观测下也能提升性能，说明重建模块增强了模型的特征表达能力
5. **首次系统研究**：首个针对 HD Map 构建任务在不完整多视角数据下的鲁棒框架

## 局限与展望

1. **仅处理相机模态**：未涉及多传感器融合（如 LiDAR-Camera 融合）场景下的传感器失效问题
2. **单帧重建**：未利用时序信息，如果引入历史帧可能进一步提升重建质量
3. **训练时需完整数据**：D-BEVC 依赖完整观测作为教师信号，训练数据本身必须包含完整六视角
4. **仅评估单视角缺失为主**：虽然做了多视角缺失实验，但主要对比实验集中在单视角缺失场景
5. **高斯分布假设**：固定方差 $\sigma$ 对所有视角同样适用可能不是最优的，自适应方差可能更好

## 相关工作与启发

- **MapTR/MapTRv2**：SafeMap 的基线方法，端到端矢量化 HD Map 构建的开创性工作
- **HIMap**：混合表示学习的 HD Map 方法，SafeMap 在其上也验证了有效性
- **M-BEV**：3D 目标检测中的 masked view 重建方法，SafeMap 的 G-PVR 是对其 Local PVR 的全面改进
- **MapBench**：HD Map 模型在传感器腐蚀下的鲁棒性评测基准
- **MetaBEV/UniBEV**：处理传感器故障的 BEV 感知方法，但不适用于 HD Map 场景

**启发**：高斯采样作为空间先验的思路可以推广到其他多视角融合任务（如 3D 检测、占用预测），蒸馏式校正策略也可用于其他需要处理缺失输入的场景。

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 高斯采样视角先验+双层重建是新颖组合 |
| 实用性 | 5 | 即插即用、参数增量极小、实际部署价值高 |
| 实验充分性 | 5 | 双数据集、多基线、丰富消融、超参敏感性分析 |
| 写作质量 | 4 | 结构清晰，动机阐述充分 |
| 综合评价 | 4.5 | 实用性强、实验扎实的 ICML 工作 |

<!-- RELATED:START -->

## 相关论文

- [Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](../../CVPR2025/autonomous_driving/uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2025/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

<!-- RELATED:END -->
