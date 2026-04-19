---
title: >-
  [论文解读] PseudoMapTrainer: Learning Online Mapping without HD Maps
description: >-
  [ICCV2025][3D视觉][在线建图] 提出 PseudoMapTrainer，首次实现**完全不依赖 GT HD Map** 训练在线建图模型：利用 2D Gaussian Splatting（RoGS）从多视角相机图像重建道路表面并结合预训练语义分割（Mask2Former）生成矢量化伪标签，同时设计 mask-aware 匹配算法与损失函数处理部分遮挡的伪标签，支持单次行程和多次行程（众包数据）两种模式。
tags:
  - ICCV2025
  - 3D视觉
  - 在线建图
  - 伪标签
  - Gaussian Splatting
  - HD Map
  - 半监督学习
  - 矢量化地图
---

# PseudoMapTrainer: Learning Online Mapping without HD Maps

**会议**: ICCV2025  
**arXiv**: [2508.18788](https://arxiv.org/abs/2508.18788)  
**代码**: [github.com/boschresearch/PseudoMapTrainer](https://github.com/boschresearch/PseudoMapTrainer)  
**领域**: 自动驾驶 / 在线建图  
**关键词**: 在线建图, 伪标签, Gaussian Splatting, HD Map, 半监督学习, 矢量化地图

## 一句话总结

提出 PseudoMapTrainer，首次实现**完全不依赖 GT HD Map** 训练在线建图模型：利用 2D Gaussian Splatting（RoGS）从多视角相机图像重建道路表面并结合预训练语义分割（Mask2Former）生成矢量化伪标签，同时设计 mask-aware 匹配算法与损失函数处理部分遮挡的伪标签，支持单次行程和多次行程（众包数据）两种模式。

## 研究背景与动机

- **在线建图的核心矛盾**：当前在线建图模型（如 MapTR、MapTRv2、MapVR）在推理时仅需多视角相机图像即可预测矢量化地图，但**训练阶段仍依赖昂贵的高精地图（HD Map）作为 GT**。HD Map 标注成本极高，且地理覆盖范围有限，导致模型难以泛化到未标注区域。
- **现有方案的局限**：
    - HD Map 数据集（如 nuScenes）仅覆盖少数城市，标注场景单一
    - 众包采集的海量行车数据无法被利用，因为缺少对应的 GT 地图
    - 没有任何先前工作探索过不使用 GT HD Map 训练在线建图模型的可能性
- **核心问题**：能否从无标注的传感器数据（相机/LiDAR）自动生成伪标签来替代 GT HD Map，实现零 HD Map 依赖的在线建图训练？

## 方法详解

### 整体流程

PseudoMapTrainer 包含两个阶段：

1. **伪标签生成**：从无标注行车数据自动构建矢量化地图伪标签
2. **在线模型训练**：使用伪标签替代 GT HD Map 训练在线建图模型

### 阶段一：伪标签生成

伪标签生成分为两个子步骤：

#### 1.1 透视图语义分割（Mask2Former）

- 使用 Mask2Former（Swin-Large backbone）在 **Mapillary Vistas V2** 数据集上训练语义分割模型
- 选择 6 个与道路相关的语义类别（车道线、人行道、道路边界等）
- 对 nuScenes 的多视角相机图像进行推理，获取每帧每个视角的像素级语义标签
- **关键**：Mapillary Vistas 是开放数据集，不包含任何 HD Map 信息，因此此步骤不引入 GT 地图依赖

#### 1.2 道路表面重建与矢量化（RoGS）

- 基于 **RoGS（Road Gaussian Splatting）** 从多视角图像重建道路表面的 3D 表示
- RoGS 使用 **2D Gaussian Splatting** 技术：
    - 将道路表面建模为一组 2D 高斯椭圆盘（Gaussian splats）
    - 同时优化 RGB 外观和语义标签通道
    - 利用车辆位姿信息将多帧多视角观测融合到统一的 3D 空间
- 从重建的高斯表示中提取 BEV（鸟瞰图）语义地图
- 对 BEV 语义地图进行矢量化处理，得到由折线（polyline）表示的地图元素（车道分隔线、道路边界、人行横道）
- **两种模式**：
    - **Single-trip**：使用单次行程数据生成伪标签，覆盖范围有限但无需数据对齐
    - **Multi-trip**：聚合同一位置的多次行程数据，通过增加观测密度提升伪标签质量，模拟众包场景

### 阶段二：在线模型训练

#### Mask-Aware 匹配与损失

伪标签的核心挑战在于**部分可观测性**：由于遮挡和视野限制，伪标签只能覆盖场景的部分区域，其余区域处于未知状态（而非"无地图元素"）。

- **传统方法的问题**：标准的匈牙利匹配会将预测结果与伪标签做全局最优匹配，但伪标签中未覆盖区域的"缺失"会被错误地当作负样本，惩罚模型的正确预测
- **Mask-Aware 匹配算法**：
    - 为每个伪标签样本生成一个**可见性掩码（visibility mask）**，标记哪些区域被有效观测
    - 在匈牙利匹配时，仅对掩码覆盖区域内的预测-GT 对计算匹配代价
    - 掩码外区域的预测不参与匹配，不产生损失
- **Mask-Aware 损失函数**：
    - 分类损失：仅对掩码内的预测计算 focal loss
    - 回归损失：仅对掩码内的匹配对计算 L1 距离和方向损失
    - 掩码外的预测既不被奖励也不被惩罚，避免引入噪声梯度

#### 半监督预训练策略

- 先在大量无标注数据（使用伪标签）上预训练在线建图模型
- 再在少量有 GT HD Map 的数据上微调
- 这种策略允许利用海量众包行车数据进行预训练，显著提升模型性能

### 在线建图骨干网络

- 采用 **MapVR / MapTRv2** 作为在线建图模型
- Transformer decoder 结构，输入多视角相机特征，输出矢量化地图元素
- 使用地理分割（Geo-split）替代传统的随机分割，确保训练集和验证集地理位置不重叠

## 实验关键数据

### 数据集

- **nuScenes**：6 个相机视角，包含 1000 个场景的自动驾驶数据集
- **Mapillary Vistas V2**：用于训练语义分割模型

### 伪标签质量评估

| 模式 | 可观测区域 mAP | 全范围 mAP |
|------|---------------|-----------|
| Single-trip | 较高 | 受限于覆盖范围 |
| Multi-trip | 更高 | 显著提升 |

- Multi-trip 模式通过聚合多次行程显著提升伪标签覆盖率和质量
- 可选使用 LiDAR 深度先验进一步约束高斯重建精度

### 在线模型性能

- **核心结论**：使用伪标签训练的在线建图模型达到了**与使用 GT HD Map 训练的模型可比的性能水平**
- 半监督设置（伪标签预训练 + GT 微调）**超越了纯 GT 训练的基线**
- Mask-aware 匹配与损失对伪标签训练至关重要，去掉后性能显著下降
- Multi-trip 伪标签优于 Single-trip 伪标签训练的模型

### 消融实验

- **Mask-aware vs 标准匹配**：mask-aware 方案显著优于忽略可见性的标准匹配
- **Single-trip vs Multi-trip**：Multi-trip 提供更完整的伪标签，训练效果更好
- **LiDAR 辅助**：可选使用 LiDAR 点云约束高斯重建的 z 轴（高度），进一步提升质量
- **半监督预训练**：伪标签预训练 + GT 微调 > 纯 GT 训练，证明了大规模无标注数据的价值

## 亮点与洞察

1. **零 HD Map 训练的开创性**：首次证明在线建图模型可以完全不依赖 GT HD Map 进行训练，打破了该领域对昂贵标注数据的刚性依赖
2. **Gaussian Splatting 的巧妙应用**：将 2D Gaussian Splatting 从新视角合成任务迁移到道路表面重建，同时优化 RGB 和语义通道，生成高质量 BEV 伪标签
3. **Mask-Aware 设计的必要性**：伪标签的部分可观测性是核心技术挑战，mask-aware 匹配与损失的设计使得模型能够在不完整监督下有效学习
4. **众包数据的潜力释放**：Multi-trip 模式和半监督预训练策略为利用海量众包行车数据铺平了道路，这在工业界具有巨大实用价值
5. **模块化设计**：伪标签生成（Mask2Former + RoGS）与在线模型训练（MapVR）完全解耦，可以方便地替换任一模块

## 局限与展望

- **伪标签生成成本**：虽然不需要 HD Map，但 Gaussian Splatting 重建仍需要较高计算资源，每个场景需独立优化
- **语义类别受限**：当前仅处理 6 个道路相关类别，未覆盖红绿灯、交通标志等更复杂的地图元素
- **依赖车辆位姿**：RoGS 重建需要准确的车辆位姿信息（通常来自 GNSS/IMU），位姿噪声会影响伪标签质量
- **仅在 nuScenes 上验证**：虽然方法理论上可泛化，但缺少在 Argoverse 2、Waymo 等其他数据集上的验证
- **时机选择局限**：单次行程的伪标签在遮挡严重（如停车场景、拥堵路段）时质量可能大幅下降
- **未考虑时序一致性**：当前伪标签生成是逐场景独立的，未利用场景间的时序关联

## 相关工作与启发

- **MapTR / MapTRv2**：端到端在线建图的基线模型，PseudoMapTrainer 的在线模型部分基于 MapTRv2
- **MapVR**：Vector-Representation-based 在线建图方法，提供了训练框架
- **RoGS**：道路表面高斯重建方法（Feng et al. 2024），PseudoMapTrainer 的伪标签生成核心
- **Mask2Former**：通用语义分割模型，用于透视图语义标签生成
- **3D Gaussian Splatting**：新视角合成的基础方法，RoGS 是其在道路场景的专用变体
- **ScalableMap / StreamMapNet**：其他在线建图方法，可作为本方法的替代骨干
- **启发**：该工作表明 3D 重建技术（如 Gaussian Splatting、NeRF）可以作为连接无标注数据与监督训练的桥梁，这一思路可推广到其他自动驾驶感知任务（如 3D 检测、语义地图构建）

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [Online Language Splatting](online_language_splatting.md)
- [Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [IM360: Large-scale Indoor Mapping with 360 Cameras](im360_large-scale_indoor_mapping_with_360_cameras.md)

<!-- RELATED:END -->
