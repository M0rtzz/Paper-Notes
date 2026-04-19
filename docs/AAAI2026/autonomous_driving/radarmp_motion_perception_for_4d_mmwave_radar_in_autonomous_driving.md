---
title: >-
  [论文解读] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving
description: >-
  [AAAI 2026][自动驾驶][4D毫米波雷达] 提出 RadarMP——首个联合解决毫米波雷达目标检测和场景流估计的统一架构，利用相邻帧雷达回波信号（tesseract）的能量流一致性进行自监督训练，在目标检测概率上达到 69.5%（远超现有方法的 44.1%），同时实现精确的 3D 场景运动感知。
tags:
  - AAAI 2026
  - 自动驾驶
  - 4D毫米波雷达
  - 场景流估计
  - 目标检测
  - 自监督学习
  - 运动感知
---

# RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving

**会议**: AAAI 2026  
**arXiv**: [2511.12117](https://arxiv.org/abs/2511.12117)  
**代码**: [github.com/chengrui7/RadarMP](https://github.com/chengrui7/RadarMP)  
**领域**: 自动驾驶  
**关键词**: 4D毫米波雷达, 场景流估计, 目标检测, 自监督学习, 运动感知

## 一句话总结

提出 RadarMP——首个联合解决毫米波雷达目标检测和场景流估计的统一架构，利用相邻帧雷达回波信号（tesseract）的能量流一致性进行自监督训练，在目标检测概率上达到 69.5%（远超现有方法的 44.1%），同时实现精确的 3D 场景运动感知。

## 研究背景与动机

### 毫米波雷达的优势与挑战

4D 毫米波雷达凭借其全天候工作能力（可穿透雨雪雾等恶劣天气），已成为自动驾驶系统的关键传感器。然而，传统 CFAR 目标检测方法依赖统计假设，缺乏对复杂背景杂波和动态场景的建模能力，导致检测性能下降并产生稀疏、噪声严重的雷达点云。

### 现有方法的局限

**检测与运动估计解耦**：现有方法将雷达目标检测和运动估计分为两个独立阶段，检测阶段产生的稀疏噪声点云直接影响后续场景流估计的精度

**光学传感器监督的矛盾**：使用 LiDAR/相机来监督雷达检测模型（如 RPDNet），迫使雷达关注一些低反射特征，反而削弱了多模态感知的互补性

**雷达场景流研究稀缺**：目前仅有 RaFlow 和 CMFlow 两个工作研究雷达点云场景流，且性能远不如 LiDAR 方法

### 核心动机

**目标点在相邻帧雷达回波信号中的能量流方向应与运动场一致，而噪声点的能量流则是无序和不规则的。** 这一关键观察启发了同时进行目标检测和运动估计的联合建模方法。

## 方法详解

### 整体框架

RadarMP 以两个连续的 4D 雷达 tesseract（$\mathbf{S}, \mathbf{T} \in \mathbb{R}^{D \times R \times A \times E}$，分别对应多普勒、距离、方位角、俯仰角维度）作为输入，同时输出：
1. **分割掩码** $\mathbf{M} \in \{0,1\}^{R \times A \times E}$：区分目标点和噪声点
2. **3D 场景流场** $\mathbf{F} = \{\mathbf{f}_i\}$：每个目标点沿距离-方位-俯仰轴的位移向量

整体流程为：

Tesseract → 多普勒通道编码 → 3D 特征金字塔 → 多尺度可变形交叉注意力（关联特征提取） → 全局运动模式感知 → 分割+流预测解码

### 关键设计

#### 1. 多普勒通道编码（Doppler Channel Encoding）

**功能**：将 tesseract 中的多普勒维度从冗余表示转化为紧凑的运动感知特征。

**核心思路**：不同于以往工作简单地对多普勒维度做平均/最大池化，本文将多普勒轴视为特征通道，通过 MLP 编码。同时引入 Softmax 和 Gumbel-Softmax 对多普勒速度进行概率编码：

$$\digamma_{v1} = \mathrm{sum}(\mathrm{matmul}(Ax_d, \mathrm{Softmax}(P_d)))$$
$$\digamma_{v2} = \mathrm{sum}(\mathrm{matmul}(Ax_d, \mathrm{GumbelSoftmax}(P_d)))$$

其中 $Ax_d \in \mathbb{R}^D$ 是多普勒轴，$P_d \in \mathbb{R}^D$ 是原始多普勒能量值。

**设计动机**：多普勒轴在每个空间位置编码了关键的运动相关属性——能量分布反映了每个空间位置对不同多普勒速度的置信度。保留这些信息对于分割（语义线索）和场景流估计（物理线索）都至关重要。通过此编码，多普勒维度从 $D$ 压缩到 $D/8$，同时保留了关键的运动特征。

#### 2. 关联特征提取（Correlation Feature Extraction）

**功能**：在两帧 tesseract 之间建立密集的运动关联。

**核心思路**：采用多尺度可变形交叉注意力机制，将源帧作为 Query、目标帧作为 Value，提取帧间关联特征。

**关联参考点生成**：将两帧投影到 RA、RE、AE 三个 2D 平面，使用预训练的 PWC-Net 预测能量流方向，得到三个多尺度 2D 流分量，然后取平均得到 3D 参考点坐标：

$$r'_l = r_l + \frac{1}{2}(\mathbf{f}^l_{ra}(r) + \mathbf{f}^l_{re}(r))$$
$$a'_l = a_l + \frac{1}{2}(\mathbf{f}^l_{ra}(a) + \mathbf{f}^l_{ae}(a))$$
$$e'_l = e_l + \frac{1}{2}(\mathbf{f}^l_{re}(e) + \mathbf{f}^l_{ae}(e))$$

**多尺度可变形交叉注意力**：通过 ResNet3D 提取三级特征金字塔，使用多尺度可变形注意力实现跨尺度的帧间关联：

$$\digamma^{\mathbf{C}}_l = \mathrm{MSDeformAttn}(\mathbf{q}, \mathbf{p}, \{\mathbf{v}_{\mathbf{T}}^l\}), \quad \mathbf{q} \in \mathbf{q}_{\mathbf{S}}^l$$

最后通过 FPN 聚合得到关联表示 $\digamma_c \in \mathbb{R}^{C_c \times R \times A \times E}$。

**设计动机**：直接在 3D 球坐标空间做密集关联会导致严重的内存开销。可变形注意力通过学习采样偏移和注意力权重，以远低于暴力搜索的计算复杂度实现精准关联。

#### 3. 全局运动模式感知模块（Global Motion Pattern-Aware Module）

**功能**：捕捉全局运动上下文，区分噪声的无序运动、静态目标的全局关联运动和动态目标的局部关联运动。

**核心思路**：设计两种自注意力机制：

- **Global Patch Self-Attention**：将关联特征分成 $4 \times 4 \times 4$ 的 patch，每个 patch 作为一个 token 输入 Transformer 编码器，使用极坐标位置编码
- **Direction Slice Self-Attention**：沿 AE 平面切片，将同一 $(a,e)$ 位置所有距离 bin 作为一个 token，使用方向向量作为位置编码

**设计动机**：不同类型点的运动模式有本质差异——噪声无序、静态目标全局一致、动态目标局部一致。两种注意力分别从体积级和方向级捕捉这些模式，提供充分的分割线索。

### 损失函数 / 训练策略

本文设计了三个专门针对雷达特性的自监督损失函数，**完全不需要显式标注**：

$$\mathcal{L} = \mathcal{L}_{se} + \mathcal{L}_{ef} + \mathcal{L}_{rfs}$$

**1. 分割能量损失 $\mathcal{L}_{se}$**：基于能量分布监督分割——能量越高越可能是目标，且两帧的分割掩码应保持一致

$$\mathcal{L}_{se} = \mathbf{M}_s - \mathrm{sigmoid}(E_f^{\mathbf{S}} - \tau_f^{\mathbf{S}}) + \mathbf{M}_s \times (\mathrm{warp}(\mathbf{M}_s, \mathbf{F}_s) - \mathrm{sigmoid}(E_f^{\mathbf{T}} - \tau_f^{\mathbf{T}}))$$

**2. 能量流损失 $\mathcal{L}_{ef}$**：目标点的流场必须与其能量流方向一致，使用能量强度加权减弱噪声影响

$$\mathcal{L}_{ef} = E_f^{\mathbf{S}} \times (E_f^{\mathbf{S}} - \mathrm{warp}(E_f^{\mathbf{T}}, \mathbf{F}_s))$$

**3. 径向流分割损失 $\mathcal{L}_{rfs}$**：多普勒值乘以帧间时间应近似于目标点真实流的径向投影

$$\delta_v = \digamma_v - \frac{\mathrm{warp}(C, \mathbf{F}_s) - C}{\Delta t} \odot O$$
$$\mathcal{L}_{rfs} = \mathbf{M}_s - \mathrm{sigmoid}(\alpha(\beta - \delta_v^2))$$

**训练细节**：Adam 优化器，初始学习率 0.001，每 2 个 epoch 按 0.9 衰减；3 台 3090 GPU 训练 250 epochs；推理速度 7.6 fps，显存 7.5 GB。

## 实验关键数据

### 主实验

#### 目标检测结果（K-Radar 数据集）

| 方法 | $P_d$ (%)↑ | $P_{fa}$ (%)↓ | CD (m)↓ | SNR (dB)↑ |
|------|-----------|--------------|---------|-----------|
| OS-CFAR | 1.643 | **0.311** | 10.030 | 5.477 |
| RPDNet | 9.311 | 1.821 | 7.590 | 5.175 |
| Radelft | 44.121 | 6.200 | 6.553 | 4.329 |
| **RadarMP** | **69.458** | 1.335 | **3.378** | **5.232** |

RadarMP 的检测概率（69.5%）远超此前最优方法 Radelft（44.1%），提升 **57.4%**，同时保持低虚警率（1.34%）和最优 Chamfer 距离（3.38m）。

#### 场景流估计结果

| 方法 | 分割方式 | EPE3D (m)↓ | AccS3D (%)↑ | AccR3D (%)↑ | Outlier3D (%)↓ |
|------|----------|-----------|------------|------------|---------------|
| RaFlow + OS-CFAR | 传统 | 0.329 | 11.635 | 20.887 | 82.399 |
| CMFlow + Radelft | 学习 | 0.190 | 20.151 | 46.584 | 65.263 |
| CMFlow + RadarMP-P | 本文检测 | 0.168 | 20.396 | 47.985 | 50.841 |
| **RadarMP（联合）** | **联合** | **0.157** | **21.365** | **46.872** | **44.734** |

联合建模（RadarMP）的 EPE3D 为 0.157m，优于所有解耦方案，Outlier3D 从 65.3% 降至 44.7%。

### 消融实验

#### 损失函数消融

| $\mathcal{L}_{se}$ | $\mathcal{L}_{ef}$ | $\mathcal{L}_{rfs}$ | $P_d$ (%)↑ | $P_{fa}$ (%)↓ | EPE3D (m)↓ |
|:---:|:---:|:---:|-----------|--------------|-----------|
| ✓ | ✓ | ✗ | 62.033 | 2.258 | 0.209 |
| ✓ | ✗ | ✓ | 56.224 | 3.847 | 0.788 |
| ✗ | ✓ | ✓ | 19.846 | 17.136 | 0.621 |
| **✓** | **✓** | **✓** | **69.458** | **1.335** | **0.157** |

三个损失函数缺一不可：缺少分割能量损失导致检测概率骤降至 19.8%；缺少能量流损失导致 EPE3D 恶化至 0.788m；完整配置性能最优。

### 关键发现

1. **联合建模优于解耦方案**：同时检测和估计运动使两个任务相互增益
2. **能量流一致性是有效的自监督信号**：无需标注即可实现强大的运动感知
3. **全天候稳定性**：在大雪等相机/LiDAR 严重退化的条件下仍保持可靠的检测和运动估计
4. **PWC-Net 参考点的作用**：去除 PWC-Net 后 EPE3D 恶化 0.31m，说明初始运动估计对可变形注意力至关重要

## 亮点与洞察

- **首个联合目标检测+场景流估计的雷达框架**：将两个长期分离的任务统一到一个架构中
- **完全自监督**：不依赖 LiDAR 的监督信号，保持雷达的传感独立性和互补性
- **从低级信号入手**：直接使用雷达 tesseract（4D回波信号）而非处理后的点云，避免了传统预处理引入的稀疏性和噪声
- **运动一致性驱动检测**：利用物理先验（能量流方向 = 运动方向）设计损失函数

## 局限与展望

- 雷达分辨率有限，无法提供 LiDAR 级别的纹理信息
- 低 RCS 目标（如穿低反射率衣服的行人）邻近杂波时仍然困难
- tesseract 占用大量内存（K-Radar 每帧约 300 MB），需要仔细的维度裁剪
- 缺乏精确的雷达点级标注，难以更细致地分析检测性能
- 未来可探索多帧融合以提升时序一致性

## 相关工作与启发

- **RaFlow/CMFlow**：唯一的雷达场景流工作，使用自/跨模态监督，但性能有限
- **Deformable DETR**：可变形注意力在减少计算复杂度方面的成功被迁移到3D雷达领域
- **PWC-Net**：光流estimation的经典方法被用于2D能量流预测，为3D参考点提供初始化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首个联合检测+流估计的雷达架构，自监督损失设计精巧）
- 实验充分度: ⭐⭐⭐⭐ （消融充分，但仅在一个数据集上验证）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机阐述到位）
- 价值: ⭐⭐⭐⭐⭐ （为4D雷达运动感知开辟新范式，对全天候自动驾驶有重要意义）

<!-- RELATED:START -->

## 相关论文

- [V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](../../CVPR2026/autonomous_driving/adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)
- [LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)

<!-- RELATED:END -->
