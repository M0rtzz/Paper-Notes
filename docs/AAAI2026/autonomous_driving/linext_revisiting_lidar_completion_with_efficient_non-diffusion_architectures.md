---
title: >-
  [论文解读] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures
description: >-
  [AAAI 2026][自动驾驶][LiDAR场景补全] 提出 LiNeXt，一种轻量级非扩散网络用于LiDAR 3D场景补全，通过**距离感知选择性重复策略**、**Noise-to-Coarse模块**和**Refine模块**直接重建完整点云，在SemanticKITTI上实现了比LiDiff快**199.8倍**的推理速度，Chamfer Distance降低**50.7%**，参数量仅为其**6.1%**。
tags:
  - AAAI 2026
  - 自动驾驶
  - LiDAR场景补全
  - 点云补全
  - 非扩散模型
  - 稀疏卷积
  - 实时推理
---

# LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures

**会议**: AAAI 2026  
**arXiv**: [2511.10209](https://arxiv.org/abs/2511.10209)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: LiDAR场景补全, 点云补全, 非扩散模型, 稀疏卷积, 实时推理

## 一句话总结

提出 LiNeXt，一种轻量级非扩散网络用于LiDAR 3D场景补全，通过**距离感知选择性重复策略**、**Noise-to-Coarse模块**和**Refine模块**直接重建完整点云，在SemanticKITTI上实现了比LiDiff快**199.8倍**的推理速度，Chamfer Distance降低**50.7%**，参数量仅为其**6.1%**。

## 研究背景与动机

### 问题定义
自动驾驶感知系统依赖LiDAR获取3D点云，但LiDAR测量固有稀疏且存在频繁遮挡，导致大量未观测区域。场景补全的目标是从稀疏点云推断并重建缺失的空间结构，为目标检测、位姿估计、建图等下游任务提供完整的3D表示。

### 现有方法的不足

**体素/SDF方法**：受限于分辨率权衡——低分辨率无法捕捉精细几何细节，高分辨率则内存和计算成本高昂

**扩散模型方法**（LiDiff, LiDPM, ScoreLiDAR）：
   - 虽然生成质量高，但**多步迭代采样**带来巨大计算开销（LiDiff推理33.4秒/帧）
   - 高噪声幅度导致点位移严重，增大噪声估计和去除的难度
   - 网络架构复杂，参数量庞大（LiDiff 32.67M，LiDiff† 54.40M）

### 核心动机
- **直接最小化Chamfer Distance**比扩散模型的去噪目标更简单有效——为什么不直接用轻量网络重建场景？
- LiDAR点云存在**距离依赖的空间分布**（近密远疏），现有方法均匀复制输入导致分布不均衡
- 关键观察：点云作为表示直接编码复杂几何和精细空间细节，无需体素化的分辨率妥协

## 方法详解

### 整体框架

LiNeXt的流程分为三步：
1. **距离感知选择性重复（DSR）**：将输入点云按距离分组，不同组使用不同重复因子，生成更均匀分布的带噪点云
2. **Noise-to-Coarse（N2C）模块**：单次前向传播直接从带噪点云重建粗糙场景结构
3. **Refine模块**：利用N2C的中间特征对粗糙输出进行精细化

### 关键设计

#### 1. 距离感知选择性重复策略（DSR）

**功能**：解决现有方法均匀复制输入点云导致的"近密远疏"不均衡问题。

**核心思路**：按距离对点云排序后分为四组，近处少复制、远处多复制：

给定输入点云 $P_{input} = \{p_i\}_{i=1}^N$，计算每点到原点距离 $d_i = \|p_i\|$，按距离升序排列后等分为四组 $G_1, G_2, G_3, G_4$，分配重复次数 $\{r_1=5, r_2=8, r_3=12, r_4=15\}$。对重复后的点集添加高斯噪声得到 $P_{noise}$。

**设计动机**：LiDAR的物理特性决定了近处采样密集、远处稀疏，均匀复制会过度采样近场而欠采样远场。DSR确保了跨距离的均匀覆盖，为后续N2C模块提供更丰富均匀的输入。

#### 2. 多尺度稀疏卷积模块（MSSC）

**功能**：在多个体素分辨率上并行提取点云特征，捕获精细局部几何和粗粒度全局上下文。

**核心思路**：在 $N_{vox}$ 个体素尺度 $g_k \in \{0.01 \times 2^{i-1} | i=1,...,N_{vox}\}$ 上进行并行稀疏卷积。对每个尺度：
- 体素化：$\hat{P}_k = \lfloor P/g_k \rfloor$
- 特征编码：$F_k = \text{MLP}_k(X)$
- 双残差稀疏卷积：$\mathcal{T}_k'' = \text{spconv}_{k,2}(\mathcal{T}_k') + \mathcal{T}_k'$
- 多尺度融合：$F = \text{MLP}_{end}(\text{CONCAT}(O_k))$

**设计动机**：不同尺度的体素分辨率捕获不同层次的空间信息——细粒度保留几何细节，粗粒度捕获全局上下文。残差连接保持几何保真度。

#### 3. 交叉点注意力模块（CPA）

**功能**：在全局场景和局部部分表示之间进行鲁棒的特征融合，通过显式编码空间关系来加强对缺失结构的推断。

**核心思路**：给定主点云坐标 $P_{key}$ 和部分坐标 $P_{query}$（带query和value）：

1. **KNN搜索**建立局部对应关系：$idx = \text{KNN}(P_{query}, P_{key}, k)$
2. **空间嵌入**计算相对位移：$\alpha = \text{MLP}_{pos}(P_{key} - \mathcal{G}(P_{key}, idx))$
3. **关系特征**增强几何感知差异：$Q_{rel} = query - \mathcal{G}(key, idx) + \alpha$
4. **序列化分段最大池化（SSMP）**：将邻域分成 $\hat{K}$ 段，对每段取最大值，实现关键维度压缩
5. **注意力聚合**：$\mathcal{A} = \text{SoftMax}(\text{MLP}_{attn}(\hat{Q}_{rel}))$，然后加权聚合

**设计动机**：标准交叉注意力的二次复杂度在点云规模下不可行（24GB显存限制）。CPA通过分段池化压缩维度，在保持判别性模式的同时实现计算效率。Z-order和Hilbert-order序列化保持空间局部性。

#### 4. Noise-to-Coarse（N2C）模块

**功能**：通过分层蒸馏结构先验，从带噪点云直接生成粗糙去噪点云。

**核心流程**：
1. **初始特征提取**：用MSSC分别提取输入和噪声点云的特征 $F_0, F_{noise}$
2. **层次种子生成**：N阶段FPS下采样 + CPA逐层精炼→输出全局种子 $P_{seed}, F_{seed}$
3. **粗重建**：用PointNet融合噪声坐标与种子特征，最后一个CPA模块回归粗糙坐标和特征

#### 5. Refine模块

**功能**：从粗糙输出恢复结构完整性和几何细节精度。

**核心思路**：从种子集检索区域特征 → PointNet生成关系感知value → CPA精炼 → 反卷积上采样。相当于"Coarse-to-Fine"的第二级。

### 损失函数 / 训练策略

使用Chamfer Distance直接训练，ground truth下采样至180,000点：

$$L_{CD}(P, \hat{P}) = \frac{1}{|P|}\sum_{x \in P}\min_{y \in \hat{P}}\|x-y\|_2^2 + \frac{1}{|\hat{P}|}\sum_{y \in \hat{P}}\min_{x \in P}\|y-x\|_2^2$$

Coarse和Refine两阶段独立训练。仅在SemanticKITTI上训练，不在KITTI-360上微调。

## 实验关键数据

### 主实验

**SemanticKITTI数据集：**

| 方法 | CD↓ | JSD 3D↓ | JSD BEV↓ | IoU(0.5m)↑ | IoU(0.2m)↑ | IoU(0.1m)↑ |
|---|---|---|---|---|---|---|
| LMSCNet | 0.641 | - | 0.431 | 30.83 | 12.09 | 3.65 |
| LiDiff | 0.434 | 0.564 | 0.444 | 31.47 | 16.79 | 4.67 |
| LiDPM | 0.446 | 0.532 | 0.440 | 34.09 | 19.45 | 6.27 |
| ScoreLiDAR | 0.406 | - | 0.425 | - | - | - |
| **LiNeXt** | **0.214** | **0.494** | **0.336** | **41.07** | 19.45 | 6.30 |
| LiDiff† | 0.376 | 0.573 | 0.416 | 32.43 | 22.99 | 13.40 |
| LiDPM† | 0.376 | 0.542 | 0.403 | 36.59 | 25.76 | 14.93 |
| **LiNeXt†** | **0.149** | **0.481** | **0.331** | **41.97** | **31.25** | **15.09** |

**效率对比：**

| 方法 | CD↓ | 推理时间(s) | 参数(M) |
|---|---|---|---|
| LiDiff | 0.434 | 33.359 | 32.67 |
| LiDPM | 0.446 | 15.288 | 32.67 |
| ScoreLiDAR | 0.406 | 5.047 | 32.67 |
| **LiNeXt** | **0.214** | **0.167** | **1.99** |
| LiDiff† | 0.376 | 33.531 | 54.40 |
| **LiNeXt†** | **0.149** | **0.434** | **2.10** |

核心数据：相比LiDiff，LiNeXt实现 **199.8×加速**、**50.7% CD降低**、**6.1%参数量**。

### 消融实验

| 配置 | CD↓ | JSD 3D↓ | JSD BEV↓ | IoU(0.5m)↑ | 说明 |
|---|---|---|---|---|---|
| **LiNeXt** | **0.214** | **0.494** | **0.336** | **41.07** | 完整模型 |
| w/o DSR | 0.215 | 0.508 | 0.352 | 40.00 | 全局形状一致性下降 |
| w/o MSSC | 0.221 | 0.502 | 0.350 | 39.87 | 精细结构恢复能力下降 |
| w/o CPA | 0.227 | 0.504 | 0.353 | 39.36 | 性能下降最大，层次特征聚合关键 |

### 关键发现

1. **非扩散方法可以大幅超越扩散方法**：LiNeXt不仅更快，而且补全质量更高（CD从0.434→0.214），打破了"扩散模型质量更高"的刻板印象
2. **跨数据集泛化能力强**：在SemanticKITTI上训练的模型直接在KITTI-360上测试，LiNeXt†保持CD=0.149不变，而LiDiff†从0.376退化到0.517
3. **CPA是最关键模块**：移除CPA导致最大性能下降（CD +6.1%），证明层次特征聚合对场景补全至关重要
4. **DSR策略影响全局**：虽然CD变化小，但JSD和IoU指标明显下降，说明均匀分布对全局形状一致性很重要

## 亮点与洞察

1. **挑战了扩散模型在生成任务中的必要性**：在LiDAR补全任务中，直接回归方法不仅更快，质量也更高。这表明扩散模型在某些结构化任务中并非最优选择。
2. **距离感知设计**符合LiDAR的物理特性：简单但有效，"近少远多"的重复策略几乎免费地提升了性能。
3. **分段最大池化（SSMP）**是一个巧妙的注意力压缩技巧：在保持判别性的同时大幅降低计算复杂度。
4. **1.99M参数**的极致轻量化设计适合嵌入式部署，对自动驾驶的实际应用非常重要。

## 局限与展望

1. 消融实验中CPA只能部分替换（Refine模块完全替换会超24GB显存），完整消融结果可能不同
2. 仅在SemanticKITTI和KITTI-360上评测，缺少更大规模数据集（如Waymo、nuScenes）
3. 当前仅处理静态场景，不考虑动态对象
4. IoU在细粒度（0.1m）上的优势不明显，精细结构恢复仍有提升空间

## 相关工作与启发

- **LiDiff**（扩散基准）：DDPM框架的局部引导扩散，是本文的主要比较对象
- **ScoreLiDAR**：通过知识蒸馏加速扩散采样5倍，但仍远慢于LiNeXt
- **SnowflakeNet**：点云补全中的反卷积上采样，被LiNeXt的Refine模块采用
- 启发："正确的归纳偏置 + 轻量设计"往往能超越"通用但重量级的生成模型"

## 评分

- 新颖性: ⭐⭐⭐⭐ — 挑战扩散范式的勇气值得赞赏，DSR和CPA设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 双数据集+跨数据集泛化+效率对比+消融完善
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，结构清晰，Figure 1非常直观
- 价值: ⭐⭐⭐⭐⭐ — 199.8倍加速对实际部署意义重大，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)
- [Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [HD2-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving](hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)
- [Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)

<!-- RELATED:END -->
