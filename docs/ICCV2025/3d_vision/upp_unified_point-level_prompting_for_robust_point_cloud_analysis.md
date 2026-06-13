---
title: >-
  [论文解读] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis
description: >-
  [ICCV 2025][3D视觉][点云分析] 提出统一点级提示方法UPP，将点云去噪和补全重新定义为下游任务的提示机制，通过Rectification Prompter过滤噪声、Completion Prompter补全缺失、Shape-Aware Unit捕获几何特征，在噪声和不完整点云上以6.3%参数实现超越全量微调的鲁棒分析。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "点云分析"
  - "参数高效微调"
  - "去噪"
  - "补全"
  - "提示学习"
---

# UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis

**会议**: ICCV 2025  
**arXiv**: [2507.18997](https://arxiv.org/abs/2507.18997)  
**代码**: [GitHub](https://github.com/zhoujiahuan1991/ICCV2025-UPP)  
**领域**: 3D视觉  
**关键词**: 点云分析, 参数高效微调, 去噪, 补全, 提示学习

## 一句话总结

提出统一点级提示方法UPP，将点云去噪和补全重新定义为下游任务的提示机制，通过Rectification Prompter过滤噪声、Completion Prompter补全缺失、Shape-Aware Unit捕获几何特征，在噪声和不完整点云上以6.3%参数实现超越全量微调的鲁棒分析。

## 研究背景与动机

预训练点云模型（Point-MAE、ReCon等）在各类下游任务上取得显著进展，但**现实世界采集的点云通常存在大量噪声和不完整性**（物体遮挡、反射表面、传感器分辨率限制），严重削弱模型性能。

**现有方案的问题**：

**专用去噪/补全模型 + 下游任务**（集成范式）：
   - 去噪和补全任务之间**目标冲突**：去噪删除多余点，补全添加缺失点，简单集成会相互干扰
   - 增强任务与下游任务之间存在**域差异**，导致性能不佳
   - 训练流程复杂、计算和存储开销大

**参数高效微调（PEFT）方法**（IDPT、Point-PEFT、DAPT）：
   - 仅在潜在特征空间中提升表示能力
   - **忽略输入点云中噪声和缺陷的显式抑制**
   - 处理低质量数据时特征不可区分，性能严重退化

**UPP的创新**：将去噪和补全重新定义为面向下游任务的提示机制，在**输入数据空间**而非仅特征空间进行干预，统一端到端训练。

## 方法详解

### 整体框架

冻结预训练骨干，插入三个可训练组件：
1. **Rectification Prompter**：浅层blocks后预测修正向量提示，过滤噪声
2. **Completion Prompter**：深层blocks后生成补全点提示，恢复缺失区域
3. **Shape-Aware Unit**：每个block中插入，捕获几何敏感特征

### Rectification Prompter（修正提示器）

给定含噪不完整点云 $\boldsymbol{x} \in \mathbb{R}^{S \times 3}$，编码为 $L$ 个token后通过 $d_r$ 个transformer块提取特征。通过空间插值将稀疏中心特征传播到密集点：

$$\boldsymbol{f}_r = \mathcal{F}(\boldsymbol{h}_{d_r}, \boldsymbol{c}, \boldsymbol{x}) \in \mathbb{R}^{S \times D_r}$$

MLP预测每个点的修正向量 $\boldsymbol{v}_r \in \mathbb{R}^{S \times 3}$，大幅度向量对应低可信度的噪声点，通过阈值 $\tau$ 过滤：

$$\boldsymbol{x}_r = \{\boldsymbol{x} + \boldsymbol{v}_r \cdot \alpha \mid \tau > \|\boldsymbol{v}_r\|\}$$

**训练目标**：噪声点目标是到干净表面的位移，干净点目标为零位移：

$$\mathcal{L}_{\text{rect}} = \frac{1}{S_n}\sum_{i \in \boldsymbol{n}} \|\boldsymbol{v}_r^i - \boldsymbol{v}_{gt}^i\|^2 + \frac{1}{S}\sum_{i \in \boldsymbol{x}} \|\boldsymbol{v}_r^i\|^2$$

### Completion Prompter（补全提示器）

在修正后的点云 $\boldsymbol{x}_r$ 上重新采样和编码，通过 $d_c$ 个blocks后将token下投影拼接为全局特征 $\boldsymbol{f}_c$，预测缺失区域的粗糙中心 $\boldsymbol{c}_m$。

关键设计：**复用MAE预训练解码器**重建局部patch：

$$\boldsymbol{x}_m = \mathcal{D}([\boldsymbol{h}_m + \text{Embed}(\boldsymbol{c}_m), \boldsymbol{h}_{d_c}])$$

最终通过FPS重采样合并修正点和补全点：$\boldsymbol{x}_c = \text{FPS}([\boldsymbol{x}_m, \boldsymbol{x}_r])$

**损失函数**（L1 Chamfer Distance）：

$$\mathcal{L}_{\text{comp}} = \mathcal{C}_1(\boldsymbol{c}_m, \mathcal{P}_m) + \mathcal{C}_1(\boldsymbol{x}_m, \mathcal{P}_m) + \mathcal{C}_1(\boldsymbol{x}_c, \mathcal{P}_{gt})$$

### Shape-Aware Unit

在每个transformer块中插入，包含两个创新：

1. **Shape-Aware Attention**：基于**空间距离**而非特征相似度建立连接，噪声离群点不太可能改变空间邻域关系，因此更鲁棒
2. **低秩适配器**：$\boldsymbol{h}_{i+1} = W_2 \cdot \sigma(W_1(\hat{\boldsymbol{h}}_i)) + \hat{\boldsymbol{h}}_i$，防止特征过平滑

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{rect}} + \mathcal{L}_{\text{comp}} + \mathcal{L}_{\text{task}}$$

采用分阶段优化策略提升训练稳定性。

## 实验

### 噪声点云分类（主实验）

| 方法 | 参考 | 参数(M)↓ | Noisy ModelNet40↑ | Noisy ShapeNet55↑ |
|:---|:---:|:---:|:---:|:---:|
| Point-MAE (FFT) | ECCV22 | 22.1 (100%) | 89.42 | 88.13 |
| +Point-PEFT | AAAI24 | 0.7 (3.2%) | 87.52 (-1.90) | 86.01 (-2.12) |
| +DAPT | CVPR24 | 1.1 (5.0%) | 86.43 (-2.99) | 86.33 (-1.80) |
| **+UPP (Ours)** | — | **1.4 (6.3%)** | **92.95 (+3.53)** | **90.40 (+2.27)** |
| ReCon (FFT) | ICML23 | 43.6 (100%) | 89.67 | 89.01 |
| **+UPP (Ours)** | — | **1.4 (3.2%)** | **91.69 (+2.02)** | **89.68 (+0.67)** |
| Point-FEMAE (FFT) | AAAI24 | 27.4 (100%) | 89.59 | 88.63 |
| **+UPP (Ours)** | — | **1.4 (5.1%)** | **91.94 (+2.35)** | **90.08 (+1.45)** |

UPP在三个骨干上均超越全量微调，且参数量仅为3.2%~6.3%。现有PEFT反而降低性能。

### 真实世界数据（ScanObjectNN）

| 方法 | 参数(M) | Acc.(%) |
|:---|:---:|:---:|
| Point-FEMAE (baseline) | 27.4 | 90.71 |
| +Point-PEFT | 0.7 | 89.16 |
| +DAPT | 1.1 | 89.67 |
| **+UPP (Ours)** | **1.4** | **91.39** |

### 消融实验

| 基础 | Rect. Prompter | Compl. Prompter | SA-Unit | Acc.(%) |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | ✗ | 89.42 |
| ✓ | ✓ | ✗ | ✗ | 90.90 |
| ✓ | ✗ | ✓ | ✗ | 91.36 |
| ✓ | ✗ | ✗ | ✓ | 91.28 |
| ✓ | ✓ | ✓ | ✓ | **92.95** |

三个组件各自贡献1.5~2个百分点，联合使用达到最优。

### 关键发现

1. **PEFT方法的反面效果**：现有3D PEFT方法（Point-PEFT、DAPT）在噪声数据上反而降低性能，因为它们忽略了输入噪声的显式处理
2. **输入空间干预的重要性**：UPP在数据空间而非仅特征空间进行修正/补全，更直接有效
3. **Shape-Aware Attention的鲁棒性**：基于空间距离的注意力比特征相似度更抗噪声干扰
4. **骨干无关性**：UPP在Point-MAE、ReCon、Point-FEMAE三个骨干上均有效

## 亮点与洞察

1. **范式转变**：将去噪/补全从独立前处理变为下游任务的统一提示，消除了域差异和目标冲突
2. **数据空间提示**：不同于VPT等仅在特征空间添加提示token，UPP直接在点坐标空间操作（移动/添加离散点）
3. **预训练解码器复用**：巧妙利用MAE训练后通常被丢弃的解码器权重来进行点云补全

## 局限性

1. 分阶段优化增加了训练复杂度
2. 补全提示器的点数 $M$ 为固定超参数，对不同缺失程度的适应性有限
3. 仅验证了分类任务，分割和检测任务的效果待确认

## 相关工作

- **点云预训练**：Point-MAE, ReCon, Point-FEMAE, PointGPT
- **点云增强**：ScoreDenoise, PoinTr, T-CorresNet
- **3D PEFT**：IDPT, Point-PEFT, DAPT, GAPrompt

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 将去噪/补全统一为提示机制的范式创新
- 技术深度：⭐⭐⭐⭐ — 三组件设计精巧，Shape-Aware Attention有理论分析支撑
- 实验完整性：⭐⭐⭐⭐ — 多骨干、多数据集、充分消融
- 实用价值：⭐⭐⭐⭐ — 参数高效，代码开源，直接提升现有模型鲁棒性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](../../CVPR2025/3d_vision/spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)

</div>

<!-- RELATED:END -->
