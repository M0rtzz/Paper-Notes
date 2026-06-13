---
title: >-
  [论文解读] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images
description: >-
  [ICCV 2025][遥感][遥感基础模型] 提出 SMARTIES，一个统一的传感器无关遥感基础模型，通过光谱感知投影将异构传感器数据映射到共享空间，结合跨传感器 token 混合和掩码重建进行自监督预训练，在单模态和多模态任务上超越专用传感器模型，并可泛化到预训练未见过的传感器。 遥感数据来自多种传感器（光学、SAR、…
tags:
  - "ICCV 2025"
  - "遥感"
  - "遥感基础模型"
  - "多传感器"
  - "光谱感知"
  - "掩码自编码器"
  - "传感器无关表示"
---

# SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images

**会议**: ICCV 2025  
**arXiv**: [2506.19585](https://arxiv.org/abs/2506.19585)  
**代码**: [https://gsumbul.github.io/SMARTIES](https://gsumbul.github.io/SMARTIES)  
**领域**: 遥感  
**关键词**: 遥感基础模型, 多传感器, 光谱感知, 掩码自编码器, 传感器无关表示

## 一句话总结

提出 SMARTIES，一个统一的传感器无关遥感基础模型，通过光谱感知投影将异构传感器数据映射到共享空间，结合跨传感器 token 混合和掩码重建进行自监督预训练，在单模态和多模态任务上超越专用传感器模型，并可泛化到预训练未见过的传感器。

## 研究背景与动机

遥感数据来自多种传感器（光学、SAR、热红外等），光谱范围、辐射分辨率、空间分辨率差异巨大。现有深度学习模型面临：

**传感器特定模型**：为每个传感器单独训练，无法跨传感器迁移

**多传感器基础模型的局限**：
   - 双/三模态模型（如 CROMA、SkySense）使用传感器特定编码器，增加传感器需要修改架构，计算开销大
   - 动态权重方法（如 DOFA）需要超网络和海量预训练数据（800万图像），扩展性有限

**泛化瓶颈**：在固定传感器组合上训练会产生偏置，无法迁移到未见传感器

**核心思路**：所有遥感传感器实质上都在采集电磁频谱的不同子集，可以基于波长范围定义统一的投影层，将不同传感器映射到共享的光谱感知空间。

## 方法详解

### 整体框架

SMARTIES 由四部分组成：
1. 光谱感知图像投影：将不同传感器数据投影到共享空间
2. 跨传感器 Token 混合：交换来自不同传感器的 token 打破传感器偏置
3. 光谱感知图像重建：用标准 ViT 编码器-解码器做掩码图像建模
4. 下游迁移到多样传感器：包括未见传感器的插值适配

### 关键设计

1. **光谱感知投影层**：根据波长范围定义 17 个投影层 $\mathcal{F} = \{f_1, ..., f_{17}\}$，其中 $f_1$-$f_{12}$ 对应 Sentinel-2 的 12 个波段，$f_{13}$-$f_{15}$ 对应 Maxar RGB，$f_{16}$-$f_{17}$ 对应 Sentinel-1 SAR。每个投影层 $f_i: \mathbb{R}^{S \times S} \to \mathbb{R}^D$ 为全连接层。对给定传感器图像的每个 patch，用其波段对应的投影层分别嵌入后取平均，再乘 $C_{\text{max}}=12$ 缩放以避免不同波段数的传感器间失衡。添加新传感器仅需添加新的投影层。

2. **跨传感器 Token 混合**：输入一对来自不同传感器但同一地区的配对图像 $(\mathbf{I}_a, \mathbf{I}_b)$，通过二值掩码 $\mathcal{M}$ 交换 token：
    $\mathbf{T}_{a'} = \mathcal{M} \odot \mathbf{T}_a + (1-\mathcal{M}) \odot \mathbf{T}_b$
   同时做镜像混合保留所有信息。这防止模型对特定光谱组合产生偏置。

3. **未见传感器的插值迁移**：对新传感器的未见波段，如果其中心波长 $\lambda_n^c$ 落在已学习层 $f_i$ 和 $f_j$ 的中心波长之间，通过距离加权平均组合两个投影层的输出。限制条件：仅适用于预训练频谱范围内的插值，不支持外推。

### 损失函数 / 训练策略

- 自监督 MAE 损失：对混合后的两组 patch 分别计算掩码区域的 MSE 重建损失
  $$\mathcal{L} = \mathcal{L}_{a'} + \mathcal{L}_{b'}, \quad \mathcal{L}_{a'} = \frac{\sum(\mathbf{P}_{a'}^{\text{mask}} - \hat{\mathbf{P}}_{a'}^{\text{mask}})^2}{R \cdot N_W N_H}$$
- 预训练仅 496K 图像（60K fMoW RGB-S2 对 + 188K BigEarthNet S1-S2 对），300 epochs
- ViT-B/L 骨干，AdamW 优化器，batch size 2048，8 张 A100 GPU
- 掩码率 75%，混合率 50%，输入 224×224

## 实验关键数据

### 主实验 (表格)

BigEarthNet 多标签场景分类（10% 训练数据，mAP）：

| 方法 | 骨干 | S1 (LP) | S2 (FT) | MM (LP) |
|------|------|---------|---------|---------|
| SatMAE (S2) | ViT-B | 68.4 | 85.9 | 77.8 |
| SpectralGPT | ViT-B | 57.1 | 85.6 | 68.5 |
| CROMA | ViT-B×2 | 79.8 | 87.6 | 85.2 |
| **SMARTIES** | **ViT-B** | **78.9** | **86.9** | **85.4** |
| **SMARTIES** | **ViT-L** | **80.5** | **87.7** | **86.7** |

EuroSAT 场景分类 Top-1 准确率：

| 方法 | LP | FT |
|------|-----|-----|
| SatMAE (S2) ViT-B | 96.6 | 99.2 |
| CROMA ViT-B×2 | 97.6 | 99.2 |
| **SMARTIES ViT-B** | **98.4** | **99.4** |
| **SMARTIES ViT-L** | **98.9** | **99.6** |

语义分割 PANGAEA 基准（冻结骨干 UPerNet, mIoU）：

| 方法 | BurnScars | DEN | SpaceNet7 |
|------|-----------|-----|-----------|
| CROMA | 81.8 | 38.3 | 59.9 |
| DOFA | 80.6 | 39.3 | 61.8 |
| **SMARTIES** | **82.8** | **38.5** | **62.2** |

### 消融实验 (表格)

跨传感器 Token 混合消融（EuroSAT kNN, 50 epochs 预训练）：

| 设置 | kNN 准确率 |
|------|-----------|
| 无混合 | 91.0 |
| 混合（仅 BEN） | 91.1 |
| 混合（完整） | **93.2** |

多模态融合策略（BEN-MM LP, mAP）：

| 策略 | 1% 数据 | 10% 数据 |
|------|---------|----------|
| Image Stacking | 75.9 | 83.1 |
| Feature Concatenation | 77.0 | 84.7 |
| **Mixup Concatenation** | **79.2** | **86.7** |

### 关键发现

- **单一模型超越传感器特定模型**：SMARTIES 用一个 ViT-B 在 SAR（S1）和光学（S2）任务上同时超越各自的专用模型
- **数据高效**：仅 496K 预训练图像，远少于 DOFA 的 800 万和 CROMA 的大规模数据
- **未见传感器泛化**：对 Landsat-8 热红外波段（预训练未见），通过投影插值后冻结骨干即达到 50.2 mIoU，超越从头训练的 U-Net (47.7) 和 DeepLabV3+ (48.5)
- **多尺度鲁棒**：在多尺度评估中超越专门设计的 Scale-MAE 和 Cross-Scale MAE
- 跨传感器 token 混合带来的多模态融合增益为 +2.2% mAP

## 亮点与洞察

- **基于物理的设计哲学**：利用电磁频谱的连续性和传感器波段的物理对应关系来统一表示，比纯数据驱动更有原则
- **极简但有效**：相比需要超网络（DOFA）或传感器特定编码器（CROMA）的复杂架构，SMARTIES 仅增加轻量投影层（+5.9M 参数），可保持与 vanilla MAE 相近的计算复杂度
- **添加新传感器零成本**：只需定义对应波长范围的新投影层，无需修改骨干架构
- **跨传感器 token 混合**思路简洁有效，启发了其他多模态学习场景

## 局限与展望

- 投影插值仅限于预训练频谱范围内（不支持外推），如 X 波段雷达等频段可能需要额外学习
- 未涉及时序建模，对遥感变化检测等需要多时相分析的任务尚需扩展
- 仅关注振幅相关物理量，排除了 SAR 相位信息（如 InSAR）
- 各波段使用独立全连接投影层，未利用相邻波段的光谱连续性
- 预训练场景以欧洲为主（BigEarthNet），对其他地理区域的泛化需进一步验证

## 相关工作与启发

- MAE 系列（SatMAE、SpectralGPT、S2MAE）是遥感 SSL 的主流框架
- CROMA 和 SkySense 是多模态遥感 FM 的代表但依赖传感器特定编码器
- DOFA 用超网络动态生成权重，但需海量数据和复杂架构
- SMARTIES 的光谱感知投影思路对其他多模态学习（如医学成像中的多序列 MRI）有启发

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 光谱感知空间和跨传感器 token 混合是遥感 FM 的重要范式突破
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个数据集，涵盖分类/分割/多尺度/未见传感器等全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，设计动机与物理原理结合紧密
- 价值: ⭐⭐⭐⭐⭐ 为遥感多传感器统一建模提供了高效且可扩展的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[ECCV 2024\] Masked Angle-Aware Autoencoder for Remote Sensing Images](../../ECCV2024/remote_sensing/masked_angle-aware_autoencoder_for_remote_sensing_images.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[CVPR 2025\] Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing](../../CVPR2025/remote_sensing/think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)

</div>

<!-- RELATED:END -->
