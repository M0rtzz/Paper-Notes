---
title: >-
  [论文解读] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields
description: >-
  [ECCV 2024][3D视觉][物体中心学习] 提出SlotLifter，通过将2D特征提升为3D并与Slot Attention结合的slot-guided feature lifting设计，在场景分解和新视角合成上同时达到SOTA，且训练效率提升约5倍。
tags:
  - ECCV 2024
  - 3D视觉
  - 物体中心学习
  - 辐射场
  - 注意力机制
  - 场景分解
  - 新视角合成
---

# SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields

**会议**: ECCV 2024  
**arXiv**: [2408.06697](https://arxiv.org/abs/2408.06697)  
**代码**: [项目页面](https://slotlifter.github.io)  
**领域**: 3D视觉  
**关键词**: 物体中心学习, 辐射场, Slot Attention, 场景分解, 新视角合成

## 一句话总结

提出SlotLifter，通过将2D特征提升为3D并与Slot Attention结合的slot-guided feature lifting设计，在场景分解和新视角合成上同时达到SOTA，且训练效率提升约5倍。

## 研究背景与动机

物体中心学习旨在从视觉场景中无监督提取物体级抽象表示。现有3D物体中心方法（如uORF、COLF）存在：(1) 需要额外辅助损失（对抗损失、LPIPS损失）；(2) 真实场景泛化困难；(3) 计算开销大（OSRT需64个TPUv2训练7天）。核心挑战是如何有效对齐多视角信息并从压缩的slot表示重建3D场景。

## 方法详解

### 整体框架

SlotLifter包含三个阶段：(1) 场景编码——通过Slot Attention提取slot并提升2D特征到3D；(2) 点-slot映射——通过cross-attention分配slot到3D点；(3) 渲染——基于体渲染生成图像和分割掩码。

### 关键设计

**Slot-guided Feature Lifting**: 将输入视角的2D特征图投影到3D点坐标上，得到lifted point feature $\mathbf{F}_{lift}$。对多视角特征计算均值和方差后通过MLP得到3D点特征$\mathbf{F}_p$，添加位置编码保留空间信息。

**Point-slot映射**: 使用cross-attention让3D点特征查询slot表示，引入额外的empty slot处理空白区域。最终通过attention权重$\mathbf{W}_p$得到每个3D点的slot分配和slot聚合点特征$\mathbf{F}_s$。

**Slot-based密度预测**: 直接利用映射模块的attention权重预测密度$\sigma_i = \text{sum}(\mathbf{W}_p^i \odot \text{ReLU}(\mathbf{A}_p^i))$，通过ReLU抑制不相关slot的贡献。

**随机掩码策略**: 对lifted特征随机掩码（余弦退火调度从0.99到0），防止模型退化为仅依赖lifted特征而忽略slot信息。

### 损失函数

仅使用MSE重建损失：$\mathcal{L}_{recon} = \|\mathbf{C}(r) - \hat{\mathbf{C}}(r)\|^2$，无需任何辅助损失。

## 实验关键数据

### 合成场景分解

| 方法 | CLEVR-567 NV-ARI↑ | Room-Chair NV-ARI↑ | Room-Diverse NV-ARI↑ | Room-Texture NV-ARI↑ |
|------|--------------------|--------------------|----------------------|----------------------|
| uORF | 83.8 | 74.3 | 56.9 | 57.8 |
| BO-uORF | 78.4 | 80.9 | 62.5 | 60.4 |
| COLF | 55.8 | 80.7 | 52.5 | 1.1 |
| uOCF-P | - | - | - | 70.4 |
| **SlotLifter** | **87.0** | **89.7** | **77.5** | **79.3** |

### 新视角合成

Room-Texture数据集上的定量对比：

| 方法 | LPIPS↓ | SSIM↑ | PSNR↑ | NV-ARI↑ | FG-ARI↑ |
|------|--------|-------|-------|---------|---------|
| uORF | 0.254 | 0.711 | 24.23 | 57.8 | 9.3 |
| BO-uORF | 0.215 | 0.739 | 25.26 | 60.4 | 35.4 |
| uOCF-P | 0.136 | 0.798 | 28.85 | 70.4 | 56.3 |
| **SlotLifter** | **0.131** | **0.858** | **30.68** | **79.3** | **70.7** |

### 关键发现

- Room-Diverse上ARI提升+15（77.5 vs 62.5），说明feature lifting对复杂场景特别有效
- 仅用1024采样光线和MSE损失即可训练，比uORF快约5倍
- 在真实场景ScanNet和DTU上也展示了有竞争力的新视角合成性能

## 亮点与洞察

1. **简洁优雅的设计**：仅用重建损失、无辅助损失、无额外decoder，却实现了全面SOTA
2. Feature lifting为slot学习提供了更显式的3D指导，避免了纯slot decoder的信息瓶颈
3. 只需1个辐射场（而非K个），计算量大幅降低

## 局限性

- 需要已知相机参数作为输入
- 真实场景中背景分割仍有挑战
- slot数量需要预设，对未知场景适应性有限

## 相关工作与启发

将image-based rendering的思想（如PixelNeRF、IBRNet）引入物体中心学习是一个新颖的交叉点。随机掩码策略对防止特征退化的作用值得在其他多信号学习场景中借鉴。

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
