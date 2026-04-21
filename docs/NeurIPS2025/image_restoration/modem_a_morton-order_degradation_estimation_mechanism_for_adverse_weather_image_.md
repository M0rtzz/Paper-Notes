---
title: >-
  [论文解读] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration
description: >-
  [NeurIPS 2025][图像恢复][Adverse Weather Restoration] 提出 MODEM 框架，通过 Morton 编码空间扫描与选择性状态空间模型（SSM）结合，建模空间异质性天气退化特征，配合双重退化估计模块提供全局和局部先验，实现多种天气退化图像的统一自适应复原 SOTA。
tags:
  - NeurIPS 2025
  - 图像恢复
  - Adverse Weather Restoration
  - State Space Model
  - Morton Order
  - Degradation Estimation
  - Mamba
---

# MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration

**会议**: NeurIPS 2025  
**arXiv**: [2505.17581](https://arxiv.org/abs/2505.17581)  
**代码**: 即将开源  
**领域**: 图像复原  
**关键词**: Adverse Weather Restoration, State Space Model, Morton Order, Degradation Estimation, Mamba

## 一句话总结

提出 MODEM 框架，通过 Morton 编码空间扫描与选择性状态空间模型（SSM）结合，建模空间异质性天气退化特征，配合双重退化估计模块提供全局和局部先验，实现多种天气退化图像的统一自适应复原 SOTA。

## 研究背景与动机

恶劣天气下的图像复原是计算机视觉的重要问题，其核心困难在于**天气退化的高度非均匀性和空间异质性**：雾霾表现为平滑的强度衰减，而雨丝和雪花则是局部的尖锐遮挡。

**现有方法的局限**：
- **任务特定方法**：为每种天气训练独立模型（去雨、去雾、去雪），缺乏可扩展性
- **统一框架**（TransWeather、Histoformer 等）：虽然能在单一模型中处理多种天气，但仍缺乏**显式的退化估计机制**来建模细粒度的空间变化退化模式

**核心 insight**：将图像退化视为状态空间中的潜变量演化过程。每个像素的退化特征是一个"状态"，这些状态沿空间演化，受局部（如雨丝）和非局部（如漂移雾）模式共同控制。这与状态空间模型（SSM）的隐状态递推天然对应。

**关键联系**：SSM 公式 $y_k = CA h_{k-1} + CB x_k$ 中，$CA h_{k-1}$ 建模长程退化上下文（如全局雾层），$CB x_k$ 捕捉局部退化细节（如雨滴）。

## 方法详解

### 整体框架

MODEM 采用两阶段训练策略：
- **Stage 1**：DDEM 同时接收退化图像 $I_{LQ}$ 和真值 $I_{GT}$，学习退化映射；主干网络仅接收 $I_{LQ}$ 配合退化先验复原
- **Stage 2**：DDEM 仅接收 $I_{LQ}$（Stage 1 的冻结 DDEM 提供监督），实现推理时的纯退化估计

主干网络由 $N$ 个 MDSL（MOS2D Degradation-Aware Layer）堆叠而成。

### 关键设计

1. **Morton-Order 2D-Selective-Scan (MOS2D)**：
   传统 SSM 处理图像时使用光栅扫描（逐行），但这会破坏空间局部性——空间上相邻的像素在 1D 序列中可能距离很远。作者提出使用 Morton 编码（Z-order curve）进行扫描：通过交错坐标 $(i,j)$ 的二进制位 $z = \text{interleave}(i,j)$，将 2D 特征展开为保局部性的 1D 序列。这使 SSM 能在结构化遍历中同时捕捉局部和长程依赖。

2. **Dual Degradation Estimation Module (DDEM)**：
   从输入提取退化特征 $F$ 后，生成两种互补的退化先验：
    - **全局退化描述符** $Z_0 = \sigma(\text{Linear}(\text{MLP}(\text{AvgPool}(F)))) \in \mathbb{R}^{C_d}$：编码天气类型和严重程度
    - **空间自适应退化核** $Z_1 = \text{Conv}(F) \times \text{Conv}(F)^T \in \mathbb{R}^{C_{d1} \times C_{d2}}$：编码局部退化结构和变化

3. **双重退化调制**：

    - **DAFM（退化自适应特征调制）**：用 $Z_0$ 生成通道级自适应权重和偏置，通过 FiLM 调制 $F_{\text{DAFM}} = (Z_0^w \odot F_i) + Z_0^b$
    - **DSAM（退化选择性注意力调制）**：用 $Z_1$ 构造注意力矩阵指导 SSM 的 $B$、$C$、$\Delta$ 参数生成：$F_{\text{DSAM}} = W_F F_{\text{DAFM}} \times \text{Softmax}(W_Z Z_1)$
   SSM 参数由 $F_{\text{DSAM}}$ 动态生成，确保状态演化和输出自适应于退化特征

### 损失函数 / 训练策略

- **Stage 1**：$\mathcal{L}_1 + \mathcal{L}_{cor}$（Pearson 相关系数损失）
- **Stage 2**：$\mathcal{L}_1 + \mathcal{L}_{cor} + \mathcal{L}_{KL}$（KL 散度约束退化表示 $\tilde{Z}$ 的一致性）
- 4×RTX 3090，AdamW + Cosine Annealing Restart
- 在 all-weather 联合数据集上训练

## 实验关键数据

### 主实验（Table 1，统一模型比较）

| 数据集 | 指标 | MODEM | Histoformer | 之前 SOTA | 提升 |
|--------|------|-------|-------------|-----------|------|
| Snow100K-S | PSNR | **38.08** | 37.41 | 36.92 | +0.67 |
| Snow100K-L | PSNR | **32.52** | 32.16 | 31.92 | +0.36 |
| Outdoor-Rain | PSNR | **33.10** | 32.08 | 31.39 | +1.02 |
| RainDrop | PSNR | 33.01 | **33.06** | 32.38 | -0.05 |
| **平均** | PSNR | **34.18** | 33.68 | 33.04 | **+0.50** |

### 消融实验（Table 7）

| 配置 | Outdoor PSNR | RainDrop PSNR | 说明 |
|------|-------------|---------------|------|
| 全部组件 | **33.10** | **33.01** | 完整 MODEM |
| 无 Morton | 32.89 | 32.69 | Morton 扫描对空间建模有贡献 |
| 无 DDEM | 32.37 | 32.38 | 退化估计是核心组件 |
| 无 DAFM | 32.19 | 32.62 | 全局调制对混合退化重要 |
| 无 DSAM | 32.77 | 32.72 | 局部调制对雨滴等局部退化关键 |

### 关键发现

- Outdoor-Rain 上 PSNR 提升达 1.02dB，是最显著的改进，证实了对混合退化（雨+雾）的优势
- 在真实世界雪景测试中无需额外微调即超越 Histoformer，展示了强泛化能力
- T-SNE 可视化显示 MODEM 的特征在不同天气类型间有更好的聚类分离
- 感知指标（LPIPS/Q-Align/MUSIQ）同样达到 SOTA

## 亮点与洞察

- **退化 = 状态空间演化**：将天气退化建模为 SSM 的隐状态递推是一个优雅的类比，理论联系清晰
- **Morton 扫描的局部性保持**：相比 raster/连续/局部扫描，Z-order curve 在 1D 展开后能更好保持 2D 空间邻近性
- **双重退化先验的互补性**：全局先验提供"什么类型的天气"，局部先验提供"这个位置如何退化"

## 局限与展望

- 两阶段训练增加了整体训练成本
- Morton 编码要求图像尺寸为 2 的幂次（实际中需要 padding）
- 在 RainDrop 上略逊于 Histoformer（-0.05dB），说明纯局部退化场景还有改进空间
- 未与最新的扩散模型复原方法进行充分比较

## 相关工作与启发

- 与 Histoformer 的直方图统计先验互补：MODEM 显式估计退化，Histoformer 用全局统计信息
- SSM/Mamba 在图像复原中的应用（MambaIR、FourierMamba）提供了基础，MODEM 的贡献在于退化导向的调制
- Morton 编码来自计算几何，在图像处理中的应用相对新颖

## 评分

- 新颖性: ⭐⭐⭐⭐ 退化状态空间建模视角新颖，Morton 扫描+双重退化调制设计完整
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多指标（PSNR/SSIM/LPIPS/Q-Align/MUSIQ）、任务特定对比、详尽消融
- 写作质量: ⭐⭐⭐⭐ 退化与 SSM 的联系讲解直观，图示丰富
- 价值: ⭐⭐⭐⭐ 多天气统一复原 SOTA，对恶劣天气下的实际视觉系统有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [Robust Adverse Weather Removal via Spectral-based Spatial Grouping (SSGformer)](../../ICCV2025/image_restoration/robust_adverse_weather_removal_via_spectral-based_spatial_grouping.md)
- [Restoring Images in Adverse Weather Conditions via Histogram Transformer](../../ECCV2024/image_restoration/restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](../../CVPR2025/image_restoration/visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)

<!-- RELATED:END -->
