---
title: >-
  [论文解读] CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model
description: >-
  [ECCV 2024][3D视觉][单图3D生成] 提出卷积重建模型 CRM，利用 triplane 与六个正交视图之间的空间对齐先验，用 U-Net 替代 Transformer 直接从六视图映射到 triplane，结合 FlexiCubes 端到端训练，10 秒内从单张图像生成高保真纹理网格，训练成本仅为 LRM 的 1/8。
tags:
  - ECCV 2024
  - 3D视觉
  - 单图3D生成
  - 卷积重建
  - Triplane
  - FlexiCubes
  - 多视角扩散
---

# CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model

**会议**: ECCV 2024  
**arXiv**: [2403.05034](https://arxiv.org/abs/2403.05034)  
**代码**: 有 (https://github.com/thu-ml/CRM)  
**领域**: 3D视觉  
**关键词**: 单图3D生成, 卷积重建, Triplane, FlexiCubes, 多视角扩散

## 一句话总结

提出卷积重建模型 CRM，利用 triplane 与六个正交视图之间的空间对齐先验，用 U-Net 替代 Transformer 直接从六视图映射到 triplane，结合 FlexiCubes 端到端训练，10 秒内从单张图像生成高保真纹理网格，训练成本仅为 LRM 的 1/8。

## 研究背景与动机

Feed-forward 3D 生成模型（如 LRM）展示了极快的生成速度，但存在以下问题：

**Transformer 架构未利用几何先验**：LRM 系列方法用 Transformer 生成 triplane patches，但没有利用 triplane 与输入图像之间的空间对齐关系

**3D 数据稀缺**：最大 3D 数据集 Objaverse 仅百万级，远小于 LAION 的 50 亿图像，因此在架构中融入先验知识尤为重要

**非端到端训练**：使用 NeRF 或 Gaussian Splatting 作为表示的方法需要额外后处理步骤来获取纹理网格

**训练成本高**：LRM 需要 batch size 1024 和大量 GPU 资源

**核心观察**：triplane 的可视化与六个正交视图（前后左右上下）存在**天然的空间对齐关系** — 轮廓和纹理自然对齐。这启发了用卷积 U-Net（具有强像素对齐能力）替代 Transformer。

## 方法详解

### 整体框架

CRM 的推理流程（约 10 秒）：

1. 输入单张图像 → **多视角扩散模型**生成六个正交视图（~5s）
2. 另一个扩散模型生成**正则坐标图（CCM）**（~1s）
3. 六视图 + CCM → **卷积 U-Net** → rolled-out triplane → MLP 解码 → **FlexiCubes** → 纹理网格（~4s）

### 关键设计

#### 1. 六正交视图的空间对齐

关键 insight：triplane 的三个平面（xy, xz, yz）分别与对应方向的正交视图空间对齐。因此：

- 选择六个正交视图（前/后/左/右/上/下）作为重建输入，天然对应 triplane 结构
- 六张图按位置排列为两组，每组三张拼成 256×768 图像，共 4 组拼接为 12 通道输入
- U-Net 直接将此输入映射到 rolled-out triplane（展开的 triplane）

#### 2. 卷积 U-Net 替代 Transformer

使用像素级对齐的 **U-Net 架构**而非 Transformer：

- 通道数配置：[64, 128, 128, 256, 256, 512, 512]
- 在分辨率 [32, 16, 8] 处加入 attention blocks
- 约 300M 参数

优势：
- **带宽更大**：U-shape 结构在保留输入信息方面优于 Transformer，产生更精细的 triplane 特征
- **训练收敛极快**：仅 280 次迭代（20 分钟）就出现合理重建结果
- **训练效率高**：batch size 仅需 32（LRM 需要 1024），8 卡 A800 训 6 天
- **总训练成本**仅为 LRM 的 **1/8**

#### 3. 正则坐标图（CCM）

CCM 包含每个像素在标准空间中的 3D 坐标（3 通道，值域 [0,1]），提供重要的几何信息。

- 由第二个扩散模型生成（以六视图为条件）
- 与 RGB 图像拼接后送入 U-Net
- 消融实验证明：**没有 CCM 输入时几何质量显著下降**，特别是复杂几何

#### 4. FlexiCubes 端到端训练

- 使用 FlexiCubes（网格大小 80）替代 NeRF/Gaussian Splatting
- 通过 dual marching cubes 在训练中直接提取网格
- MLP 解码 triplane 特征为 SDF、形变、权重和颜色
- 实现了以纹理网格为最终输出的**端到端训练**

#### 5. 多视角扩散模型的训练增强

- 基于 ImageDream 微调，扩展为 6 视图
- **Zero-SNR**：解决采样初始噪声与最噪训练样本的差异
- **随机缩放**：防止模型总是生成占满整个图像的物体
- **轮廓增强**：随机改变轮廓颜色，防止背面颜色过度依赖输入轮廓

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{MSE}(x, x^{GT}) + \lambda_{LPIPS}\mathcal{L}_{LPIPS}(x, x^{GT}) + \lambda_{depth}\mathcal{L}_{MSE}(x_{depth}, x_{depth}^{GT}) + \lambda_{mask}\mathcal{L}_{MSE}(x_{mask}, x_{mask}^{GT}) + \lambda_{reg}\mathcal{L}_{reg}$$

- $\lambda_{LPIPS}{=}0.1$, $\lambda_{depth}{=}0.5$, $\lambda_{mask}{=}0.5$, $\lambda_{reg}{=}0.005$
- 每个 shape 随机采样 8 个视角（共 16 个）进行监督
- 小高斯噪声加到输入上增强对多视图不一致的鲁棒性
- 重建模型训 110K 步，扩散模型训 10K 步（梯度累积 12 步，有效 batch=1536）

## 实验关键数据

### 几何质量（GSO 数据集）

| 方法 | Chamfer Dist.↓ | Vol. IoU↑ | F-Score (%)↑ |
|---|---|---|---|
| One-2-3-45 | 0.0172 | 0.4463 | 72.19 |
| SyncDreamer | 0.0140 | 0.3900 | 75.74 |
| Wonder3D | 0.0186 | 0.4398 | 76.75 |
| LGM | 0.0117 | 0.4685 | 68.69 |
| **CRM (Ours)** | **0.0094** | **0.6131** | **79.38** |

### 纹理质量（GSO 数据集）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-Sim↑ |
|---|---|---|---|---|
| OpenLRM | 14.30 | 0.8294 | 0.2276 | 84.20 |
| Magic123 | 12.69 | 0.7984 | 0.2442 | 85.16 |
| LGM | 13.28 | 0.7946 | 0.2560 | 85.20 |
| **CRM (Ours)** | **16.22** | **0.8381** | **0.2143** | **87.55** |

### 多视角扩散质量

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| SyncDreamer | 20.30 | 0.7804 | 0.2932 |
| Wonder3D | 23.76 | 0.8127 | 0.2210 |
| **CRM (Ours)** | **29.36** | **0.8721** | **0.1354** |

### 消融实验

**CCM 影响**：没有 CCM 输入时几何明显退化，尤其复杂几何（如动物）。

**多视角扩散训练技巧**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| ImageDream (6 view) | 28.99 | 0.8565 | 0.1497 |
| + Zero-SNR | 29.13 | 0.8598 | 0.1498 |
| + Random Resizing | **29.36** | **0.8721** | **0.1354** |

### 关键发现

1. CRM 在所有几何和纹理指标上全面超越所有基线
2. 训练成本仅为 LRM 的 1/8（8 卡 6 天 vs LRM）
3. 仅 280 迭代（20 分钟）就出现合理重建，说明空间对齐先验极大加速收敛
4. 多视角扩散的 PSNR 比 Wonder3D 高 5.6 分

## 亮点与洞察

1. **Triplane 空间对齐先验**：最大的 insight — 把正确的先验融入架构比堆算力更有效
2. **U-Net > Transformer（对此任务）**：卷积本身的归纳偏置比通用 Transformer 在像素对齐任务上更合适
3. **端到端网格输出**：FlexiCubes 避免了 NeRF→mesh 的后处理失真
4. **轮廓增强技巧**：虽然不改善定量指标，但大幅提升 in-the-wild 输入的鲁棒性
5. **极快收敛**：20 分钟训练即可得到合理结果，说明先验发挥了巨大作用

## 局限与展望

1. 多视角扩散模型无法保证完全一致，不一致图像会降低 3D 质量
2. FlexiCubes 网格分辨率仅 80，限制了超精细几何细节
3. 对大仰角或非标准 FoV 输入效果有限（继承自 ImageDream）
4. 六视图固定为正交视角，可能不是所有物体的最优视角选择

## 相关工作与启发

- **LRM**：Transformer 架构的 triplane 生成开创者，但未利用空间对齐先验
- **LGM**：用 Gaussian Splatting 表示，但需额外步骤转换为网格
- **SyncDreamer/Wonder3D**：多视图一致性生成，但需测试时优化重建
- 启发：**在数据稀缺时，将领域先验编入架构** 比数据增强或模型增大更有效

## 评分

| 维度 | 分数 (1-10) |
|---|---|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 写作质量 | 8 |
| 实用价值 | 9 |
| **总分** | **8.0** |

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)
- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](../../CVPR2025/3d_vision/mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)

<!-- RELATED:END -->
