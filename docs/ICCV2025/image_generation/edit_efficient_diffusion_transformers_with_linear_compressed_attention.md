---
title: >-
  [论文解读] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention
description: >-
  [图像生成] EDiT 提出线性压缩注意力机制，通过 ConvFusion 增强 query 的局部信息并用 Spatial Compressor 压缩 key/value token，实现对 DiT 和 MM-DiT 的高效加速（最高 2.2 倍），同时保持可比的图像质量。
tags:
  - 图像生成
---

# EDiT: Efficient Diffusion Transformers with Linear Compressed Attention

## 基本信息

- **会议**: ICCV 2025
- **arXiv**: 2503.16726
- **代码**: 未公开
- **领域**: 图像生成 (Image Generation)
- **关键词**: 扩散Transformer, 线性注意力, 高效推理, 多模态DiT, 知识蒸馏

## 一句话总结

EDiT 提出线性压缩注意力机制，通过 ConvFusion 增强 query 的局部信息并用 Spatial Compressor 压缩 key/value token，实现对 DiT 和 MM-DiT 的高效加速（最高 2.2 倍），同时保持可比的图像质量。

## 研究背景与动机

Diffusion Transformers (DiTs) 已成为文本到图像生成的主流架构（FLUX, PixArt-Σ, SD3 等），但标准缩放点积注意力的二次复杂度严重限制了：

**高分辨率生成**：token 数量随分辨率增长，4K/8K 图像生成的计算量极大

**边缘设备部署**：手机等资源受限设备难以运行

现有加速方案的局限：
- **SANA**：线性注意力 + MixFFN 卷积模块，但 MixFFN 引入额外计算
- **LinFusion**：多层非线性变换映射 token，但在 DiT 上效果不佳（作者及原文均证实）
- **PixArt-Σ 的 KV 压缩**：仅对 key/value 做深度卷积压缩，仍保持二次复杂度

更重要的是，**多模态 DiT（MM-DiT）** 将图像和文本 token 在联合注意力中处理，token 数更多，效率问题更加突出，但此前**无有效的线性化方案**。

## 方法详解

### 整体框架

EDiT 由两部分组成：
- **EDiT**：用于常规 DiT 的线性压缩注意力
- **MM-EDiT**：用于 MM-DiT 的混合注意力（线性 + 缩放点积）

### ConvFusion：增强 Query 的局部信息

将 LinFusion 的逐 token 处理和 SANA 的 MixFFN 统一为多层卷积映射：

$$Q^{\text{EDiT}} = \phi_{\text{CF}}(X) = \text{ReLU}(\text{Linear}(X + \text{Conv}(\text{LeakyReLU}(\text{GN}(\text{Conv}(X))))))$$

关键设计：
- 将 1D 序列 reshape 为 2D 图像形状后应用 2D 卷积，利用图像的空间局部性
- 第一个卷积（3×3）沿通道维压缩，第二个卷积（1×1）恢复通道数
- 使用 Group Normalization 提升稳定性

### Spatial Compressor：压缩 Key/Value

使用深度卷积对 key 和 value 进行空间压缩：

$$K^{\text{EDiT}} = \text{ReLU}(\phi_{\text{SC}}(X)), \quad V^{\text{EDiT}} = \phi_{\text{SC}}(X)$$
$$\phi_{\text{SC}}(X) = \text{Conv}(\text{Linear}(X))$$

使用 3×3 深度卷积、stride=2，将 key/value 数量减少 4 倍。与 PixArt-Σ 不同的是，将其集成到线性注意力中（而非二次注意力），进一步提升效率。

### 线性注意力公式

将 ConvFusion 的 query 与 Spatial Compressor 的 key/value 插入标准线性注意力：

$$y_i = \frac{Q_i \sum_{j=1}^{N}(K_j^T V_j)}{Q_i \sum_{j=1}^{N} K_j}$$

### MM-EDiT：多模态混合注意力

核心观察：联合注意力中，$Q_I K_I^T$（图像-图像注意力）的计算量最大且随分辨率二次增长。

混合策略：
- **图像-图像关系**：使用 EDiT 线性压缩注意力
- **图像-文本 / 文本-图像 / 文本-文本关系**：保持标准缩放点积注意力

$$\mathbf{A}^{\text{Hybrid}} = \begin{pmatrix} \eta_I^{\text{Lin}} \cdot \mathbf{A}^{\text{Lin}}(Q_I, K_I, V_I) + (1-\eta_I^{\text{Lin}}) \cdot \mathbf{A}(Q_I, K_P, V_P) \\ \eta_P \cdot \mathbf{A}(Q_P, K_I, V_I) + (1-\eta_P) \cdot \mathbf{A}(Q_P, K_P, V_P) \end{pmatrix}$$

归一化因子近似为 token 数量比：$\hat{\eta}^{\text{Lin}} = \frac{N_I}{N_I + N_T}$，避免了自定义 attention kernel 的实现复杂性，且实验证明该近似甚至略优于精确计算。

### 训练策略

采用知识蒸馏：
- **任务损失**：噪声预测 / rectified flow 损失
- **知识蒸馏**：最小化 student 和 teacher 预测之间的差异
- **特征蒸馏**：对齐每层自注意力输出

多阶段升分辨率训练：512 → 1024 → 2048。

## 实验关键数据

### 主实验：EDiT vs PixArt-Σ 及其他线性方案

| Method | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ |
|---|---|---|---|---|---|---|
| | **512×512** | | | **1024×1024** | | |
| PixArt-Σ (teacher) | 0.285 | 7.57 | 2.50 | 0.285 | 7.09 | 2.53 |
| **EDiT (ours)** | 0.283 | **7.06** | 2.57 | 0.290 | 7.82 | 2.64 |
| SANA-DiT | 0.283 | 8.43 | 3.31 | 0.286 | 9.31 | 3.16 |
| LinFusion-DiT | 0.289 | 15.87 | 5.98 | 0.283 | 44.66 | 11.01 |
| KV Comp. (k=2) | 0.275 | 10.69 | 3.77 | 0.283 | 10.32 | 3.50 |

EDiT 在 512 分辨率下 FID 甚至优于 teacher（7.06 vs 7.57）。LinFusion-DiT 在 DiT 上严重失效（44.66 FID）。

### 消融实验：Query/Key/Value 处理方式

| Q | K | V | FID (512) ↓ | FID (1024) ↓ |
|---|---|---|---|---|
| CF | SC | SC | **7.06** | **7.82** |
| CF | CF | - | 7.59 | 7.76 |
| - | SC | SC | 14.53 | 26.59 |
| CF | - | - | 7.06 | 7.70 |

仅使用 Spatial Compressor 而不用 ConvFusion（第三行）导致严重性能退化，证明 ConvFusion 对 query 局部信息增强至关重要。

### 延迟分析

| Resolution | PixArt-Σ | SANA-DiT | EDiT | 加速比 (vs PixArt) |
|---|---|---|---|---|
| 1024×1024 | 0.047s | 0.043s | **0.034s** | 1.4× |
| 2048×2048 | 0.387s | 0.166s | **0.121s** | 3.2× |
| 4096×4096 | 4.770s | 0.687s | **0.461s** | 10.3× |
| 8192×8192 | 72.96s | 21.76s | **1.693s** | **43×** |

8K分辨率下 EDiT 比 PixArt-Σ 快 43 倍。在三星 S25 Ultra 手机上，EDiT 也实现了 38% 的延迟降低。

### MM-EDiT vs SD-v3.5M

| Method | Hybrid | CLIP ↑ | FID (Inception) ↓ | FID (CLIP) ↓ |
|---|---|---|---|---|
| SD-v3.5M (teacher) | – | 0.283 | 10.49 | 3.86 |
| **MM-EDiT (Ours)** | ✓ | 0.285 | 11.60 | 3.91 |
| SANA-MM-DiT | × | 0.279 | 14.94 | 5.02 |
| Linear MM-DiT-α | × | 0.281 | 13.59 | 4.28 |

MM-EDiT 的混合注意力显著优于完全线性化的基线（FID 11.60 vs 13.53-14.94）。

## 亮点与洞察

1. **ConvFusion + Spatial Compressor 的互补设计**：ConvFusion 增强 query 局部信息，Spatial Compressor 压缩 key/value，两者结合实现最优质量-速度平衡
2. **混合注意力的必要性**：在 MM-DiT 中对图像-文本交互保持标准注意力至关重要，完全线性化会严重损害跨模态理解
3. **归一化因子近似**：用 token 数量比近似理论归一化因子不仅更快，还略微提升了性能（反直觉的发现）
4. **分辨率越高加速越显著**：8K 分辨率下 43 倍加速，真正解锁了超高分辨率生成

## 局限性

- 需要知识蒸馏训练，无法即插即用到现有模型
- MM-EDiT 在手机端不如 SANA-MM-DiT 快（因图像-文本交互仍用标准注意力）
- 仅在 PixArt-Σ 和 SD-v3.5M 上验证，未覆盖 FLUX 等最新模型
- 定量评估依赖 FID 和 CLIP Score，缺乏人类偏好评估

## 相关工作与启发

- **SANA** 率先在 DiT 中引入线性注意力，但将卷积放在 FFN 中而非注意力中
- **LinFusion** 的多层变换在 UNet 上有效但在 DiT 上失败，EDiT 的 ConvFusion 解决了这一问题
- **CLEAR** 是唯一其他面向 MM-DiT 的线性化工作，但依赖稀疏邻域注意力，硬件适配性差
- 混合注意力（部分线性 + 部分标准）的思想可推广到其他多模态架构

## 评分

⭐⭐⭐⭐ — 方法设计合理，实验全面覆盖 DiT 和 MM-DiT 两种架构。高分辨率加速效果令人印象深刻（43 倍）。混合注意力设计为 MM-DiT 加速提供了一个实用范式。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](../../CVPR2025/image_generation/dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)

<!-- RELATED:END -->
