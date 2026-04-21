---
title: >-
  [论文解读] ZigMa: A DiT-style Zigzag Mamba Diffusion Model
description: >-
  [ECCV 2024][图像生成] ZigMa 提出了一种 DiT 风格的 Zigzag Mamba 扩散模型，通过异构逐层锯齿形扫描方案保持空间连续性，以零参数/显存开销实现优于 Mamba 基线的生成质量，同时相比 Transformer 具备线性复杂度优势。
tags:
  - ECCV 2024
  - 图像生成
---

# ZigMa: A DiT-style Zigzag Mamba Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2403.13802](https://arxiv.org/abs/2403.13802)  
**领域**: 图像生成

## 一句话总结

ZigMa 提出了一种 DiT 风格的 Zigzag Mamba 扩散模型，通过异构逐层锯齿形扫描方案保持空间连续性，以零参数/显存开销实现优于 Mamba 基线的生成质量，同时相比 Transformer 具备线性复杂度优势。

## 研究背景与动机

扩散模型在各种视觉任务中取得了显著进展，但面临两个核心瓶颈：

**Transformer 的二次复杂度**：DiT 等基于 Transformer 的扩散骨干在自注意力机制上存在 $O(M^2)$ 的复杂度瓶颈，尽管有 Flash Attention 等优化，处理长序列 token 时仍受限

**Mamba 的空间连续性问题**：Mamba 作为线性复杂度的状态空间模型（SSM）在 1D 序列建模上优势明显，但现有视觉 Mamba 方法直接按行列主序（row/column-major）将 2D token 展平为 1D，忽略了图像中相邻 patch 的空间连续性

**多方向扫描的参数开销**：VisionMamba 等方法通过在单个 Mamba 块中采用多方向扫描来弥补空间感知不足，但这带来了额外的参数量和 GPU 显存负担

ZigMa 的核心洞察是：**可以将多方向扫描的复杂度分摊到网络的不同层中**，每一层使用不同的锯齿形扫描方案，以零额外参数实现空间连续性。

## 方法详解

### 整体框架

ZigMa 采用 DiT（Adaptive LayerNorm）风格的架构，由 L 层 Zigzag Mamba 块堆叠而成。每层包含：
- Mamba 扫描模块（长序列建模）
- Cross-attention 模块（多模态推理，如文本条件）
- AdaLN 调制（时间步和条件注入）

训练框架基于 Stochastic Interpolant，统一了扩散模型、Flow Matching 和 Normalizing Flow。

### 关键设计

**Zigzag 扫描方案**：
- 传统 sweep 扫描按行展平 2D tokens，在行末到下一行首时发生空间跳跃，破坏了相邻 patch 的连续性
- Zigzag 扫描在每行交替改变扫描方向（如蛇形路径），确保 1D 序列中相邻 token 在 2D 空间中也是相邻的
- 设计了 8 种不同的 zigzag 空间填充方案 $\mathbf{S}_j$（$j \in [0,7]$），涵盖水平/垂直方向的不同起始和走向

**异构逐层扫描**：
- 每一层使用不同的扫描方案：$\Omega_i = \mathbf{S}_{\{i \% 8\}}$
- 在输入扫描前重排 token 顺序，扫描后再恢复原始顺序：
$$z_{\Omega_i} = \text{arrange}(z_i, \Omega_i), \quad \bar{z}_{\Omega_i} = \text{scan}(z_{\Omega_i}), \quad z_{i+1} = \text{arrange}(\bar{z}_{\Omega_i}, \bar{\Omega}_i)$$
- 核心优势：不同于在同一个块中使用 k 个方向（需要 k 倍参数），异构逐层方案将扫描多样性分散到不同层，**零参数零显存额外开销**

**文本条件的 Cross-Attention**：
- 在 Mamba 块之上添加带 skip connection 的 cross-attention 块
- 条件信号（时间步 + 文本提示）通过 MLP 分别调制 Mamba 扫描和 cross-attention

**3D 视频的因式化扫描**：
- 将 3D Zigzag 分解为 2D 空间 Zigzag + 1D 时序扫描
- 采用 "sst" 方案（两次空间 + 一次时序），假设时序维度存在冗余

### 损失函数

基于 Stochastic Interpolant 的速度场估计损失：

$$\mathcal{L}_v(\theta) = \int_0^T \mathbb{E}[\|v_\theta(x_t, t) - \dot{\alpha}_t x_* - \dot{\sigma}_t \varepsilon\|^2] dt$$

采用线性路径：$\alpha_t = 1-t$，$\sigma_t = t$。

## 实验关键数据

### 主实验

**表1：FacesHQ 1024×1024 高分辨率生成（4096 tokens）**

| 方法 | FID↓ | FDD↓ |
|------|------|------|
| VisionMamba | 51.1 | 66.3 |
| **ZigMa** | **37.8** | **50.5** |
| ZigMa (bs×2) | **26.6** | **31.2** |

**表2：MS-COCO 256×256 文本条件生成**

| 方法 | FID↓ |
|------|------|
| Sweep | 195.1 |
| Zigzag-1 | 73.1 |
| VisionMamba | 60.2 |
| **Zigzag-8** | **41.8** |

**表3：UCF101 视频生成**

| 方法 | Frame-FID↓ | FVD↓ |
|------|-----------|------|
| Bidirection | 256.1 | 320.2 |
| 3D Zigzag | 238.1 | 282.3 |
| **Factorized ZigMa** | **216.1** | **210.2** |
| Bidirection (bs×4) | 146.2 | 201.1 |
| **ZigMa (bs×4)** | **121.2** | **140.1** |

### 消融实验

**表4：扫描方案数量消融（MultiModal-CelebA）**

| 方案 | FID↓ (256) | KID↓ (256) | FID↓ (512) | KID↓ (512) |
|------|-----------|-----------|-----------|-----------|
| Sweep | 158.1 | 0.169 | 162.3 | 0.203 |
| Zigzag-1 | 65.7 | 0.051 | 121.0 | 0.113 |
| Zigzag-2 | 54.7 | 0.041 | 96.0 | 0.079 |
| **Zigzag-8** | **45.5** | **0.011** | **34.9** | **0.023** |

**表5：位置编码消融（CelebA 256）**

| 方法 | 无 PE | Cosine PE | Learnable PE |
|------|------|-----------|-------------|
| VisionMamba | 21.33 | 18.47 | 16.38 |
| **ZigMa** | **14.27** | **14.04** | **13.32** |

**表6：与 Transformer 的效率对比（CelebA 256）**

| 方法 | FID↓ | Memory (GB)↓ | FLOPS (G)↓ |
|------|------|-------------|-----------|
| U-ViT | 14.50 | 35.10 | 12.5 |
| DiT | 14.64 | 29.20 | 5.5 |
| **ZigMa** | **14.27** | **17.80** | **5.2** |

### 关键发现

1. 从 Sweep 到 Zigzag-8，FID 从 158.1 降至 45.5（256 分辨率），且在 512 分辨率上增益更显著（162.3→34.9），验证了空间连续性在长序列中的重要性
2. ZigMa 即使不使用位置编码（FID=14.27）也优于 VisionMamba 使用 Cosine PE（18.47），证明 zigzag 扫描本身已内含空间归纳偏置
3. 相比 U-ViT 降低 49% 显存（35.1→17.8 GB），保持可比的生成质量
4. 视频生成中，因式化 3D Zigzag 大幅优于直接 3D Zigzag，说明空间和时序信息的分离处理更有效

## 亮点与洞察

- **极简却高效的设计哲学**：仅通过改变 token 遍历顺序就获得巨大增益，不增加任何参数，堪称 "免费午餐"
- **异构逐层扫描的核心洞察**：将扫描多样性从单块多方向转移到多层异构方案，巧妙规避了 k-Mamba 的 k 倍开销
- **空间连续性的归纳偏置**：明确指出并量化了 Mamba 从 1D 到 2D 扩展时空间连续性的重要性，为后续 SSM 视觉应用提供了基础性指导
- **Stochastic Interpolant 的大规模验证**：首次将该框架扩展到 1024×1024 分辨率图像和视频生成

## 局限性

1. 实验规模受 GPU 资源限制，FacesHQ 1024 上未能充分训练
2. Zigzag-8 之外的更复杂空间填充曲线（如 Hilbert 曲线）表现不佳，最优扫描方案的理论基础尚不清晰
3. 在复杂数据集（MS-COCO）上 FID 仍有较大改进空间
4. 当前仅支持类别条件和文本条件，未探索更多条件控制模式

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

<!-- RELATED:END -->
