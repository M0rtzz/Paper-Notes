---
title: >-
  [论文解读] MambaIR: A Simple Baseline for Image Restoration with State-Space Model
description: >-
  [ECCV 2024][图像恢复][图像修复] 本文首次将 Mamba（选择性状态空间模型）引入底层图像修复任务，通过设计残差状态空间块（RSSB）中的局部卷积增强和通道注意力机制，解决了 vanilla Mamba 在 2D 图像上的局部像素遗忘和通道冗余问题，在图像超分辨率和去噪任务上以线性复杂度实现了与 Transformer 方法相当甚至更优的性能（SR 上超过 SwinIR 0.45dB）。
tags:
  - ECCV 2024
  - 图像恢复
  - 图像修复
  - 状态空间模型
  - Mamba
  - 超分辨率
  - 图像去噪
---

# MambaIR: A Simple Baseline for Image Restoration with State-Space Model

**会议**: ECCV 2024  
**arXiv**: [2402.15648](https://arxiv.org/abs/2402.15648)  
**代码**: https://github.com/csguoh/MambaIR  
**领域**: 图像修复  
**关键词**: 图像修复, 状态空间模型, Mamba, 超分辨率, 图像去噪

## 一句话总结

本文首次将 Mamba（选择性状态空间模型）引入底层图像修复任务，通过设计残差状态空间块（RSSB）中的局部卷积增强和通道注意力机制，解决了 vanilla Mamba 在 2D 图像上的局部像素遗忘和通道冗余问题，在图像超分辨率和去噪任务上以线性复杂度实现了与 Transformer 方法相当甚至更优的性能（SR 上超过 SwinIR 0.45dB）。

## 研究背景与动机

图像修复领域长期存在一个核心矛盾：**全局感受野 vs 计算效率**。CNN 方法（如 EDSR、RCAN）计算效率高但感受野有限；Transformer 方法（如 SwinIR、HAT）虽然能建模全局依赖，但自注意力的二次复杂度使得在全图尺度上计算代价极高。即使使用窗口注意力等高效变体，也只是在全局建模和计算效率间做折中。

Mamba 作为改进的选择性结构化状态空间模型，具有线性复杂度的长程依赖建模能力，天然适合解决这一困境。然而，标准 Mamba 是为 1D NLP 序列设计的，直接应用于 2D 图像修复面临两个问题：

**局部像素遗忘**：将 2D 图像展平为 1D 序列后，空间上邻近的像素在序列中可能距离很远，递归处理时容易被遗忘

**通道冗余**：为记忆长序列依赖，隐藏状态数量通常很大，导致通道特征冗余

核心 idea：设计专门的修复模块（RSSB），通过局部卷积增强和通道注意力来弥补 Mamba 在底层视觉中的不足，使其成为 CNN 和 Transformer 之外的第三类修复 backbone。

## 方法详解

### 整体框架

MambaIR 采用三阶段架构：
1. **浅层特征提取**：一个 3×3 卷积层提取浅层特征 $F_S \in \mathbb{R}^{H \times W \times C}$
2. **深层特征提取**：多个堆叠的残差状态空间组（RSSG），每组包含若干残差状态空间块（RSSB）
3. **高质量图像重建**：浅层和深层特征逐元素相加后重建输出

### 关键设计

1. **残差状态空间块（RSSB）**：RSSB 是 MambaIR 的核心模块，打破了 Transformer 中 "Norm → Attention → Norm → MLP" 的固定范式。它包含两个子模块：

    - 第一部分：LayerNorm → VSSM（视觉状态空间模块）+ 可学习缩放因子 $s$ 的跳跃连接，即 $Z^l = \text{VSSM}(\text{LN}(F_D^l)) + s \cdot F_D^l$
    - 第二部分：LayerNorm → 瓶颈结构局部卷积 → 通道注意力（CA）+ 可学习缩放因子 $s'$ 的跳跃连接，即 $F_D^{l+1} = \text{CA}(\text{Conv}(\text{LN}(Z^l))) + s' \cdot Z^l$
   
   局部卷积用于恢复被 1D 展平破坏的邻域相似性，通道注意力用于缓解大量隐藏状态导致的通道冗余。可学习缩放因子控制跳跃连接的信息流。

2. **视觉状态空间模块（VSSM）**：采用双分支结构，第一分支通过 Linear → DWConv → SiLU → 2D-SSM → LN 提取全局特征；第二分支通过 Linear → SiLU 作为门控。两分支经 Hadamard 乘积融合后投影回原通道数。公式为：
    $X_1 = \text{LN}(\text{2D-SSM}(\text{SiLU}(\text{DWConv}(\text{Linear}(X)))))$
    $X_2 = \text{SiLU}(\text{Linear}(X))$
    $X_{out} = \text{Linear}(X_1 \odot X_2)$

3. **2D 选择性扫描模块（2D-SSM）**：为了让 Mamba 处理 2D 图像的非因果数据，采用四方向扫描策略（左上→右下、右下→左上、右上→左下、左下→右上），对每个方向的展平序列分别用离散状态空间方程建模长程依赖，最后将四个方向的结果求和并 reshape 回 2D。

### 损失函数 / 训练策略

- 图像超分辨率：L1 损失 $\mathcal{L} = \|I_{HQ} - I_{LQ}\|_1$
- 图像去噪：Charbonnier 损失 $\mathcal{L} = \sqrt{\|I_{HQ} - I_{LQ}\|^2 + \epsilon^2}$，$\epsilon = 10^{-3}$
- 训练 patch：SR 为 64×64，去噪为 128×128
- 数据增强：水平翻转 + 90°/180°/270° 随机旋转
- 优化器：Adam（$\beta_1 = 0.9, \beta_2 = 0.999$），初始学习率 $2 \times 10^{-4}$
- SR ×3 和 ×4 模型使用 ×2 预训练权重初始化
- 8× NVIDIA V100 GPU 训练

## 实验关键数据

### 主实验

经典图像超分辨率（×2 scale，PSNR/dB）：

| 数据集 | 指标 | MambaIR | SwinIR | SRFormer | 提升 |
|--------|------|---------|--------|----------|------|
| Set5 | PSNR | 38.57 | 38.42 | 38.51 | +0.15/+0.06 |
| Set14 | PSNR | 34.67 | 34.46 | 34.44 | +0.21/+0.23 |
| Urban100 | PSNR | 34.15 | 33.81 | 34.09 | +0.34/+0.06 |
| Manga109 | PSNR | 40.28 | 39.92 | 40.07 | +0.36/+0.21 |

经典图像超分辨率（×4 scale，PSNR/dB）：

| 数据集 | 指标 | MambaIR | SwinIR | SRFormer | 提升 |
|--------|------|---------|--------|----------|------|
| Set5 | PSNR | 33.03 | 32.92 | 32.93 | +0.11/+0.10 |
| Urban100 | PSNR | 27.68 | 27.45 | 27.68 | +0.23/0.00 |
| Manga109 | PSNR | 32.32 | 32.03 | 32.21 | +0.29/+0.11 |

### 消融实验

RSSB 设计消融（Set5/Set14/Urban100 PSNR）：

| 配置 | Set5 | Set14 | Urban100 | 说明 |
|------|------|-------|----------|------|
| 去掉 Conv | 38.48 | 34.54 | 34.04 | 局部增强对 Urban100 贡献大 |
| 去掉 Conv+CA | 38.55 | 34.64 | 34.06 | 直接用原始 Mamba 效果次优 |
| 用 MLP 替换 Conv+CA | 38.55 | 34.68 | 34.22 | 类 Transformer 结构不适合 SSM |
| 完整 MambaIR | 38.57 | 34.67 | 34.15 | - |

扫描方向消融：

| 配置 | Set5 | Urban100 | 说明 |
|------|------|----------|------|
| 单方向 | 38.53 | 34.06 | 信息感知范围最小 |
| 双方向 | 38.56 | 33.96 | - |
| 四方向（baseline） | 38.57 | 34.15 | 最优 |

### 关键发现

- MambaIR 在计算复杂度与 SwinIR 相当的情况下，拥有真正的全局有效感受野（ERF 可视化验证）
- 计算复杂度随输入分辨率线性增长，与窗口注意力类似但感受野更大
- SSM 和 Attention 虽然都能建模全局依赖，但行为模式不同，不能简单替换
- 轻量级版 MambaIR-light 在参数量和 MACs 相近时仍然超过 SwinIR-light 0.34dB（×4 Manga109）

## 亮点与洞察

1. **首次系统性地将 Mamba 引入底层视觉**：虽然 VMamba 等已在高层视觉任务中探索 Mamba，但 MambaIR 是第一个为图像修复量身定制的 Mamba 方法
2. **简洁有效的设计哲学**：RSSB 的两个改进（局部卷积 + 通道注意力）都极其简单，但切中了 Mamba 用于 2D 图像的两个真实痛点
3. **关于全局感受野的洞察**：通过 ERF 可视化清晰展示 MambaIR 实现了类似全注意力的全局感受野，但复杂度是线性的
4. **可学习缩放因子**：控制跳跃连接，让网络自适应调节残差信息流

## 局限与展望

1. 四方向扫描仍是一种启发式的 2D 适配方案，可能不是最优的空间建模方式
2. 消融实验显示在某些数据集上 MambaIR 的提升不是特别显著（如 Set5 上仅 +0.06 vs SRFormer）
3. 没有在真实世界退化场景（Real-world SR）上做广泛实验
4. 对于不同修复任务（SR、去噪、JPEG去伪影），模型配置需要分别调整
5. 推理速度虽然比原始 Transformer 好，但文中没有详细的延迟对比数据

## 相关工作与启发

- **VMamba**：MambaIR 的 VSSM 和 2D-SSM 直接沿用了 VMamba 的设计，说明高层视觉的 Mamba 适配方案可以迁移到底层视觉
- **SwinIR**：作为主要对标方法，窗口注意力的局限性（非全局感受野）是 MambaIR 的出发点
- **HAT**：通过激活更多像素提升修复性能的观察，为 Mamba 的全局建模提供了理论支撑
- 对后续工作的启发：可以探索更好的 2D 扫描策略、结合 Mamba 和注意力的混合架构、以及 Mamba 在视频修复等更长序列任务中的应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 Mamba 引入图像修复，虽然具体模块设计较简单
- 实验充分度: ⭐⭐⭐⭐ 覆盖 SR、去噪等多个任务，消融实验全面，但缺少真实世界场景
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，问题定义准确，实验安排合理
- 价值: ⭐⭐⭐⭐⭐ 作为 baseline 工作为后续 Mamba+修复研究奠定基础，影响力大

<!-- RELATED:START -->

## 相关论文

- [MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](../../AAAI2026/image_restoration/mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [MambaIRv2: Attentive State Space Restoration](../../CVPR2025/image_restoration/mambairv2_attentive_state_space_restoration.md)

<!-- RELATED:END -->
