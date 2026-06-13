---
title: >-
  [论文解读] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration
description: >-
  [CVPR 2025][图像恢复][低光图像增强] 提出 URWKV 模型，在 RWKV 架构中引入多状态（intra-stage 和 inter-stage）视角，通过亮度自适应归一化（LAN）、多状态聚合的 token shift（SQ-Shift）和状态感知选择性融合（SSF）模块，用一个统一模型处理低光图像的动态耦合退化（噪声、亮度失真、运动模糊），参数量仅 2.25M 即在 8 个基准数据集上全面超越现有方法。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "低光图像增强"
  - "RWKV"
  - "多状态机制"
  - "统一模型"
  - "图像去模糊"
---

# URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2505.23068](https://arxiv.org/abs/2505.23068)  
**代码**: [https://github.com/FZU-N/URWKV](https://github.com/FZU-N/URWKV)  
**领域**: 图像复原 / 低光图像增强  
**关键词**: 低光图像增强, RWKV, 多状态机制, 统一模型, 图像去模糊

## 一句话总结

提出 URWKV 模型，在 RWKV 架构中引入多状态（intra-stage 和 inter-stage）视角，通过亮度自适应归一化（LAN）、多状态聚合的 token shift（SQ-Shift）和状态感知选择性融合（SSF）模块，用一个统一模型处理低光图像的动态耦合退化（噪声、亮度失真、运动模糊），参数量仅 2.25M 即在 8 个基准数据集上全面超越现有方法。

## 研究背景与动机

**领域现状**：低光环境带来噪声增加、细节丢失、对比度降低和色彩失真等多重退化。现有方法分为三类：低光图像增强（LLIE）模型专注亮度提升和去噪，联合 LLIE-去模糊模型处理低光+运动模糊的耦合退化，以及 Restormer, MambaIR 等统一图像复原模型。

**现有痛点**：(1) LLIE 模型无法处理低光场景中频繁出现的运动模糊退化；(2) LLIE-去模糊模型（如 LEDNet, PDHAT）受限于预定义的退化类别，无法适应实际场景中动态耦合的退化组合；(3) 通用统一模型（如 Restormer）缺乏针对低光环境的自适应机制，往往放大退化或引入新伪影；(4) 参数量和计算量大是实际部署的关键瓶颈。

**核心矛盾**：如何用一个参数高效的统一模型灵活处理低光场景中动态变化的耦合退化？

**本文目标**：构建一个能感知和分析复杂退化的统一模型，不依赖预定义退化类型，而是通过多状态表示动态适应不同退化组合。

**切入角度**：借鉴人眼瞳孔根据亮度自适应调节的机制，以及 RWKV 架构线性复杂度和序列建模能力，从"多状态"角度出发——利用 stage 间状态做亮度自适应，利用 stage 内状态捕获长程退化依赖。

**核心 idea**：将 RWKV 的单状态 token shift 扩展为多状态机制，通过 EMA 聚合历史状态捕获长程依赖，通过跨 stage 状态动态调制归一化参数实现场景感知的亮度适应。

## 方法详解

### 整体框架

标准 encoder-decoder 架构，encoder 和 decoder 各 3 个 stage，核心处理单元为 URWKV block。encoder 每个 stage 含 $N_1=3$ 个 URWKV block + 下采样层，decoder 每个 stage 含 $N_2=2$ 个 URWKV block + 上采样层。每个 URWKV block 包含两个子块：多状态空间混合子块和多状态通道混合子块。encoder 到 decoder 之间通过 SSF 模块替代朴素 skip connection 传递特征。

### 关键设计

1. **亮度自适应归一化（LAN）**:

    - 功能：替代标准 LayerNorm，根据整个复原流程中的多个历史 stage 状态动态调节归一化参数，实现场景感知的亮度调制
    - 核心思路：收集当前输入 $X_t$ 和所有历史 stage 输出 $M_i$ 的全局亮度向量（GAP），零填充到统一维度后堆叠为 2D 亮度图。用多核（1×T, 3×T, 5×T）1D 卷积聚合跨状态的亮度变化模式，拼接后通过 MLP+tanh 预测亮度调制参数 $\Delta\gamma_t$，更新 LayerNorm 的缩放参数为 $\hat{\gamma_t} = \gamma + \Delta\gamma_t$
    - 设计动机：标准 LayerNorm 的固定参数无法适应低光场景中复杂的亮度变化。灵感来自人眼瞳孔根据环境光自适应调节的机制。利用 inter-stage 状态使归一化参数能动态感知整个复原过程中的亮度演变

2. **多状态四方向 Token Shift（SQ-Shift）**:

    - 功能：扩展原始 RWKV 的单状态 Q-Shift，捕获跨多个状态的长程退化依赖
    - 核心思路：在执行标准 Q-Shift（四方向空间 token 移位）之前，用指数移动平均（EMA）聚合当前状态与同一 stage 内所有先前状态：$\text{MSA}(X_t^{LAN}) = \alpha \odot X_t^{LAN} + (1-\alpha) \odot \text{MSA}(H_{t-1})$，其中 $\alpha=0.5$ 为衰减因子。这样每个 block 不仅感知邻近 token 的空间关系，还融合了历史状态的退化修复信息
    - 设计动机：原始 RWKV 单状态机制导致早期信息逐渐消失，无法有效捕获耦合退化间的复杂依赖

3. **状态感知选择性融合（SSF）模块**:

    - 功能：替代朴素 skip connection，动态对齐和选择性融合跨 encoder stage 的多状态特征
    - 核心思路：对三个 encoder 输出分别做通道均值压缩（减少语义干扰），自适应对齐到 decoder 目标分辨率，堆叠后用 inception 风格的多尺度卷积（1×1, 3×3, 5×5）聚合退化模式，最终通过 sigmoid 生成空间权重 $W_s$，以 $D_1' = ([W_s \odot E_3, D_1])W_p$ 方式选择性地将 encoder 特征传递到 decoder
    - 设计动机：朴素 skip connection（add/concat）在低光场景下容易传播噪声和无关信息，且存在 stage 间语义鸿沟。SSF 通过预测空间引导权重选择性地过滤有用信息

### 损失函数 / 训练策略

统一损失函数：$L_1$ Loss + SSIM Loss + Perceptual Loss。使用 Adam 优化器（$\beta_1=0.9, \beta_2=0.99$），初始学习率 $2\times10^{-4}$，余弦退火衰减至 $10^{-6}$。训练采用翻转、旋转等数据增强。测试时直接处理任意形状输入，无需裁剪或缩放。输入通道数 $C=32$，NVIDIA Tesla A40 GPU 训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | URWKV | Retinexformer | Restormer | PDHAT | 参数量 |
|--------|------|-------|---------------|-----------|-------|--------|
| LOL-v2-real | PSNR/SSIM | **23.11/0.874** | 22.79/0.839 | 18.60/0.789 | 20.16/0.841 | **2.25M** |
| LOL-v2-syn | PSNR/SSIM | **26.36/0.944** | 25.67/0.928 | 21.41/0.831 | 24.94/0.937 | - |
| SDSD-indoor | PSNR/SSIM | **31.24/0.911** | 29.78/0.895 | 28.49/0.892 | 26.37/0.884 | - |
| LOL-blur | PSNR/SSIM | **27.27/0.890** | 25.25/0.821 | 26.38/0.860 | 26.71/0.879 | - |

URWKV 仅 2.25M 参数、18.34G FLOPs，远低于 Restormer (26.11M/140.99G) 和 MIRNet (31.76M/785G)。

### 消融实验

| 配置 | PSNR | SSIM | Params | FLOPs |
|------|------|------|--------|-------|
| Baseline (无LAN无SSF) | 21.33 | 0.856 | 1.64M | 18.25G |
| + SSF | 21.40 | 0.861 | 1.65M | 18.29G |
| + LAN | 22.71 | 0.869 | 2.25M | 18.30G |
| + LAN + SSF (Full) | **23.11** | **0.874** | 2.25M | 18.34G |

多状态聚合消融：多状态 EMA + Q-Shift 组合比单状态或单独 Q-Shift 高 0.5-0.9 dB。

### 关键发现

- LAN 贡献最大（+1.38 dB），表明低光场景中亮度自适应归一化至关重要
- SSF 相比朴素 skip connection（Add/Cat）不仅性能更好，还避免了多状态特征直接融合带来的噪声传播
- 多状态 EMA 聚合比单状态和标准 Q-Shift 都有显著提升，验证了跨状态长程依赖的重要性
- 在 LOL-blur 数据集上超越专门的 LLIE-去模糊模型（PDHAT），证明统一模型也能处理耦合退化

## 亮点与洞察

- **极致的参数效率**：2.25M 参数就超越了 26M 的 Restormer 和 31M 的 MIRNet，RWKV 架构的线性复杂度是关键。这对边缘设备部署非常有价值
- **多状态视角的创新性**：将 RWKV 的状态概念从单纯的序列依赖扩展为复原过程中的多层次上下文感知，既有 intra-stage（退化修复进程内）又有 inter-stage（跨处理阶段），是一种优雅的设计
- **LAN 的生物启发**：模仿瞳孔自适应调节的 LAN 设计直觉清晰，且 multi-kernel 聚合策略能同时捕获局部和全局亮度变化模式

## 局限与展望

- 目前主要验证在低光场景，是否能无缝扩展到更多退化类型（如雨雾、压缩伪影等）尚未验证
- EMA 的衰减因子 $\alpha$ 固定为 0.5，可能对不同退化程度不是最优，自适应 $\alpha$ 值得探索
- SSF 模块对 encoder 输出做了通道均值压缩来减少干扰，这可能也丢失了有用信息
- 模型在 SID 数据集上略低于 Retinexformer，说明极暗场景下仍有改进空间

## 相关工作与启发

- **vs Retinexformer**: 基于 Retinex 理论的专用 LLIE 模型，在纯增强任务上竞争力强，但无法处理运动模糊等耦合退化。URWKV 统一处理的能力是关键优势
- **vs Restormer**: 通用统一模型但缺乏低光自适应机制，在低光数据集上表现不佳。URWKV 通过 LAN 和多状态机制弥补了这一缺陷
- **vs MambaIR**: 同为高效线性复杂度架构，但 MambaIR 缺乏低光专用组件，在多个低光数据集上明显不如 URWKV
- **vs PDHAT**: 专用 LLIE-去模糊模型，在 LOL-blur 上竞争力强，但在纯 LLIE 任务上表现不稳定。URWKV 在两种场景下都更优

## 评分

- 新颖性: ⭐⭐⭐⭐ 多状态 RWKV 视角有新意，LAN 和 SSF 设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 8个数据集、多种退化类型、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分
- 价值: ⭐⭐⭐⭐ 2.25M 参数的统一模型在实际部署中很有吸引力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)

</div>

<!-- RELATED:END -->
