---
title: >-
  [论文解读] Efficient Visual State Space Model for Image Deblurring
description: >-
  [CVPR 2025][图像恢复][图像去模糊] 本文提出 EVSSM，通过在单方向 SSM 扫描前施加交替的几何变换（转置/翻转）来高效捕获非局部信息，并设计高效判别性频域 FFN (EDFFN) 增强局部细节，在图像去模糊任务上以仅 1/4 的计算量超越了现有 SSM 方法并达到 SOTA。 领域现状：图像去模糊旨在从模…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "图像去模糊"
  - "状态空间模型"
  - "Mamba"
  - "频域FFN"
  - "几何变换扫描"
---

# Efficient Visual State Space Model for Image Deblurring

**会议**: CVPR 2025  
**arXiv**: [2405.14343](https://arxiv.org/abs/2405.14343)  
**代码**: [https://github.com/kkkls/EVSSM](https://github.com/kkkls/EVSSM)  
**领域**: 图像复原  
**关键词**: 图像去模糊、状态空间模型、Mamba、频域FFN、几何变换扫描

## 一句话总结

本文提出 EVSSM，通过在单方向 SSM 扫描前施加交替的几何变换（转置/翻转）来高效捕获非局部信息，并设计高效判别性频域 FFN (EDFFN) 增强局部细节，在图像去模糊任务上以仅 1/4 的计算量超越了现有 SSM 方法并达到 SOTA。

## 研究背景与动机

**领域现状**：图像去模糊旨在从模糊图像中恢复清晰图像，主流方法分为 CNN 和 Transformer 两大类。CNN 方法受限于卷积操作的空间不变性和有限感受野，难以捕获空间变化特性和非局部信息。Transformer 通过自注意力机制建模全局依赖，效果更好，但其计算复杂度与 token 数量呈二次方关系，处理高分辨率图像时成本不可接受。

**现有痛点**：为降低 Transformer 的计算开销，现有方法采用局部窗口注意力、转置注意力、频域近似等策略，但这些方法在降低计算量的同时牺牲了对非局部信息或空间信息的建模能力，限制了恢复质量。近期状态空间模型 (SSM/Mamba) 展示了以线性复杂度建模长程依赖的潜力，但现有视觉 SSM 方法（如 VMamba）采用多方向扫描机制，计算成本是单方向的 4 倍，效率增益被大幅抵消。

**核心矛盾**：如何在保持线性计算复杂度的前提下，让 SSM 有效地探索二维图像中的非局部信息？多方向扫描虽然覆盖全面但计算代价过高，而单方向扫描又无法充分利用空间结构。

**本文目标** (1) 设计高效的视觉扫描策略，在不显著增加计算量的情况下捕获多方向的非局部信息；(2) 解决 SSM 参数 B/C/Δ 从相同线性变换导出、空间信息单一的问题；(3) 降低频域 FFN 的计算开销，同时保持局部细节增强能力。

**切入角度**：作者观察到，与其在多个方向重复扫描，不如在每次扫描前对输入特征进行简单的几何变换（转置或翻转），这样单方向扫描就能自动覆盖不同方向的信息。由于卷积具有平移不变性，几何变换不影响卷积本身，仅改变选择性扫描的行为。

**核心 idea**：用交替几何变换 + 单方向扫描代替多方向扫描，以近乎零开销实现多方向非局部信息探索。

## 方法详解

### 整体框架

EVSSM 采用经典的三级对称编码器-解码器架构。输入模糊图像 $I_{blur} \in \mathbb{R}^{H \times W \times 3}$ 首先通过 3×3 卷积提取浅层特征 $F_s \in \mathbb{R}^{H \times W \times C}$（C=48），送入三级编码器-解码器。每级编码器/解码器由若干个 EVSS 模块堆叠而成（各级数量为 [6, 6, 12]），级间通过双线性插值和 1×1 卷积实现上下采样，并加入跳跃连接。最终通过 3×3 卷积输出残差图像 R，加上输入得到去模糊结果 $I_{deblur} = R + I_{blur}$。

### 关键设计

1. **高效视觉扫描块 (EVS Block)**:

    - 功能：以最小计算代价实现多方向非局部信息探索
    - 核心思路：对于第 $i$ 个 EVSS 模块，在扫描前根据模块索引交替施加几何变换：$i \% 2 = 0$ 时做特征转置，$i \% 2 = 1$ 时做水平+垂直翻转。这样每 4 个 EVSS 模块自动恢复到原始空间结构。变换后，通过线性层拆分为两个分支 $X_1, X_2$，$X_1$ 经 3×3 深度卷积和 SiLU 激活后送入 S6 选择性扫描，$X_2$ 作为门控信号与扫描结果相乘输出
    - 设计动机：几何变换几乎零开销（仅涉及内存重排），却能让单方向扫描在不同模块中"看到"不同方向的信息排列，等效实现了多方向扫描的效果

2. **1D 深度卷积增强 SSM 参数多样性**:

    - 功能：使 SSM 的 B、C、Δ 参数编码不同的空间信息
    - 核心思路：在线性投影导出 B、C、Δ 后，分别对每个参数施加 kernel size=7 的 1D 深度卷积。由于前序有几何变换，1D 卷积实际上在原始 2D 输入上聚合了多方向信息，使各参数具有差异化的空间表征
    - 设计动机：原始 Mamba 中 B、C、Δ 都由相同输入的线性变换得到，编码了相同的空间信息，限制了模型捕获多样化空间模式的能力。加入 1D 深度卷积后各参数有了独立的局部关注能力

3. **高效判别性频域 FFN (EDFFN)**:

    - 功能：增强 SSM 未充分覆盖的局部细节信息
    - 核心思路：在 FFN 末端（而非中间）对特征执行 FFT 并学习一个量化矩阵 W，自适应筛选需要保留的频率信息。由于 FFN 末端特征通道数远小于中间层（原 DFFN 在 3 倍通道扩展的中间层做 FFT），计算开销大幅降低
    - 设计动机：FFTformer 的 DFFN 在 FFN 中间做 FFT，此时通道数是输入的 3 倍，导致 FFT 计算量巨大。将频域筛选移至 FFN 末端，在不影响性能的前提下显著降低计算时间

### 损失函数 / 训练策略

训练损失由像素域 L1 损失和频域 L1 损失组成：$\mathcal{L} = \|I_{deblur} - I_{gt}\|_1 + 0.1 \|\mathcal{F}(I_{deblur}) - \mathcal{F}(I_{gt})\|_1$。采用渐进式训练：先用 128×128 patch + batch 64 训练 300K 迭代，再切换到 256×256 patch + batch 16 继续 300K 迭代，均使用 AdamW 优化器和余弦退火策略。

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR (dB) | SSIM |
|--------|------|-----------|------|
| GoPro | FFTformer | 34.21 | 0.9692 |
| GoPro | GRL | 33.93 | 0.9680 |
| GoPro | **EVSSM** | **34.51** | **0.9713** |
| HIDE | FFTformer | 31.62 | 0.9455 |
| HIDE | GRL | 31.65 | 0.9470 |
| HIDE | **EVSSM** | **31.99** | **0.9503** |
| RealBlur-R | FFTformer | 40.11 | 0.9753 |
| RealBlur-R | **EVSSM** | **41.27** | **0.9776** |
| RealBlur-J | FFTformer | 32.62 | 0.9326 |
| RealBlur-J | **EVSSM** | **34.34** | **0.9456** |

### 消融实验

| 配置 | 计算特点 | 说明 |
|------|---------|------|
| VMamba 四方向扫描 | 4× 计算量 | 现有 SSM 视觉方法的通用做法 |
| EVSSM 几何变换+单方向 | 1× 计算量 | 仅增加几何变换的微小开销 |
| CU-Mamba (另一 SSM 方法) | GoPro PSNR 33.53 | 明显低于 EVSSM 的 34.51 |

此外在去雨（PSNR 49.00 vs Restormer 47.98）和去雾（PSNR 32.05 vs DehazeFormer 31.45）任务上也验证了 EVSSM 的泛化性。

### 关键发现
- 几何变换策略是核心贡献：以几乎为零的计算代价，实现了等效于多方向扫描的信息覆盖
- EDFFN 将频域筛选移至 FFN 末端，在保持相同性能的前提下显著减少运行时间
- 1D 深度卷积对 B、C、Δ 参数的增强也有独立贡献，使得扫描过程中参数具有空间差异性
- 在真实模糊数据集 RealBlur 上提升特别明显（+1.16 dB / +1.72 dB），体现了对真实退化的鲁棒性

## 亮点与洞察
- **几何变换替代多方向扫描**是本文最精妙的设计——利用转置/翻转这种零成本操作改变了信息在序列中的相对位置，让单方向扫描在不同层中等效扫描了不同空向，是一种非常优雅的工程化思路
- **频域 FFN 的位置优化**（从中间移至末端）是一个简单但高效的改进思路，具有广泛的可迁移性——任何在特征中间层做昂贵变换的网络都可以考虑将变换后移到通道数更小的位置
- EVSSM 在去模糊之外还成功应用于去雨和去雾，说明该架构具有通用的图像复原能力

## 局限与展望
- 几何变换策略虽然简洁，但变换模式固定（交替转置和翻转），未探索自适应选择变换类型的可能性
- 文中未充分分析几何变换对不同尺度特征的影响差异，也未与可学习的扫描方向进行对比
- EDFFN 将频域筛选移至末端虽然降低了成本，但可能也损失了一些在高通道特征上的表达能力，这一点缺乏消融验证

## 相关工作与启发
- **vs FFTformer**: FFTformer 使用频域 Transformer，EVSSM 用 SSM 替代注意力机制实现线性复杂度，同时改进了 FFTformer 的 DFFN 组件。EVSSM 在所有基准上均超过 FFTformer
- **vs VMamba/CU-Mamba**: 这些方法采用多方向扫描，EVSSM 通过几何变换实现等效效果但仅需 1/4 计算量
- **vs NAFNet/Restormer**: CNN/Transformer 方法难以高效建模非局部信息，EVSSM 的线性复杂度优势在高分辨率场景下尤为突出

## 评分
- 新颖性: ⭐⭐⭐⭐ 几何变换替代多方向扫描的思路新颖且优雅，但整体框架仍沿用标准编码器-解码器
- 实验充分度: ⭐⭐⭐⭐ 在多个去模糊基准和其他复原任务上验证，但消融实验可以更详细
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机推导合理，图表展示清楚
- 价值: ⭐⭐⭐⭐ 为视觉 SSM 的高效化提供了简洁实用的方案，对相关领域有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[CVPR 2025\] MambaIRv2: Attentive State Space Restoration](mambairv2_attentive_state_space_restoration.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)

</div>

<!-- RELATED:END -->
