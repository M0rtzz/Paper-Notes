---
title: >-
  [论文解读] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration
description: >-
  [ICCV 2025][图像恢复][Vision Mamba] 本文提出EAMamba框架，通过多头选择性扫描模块（MHSSM）和全方位扫描策略（all-around scanning），在不增加计算复杂度和参数量的情况下实现多方向扫描，解决了Vision Mamba在图像恢复中的计算开销和局部像素遗忘问题，在超分辨率、去噪、去模糊、去雾等任务上取得了31-89%的FLOPs降低同时保持优异性能。
tags:
  - ICCV 2025
  - 图像恢复
  - Vision Mamba
  - state space model
  - multi-head selective scan
  - all-around scanning
---

# EAMamba: Efficient All-Around Vision State Space Model for Image Restoration

**会议**: ICCV 2025  
**arXiv**: [2506.22246](https://arxiv.org/abs/2506.22246)  
**代码**: https://github.com/daidaijr/EAMamba  
**领域**: 图像恢复  
**关键词**: Vision Mamba, state space model, image restoration, multi-head selective scan, all-around scanning

## 一句话总结
本文提出EAMamba框架，通过多头选择性扫描模块（MHSSM）和全方位扫描策略（all-around scanning），在不增加计算复杂度和参数量的情况下实现多方向扫描，解决了Vision Mamba在图像恢复中的计算开销和局部像素遗忘问题，在超分辨率、去噪、去模糊、去雾等任务上取得了31-89%的FLOPs降低同时保持优异性能。

## 研究背景与动机
- **领域现状**：图像恢复领域经历了CNN → Vision Transformer → Vision Mamba的演进，Mamba以线性复杂度建模长程依赖成为新方向
- **现有痛点**：当前Vision Mamba方法（如MambaIR）采用二维选择性扫描（2DSS），每增加一个扫描方向就需要相应增加计算开销和参数量，限制了扫描模式的扩展性
- **核心矛盾**：（1）2DSS的计算复杂度随扫描序列数线性增长；（2）二维扫描导致"局部像素遗忘"——空间上相邻的像素在展平为一维序列后可能相距很远，丢失局部空间关系
- **本文要解决的问题**：在保持甚至降低计算复杂度的同时，增加扫描方向以捕获更全面的空间信息
- **切入角度**：借鉴多头注意力的思想，将通道分组后分别进行不同方向的扫描
- **核心idea**：通道分组扫描（Multi-Head Selective Scan）消除了多方向扫描的计算开销，使全方位扫描成为可能

## 方法详解

### 整体框架
EAMamba采用UNet编码器-解码器架构，包含4级编码器（[4,6,6,7]个MambaFormer块）、瓶颈模块、4级解码器和精炼模块。输入低质量图像经编码器提取多尺度特征，解码器逐步重建残差图像，最终与原始输入相加得到高质量输出。核心创新在MambaFormer块中的MHSSM模块。

### 关键设计
1. **多头选择性扫描模块（MHSSM）**:

    - 功能：替代标准2DSS，高效处理多方向一维序列
    - 核心思路：将输入特征 $X \in \mathbb{R}^{H \times W \times C}$ 沿通道维度分为 $n$ 组，每组独立进行不同方向的扫描，输出拼接后得到最终结果。处理流程：$Y = \text{LN}(\text{MHSS}(\text{SiLU}(\text{DWConv2D}(\text{Linear}(X)))))$，$X_{out} = \text{Linear}(Y \otimes \text{SiLU}(\text{Linear}(X)))$
    - 设计动机：标准2DSS对全通道进行每个方向的扫描，复杂度随方向数线性增长；MHSS通过通道分组使总复杂度与单方向扫描相当，突破了扩展性瓶颈

2. **全方位扫描策略（All-Around Scanning）**:

    - 功能：在水平、垂直方向的基础上增加对角线方向扫描
    - 核心思路：组合二维扫描（水平+垂直+各反向）和对角扫描（正对角+翻转对角+各反向），共8个方向覆盖完整空间邻域
    - 设计动机：有效感受野（ERF）可视化显示，仅二维扫描难以捕获对角方向信息，导致局部像素遗忘；全方位扫描通过互补的多方向覆盖解决了这一问题

3. **MambaFormer块**:

    - 功能：作为编解码器的基本构建单元
    - 核心思路：$X' = X + \text{MHSSM}(\text{LN}(X))$，$X'' = X' + \text{Channel MLP}(\text{LN}(X'))$，类似Transformer块但用MHSSM替代注意力层
    - 设计动机：保持与Transformer相似的残差连接和归一化结构，便于组合和扩展

### 损失函数 / 训练策略
- L1损失函数
- 渐进式训练策略：初始patch 128×128/batch 64，逐步增大到384×384/batch 8
- AdamW优化器，初始学习率 $3 \times 10^{-4}$，余弦退火至 $1 \times 10^{-6}$
- 数据增强：随机水平/垂直翻转 + 90°旋转
- 总计450K迭代

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | EAMamba | MambaIR(-UNet) | FLOPs降低 |
|------------|------|--------|----------------|----------|
| SIDD（真实去噪） | PSNR | 39.87 | 39.89 | 41% (137G vs 230G) |
| CBSD68 σ=50（合成去噪） | PSNR | 28.62 | 28.61 | 89% (137G vs 1290G) |
| GoPro（去模糊） | PSNR | 33.58 | - | 超SFNet 0.31dB |
| SOTS-Indoor（去雾） | PSNR | 43.19 | - | 超DehazeFormer-L 3.14dB |
| RealSR ×4（超分辨率） | PSNR | 29.60 | 29.53 | 40% (137G vs 230G) |

### 消融实验

| 配置 | Param.(M) | FLOPs(G) | Urban100 σ=25 PSNR | 说明 |
|------|-----------|----------|---------------------|------|
| Baseline (2DSSM) | 31.1 | 286 | 33.00 | MambaIR基线 |
| + MHSSM | 25.3 | 137 | 32.89 | FLOPs减半，PSNR仅降0.1% |
| + all-around scan | 25.3 | 137 | 32.93 | 全方位扫描恢复部分性能 |

| 扫描组合 | SIDD | RealSRx4 | GoPro | SOTS-Indoor |
|---------|------|----------|-------|-------------|
| 2D + Diagonal (默认) | 39.87 | 29.60 | 33.58 | 43.19 |
| 2D + Z-order | 39.82 | 29.58 | 33.51 | 43.20 |
| 2D + Hilbert | 39.83 | 29.51 | 33.66 | 43.07 |

### 关键发现
- MHSSM将FLOPs减少约50%，PSNR仅降低约0.1dB，展现了卓越的效率-性能权衡
- 全方位扫描在所有任务上一致优于单一二维扫描，ERF可视化直观展示了更完整的局部覆盖
- 2D + Diagonal组合在多数任务上表现最均衡
- EAMamba在去模糊（GoPro）和去雾（SOTS-Indoor）上分别超越第二名0.31dB和3.14dB

## 亮点与洞察
- 多头分组扫描是一个优雅的设计：以"通道分组+方向分配"打破了扫描方向数与计算量的线性关系
- ERF可视化有效直观地展示了不同扫描策略的空间覆盖差异，为方法选择提供了清晰依据
- 全方位扫描策略具有良好的可扩展性，可以灵活组合不同扫描方式适配特定任务
- 在去雾任务上的巨大提升（+3.14dB）值得关注

## 局限与展望
- 通道分组可能导致各组特征的信息隔离，缺乏组间交互机制
- 全方位扫描虽然高效，但对非规则图像区域（如边界、不规则形状）的覆盖可能不均匀
- 仍依赖固定的扫描模式组合，未探索自适应或可学习的扫描策略
- 在真实去噪和超分辨率上与Restormer等非Mamba方法的优势不够明显

## 相关工作与启发
- **MambaIR**：首个将Mamba应用于图像恢复的工作，但2DSS计算开销大
- **VMambaIR**：提出Omni Selective Scan，方向更多但计算量相应增加
- **Restormer**：Transformer-based方法的代表，在部分任务上仍具竞争力
- **启发**：多头分组的思想可推广到其他需要多方向/多尺度处理的任务；通道分组策略与组卷积、深度可分离卷积有异曲同工之妙

## 评分
- 新颖性: ⭐⭐⭐⭐ 多头选择性扫描是巧妙的设计，全方位扫描有效解决局部遗忘问题
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖去噪/超分/去模糊/去雾四类任务，多数据集验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，ERF可视化和架构图制作精良，实验对比详尽
- 价值: ⭐⭐⭐⭐ 为Vision Mamba在低级视觉任务的效率化提供了通用解决方案

<!-- RELATED:START -->

## 相关论文

- [Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [MambaIRv2: Attentive State Space Restoration](../../CVPR2025/image_restoration/mambairv2_attentive_state_space_restoration.md)
- [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [QMambaBSR: Burst Image Super-Resolution with Query State Space Model](../../CVPR2025/image_restoration/qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)

<!-- RELATED:END -->
