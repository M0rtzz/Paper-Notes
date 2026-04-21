---
title: >-
  [论文解读] MambaVision: A Hybrid Mamba-Transformer Vision Backbone
description: >-
  [CVPR 2025][图像分割][Mamba] NVIDIA 提出 MambaVision——首个系统研究 Mamba 与 Transformer 混合方式的视觉骨干网络，通过重设计的 MambaVision Mixer + 在最后几层加入 self-attention 来弥补 SSM 的全局上下文不足，在 ImageNet-1K 上达到精度-吞吐量的新 Pareto 前沿，同时在检测和分割下游任务中也优于同等规模的竞争模型。
tags:
  - CVPR 2025
  - 图像分割
  - Mamba
  - Transformer
  - 混合架构
  - SSM
  - 视觉骨干
---

# MambaVision: A Hybrid Mamba-Transformer Vision Backbone

**会议**: CVPR 2025  
**arXiv**: [2407.08083](https://arxiv.org/abs/2407.08083)  
**代码**: https://github.com/NVlabs/MambaVision  
**领域**: 视觉骨干网络 / 图像分类 / 目标检测 / 语义分割  
**关键词**: Mamba, Transformer, 混合架构, SSM, 视觉骨干

## 一句话总结
NVIDIA 提出 MambaVision——首个系统研究 Mamba 与 Transformer 混合方式的视觉骨干网络，通过重设计的 MambaVision Mixer + 在最后几层加入 self-attention 来弥补 SSM 的全局上下文不足，在 ImageNet-1K 上达到精度-吞吐量的新 Pareto 前沿，同时在检测和分割下游任务中也优于同等规模的竞争模型。

## 研究背景与动机

1. **领域现状**：Transformer 在视觉领域已成为主流骨干，但 attention 的二次复杂度是硬伤。Mamba 提出了基于 SSM 的线性复杂度替代方案，已在 NLP 中见效。Vision Mamba（Vim）和 VMamba 等将 Mamba 引入视觉任务。
2. **现有痛点**：(1) Mamba 的自回归公式处理图像时有天然劣势——像素之间没有顺序依赖，空间关系是局部且并行的; (2) 自回归模型无法在一次前向传播中捕获全局上下文; (3) 双向 SSM（如 Vim）引入了额外延迟且训练困难; (4) 因此 ViT 和 CNN 骨干仍然优于最佳的 Mamba 视觉模型。
3. **核心矛盾**：Mamba 在序列建模上高效但缺乏全局感受野，Transformer 有全局感受野但计算昂贵。如何将两者优势互补是关键。
4. **本文目标**：系统设计一个 Mamba+Transformer 的混合架构，在精度和吞吐量上同时超越纯 Mamba 和纯 Transformer 模型。
5. **切入角度**：作者系统实验了不同混合模式（前层/中层/末层/均匀间隔加 Transformer），发现在最后几层加 self-attention 效果最好——SSM 在前面做局部特征提取，attention 在后面恢复全局上下文。
6. **核心 idea**：前两层用 CNN 快速提取特征，中间层用改进的 Mamba mixer，最后层用 self-attention 恢复全局信息，形成 "CNN → Mamba → Transformer" 的层级混合。

## 方法详解

### 整体框架
MambaVision 采用 4 阶段层级架构。输入图像通过 stem（两层 3×3 conv stride 2）转换为 H/4×W/4×C 的 patch。Stage 1-2 使用 CNN 残差块（BN + 3×3 conv + GELU），Stage 3-4 使用 MambaVision Mixer + Transformer 块。在 Stage 3/4 内部，前半数层用 MambaVision Mixer，后半数层用 self-attention。阶段间通过 3×3 conv stride 2 下采样。

### 关键设计

1. **MambaVision Mixer（重设计的视觉 SSM 块）**:
    - 功能：替代原始 Mamba block，使其更适合视觉任务
    - 核心思路：对输入 $X_{in}$ 做两个并行分支：(1) SSM 分支——通过线性层降为 $C/2$ 维，经 regular conv（替换 causal conv）+ SiLU + selective scan 得到 $X_1$; (2) 对称卷积分支——同样线性降维 + conv + SiLU 得到 $X_2$（无 SSM）。两个分支 concat 后通过线性层投影回 $C$ 维。公式：$X_{out} = \text{Linear}(\text{Concat}(X_1, X_2))$
    - 设计动机：(1) 用 regular conv 替换 causal conv 是因为视觉任务不需要因果约束; (2) 增加无 SSM 的对称分支是为了补偿 SSM 固有的序列化信息损失，确保全局空间信息不丢失; (3) 每个分支降为 $C/2$ 维以保持参数量与原始 Mamba block 相当

2. **层级混合策略（Transformer 放在末层）**:
    - 功能：在模型末端恢复全局上下文信息
    - 核心思路：在 Stage 3/4 中，给定 $N$ 层，前 $N/2$ 层用 MambaVision Mixer + MLP，后 $N/2$ 层用 self-attention + MLP。self-attention 采用窗口机制（Stage 3 窗口 14，Stage 4 窗口 7）
    - 设计动机：系统消融实验表明，将 Transformer 块放在最后几层比放在前面/中间/均匀分布都更好。原因是 SSM 在前面处理时已经积累了丰富的局部特征，最后用 attention 可以在紧凑的 token 空间中高效捕获全局依赖

3. **CNN 前端快速特征提取**:
    - 功能：在高分辨率阶段用 CNN 替代 Mamba/Transformer 以获得高吞吐量
    - 核心思路：Stage 1-2 使用简单的残差 CNN 块（两层 3×3 conv + BN + GELU + 残差连接），处理 H/4 和 H/8 分辨率的特征
    - 设计动机：高分辨率阶段 token 数量大，用 Mamba 或 attention 都会成为速度瓶颈。CNN 块计算密度高、硬件友好，在前两个 stage 用 CNN 可以大幅提升整体吞吐量

### 损失函数 / 训练策略
标准 ImageNet-1K 训练方案：300 epochs，32 块 A100 GPU，使用 DeiT 训练配方。下游检测用 Cascade Mask-RCNN 3× schedule，语义分割用 UperNet。

## 实验关键数据

### 主实验

| 模型 | Params | FLOPs | 吞吐量(Img/s) | Top-1 Acc |
|------|--------|-------|-------------|-----------|
| MambaVision-T | 31.8M | 4.4G | 6298 | 82.3% |
| Swin-T | 28.3M | 4.4G | 2758 | 81.3% |
| VMamba-T | 30.0M | 4.9G | 1282 | 82.6% |
| MambaVision-S | 50.1M | 7.5G | 4700 | 83.3% |
| Swin-S | 49.6M | 8.5G | 1720 | 83.2% |
| MambaVision-B | 97.7M | 15.0G | 3670 | 84.2% |
| ConvNeXt-B | 88.6M | 15.4G | 1485 | 83.8% |
| VMamba-B | 89.0M | 15.4G | 645 | 83.9% |
| MambaVision-L2 | 241.5M | 37.5G | 1021 | 85.3% |

MambaVision 在同等精度下吞吐量远超竞争对手（如 MambaVision-T 比 VMamba-T 快 ~5x）。

### 消融实验

**COCO 检测（Cascade Mask-RCNN）**:

| Backbone | AP_box | AP_mask |
|----------|--------|---------|
| MambaVision-T | 51.1 | 44.3 |
| Swin-T | 50.4 | 43.7 |
| ConvNeXt-T | 50.4 | 43.7 |
| MambaVision-S | 52.3 | 45.2 |
| MambaVision-B | 52.8 | 45.7 |

**ADE20K 语义分割（UperNet）**:

| Backbone | mIoU |
|----------|------|
| MambaVision-T | 46.0 |
| Swin-T | 44.5 |
| MambaVision-B | 49.1 |
| Swin-B | 48.1 |

### 关键发现
- **吞吐量优势巨大**：MambaVision-T（6298 img/s）比 VMamba-T（1282 img/s）快近 5 倍，比 Swin-T（2758 img/s）快 2 倍以上
- **Transformer 放在末层是最佳混合策略**：实验证明 SSM 做前端特征提取、attention 做后端全局聚合的组合最优
- **MambaVision Mixer 中的对称分支至关重要**：去掉无 SSM 的对称 conv 分支后精度显著下降
- **下游任务一致性好**：在检测和分割任务中持续优于 ConvNeXt 和 Swin

## 亮点与洞察
- **系统性混合研究**：不是随意堆叠 Mamba+Transformer，而是穷举了多种混合模式并给出最佳方案。"SSM 在前、attention 在后"的结论对后续混合架构设计有指导意义
- **对称无 SSM 分支**：在 SSM 分支旁边加一条纯 conv 分支来补偿序列化信息损失，简洁有效。这种"主路径+补偿路径"的设计模式可以迁移到任何将序列模型应用于非序列数据的场景
- **A100 实测吞吐量**：论文一直强调实测吞吐量而非 FLOPs，更具实际参考价值

## 局限与展望
- 论文未对 MambaVision Mixer 的各组件做细粒度消融（如对称分支的维度比例、conv kernel size 等）
- MambaVision-L2 虽然达到 85.3%，但参数量（241M）和 FLOPs（37.5G）已经偏大
- 窗口 attention 的窗口大小（14/7）似乎是手动设定的，未做充分搜索
- 与 MambaOut 的结论形成有趣对比——MambaOut 认为分类不需要 SSM，但 MambaVision 用 SSM+attention 混合仍然获得了竞争力

## 相关工作与启发
- **vs MambaOut**: MambaOut 证明纯 Gated CNN 在分类上够用，MambaVision 则证明 SSM+attention 混合能在保持高吞吐量的同时获得更好的精度-速度权衡
- **vs VMamba**: VMamba 用四向 Cross-Scan 但吞吐量低（645 img/s），MambaVision 用重设计的单向 SSM+对称分支，吞吐量高 5 倍
- **vs Swin Transformer**: MambaVision 在所有尺度上精度+吞吐量双赢，特别是 MambaVision-T 比 Swin-T 精度高 1% 且吞吐量快 2.3 倍
- **vs EfficientVMamba**: EfficientVMamba 在高分辨率用 SSM、低分辨率用 CNN；MambaVision 完全相反——高分辨率用 CNN、低分辨率用 SSM+attention，效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统研究 Mamba+Transformer 混合的视觉骨干，MambaVision Mixer 设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分类/检测/分割，多尺度对比，吞吐量实测，消融设计合理
- 写作质量: ⭐⭐⭐⭐ 结构完整，图示清晰，但技术细节部分可以更精炼
- 价值: ⭐⭐⭐⭐ 提供了实用的高效视觉骨干和系统性的混合架构设计指导

<!-- RELATED:START -->

## 相关论文

- [MambaOut: Do We Really Need Mamba for Vision?](mambaout_do_we_really_need_mamba_for_vision.md)
- [TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](../../ICCV2025/segmentation/tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [Rethinking Query-Based Transformer for Continual Image Segmentation](rethinking_query-based_transformer_for_continual_image_segmentation.md)
- [QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)

<!-- RELATED:END -->
