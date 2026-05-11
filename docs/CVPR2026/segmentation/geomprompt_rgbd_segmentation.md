---
title: >-
  [论文解读] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth
description: >-
  [CVPR 2026][图像分割][RGB-D语义分割] GeomPrompt 为冻结的 RGB-D 分割模型学习轻量的几何提示模块，从 RGB 合成任务驱动的深度代理信号（无深度监督），在深度缺失时提升 6.1 mIoU，在深度退化时提升最高 3.6 mIoU。
tags:
  - CVPR 2026
  - 图像分割
  - RGB-D语义分割
  - 深度缺失
  - 模态鲁棒性
  - 几何提示
  - 轻量适配
---

# GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth

**会议**: CVPR 2026  
**arXiv**: [2604.11585](https://arxiv.org/abs/2604.11585)  
**代码**: [https://geomprompt.github.io](https://geomprompt.github.io)  
**领域**: 语义分割  
**关键词**: RGB-D语义分割, 深度缺失, 模态鲁棒性, 几何提示, 轻量适配

## 一句话总结
GeomPrompt 为冻结的 RGB-D 分割模型学习轻量的几何提示模块，从 RGB 合成任务驱动的深度代理信号（无深度监督），在深度缺失时提升 6.1 mIoU，在深度退化时提升最高 3.6 mIoU。

## 研究背景与动机

**领域现状**：RGB-D 语义分割通过融合深度信息提升性能，但实际部署中深度传感器经常失效、不完整或噪声严重（反射/透明表面、传感器故障等）。

**现有痛点**：(1) 深度作为特权信息蒸馏到 RGB 仍需深度监督；(2) 单目深度估计作为代理需要额外模型且目标是重建深度而非优化分割；(3) 缺乏直接针对"深度缺失/退化时如何维持分割性能"的轻量解决方案。

**核心矛盾**：RGB-D 分割器期望深度输入提供几何先验，但部署时深度可能不可用或不可靠。关键问题是：能否学习一个"足够好"的几何信号来满足分割器，而不需要真正重建深度？

**核心 idea**：学习"任务驱动的几何提示"而非"重建深度"——只用分割损失训练提示生成模块，让它自动发现对分割最有用的几何信号。

## 方法详解

### 整体框架
冻结的 RGB-D 分割模型（如 DFormer、GeminiFusion）→ GeomPrompt 模块从 RGB 生成几何提示替代缺失深度 / GeomPrompt-Recovery 模块在退化深度上预测校正残差 → 提示经归一化+PromptAdapter+低通投影后送入分割器的深度通道。

### 关键设计

1. **GeomPrompt（深度缺失）**:

    - 功能：从 RGB 合成任务相关的几何提示
    - 核心思路：ViT-S/16 编码器提取 RGB 特征 → 轻量 CNN 解码器预测低分辨率残差图 → 抗锯齿上采样到全分辨率 → 以中性灰 127.5 为先验加上 $s \cdot \tanh$ 有界残差 → 归一化+PromptAdapter（零初始化残差模块）+低通投影。全程只用分割损失训练
    - 设计动机：不重建深度而是学习"对分割有用的几何信号"，避免了深度估计器和分割器目标不一致的问题

2. **GeomPrompt-Recovery（深度退化）**:

    - 功能：修复退化深度中对分割有害的部分
    - 核心思路：双路径——RGB ViT 分支 + 轻量深度条件编码器（4 层 stride-2 CNN），特征拼接融合后预测有界残差校正 $p_{raw} = \text{clamp}(\tilde{d} + s \cdot \tanh(\Delta_{full}), 0, 255)$。校正头零初始化，起始时为恒等映射
    - 设计动机：退化深度可能大部分仍有用，只需修复对分割有害的部分。零初始化确保模型从"不改变深度"开始，只学习必要的修正

3. **参数化与正则化**:

    - 功能：确保生成的提示稳定且在分割器的预期输入空间内
    - 核心思路：有界残差（tanh 限幅）+ 渐进缩放因子（训练早期残差幅度小，逐步放开）+ TV 平滑正则 + L1 幅度正则 + 低通投影（抑制高频伪影）
    - 设计动机：分割器在正常深度上训练，生成的提示必须"看起来像深度"才能被正确处理

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{seg}(\text{OHEM CE}) + \lambda_{tv} \mathcal{L}_{tv}(p_{raw}) + \lambda_\delta \|\Delta\|_1$。GeomPrompt-Recovery 训练时随机合成多种深度退化（空间丢失、量化、噪声）。

## 实验关键数据

### 主实验

| 设置 | 模型 | 基线(RGB-only) | + GeomPrompt | 提升 |
|------|------|---------------|-------------|------|
| 深度缺失 | DFormer | 43.8 mIoU | 49.9 mIoU | +6.1 |
| 深度缺失 | GeminiFusion | 47.2 mIoU | 50.2 mIoU | +3.0 |
| 深度退化(严重) | DFormer | 45.x mIoU | +3.6 mIoU | 改善 |

### 消融实验

| 配置 | mIoU | 延迟 | 说明 |
|------|------|------|------|
| GeomPrompt | 49.9 | 7.8ms | 轻量高效 |
| Depth Anything V2 | 50.1 | 38.3ms | 类似精度但慢 5x |
| Metric3Dv2 | 49.6 | 71.9ms | 更慢且精度不如 |
| 中性灰填充 | 43.8 | 0ms | 基线 |

### 关键发现
- GeomPrompt 以 7.8ms 延迟达到与 38.3ms 的 Depth Anything V2 竞争性的精度，效率优势明显
- 任务驱动的几何提示不需要是精确深度图——分割器只需要"足够好"的几何先验
- 零初始化+渐进缩放的训练策略对稳定性至关重要

## 亮点与洞察
- **范式转换**：从"估计深度"到"生成对任务有用的几何信号"，省去了深度监督和额外预训练
- **即插即用**：可应用于任何冻结的 RGB-D 分割器，无需修改骨干

## 局限与展望
- 需要为每个分割器分别训练 GeomPrompt
- 极端退化下的恢复能力有限
- 未来可探索跨分割器通用的几何提示

## 相关工作与启发
- **vs Depth Anything V2**: 通用深度估计目标与分割目标不完全一致，GeomPrompt 直接优化分割
- **vs 特权信息蒸馏**: 蒸馏方法修改骨干权重，GeomPrompt 完全不动骨干

## 评分
- 新颖性: ⭐⭐⭐⭐ "任务驱动的几何提示"视角新颖
- 实验充分度: ⭐⭐⭐⭐ 缺失+退化两种设置都有评估
- 写作质量: ⭐⭐⭐⭐⭐ 动机极其清晰，参数化设计细致
- 价值: ⭐⭐⭐⭐ 对机器人/嵌入式感知有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [\[CVPR 2026\] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)

</div>

<!-- RELATED:END -->
