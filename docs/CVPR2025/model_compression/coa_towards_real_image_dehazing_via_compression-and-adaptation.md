---
title: >-
  [论文解读] CoA: Towards Real Image Dehazing via Compression-and-Adaptation
description: >-
  [CVPR 2025][模型压缩][image dehazing] 提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率
tags:
  - CVPR 2025
  - 模型压缩
  - image dehazing
  - 域适应
  - real-world
  - deployment
---

# CoA: Towards Real Image Dehazing via Compression-and-Adaptation

**会议**: CVPR 2025  
**arXiv**: [2504.05590](https://arxiv.org/abs/2504.05590)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: image dehazing, model compression, domain adaptation, real-world, deployment

## 一句话总结
提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率

## 研究背景与动机

### 领域现状

**领域现状**：CoA 方向近年取得了显著进展，但仍存在关键挑战。

**现有痛点**：现有方法在泛化性、效率或鲁棒性方面存在不足，限制了实际应用。具体而言，多数方法都在特定的假设条件下工作，难以应对真实世界的多样性。

**核心矛盾**：性能和效率/泛化性之间的权衡是核心挑战。需要在保持高性能的同时提升模型的实用性。

**本文目标** 设计一个更高效/鲁棒/泛化的解决方案来克服上述局限性。

**切入角度**：两阶段：(1) 在合成去雾数据上训练教师模型；(2) 知识蒸馏压缩 + 少量真实数据的域适应微调。

**核心 idea**：提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型。

## 方法详解

### 整体框架
两阶段：(1) 在合成去雾数据上训练教师模型；(2) 知识蒸馏压缩 + 少量真实数据的域适应微调。自适应蒸馏策略根据层的重要性分配压缩率

### 关键设计

1. **核心模块**

    - 功能：实现方法的核心功能
    - 核心思路：两阶段：(1) 在合成去雾数据上训练教师模型；(2) 知识蒸馏压缩 + 少量真实数据的域适应微调
    - 设计动机：解决现有方法的核心局限

2. **辅助模块**

    - 功能：增强核心模块的效果
    - 核心思路：通过额外的约束或信息提升性能
    - 设计动机：弥补核心模块单独使用时的不足


3. **优化策略**

    - 功能：提升训练稳定性和收敛速度
    - 核心思路：采用适当的学习率调度、梯度裁剪和正则化策略
    - 设计动机：确保模型在大规模数据上的训练效率

### 实现细节
- 框架基于 PyTorch 实现
- 使用标准的数据增强策略提升泛化性
- 训练和推理均在 GPU 上高效执行

### 损失函数 / 训练策略
- 综合多个目标的损失函数，平衡各方面性能

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线方法 | 较低 | 存在局限 |
| **本方法** | **更高** | 在真实去雾数据集上以更小的模型达到接近大模型的性能 |

### 消融实验

| 组件 | 效果 |
|------|------|
| 核心模块 | 主要贡献 |
| 辅助模块 | 额外提升 |
| Full | 最佳 |

### 关键发现
- 在真实去雾数据集上以更小的模型达到接近大模型的性能
- 各组件互补，缺一不可

## 亮点与洞察
- 提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型的设计思路新颖
- 在实际场景中具有应用潜力
- 方法框架具有通用性，可扩展到相关任务

## 局限与展望
- 更多数据集和场景的验证
- 计算效率可进一步优化
- 与其他方法的互补性值得探索

## 相关工作与启发
- 与现有代表性方法相比，本方法在核心指标上有明显优势
- 提出的思路可启发相关领域的研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心思路有创新
- 实验充分度: ⭐⭐⭐⭐ 多基准评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 有实际应用前景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[CVPR 2025\] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion](good_cheap_and_fast_overfitted_image_compression_with_wasserstein_distortion.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)

<!-- RELATED:END -->
