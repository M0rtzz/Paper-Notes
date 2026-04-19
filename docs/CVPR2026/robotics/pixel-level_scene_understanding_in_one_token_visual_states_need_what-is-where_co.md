---
title: >-
  [论文解读] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition
description: >-
  [CVPR 2026][机器人][视觉状态表示] 本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息，在Franka Kitchen和DMC机器人策略学习benchmark上达到SOTA。
tags:
  - CVPR 2026
  - 机器人
  - 视觉状态表示
  - 自监督学习
  - 机器人策略学习
  - 瓶颈token
  - 全局-局部重建
---

# Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition

**会议**: CVPR 2026  
**arXiv**: [2603.13904](https://arxiv.org/abs/2603.13904)  
**代码**: [项目主页](https://seokminlee-chris.github.io/CroBo-ProjectPage)  
**领域**: 机器人 / 自监督学习  
**关键词**: 视觉状态表示, 自监督学习, 机器人策略学习, 瓶颈token, 全局-局部重建

## 一句话总结
本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息，在Franka Kitchen和DMC机器人策略学习benchmark上达到SOTA。

## 研究背景与动机
1. **领域现状**：自监督视觉表示学习（MAE、DINO、DINOv2等）在图像分类和语义分割等任务上表现优异，但这些方法没有显式考虑"好的视觉状态表示应该编码什么"。对于机器人的序列决策任务，需要将原始视觉观察压缩为紧凑的视觉状态。
2. **现有痛点**：(1) 对比学习方法（DINO）学习的是全局语义特征，对空间位置不敏感；(2) MAE通过patch级重建学习局部特征，但没有全局场景理解；(3) SiamMAE/CropMAE学习patch级对应关系，不适合需要单个紧凑表示的下游任务；(4) ToBo引入了瓶颈token概念，但基于时序帧重建，未显式约束空间组成编码。
3. **核心矛盾**：机器人需要检测场景中的微妙变化（如机械臂移动了几厘米），这要求视觉状态同时编码"是什么物体"和"在哪里"。但现有SSL方法要么丢失空间信息（对比学习的增强不变性），要么没有单一紧凑表示（patch级方法输出tokens序列）。
4. **本文目标** 学习一个单token的视觉状态表示，使其编码完整的"what-is-where"场景组成——哪些语义实体存在、它们在哪里、如何排列。
5. **切入角度**：如果一个紧凑表示能够从极少线索中恢复出任意局部裁剪区域的内容，那它必然编码了整个场景的语义和空间信息。
6. **核心 idea**：用单个[CLS] token作为全局瓶颈，重建90%遮蔽的局部裁剪视图，迫使瓶颈token成为像素级场景组成的压缩表示。

## 方法详解

### 整体框架
从视频中采样单帧，构建两个视图：全局源视图$\mathbf{x}^g$（大裁剪，尺度[0.5, 1.0]）和局部目标视图$\mathbf{x}^l$（从源视图中小裁剪，尺度[0.3, 0.6]）。共享权重的Siamese编码器处理两个视图：源视图不遮蔽，目标视图90%遮蔽。Transformer解码器利用源视图的[CLS]瓶颈token和目标视图的少量可见patch token来重建被遮蔽的目标patch。

### 关键设计

1. **全局-局部重建目标**:

    - 功能：迫使[CLS]瓶颈token编码全场景的语义-空间组成信息
    - 核心思路：目标视图$\mathbf{x}^l$是源视图$\mathbf{x}^g$的子区域，90%的目标patch被遮蔽后仅剩约14个可见patch。解码器需要从两个信号重建：(a) 源视图的[CLS]token提供全局场景上下文；(b) 目标视图的少量可见patch提供局部线索。由于可见patch太少，解码器必须重度依赖[CLS]token来推断遮蔽区域的内容，因此[CLS]token被迫编码"每个物体是什么、在场景中的哪个位置"
    - 设计动机：传统MAE从同一视图的可见patch重建被遮蔽patch，75%的遮蔽率下仍有足够的局部上下文。而CroBo的跨视图设计+90%的极高遮蔽率，使得仅靠局部线索无法完成重建，必须利用全局瓶颈

2. **Siamese编码器 + 高遮蔽率**:

    - 功能：共享权重确保表示空间一致，高遮蔽率强制依赖瓶颈
    - 核心思路：源视图和目标视图使用相同的ViT编码器。源视图全部patch输入编码器产生$N$个patch token和1个[CLS] token；目标视图仅10%的可见patch输入编码器。遮蔽率从75%（MAE标准）提高到90%，实验进一步证明95%时性能更好
    - 设计动机：如果遮蔽率太低（如75%），解码器可能仅从目标视图的可见patch就能重建，绕过瓶颈token。90%的遮蔽率确保可见信息不足以独立重建

3. **Crop vs Time的设计选择**:

    - 功能：选择最优的源-目标视图构建方式
    - 核心思路：对比了三种构建方式——Time（不同帧）、Crop（同帧不同裁剪）、Time+Crop（不同帧+裁剪）。Crop一致优于Time，因为目标视图是源视图的空间子集，重建目标明确且well-defined。Time引入运动/光照变化的歧义。Time+Crop效果最差，两种变化叠加使重建目标过于模糊
    - 设计动机：虽然CroBo不使用时序训练数据（仅用单帧），但在视频数据集上训练，目的是与先前工作公平对比

### 损失函数 / 训练策略
损失为遮蔽patch的MSE重建损失：$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{\mathbf{x}}_i^l - \mathbf{x}_i^l \|_2^2$。使用归一化像素目标。ViT-S/16编码器，8层解码器（512维），AdamW优化，batch size 1536，在Kinetics-400上训练400 epochs。

## 实验关键数据

### 主实验（Franka Kitchen成功率% + DMC归一化分数）

| 方法 | Knob on | Light on | Sdoor | Ldoor | Micro | walker/stand | walker/walk | reacher/easy |
|------|---------|----------|-------|-------|-------|-------------|-------------|-------------|
| MAE | 12.0 | 24.3 | 71.5 | 12.8 | 10.0 | - | - | - |
| DINOv2 | 25.0 | 46.6 | 87.8 | 17.6 | 21.8 | 81.2 | 38.2 | 87.6 |
| ToBo | 58.4 | 80.6 | 98.4 | 44.2 | 51.2 | 87.0 | 77.7 | 87.5 |
| **CroBo** | **65.6** | **87.6** | **99.4** | 41.2 | **64.8** | **92.0** | **80.8** | **95.8** |
| Δ SOTA | +7.2 | +7.0 | +1.0 | -3.0 | +13.6 | +5.0 | +3.1 | +8.3 |

### 消融实验

| 配置 | Knob1 | Light | Sdoor | Ldoor | Micro |
|------|-------|-------|-------|-------|-------|
| Time（时序帧） | 44.4 | 67.6 | 96.4 | 36.8 | 49.6 |
| **Crop（空间裁剪）** | **57.6** | **81.6** | **98.6** | **36.8** | **50.4** |
| Time+Crop | 38.2 | 62.8 | 90.8 | 22.2 | 28.4 |
| 遮蔽率75% | 41.4 | 70.6 | 94.0 | 22.6 | 35.0 |
| 遮蔽率90% | 57.6 | 81.6 | 98.6 | 36.8 | 50.4 |
| 遮蔽率95% | **59.0** | **86.6** | **99.4** | **41.2** | **58.0** |

### 关键发现
- **Crop显著优于Time**：同帧裁剪提供well-defined的重建目标，时序帧引入过多歧义。这颠覆了"需要视频才能学习动态表示"的直觉
- **遮蔽率越高越好**：从75%到95%，所有任务性能持续提升，说明更少的目标线索迫使模型更充分利用瓶颈token
- **ViT-S超越所有ViT-L baseline**：小模型（65.0%平均成功率）超过了所有大模型baseline（最好63.3%），增益来自更好的表示而非更大的模型
- **感知直线性分析**：CroBo在DAVIS视频上的表示曲率仅75.4°，远低于DINOv2的103.28°，说明CroBo的表示在时序上更线性，更好地编码了"什么在哪里移动"
- **重建可视化**：即使目标视图90%被遮蔽，CroBo能正确恢复被完全遮挡的物体（如CLEVR中两个青色球体都不可见但位置被正确重建），证明瓶颈token确实编码了完整的what-is-where信息

## 亮点与洞察
- **"不需要视频就能学动态表示"的发现**非常反直觉：CroBo仅用单帧的空间裁剪训练，在机器人动态任务上优于基于视频时序的方法。这说明空间组成理解是动态理解的基础
- **极高遮蔽率作为信息瓶颈的设计**巧妙结合了两个思想：MAE的重建学习 + 信息瓶颈原理。遮蔽率越高，瓶颈越紧，表示越被迫编码全局信息
- **感知直线性指标**为表示质量提供了新的度量维度：好的视觉状态表示应该使得观察轨迹在表示空间中近似线性，便于预测和控制

## 局限与展望
- 仅在Franka Kitchen和DMC两个仿真环境中评估，缺乏真实机器人实验
- 瓶颈token是单个token，信息容量有限，对于更复杂的场景可能不足
- 仅使用ViT-S/B/L评估，未探索更大或更新的视觉骨干（如ViT-G、DINOv2-L+）
- 重建目标使用像素MSE，可能不是最优的，感知损失或特征级重建可能更好
- Ldoor任务上不如ToBo，暗示某些需要强时序关联的任务仍需时序信息

## 相关工作与启发
- **vs ToBo**: CroBo的直接前身，ToBo用时序帧重建，CroBo用空间裁剪重建，后者更清晰
- **vs CropMAE**: 类似的空间裁剪思路，但CropMAE学习patch级对应关系，CroBo将信息压缩为单token
- **vs SiamMAE**: SiamMAE也是Siamese编码器+视频帧对，但没有瓶颈token设计
- **vs DINOv2**: 通用表示但不保留精细空间信息，在机器人任务上明显不如CroBo

## 评分
- 新颖性: ⭐⭐⭐⭐ what-is-where的概念清晰，全局-局部重建设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark、完整消融、重建可视化、感知直线性分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事逻辑性强，从"什么是好的视觉状态"出发推导出方法设计
- 价值: ⭐⭐⭐⭐ 对机器人视觉表示学习有重要启发，单帧训练胜时序训练的发现有广泛影响

<!-- RELATED:START -->

## 相关论文

- [DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [Towards Training-Free Scene Text Editing](towards_training-free_scene_text_editing.md)
- [ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)
- [FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](force_transferable_visual_jailbreaking_attacks_via_feature_over-reliance_correct.md)

<!-- RELATED:END -->
