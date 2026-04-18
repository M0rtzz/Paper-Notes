---
title: >-
  [论文解读] Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency
description: >-
  [CVPR 2025][Vision Mamba] 提出 Adventurer 系列视觉模型，通过"头部平均池化 token"和"层间翻转"两个简单设计将图像输入适配到单向因果扫描框架中，使 Mamba 架构在视觉任务上实现 4-6 倍于现有 Vision Mamba 的训练速度，同时保持与 ViT 相当甚至更优的精度。
tags:
  - CVPR 2025
  - Vision Mamba
  - 因果图像建模
  - 线性复杂度
  - 状态空间模型
  - 高效视觉架构
---

# Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency

**会议**: CVPR 2025  
**arXiv**: [2410.07599](https://arxiv.org/abs/2410.07599)  
**代码**: https://github.com/wangf3014/Adventurer  
**领域**: 模型架构 / 视觉骨干网络  
**关键词**: Vision Mamba, 因果图像建模, 线性复杂度, 状态空间模型, 高效视觉架构

## 一句话总结
提出 Adventurer 系列视觉模型，通过"头部平均池化 token"和"层间翻转"两个简单设计将图像输入适配到单向因果扫描框架中，使 Mamba 架构在视觉任务上实现 4-6 倍于现有 Vision Mamba 的训练速度，同时保持与 ViT 相当甚至更优的精度。

## 研究背景与动机

**领域现状**：Vision Transformer (ViT) 是当前主流的视觉骨干网络，但其自注意力机制的二次复杂度在处理高分辨率、细粒度图像时面临严重的计算和内存瓶颈。Mamba 等状态空间模型 (SSM) 因线性复杂度被引入视觉领域，但现有 Vision Mamba 架构（如 Vim、VMamba）通常需要多路扫描来弥补因果建模的信息不均衡问题。

**现有痛点**：多路扫描虽然不显著增加参数量，但实际计算量和推理时间成倍增加，导致 Vision Mamba 在实际训练速度上甚至不如 ViT。例如 Vim-Base 的训练吞吐量仅为约 200 images/s，远低于 DeiT-Base 的 861 images/s。

**核心矛盾**：因果建模中的信息不均衡问题——序列尾部的 token 能聚合前面所有 token 的信息，而序列头部的 token 缺乏全局上下文，形成严重的表征质量差异。现有方案用多路扫描解决这个问题，但牺牲了效率。

**本文目标** 如何在只使用单向扫描的前提下，解决因果视觉建模中的信息不均衡问题，从而真正释放 Mamba 的线性复杂度优势。

**切入角度**：作者从人眼的扫视机制 (Saccade Mechanism) 获得启发——人眼每次只能聚焦很小的区域，通过快速扫视来理解复杂场景，与因果建模的单向扫描方式天然契合。

**核心 idea**：用头部平均 token 提供全局信息起点 + 层间翻转消除位置偏差，仅用单向扫描就能达到多路扫描的效果。

## 方法详解

### 整体框架
Adventurer 遵循 ViT 的基本架构：图像分块 → patch embedding → 位置编码 → L 个因果 block → class token 分类。每个 block 由一个因果 token mixer（默认 Mamba-2）和一个 channel mixer（SwiGLU MLP）组成。关键不同在于两处改造：(1) 每层输入前插入一个全局平均池化 token 作为序列起点；(2) 每两层之间翻转 patch token 序列顺序。class token 放在序列末尾。

### 关键设计

1. **Heading Average（头部平均 token）**:

    - 功能：为因果序列提供全局上下文起点，解决序列头部 token 信息匮乏问题
    - 核心思路：在每一层的输入开头插入一个平均 token $x_{\text{AVG}} = \frac{1}{n+1}\sum_j x_j$，它压缩了所有 patch token 的全局信息。因果扫描时，序列中的每个 token 都能至少通过这个头部 token "看到"全局概况。每层处理完毕后丢弃该 token 的输出，在下一层重新计算——确保每层都有最新的全局信息
    - 设计动机：直接解决因果建模中头部 token 无法获取后方信息的根本问题。实验对比了多种替代方案（复制 cls token、可学习新 token、多个细粒度 token），全局平均效果最好且最简单

2. **Inter-Layer Flipping（层间翻转）**:

    - 功能：消除因位置差异导致的信息不均衡，使模型学到方向不变的特征
    - 核心思路：每两个 block 之间，将 patch token 序列反转（cls token 位置不变）。这样原本处于序列头部（信息较少）的 token，在下一层变成尾部（信息最丰富），反之亦然。交替进行的扫描方向使每个 token 在不同层中交替获得不同方向的上下文
    - 设计动机：相比多路扫描（2 路、4 路），层间翻转不增加任何计算量（翻转本身几乎零开销），但能达到相似的性能。实验表明翻转比头部平均贡献更大（+0.9% vs +0.5%），可能因为翻转还额外促进了方向不变特征的学习

3. **Mamba-2 Token Mixer + SwiGLU Channel Mixer**:

    - 功能：提供高效的序列建模和通道混合
    - 核心思路：Token mixer 采用 Mamba-2（结构化 SSM 的最新版本）替代自注意力，扩展比 2x，特征维度为 256 的倍数以充分利用并行效率。Channel mixer 采用 SwiGLU MLP，隐层维度为输入的 2.5x（而非标准 MLP 的 4x），在减少计算量的同时通过门控机制增强表达能力
    - 设计动机：消融实验显示纯 Mamba 层（无 channel mixer）虽然可行但速度慢 1.3x，且精度略低。SwiGLU MLP 比标准 MLP 效果更好（+0.1~0.2%），且线性层对硬件更友好

### 损失函数 / 训练策略
采用多阶段训练：300 epoch 在 128×128 预训练 → 100 epoch 在 224×224 训练 → 20 epoch 在 224×224 微调（更强数据增强和更高 drop path rate）。等效约 230 epoch 的 224×224 训练，优于常用的 300 epoch 方案。

## 实验关键数据

### 主实验

| 模型 | Token Mixer | 输入 | 参数量 | 吞吐量 (img/s) | ImageNet Acc |
|------|------------|------|--------|---------------|-------------|
| DeiT-Small | Self-Attn | 224 | 22M | 1924 | 79.8% |
| Vim-Small | Mamba | 224 | 26M | 395 | 80.5% |
| MambaReg-S | Mamba | 224 | 28M | 391 | 81.4% |
| **Adventurer-Small** | Mamba | 224 | 44M | **1405** | **81.8%** |
| DeiT-Base | Self-Attn | 224 | 86M | 861 | 81.8% |
| Vim-Base* | Mamba | 224 | 98M | ~200 | ~81.9% |
| **Adventurer-Base** | Mamba | 224 | 99M | **856** | **82.6%** |
| DeiT-Base | Self-Attn | 384 | 86M | 201 | 83.1% |
| **Adventurer-Base** | Mamba | 448 | 99M | **216** | **84.3%** |

下游任务（ADE20k 语义分割 / COCO 检测）：Adventurer-Base 在 ADE20k 上达 46.6% mIoU（同等速度下最优），COCO 上 AP^b 48.4%。

### 消融实验

| 配置 | Tiny Acc | Small Acc | Base Acc |
|------|----------|-----------|----------|
| Naive Causal (无 HA/ILF) | - | 80.3% | - |
| + Heading Average | - | 80.8% (+0.5) | - |
| + Inter-Layer Flipping | - | 81.2% (+0.9) | - |
| + Both (完整模型) | 78.2% | **81.8% (+1.5)** | 82.6% |
| DeiT Causal → +Both | 78.8→79.9 | - | - |
| DeiT Standard (非因果) | 79.9 | - | - |

Channel mixer 消融：纯 Mamba 层 81.6%，+标准 MLP 81.7%，+SwiGLU 81.8%（Small）。

### 关键发现
- 层间翻转 (ILF) 的贡献约为头部平均 (HA) 的两倍（+0.9 vs +0.5），且两者互补
- 配合 HA+ILF 后，因果 DeiT 完全匹配标准 DeiT 的精度，证明因果建模不损失表达能力
- 在 1280×1280 高分辨率下，Adventurer-Base 比 ViT-Base 快 11.7x、省内存 14.0x
- 随着 patch 大小减小（序列变长），Adventurer 精度持续提升且速度优势更加显著

## 亮点与洞察
- **单向扫描等效多路扫描**：用零计算开销的层间翻转替代计算量翻倍的多路扫描，是非常优雅的工程与理论结合。这个思路可以迁移到任何需要双向信息流的序列模型
- **因果建模 = 标准 ViT 精度**：消融实验严格证明了配合简单机制后，因果模型完全不损失表达能力，暗示标准 ViT 的全可见注意力中约一半计算是冗余的
- **高分辨率场景的杀手级优势**：随着输入分辨率增加，二次复杂度的 ViT 急剧变慢，而 Adventurer 的线性复杂度使其能在 3000+ token 序列上以 5x 速度运行

## 局限与展望
- 模型仅在 ImageNet-1k 上训练，未探索大规模预训练（如 ImageNet-21K 或自监督预训练）的效果
- 位置编码仍用简单可学习矩阵，未针对因果场景优化（如 RoPE 等旋转位置编码）
- 当前只验证了分类、分割、检测任务，未涉及生成式任务或多模态场景
- SwiGLU MLP 的 2.5x 扩展比是经验选择，缺乏系统的超参搜索

## 相关工作与启发
- **vs Vim**: Vim 使用前向+后向双路扫描，Adventurer 用层间翻转的单路扫描达到相似效果但快 4-5x
- **vs VMamba**: VMamba 结合 Mamba 和 2D 卷积进行多方向扫描，结构更复杂，Adventurer 更简洁高效
- **vs MambaReg**: MambaReg 引入注册 token 并多路扫描，精度略高但训练成本 3-4x
- 对高分辨率视觉理解（如医学图像、遥感图像）有直接应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心创新（HA+ILF）简单但有效，思路清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 分类/分割/检测全覆盖，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 火炬探险者的比喻生动，结构清晰
- 价值: ⭐⭐⭐⭐ 对高分辨率视觉任务有实际价值，但需要更多大规模验证
