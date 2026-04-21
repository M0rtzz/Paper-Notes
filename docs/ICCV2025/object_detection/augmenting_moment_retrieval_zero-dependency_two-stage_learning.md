---
title: >-
  [论文解读] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning
description: >-
  [ICCV 2025][目标检测][时刻检索] 提出 AMR 框架，通过 Splice-and-Boost 数据增强策略和冷启动-蒸馏两阶段训练，在不依赖任何外部数据/预训练模型的前提下，大幅提升视频时刻检索的边界感知能力和语义辨别力，在 QVHighlights 上超越 SOTA +5%。
tags:
  - ICCV 2025
  - 目标检测
  - 时刻检索
  - 数据增强
  - 知识蒸馏
  - DETR
  - 视频时序定位
---

# Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning

**会议**: ICCV 2025  
**arXiv**: [2510.19622](https://arxiv.org/abs/2510.19622)  
**代码**: https://github.com/SooLab/AMR  
**领域**: 视频理解  
**关键词**: 时刻检索, 数据增强, 知识蒸馏, DETR, 视频时序定位

## 一句话总结

提出 AMR 框架，通过 Splice-and-Boost 数据增强策略和冷启动-蒸馏两阶段训练，在不依赖任何外部数据/预训练模型的前提下，大幅提升视频时刻检索的边界感知能力和语义辨别力，在 QVHighlights 上超越 SOTA +5%。

## 研究背景与动机

**领域现状**：Moment Retrieval（时刻检索）旨在给定文本描述后，精确定位长视频中目标片段的起止时间。当前主流方法基于 DETR 框架进行端到端定位，但受限于标注稀缺和标注模糊性。

**现有痛点**：
   - **数据稀缺**：单段视频的密集标注代价高昂，导致训练样本不足，模型只学到文本关键词与局部视觉特征的浅层关联，而非理解动作的时间完整性。
   - **边界模糊**：相邻事件（如"准备投篮"和"完成投篮"）之间的过渡区域，边界信号不明确。
   - **细粒度语义混淆**：现有方法难以区分语义相近但逻辑不同的片段（如"踢球"与"扔球"都涉及球的运动）。

**现有尝试的局限**：已有方法沿三条路径探索——多模态交互模块（注意力区分能力有限）、大规模预训练迁移（参数量过大，部署成本高）、外部大模型合成数据（引入外部依赖）。均未解决核心矛盾：如何在零外部依赖下从有限标注中挖掘判别性时空推理模式。

**切入角度**：作者观察到可以通过结构化拆解和重组已有数据来构造具有清晰边界和语义干扰的增强样本，并设计两阶段训练策略实现从增强数据到真实数据的知识迁移。

**核心idea**：Splice-and-Boost 增强 + 冷启动蒸馏两阶段训练 = 零外部依赖的 Moment Retrieval 性能突破。

## 方法详解

### 整体框架

AMR 遵循标准的 DETR 框架流程：视频特征通过 CLIP + SlowFast 提取（$V \in \mathbb{R}^{L \times C}$），文本特征通过 CLIP 文本编码器提取（$T \in \mathbb{R}^{K \times C}$），经 Transformer 编码器进行跨模态交互，解码器使用 $N$ 个可学习 query 生成时间段预测。创新集中在数据增强和训练策略两个维度。

### 关键设计

1. **Splice（时序拼接）增强**：

    - 功能：构造具有清晰边界的增强训练样本
    - 核心思路：从原始样本中提取前景目标片段，随机降采样后，在另一视频的背景时间线上随机找到等长窗口删除，然后将目标片段无缝插入该空缺位置
    - 设计动机：通过将片段语义与背景依赖解耦，强化模型独立于上下文进行事件定位的能力，增强边界感知

2. **Boost（语义增强）机制**：

    - 功能：引入语义模糊的困难负样本，提升模型对相似事件的辨别力
    - 核心思路：采用交叉验证范式——将训练集分成两个互斥子集 $\mathcal{D}_1$ 和 $\mathcal{D}_2$，在一个子集上训练模型，在另一个子集上检测高置信度误报（$\text{IoU}(W_i, W_j^{gt}) = 0$ 且置信度 $S_i > \theta$）。交换子集角色迭代收集模糊片段，与真实目标片段一起拼接到背景视频中
    - 设计动机：联合优化真实片段的定位与模糊片段的抑制，在不引入外部数据的前提下平衡多样性和判别力

3. **冷启动阶段（Cold-Start）**：

    - 功能：在增强数据上建立基础的边界定位和语义辨别能力
    - 核心思路：课程学习策略，仅使用增强数据训练 40 epoch。增强数据中显式区分前景/背景并包含挑战性片段，建立基础判别力
    - 设计动机：为蒸馏阶段提供强知识基线；但增强数据的锐利边界可能导致模型过度依赖突变视觉特征

4. **蒸馏适应阶段（Dual-path Distillation）**：

    - 功能：将冷启动知识迁移到真实数据分布，同时防止遗忘
    - 核心思路：引入双路查询——原始查询 $Q^{\text{ori}}$ 保持 DETR 定位能力，活跃查询 $Q^{\text{act}}$ 动态适应真实数据，各配独立 FFN。蒸馏损失约束原始查询与冻结基础查询的分布一致性：$\mathcal{L}_{\text{dill}} = 1 - \frac{1}{N}\sum_{i}\frac{(\hat{Q}^{\text{ori}}_i)^T \hat{Q}^{\text{base}}_i}{\|\hat{Q}^{\text{ori}}_i\| \|\hat{Q}^{\text{base}}_i\|}$
    - 设计动机：通过参数隔离和相似性驱动的正则化桥接合成到真实的域间差距

5. **判别对比损失（Discriminative Contrastive Loss）**：

    - 功能：保持模型对模糊片段的辨别力
    - 核心思路：帧级文本-视频余弦相似度 $M_i$，构造相对排序约束而非绝对分数压制：$\mathcal{L}_{\text{disc}} = -\log\frac{\exp(p/\tau)}{\exp(p/\tau) + \exp(n/\tau)}$，其中 $p$/$n$ 分别为真实/模糊区域的平均匹配分数
    - 设计动机：鼓励真实事件与语义邻居之间保持更高的区分边界

### 损失函数 / 训练策略

- **冷启动阶段**：$\mathcal{L} = \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}} + \lambda_{\text{sal}}\mathcal{L}_{\text{sal}}$
- **蒸馏阶段**：$\mathcal{L} = \mathcal{L}_{\text{stage1}} + 0.5 \cdot \mathcal{L}_{\text{dill}} + 0.5 \cdot \mathcal{L}_{\text{disc}}$
- 匈牙利匹配建立预测-真值对应；第一阶段 40 epoch，第二阶段 100 epoch；AdamW，lr=1e-4

## 实验关键数据

### 主实验

**QVHighlights 数据集（验证集/测试集）**：

| 方法 | R1@0.5 (Val) | R1@0.7 (Val) | mAP Avg. (Val) | R1@0.5 (Test) | mAP Avg. (Test) |
|------|-------------|-------------|---------------|--------------|----------------|
| M-DETR | 53.94 | 34.84 | 32.20 | 52.89 | 30.73 |
| QD-DETR | 62.68 | 46.66 | 41.22 | 62.40 | 39.86 |
| TR-DETR | 67.10 | 51.48 | 45.09 | 64.66 | 42.62 |
| BAM-DETR | 65.10 | 51.61 | 47.61 | 62.71 | 45.36 |
| **AMR (Ours)** | **70.13** | **56.65** | **51.66** | **68.22** | **48.43** |

验证集上比 BAM-DETR 的 R1@0.5 提升 +5.03%，R1@0.7 提升 +5.04%，平均 mAP 提升 +4.05%。

**Charades-STA & TACoS**：

| 方法 | Charades R1@0.5 | Charades mIoU | TACoS R1@0.5 | TACoS mIoU |
|------|----------------|--------------|-------------|-----------|
| UniVTG | 58.01 | 50.10 | 34.97 | 33.60 |
| UVCOM | 59.25 | - | 36.39 | - |
| **AMR** | **62.02** | **52.58** | **40.91** | **38.22** |

### 消融实验

| 设置 | Splice | Boost | 两阶段 | Dill | DCL | mAP Avg. |
|------|--------|-------|--------|------|-----|---------|
| (a) Baseline | - | - | - | - | - | 44.82 |
| (c) 仅增强 | ✓ | ✓ | - | - | - | 43.88 |
| (f) 增强+两阶段 | ✓ | ✓ | ✓ | - | - | 48.07 |
| (l) 全部模块 | ✓ | ✓ | - | ✓ | ✓ | **51.66** |

- 单用数据增强（无两阶段训练）反而低于 baseline（43.88 vs 44.82），验证了增强-真实数据分布差异问题
- 两阶段训练 + 增强显著提升至 48.07；蒸馏和对比损失进一步推至 51.66

## 个人思考

- **亮点**：零外部依赖的设计思路实用，Boost 机制用交叉验证挖掘困难负样本方法巧妙，双路查询蒸馏范式优雅
- **局限**：两阶段训练增加复杂度（140 epoch）；Splice 拼接可能引入不自然的视觉突变
- **启发**：双路查询 + 蒸馏的合成-真实知识迁移范式，可推广到其他需要数据增强辅助的视频理解任务

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)
- [Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)
- [Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](../../ECCV2024/object_detection/spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)
- [Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)

<!-- RELATED:END -->
