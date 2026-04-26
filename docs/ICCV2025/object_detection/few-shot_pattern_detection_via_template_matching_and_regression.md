---
title: >-
  [论文解读] Few-Shot Pattern Detection via Template Matching and Regression
description: >-
  [ICCV 2025][目标检测][few-shot detection] 本文提出TMR方法，通过经典模板匹配结合支持条件化边界框回归，实现了对任意模式（包括非物体级模式）的小样本检测，同时引入RPINE数据集覆盖更广泛的重复模式，在多个基准上超越现有FSCD方法并展现出强大的跨数据集泛化能力。
tags:
  - ICCV 2025
  - 目标检测
  - few-shot detection
  - template matching
  - pattern detection
  - repetitive patterns
  - anchor-free detection
---

# Few-Shot Pattern Detection via Template Matching and Regression

**会议**: ICCV 2025  
**arXiv**: [2508.17636](https://arxiv.org/abs/2508.17636)  
**代码**: https://cvlab.postech.ac.kr/research/TMR  
**领域**: 目标检测 / 小样本检测  
**关键词**: few-shot detection, template matching, pattern detection, repetitive patterns, anchor-free detection

## 一句话总结
本文提出TMR方法，通过经典模板匹配结合支持条件化边界框回归，实现了对任意模式（包括非物体级模式）的小样本检测，同时引入RPINE数据集覆盖更广泛的重复模式，在多个基准上超越现有FSCD方法并展现出强大的跨数据集泛化能力。

## 研究背景与动机
- **领域现状**：小样本目标检测（FSOD）和小样本计数检测（FSCD）取得了显著进展，但现有方法高度依赖物体级先验，主要面向有明确边界的物体类别
- **现有痛点**：主流方法（如Counting-DETR、GeCo、PseCo）通过全局平均池化将支持样本压缩为原型向量（prototype），这一操作丢失了样本的空间结构信息
- **核心矛盾**：当检测目标从物体扩展到任意模式（如纹理、几何结构、物体部件等非物体模式）时，空间布局信息至关重要，但原型匹配恰恰丢弃了这些信息
- **本文要解决的问题**：如何设计一个不依赖物体先验、保留空间结构的小样本模式检测器
- **切入角度**：回归经典模板匹配（template matching），利用二维交叉相关保留示例的空间布局
- **核心idea**：用通道级模板匹配替代原型匹配，并通过支持条件化回归自适应调整边界框尺寸

## 方法详解

### 整体框架
TMR采用极简结构：冻结的SAM-ViT/H骨干提取特征图 → RoIAlign提取模板特征 → 通道级模板匹配产生相关图 → 拼接原始特征与匹配特征 → 边界框回归器 + 存在分类器预测结果 → NMS后处理。整个检测头仅由少量3×3卷积和线性层组成，无交叉注意力等复杂模块。

### 关键设计
1. **通道级模板匹配（Channel-wise Template Matching）**:

    - 功能：将模板特征 $\mathbf{T} \in \mathbb{R}^{t_h \times t_w \times D}$ 以滑窗方式在图像特征图 $\mathbf{F}$ 上进行交叉相关
    - 核心思路：$\mathbf{F}_{\text{TM}}(x,y) = \frac{1}{t_w t_h} \sum_{x',y'} \mathbf{F}(x+x'-\lfloor t_w/2 \rfloor, y+y'-\lfloor t_h/2 \rfloor) \mathbf{T}(x',y')$，结果保留通道维度 $\mathbf{F}_{\text{TM}} \in \mathbb{R}^{H \times W \times D}$
    - 设计动机：与原型匹配不同，通道级模板匹配保留了示例的空间结构和几何特征，这对检测非物体模式至关重要。消融实验表明，使用池化原型替代2D模板后AP显著下降（RPINE: 33.59→20.94）

2. **自适应模板提取（Adaptive Template Extraction）**:

    - 功能：使用RoIAlign从图像特征图中裁切模板特征
    - 核心思路：模板尺寸根据支持示例的实际尺寸自适应确定（向上取整），而非固定尺寸池化
    - 设计动机：保持模板与特征图之间的平移对齐关系，避免空间信息损失

3. **支持条件化边界框回归（Support-Conditioned Box Regression）**:

    - 功能：预测相对于支持示例尺寸的缩放和偏移参数，而非绝对坐标
    - 核心思路：对每个特征点预测 $(\Delta x, \Delta y, \alpha_w, \alpha_h)$，最终框为 $(x + s_w \Delta x, y + s_h \Delta y, e^{\alpha_w} s_w, e^{\alpha_h} s_h)$
    - 设计动机：利用示例尺寸作为参考基准，使模型能动态适应不同大小的示例和目标，消融实验证实条件化回归优于直接回归（AP: 36.01 vs 17.01）

### 损失函数 / 训练策略
- **存在损失** $\mathcal{L}_P$：带中心点扩展边距的二值交叉熵损失
- **边界框损失** $\mathcal{L}_B$：gIoU损失，仅在存在目标的位置计算
- **总损失**：$\mathcal{L} = \mathcal{L}_P + \mathcal{L}_B$
- 骨干网络SAM-ViT/H保持冻结，仅训练检测头（约19M可训练参数）
- 特征图从64×64插值到128×128以提高密集预测精度

## 实验关键数据

### 主实验

| 数据集 | 指标 | TMR | 之前SOTA (GeCo) | 提升 |
|--------|------|------|----------|------|
| RPINE (1-shot) | AP | 33.59 | 23.33 | +10.26 |
| RPINE (1-shot) | AP50 | 64.05 | 45.93 | +18.12 |
| FSCD-LVIS seen (3-shot) | AP | 27.49 | 22.37 (PseCo) | +5.12 |
| FSCD-LVIS unseen (3-shot) | AP | 22.71 | 11.47 (GeCo) | +11.24 |
| FSCD-147 (1-shot) | AP | 36.01 | 32.71 (GeCo) | +3.30 |
| FSCD-147 (3-shot) | AP | 38.57 | 32.49 (GeCo) | +6.08 |

### 消融实验

| 配置 | RPINE AP | FSCD-147 AP | 说明 |
|------|---------|-------------|------|
| 仅图像特征 $\mathbf{F}$ | 11.44 | 20.95 | 无示例信息下界 |
| 仅模板匹配特征 $\mathbf{F}_{\text{TM}}$ | 32.55 | 31.96 | 模板匹配的有效性 |
| $\mathbf{F} \oplus \mathbf{F}_{\text{PM}}$（原型匹配） | 20.94 | 28.91 | 池化原型丢失空间信息 |
| $\mathbf{F} \oplus \mathbf{F}_{\text{TM}}$（完整模型） | 33.59 | 36.01 | 空间结构+外观信息最优 |

### 关键发现
- SAM解码器在RPINE上反而降低性能（AP: 33.59→29.66），因为SAM倾向于对齐边缘，对非物体模式有害
- TMR的FLOPs（3.04T）显著低于PseCo（5.08T）和GeCo（4.72T）
- 跨数据集评估中TMR展现压倒性优势：在RPINE训练、FSCD-147测试时AP达41.39 vs GeCo的36.99

## 亮点与洞察
- 回归经典方法的成功案例：模板匹配这一古老技术配合现代特征提取器展现出惊人效果
- 极简架构设计：仅3×3卷积+线性层，无注意力机制，却达到SOTA
- RPINE数据集填补了非物体模式检测评测的空白，支持多模式标注（每张图最多3个不同模式，3个标注者独立标注）
- 揭示了原型匹配方法对物体语义的过拟合问题
- 关于SAM解码器的洞察特别有价值：边缘对齐特性对非物体模式有害，提醒社区不要盲目使用SAM后处理

## 局限与展望
- 依赖冻结的SAM骨干，对小实例的分辨率受限于ViT的patch大小
- RPINE数据集较小（4362张图），可能限制模型的多样性学习
- 仅处理2D模式，未涉及3D或视频中的时空模式
- 多尺度推理的计算开销可进一步优化
- 模板匹配对旋转和大尺度变化的鲁棒性有待验证
- 作者指出未来可探索轻量级模式特定架构，减少对物体级边缘先验的依赖

## 相关工作与启发
- **GeCo/PseCo**：基于原型匹配的FSCD方法，性能受限于空间信息损失
- **传统模板匹配**：TMR成功将经典方法与深度特征结合，值得其他检测任务借鉴
- **SAM骨干的通用性**：冻结SAM特征在小样本检测中表现优异，说明大模型特征的可迁移性
- **FSCD-147/FSCD-LVIS**：现有标准FSCD数据集，但仅覆盖物体级模式
- **启发**：在需要保留空间结构的匹配任务中（如点云配准、纹理分析），经典方法+现代特征可能是被忽视的有效组合
- **SEM图像应用**：作者展示了在扫描电子显微镜图像上的跨域检测效果，表明方法的实际应用潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ 将经典模板匹配与现代检测框架结合，提出支持条件化回归，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、跨数据集评估、详尽消融、计算复杂度分析、实际应用验证
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，motivation充分，图表丰富
- 价值: ⭐⭐⭐⭐ 将小样本检测从物体扩展到任意模式，开辟了新方向，RPINE数据集有长期价值

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Few-Shot Learner Generalizes Across AI-Generated Image Detection](../../ICML2025/object_detection/few-shot_learner_generalizes_across_ai-generated_image_detection.md)
- [\[ECCV 2024\] Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](../../ECCV2024/object_detection/tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](../../ECCV2024/object_detection/adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](../../ECCV2024/object_detection/openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [\[ICCV 2025\] UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)

<!-- RELATED:END -->
