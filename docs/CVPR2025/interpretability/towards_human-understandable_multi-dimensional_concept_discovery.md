---
title: >-
  [论文解读] Towards Human-Understandable Multi-Dimensional Concept Discovery
description: >-
  [CVPR 2025][概念发现] 提出 HU-MCD 框架，用 SAM 替代传统分割方法发现人类可理解的视觉概念，配合 CNN 专用的输入遮罩方案减少噪声干扰，在 MCD 的完备性框架下实现可理解性和忠实性兼顾的概念级模型解释。
tags:
  - CVPR 2025
  - 概念发现
  - 可解释AI
  - SAM分割
  - CNN解释
  - 人类理解性
---

# Towards Human-Understandable Multi-Dimensional Concept Discovery

**会议**: CVPR 2025  
**arXiv**: [2503.18629](https://arxiv.org/abs/2503.18629)  
**代码**: https://github.com/grobruegge/hu-mcd  
**领域**: 其他  
**关键词**: 概念发现, 可解释AI, SAM分割, CNN解释, 人类理解性

## 一句话总结
提出 HU-MCD 框架，用 SAM 替代传统分割方法发现人类可理解的视觉概念，配合 CNN 专用的输入遮罩方案减少噪声干扰，在 MCD 的完备性框架下实现可理解性和忠实性兼顾的概念级模型解释。

## 研究背景与动机
1. **领域现状**：概念级可解释 AI（C-XAI）旨在用人类可理解的视觉概念替代像素级显著性图解释模型决策。代表方法有 ACE（超像素分割+聚类）、ICE（NMF 特征分解）、CRAFT（区域 NMF）、MCD（多维子空间分解）。
2. **现有痛点**：(a) ACE 需要将分割区域 inpainting 和 resize 到模型输入尺寸，引入噪声干扰模型预测；(b) MCD 虽有完备性理论保障忠实性，但生成的概念人类难以理解（不同概念高度相似、语义模糊）；(c) 可理解性和忠实性之间存在固有矛盾——越对齐人类认知的分割可能越偏离模型内部表示。
3. **核心矛盾**：要发现人类可理解的概念就需要好的分割，但分割后的不规则区域难以喂给 CNN；不做分割直接用特征图虽忠实但概念不可理解。
4. **本文目标**：同时实现概念的人类可理解性和对模型决策的忠实解释。
5. **切入角度**：用 SAM 做实例分割获得高质量语义区域 + CNN layer masking 避免 inpainting 噪声。
6. **核心 idea**：SAM 分割 → CNN-specific 层级遮罩传播 → SSC 聚类 → MCD 完备性分解。

## 方法详解

### 整体框架
两阶段：**概念发现**——用 SAM 分割类别图像，通过 CNN 层级遮罩方案提取每个区域的特征嵌入，用稀疏子空间聚类（SSC）将相似区域聚为概念；**概念评分**——采用 MCD 框架在特征空间中为每个概念计算激活度和重要性评分，满足局部+全局完备性。

### 关键设计

1. **SAM 驱动的概念发现**

    - 功能：生成语义有意义、边界精确的图像区域作为概念候选
    - 核心思路：对每张类别图像调用 SAM（ViT-h 编码器），选择覆盖面积 ≥1% 的最细粒度分割掩码。聚类数量根据每张图像的平均分割数自动确定（不像 ACE 手动设定 25 个）。SAM 在人工标注分割掩码上训练，天然产生人类直觉的区域划分。
    - 设计动机：ACE 的超像素分割语义性弱，SAM 的零样本实例分割能力生成更有语义意义的区域

2. **CNN-specific 层级遮罩方案**

    - 功能：在不引入 inpainting/resize 噪声的前提下提取不规则区域的 CNN 特征
    - 核心思路：受 Balasubramanian & Feizi 启发，将图像和对应遮罩同时逐层传播，在每个卷积层后用遮罩丢弃仅依赖被遮盖区域的激活值。边界处用邻域非遮罩像素的均值做 padding 避免边缘伪影。特殊处理：第一个卷积层（如 ResNet50 的 7×7）允许访问遮罩边缘的窄带上下文保留形状信息；大遮罩（>25% 面积）收缩 kernel size 避免暴露物体轮廓。
    - 设计动机：传统方法用 mean-padding 或 inpainting 填充遮罩区域会引入虚假特征干扰模型预测，层级遮罩方案从源头消除噪声

3. **MCD 完备性框架的适配**

    - 功能：为每个 SAM 发现的概念提供有完备性保障的重要性评分
    - 核心思路：对聚类成员的隐层表示做 PCA，取主成分作为概念子空间基底。所有概念子空间加上正交补空间构成特征空间的完整分解。**概念激活度**：将区域特征投影到子空间测量概念存在强度。**局部概念相关性**：分解最终分类 logit 为各概念子空间的贡献，求和严格等于原始 logit（完备性）。**全局概念相关性**：将分类权重向量投影到各子空间。
    - 设计动机：MCD 的完备性保证了概念重要性评分"忠实"地反映模型决策过程，不会遗漏信息

### 损失函数 / 训练策略
HU-MCD 是一个后验解释方法，不需要训练。使用预训练的 ResNet50 (timm) + 预训练的 SAM (ViT-h)。

## 实验关键数据

### 主实验（人类实验 + ImageNet 10 类）

| 指标 | HU-MCD | ACE | MCD |
|------|--------|-----|-----|
| 预测准确率↑ | **70.24%** | 42.93% | 31.22% |
| 可识别概念比例↑ | **67.12%** | 45.66% | 50.34% |
| 概念内描述相似度↑ | **0.49** | 0.39 | 0.41 |
| 概念间描述相似度↓ | **0.28** | 0.29 | 0.38 |

### 消融/忠实性实验

| 方法 | C-Insertion AUC↑ | C-Deletion AUC↓ | 说明 |
|------|-----------------|----------------|------|
| HU-MCD | **best** | **best** | 概念重要性评分最忠实 |
| ACE | 中等 | 中等 | 分割噪声影响忠实性 |
| MCD | 中等 | 中等 | 概念不可区分影响评估 |

### 关键发现
- HU-MCD 预测准确率 70.24% vs MCD 31.22%，可理解性提升巨大
- MCD 的概念间描述相似度高达 0.38（接近概念内的 0.41），说明其概念高度同质化，人类难以区分
- SAM 分割的概念能发现数据偏差（如"蛙"类中的"人手"概念），具有发现虚假相关的实用价值
- 层级遮罩方案使模型准确度保持率显著优于 mean-padding 方案
- 41 名人类被试的实验（含注意力检查），结果具有统计显著性（ANOVA p<0.001）

## 亮点与洞察
- **将 SAM 引入 C-XAI** 是一个自然但有效的组合——SAM 的零样本分割能力弥补了传统概念发现方法的分割短板
- **层级遮罩方案**解决了一个长期困扰 C-XAI 的工程问题——如何将不规则区域喂给 CNN 而不引入噪声
- **人类实验设计**非常严谨——任务预测、可识别性评估、描述一致性多维度验证可理解性

## 局限与展望
- 仅验证了 ResNet50，Transformer 架构的适用性待探索
- SAM 的计算开销较大，限制了大规模应用
- 10 个类的验证规模较小
- 未来可探索将发现的概念用于模型改进（不仅解释）

## 相关工作与启发
- **vs ACE**: ACE 用超像素分割+inpainting，概念可理解性中等但忠实性受噪声影响；HU-MCD 用 SAM+层级遮罩双重改进
- **vs MCD**: MCD 忠实性好但概念不可理解（高度同质化）；HU-MCD 用 SAM 保证可理解性，继承 MCD 的完备性保证忠实性
- **vs CRAFT**: CRAFT 用方形 patch 避免 inpainting 但失去了精确的区域边界

## 评分
- 新颖性: ⭐⭐⭐⭐ SAM+CNN遮罩+MCD的组合创新，各组件有前人基础但组合有效
- 实验充分度: ⭐⭐⭐⭐⭐ 41人人类实验+多指标+统计检验，非常严谨
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，人类实验设计规范，图表直观
- 价值: ⭐⭐⭐⭐ 在可解释AI领域推进了可理解性与忠实性的平衡

<!-- RELATED:START -->

## 相关论文

- [Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)
- [Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)

<!-- RELATED:END -->
