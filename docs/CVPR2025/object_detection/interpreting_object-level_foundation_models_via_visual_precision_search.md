---
title: >-
  [论文解读] Interpreting Object-level Foundation Models via Visual Precision Search
description: >-
  [CVPR 2025][目标检测][模型可解释性] 针对 Grounding DINO 和 Florence-2 等目标级基础模型的可解释性问题，本文提出 Visual Precision Search (VPS) 方法，通过超像素稀疏化+子模函数引导的贪心搜索精确定位关键决策子区域，在 MS COCO/RefCOCO/LVIS 上的忠实度指标(Insertion)分别超过 SOTA 方法 D-RISE 达 23.7%/20.1%/31.6%。
tags:
  - CVPR 2025
  - 目标检测
  - 模型可解释性
  - 视觉精确搜索
  - 子模函数
  - 归因分析
---

# Interpreting Object-level Foundation Models via Visual Precision Search

**会议**: CVPR 2025  
**arXiv**: [2411.16198](https://arxiv.org/abs/2411.16198)  
**代码**: https://github.com/RuoyuChen10/VPS  
**领域**: 目标检测  
**关键词**: 模型可解释性, 目标检测, 视觉精确搜索, 子模函数, 归因分析

## 一句话总结
针对 Grounding DINO 和 Florence-2 等目标级基础模型的可解释性问题，本文提出 Visual Precision Search (VPS) 方法，通过超像素稀疏化+子模函数引导的贪心搜索精确定位关键决策子区域，在 MS COCO/RefCOCO/LVIS 上的忠实度指标(Insertion)分别超过 SOTA 方法 D-RISE 达 23.7%/20.1%/31.6%。

## 研究背景与动机

1. **领域现状**：随着多模态预训练的发展，Grounding DINO、Florence-2等目标级基础模型在视觉定位和目标检测中表现出色。但这些模型参数量巨大、语义丰富，透明度和可解释性急剧下降。

2. **现有痛点**：
    - **梯度方法(如ODAM)**：由于基础模型中视觉和文本特征深度融合，梯度回传时受双模态影响，无法准确定位到视觉层面的关键区域，导致归因图弥散
    - **扰动方法(如D-RISE)**：通过随机采样生成显著性图，引入大量采样噪声，导致归因图粗糙、细粒度可解释性不足
    - 现有方法无法有效解释模型的检测失败原因

3. **核心矛盾**：基础模型的视觉-文本融合架构使传统梯度归因失效；而扰动方法的随机采样策略又过于粗糙。

4. **本文目标** 设计一种不依赖模型内部参数、能精确定位少量关键决策区域的归因方法，适用于不同架构的目标级基础模型。

5. **切入角度**：将归因问题转化为子区域的子集选择问题，利用子模优化的理论保证进行贪心搜索。

6. **核心 idea**：用超像素将输入稀疏化为子区域集合，设计满足子模性的评分函数（线索分+协作分），通过贪心搜索精确排序关键区域。

## 方法详解

VPS 的核心思路是：不去探究模型内部参数（黑盒方法），而是将输入图像划分为少量有意义的子区域，然后通过子模优化找到"用最少区域即可让检测器正确工作"的关键区域集合。

### 整体框架

输入：图像 $\mathbf{I}$、目标边界框 $\boldsymbol{b}_{target}$、目标类别 $c$。
处理流程：
1. SLICO超像素分割将图像分为 $m$ 个子区域 $V = \{\mathbf{I}_1^s, ..., \mathbf{I}_m^s\}$
2. 定义子模函数 $\mathcal{F}(S)$ 评估子区域集合的可解释性
3. 贪心搜索迭代选择使 $\mathcal{F}$ 最大化的子区域
4. 根据边际贡献为每个子区域评分，生成最终显著性图

### 关键设计

1. **线索分 (Clue Score)**:
    - 功能：评估给定子区域集合能否让模型正确定位和识别目标
    - 核心思路：对于子区域集合 $S$，计算模型输出中与目标框IoU和类别置信度的乘积最大值：$s_{clue}(S, \boldsymbol{b}_{target}, c) = \max_{(\boldsymbol{b}_i, s_{c,i}) \in f(S)} \text{IoU}(\boldsymbol{b}_{target}, \boldsymbol{b}_i) \cdot s_{c,i}$。与D-RISE不同，此方法考虑所有候选框而非仅高置信框，避免搜索陷入局部最优
    - 设计动机：一个好的解释应该能让模型用尽可能少的输入区域就做出正确检测

2. **协作分 (Collaboration Score)**:
    - 功能：评估子区域的组合效应——某些区域只有与其他特定区域组合时才对决策有贡献
    - 核心思路：计算移除子区域集合 $S$ 后模型检测能力的下降程度：$s_{colla.}(S, \boldsymbol{b}_{target}, c) = 1 - \max_{(\boldsymbol{b}_i, s_{c,i}) \in f(V \setminus S)} \text{IoU}(\boldsymbol{b}_{target}, \boldsymbol{b}_i) \cdot s_{c,i}$。移除关键区域后，如果没有检测框能准确定位目标，则这些区域的协作分高
    - 设计动机：在搜索初期，某些微妙但关键的区域（如辅助上下文线索）仅通过加入测试可能效果不明显，但移除它们会导致检测失败。协作分能有效捕捉这类区域

3. **子模函数与贪心搜索**:
    - 功能：将两种分数组合为具有理论保证的优化目标
    - 核心思路：子模函数 $\mathcal{F}(S) = s_{clue}(S) + s_{colla.}(S)$。论文证明了该函数在合理假设下满足边际递减和单调非负性质（子模性），因此贪心搜索能保证 $(1-1/e)$ 近似比的最优解。搜索完成后，用边际贡献差 $\mathcal{A}_i = \mathcal{A}_{i-1} - |\mathcal{F}(S_{[i]}) - \mathcal{F}(S_{[i-1]})|$ 为每个子区域评分，归一化后生成显著性图
    - 设计动机：子模优化提供了理论上的最优性保证，比D-RISE的随机采样和D-HSIC的SHAP估计更加原理化

### 损失函数 / 训练策略

VPS是无需训练的推理时方法。超像素数默认设为100个。每个被评估的子区域集合需要一次模型前向推理。

## 实验关键数据

### 主实验

**Grounding DINO 忠实度评估 (正确检测样本)**

| 数据集 | 方法 | Insertion↑ | Deletion↓ | Avg. Highest Score↑ |
|--------|------|-----------|-----------|---------------------|
| MS COCO | D-RISE | 0.4412 | 0.0402 | 0.6215 |
| MS COCO | **VPS(Ours)** | **0.5459** (+23.7%) | **0.0375** | **0.6873** |
| RefCOCO | D-RISE | 0.6178 | 0.1605 | 0.8471 |
| RefCOCO | **VPS(Ours)** | **0.7419** (+20.1%) | **0.1250** | **0.8842** |
| LVIS(rare) | D-RISE | 0.2808 | 0.0289 | 0.4289 |
| LVIS(rare) | **VPS(Ours)** | **0.3695** (+31.6%) | **0.0277** | **0.4969** |

**Florence-2 忠实度评估**

| 数据集 | 方法 | Insertion↑ | Deletion↓ |
|--------|------|-----------|-----------|
| MS COCO | D-RISE | 0.7477 | 0.0972 |
| MS COCO | **VPS(Ours)** | **0.7761** | **0.0479** (-50.7%) |
| RefCOCO | D-RISE | 0.8107 | 0.1275 |
| RefCOCO | **VPS(Ours)** | **0.8604** | **0.0422** (-66.9%) |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 仅Clue Score | Insertion显著下降 | 缺少协作分无法捕捉上下文依赖 |
| 仅Collaboration Score | 初始搜索不够精准 | 缺少直接的检测引导 |
| Full (Clue+Colla.) | 最优 | 两者互补 |
| 超像素数量 | 100个最优 | 过少→粒度粗；过多→搜索空间大 |

**失败案例分析 (Grounding DINO)**

| 任务 | 方法 | Insertion↑ | ESR↑ |
|------|------|-----------|------|
| 视觉定位失败 | D-RISE | 0.3430 | 39.5% |
| 视觉定位失败 | **VPS** | **0.4901** (+42.9%) | **42.5%** |
| 误分类(COCO) | D-RISE | 0.3021 | — |
| 误分类(COCO) | **VPS** | **0.4674** (+54.7%) | — |

### 关键发现
- **梯度方法在基础模型上失效**：Grad-CAM/ODAM在Grounding DINO上的归因高度弥散，Insertion指标仅为VPS的一半以下，证实了视觉-文本融合对梯度归因的严重干扰
- **VPS在失败分析上尤其强大**：不仅能解释正确检测，还能识别导致误分类和漏检的输入级因素
- **线索分+协作分的互补性**：线索分在后期搜索中主导（引导模型做出正确检测），协作分在早期搜索中更重要（快速锁定关键的上下文区域）
- **Florence-2的Deletion指标提升最显著**：50.7%和66.9%的改善，因为没有置信度分数时搜索比随机采样优势更大

## 亮点与洞察
- **搜索代替采样**：将随机扰动替换为有理论保证的子模优化搜索，核心巧妙在于"用更少区域做出正确检测"这个目标天然适合子模优化的边际递减特性
- **黑盒通用性**：不访问模型内部参数，适用于完全不同架构的模型（Grounding DINO的多阶段融合 vs Florence-2的MLLM架构），这种方法论可以迁移到任何未来的新架构
- **失败分析能力**：除了解释正确决策，VPS还能定位导致错误的输入因素（如共现混淆），这对自动驾驶等安全关键场景有实际价值

## 局限与展望
- 贪心搜索的计算开销：每轮搜索中每个候选子区域都需要一次模型前向推理，$m$ 个子区域需要 $O(m^2)$ 次推理，对于大模型可能较慢
- 超像素分割质量直接影响結果，如果关键区域被不合理地分割则无法正确归因
- MLLM模型(如Florence-2)不直接输出置信度分数，使得clue score只能依赖IoU，解释精度受限
- 目前仅验证了静态图像的目标检测，视频检测和其他视觉基础任务（如分割）未涉及
- 子模性的理论证明依赖"区域对检测有正贡献"的假设，异常输入可能违反

## 相关工作与启发
- **vs D-RISE**: D-RISE通过随机掩码采样估计每个像素的重要性，产生噪声归因图；VPS通过结构化搜索精确定位少量关键子区域，Insertion指标提升20-30%
- **vs ODAM**: ODAM是梯度方法(Grad-CAM)在检测上的扩展，在多模态融合模型上梯度归因被文本特征污染导致弥散；VPS作为黑盒方法完全规避了这个问题
- **vs VX-CODE**: VX-CODE也使用贪心搜索+SHAP，但缺少子模优化的理论保证和协作分的设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 将子模优化引入目标检测可解释性，clue+collaboration双分设计有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集+两个模型+正确/失败分析+多种评估指标，非常全面
- 写作质量: ⭐⭐⭐⭐ 理论分析和实验验证结合良好，逻辑清晰
- 价值: ⭐⭐⭐⭐ 为基础模型的可解释性提供了通用黑盒方案，对安全关键应用有实际意义

<!-- RELATED:START -->

## 相关论文

- [Can OOD Object Detectors Learn from Foundation Models?](../../ECCV2024/object_detection/can_ood_object_detectors_learn_from_foundation_models.md)
- [Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)
- [Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](../../AAAI2026/object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](../../ICLR2026/object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)

<!-- RELATED:END -->
