---
title: >-
  [论文解读] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification
description: >-
  [CVPR 2025][社会计算][CLIP] 提出 Classifier-guided CLIP Distillation（CCD），通过 CAM 引导的局部视图标签聚合和 CLIP 预测去偏两项核心技术，在完全无标注的条件下达到与全监督方法持平的多标签分类性能（VOC12 上 90.1% mAP）。
tags:
  - "CVPR 2025"
  - "社会计算"
  - "CLIP"
  - "unsupervised multi-label classification"
  - "CAM"
  - "debiasing"
  - "伪标签"
---

# Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification

**会议**: CVPR 2025  
**arXiv**: [2503.16873](https://arxiv.org/abs/2503.16873)  
**代码**: [https://github.com/k0u-id/CCD](https://github.com/k0u-id/CCD)  
**作者**: Dongseob Kim, Hyunjung Shim  
**机构**: Samsung Electronics, KAIST  
**领域**: 社会计算  
**关键词**: CLIP, unsupervised multi-label classification, CAM, debiasing, pseudo-labels

## 一句话总结
提出 Classifier-guided CLIP Distillation（CCD），通过 CAM 引导的局部视图标签聚合和 CLIP 预测去偏两项核心技术，在完全无标注的条件下达到与全监督方法持平的多标签分类性能（VOC12 上 90.1% mAP）。

## 研究背景与动机
**领域现状**：多标签图像分类（multi-label classification）要求模型对一张图片同时预测多个语义标签。传统全监督方法依赖大量人工标注，成本极高——一张图片平均需要标注2-3个正类和几十个负类。CLIP 等视觉-语言预训练模型的出现为无监督分类带来了新的可能性，因为它们在海量图文对上预训练后具备了zero-shot分类能力。

**现有痛点**：(1) CLIP 的预测具有**视角依赖性（view-dependent predictions）**——对同一图片的不同裁切区域，CLIP 会给出截然不同的类别概率。例如一张包含"人+马"的图片，裁切到人的区域时只会预测"人"，裁切到马的区域时只预测"马"，全图时可能两者概率都不高。(2) CLIP 存在**固有预测偏差（inherent prediction bias）**——某些类别（如"person"）的预测概率系统性偏高，另一些类别（如"sofa"）系统性偏低，即使它们在图中同样显著。这种偏差来源于预训练语料中的类别频率不均衡。(3) 现有无监督方法（如 CDUL）虽然尝试利用 CLIP，但未能有效解决上述两个问题，性能与全监督方法仍有明显差距。

**核心矛盾**：CLIP 具备丰富的视觉-语义知识，理论上足以支撑高质量的多标签分类，但视角依赖性和预测偏差这两个系统性缺陷阻碍了知识的有效转移。现有方法要么忽视这些缺陷，要么仅部分处理。

**本文目标** 如何系统性地解决 CLIP 的视角依赖和预测偏差问题，将 CLIP 的知识无损蒸馏到下游分类器中，实现无需人工标注的高质量多标签分类？

**切入角度**：利用下游分类器自身训练产生的 CAM（Class Activation Map）来反向指导 CLIP 的标签生成——CAM 指示每个类别在图中的空间位置，据此裁切局部视图给 CLIP 做局部预测，再聚合为标签。这种"分类器指导CLIP"的反向蒸馏思路是核心创新。

**核心 idea**：用分类器产生的CAM引导CLIP对局部区域做预测并聚合标签，同时通过训练集统计量对CLIP概率去偏，实现无标注多标签分类。

## 方法详解

### 整体框架
CCD 分为四个阶段：(1) 利用 CLIP 生成初始伪标签；(2) 用分类器的 CAM 引导局部视图选择，更新伪标签；(3) 对 CLIP 预测进行去偏处理；(4) 使用一致性损失进行最终训练。整个流程无需任何人工标注，仅依赖预训练的 CLIP 模型和类别名称列表。

### 关键设计

1. **Phase 1：初始伪标签生成**:

    - 功能：为训练集每张图片生成初始的多标签伪标签
    - 核心思路：用冻结的 CLIP 模型计算图像与每个类别文本（"a photo of {class}"）的余弦相似度，得到类别概率向量。对全图和多个随机裁切区域分别计算概率，取最大值作为初始概率。使用固定阈值（0.5）二值化为伪标签
    - 设计动机：全图预测可能遗漏小物体，随机裁切提供了一定的局部视角覆盖。但这一阶段的标签质量较低（约86.4% mAP on VOC12），因为裁切是随机的而非语义引导的

2. **Phase 2：分类器引导的标签更新（Classifier-Guided Label Update）**:

    - 功能：利用训练中的分类器的 CAM 信息，针对性地裁切包含特定物体的局部区域，让 CLIP 重新预测
    - 核心思路：(a) 用当前分类器对图像做前向传播，获得每个类别的 CAM 热力图；(b) 对每个 CAM 做阈值化获得激活区域的 bounding box；(c) 在 bounding box 附近随机裁切多个（约10个）局部视图；(d) 用 CLIP 对每个局部视图计算类别概率，通过 max pooling 聚合——只要任一局部视图认为该类存在就保留；(e) 将局部聚合标签与初始标签加权融合：$\hat{y} = \alpha \cdot y_{init} + (1-\alpha) \cdot y_{local}$，$\alpha=0.4$
    - 设计动机：CAM 天然指示了"分类器认为该类别在图中的位置"，用它来指导裁切确保局部视图确实包含目标物体。Max pooling 聚合是因为多标签场景下宁可多检测也不要漏检（高召回优先）

3. **Phase 3：CLIP 预测去偏（Debiasing CLIP Predictions）**:

    - 功能：消除 CLIP 预测中的类别频率偏差
    - 核心思路：在训练集上统计每个类别被 CLIP 预测为正类的频率（出现计数 $n_c$），用该频率归一化 CLIP 的原始概率：$p_{debiased}(c) = p_{raw}(c) / n_c^{\gamma}$。$\gamma$ 控制去偏强度。直觉上，如果 CLIP 总是给"person"打高分，那么$n_{person}$会很大，归一化后其概率会被适当降低
    - 设计动机：CLIP 在 web 数据上预训练，"person"等高频类别的文本-图像配对远多于"sofa"等低频类别，导致系统性偏差。去偏操作类似于信息检索中的 IDF（逆文档频率），高频类别的权重被降低

4. **Phase 4：一致性损失训练（Training with Consistency Loss）**:

    - 功能：利用去偏后的伪标签训练最终分类器
    - 核心思路：总损失为 $L_{total} = L_{ce} + \beta \cdot L_{consist}$。$L_{ce}$ 是伪标签上的标准交叉熵损失。$L_{consist}$ 是一致性正则化——对同一图像的两种不同数据增强（如随机裁切 + 颜色抖动），要求模型输出的概率分布一致（KL散度最小化）。训练流程：前2个epoch做warm-up（仅用 $L_{ce}$），之后加入一致性损失继续训练至第10个epoch
    - 设计动机：一致性损失是半监督学习中的经典技术（如FixMatch），在伪标签存在噪声的条件下提供额外的学习信号。Warm-up阶段让模型先建立基础分类能力，避免在训练初期一致性目标与主损失冲突

### 模型架构
- 下游分类器骨干网络：ImageNet 预训练的 ResNet-101
- CLIP 模型：ResNet50×64 变体（冻结，仅用于推理）
- CAM 方法：GradCAM（对最后一个卷积层）

## 实验关键数据

### 主实验：无监督 vs 全监督多标签分类（mAP %）

| 方法 | 监督类型 | VOC12 | VOC07 | COCO | NUS-WIDE |
|------|---------|-------|-------|------|----------|
| BCE (ResNet-101) | 全监督 | 90.1 | 91.3 | 78.5 | 50.7 |
| ASL | 全监督 | 91.2 | 91.8 | 80.2 | — |
| ADDS | 半监督 (10%) | 87.0 | 88.5 | 72.1 | — |
| DualCoOp | Zero-shot (CLIP) | 85.0 | 83.3 | 64.2 | 39.8 |
| CDUL | 无监督 | 88.6 | 89.0 | 69.2 | 44.0 |
| **CCD (本文)** | **无监督** | **90.1** | **91.0** | **70.3** | **44.5** |

### 消融实验（VOC12 mAP %）

| 配置 | mAP | 相比基线变化 | 说明 |
|------|-----|------------|------|
| Baseline (仅全图CLIP) | 86.4 | — | 无局部视图、无去偏 |
| + Label Update (Phase 2) | 88.7 | +2.3 | CAM引导的局部视图聚合 |
| + Debiasing (Phase 3) | 89.4 | +3.0 | CLIP预测去偏 |
| + Consistency (Phase 4) | **90.1** | **+3.7** | 完整CCD框架 |
| 随机裁切替代CAM引导 | 87.9 | +1.5 | CAM引导优于随机裁切 |
| Mean pooling替代Max pooling | 88.1 | +1.7 | Max pooling更适合多标签 |

### 去偏对低频类别的提升（VOC12 per-class AP 变化）

| 类别 | 去偏前 AP | 去偏后 AP | 提升 |
|------|---------|---------|------|
| Plant | 83.2 | 86.5 | +3.3 |
| Sofa | 72.4 | 76.0 | +3.6 |
| TV Monitor | 80.1 | 82.6 | +2.5 |
| Person | 96.8 | 96.2 | -0.6 |
| Car | 93.5 | 93.1 | -0.4 |

### 关键发现
- **无监督可媲美全监督**：CCD 在 VOC12 上达到 90.1% mAP，与 ResNet-101 全监督基线（90.1%）完全持平，在 VOC07 上也仅差 0.3 个百分点（91.0 vs 91.3）。这证明 CLIP 的视觉-语义知识在正确引导下足以替代人工标注
- **CAM 引导显著优于随机裁切**：使用 CAM 引导的局部视图选择比随机裁切高 0.8% mAP（88.7 vs 87.9），说明语义引导的裁切能更精准地覆盖目标物体区域
- **去偏对长尾类别提升明显**：低频类别（Plant +3.3, Sofa +3.6）获得显著提升，而高频类别（Person, Car）仅微降 0.4-0.6，整体增益为正。这验证了 CLIP 偏差主要影响低频类别
- **局部推理次数约10次最优**：增加到10次局部推理后 mAP 趋于饱和，进一步增加会引入噪声。这为实际部署提供了效率-性能的平衡点
- **在复杂数据集上差距仍存在**：COCO 和 NUS-WIDE 上与全监督方法差距仍较大（70.3 vs 78.5, 44.5 vs 50.7），原因在于这些数据集类别更多、物体共现更复杂，CAM 的共现问题（co-occurrence issue）导致局部裁切中包含多个物体
- **Max pooling vs Mean pooling**：Max pooling 在多标签场景下显著优于 Mean pooling（88.7 vs 88.1），因为多标签需要高召回率——任一局部视图检测到物体就应保留

## 亮点与洞察
- **"分类器反向指导 CLIP"的创新思路**：大多数 CLIP 蒸馏方法是单向的（CLIP→分类器），本文提出双向反馈——分类器产生的 CAM 反过来指导 CLIP 在哪里看。这种相互增强的框架简单但非常有效
- **去偏方法的优雅性**：仅用训练集上的类别出现频率做简单归一化就能消除系统性偏差，无需复杂的校准技术。这个方法可以直接迁移到任何使用 CLIP 做伪标签的场景中
- **无监督达到全监督水平的里程碑**：在 VOC12 上首次实现无监督方法与全监督持平，这对于标注成本敏感的实际应用（如医学影像、遥感图像）具有重要的实际意义
- **模块化设计**：四个阶段可以独立替换和改进，例如可以用更好的 CAM 方法（如 ScoreCAM）替代 GradCAM，或用其他 VLM 替代 CLIP

## 局限与展望
- **CAM 共现问题**：当多个物体在图像中空间重叠时，CAM 无法干净地分离各个物体区域，导致局部裁切中混入其他类别的物体，在 COCO/NUS-WIDE 等复杂数据集上表现为性能瓶颈
- **依赖 CLIP 的类别先验**：方法假设 CLIP 对所有目标类别都有一定的识别能力，对于 CLIP 预训练数据中极少出现的细粒度类别（如特定鸟类亚种）可能失效
- **固定阈值的局限**：伪标签二值化使用固定阈值 0.5，未考虑不同类别可能需要不同的最优阈值。自适应阈值策略可能进一步提升性能
- **ResNet-101 骨干网络**：实验仅使用 ResNet-101 作为下游分类器骨干，未验证在 Vision Transformer 等更现代架构上的效果
- **训练效率**：每轮标签更新需要对所有训练图像做多次 CLIP 推理（约10次局部推理/图像），计算成本较高

## 相关工作与启发
- **vs CDUL**：CDUL 也利用 CLIP 做无监督多标签分类，但未处理视角依赖性和预测偏差，CCD 通过 CAM 引导和去偏两项技术系统性地解决了这些问题
- **vs DualCoOp**：DualCoOp 通过学习正负 prompt 来适应 CLIP，但属于 zero-shot（需要类别定义），且性能（85.0% on VOC12）明显低于 CCD
- **vs FixMatch/MixMatch**：半监督学习方法需要少量标注数据，CCD 完全无需标注但借鉴了一致性正则化的思想
- **vs CAM-based WSS**：弱监督语义分割中广泛使用 CAM，但是用图像级标签生成像素级伪标签；CCD 反过来用 CAM 来改进图像级标签质量，是一种新颖的CAM应用方式

## 评分
- 新颖性: ⭐⭐⭐⭐ 分类器反向指导 CLIP、CLIP 去偏都是新颖且有效的设计
- 实验充分度: ⭐⭐⭐⭐ 4个数据集、充分消融、per-class分析，但缺少 ViT 骨干实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程易懂
- 价值: ⭐⭐⭐⭐ 无监督首次达到全监督水平，实际应用潜力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)
- [\[CVPR 2025\] Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness](project-probe-aggregate_efficient_fine-tuning_for_group_robustness.md)
- [\[NeurIPS 2025\] IF-GUIDE: Influence Function-Guided Detoxification of LLMs](../../NeurIPS2025/social_computing/if-guide_influence_function-guided_detoxification_of_llms.md)
- [\[ACL 2026\] Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation](../../ACL2026/social_computing/estimating_the_black-box_llm_uncertainty_with_distribution-aligned_adversarial_d.md)

</div>

<!-- RELATED:END -->
