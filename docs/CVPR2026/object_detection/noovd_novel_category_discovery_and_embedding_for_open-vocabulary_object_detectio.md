---
title: >-
  [论文解读] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection
description: >-
  [CVPR 2026][目标检测][开放词汇目标检测] 提出NoOVD框架，在基于冻结VLM的OVD训练中通过无参数K-FPN保留CLIP知识来发现潜在新类别目标、通过自蒸馏将新类别知识嵌入检测器、通过R-RPN在推理时提升新类别召回率，在OV-LVIS/OV-COCO/Objects365上取得SOTA。
tags:
  - CVPR 2026
  - 目标检测
  - 开放词汇目标检测
  - 新类别发现
  - 自蒸馏
  - K-FPN
  - 冻结VLM
---

# NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection

**会议**: CVPR 2026  
**arXiv**: [2603.21069](https://arxiv.org/abs/2603.21069)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 开放词汇目标检测、新类别发现、自蒸馏、K-FPN、冻结VLM

## 一句话总结
提出NoOVD框架，在基于冻结VLM的OVD训练中通过无参数K-FPN保留CLIP知识来发现潜在新类别目标、通过自蒸馏将新类别知识嵌入检测器、通过R-RPN在推理时提升新类别召回率，在OV-LVIS/OV-COCO/Objects365上取得SOTA。

## 研究背景与动机

1. **领域现状**：开放词汇目标检测（OVD）旨在让检测器识别训练时未见过的新类别。主流方法构建在冻结VLM（如CLIP）之上，只训练检测模块（FPN、RPN、RoI head），通过VLM的零样本迁移能力实现对新类别的识别。
2. **现有痛点**：训练和测试之间存在显著gap——训练时只有基类标注，所有未标注的新类别目标被强制视为背景。RPN阶段新类别proposal得分低被过滤掉，RoI阶段新类别特征被强制对齐到背景文本嵌入。测试时这些proposal同样得分低，在后处理时被移除，导致新类别召回率大幅下降。
3. **核心矛盾**：训练时没有新类别标注，模型被迫将新类别当背景学习；但测试时又要求模型识别这些类别。已有解决思路要么依赖大规模额外数据（成本高），要么使用伪标签（引入噪声）。
4. **本文目标** (1) 在不引入额外数据和伪标签噪声的前提下发现潜在新类别目标；(2) 将新类别知识嵌入检测器；(3) 提升推理时新类别的召回率。
5. **切入角度**：利用冻结CLIP自身的零样本识别能力来发现前景目标，用类别无关的通用前景/背景文本描述代替具体类别名称，无需知道新类别的名字就能区分前景和背景。
6. **核心 idea**：用冻结CLIP的零样本能力做类别无关的前景发现，然后通过自蒸馏把新类别知识注入检测器，避免新类别特征被错误对齐到背景。

## 方法详解

### 整体框架
基于冻结CLIP的两阶段检测框架。训练时：(1) K-FPN从冻结CLIP多层特征构建无参数特征金字塔，保留CLIP对新类别的知识；(2) 利用类别无关的前景/背景文本描述 + CLIP零样本能力发现潜在新类别proposal；(3) 对发现的新类别proposal做自蒸馏，将CLIP特征对齐到RoI特征。测试时：(4) R-RPN用同样的发现策略提升新类别proposal的置信度。

### 关键设计

1. **K-FPN (Knowledge-retentive FPN)**:

    - 功能：从冻结CLIP构建保留原始知识的特征金字塔
    - 核心思路：以CLIP ViT-B/16为例，提取第[5,7,11]层特征图作为 {P2,P3,P4}，按FPN方式做自顶向下特征融合，再用CLIPSelf的冻结参数降维头将768维降到512维（对齐文本嵌入维度），得到 {C2,C3,C4}。将高层 C4 上采样与 C3、C2 拼接得到高分辨率特征图 {F2,F3,F4}，对 C4 做两次max pooling得到 {F5,F6}。整个过程完全无可学习参数。
    - 设计动机：标准FPN有可学习参数，训练时只有基类数据会导致CLIP特征在经过FPN后发生漂移，丧失新类别知识。K-FPN通过完全无参数操作最大程度保留CLIP的原始表示能力，为后续新类别发现提供可靠的特征基础。

2. **新类别发现与嵌入**:

    - 功能：训练时发现潜在新类别目标并将知识注入检测器
    - 核心思路：(1) 用ChatGPT-o1生成30条类别无关的前景描述（如 "This is an object, specifically a plant"）和30条背景描述（如 "This is a background area"），提取冻结CLIP文本嵌入；(2) 将RPN proposal映射到K-FPN上做RoI Align提取特征，计算与前景/背景文本嵌入的余弦相似度，保留前景得分高于背景的proposal，排除与GT Bbox重叠高的基类proposal，剩下的就是潜在新类别目标；(3) 对这些proposal从原图裁剪区域送入冻结CLIP提取特征，与RoI head输出特征做L2 loss自蒸馏：$\mathcal{L}_{kd} = \|F_{proposals+}^{RoI} - F_{proposals+}^{Image}\|_2^2$。
    - 设计动机：不依赖具体类别名称，用通用"前景vs背景"描述就能发现新类别目标。自蒸馏让检测器在训练时就学到新类别知识，而不是把它们当背景来学。不需要额外数据，不构造伪图文对，从源头消除伪标签噪声。

3. **R-RPN (Re-weighted RPN)**:

    - 功能：推理时提升新类别proposal的召回率
    - 核心思路：在RPN后处理前，用训练阶段相同的前景发现策略对NMS后的proposal进行前景分类，将K-FPN前景得分与原始RPN得分加权融合：$S_{R-RPN} = \alpha \cdot S_{RPN} + (1-\alpha) \cdot S_{K-FPN}$，$\alpha=0.5$。用融合后的得分重新排序，取top-1000送入RoI head。
    - 设计动机：新类别proposal在RPN中得到低分（因为训练时被当背景），在后处理中被过滤掉是新类别检测失败的主要原因。R-RPN通过注入CLIP的前景知识来"拯救"这些被遗漏的proposal。

### 损失函数 / 训练策略
- 总损失 $\mathcal{L}_{total} = \mathcal{L}_{cls-RPN} + \mathcal{L}_{reg-RPN} + \mathcal{L}_{reg-RoI} + \mathcal{L}_{cons} + \mathcal{L}_{kd}$
- $\mathcal{L}_{cons}$为RoI head的对比损失，$\mathcal{L}_{kd}$为新类别自蒸馏L2损失（权重=1）
- 冻结CLIP image/text encoder，只训练FPN、RPN、RoI head
- OV-COCO训练5 epochs，OV-LVIS训练50 epochs
- 16个NVIDIA 3090，batch size 10/GPU，AdamW lr=$10^{-4}$

## 实验关键数据

### 主实验 - OV-LVIS

| 方法 | Backbone | AP_r (rare/新) | AP (总) |
|------|----------|---------------|---------|
| CLIPSelf + F-ViT | ViT-L/14 | 34.9 | 35.1 |
| DeCLIP + F-ViT | ViT-L/14 | 37.2 | 36.0 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **39.2 (+2.0)** | **37.7 (+1.7)** |
| CLIPSelf + NoOVD | ViT-B/16 | 28.3 (+2.9) | 26.7 (+1.3) |
| YOLOE | YOLOv11-L | 29.1 | 35.2 |
| RO-ViT | ViT-H/16 | 34.1 | 35.1 |

### 主实验 - OV-COCO

| 方法 | Backbone | AP_novel^50 | AP^50 |
|------|----------|-------------|-------|
| DeCLIP + F-ViT | ViT-L/14 | 46.2 | 60.3 |
| **DeCLIP + NoOVD** | **ViT-L/14** | **47.5 (+1.3)** | **61.0 (+0.7)** |
| CORA+ | RN50x4 | 43.1 | 56.2 |

### 消融实验

| 配置 | AP_r | AP |
|------|------|-----|
| Baseline (F-ViT) | 25.4 | 25.4 |
| + CLIP-top (简单顶层特征) | 26.4 | 26.1 |
| + K-FPN | 27.5 | 26.4 |
| + R-RPN | 26.7 | 25.9 |
| + K-FPN + R-RPN (完整) | **28.3** | **26.7** |

### 跨数据集迁移 (LVIS→Objects365)

| 方法 | Backbone | AP_r | AP50 |
|------|----------|------|------|
| CLIPSelf + F-ViT | ViT-L/14 | 21.7 | 39.2 |
| CLIPSelf + NoOVD | ViT-L/14 | **22.8 (+1.1)** | **40.2 (+1.0)** |

### 关键发现
- **K-FPN vs 简单CLIP顶层特征**：K-FPN多尺度特征金字塔比单层特征在新类别检测上多提升1.1%，说明多尺度对不同大小新类别目标的发现至关重要。
- **K-FPN和R-RPN互补**：K-FPN解决训练时的知识保留和注入问题，R-RPN解决推理时的召回问题，两者组合效果最佳。
- **OV-LVIS比OV-COCO更稳定**：OV-COCO标注不完整，训练时被NoOVD正确检出的新类别目标在测试时反而被计为误检（false positive），导致增益被低估。
- **融合权重W=0.3最优**：K-FPN特征融合中高层语义和底层细节的平衡很重要，比例过大或过小都会掉点。

## 亮点与洞察
- **无额外数据、无伪标签的新类别发现**：利用通用前景/背景文本描述和冻结CLIP的零样本能力就能发现新类别，无需知道具体类别名称，也无需图文匹配构造伪标签。这种"类别无关但前景感知"的策略非常巧妙精简。
- **K-FPN完全无参数设计**：通过纯插值、拼接、池化从冻结CLIP特征构建特征金字塔，最大程度避免了基类训练对新类别知识的破坏。这个设计虽然简单但直击问题本质。
- **训练-推理一致的发现策略**：训练时用前景发现做自蒸馏，推理时同样的策略给R-RPN做分数重加权，逻辑自洽。

## 局限与展望
- 前景/背景文本描述是用ChatGPT生成的固定30+30条，可能无法覆盖所有场景语义，自动化/自适应prompt设计可能更好
- 自蒸馏的新类别proposal选择依赖阈值，可能漏掉与背景视觉相似的新类别目标（如road上的manhole cover）
- R-RPN的 $\alpha=0.5$ 是固定值，对不同数据集/不同新基类比例可能不是最优
- 当前框架是two-stage detector，与one-stage/DETR-based检测器的结合未探索

## 相关工作与启发
- **vs Detic**: Detic用ImageNet图像级标签扩展类别，依赖额外大规模数据；NoOVD完全不需要额外数据，只利用CLIP自身能力
- **vs CLIPSelf/DeCLIP**: 这些方法优化了CLIP的区域级表示但仍在训练中将新类别当背景；NoOVD从训练流程本身修正了这个问题
- **vs F-VLM**: 同样基于冻结VLM但F-VLM没有主动发现新类别的机制；NoOVD通过K-FPN+自蒸馏主动挖掘和学习新类别知识

## 评分
- 新颖性: ⭐⭐⭐⭐ 类别无关的新类别发现+K-FPN无参数设计思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ OV-LVIS/OV-COCO/Objects365三个数据集、多backbone、跨数据集验证、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，图示丰富
- 价值: ⭐⭐⭐⭐ 提供了一种无需额外数据的OVD新范式，对开放词汇检测社区有参考价值

<!-- RELATED:START -->

## 相关论文

- [Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)
- [Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation.md)
- [CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](../../NeurIPS2025/object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)
- [DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)

<!-- RELATED:END -->
