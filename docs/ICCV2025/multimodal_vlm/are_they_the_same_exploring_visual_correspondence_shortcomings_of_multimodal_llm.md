---
title: >-
  [论文解读] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs
description: >-
  [ICCV 2025][多模态][视觉对应] 本文首次系统研究了多模态大模型（MLLM）在视觉对应匹配方面的不足，构建了含1510样本的MMVM基准和220K匹配数据集，并提出CoLVA方法通过目标级对比学习和细粒度视觉专家显著提升了MLLM的跨图像实例匹配能力。
tags:
  - ICCV 2025
  - 多模态
  - 视觉对应
  - 多模态大模型
  - 对比学习
  - 视觉匹配
  - benchmark
---

# Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs

**会议**: ICCV 2025  
**arXiv**: [2501.04670](https://arxiv.org/abs/2501.04670)  
**代码**: https://zhouyiks.github.io/projects/CoLVA/  
**领域**: Multimodal VLM  
**关键词**: 视觉对应, 多模态大模型, 对比学习, 视觉匹配, benchmark

## 一句话总结

本文首次系统研究了多模态大模型（MLLM）在视觉对应匹配方面的不足，构建了含1510样本的MMVM基准和220K匹配数据集，并提出CoLVA方法通过目标级对比学习和细粒度视觉专家显著提升了MLLM的跨图像实例匹配能力。

## 研究背景与动机

尽管MLLM在视觉感知、推理和视觉语言理解方面取得了显著进展，但视觉匹配能力这一计算机视觉中的基础能力却鲜有研究。作者发现即便是GPT-4o这样的强模型，在面对简单的跨图像物体匹配问题时也表现不佳。实验显示，当前最强的开源MLLM Qwen2-VL-72B在MMVM基准上的总体准确率仅38%，说明现有MLLM在视觉对应理解上存在系统性缺陷。

通过分析，作者归纳了两个核心原因：（1）现有MLLM虽具备识别物体外观和位置的基本能力，但缺少教其如何利用这些能力进行视觉匹配的对应数据；（2）MLLM依赖CLIP模型理解图像，无法提取细粒度、判别性的视觉特征——这对于区分语义高度相似的候选物体至关重要。PCA可视化进一步证实了这一点：匹配目标与其他候选物体在特征空间中聚集在一起，远离查询物体。

## 方法详解

### 整体框架

CoLVA基于InternVL2构建，引入两项核心技术：（1）带目标级对比学习（OCL）的细粒度视觉专家；（2）指令增强策略（IA）。训练分为预训练阶段（对齐附加视觉编码器的特征空间）和SFT阶段（使用MMVM数据微调）。

### 关键设计

1. **MMVM数据集与基准构建**:

    - 从5个视频分割数据集采样帧对，利用已有标注生成220K多选匹配QA对
    - 通过InternVL2-76B为每个QA对自动生成匹配推理文本（chain-of-thought风格）
    - 基准集包含1510个手工标注样本，覆盖8种匹配线索（颜色、形状/姿态、文本/LOGO标记、尺寸、相对位置、朝向/运动、绑定关系、物体标记）
    - 候选选项数范围2-37，平均10个

2. **目标级对比学习（OCL）**:

    - 在MLLM视觉编码器与附加视觉专家（RADIO）之间进行对比学习，而非共享跟踪头
    - 通过Masked Average Pooling提取目标级表示
    - 对比损失定义为：$\mathcal{L} = \frac{\exp(\mathbf{O} \cdot \mathbf{O}^+)}{\exp(\mathbf{O} \cdot \mathbf{O}^+) + \sum_{\mathbf{O}^-} \exp(\mathbf{O} \cdot \mathbf{O}^-)}$
    - 设计动机：让视觉专家在MLLM语义空间内学习判别性特征，同时实现模态对齐
    - 直接在MLLM的CLIP骨干上做OCL效果有限（34.05 vs 32.38），因CLIP缺乏细粒度特征

3. **细粒度视觉专家（RADIO）**:

    - RADIO蒸馏自SAM、DINOv2、CLIP等多个视觉基础模型，兼具细粒度特征和图文对齐能力
    - 预训练阶段冻结InternVL2视觉编码器和RADIO，仅训练RADIO Adapter
    - 输入图像对中，一张通过MLLM原始编码器，另一张通过RADIO，再在目标表示上做OCL
    - 利用图像分割数据集模拟伪视频数据来扩充有限的带分割标注的图像对

4. **指令增强（IA）**:

    - 原始方式：在图像上叠加高亮轮廓和ID标签来指代物体，梯度无法直接回传到图像特征
    - 增强方式：直接将目标级表示嵌入指令中（"object-1: \<Obj 1\>, object-2: \<Obj 2\>, ..."），使梯度可直接回传到对应图像特征
    - 两种格式随机切换使用，增强模型对多物体引用的指令跟随能力

### 损失函数 / 训练策略

- 预训练阶段：目标级对比损失，对齐RADIO与MLLM特征空间
- SFT阶段：LLaVA SFT数据（665K）+ MMVM数据（220K），标准自回归训练目标

## 实验关键数据

### 主实验

| 模型 | 参数量 | Overall | CL | RP |
|------|--------|---------|-----|-----|
| InternVL2-4B | 4B | 17.62 | 14.73 | 10.28 |
| Qwen2-VL-72B | 72B | 38.08 | 37.64 | 32.28 |
| GPT-4o | 未知 | 42.65 | 39.28 | 32.28 |
| **CoLVA-InternVL2-4B** | **4B** | **49.80** | **42.72** | **44.86** |

CoLVA仅用4B参数即超越GPT-4o 7.15%，超越Qwen2-VL-72B 11.72%。

### 消融实验

| 配置 | Overall Acc. | 提升 |
|------|-------------|------|
| InternVL2-4B (baseline) | 17.62 | - |
| + MMVM数据 | 32.38 | +14.76 |
| + 数据 + OCL (无VE) | 34.05 | +1.67 |
| + 数据 + VE (无OCL) | 32.25 | -0.13 |
| + 数据 + OCL + VE | 40.45 | +8.07 |
| + 数据 + OCL + VE + IA | **45.83** | **+5.38** |

### 关键发现

- MMVM数据本身贡献最大（+14.76%），验证了数据缺失是匹配能力不足的主因
- OCL+VE协同作用明显（+8.07%），单独用VE几乎无效（-0.13%），说明对比学习是VE发挥作用的前提
- 指令增强额外贡献+5.38%，梯度直接回传到物体级特征是关键
- CoLVA不影响通用VQA能力，在MME、POPE、NaturalBench等基准上甚至有提升
- 方法可迁移到Qwen2VL-2B（47.48%）和LLaVA1.5-7B（36.56%）

## 亮点与洞察

- 首次系统性地定义和基准测试了MLLM的视觉对应匹配能力，填补了领域空白
- 精巧的双编码器对比学习设计——不在同一编码器上做对比，而是让辅助编码器在主编码器语义空间中学习判别性特征，兼顾细粒度与对齐
- 自动化数据生成pipeline设计合理：MLLM不直接做匹配（它们不擅长），而是基于给定标注总结视觉线索

## 局限与展望

- 基准集规模（1510样本）相对有限，更大规模评估有待验证
- 预训练需要带分割标注的图像对，数据获取成本较高（虽然通过伪视频缓解）
- 4B模型虽超越72B和GPT-4o，但绝对准确率仍不到50%，说明视觉匹配仍极具挑战

## 相关工作与启发

- 与视觉跟踪、重识别等传统对应学习一脉相承，但将其引入MLLM框架是全新方向
- RADIO作为多模型蒸馏产物的成功使用，启示在MLLM中引入互补视觉编码器的潜力
- 指令增强策略（将物体表示嵌入文本指令）可能适用于其他需要细粒度物体引用的MLLM任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义视觉对应匹配问题并构建完整数据+基准+方法体系
- 实验充分度: ⭐⭐⭐⭐ 36个模型基准测试、多维消融、跨模型迁移验证，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义与动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ 开辟了MLLM视觉对应理解的新研究方向

<!-- RELATED:START -->

## 相关论文

- [MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](../../ACL2025/multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)
- [Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging](../../ACL2025/multimodal_vlm/exploring_compositional_generalization_of_multimodal_llms_for_medical_imaging.md)
- [Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images](visual_chronicles_using_multimodal_llms_to_analyze_massive_collections_of_images.md)
- [Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](../../ACL2025/multimodal_vlm/conflictvis_vision_knowledge_conflict.md)

<!-- RELATED:END -->
