---
title: >-
  [论文解读] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
description: >-
  [NeurIPS 2025][图像分割][3D部件分割] 提出 PartNeXt，一个包含 23,519 个高质量带纹理 3D 模型、跨 50 个类别的细粒度层级部件标注数据集，并建立了类别无关部件分割和 3D 部件问答两个基准测试，揭示了当前方法在细粒度部件理解上的显著不足。
tags:
  - NeurIPS 2025
  - 图像分割
  - 3D部件分割
  - 数据集
  - 层级标注
  - 众包标注
  - 3D视觉语言模型
---

# PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2510.20155](https://arxiv.org/abs/2510.20155)  
**代码**: [GitHub](https://authoritywang.github.io/partnext)  
**领域**: Segmentation / 3D部件理解  
**关键词**: 3D部件分割, 数据集, 层级标注, 众包标注, 3D视觉语言模型

## 一句话总结

提出 PartNeXt，一个包含 23,519 个高质量带纹理 3D 模型、跨 50 个类别的细粒度层级部件标注数据集，并建立了类别无关部件分割和 3D 部件问答两个基准测试，揭示了当前方法在细粒度部件理解上的显著不足。

## 研究背景与动机

**领域现状**：PartNet 数据集推动了 3D 部件级理解的发展，提供了 26K 模型、24 类别的 573K 部件标注。

**现有痛点**：
   - PartNet 的标注工具需要重新网格化（remeshing），导致部分物体纹理丢失和几何变形，限制了视觉线索的利用。
   - PartNet 的标注界面要求 3D 建模专业知识（手动画曲线切割网格、检查切面截面），不适合众包扩展。
   - 现有数据集类别覆盖有限（24类），且以无纹理几何体为主。

**核心矛盾**：细粒度 3D 部件标注的高质量需求 vs. 标注的可扩展性和易用性之间的矛盾。

**本文目标**：创建一个高质量、可扩展的下一代 3D 部件标注数据集，同时提供新的评测基准。

**切入角度**：设计全 Web 化众包标注界面 + AI 辅助层级定义 + 直接在带纹理网格上标注。

**核心idea**：通过标注工具创新和 AI 辅助实现大规模、细粒度的带纹理 3D 部件标注。

## 方法详解

### 整体框架

PartNeXt 的构建包含四个环节：
1. 数据收集与预处理（从 Objaverse、ABO、3D-FUTURE 筛选高质量模型）
2. 层级定义与示例生成（GPT-4o 辅助 + 人工审核）
3. 标注系统设计（Web 化众包平台）
4. 基准测试建立（部件分割 + 部件问答）

### 关键设计

1. **数据收集与 CLIP 过滤**

    - **功能**：从三个大规模 3D 数据集中筛选高质量、类别一致的模型。
    - **怎么做**：
        - ABO 和 3D-FUTURE 直接按类别过滤
        - Objaverse 规模大但质量参差：先过滤动画模型、超大面数（>130K）、扫描/建筑模型
        - 使用 CLIP 文本编码器编码约 100 个类别名和 Cap3D 提供的描述，通过余弦相似度分类
        - 最高相似度 < 0.75 的模型被丢弃
        - 最终选择模型数最多的 50 个类别

2. **AI 辅助层级定义**

    - **功能**：为每个类别定义细粒度、一致的部件层级结构。
    - **为什么**：手动定义详细层级耗时费力，特别是枚举多种部件变体时。
    - **怎么做**：
        - 制定 5 条层级设计准则：功能感知、层级化、穷举变体、原子性、一致性
        - 用 GPT-4o 生成粗层级 → 渲染图辅助优化 → 人工专家审核
        - 用 GPT-4o 图像生成能力为每个部件节点生成可视化参考示例
    - 最终层级深度范围：4~10 层

3. **Web 众包标注系统**

    - **功能**：设计高效、易用的 3D 部件标注界面。
    - **三大核心特性**：
        - **层级标注工作流**：可折叠树结构，逐步展开标注叶节点，支持 "Other" 节点处理意外部件
        - **双面板界面**：左侧显示未分割网格，右侧显示已分割结果（同一视角）。标注部件从左转移到右，每个部件唯一颜色标识。特别适合标注被遮挡的内部部件。
        - **面选择工具套件**：
       - 连通分量选择：点击自动选择整个连通区域
       - 包围框选择：2D 框投影选择所有可见面
       - 逐面选择：精细控制
    - **区别于 PartNet**：直接在带纹理的原始网格上操作，无需 remeshing，保留纹理信息

4. **标注质量控制**

    - 35 名专业标注员 + 5 名顶级标注员负责数据验证
    - 标注员完成两天培训
    - 每个标注至少经过一次审核，共 5,211 次修正
    - 单个模型平均标注时间约 5-6 分钟

### 数据集规模

| 维度 | 数据 |
|------|------|
| 模型总数 | 23,519 |
| 部件实例总数 | 350,187 |
| 类别数 | 50 |
| 数据来源 | Objaverse (14,811) + ABO (2,633) + 3D-FUTURE (6,075) |
| 层级深度 | 4~10 |

## 实验关键数据

### 主实验一：类别无关 3D 部件实例分割

评估 250 个物体（50 类 × 5 个），使用叶节点作为 ground truth。

| 方法 | Bed | Bottle | Chair | Knife | Table | Controller | Fan | Glasses | Monitor | Wrench | mIoU |
|-----|-----|--------|-------|-------|-------|-----------|-----|---------|---------|--------|------|
| SAMPart3D | 17.51 | 47.71 | 28.49 | 61.08 | 25.86 | 24.00 | 31.12 | 28.34 | 25.70 | 40.53 | 36.78 |
| PartField | 24.77 | 67.91 | 43.78 | 68.22 | 53.26 | 41.57 | 46.66 | 55.57 | 45.97 | 60.53 | 50.22 |
| SAMesh | **82.59** | 35.63 | **72.57** | 51.19 | **64.81** | **47.71** | **56.72** | 33.38 | 45.16 | 52.17 | **51.57** |

**发现**：三种 SOTA 方法在 PartNeXt 上表现均较差（最佳 mIoU 仅 51.57%），说明细粒度部件分割仍是巨大挑战。各方法各有特点：SAMesh 擅长细粒度但过分割，PartField 对连通区域分割不足，SAMPart3D 在弱纹理区域连续性差。

### 主实验二：3D 部件问答

| 任务 | 指标 | 3DLLM | PointLLM | ShapeLLM |
|------|------|-------|----------|----------|
| Part Count (有类别) | MAE↓ | 2.16 | 1.87 | 1.72 |
| Part Count (无类别) | MAE↓ | 2.46 | 1.79 | 1.85 |
| Classification (有类别) | Acc↑ | - | 0.22 | 0.25 |
| Classification (无类别) | Acc↑ | - | 0.18 | 0.08 |
| Grounding (有类别) | IoU↑ | - | - | 0.33 |
| Grounding (无类别) | IoU↑ | - | - | 0.30 |

**发现**：当前 3D 视觉语言模型在部件级推理上严重不足。部件计数 MAE 接近 2、分类准确率仅约 20%、定位 IoU 仅约 30%。

### 消融实验：Point-SAM 训练数据消融

| 评估集 | 训练集 | IoU@1 | IoU@3 | IoU@5 | IoU@7 | IoU@10 |
|-------|--------|-------|-------|-------|-------|--------|
| PartNet-Mobility | PartNet | 39.0 | 53.7 | 58.6 | 60.9 | 62.9 |
| PartNet-Mobility | **PartNeXt** | 40.2 | 57.5 | 63.2 | 65.0 | 67.4 |
| PartNet-Mobility | Mixture | **40.4** | **58.3** | **64.1** | **66.9** | **68.7** |
| PartNeXt | PartNet | 39.9 | 53.9 | 58.4 | 60.4 | 60.3 |
| PartNeXt | **PartNeXt** | 44.3 | 60.1 | 63.2 | 64.8 | 65.9 |
| PartNeXt | Mixture | **45.3** | **61.7** | **65.3** | **66.6** | **67.6** |

**发现**：仅用 PartNeXt 训练的 Point-SAM 即显著超过 PartNet 训练版本（IoU@10: 67.4 vs 62.9），证实了数据集的高质量和多样性。

### 关键发现

- 当前 SOTA 部件分割方法在细粒度层级分割上仍有巨大提升空间（最佳 mIoU 仅 51.57%）。
- 3D LLM 在部件级推理与定位上能力严重不足，部件问答是有价值的新方向。
- 更丰富的训练数据（PartNeXt）能直接带来交互式分割模型的显著提升。

## 亮点与洞察

- **标注工具的系统性创新**：双面板界面 + 三种面选择工具 + 层级标注工作流的组合设计非常实用，大幅降低标注门槛。
- **AI 辅助流程**：CLIP 过滤 + GPT-4o 层级定义 + GPT-4o 参考图生成，形成了完整的 AI 辅助数据构建流程。
- **保留纹理的标注**：直接在纹理网格上标注避免了 PartNet 的 remeshing 问题，为需要纹理信息的下游任务提供支持。
- **两个新基准**：部件分割和部件问答基准揭示了当前方法的不足，为社区指明了方向。

## 局限与展望

- **数据集规模**仍有限（23.5K 模型），未来计划从 Objaverse-XL 扩展。
- **需要预定义层级**：每个类别需精心设计部件层级，限制了开放词汇标注能力。
- **缺乏语义描述标注**：当前仅提供部件名称标注，未提供 caption 或物理属性标注。
- 标注界面仍需手工操作，未来可探索 VLM 辅助的半自动标注。
- 评测中每类仅用 5 个物体，结果可能受样本选择影响。

## 相关工作与启发

- **PartNet → PartNeXt 的演进**：从无纹理 + 专家标注 → 带纹理 + 众包标注，反映了 3D 数据集构建范式的升级。
- **AI 辅助数据构建**：GPT-4o 在层级定义和参考图生成中的应用展示了大模型辅助数据工程的潜力。
- **3D LLM 的局限暴露**：ShapeLLM、PointLLM 等在部件级任务上的不足，提示细粒度 3D 理解可能是下一代 3D 基础模型的重要方向。
- **启发**：数据集质量和多样性是提升模型泛化性的最直接路径。

## 评分

- 新颖性: ⭐⭐⭐⭐ 标注系统设计有创新，部件问答基准为新任务
- 实验充分度: ⭐⭐⭐⭐ 分割和问答双基准测试全面，Point-SAM 消融有说服力
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，图表精美，数据统计详尽
- 价值: ⭐⭐⭐⭐⭐ 高质量数据集对社区有长期价值，基准测试揭示了重要研究缺口

<!-- RELATED:START -->

## 相关论文

- [GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)
- [LangHOPS: Language Grounded Hierarchical Open-Vocabulary Part Segmentation](langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)
- [FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](partonomy_large_multimodal_models_with_part-level_visual_understanding.md)
- [Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](../../CVPR2025/segmentation/fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)

<!-- RELATED:END -->
