---
title: >-
  [论文解读] LMM-Det: Make Large Multimodal Models Excel in Object Detection
description: >-
  [ICCV 2025][目标检测][大型多模态模型] 提出 LMM-Det，通过系统分析发现大型多模态模型在目标检测中核心瓶颈是低召回率，并通过数据分布调整（伪标签增强）和推理优化（按类别逐一检测）将 LMM 的 COCO AP 从 0.2 提升至 47.5，无需任何额外专用检测模块。
tags:
  - "ICCV 2025"
  - "目标检测"
  - "大型多模态模型"
  - "召回率"
  - "数据分布调整"
  - "推理优化"
---

# LMM-Det: Make Large Multimodal Models Excel in Object Detection

**会议**: ICCV 2025  
**arXiv**: [2507.18300](https://arxiv.org/abs/2507.18300)  
**代码**: [github.com/360CVGroup/LMM-Det](https://github.com/360CVGroup/LMM-Det)  
**领域**: 目标检测  
**关键词**: 大型多模态模型, 目标检测, 召回率, 数据分布调整, 推理优化

## 一句话总结

提出 LMM-Det，通过系统分析发现大型多模态模型在目标检测中核心瓶颈是低召回率，并通过数据分布调整（伪标签增强）和推理优化（按类别逐一检测）将 LMM 的 COCO AP 从 0.2 提升至 47.5，无需任何额外专用检测模块。

## 研究背景与动机

### 领域现状

大型多模态模型（LMMs）如 LLaVA、InternVL 等在图像描述、VQA、视觉定位等任务上表现优异，展现了出色的多模态理解和推理能力。然而，在标准目标检测（定位并分类图像中所有物体）这一基础视觉任务上，LMMs 与专用检测器（如 RT-DETR、Salience-DETR）之间存在巨大的性能鸿沟。

### 现有痛点

**现有方案依赖额外检测模块**：Groma 集成了 RPN，VisionLLM v2 使用 Grounding DINO 作为外接检测头。这些方案受限于额外模块的性能，引入额外延迟，且未挖掘 LMM 自身的检测潜力

**LMM 原生检测能力极差**：LLaVA 在 COCO 上零样本仅 0.2 AP。即使用 COCO + Object365 训练并提升分辨率，也只达到 38.7 AP，远低于专用检测器的 55+ AP

**核心问题未被诊断**：之前的工作未系统分析 LMM 在检测任务上失败的根本原因

### 核心发现与切入

通过系统的可视化和分布分析，作者发现**核心瓶颈是低召回率**：LMM 训练后的预测框数量分布会逼近训练集分布，而 COCO 的不完整标注导致模型过早截断预测，每张图片平均只生成约 7 个框。此外，LMM 的自回归预测机制天然难以生成大量高质量 proposal。

## 方法详解

### 整体框架

LMM-Det 由视觉编码器（OWLv2-ViT）、线性投影器和大语言模型（Vicuna-1.5-7B）组成。训练分四个阶段：Stage I 对齐视觉-语言模块；Stage II 在 Object365 上预训练检测能力；Stage III 在 COCO 上微调；可选 Stage IV 混合 LLaVA 数据保持多模态通用能力。

### 关键设计

#### 1. **数据分布调整（Data Distribution Adjustment）**
- **功能**：通过伪标签增强训练数据，增加每张图片的标注框数量，从而提升模型的召回率
- **核心思路**：使用预训练的专用检测器（Salience-DETR）为每张训练图片生成伪标签，与原始 GT 标注通过 NMS 合并。同时让模型输出每个框的坐标和置信度分数（GT 标注置信度为 1，伪标签的置信度由检测器给出）
- **设计动机**：训练后的 LMM 会逼近训练数据分布。COCO 标注不完整导致模型预测过早截断——通过增加标注密度，可以引导模型生成更多候选框，从而提升召回率。注意：伪标签仅用于扩充数据，LMM-Det 的推理阶段完全不依赖任何额外检测模块

#### 2. **推理优化（Inference Optimization）**
- **功能**：将"一次性输出所有框"改为"按类别逐一检测"
- **核心思路**：不让 LMM 在一步内输出所有类别的所有框，而是对每个类别独立提问。为保持训练和推理的一致性，重构了指令数据格式，采用类别特定的预测策略
- **设计动机**：当前 LMM 使用固定采样策略在单步预测中难以生成足够数量的细粒度 proposals。按类别独立检测可以显著增加总 proposal 数量。消融实验显示该策略将 AP 从 44.2% 提升至 47.5%，AR@100 从 56.0% 提升至 63.6%

#### 3. **Token 表示验证**
- **功能**：探索坐标和置信度的 token 表示方式
- **核心思路**：对比直接输出 token 预测 vs 扩展词汇表两种方案
- **设计动机**：直接输出 token 不需要额外训练词汇嵌入，且实验表明其检测精度更优

### 损失函数 / 训练策略

标准的语言建模损失（next-token prediction）：
$$\max_\theta \sum_{i=1}^L \log p_\theta(\tilde{\mathbf{y}}_i | \mathbf{x}_v, \mathbf{x}_t, \mathbf{y}_{1:i-1})$$

训练使用 595K 图文对 + 1.86M 图像，在 6 节点（每节点 8×H800）上总计 176 小时。

## 实验关键数据

### 主实验（零样本，COCO val）

| 方法 | 视觉编码器 | LLM | 额外检测模块 | AP | AP50 | AR@100 |
|------|----------|-----|-----------|------|------|--------|
| LLaVA | CLIP-L | Vicuna-7B | 无 | 0.2 | 0.6 | 11.2 |
| KOSMOS-2 | CLIP-L | MAGNETO | 无 | 7.6 | 13.7 | 18.2 |
| InternVL-2.5 | InternViT | Internlm2.5-7B | 无 | 11.8 | 18.4 | 27.5 |
| Groma | DINOv2 | Vicuna-7B | **有** | 12.8 | 17.0 | 22.5 |
| **LMM-Det** | OWLv2-L | Vicuna-7B | **无** | **24.5** | **34.7** | **46.6** |

### 主实验（微调后，COCO val）

| 方法 | 额外模块 | AP | AP50 | AP75 | AR@100 |
|------|---------|------|------|------|--------|
| RT-DETR (专用检测器) | - | 55.3 | 73.4 | 60.0 | 74.4 |
| Salience-DETR (专用检测器) | - | 57.3 | 75.5 | 62.4 | 75.4 |
| Groma (DINOv2) | 有 | 43.6 | - | - | - |
| VisionLLM v2 | 有 (Grounding DINO) | 56.3↓ | 74.3 | 61.6 | - |
| LLaVA* (重训) | 无 | 38.7 | 55.8 | 41.3 | 50.5 |
| **LMM-Det** | **无** | **47.5** | **66.5** | **51.1** | **63.6** |

### 消融实验

| 配置 | AP | AP50 | AP75 | AR@100 | 说明 |
|------|------|------|------|--------|------|
| Baseline (CLIP-ViT) | 38.7 | 55.8 | 41.3 | 50.5 | LLaVA* 重训 |
| + OWLv2-ViT | 42.1 | 57.8 | 45.8 | 51.3 | +3.4 AP |
| + 数据分布调整 | 44.2 | 61.3 | 47.5 | 56.0 | +2.1 AP, +4.7 AR |
| + 推理优化 | **47.5** | **66.5** | **51.1** | **63.6** | +3.3 AP, +7.6 AR |

### 多模态能力保持

| 模型 | COCO AP | 图像描述 CIDEr | VQAv2 Accuracy |
|------|---------|-------------|---------------|
| LLaVA | 0.2 | 108.9 | 78.5 |
| LMM-Det† | 47.1 | 99.0 | 74.1 |

### 关键发现

- LMM 检测能力差的根本原因是**召回率不足**（AR@100 仅 50.5），而非定位精度
- 预测框分布会逼近训练集分布——COCO 不完整标注导致模型学会了"少预测"
- OWLv2-ViT 比 CLIP-ViT 更适合检测任务，提供更好的高分辨率输入支持
- LMM-Det† 版本在获得检测能力的同时，VQA 和描述能力仅有轻微下降（约 4%）
- 单张图像推理需要约 4 秒，在实时检测场景下竞争力不足

## 亮点与洞察

1. **诊断优先的方法论**：先系统分析失败原因（低召回率），再针对性设计解决方案，这种研究范式值得学习
2. **数据分布视角的洞察**：LMM 会学习训练数据的分布特征（如每张图的平均框数），不完整标注导致的"分布截断"是一个被忽视但重要的问题
3. **最小化架构改动**：不引入任何额外检测模块，仅通过数据和推理策略的优化就大幅提升性能，证明了 LMM 内在的检测能力
4. **多任务兼容性**：可选的 Stage IV 展示了检测能力与通用多模态能力的良好兼容

## 局限与展望

1. **推理速度慢**：每张图约 4 秒（需逐类别预测），远不如实时检测器
2. **与专用检测器仍有差距**：47.5 vs 57.3 AP，约 10 个点的差距
3. **依赖伪标签质量**：数据分布调整依赖 Salience-DETR 生成的伪标签，质量受检测器限制
4. **仅评估 COCO**：未在 LVIS、Objects365 等更大规模/长尾数据集上验证
5. **缺少开放词汇检测评估**：仅评估固定类别集的检测，未探索 LMM 在开放词汇检测上的优势

## 相关工作与启发

- Shikra 和 KOSMOS-2 展示了 LMM 在 REC 任务上的定位能力，但整图检测（需枚举所有物体）是更难的任务
- VisionLLM v2 集成 Grounding DINO 后反而导致原有检测性能下降，说明简单拼接模块并非最优解
- 按类别逐一检测的推理策略本质上将"一次生成所有"转化为多个"条件生成"任务，更适合 LMM 的自回归生成范式

## 评分

- **新颖性**: ⭐⭐⭐ — 方法本身（伪标签 + 按类别检测）比较直接，但核心贡献在于系统的诊断分析
- **实验充分度**: ⭐⭐⭐⭐ — 零样本 + 微调 + 消融 + 多任务评估，覆盖全面
- **写作质量**: ⭐⭐⭐⭐ — 从探索性实验到分析再到方案的逻辑链清晰流畅
- **价值**: ⭐⭐⭐⭐ — 首次系统证明 LMM 无需额外模块即可做目标检测，为社区提供了重要基线和分析框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](../../CVPR2025/object_detection/towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](../../CVPR2025/object_detection/large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)

</div>

<!-- RELATED:END -->
