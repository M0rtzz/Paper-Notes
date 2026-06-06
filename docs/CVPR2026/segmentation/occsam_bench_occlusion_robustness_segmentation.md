---
title: >-
  [论文解读] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models
description: >-
  [CVPR 2026][语义分割][遮挡鲁棒性] 提出 OccSAM-Bench 基准，通过合成手术器械遮挡系统评估 SAM 系列模型在内窥镜场景下的鲁棒性，并设计三区域评估协议揭示模型在遮挡下的两种行为模式：遮挡感知型和遮挡无关型。
tags:
  - "CVPR 2026"
  - "语义分割"
  - "遮挡鲁棒性"
  - "SAM"
  - "内窥镜"
  - "基准测试"
  - "分割"
---

# Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models

**会议**: CVPR 2026  
**arXiv**: [2604.11711](https://arxiv.org/abs/2604.11711)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 遮挡鲁棒性, SAM, 内窥镜, 基准测试, 分割

## 一句话总结

提出 OccSAM-Bench 基准，通过合成手术器械遮挡系统评估 SAM 系列模型在内窥镜场景下的鲁棒性，并设计三区域评估协议揭示模型在遮挡下的两种行为模式：遮挡感知型和遮挡无关型。

## 研究背景与动机

**领域现状**：SAM 及其后续模型（SAM 2、SAM 3、MedSAM 等）在医学图像分割中展现了出色的零样本泛化能力，但现有评估几乎都在干净、精心策划的医学图像上进行。

**现有痛点**：临床内窥镜操作中，目标组织经常被手术器械遮挡，但目前没有基准系统性地量化基础分割模型在手术器械遮挡下的鲁棒性。更严重的是，标准的全掩码评估在手术场景中存在根本性缺陷——一个错误地将组织"幻觉"到手术器械上的模型可能因为恰好与隐藏的真值重叠而获得高分。

**核心矛盾**：全掩码（amodal）评估指标无法区分"正确拒绝遮挡物"和"错误穿过遮挡物预测"这两种截然不同的临床行为。

**本文目标**：(1) 建立受控的手术遮挡生成框架；(2) 提出能区分不同模型行为的评估协议；(3) 系统评估 SAM 系列模型的遮挡鲁棒性。

**切入角度**：在已知真值的条件下合成遮挡，从而可以精确计算可见区域、不可见区域和完整区域各自的分割性能。

**核心 idea**：设计三区域评估协议（可见、不可见、完整），替代传统的全掩码单一评估，揭示模型在遮挡下的真实行为。

## 方法详解

### 整体框架

OccSAM-Bench 包含三个核心组件：(1) 受控遮挡生成管线，在三个公开息肉数据集上合成两种遮挡类型和三个严重程度级别；(2) 三区域评估协议，将分割性能分解为可见、不可见和完整三个目标；(3) 系统评估七个 SAM 系列模型的零样本性能。

### 关键设计

1. **受控遮挡生成**:

    - 功能：模拟手术场景中的两种遮挡类型
    - 核心思路：手术器械粘贴（从 Kvasir-Instrument 数据集采样真实器械掩码，随机缩放和旋转后叠加到目标图像上）和 CutOut 遮挡（在目标区域内放置矩形掩码移除图像内容）。通过遮挡比率 $r = |M_{full} \cap M_{occluder}| / |M_{full}|$ 控制三个严重程度级别：低（0-20%）、中（20-40%）、高（40-60%）
    - 设计动机：手术器械粘贴引入视觉混淆，CutOut 仅移除信息不引入外来内容，两者互补地隔离视觉混淆和数据缺失的影响

2. **三区域评估协议**:

    - 功能：分解分割性能以揭示模型在遮挡下的真实行为
    - 核心思路：定义 $M_{vis} = M_{full} \setminus M_{occ}$（可见掩码）和 $M_{inv} = M_{full} \cap M_{occ}$（不可见掩码），分别评估模型对可见组织的分割能力、对遮挡区域的预测倾向以及整体性能。可见 DSC 惩罚预测延伸到器械区域的假阳性
    - 设计动机：标准全掩码评估可能奖励临床上错误的预测（穿过器械预测组织），三区域协议直接对齐手术安全约束

3. **模型行为分类**:

    - 功能：将 SAM 系列模型分为两种行为原型
    - 核心思路：遮挡感知型（SAM、SAM 2、SAM 3、MedSAM3）优先准确分割可见组织并拒绝器械；遮挡无关型（MedSAM、MedSAM2）自信地预测到遮挡区域，表现出 amodal 补全行为
    - 设计动机：模型选择应由临床意图驱动（保守的可见组织分割还是隐藏解剖结构的推断），而非仅看干净图像上的性能

### 损失函数 / 训练策略

本文是基准测试论文，不涉及模型训练。评估使用 DSC 和 95% Hausdorff 距离作为指标，支持边界框和点提示两种模式。

## 实验关键数据

### 主实验

| 模型 | 类型 | 可见 DSC (工具高) | 不可见 DSC (工具高) | 全掩码 DSC (工具高) |
|------|------|------------------|-------------------|-------------------|
| SAM 3 | 遮挡感知 | **0.72** | 0.15 | 0.58 |
| MedSAM3 | 遮挡感知 | 0.70 | 0.18 | 0.56 |
| MedSAM2 | 遮挡无关 | 0.65 | **0.52** | **0.62** |
| MedSAM | 遮挡无关 | 0.58 | 0.45 | 0.55 |
| SAM-Med2D | 均不符合 | 0.42 | 0.22 | 0.35 |

### 消融实验

| 遮挡类型 | 严重程度 | SAM 3 可见DSC | MedSAM2 可见DSC |
|---------|---------|-------------|---------------|
| 工具 | 低 | 0.85 | 0.78 |
| 工具 | 中 | 0.78 | 0.72 |
| 工具 | 高 | 0.72 | 0.65 |
| CutOut | 低 | 0.83 | 0.80 |
| CutOut | 中 | 0.76 | 0.74 |
| CutOut | 高 | 0.68 | 0.66 |

### 关键发现

- MedSAM2 是唯一在保持竞争力的可见 DSC 同时实现高不可见分数的模型，可能得益于其基于视频的微调策略
- SAM-Med2D 在所有条件下表现不佳，与两种行为模式都不匹配
- 全掩码评估确实会误导：遮挡感知和遮挡无关模型可能获得相似的全掩码分数，但临床行为截然不同

## 亮点与洞察

- 三区域评估协议是一个简单但深刻的贡献：可见 DSC 作为主要鲁棒性指标，直接惩罚临床危险的过度分割，这个思路可推广到任何有遮挡的分割场景
- 发现医学微调方向决定了遮挡行为：通用 SAM 微调产生遮挡感知行为，而将医学图像当作视频序列处理（MedSAM2）则产生 amodal 补全倾向

## 局限与展望

- 合成遮挡无法完全模拟真实手术中的光学物理（组织变形、镜面反射等）
- 仅评估了息肉分割，未涉及其他解剖结构
- 未探索多点击或正负点对等更复杂的提示策略
- 可扩展到视频场景，评估时间维度的遮挡鲁棒性

## 相关工作与启发

- **vs SAMEO**: SAMEO 针对自然图像的 amodal 分割，本文指出直接迁移 amodal 评估到手术环境是有问题的，因为器械是明确的非目标遮挡物
- **vs 标准医学评估**: 现有评估如 SAMed 仅在干净图像上进行，忽略了遮挡这一关键临床挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ 三区域协议和行为分类是新颖的评估范式
- 实验充分度: ⭐⭐⭐⭐ 7个模型×3个数据集×2种遮挡×3个级别的全面评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，协议描述严谨
- 价值: ⭐⭐⭐⭐ 对医学分割模型部署有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](../../NeurIPS2025/segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2025\] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models](../../CVPR2025/segmentation/sketchfusion_learning_universal_sketch_features_through_fusing_foundation_models.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)

</div>

<!-- RELATED:END -->
