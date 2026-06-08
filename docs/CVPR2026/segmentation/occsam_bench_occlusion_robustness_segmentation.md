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

OccSAM-Bench 想回答一个被现有评估回避的问题：当手术器械挡住目标组织时，SAM 这类基础分割模型到底还可不可靠。它的整条管线是这样转的——先在三个公开息肉数据集上，用已知真值合成两种类型、三个严重程度的器械遮挡；再把每张遮挡图喂给模型做零样本分割；最后不再用一个全掩码分数笼统打分，而是把预测拆到可见区、不可见区、完整区三个子目标上分别算 DSC，从而看清模型是"老实分割看得见的部分"还是"硬穿过器械去猜"。整个基准覆盖七个 SAM 系列模型，支持边界框和点提示两种调用方式。

### 关键设计

**1. 受控遮挡生成：在已知真值上合成遮挡，才能精确算出哪块被挡了。**

真实手术图里组织被器械挡住时，我们并不知道被挡部分的真值长什么样，自然没法量化"遮挡到底损害了多少分割质量"。OccSAM-Bench 反过来从干净的息肉图出发主动加遮挡：一种是**手术器械粘贴**，从 Kvasir-Instrument 采样真实器械掩码，随机缩放旋转后叠到目标上，既挡住组织又引入逼真的器械纹理；另一种是 **CutOut**，直接在目标区内放一个矩形掩码抠掉图像内容。两者刻意互补——器械粘贴同时带来"遮挡"和"视觉混淆"（模型可能把器械误当组织），CutOut 则只移除信息、不引入外来纹理，于是能把"数据缺失"和"视觉混淆"两种干扰因素分开来归因。遮挡程度由比率 $r = |M_{full} \cap M_{occluder}| / |M_{full}|$ 量化，分成低（0–20%）、中（20–40%）、高（40–60%）三档，让后续能画出鲁棒性随遮挡加重的衰减曲线。

**2. 三区域评估协议：可见 DSC 把"穿过器械乱猜"直接判成失分。**

标准的全掩码（amodal）评估只比对模型预测和完整真值的重叠，这在手术场景里会奖励一种危险行为：一个把组织"幻觉"到器械上的模型，恰好和被挡住的真值重叠，反而拿到高分。为此协议把完整真值 $M_{full}$ 沿遮挡掩码切成两块——可见掩码 $M_{vis} = M_{full} \setminus M_{occ}$ 和不可见掩码 $M_{inv} = M_{full} \cap M_{occ}$，再分别算三个分数。关键是**可见 DSC**：它只看模型对露出来那部分组织分得准不准，而模型一旦把预测延伸进器械区域，这些落在 $M_{vis}$ 之外的像素就成了假阳性被扣分。这样一来，"老实分割可见组织、拒绝器械"和"穿过器械硬猜"两种在全掩码下可能拿到相近分数的行为，被可见/不可见两个维度清晰地分了开，评估口径也直接对齐了手术安全约束。

**3. 模型行为分类：把七个模型归到两种原型，让选型由临床意图决定。**

有了三区域分数后，模型的差异不再是一个标量高低，而是呈现出两种稳定的行为原型。一类是**遮挡感知型**（SAM、SAM 2、SAM 3、MedSAM3），它们倾向于准确分割可见组织、主动拒绝器械区域，可见 DSC 高但不可见 DSC 低；另一类是**遮挡无关型**（MedSAM、MedSAM2），它们会自信地把预测补到遮挡区域里，表现出 amodal 补全倾向，不可见 DSC 明显更高。这套分类的意义在于：它把"该选哪个模型"从"谁在干净图上分数高"变成了"临床上你想要什么"——若要保守地只分割确实看得见的组织就选遮挡感知型，若要推断被挡住的隐藏解剖结构就选遮挡无关型。

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
