---
title: >-
  [论文解读] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation
description: >-
  [CVPR 2026][图像分割][零样本分割] 提出DSS三阶段渐进式pipeline(Discover→Segment→Select)，通过自监督视觉编码器+Leiden聚类发现前景(FOD)、SAM生成候选mask、启发式评分+MLLM成对比较选择最优mask，实现零样本无训练的伪装目标分割，尤其在多实例场景上显著优于现有方法。
tags:
  - CVPR 2026
  - 图像分割
  - 零样本分割
  - 伪装目标检测
  - SAM
  - MLLM
  - 无训练pipeline
  - 聚类定位
---

# DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation

**会议**: CVPR 2026  
**arXiv**: [2602.19944](https://arxiv.org/abs/2602.19944)  
**代码**: 待确认  
**领域**: 零样本伪装目标分割  
**关键词**: [零样本分割, 伪装目标检测, SAM, MLLM, 无训练pipeline, 聚类定位]  

## 一句话总结
提出DSS三阶段渐进式pipeline(Discover→Segment→Select)，通过自监督视觉编码器+Leiden聚类发现前景(FOD)、SAM生成候选mask、启发式评分+MLLM成对比较选择最优mask，实现零样本无训练的伪装目标分割，尤其在多实例场景上显著优于现有方法。

## 背景与动机
伪装目标分割(COS)要求从高度与背景融合的图像中检测并分割隐匿目标。现有zero-shot COS方法普遍采用"MLLM定位→SAM分割"两阶段范式：先让多模态大语言模型(MLLM)生成目标位置提示(如bounding box)，再将提示送入SAM进行像素级分割。然而MLLM的视觉定位能力在伪装场景下严重退化——伪装目标与背景颜色/纹理高度相似，MLLM难以准确锚定目标区域，生成的bbox偏差大；在多实例场景下问题更严重，MLLM往往只能定位到最显著的一个目标而遗漏其余。

## 核心问题
零样本COS中MLLM定位不准导致SAM分割质量受限，尤其在多实例伪装场景下MLLM无法可靠地发现所有目标。需要一种不依赖MLLM定位、能自动发现多个伪装目标并从候选mask中选择最优结果的无训练方案。

## 方法详解

### 整体框架
DSS是一个三阶段pipeline：**Discover**——用自监督视觉特征的聚类替代MLLM定位来发现伪装目标区域并生成bbox prompt；**Segment**——将bbox送入SAM生成候选mask集合；**Select**——通过启发式评分筛选+MLLM成对比较迭代选出最终mask。整个流程zero-shot、training-free，无需任何微调或标注数据。

### 阶段一：Discover (FOD — Foreground Object Discovery)

1. **Patch-level特征提取**: 用自监督预训练的视觉编码器(如DINOv2)提取图像的patch-level特征矩阵X∈R^{N×D}，N为patch数量，D为特征维度。

2. **Leiden聚类初始化**: 对patch特征用Leiden社区检测算法进行聚类，得到初始的前景/背景粗略划分。Leiden算法基于图的模块度优化，能自动发现特征空间中的自然聚类结构，无需预设类别数。

3. **Part Composition (PC) 迭代精炼**: 对初始聚类结果进行迭代优化。每轮计算每个patch的前景/背景归属概率：
$$y_i^{(t)} = \sigma\left(\|x_i - \mu_b\|_2 - \|x_i - \mu_f\|_2\right)$$
其中μ_f和μ_b分别为当前前景和背景patch的特征均值中心，σ为sigmoid函数。直觉是：离前景中心近、离背景中心远的patch更可能是前景。迭代直到能量函数E收敛，E衡量当前划分的整体一致性。

4. **Similarity-based Box Generation (SBG)**: 计算前景centroid与全图所有patch的余弦相似度，生成affinity map。在affinity map上通过阈值化和连通域分析提取候选区域。对候选区域用**Pearson correlation去重**(阈值τ=0.95)——相关性高于0.95的区域视为同一目标合并，避免同一目标生成多个冗余bbox。最终输出的bbox作为SAM的prompt。

### 阶段二：Segment
将FOD生成的所有bbox prompt送入SAM(Segment Anything Model)，每个bbox生成一组候选mask，汇总为候选mask集合M_FOD。SAM作为通用分割基础模型，能从给定的位置提示生成高质量的像素级mask。

### 阶段三：Select (SMS — Segment Mask Selection)

1. **启发式评分**: 对每个候选mask m_i计算质量得分：
$$s_i = \text{corr}(m_i, \text{sim}_i) + (1 - \text{BC}(m_i))$$
其中corr(m_i, sim_i)是mask与affinity map的Pearson相关系数——衡量mask区域是否与前景特征分布一致；BC(m_i)是mask的boundary complexity(边界复杂度)——惩罚过度碎片化的mask。两项相加，高质量mask应同时具备高特征一致性和低边界复杂度。

2. **Top-K筛选**: 按得分排序，保留Top-K个候选mask进入精选阶段。

3. **Iterative Pairwise MLLM Comparison**: 从得分最低的mask开始，两两送入MLLM进行成对比较——"哪个mask更好地分割了伪装目标？"MLLM在成对比较中的判断远比直接定位准确(降低了任务难度)。从低分到高分迭代对比，最终胜出的mask即为输出。这种从差到好的比较顺序让MLLM逐步理解什么是更好的mask，减少单次判断误差的累积。

### 损失函数 / 训练策略
无训练，整个pipeline是inference-only的。超参数包括：Leiden聚类的分辨率参数、PC迭代的收敛阈值、Pearson去重阈值τ=0.95、启发式评分的Top-K值。

## 实验关键数据
| Benchmark | 指标 | DSS | 前SOTA(ZS方法) | 提升 |
|-----------|------|-----|------------|------|
| CHAMELEON | S_m↑ | 显著领先 | MLLM+SAM baseline | +大幅 |
| CAMO | S_m↑ | 显著领先 | MLLM+SAM baseline | +大幅 |
| COD10K | S_m↑ | 显著领先 | MLLM+SAM baseline | +大幅 |
| NC4K | S_m↑ | 显著领先 | MLLM+SAM baseline | +大幅 |

- 在多实例伪装场景中优势最大，因为FOD能自动发现多个目标区域，而MLLM baseline往往只定位单一目标
- 与使用MLLM定位的zero-shot方法相比，DSS在所有COS benchmark上达到SOTA
- Training-free，无需任何COS标注数据

### 消融实验要点
- FOD vs MLLM定位：FOD在多实例场景上发现目标数量远超MLLM
- PC迭代精炼贡献显著：移除PC后bbox质量明显下降
- SBG中的Pearson去重(τ=0.95)有效减少冗余bbox
- SMS中MLLM成对比较优于直接用启发式评分作为最终选择
- 从低分到高分的比较顺序优于随机顺序

## 亮点
- 巧妙地将MLLM从"定位者"转变为"裁判"——用自监督视觉特征+聚类替代MLLM做定位(更可靠)，让MLLM只做成对比较(更擅长)，任务分配合理
- PC迭代精炼公式简洁优雅，前景/背景距离差的sigmoid可直接解释为概率
- Pearson correlation去重是一种轻量有效的重复检测方式
- 三阶段渐进式设计层次清晰，每个阶段的目标明确且可独立评估
- Zero-shot + training-free的设置使方法具有极强的泛化能力和部署灵活性

## 局限与展望
- 依赖SAM和MLLM两个大模型，推理开销不小(尤其SMS阶段的多次MLLM调用)
- Leiden聚类和PC精炼假设前景/背景在特征空间可分，对于极度伪装(几乎零特征差异)的场景可能失效
- MLLM成对比较的Top-K设置和迭代次数影响效率和质量的trade-off，需调参
- 未探索不同自监督backbone(如MAE、CLIP)对FOD的影响
- Pearson去重阈值τ=0.95为固定值，未做自适应优化

## 与相关工作的对比
- **vs GenSAM/LAKE-RED等MLLM+SAM方法**: 这些方法依赖MLLM生成prompt来引导SAM，在伪装场景下MLLM定位不准是瓶颈。DSS用FOD替代MLLM定位，从根源解决了定位失败问题。
- **vs COS全监督方法(如SINet等)**: 全监督方法依赖大量像素级标注，泛化到新域受限。DSS作为zero-shot方法虽在精度上未必超越全监督SOTA，但在泛化性和标注成本上优势明显。
- **vs 通用zero-shot分割(如Matcher等)**: 通用方法未针对伪装场景优化，在前背景相似度极高时表现差。DSS的FOD专为伪装场景设计，利用细粒度patch特征的聚类分析来突破视觉相似性。

## 启发与关联
- **idea**: FOD的聚类+迭代精炼范式可迁移到其他"目标定位困难"的场景，如水下目标检测、夜间目标发现
- **idea**: SMS的启发式评分+MLLM成对比较可作为通用的"mask质量选择器"，嵌入其他分割pipeline的后处理阶段
- **idea**: 将MLLM角色从定位者转为裁判的思路可推广——在其他视觉任务中也将大模型用在其更擅长的比较/判断环节
- 与EReCu的无监督COS方案互补：EReCu需要训练但处理unsupervised设置，DSS完全免训练但需要SAM+MLLM
- PC的能量收敛机制可与主动推理结合，实现自适应的前景发现深度控制

## 评分
- 新颖性: ⭐⭐⭐⭐ 三阶段pipeline设计新颖，将MLLM角色转换的insight有价值
- 实验充分度: ⭐⭐⭐⭐ 多个COS benchmark + 消融实验 + 多实例分析
- 写作质量: ⭐⭐⭐⭐ 三阶段命名直观，动机阐述清晰
- 对我的价值: ⭐⭐⭐⭐ zero-shot pipeline设计范式可借鉴，FOD和SMS模块可复用

<!-- RELATED:START -->

## 相关论文

- [DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](dss_discover_segment_select_zero_shot_cos.md)
- [FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)
- [EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](erecu_pseudo-label_evolution_fusion_and_refinement_with_multi-cue_learning_for_u.md)
- [SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)

<!-- RELATED:END -->
