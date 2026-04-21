---
title: >-
  [论文解读] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models
description: >-
  [CVPR 2026][多模态][心智理论] 提出VisionToM，一个基于视觉的轻量级干预框架，通过探测和干预MLLM中对视觉输入和ToM推理敏感的注意力头，在不微调模型的情况下显著增强多模态大语言模型的心智理论推理能力，在EgoToM基准上大幅提升表现。
tags:
  - CVPR 2026
  - 多模态
  - 心智理论
  - 多模态大语言模型
  - 注意力干预
  - 视觉推理
  - 幻觉缓解
---

# Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.24484](https://arxiv.org/abs/2603.24484)  
**代码**: 无（项目页面: https://founce.github.io/VisionToM）  
**领域**: 多模态VLM / 心智理论  
**关键词**: 心智理论, 多模态大语言模型, 注意力干预, 视觉推理, 幻觉缓解

## 一句话总结

提出VisionToM，一个基于视觉的轻量级干预框架，通过探测和干预MLLM中对视觉输入和ToM推理敏感的注意力头，在不微调模型的情况下显著增强多模态大语言模型的心智理论推理能力，在EgoToM基准上大幅提升表现。

## 研究背景与动机

1. **领域现状**：心智理论（Theory of Mind, ToM）指推断自我和他人心理状态（欲望、信念、意图）以预测行为的能力。随着LLM的发展，其ToM能力越来越受关注。但现有ToM评估主要集中在文本输入，基于纯视觉信息的场景研究不足。

2. **现有痛点**：(1) 大多数MLLM在仅视觉输入的ToM任务上表现不佳，尤其在Belief和Action推理上与人类基线差距巨大；(2) 现有方法将模型视为黑盒，很少探究注意力在多选问答中的内部行为；(3) LLM幻觉对ToM任务的影响从可解释性角度尚未充分研究；(4) 多模态ToM基准大多依赖模拟环境，缺乏真实世界的生态效度。

3. **核心矛盾**：MLLM在处理ToM任务时过度依赖语言先验而忽视视觉证据。当视觉信息与语言先验冲突时，模型倾向于基于语言模式产生不准确的推断，导致幻觉。现有的可解释性增强方法仅限于文本模态。

4. **本文目标** 如何在不微调MLLM的情况下，通过干预模型内部表征来增强其视觉注意力和ToM推理能力，减少对虚假语言先验的依赖？

5. **切入角度**：通过可解释性分析发现，MLLM在多个ToM任务中展示了视觉注意力的跨任务一致性，而ToM推理的内部表征在任务间分化但任务内一致。这为有针对性的干预提供了依据。

6. **核心 idea**：通过线性探针找到对视觉输入和ToM推理敏感的注意力头，计算从错误到正确的干预向量，在推理时注入这些向量来引导模型关注视觉证据并做出正确推理。

## 方法详解

### 整体框架

VisionToM分为四个阶段：(1) 提取内部表征——构造正负样本对，从MLLM注意力头中提取视觉注意力和ToM推理的表征；(2) 探测——用线性分类器识别哪些注意力头对视觉输入/ToM推理最敏感；(3) 分离ToM推理表征——用聚类+编码器将负样本表征向正样本方向对齐；(4) 干预——在推理时将计算的干预向量注入敏感注意力头。整个过程中MLLM backbone保持冻结。

### 关键设计

1. **视觉注意力增强（Visual Attention Enhancement）**:
    - 功能：减少MLLM对语言先验的过度依赖，增强对视觉输入的关注
    - 核心思路：用PGD对抗攻击（$\epsilon=16/255$，300步）生成视觉扰动样本作为负样本，保持文本问题不变。从正负样本对中提取所有注意力头的激活值，计算平均偏移向量：$\{\delta_{V,l}^h\} = \frac{1}{S}\sum_{i=1}^{S}(X_{V,i,l}^{pos,h} - X_{V,i,l}^{neg,h})$。该偏移向量编码了"视觉信息正确→视觉信息被扰动"的方向，反向施加可引导模型更关注真实视觉信息
    - 设计动机：PGD攻击比随机噪声更有效地暴露注意力失败模式——PGD攻击后Goal准确率从61.5%降到29.1%，而随机噪声仅降到47.0%，说明对抗样本提供了更准确的梯度方向

2. **ToM推理引导（ToM Reasoning Guidance）**:
    - 功能：增强模型的ToM推理能力，引导正确的心理状态推断
    - 核心思路：固定视觉输入，以正确答案为正样本、错误答案为负样本。由于负样本语义多样导致表征分布不均匀，无法直接算偏移向量。采用基于聚类的方法：对每个敏感注意力头的负样本做聚类（k在2-15间自动选择），为每个聚类训练专门的编码器网络 $f_{h,c}$，学习从负样本到正样本的修正向量 $\delta_{h,c,i} = f_{h,c}(x_{T,i}^{neg,h})$。推理时按最近聚类中心选择对应编码器
    - 设计动机：不同类型的推理失败需要不同方向的干预。直接取平均会模糊不同失败模式的区别，聚类+专用编码器的设计实现了更精细的"对症下药"

3. **探测与干预机制（Probing & Intervention）**:
    - 功能：识别敏感头并在推理时精确施加干预
    - 核心思路：对每个注意力头训练独立的线性二分类探针（逻辑回归），验证集准确率高的头被认定为"敏感头"。关键发现：视觉注意力敏感头分布在不同层且跨任务一致，ToM推理敏感头集中在中间层且任务间分化。干预时选择Top-K=64个敏感头，将视觉和ToM两个方向的干预向量相加后注入：$T_{l+1} = T_l + \sum_{h=1}^H(Attn_l^h(P_l^hT_l) + \alpha \times \Delta) \cdot W_l^o$，干预强度$\alpha=1.0$
    - 设计动机：视觉注意力跨任务一致性使得可以共用同一组敏感头；ToM推理任务内一致性使得VisionToM能探测任务特定的ToM嵌入。反向干预（$-\alpha\Delta$）导致性能暴跌，验证了方向正确性

### 损失函数 / 训练策略

- 探针训练：标准交叉熵损失优化logistic回归参数
- 编码器训练：$L_{total} = \sum_h \sum_{c=1}^{k_h^*} \frac{1}{|C_{h,c}|} \sum_{i \in C_{h,c}} \|(x_{T,i}^{neg,h} + \delta_{h,c,i}) - x_{T,i}^{pos,h}\|^2$
- 探针和编码器在30%校准集上训练，70%评估集上推理
- 一次性校准：探针训练约0.2小时，编码器训练约1小时，MLLM backbone保持冻结

## 实验关键数据

### 主实验

| 模型 | 任务 | Baseline | +VisionToM | 提升 |
|--------|------|------|----------|------|
| LLaVA-Next-Video-7B | Goal | 61.5% | 74.5% | +13.0% |
| LLaVA-Next-Video-7B | Belief | 38.9% | 45.3% | +6.4% |
| LLaVA-Next-Video-7B | Actions | 24.0% | 29.7% | +5.7% |
| Qwen2.5-VL-7B | Goal | 86.9% | 88.9% | +2.0% |
| Qwen2.5-VL-7B | Belief | 35.6% | 42.0% | +6.4% |
| Qwen2.5-VL-7B | Actions | 31.1% | 37.6% | +6.5% |
| 人类基线 | Goal/Belief/Actions | 88/72/78% | - | - |

### 消融实验

| 配置 | Goal | Belief | Actions | 说明 |
|------|---------|------|------|------|
| LLaVA Baseline | 61.5% | 38.9% | 24.0% | 无干预 |
| 仅视觉干预 (w/o $\delta_T$) | 73.2% | 39.2% | 25.3% | Goal提升大，Belief/Actions小 |
| 仅ToM干预 (w/o $\delta_V$) | 72.6% | 45.3% | 29.0% | Belief/Actions提升大 |
| 随机干预 (Rnd-$\Delta$) | 62.1% | 39.2% | 25.4% | 随机方向几乎无效 |
| 反向干预 ($-\alpha\Delta$) | 50.6% | 20.6% | 10.1% | 性能暴跌，验证方向正确 |
| 完整 (+$\alpha\Delta$) | 74.5% | 45.3% | 29.7% | 两种干预互补叠加 |

### 关键发现

- 视觉注意力增强对Goal任务效果最显著（+11.7%），因为目标推理更依赖视觉线索
- ToM推理干预对Belief和Actions任务至关重要，因为这些任务需要深层认知推理
- 两种干预方向正交互补，同时施加效果优于单独使用
- PGD攻击比随机噪声提供更精确的干预方向：PGD干预后Goal从29.1%恢复到74.5%，随机噪声干预后仅从47.0%恢复到70.4%
- VisionToM对开放式生成任务同样有效：LLaVA-Next-Video的True∧Info从8.5%提升到27.2%
- Qwen2.5-VL在Goal上的+VisionToM结果（88.9%）已接近人类基线（88%）

## 亮点与洞察

- 可解释性分析的深刻发现：视觉注意力跨任务一致但ToM推理表征任务内聚合、任务间分化，这一洞察为精确干预奠定了理论基础
- 对抗攻击作为探测工具的巧妙使用——PGD攻击不是用来攻击模型，而是用来发现视觉注意力的脆弱方向，反向修正即可增强
- 聚类+专用编码器的ToM推理纠偏策略考虑了推理失败的多样性，比简单平均方向更精细
- 整个方法轻量级且backbone冻结——探针是线性分类器，编码器是两层MLP，一次性校准后干预向量可复用

## 局限与展望

- 当前仅在EgoToM基准上评估，泛化到其他ToM基准（如MMToM-QA、GridToM）的效果未知
- 干预向量在校准集上一次性计算，对于分布外的视频场景可能需要重新校准
- 聚类数量的自动确定（Silhouette + Elbow + CH Index）虽然合理，但对小样本聚类可能不稳定
- Belief和Action任务上仍与人类基线有较大差距（45.3% vs 72%，29.7% vs 78%），表明注意力干预仅是改善手段之一

## 相关工作与启发

- **vs GridToM**: GridToM从线性探针的系数向量推导干预方向，是二分类的简单场景；VisionToM引入聚类+编码器处理多类负样本的异质性，更细粒度
- **vs ICT (CVPR'25)**: ICT用随机噪声引导视觉注意力；VisionToM用PGD对抗样本，提供更精确的方向估计
- **启发**：视觉注意力干预的思路可迁移到其他需要增强VLM视觉推理的任务（如视觉常识推理、因果推断），核心模式是"探测→找方向→干预"

## 评分

- 新颖性: ⭐⭐⭐⭐ 将可解释性探测与干预结合用于ToM增强，视角新颖
- 实验充分度: ⭐⭐⭐⭐ 两个模型、三个任务、消融全面，但仅一个基准数据集  
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，可视化（PCA、KDE）有助理解
- 价值: ⭐⭐⭐⭐ 提供了增强MLLM认知推理的可解释方法，但实际应用场景有限

<!-- RELATED:START -->

## 相关论文

- [From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models](../../ICML2025/multimodal_vlm/from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)
- [GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

<!-- RELATED:END -->
