---
title: >-
  [论文解读] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels
description: >-
  [ACL 2025][细粒度评分] 提出 FRACTAL 方法，将回复级别（response-level）的聚合标签分解为句子级别（sentence-level）的伪标签，利用多实例学习（MIL）和标签比例学习（LLP）技术结合先验信息（文档-句子余弦相似度）训练句子级评分模型，覆盖检索、问答、摘要和数学推理四类任务。
tags:
  - "ACL 2025"
  - "细粒度评分"
  - "多实例学习"
  - "标签比例学习"
  - "伪标签"
  - "LLM评估"
---

# FRACTAL: Fine-Grained Scoring from Aggregate Text Labels

**会议**: ACL 2025  
**arXiv**: [2404.04817](https://arxiv.org/abs/2404.04817)  
**代码**: 无  
**领域**: 其他  
**关键词**: 细粒度评分, 多实例学习, 标签比例学习, 伪标签, LLM评估

## 一句话总结

提出 FRACTAL 方法，将回复级别（response-level）的聚合标签分解为句子级别（sentence-level）的伪标签，利用多实例学习（MIL）和标签比例学习（LLP）技术结合先验信息（文档-句子余弦相似度）训练句子级评分模型，覆盖检索、问答、摘要和数学推理四类任务。

## 研究背景与动机

LLM 的评估和优化反馈通常在回复级别提供（如整条回复的好坏评分），这种粗粒度反馈虽然效率高、成本低，但存在明显局限：

**定位不精确**：回复级标签无法识别回复中哪些句子是高质量的、哪些存在问题（如事实错误、冗余或不相关）。

**可解释性差**：无法为 LLM 微调提供细粒度的优化目标。

**最新研究表明**：句子级标签能为 LLM 优化提供更准确和可解释的反馈（Amplayo et al., 2022; Lightman et al., 2023），但收集细粒度人工反馈通常成本极高，特别是 Side-by-Side (SxS) 偏好评估。

核心问题：**能否从容易获取的回复级粗标签中推断出有用的句子级细标签？**

这本质上是一个弱监督学习问题：已知"袋"（回复）的聚合标签，需要推断"实例"（句子）的标签。

## 方法详解

### 整体框架

FRACTAL 由三个关键组件组成：
1. 损失函数设计：将回复建模为"袋"、句子建模为"实例"，设计兼顾袋标签和先验信息的联合损失。
2. 聚合函数的可微近似：对 MIN/MAX/AVG 等聚合操作提供可微替代。
3. 最大似然伪标签：利用训练好的模型预测生成与袋标签一致的实例级伪标签，再进行第二轮训练。

### 关键设计

1. **任务到 MIL/LLP 的映射**：不同 NLP 任务对应不同的聚合函数：

    - **检索（Retrieval）**：聚合函数为 AVG（句子级相关性的平均反映整体相关性），对应 LLP 问题。
    - **问答（QA）**：聚合函数为 MAX（至少有一个句子包含答案），对应 MIL 问题。
    - **摘要（Summarization）**：聚合函数为 MIN（所有句子都需要被蕴含），对应 MIL 问题。
    - **数学推理**：聚合函数为 MIN（所有步骤都正确才算整体正确），对应 MIL 问题。

2. **带先验的袋损失（PriorsBagLoss）**：在标准 BagLoss 基础上引入两种先验信息：

    - **余弦相似度先验 P1**：衡量句子嵌入与参考文档/查询嵌入之间的相似度，归一化到 [0,1]：$P1(x) = \frac{1}{2}(1 + \frac{\langle x, U \rangle}{\|x\|_2 \|U\|_2})$
    - **相关性先验 P2**：衡量句子对之间的 Pearson 相关系数：$P2(x, z) = \frac{1}{2}(1 + \rho_{xz})$
   
   总损失为袋损失与先验损失的加权组合：$L_{tot} = \lambda L_{totbag} + \lambda_1 L_{totprior1} + \lambda_2 L_{totprior2}$

3. **聚合函数的可微近似**：

    - 对 MIN 使用 TensorFlow 内置的 tf_reduce_min
    - MAX 通过对翻转变量应用 MIN 推导
    - AVG 本身即可微

4. **最大似然伪标签（PsLab）**：利用训练好的模型 M 的预测，为每个袋中的实例生成与袋标签一致的最大似然标签配置。对于 MIN 聚合且袋标签为 0 的情况：首先按 $M(x) > 0.5$ 二值化所有实例；如果所有实例都被标 1，则翻转 $M(z)$ 最小的那个实例为 0（保证至少有一个 0）。然后在生成的伪标签上重新训练模型。

5. **偏好袋损失（PrefBagLoss）**：针对 SxS 偏好标签（成对比较），使用 Bradley-Terry 模型定义偏好损失：$L_{pref}(B_1, B_2, y_{B_1 B_2}) = y_{B_1 B_2} \log \frac{y_{B_2}}{y_{B_1}}$

### 损失函数 / 训练策略

采用 mini-batch 训练：每步采样 q 个袋，计算袋损失和先验损失的加权和，使用标准优化器更新模型权重。对于偏好标签场景，则采样袋对进行训练。损失函数中的权重超参数 $\lambda, \lambda_1, \lambda_2 \in [0, 1]$ 且 $\lambda + \lambda_1 + \lambda_2 = 1$。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | FRACTAL | BagLoss | Supervised | 说明 |
|--------|------|------|---------|---------|------------|------|
| MultiSpanQA | 检索 | AUC-ROC | **0.693** | 0.661 | 0.729 | 缩小与监督上界差距 |
| QA-Feedback | 偏好QA | AUC-ROC | **0.532** | 0.509 | 0.651 | 偏好标签场景 |
| AquaMuSe | 摘要 | AUC-ROC | **0.814** | 0.751 | 0.876 | 最大提升 +6.3% |
| WikiCatSum | 摘要 | AUC-ROC | **0.645** | 0.477 | 0.837 | 提升 +16.8% |
| PRM800K | 数学推理 | AUC-ROC | **0.597** | 0.569 | 0.613 | 接近监督上界 |
| FirA | 检索回归 | MAE↓ | **0.294** | 0.304 | 0.283 | 回归任务 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| BagLoss vs PriorBagLoss | 大多数数据集提升 | 先验信息有效 |
| PriorBagLoss vs PsLab | PsLab 在 4/5 数据集更优 | 伪标签二次训练有效 |
| cos-sim 直接作为评分器 | 远低于训练方法 | 纯先验不足够 |
| NLI 评分器 | 在摘要任务表现良好 | 但需 T5x-11B 模型 |

### 关键发现

- FRACTAL 在全部 6 个数据集中 5 个超过 BagLoss 基线，且在所有数据集上性能介于 BagLoss 和 Supervised（全监督上界）之间。
- 先验信息的贡献：余弦相似度先验在检索和QA任务中更有效，相关性先验在摘要任务中更有效，两者组合在 QA-Feedback 上效果最佳。
- PsLab 伪标签方法在有确定性标签（0/1二值）的任务中表现最好（如 MultiSpanQA、AquaMuSe），在偏好标签或连续值场景中不可用。
- 在 PRM800K 数学推理任务中，FRACTAL 分数非常接近全监督上界（0.597 vs 0.613），说明数学推理中步骤级错误的定位对回复级信号较为敏感。

## 亮点与洞察

- **问题建模优雅**：将不同 NLP 任务统一到 MIL/LLP 框架下，定义清晰，数学推导严谨。
- **先验信息的设计巧妙**：文档-句子余弦相似度和句间相关性都是容易计算、普遍适用的信号，无需额外标注。
- **伪标签策略简单有效**：最大似然配置赋值加一致性检查，避免了复杂的 EM 或正则化技术。
- **端到端微调验证**：不仅评估了句子级评分的准确性，还验证了用推断出的句子级标签微调 LLM 后性能可比拟使用人工标注细粒度标签训练的模型。
- **跨任务统一方法**：同一个框架适用于检索、问答、摘要、数学推理四个差异巨大的任务。

## 局限与展望

1. 聚合函数的选择（MIN/MAX/AVG）依赖人工为每个任务指定，自动选择聚合函数值得研究。
2. 先验信息目前仅使用了余弦相似度和 Pearson 相关性，未来可探索更丰富的先验（如 NLI 分数、句法特征）。
3. PsLab 伪标签方法不适用于偏好标签和连续值标签场景，限制了其通用性。
4. 评估主要在中等规模数据集上进行，大规模 LLM 评估场景下的扩展性有待验证。
5. 伪标签的质量高度依赖第一阶段模型的准确性，误差可能在二次训练中被放大。

## 相关工作与启发

- MIL 的经典工作始于 Dietterich et al. (1997) 的药物活性检测，后扩展到信息检索、医学影像等领域。FRACTAL 将其首次系统性地应用于 NLP 场景的细粒度评估。
- LLP 方面，DLLP (Ardehaly & Culotta, 2017) 的袋损失方法是重要基线。
- 与 Lightman et al. (2023) 的 PRM800K 工作直接关联——后者收集了步骤级标注，本文则研究如何从聚合标签推断步骤级评分。
- 对 RLHF/RLAIF 有重要启示：如果能从容易获取的回复级反馈中推断细粒度信号，可大幅降低人工标注成本。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 MIL/LLP 引入 NLP 细粒度评价新颖，先验设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个数据集、4 类任务、多种基线和变体、端到端微调验证
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但符号较多，部分段落较密集
- 价值: ⭐⭐⭐⭐ 对 LLM 评估和 RLHF 的细粒度反馈有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)

</div>

<!-- RELATED:END -->
