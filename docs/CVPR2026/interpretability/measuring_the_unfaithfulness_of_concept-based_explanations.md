---
title: >-
  [论文解读] Measuring the (Un)Faithfulness of Concept-Based Explanations
description: >-
  [CVPR 2026][可解释性][概念解释] 本文揭示了现有无监督概念解释方法 (U-CBEMs) 的忠实度被高估——原因是使用了过于复杂的代理模型和有缺陷的删除式评估。作者提出 SURF（Surrogate Faithfulness），一个简单的线性代理 + 双空间度量框架…
tags:
  - "CVPR 2026"
  - "可解释性"
  - "概念解释"
  - "忠实度度量"
  - "无监督概念方法"
  - "代理模型"
  - "可解释性评估"
---

# Measuring the (Un)Faithfulness of Concept-Based Explanations

**会议**: CVPR 2026  
**arXiv**: [2504.10833](https://arxiv.org/abs/2504.10833)  
**代码**: 有（论文声明已释放代码）  
**领域**: 可解释AI / 模型解释性  
**关键词**: 概念解释, 忠实度度量, 无监督概念方法, 代理模型, 可解释性评估

## 一句话总结

本文揭示了现有无监督概念解释方法 (U-CBEMs) 的忠实度被高估——原因是使用了过于复杂的代理模型和有缺陷的删除式评估。作者提出 SURF（Surrogate Faithfulness），一个简单的线性代理 + 双空间度量框架，通过"随机概念应该更不忠实"的 sanity check 验证了其正确性，并首次系统地揭示了多个 SOTA U-CBEMs 实际上并不忠实。

## 研究背景与动机

1. **领域现状**：深度视觉模型难以解释。概念解释方法 (CBEMs) 通过将模型中间表示分解为人类可理解的语义概念（如边缘、颜色、物体部件）来提高可解释性。无监督 CBEMs (U-CBEMs) 自动发现概念激活向量 (CAVs) 及其重要性得分，避免了人工标注概念的需求。

2. **现有痛点**：U-CBEMs 的核心评估指标是"忠实度"——解释是否真正反映模型的内部计算。但现有评估存在两个系统性问题：(a) **代理模型过于复杂**——ICE-Eval 先用 NMF 重建 embedding 再过原始模型，C-SHAP-Eval 额外训练了一个 MLP，这些复杂代理让 U-CBEMs "看起来"忠实，但解释本身并未清楚地导向模型输出；(b) **删除式评估不可靠**——通过删除概念再观察性能下降来间接推断忠实度，但删除后的输入可能偏离数据流形，模型在偏离流形处的行为不可预测。

3. **核心矛盾**：可解释性和忠实度之间存在天然矛盾——为了让解释简单易懂（可解释），必然丢失信息（降低忠实度）。过去的评估方法通过允许复杂代理和单一类别度量，人为地让 U-CBEMs 同时"显示"高可解释性和高忠实度，但实际上解释并不能清楚地推导出模型的输出。

4. **本文目标** (a) 统一现有碎片化的忠实度评估框架；(b) 设计满足三个准则（简单代理、包含概念重要性、全输出度量）的忠实度度量；(c) 对现有 U-CBEMs 做首次公平的忠实度基准评测。

5. **切入角度**：作者提出了一个极其简洁的 sanity check——如果把概念替换为随机向量，忠实度应该下降。令人震惊的是，现有的 ICE-Eval 和 C-SHAP-Eval 在这个检查上都失败了。

6. **核心 idea**：用模型本身的最终线性层结构作为代理模板，提出零参数线性代理 SURF，搭配 logit 空间 (MAE) 和概率空间 (EMD) 的双度量，实现对 U-CBEM 忠实度的准确、可靠评估。

## 方法详解

### 整体框架

SURF 的评估流程：对于任何 U-CBEM 生成的解释（包含 CAVs $V_i$ 和概念重要性 $A_i$），SURF 用一个线性代理将概念表示映射到模型输出空间，然后在 logit 和概率两个空间上度量解释输出与真实模型输出的差异。整个过程不引入任何可训练参数，计算量极低（200 FLOPs vs C-SHAP-Eval 的 205M FLOPs）。

### 关键设计

1. **统一的忠实度框架**:

    - 功能：将所有现有的 U-CBEM 忠实度评估方法纳入一个统一视角
    - 核心思路：任何忠实度度量都由三部分组成——度量 $d$（如何比较输出差异）、代理 $s$（如何从解释得到输出）、概念投影 $\mathcal{P}$（如何将 embedding 映射到概念空间）。删除式方法通过在"删除空间"（像素/权重/概念）中移除概念后观察模型退化来评估；代理式方法直接用代理近似 Eq.3 的忠实度积分。
    - 设计动机：每个 U-CBEM 论文都提出自己的忠实度度量，无法跨方法比较。统一框架暴露了各方法的系统性差异。

2. **SURF 代理设计**:

    - 功能：用最简单的方式从 U-CBEM 解释预测模型输出
    - 核心思路：观察模型最终线性层的计算 $y_i = \sum_j \mathbf{h}_j^T \mathbf{f}_{i,j}$，可以分解为 $y_i = \sum_j \mathbf{h}_j^T \mathbf{v}_{i,j} \alpha_{i,j}$，其中 $\mathbf{v}_{i,j}$ 是归一化方向（即 CAV），$\alpha_{i,j}$ 是其范数（即重要性）。SURF 的代理直接模仿这个结构：$\hat{y}_i = \sum_j \sum_k \alpha_{i,k} \mathcal{P}(\mathbf{h}_j; V_i)_k$，用 U-CBEM 发现的 CAV 和重要性替代最终层权重。
    - 设计动机：这个代理是零参数的、非重建式的——不需要先重建 embedding 再过原始模型。它直接检验"解释中的概念和重要性能否通过线性组合预测模型输出"，这恰好是人类解释者需要执行的心理运算。

3. **三大准则 (Desiderata)**:

    - 功能：定义"好的"忠实度度量应满足的条件
    - 核心思路：(1) **代理尽可能简单**——复杂代理只是把解释的复杂性推到了下游，人类解释者仍然不知道解释如何导向预测；(2) **使用解释的所有组件**——特别是概念重要性 $A_i$，如果代理不使用 $A_i$，错误的重要性不会影响忠实度得分；(3) **度量所有输出类别的误差**——只看预测类或 GT 类会忽略其他类别上的大偏差。
    - 设计动机：现有的 ICE-Eval 和 C-SHAP-Eval 全部违反这三个准则——前者基于重建且不使用 $A_i$，后者引入可训练 MLP 且不使用 $A_i$，两者都只度量单个类别。

4. **SURF 双度量**:

    - 功能：在 logit 和概率两个互补空间上全面评估忠实度
    - 核心思路：$\text{SURF}_{\text{MAE}} = \frac{1}{|\mathcal{V}|C} \sum_{\mathbf{x}} \sum_i |y_i - \hat{y}_i|$ 衡量 logit 空间上的绝对误差；$\text{SURF}_{\text{EMD}} = \frac{1}{2|\mathcal{V}|} \sum_{\mathbf{x}} \sum_i |p_i - \hat{p}_i|$ 衡量概率空间上的分布距离。
    - 设计动机：logit 空间不受归一化影响但范围不可控，概率空间归一化了但 softmax 会放大预测类、压缩其他类。用两个度量互相弥补：低 $\text{SURF}_{\text{MAE}}$ 保证 logit 精度，低 $\text{SURF}_{\text{EMD}}$ 保证概率分布准确。

### 损失函数 / 训练策略

SURF 本身不需训练，是纯评估 metric。实验中 U-CBEMs 按各自论文设置运行，每个输出类发现 5 个概念。

## 实验关键数据

### Measure-over-Measure 比较（Sanity Check）

| 设置 | SURF_MAE ↓ | SURF_EMD ↓ | Top-1 ↑ | C-SHAP Top-1 ↑ | ICE Top-1 ↑ |
|------|------------|------------|---------|-----------------|-------------|
| Perfect | 0.00 | 0.000 | 100% | 9.02% | 100% |
| Rand Imp | 2.70 | 0.862 | 97.5% | 9.02% | 100% |
| Full Rand | 3.17 | 0.883 | 1.3% | **97.6%** | 3.3% |

关键发现：**C-SHAP-Eval 在完全随机解释 (Full Rand) 下反而报告了 97.6% 的准确率**，比 Perfect 设置还高！ICE-Eval 在随机重要性 (Rand Imp) 下仍报告 100% 忠实度。只有 SURF 在三个设置中表现正确——Perfect 完美、Rand Imp 下降、Full Rand 最差。

### U-CBEM 基准评测（Object Classification, ResNet-50）

| U-CBEM | SURF_MAE ↓ | SURF_EMD ↓ | Top-1 ↑ | Rank Corr ↑ |
|--------|------------|------------|---------|-------------|
| CDISCO | 3.40 | 0.932 | 0.2% | 0.002 |
| ICE | 3.33 | 0.628 | 98.9% | 0.093 |
| CRAFT | 3.19 | 0.878 | 90.6% | 0.068 |
| C-SHAP | 3.28 | 0.882 | 6.3% | 0.005 |
| MCD | 2.60 | 0.426 | 99.4% | 0.145 |
| HU-MCD | 1.97 | 0.384 | 99.7% | 0.149 |
| SAE | **1.04** | **0.195** | 99.2% | **0.366** |

### 关键发现

- **没有一个现有 U-CBEM 是真正忠实的**——最好的 SAE 的 $\text{SURF}_{\text{EMD}}$ 也达到 0.195，意味着概率分布有显著偏差。多数方法的 $\text{SURF}_{\text{EMD}}$ 在 0.4-0.93 之间，几乎与随机一样差。
- **Top-1 Accuracy 是误导性指标**——ICE 和 MCD 的 Top-1 高达 98-99%，但 SURF_EMD 和 Rank Corr 揭示它们在非预测类上的误差极大。只看 Top-1 会严重高估忠实度。
- **SAE 在所有任务中最忠实**——无论是分类、多属性预测还是年龄回归，SAE 都是最好的，可能因其在完备性准则上的优势。
- **增加概念数不一定提高忠实度**——CDISCO、CRAFT、C-SHAP、ICE 在增加概念数后忠实度几乎不变甚至下降。只有 MCD/HU-MCD 随概念数增加而单调改善，并出现自然的饱和拐点。

## 亮点与洞察

- **"随机概念应该更不忠实"这个 sanity check 极其简洁有力**——用一个任何人都能理解的直觉检验，一击命中了所有先前度量的要害。这类简单而致命的检验在 XAI 领域非常有价值（类似 Adebayo et al. 的 saliency map sanity check）。
- **SURF 的零参数设计是关键创新**——通过观察到最终线性层本身就定义了"完美解释"的形式，避免了任何额外参数或重建操作。200 FLOPs vs 205M FLOPs 的差距不只是效率问题，更是"代理复杂度是否污染忠实度评估"的根本性区别。
- **双度量设计弥补了单一度量的盲区**——Top-1 只关注最大类、Norm L1 只关注 GT 类、SURF_MAE 不区分类别重要性、SURF_EMD 补充概率域评估。

## 局限与展望

- **目前 SURF 只适用于最终线性层**——对中间层的解释无法评估，因为中间层到输出的映射是非线性的。扩展 SURF 到中间层是关键的未来工作。
- **只评估了忠实度，没有联合评估可解释性**——一个忠实度低但高度可解释的方法可能仍有实用价值。理想的框架应该同时量化忠实度-可解释性 tradeoff。
- **实验中 U-CBEMs 每类只发现 5 个概念**——这个设置对某些方法可能不公平。虽然 Fig.2 展示了变化概念数的趋势，但只在一个任务上做了。
- **分类任务为主**——只有一个回归任务（年龄估计），对其他任务类型（如分割、生成）的推广未验证。

## 相关工作与启发

- **vs ICE / ICE-Eval**: ICE 用 NMF 发现概念并重建 embedding 评估忠实度。SURF 揭示 ICE-Eval 不使用概念重要性，因此随机重要性和完美重要性得分一样——根本性缺陷。
- **vs C-SHAP / C-SHAP-Eval**: C-SHAP 用 Shapley 值计算概念重要性并引入 MLP 代理。SURF 揭示 MLP 代理有时在随机概念上反而表现更好——可能是因为 MLP 学到了与概念无关的从 embedding 到输出的映射。
- **vs CRAFT**: CRAFT 用 NMF 递归分解子概念，使用概念空间删除评估。SURF 的代理式方法避免了删除操作的离流形问题。
- **vs SAE (Sparse Autoencoders)**: SAE 在 SURF 下表现最佳，可能成为未来视觉模型解释的主流方向。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示 U-CBEM 忠实度评估的系统性缺陷，SURF 设计优雅有力
- 实验充分度: ⭐⭐⭐⭐ 三个任务、七个 U-CBEMs、measure-over-measure 比较全面，但中间层评估缺失
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑极为严谨，从框架统一到准则提出到 sanity check 到 benchmark 一气呵成
- 价值: ⭐⭐⭐⭐⭐ 对 XAI 社区有重大影响，可能改变 U-CBEM 领域的评估标准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings](inside-out_measuring_generalization_in_vision_transformers_through_inner_working.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](../../NeurIPS2025/interpretability/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](../../ICML2025/interpretability/what_makes_an_ensemble_un_interpretable.md)

</div>

<!-- RELATED:END -->
