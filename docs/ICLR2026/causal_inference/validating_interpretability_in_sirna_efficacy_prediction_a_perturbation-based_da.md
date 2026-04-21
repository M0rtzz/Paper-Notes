---
title: >-
  [论文解读] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol
description: >-
  [ICLR 2026][siRNA] 提出一个标准化的扰动式显著性忠实性验证协议用于 siRNA 效能预测，作为"合成前关卡"检验显著性图是否可信；同时提出 BioPrior 生物信息正则化提升解释忠实性，发现 19/20 折instances 通过验证，但跨数据集迁移暴露两种失败模式。
tags:
  - ICLR 2026
  - siRNA
  - 显著性图
  - 忠实性验证
  - 扰动测试
  - 生物信息正则化
---

# Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol

**会议**: ICLR 2026  
**arXiv**: [2602.10152](https://arxiv.org/abs/2602.10152)  
**代码**: https://github.com/shadi97kh/BioPrior  
**领域**: 医学/生物 可解释AI  
**关键词**: siRNA, 显著性图, 忠实性验证, 扰动测试, 生物信息正则化

## 一句话总结
提出一个标准化的扰动式显著性忠实性验证协议用于 siRNA 效能预测，作为"合成前关卡"检验显著性图是否可信；同时提出 BioPrior 生物信息正则化提升解释忠实性，发现 19/20 折instances 通过验证，但跨数据集迁移暴露两种失败模式。

## 研究背景与动机

**领域现状**：siRNA 药物（如 patisiran、givosiran）已获 FDA 批准。深度学习模型用于预测 siRNA 敲低效能，研究者会检查显著性图来推断哪些核苷酸位置"重要"，指导序列编辑。

**现有痛点**：显著性方法（梯度、积分梯度等）在 siRNA 领域被广泛使用但很少被验证。归因方法不保证反映真正的特征重要性，尤其在协议/分布偏移下可能悄然失效。

**核心矛盾**：模型可能在某个数据集上预测准确且显著性看起来合理，但当迁移到不同实验方案（如不同检测方法）时，显著性可能完全不可靠——而这种失败在部署前无法察觉。

**本文目标**：(a) 提供标准化的显著性忠实性测试协议；(b) 发现和分类跨数据集迁移的失败模式；(c) 用生物学先验正则化提升显著性忠实性。

**切入角度**：定义"反事实忠实性"——突变高显著性位置是否比对照引起更大的预测变化？用这种可操作的测试作为部署前的"合成前关卡"。

**核心 idea**：expected-effect 扰动操作符（对每个位置平均 3 种替代碱基的预测变化）+ 核苷酸组成匹配的随机基线对照 + 配对 Wilcoxon 检验 → pass/fail 判定。

## 方法详解

### 整体框架

训练：Conv→BiLSTM→Transformer 混合编码器 + 双向交叉注意力（siRNA↔mRNA）+ MLP 预测头 + BioPrior 可微生物正则化。验证：计算梯度显著性 → 选 top-k 位置 → 计算 expected-effect → 与组成匹配的随机基线比较 → 统计检验。

### 关键设计

1. **反事实忠实性验证协议**:

    - **功能**：测试高显著性位置的模型敏感度是否高于匹配对照
    - **核心思路**：$\Delta_i = \frac{1}{3}\sum_{b \neq x_i} |\hat{y}(\mathbf{X}) - \hat{y}(\mathbf{X}^{i \leftarrow b})|$ 计算每个位置的 expected-effect。选 top-k 位置，比较 $\Delta(T)$ 与核苷酸组成匹配的随机基线 $\Delta_{match}$。pass 条件：p < 0.05 且 $d_z > 0.2$ 且 win rate > 50%
    - **设计动机**：与标准 ISM 的区别：(1) expected-effect 操作符而非单次突变；(2) 组成匹配基线控制核苷酸特异性偏差；(3) 显式 pass/fail 标准；(4) 跨数据集诊断分类

2. **BioPrior 生物信息正则化**:

    - **功能**：将已知 siRNA 设计规则编码为可微惩罚项
    - **核心规则**：热力学不对称性、种子区组成约束、全局 GC 启发式、免疫基序回避、双链稳定性代理。$\mathcal{L}_{bio} = \sum_c \bar{\alpha}_c \mathcal{L}_c$
    - **设计动机**：生物先验使模型学到的特征更符合已知机制，从而提升显著性忠实性

3. **迁移失败模式分类学**:

    - **faithful-but-wrong**：显著性测试通过但预测失败（模型内部一致但学了错误规则）
    - **inverted saliency**：高显著性位置的敏感度反而低于随机（$d_z < 0$）——这是烟幕弹失败

### 损失函数 / 训练策略

$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda(t) \mathcal{L}_{bio} + \lambda_{aux} \mathcal{L}_{aux}$，其中 $\lambda(t)$ 使用 warmup+ramp 调度（8 epoch 后从 0.10 线性增长到 0.30），先学预测特征再逐步引入生物正则化。

## 实验关键数据

### 主实验

| 数据集 | 模型 | AUC | Pearson r | 忠实性 win rate | Cohen's $d_z$ |
|--------|------|-----|-----------|----------------|-------------|
| Huesken (2431 siRNAs) | +BioPrior | ~0.78 | ~0.65 | 85.2% | 0.86 |
| Huesken | Baseline | ~0.77 | ~0.64 | 82% | 0.77 |
| Katoh (702 siRNAs) | +BioPrior | ~0.76 | ~0.58 | 80%+ | 0.82 |
| Mix (581 siRNAs) | +BioPrior | ~0.77 | ~0.62 | 83%+ | 0.79 |

19/20 折-数据集组合通过忠实性测试。

### 消融实验

| 配置 | 忠实性 $d_z$ | AUC 变化 | 说明 |
|------|------------|---------|------|
| +BioPrior（完整） | 0.86 | +0.01 | 忠实性提升 |
| Baseline（无BioPrior） | 0.77 | 基线 | |
| 随机权重 | -0.45~0.03 | N/A | 负控制，确认失败 |
| 打乱标签 | <0.03 | N/A | 负控制 |
| 底部-k（低显著性） | 失败 | N/A | 反向控制 |

### 关键发现
- **跨数据集迁移揭示关键问题**：Katoh（荧光素酶报告基因）数据集与其他三个数据集（mRNA 水平测定）之间迁移失败——模型在一种实验方案上学到的显著性在另一种上可能完全无效
- **两种失败模式**：faithful-but-wrong（预测失败但显著性通过）和 inverted saliency（Taka→Hu 时 $d_z = -1.25$）
- **BioPrior 提升忠实性但预测改善有限**：+0.01 AUC，但忠实性 $d_z$ 从 0.77 提升到 0.86
- **高显著性位置聚集在功能区域**：种子区（5'端）和 3'端——与生物学先验一致

## 亮点与洞察
- **"合成前关卡"概念的实用价值**：在实验室-AI 循环中，显著性验证应该是标准操作流程，类似于统计检验的显著性阈值
- **迁移失败的预警价值**：协议/实验方案偏移可能悄然无效化部署——即使域内性能看起来很好
- **负控制设计严谨**：随机权重、打乱标签、打乱显著性、底部-k 四种负控制都失败，确认测试有区分力

## 局限与展望
- 仅验证了模型敏感度忠实性，不等于生物因果性（需要湿实验验证）
- RNA-FM 嵌入在扰动时保持固定（出于计算考虑），可能引入误差
- BioPrior 的规则是手动编码的，更多规则或数据驱动的先验可能更好
- 4 个数据集规模有限

## 相关工作与启发
- **vs ISM（体外突变扫描）**：ISM 是解释输出，本文协议是统计接受测试——目的不同
- **vs OligoFormer**：共享架构基础，但本文增加了 BioPrior 和忠实性验证
- **与 physics-informed ML 的联系**：BioPrior 类似 PINN 中的物理约束，但生物系统的先验更不确定

## 评分
- 新颖性: ⭐⭐⭐⭐ 组合了已知组件但在 siRNA 领域的应用新颖，失败模式分类学有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 4 数据集、5 折 CV、跨数据集迁移、多种负控制、消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，协议描述详细可复现
- 价值: ⭐⭐⭐⭐ 对生物序列模型的可解释性验证有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](../../ICML2025/causal_inference/classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)
- [LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](../../CVPR2025/causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)

<!-- RELATED:END -->
