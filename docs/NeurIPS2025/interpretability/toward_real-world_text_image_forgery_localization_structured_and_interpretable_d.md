---
title: >-
  [论文解读] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis
description: >-
  [NeurIPS 2025][文本图像篡改] 提出基于傅里叶级数的篡改合成框架 FSTS，通过从67名人类参与者收集的16750个真实篡改实例中建模"不可见分布"（篡改操作参数的高维分布），生成更贴近真实世界的合成训练数据，显著提升文本图像篡改定位模型的泛化能力。
tags:
  - NeurIPS 2025
  - 文本图像篡改
  - 合成数据
  - 不可见分布建模
  - 傅里叶级数
  - 篡改参数
---

# Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2511.12658](https://arxiv.org/abs/2511.12658)  
**代码**: [Project Page](https://github.com/)（论文提到有Project Page发布数据集）  
**领域**: 图像伪造检测 / 文本图像篡改定位  
**关键词**: 文本图像篡改, 合成数据, 不可见分布建模, 傅里叶级数, 篡改参数

## 一句话总结

提出基于傅里叶级数的篡改合成框架 FSTS，通过从67名人类参与者收集的16750个真实篡改实例中建模"不可见分布"（篡改操作参数的高维分布），生成更贴近真实世界的合成训练数据，显著提升文本图像篡改定位模型的泛化能力。

## 研究背景与动机

文本图像（如文件、发票、新闻截图等）包含大量敏感信息，容易成为伪造目标。现有文本图像篡改定位（T-IFL）方法依赖大规模高质量标注数据，但真实世界篡改数据集规模受限（如 FindIt 仅240张，STFD 仅4094张），难以满足深度学习的数据需求。

现有的合成方法（如 DocTamper）尝试自动生成篡改数据，但它们主要关注**可见分布**——场景多样性、数据规模、语言多样性等表面属性。然而，真实世界的篡改涉及复杂的**不可见分布**：伪造者会根据场景选择不同篡改类型，执行一系列主处理（区域选择、文本插入、几何变换）和后处理（模糊、滤波、颜色调整、JPEG压缩等）。这些篡改参数的高维向量对人眼不可见，但对篡改痕迹的多样性和检测模型的泛化能力至关重要。

核心挑战在于：(1) 如何有效收集真实篡改参数？多数篡改图像只保留最终结果，无法恢复操作历史。(2) 收集多少数据才能充分建模分布？穷举收集不现实，需要从有限样本中泛化。

## 方法详解

### 整体框架

FSTS 分三步走：收集真实篡改参数 → 层次化建模参数分布 → 从分布中采样生成合成数据。核心思想受傅里叶级数启发——就像复杂波形可分解为基函数的加权和，复杂的篡改行为可分解为基本操作-参数配置的加权组合。

### 关键设计

1. **结构化篡改参数收集管线**: 招募67名专家和志愿者，使用 Photoshop 对不同场景（照片、截图、扫描件）执行5种篡改类型（copy-move、splicing、removal、insertion、replacement），共产生16750个篡改实例。通过视频录制、PSD文件、操作日志等多格式记录自动捕获每次编辑的参数。这解决了"如何收集 $t$"的问题——不是逆向分析篡改图像，而是在篡改过程中直接记录。

2. **个体级分布建模**: 分析发现个体篡改者倾向于重复使用类似的参数配置（如篡改者1在67%的替换样本中使用 Content-Aware Fill，41.4%使用高斯模糊）。因此，每个个体 $i$ 的篡改分布可建模为 $K$ 种篡改类型 $\phi_k$ 的加权组合：

$$P_S^{(i)}(t) = \sum_{k=1}^{K} a_k^{(i)} \phi_k(t_k^{(i)})$$

其中 $a_k^{(i)}$ 是篡改类型 $\phi_k$ 的频率权重，$t_k^{(i)}$ 是代表性操作-参数配置（选取超过2%使用率阈值的最频繁配置）。随着样本数增大，每种类型的统计特征趋于稳定，因此可用单一代表性配置近似。

3. **群体级分布建模**: 观察到不同篡改者之间存在共享偏好（如61.7%使用 Content-Aware Fill，39.7%使用高斯模糊）。因此，群体级分布通过聚合所有个体分布得到：

$$P_S(t) = \sum_{k=1}^{K} a_k \phi_k(t_k)$$

其中 $a_k = \sum_{i=1}^{I} a_k^{(i)}$，$t_k$ 从所有个体的 $\{t_k^{(i)}\}$ 中选取至少5%个体共享的配置。最终优化目标简化为在基配置 $t_k \approx \hat{t}_k$ 的假设下，对齐合成权重 $\{a_k\}$ 与真实权重 $\{\hat{a}_k\}$。

### 损失函数 / 训练策略

FSTS 本身是数据合成框架，不涉及额外的模型训练损失。下游检测模型使用各自原始默认配置训练——Protocol 1/2 训练50个 epoch，Protocol 3/4 训练25个 epoch。合成图像生成过程为：根据建模分布采样操作-参数配置，对原始图像 $I^o$ 执行对应篡改操作：

$$I^s = \text{Generator}(I^o | \{a_k, t_k, \phi_k\}_{k=1}^K)$$

## 实验关键数据

### 主实验（Protocol 2: 合成训练 → 真实测试）

| 方法 | 训练数据 | 真实数据集平均F1 | 真实数据集平均AUC | 相比DocT-T的F1提升 |
|------|---------|----------------|-----------------|------------------|
| RRU-Net | DocT-T | .199 | .765 | - |
| RRU-Net | FSTS-T | .342 | .864 | +.143 |
| DFCN | DocT-T | .102 | .782 | - |
| DFCN | FSTS-T | .327 | .889 | +.225 |
| MVSS-Net | DocT-T | .168 | .697 | - |
| MVSS-Net | FSTS-T | .386 | .812 | +.218 |
| TruFor | DocT-T | .198 | .785 | - |
| TruFor | FSTS-T | .477 | .912 | +.279 |
| STFL-Net | DocT-T | .205 | .781 | - |
| STFL-Net | FSTS-T | .399 | .892 | +.194 |

所有方法在 FSTS-T 训练后的真实世界测试中均大幅超越 DocT-T 训练。

### 消融实验（Protocol 4: 合成预训练 + 真实微调）

| 配置 | 平均F1 | 平均AUC | 说明 |
|------|--------|---------|------|
| Direct（仅真实数据训练） | .405 (MVSS-Net) | .758 | 基线，在跨数据集上泛化差 |
| DocT-T预训练 + 微调 | .406 | .758 | 几乎无提升，说明DocTamper合成数据分布偏差大 |
| FSTS-T预训练 + 微调 | .434 | .807 | 一致性提升，证实FSTS合成数据的预训练价值 |

### 关键发现

- 在 Protocol 2 中，FSTS-T 训练的模型在真实数据集上的平均 F1 提升超过14%，部分模型超过21%
- FSTS-T 合成数据在跨域泛化上甚至可以超越用真实数据直接训练的模型（Protocol 3 vs Protocol 2）
- DocT-T 预训练往往导致性能下降（负增益），而 FSTS-T 预训练几乎总是正增益
- 5种篡改类型中，群体级参数分布展现出明确的共享偏好模式，验证了层次建模的合理性

## 亮点与洞察

- **从"可见分布"到"不可见分布"**的视角转换非常有洞察力。现有合成方法只关注图像层面的多样性，而忽略了篡改操作本身的参数分布——这一隐藏变量对生成逼真训练数据至关重要
- 傅里叶级数的类比虽非严格数学对应，但提供了直觉上优美的框架：复杂篡改行为 = 基本操作模式的加权叠加
- 数据收集管线的设计有实用价值：通过记录人类的编辑过程（而非逆向工程篡改图像）来获取真实参数

## 局限与展望

- 当前只覆盖5种篡改类型，未涵盖基于深度生成模型（GAN/Diffusion）的新型篡改手段
- 假设 $t_k \approx \hat{t}_k$（合成基配置近似真实基配置）是否成立缺乏理论保证
- 67人的参与者池是否足够代表全球伪造者的行为模式有待验证
- 某些方法在特定数据集上仍表现不佳（如 PSCC-Net 在 AFAC 上），说明合成数据无法完全弥补模型架构的局限

## 相关工作与启发

- 与 DocTamper 的核心区别：DocTamper 使用预定义规则生成篡改，FSTS 从真实人类行为建模参数分布
- 可迁移到其他领域的思路：任何需要合成训练数据的场景都可以考虑"从人类操作日志建模隐藏参数分布"
- 与计算机取证领域的联系：不可见分布建模可视为逆向数字取证的正向建模对偶

## 评分

- **新颖性**: ⭐⭐⭐⭐ 不可见分布建模视角新颖，傅里叶级数类比有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ 4种评估协议，7种基线方法，5个真实数据集，实验设计系统全面
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，从问题定义到方法推导层层递进
- **价值**: ⭐⭐⭐⭐ 对文档安全和数字取证领域有直接应用价值，数据集将公开

<!-- RELATED:START -->

## 相关论文

- [CHiQPM: Calibrated Hierarchical Interpretable Image Classification](chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](../../ACL2025/interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [Why Does It Look There? Structured Explanations for Image Classification](../../CVPR2025/interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)
- [Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)

<!-- RELATED:END -->
