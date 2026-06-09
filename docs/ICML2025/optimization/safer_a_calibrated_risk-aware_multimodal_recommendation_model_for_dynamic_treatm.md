---
title: >-
  [论文解读] SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes
description: >-
  [ICML2025][优化/理论][动态治疗策略(DTR)] 提出 SAFER 框架，融合结构化 EHR 与临床笔记的多模态信息，通过 KL 散度度量标签不确定性并结合保形推断控制 FDR，为高风险动态治疗推荐提供统计安全保障。
tags:
  - "ICML2025"
  - "优化/理论"
  - "动态治疗策略(DTR)"
  - "多模态融合"
  - "不确定性量化"
  - "保形推断(Conformal Prediction)"
  - "脓毒症"
  - "EHR"
---

# SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes

**会议**: ICML2025  
**arXiv**: [2506.06649](https://arxiv.org/abs/2506.06649)  
**代码**: [yishanssss/SAFER](https://github.com/yishanssss/SAFER)  
**领域**: 医学图像  
**关键词**: 动态治疗策略(DTR), 多模态融合, 不确定性量化, 保形推断(Conformal Prediction), 脓毒症, EHR

## 一句话总结

提出 SAFER 框架，融合结构化 EHR 与临床笔记的多模态信息，通过 KL 散度度量标签不确定性并结合保形推断控制 FDR，为高风险动态治疗推荐提供统计安全保障。

## 研究背景与动机

- **动态治疗策略 (DTR)** 旨在根据患者不断变化的临床状态做出实时、个性化的治疗决策，是精准医疗的核心问题
- 现有方法存在三个关键瓶颈：
    1. **标签不确定性**：死亡患者的治疗标签可能不代表最优决策（可能是治疗正确但不足以挽救，也可能是治疗错误导致不良结局），现有方法普遍忽视这种标签模糊性
    2. **模态单一**：大多数 DTR 方法仅使用结构化 EHR 数据（生命体征、实验室检查），忽略了临床笔记中蕴含的医生判断和患者病程信息
    3. **缺乏安全保障**：现有方法没有对推荐质量提供理论性的错误率控制，在高风险临床场景中难以获得医生信任

## 方法详解

SAFER 由三个核心模块组成：多模态表示学习 → 风险感知微调 → 保形选择与 FDR 控制。

### 1. 多模态表示学习

**输入**：每位患者 $i$ 的时序序列 $\mathbf{r}_i = \{(\mathbf{e}_i^1, \mathbf{o}_i^1), \ldots, (\mathbf{e}_i^T, \mathbf{o}_i^T)\}$，其中 $\mathbf{e}$ 为结构化 EHR，$\mathbf{o}$ 为临床笔记。

- **编码器**：临床笔记通过 BioClinicalBERT 编码；结构化数据通过归一化 + one-hot 编码
- **模态内时序建模**：对每种模态分别施加带因果掩码的自注意力机制：

$$\mathbf{S}_i^A = \text{Softmax}\left(\frac{(\mathbf{X}_i^A \mathbf{W}_A^Q)(\mathbf{X}_i^A \mathbf{W}_A^K)^\top + \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{X}_i^A \mathbf{W}_A^V + \text{PE}$$

- **跨模态融合**：设计双向交叉注意力机制，让 EHR 与临床笔记相互学习上下文信息，最终拼接静态人口学特征获得统一患者嵌入 $\mathbf{h}_i \in \mathbb{R}^{3d_k}$
- **分类头**：前馈网络将嵌入映射至药物类别分布，使用交叉熵损失训练

### 2. 风险感知微调

核心思想：存活患者的标签可靠，死亡患者的标签不确定。

- **不确定性估计模块 $f_\phi$**：在第一阶段模型收敛后，引入一个仅在存活患者上训练的 MLP 模块，学习更纯净的预测分布
- **不确定性度量**：通过两个模块输出分布的 KL 散度衡量标签不确定性：

$$\kappa_i = D_{\text{KL}}(p_\theta(\mathbf{h}_i) \| p_\phi(\mathbf{h}_i)) = \sum_{l=1}^{L} p_\theta(\hat{y}_i = l | \mathbf{h}_i) \ln \frac{p_\theta(\hat{y}_i = l | \mathbf{h}_i)}{p_\phi(\hat{y}_i = l | \mathbf{h}_i)}$$

- **理论保障 (Theorem 4.1)**：在 $f_\phi$ 满足 Lipschitz 连续条件下，死亡患者的期望 $\kappa$ 严格高于存活患者
- **风险感知损失**：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}(1 - \hat{\kappa}_i)\sum_{l=1}^{L} y_i \log p_\theta(\hat{y}_i = l | h_i) + \gamma \kappa_i^2$$

其中 $(1-\hat{\kappa}_i)$ 降低不确定样本的权重，$\gamma\kappa_i^2$ 正则项惩罚高风险样本的过度自信预测。

### 3. 保形选择与 FDR 控制

- 对校准集和测试集计算不确定性分数 $\hat{\kappa}$，构造保形 p 值
- 通过 Benjamini-Hochberg (BH) 程序控制误发现率 (FDR)：仅推荐 p 值排名前 $k$ 的治疗方案
- **理论保障 (Theorem 5.1)**：在 i.i.d. 和有界不确定性条件下，推荐集合的 FDR $\leq \alpha$（用户指定阈值）

## 实验关键数据

在两个公开脓毒症数据集（MIMIC-III / MIMIC-IV）上评估，治疗空间为 $5 \times 5$ 的液体-血管升压素组合。

| 方法 | MI-AUC (III) | MA-AUC (III) | HR@3 (III) | MRR@3 (III) | ↓Mortality (III) |
|------|-------------|-------------|-----------|------------|-----------------|
| LSTM | 0.9122 | 0.7934 | 0.7481 | 0.8015 | 0.0915 |
| RETAIN | 0.9257 | 0.8219 | 0.8324 | 0.8153 | 0.1994 |
| ACIL | 0.8219 | 0.7012 | 0.8013 | 0.8313 | 0.3212 |
| **SAFER** | **0.9407** | **0.8672** | **0.8517** | **0.9017** | **0.3891** |

| 方法 | MI-AUC (IV) | MA-AUC (IV) | HR@3 (IV) | MRR@3 (IV) | ↓Mortality (IV) |
|------|------------|------------|----------|-----------|----------------|
| LSTM | 0.9213 | 0.8121 | 0.7551 | 0.8066 | 0.1051 |
| RETAIN | 0.9279 | 0.7851 | 0.8017 | 0.8052 | 0.1863 |
| ACIL | 0.8854 | 0.7135 | 0.8319 | 0.8441 | 0.3782 |
| **SAFER** | **0.9356** | **0.8755** | **0.8713** | **0.8698** | **0.4562** |

- SAFER 在所有推荐指标上全面超越 SOTA，反事实死亡率降低最多（↓Mortality 越高表示模型推荐的治疗越能降低死亡率）
- 在 MIMIC-IV 上 MA-AUC 提升了约 5.8%（vs RETAIN），HR@3 提升了约 4.7%

## 亮点与洞察

1. **标签不确定性的系统性建模**：首次在 DTR 中显式建模死亡患者标签的不确定性，通过 KL 散度量化并融入损失函数，思路优雅且有理论支撑
2. **保形推断 + FDR 控制**：将保形预测引入治疗推荐，提供可量化的安全边界，这在高风险医学场景中极具实用价值
3. **真正的多模态融合**：首次将临床笔记与结构化 EHR 同时用于 DTR，双向交叉注意力设计让两种模态互相增强
4. **端到端框架**：多模态学习、不确定性量化与统计推断三位一体，设计完整

## 局限与展望

1. **仅验证脓毒症**：虽然框架是通用的，但实验只在脓毒症场景（MIMIC 数据集）上测试，对其他疾病/治疗场景的泛化性未知
2. **临床笔记质量假设**：依赖 BioClinicalBERT 编码临床笔记，对笔记缺失或质量差的场景鲁棒性有待验证
3. **治疗空间离散化**：将液体和血管升压素剂量离散为 $5 \times 5$ 的空间，可能丢失连续剂量的精细信息
4. **KL 散度的局限**：不确定性度量依赖两个模块输出分布的差异，当两个模块都出错时可能产生虚假的"低不确定性"
5. **保形推断的 i.i.d. 假设**：实际临床数据往往存在分布漂移，i.i.d. 假设在部署中可能不完全满足

## 评分

- 新颖性: ⭐⭐⭐⭐ — 标签不确定性建模 + 保形推断的组合在 DTR 领域首创
- 实验充分度: ⭐⭐⭐⭐ — 两个大规模公开数据集、多个baseline、消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，理论推导严谨
- 价值: ⭐⭐⭐⭐ — 为高风险治疗推荐提供安全保障的思路有重要临床价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)
- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](../../CVPR2026/optimization/bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](../../NeurIPS2025/optimization/dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[ICML 2025\] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
