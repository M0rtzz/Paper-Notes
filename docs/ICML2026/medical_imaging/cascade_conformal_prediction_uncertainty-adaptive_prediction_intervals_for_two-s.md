---
title: >-
  [论文解读] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support
description: >-
  [ICML2026][医学图像][保形预测] 提出 CASCADE 框架，将两阶段临床决策系统中第一阶段分类器的认知不确定性（通过 Venn-Abers 预测器量化）传播到第二阶段回归预测区间，使高置信患者的预测区间缩窄 38.9%，同时为不确定病例自动扩展安全缓冲，实现自适应覆盖保证。
tags:
  - "ICML2026"
  - "医学图像"
  - "保形预测"
  - "不确定性量化"
  - "两阶段决策"
  - "Venn-Abers校准"
  - "帕金森病"
---

# CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support

**会议**: ICML2026  
**arXiv**: [2605.20468](https://arxiv.org/abs/2605.20468)  
**代码**: https://github.com/rdiazrincon/cascade_conformal_pd  
**领域**: 医学AI / 临床决策支持  
**关键词**: 保形预测, 不确定性量化, 两阶段决策, Venn-Abers校准, 帕金森病

## 一句话总结
提出 CASCADE 框架，将两阶段临床决策系统中第一阶段分类器的认知不确定性（通过 Venn-Abers 预测器量化）传播到第二阶段回归预测区间，使高置信患者的预测区间缩窄 38.9%，同时为不确定病例自动扩展安全缓冲，实现自适应覆盖保证。

## 研究背景与动机

**领域现状**：帕金森病（PD）的药物管理是一个典型的两阶段决策问题——先判断患者是否需要调药（分类），再预测调整多少剂量（回归）。左旋多巴等效日剂量（LEDD）是衡量药物负荷的标准指标，但最优滴定过程仍高度依赖临床经验的试错法。近年来 AI 临床决策支持系统开始采用两阶段架构来辅助这一流程。

**现有痛点**：标准的保形预测（Conformal Prediction）方法在回归阶段独立工作，完全忽略了第一阶段分类决策的不确定性。这意味着：对于一个分类器 99% 确信需要调药的患者 A 和 55% 置信度的边界患者 B，回归模型会给出相同宽度的预测区间。患者 B 的预测在临床上风险极高——过度自信的剂量推荐可能导致左旋多巴诱导的运动障碍（LID）。

**核心矛盾**：两阶段架构在决策边界处存在**信息丢失**问题。一旦患者越过分类阈值，第一阶段的概率模糊性就被完全丢弃，导致下游回归预测的可靠性与上游决策的确定性脱节。标准保形方法假设同方差性，对所有样本施加全局一致的非一致性阈值，无法根据局部认知风险调整区间。

**本文目标**：设计一个能将分类不确定性显式传播到回归区间校准中的保形预测框架，使预测区间在高置信病例中收紧、在模糊病例中扩展，实现临床风险自适应的不确定性量化。

**核心 idea**：用 Venn-Abers 预测器从第一阶段分类器中提取认知不确定性分数，将其映射为第二阶段保形预测的非一致性分数缩放因子，实现跨任务不确定性传递（cross-task uncertainty transfer），无需训练额外的误差预测模型。

## 方法详解

### 整体框架
CASCADE（Calibrated Adaptive Scaling via Conformal And Distributional Estimation）是一个两阶段保形预测框架。输入为患者特征向量 $x \in \mathbb{R}^d$（年龄、临床变量等），输出为 LEDD 变化百分比的自适应预测区间。数据被划分为训练集 $D_{\text{train}}$ 和校准集 $D_{\text{cal}}$（占 20%）。框架包含两个阶段：（1）第一阶段分类器判断是否需要调药，并通过 Venn-Abers 校准提取认知不确定性分数 $u_{\text{VA}}(x)$；（2）第二阶段回归器预测剂量变化幅度，其预测区间根据 $u_{\text{VA}}(x)$ 动态缩放——这就是"级联效应"（cascade effect）。

### 关键设计

1. **Venn-Abers 认知不确定性提取**:

    - 功能：从第一阶段分类器中提取稳定的认知不确定性度量
    - 核心思路：与标准 softmax 概率不同，Venn-Abers 预测器输出一个多概率区间 $[p_0(x), p_1(x)]$，不依赖分布假设。不确定性分数定义为该区间的宽度 $u_{\text{VA}}(x) = p_1(x) - p_0(x)$。宽区间意味着临床决策的高模糊性——系统无法确定患者是否真正需要调药
    - 设计动机：标准概率估计（如 softmax）在非线性模型中通常校准不佳，单一标量无法反映决策的认知不确定性。Venn-Abers 提供了理论严格的无分布不确定性度量，可作为下游可靠性的直接代理

2. **连续级联缩放机制（Continuous CASCADE）**:

    - 功能：将 $u_{\text{VA}}(x)$ 映射为回归预测区间的连续缩放因子
    - 核心思路：定义均值中心缩放函数 $\sigma(x) = 1 + \beta \left( \frac{u_{\text{VA}}(x)}{\bar{u}_{\text{VA}}} - 1 \right)$，其中 $\bar{u}_{\text{VA}}$ 为校准集的平均 VA 不确定性，$\beta \geq 0$ 为灵敏度参数。当 $u_{\text{VA}}(x) \approx \bar{u}_{\text{VA}}$ 时 $\sigma(x) \approx 1$（标准长度）；当不确定性高于均值时区间扩展；低于均值时区间收缩。归一化非一致性分数为 $S_i = |y_i - \hat{f}(x_i)| / \sigma(x_i)$，最终预测区间为 $\hat{C}(x) = [\hat{f}(x) \pm Q_{1-\alpha} \cdot \sigma(x)]$
    - 设计动机：替代离散 Mondrian CP 的分桶策略。Mondrian 将校准集按 $u_{\text{VA}}$ 分位数划分为 $K$ 个层（如 $K=3$），在每层内独立校准，导致有效样本量碎片化（每层仅 $N_{\text{cal}}/K$）。连续缩放使用全部校准样本估计单一分位数 $Q_{1-\alpha}$，消除离散化伪影，同时保持统计功效

3. **灵敏度参数 $\beta$ 的风险-自适应权衡**:

    - 功能：让临床从业者显式控制系统对不确定性的响应强度
    - 核心思路：$\beta = 0$ 时退化为标准保形预测（无自适应），$\beta$ 越大自适应越强。通过消融实验在覆盖率约束下搜索最优值，$\beta = 0.7$ 时级联比（Cascade Ratio）达到 4.23 同时保持 80.1% 的边际覆盖率。$\beta \geq 0.9$ 时覆盖率保证被违反
    - 设计动机：提供一个可解释的"旋钮"，让临床系统在精度和安全性之间做出显式权衡，而非依赖固定的全局保守性

## 实验关键数据

### 主实验

数据来自佛罗里达大学健康中心 631 例帕金森病住院患者十年数据，使用 XGBoost 作为分类器和回归器，评估在真实需要调药的患者子集（$y_i \neq 0$）上的表现。目标覆盖率 $1-\alpha = 80\%$。

| 方法 | 边际覆盖率 | 平均区间长度 | 级联比 CR |
|------|-----------|-------------|----------|
| Naïve | 52.5% | 0.031 | 1.00 |
| Standard CP | 84.0% | 0.113 | 1.00 |
| CV+ | 83.5% | 0.100 | 1.06 |
| J+aB | 60.6% | 0.132 | 0.97 |
| Mondrian (K=3) | 86.5% | 0.118 | 2.02 |
| **Cont. CASCADE (β=0.7)** | **80.1%** | **0.148** | **4.23** |

### 分层分析（按不确定性三分位）

| 不确定性层 | 方法 | 覆盖率 | 区间长度 | 相对变化 |
|-----------|------|--------|---------|---------|
| 低（底 33%） | Standard CP | 81.1% | 0.113 | — |
| 低（底 33%） | CASCADE | 69.7% | 0.069 | **−38.9%** |
| 中 | Standard CP | 86.5% | 0.113 | — |
| 中 | CASCADE | 82.0% | 0.100 | −10.9% |
| 高（顶 33%） | Standard CP | 85.4% | 0.113 | — |
| 高（顶 33%） | CASCADE | 91.7% | 0.292 | **+158.9%** |

### 关键发现
- **级联效应显著**：CASCADE 将低不确定性患者的区间缩窄 38.9%（0.113→0.069），同时将高不确定性患者的区间扩展 158.9%（0.113→0.292），覆盖率从 85.4% 提升至 91.7%
- **统计验证充分**：KS 检验 $D=0.62$（$p<10^{-54}$）证实 CASCADE 与标准 CP 产生了统计上显著不同的区间分布；Spearman 相关性 $\rho=0.999$ 验证区间长度与 VA 不确定性分数单调相关
- **连续优于离散**：当 Mondrian 的分桶数增加到 $K=7$ 时，平均区间长度膨胀到 0.170（比 $K=3$ 增加 44%），而 Continuous CASCADE 在 $K=7$ 评估下仍保持 CR=6.83 且无碎片化惩罚
- **$\beta$ 消融**：$\beta \leq 0.5$ 自适应不足（CR<3.0）；$\beta \in [0.9,1.0]$ 违反覆盖率保证；$\beta=0.7$ 是安全约束下的最大自适应点

## 亮点与洞察
- **跨任务不确定性传递是核心创新**：不需要训练额外的残差回归模型来估计预测难度，直接复用第一阶段分类器的 Venn-Abers 不确定性作为缩放信号，计算开销几乎为零，且理论上更合理——因为分类决策的模糊性本身就是回归可靠性的最直接代理
- **均值中心缩放的设计非常优雅**：$\sigma(x)$ 以群体平均不确定性为枢轴（pivot），使得"标准患者"获得标准区间、困难患者获得更宽区间、简单患者获得更窄区间，保持了全局校准集的统计效率
- **通用两阶段架构的即插即用模块**：任何"先分类再回归"的级联决策系统（如脑深部电刺激参数设置、肉毒杆菌毒素剂量计算）都可直接应用此框架，只需从分类阶段提取 Venn-Abers 分数即可

## 局限与展望
- 当前采用**对称缩放**，在临床上某些场景下过量用药和用药不足的风险并不对等，需要非对称缩放策略
- 评估主要在**真实标签筛选的子集**（$y_i \neq 0$）上进行，未充分考虑第一阶段分类器的误差传播对整体系统的影响
- 仅在**单一疾病（PD）的单中心数据**（631 例）上验证，样本量有限，缺乏多中心、多疾病的泛化性验证
- 缺少**拒绝预测机制**：对于极高不确定性的患者，系统应能主动放弃预测并转交人类专家
- $\beta$ 参数需要在特定数据集上通过消融实验确定，缺乏自动选择的理论指导

## 相关工作与启发
- **保形预测基础**：Vovk et al. (2005) 的保形预测提供无分布覆盖保证；Mondrian CP 通过分层实现组条件有效性；Normalized CP (Lei et al., 2018) 通过局部缩放实现自适应
- **Venn-Abers 预测器**：Vovk & Petej (2012) 提出的多概率校准方法，本文创新性地将其从校准工具转化为不确定性传播的信号源
- **两阶段临床系统**：Diaz-Rincon et al. (2025) 的 PD 两阶段架构是本文的直接前驱，CASCADE 解决了其决策边界处的信息丢失问题
- 启发：保形预测与认知不确定性的结合可推广到更多级联决策场景，如自动驾驶中的"检测→规划"、医学影像中的"分割→诊断"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](../../AAAI2026/medical_imaging/provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_imaging/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ACL 2026\] ReMedi: Reasoner for Medical Clinical Prediction](../../ACL2026/medical_imaging/remedi_reasoner_for_medical_clinical_prediction.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](../../AAAI2026/medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)

</div>

<!-- RELATED:END -->
