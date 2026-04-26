---
title: >-
  [论文解读] ICYM2I: The Illusion of Multimodal Informativeness under Missingness
description: >-
  [ICLR 2026][多模态][多模态缺失] 揭示了多模态学习中被忽视的问题：模态缺失（missingness）导致的分布偏移会使模态价值评估产生严重偏差，提出 ICYM2I 框架通过双重逆概率加权（IPW）纠正训练和评估中的偏差，在 MAR 假设下实现对模态预测效用和信息论价值的无偏估计。
tags:
  - ICLR 2026
  - 多模态
  - 多模态缺失
  - 分布偏移
  - 逆概率加权
  - 信息分解
  - 模态价值评估
---

# ICYM2I: The Illusion of Multimodal Informativeness under Missingness

**会议**: ICLR 2026  
**arXiv**: [2505.16953](https://arxiv.org/abs/2505.16953)  
**代码**: https://github.com/reAIM-Lab/ICYM2I  
**领域**: 多模态VLM / 机器学习理论  
**关键词**: 多模态缺失, 分布偏移, 逆概率加权, 信息分解, 模态价值评估

## 一句话总结

揭示了多模态学习中被忽视的问题：模态缺失（missingness）导致的分布偏移会使模态价值评估产生严重偏差，提出 ICYM2I 框架通过双重逆概率加权（IPW）纠正训练和评估中的偏差，在 MAR 假设下实现对模态预测效用和信息论价值的无偏估计。

## 研究背景与动机

**领域现状**：多模态学习广泛应用于医疗、自动驾驶、推荐系统等场景，核心假设是"多模态>单模态"。实践中常通过消融实验（去掉某个模态看性能下降多少）来评估模态价值。

**现有痛点**：现实中数据采集存在大量缺失——传感器故障、成本限制、隐私约束等导致某些模态在某些样本上不可用。当前做法是直接丢弃不完整样本，在完整子集上训练和评估。但如果缺失不是完全随机的（即 MCAR），丢弃后的数据子集分布与真实分布不同，导致模态价值评估有偏。

**核心矛盾**：缺失机制与模态信号混淆——如果某模态的缺失与标签相关（MAR），则在完整子集中该模态的"表现"会被系统性高估或低估，而这种偏差在当前研究中几乎完全被忽视。

**本文目标**：在存在非随机缺失的条件下，如何无偏地评估一个模态的（a）预测效用（加入该模态后性能提升多少）和（b）信息论价值（该模态携带的唯一/共享/互补信息）？

**切入角度**：借鉴因果推断中的逆概率加权（IPW）方法，将缺失引起的分布偏移视为一种可纠正的 selection bias。

**核心 idea**：用逆概率加权同时纠正训练损失和评估指标，使多模态模型在有缺失的观测数据上也能无偏评估模态价值。

## 方法详解

### 整体框架

ICYM2I 框架包含两个互补工具：
- **ICYM2I-learn**：用 IPW 加权训练和评估多模态/单模态模型，纠正缺失偏差
- **ICYM2I-PID**：用 IPW 纠正 Partial Information Decomposition（PID）估计，分解模态的唯一/共享/互补信息

前提假设：MAR（Missing At Random）+ Positivity（任意协变量组合下完整观测概率 > 0）

### 关键设计

1. **IPW 加权训练（ICYM2I-learn）**:

    - 功能：在有缺失观测数据 $\Omega_{obs}$ 上训练，但让模型学到的是真实分布 $\Omega$ 上的映射
    - 核心思路：对每个完整样本 $(x_1, x_2, y)$，用逆概率权重 $w = \frac{1}{1 - p(m_1, m_2, m_y | C)}$ 重新加权损失函数。$p(m|C)$ 是缺失概率模型（logistic regression），用观测协变量 $C$ 预测
    - 设计动机：在缺失数据中，某些"完整样本"是过度代表的（因为它们更容易被观测到），IPW 通过降低这些过度代表样本的权重来恢复分布平衡

2. **IPW 纠正评估**:

    - 功能：在完整子集上评估模型时也做 IPW 纠正
    - 核心思路：标准评估指标（如 AUC）在 $\Omega_{obs}$ 上计算会有偏，需要用 IPW 调整为 $\Omega$ 上的无偏估计
    - 设计动机：仅纠正训练不够——如果在偏倚的测试集上评估，仍然无法正确判断模态价值

3. **ICYM2I-PID（信息论分解纠正）**:

    - 功能：无偏估计模态间的 Partial Information Decomposition（PID）——包括唯一信息（Unique）、共享信息（Shared）、互补信息（Complementary）
    - 核心思路：将 Bertschinger et al. 的 PID 框架与 IPW 结合。关键创新在于纠正三路互信息 $I(Y:(X_1,X_2))$ 的估计：用 IPW 加权样本重建真实分布下的互信息，再用修正的 Sinkhorn-Knopp 过程约束边缘分布匹配
    - 设计动机：仅看预测性能无法区分模态信息是"唯一的"还是"冗余的"——PID 提供更精细的信息结构分析，但标准 PID 估计也会被缺失偏差污染

### 损失函数 / 训练策略

- 加权交叉熵损失：$l_{\Omega}(x_1,x_2,y) = \frac{1}{1-p(m_1,m_2,m_y|C)} \cdot l_{\Omega_{obs}}(x_1,x_2,y)$
- 缺失概率模型：用观测协变量 $C$ 训练 logistic regression
- PID 优化：用参数化神经网络 + Sinkhorn-Knopp 迭代求解

## 实验关键数据

### 主实验

比特逻辑运算实验（AND/OR/XOR，50% MAR 缺失）：

| 算子 | 方法 | X1 AUC | X2 AUC | Unique1 | Unique2 | Shared | Compl. |
|------|------|--------|--------|---------|---------|--------|--------|
| AND | Oracle | 0.83 | 0.84 | 0.05 | 0.03 | 0.26 | 0.47 |
| AND | Observed | 0.66 | 0.93 | **0.44** | 0.00 | 0.15 | 0.36 |
| AND | **ICYM2I** | **0.83** | **0.85** | **0.03** | **0.03** | **0.27** | **0.45** |
| XOR | Oracle | 0.51 | 0.49 | 0.00 | 0.00 | 0.00 | 0.99 |
| XOR | Observed | 0.52 | **0.80** | **0.34** | 0.07 | -0.07 | 0.62 |
| XOR | **ICYM2I** | **0.53** | **0.49** | **0.00** | **0.00** | **0.01** | **0.96** |

Observed 方法在 XOR 中严重高估 X2 的 AUC（0.80 vs 真实 0.49）和 Unique1（0.34 vs 真实 0.00），ICYM2I 完美纠正。

### 消融实验

训练-评估纠正组合分析（AUC RMSE vs Oracle）：

| 训练方式 | 评估方式 | AUC RMSE ↓ |
|----------|----------|------------|
| Standard | Standard | 高（有偏） |
| IPW | Standard | 中（训练纠正但评估仍偏） |
| Standard | IPW | 中 |
| **IPW** | **IPW** | **最低（双重纠正）** |

### 关键发现

- 缺失偏差的方向取决于缺失机制：对 OR 算子，X1 被高估（因为 X1=1 时 X2 更可能缺失，观测子集中 X1 的预测力被放大）；对 AND 算子则相反
- **XOR 是最极端的 case**：两个模态的唯一信息均为 0（所有信息都是互补的），但不纠正缺失时 Unique1 被估计为 0.34——这会严重误导"X1 单独就有价值"的错误结论
- 训练和评估的纠正都是必要的，缺一不可
- 在真实医疗数据（乳腺癌筛查）上也验证了 ICYM2I 的有效性

## 亮点与洞察

- **视角独特且影响深远**：之前所有多模态工作都隐式假设"完整样本子集代表全集"，ICYM2I 首次形式化了这个假设的脆弱性。这不是一个边缘问题——在医疗、自动驾驶等高风险场景中，缺失往往与关键因素相关，偏差后果严重
- **因果推断工具迁移到多模态**：IPW 是因果推断经典工具，本文巧妙地将其应用于多模态学习的缺失问题，是很好的跨领域方法迁移
- **区分了两种完全不同的模态缺失问题**：(1) 目标环境缺失（传统问题：如何 robust 于部署时的传感器故障）和 (2) 源环境缺失（本文关注：训练数据中的缺失如何偏置模态价值评估）

## 局限与展望

- **MAR 假设可能不成立**：如果缺失依赖于未观测变量（MNAR），IPW 无法纠正。论文在附录中讨论了 MNAR 下的鲁棒性但承认局限
- **缺失概率模型的准确性至关重要**：IPW 权重来自 logistic regression 对缺失概率的估计，如果该模型不准确，纠正也会有偏
- **极端 IPW 权重问题**：当某些样本的观测概率极低时，IPW 权重极大导致高方差。论文未讨论权重截断等稳定化策略
- **实验规模较小**：主要在合成/半合成数据和小规模医疗数据上验证，需要在大规模多模态 benchmark 上验证实用性

## 相关工作与启发

- **vs 标准多模态 robustness 方法**（如 imputation、knowledge distillation）：这些方法关注"目标环境模态缺失时如何维持性能"，而 ICYM2I 关注的是更基础的问题——"源数据缺失如何偏置我们对模态价值的判断"
- **vs PID 分解方法**（Liang et al. 2024a）：PID 分解隐式假设观测分布=真实分布，ICYM2I 证明在缺失下这会产生严重偏差，并提供了纠正方案
- **对实际系统的启发**：在决定"是否值得采集某个昂贵模态"时（如医疗中的活检），不能简单看回顾性数据中的消融实验结论——必须先纠正缺失偏差

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次形式化多模态缺失偏差问题，视角独特且有广泛影响
- 实验充分度: ⭐⭐⭐ 实验较小规模，缺乏大型多模态 benchmark 上的验证
- 写作质量: ⭐⭐⭐⭐⭐ 形式化严谨，动机示例（比特逻辑运算）直观有力
- 价值: ⭐⭐⭐⭐ 对多模态学习的评估实践有重要指导意义，但需要更多实证支持

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](../../CVPR2026/multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[CVPR 2025\] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](../../CVPR2025/multimodal_vlm/counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](../../ICCV2025/multimodal_vlm/mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)

<!-- RELATED:END -->
