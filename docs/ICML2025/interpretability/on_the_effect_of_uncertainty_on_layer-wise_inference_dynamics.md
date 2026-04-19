---
title: >-
  [论文解读] On the Effect of Uncertainty on Layer-wise Inference Dynamics
description: >-
  [ICML 2025 (Actionable Interpretability Workshop)][uncertainty] 使用 Tuned Lens 系统分析 5 个 LLM 在 11 个数据集上各层的 token 概率演化轨迹，发现确定性和不确定性预测的层间推理动力学高度对齐（信心突变出现在相似的层），表明不确定性并不影响模型的推理动态结构，挑战了通过简单中间层特征检测不确定性的方法可行性。
tags:
  - ICML 2025 (Actionable Interpretability Workshop)
  - uncertainty
  - layer-wise dynamics
  - Tuned Lens
  - interpretability
  - hallucination detection
---

# On the Effect of Uncertainty on Layer-wise Inference Dynamics

**会议**: ICML 2025 (Actionable Interpretability Workshop)  
**arXiv**: [2507.06722](https://arxiv.org/abs/2507.06722)  
**代码**: 无  
**领域**: 可解释性 / LLM  
**关键词**: uncertainty, layer-wise dynamics, Tuned Lens, interpretability, hallucination detection

## 一句话总结

使用 Tuned Lens 系统分析 5 个 LLM 在 11 个数据集上各层的 token 概率演化轨迹，发现确定性和不确定性预测的层间推理动力学高度对齐（信心突变出现在相似的层），表明不确定性并不影响模型的推理动态结构，挑战了通过简单中间层特征检测不确定性的方法可行性。

## 研究背景与动机

**领域现状**：理解 LLM 内部如何表示和处理预测是当前可解释性研究的核心问题之一。已有研究表明模型能够在隐藏状态中编码不确定性信息，这催生了一系列基于中间层特征检测幻觉和不确定性的方法（如 probing classifiers、early exit 策略等）。

**现有痛点**：虽然知道模型"在哪编码了不确定性"，但对不确定性"如何影响模型处理过程"了解甚少。现有的不确定性检测方法隐含假设：确定和不确定的预测在层间表现出不同的处理模式（例如不确定预测可能在中间层呈现更多"犹豫"），但这个假设从未被系统验证过。

**核心矛盾**：如果模型对确定和不确定的输入采用相同的推理动力学模式（即相同的层间处理流程），那么试图从中间层统计量来区分两者的所有简单方法都将面临根本性困难。

**本文目标**：系统验证不确定性是否影响 LLM 的层间推理动力学——确定和不确定预测的概率轨迹在层间是否呈现不同模式。

**切入角度**：利用 Tuned Lens（Logit Lens 的改进版）将每层隐藏状态投影到词表空间获取概率分布，追踪最终预测 token 在各层的概率演化轨迹。用"预测正确/错误"作为确定性/不确定性的代理指标。

**核心 idea**：通过比较确定与不确定预测的层间概率轨迹，发现两者高度对齐，说明不确定性不改变推理动态的结构。

## 方法详解

### 整体框架

对 5 个不同规模的 LLM（涵盖不同架构和参数量），在 11 个涵盖不同任务类型的数据集上进行系统分析。具体流程：(1) 对每个 (模型, 数据集) 组合，用 Tuned Lens 计算每层对最终预测 token 的概率；(2) 将所有样本按预测正确/错误分成两组（正确=确定，错误=高认知不确定性）；(3) 计算两组在每层的平均概率并绘制轨迹曲线；(4) 比较两组轨迹的形状、信心突变层位置、收敛模式等特征。

### 关键设计

1. **Tuned Lens 作为层间概率探针**:

    - 功能：将每层隐藏状态映射到词表空间，获得该层对最终 token 的概率估计
    - 核心思路：Logit Lens 直接用最后一层的 unembedding matrix 投影中间层隐藏状态，但中间层的表示分布与最后一层差异大，导致噪声严重。Tuned Lens 为每层学习一个独立的仿射变换 $p_l = \text{softmax}(W_l h_l + b_l)$，其中 $W_l, b_l$ 通过在验证集上拟合最终层输出来训练。这使得中间层的概率估计更准确，轨迹更平滑
    - 设计动机：Tuned Lens 最大程度减少了工具本身引入的噪声，使得观察到的轨迹差异（或缺乏差异）可信地归因于模型本身而非测量工具

2. **错误预测作为认知不确定性代理**:

    - 功能：将样本分为"确定预测"和"不确定预测"两组
    - 核心思路：如果模型给出正确答案，说明模型对相关知识有较高确信度；如果给出错误答案，则反映模型对该问题存在较高的认知不确定性（epistemic uncertainty）。这种二分法虽然粗糙，但覆盖面广，可以在大规模实验中系统使用
    - 设计动机：直接测量 epistemic uncertainty 需要多次采样或集成方法，计算成本高且不适用于所有模型。用预测正误做代理简单直接，且在统计意义上是合理的

3. **跨模型-跨数据集的系统性比较框架**:

    - 功能：验证结论的鲁棒性和普遍性
    - 核心思路：选取 5 个不同规模和架构的模型以及 11 个涵盖 QA、常识推理、知识查询等任务的数据集，形成 55 个 (模型, 数据集) 组合。对每个组合独立分析后，汇总比较模式是否一致
    - 设计动机：如果结论只在特定模型或数据集上成立，则价值有限。55 个组合的覆盖面确保了发现的通用性

### 损失函数 / 训练策略

纯分析性工作，无模型训练。Tuned Lens 本身是预训练好的轻量模块。

## 实验关键数据

### 主实验表格（确定 vs 不确定预测的层间动态特征对比）

| 特征维度 | 确定预测（正确） | 不确定预测（错误） | 差异程度 |
|---------|----------------|-------------------|---------|
| 信心突变层位置 | 中间偏后层 | 中间偏后层 | 极小，几乎对齐 |
| 概率轨迹形状 | 先平稳后突升 | 先平稳后突升 | 形状一致 |
| 最终层置信度 | 高（~0.7-0.9） | 较低（~0.3-0.5） | 预期内差异 |
| 突变幅度 | 大幅跃升 | 大幅跃升 | 幅度相似 |
| 突变后曲线斜率 | 趋于平稳 | 趋于平稳 | 收敛模式一致 |

### 消融表格（跨不同模型和数据集的鲁棒性）

| 分析维度 | 组合数量 | 轨迹对齐观察 | 例外情况 |
|---------|---------|------------|---------|
| 5 个模型 | 全部 | 趋势一致，轨迹对齐 | 更强模型有微弱差异化迹象 |
| 11 个数据集 | 全部 | 跨任务稳健 | 无显著例外 |
| 模型能力分级 | 高能力 vs 低能力 | 高能力模型有轻微差异 | 差异不显著 |
| 不同层区间 | 早/中/晚期层 | 早期层几乎无信息 | 信息集中在中后期 |

### 关键发现

- **确定和不确定预测的层间概率轨迹高度对齐**：两组样本经历极为相似的"先平稳—后突变—再平稳"过程，信心突变发生在相似的层位置
- **不确定性不影响"推理管线"的结构**：模型并没有为不确定的输入采用不同的处理路径，只是最终输出的置信度不同——这是一个重要的否定性发现
- **更有能力的模型可能学会差异化处理**：实验中观察到更强模型有微弱的轨迹差异迹象，暗示模型能力提升可能伴随着对不确定性更精细的处理，但此趋势不显著
- **直接挑战简单的层间不确定性检测方法**：如果两种预测的动态模式相同，则仅基于层间轨迹统计量的检测器可能接近"随机猜测"

## 亮点与洞察

- **重要的"否定结果"**：证明一件事"不行"有时比证明"行"更有价值。本文系统性地表明简单的层间检测思路面临根本困难，这能帮助社区避免在死胡同中浪费精力
- **可解释性工具研究不确定性**的跨界视角新颖：用 Tuned Lens 这个本来用于理解模型计算过程的工具来探究不确定性处理机制，展示了可解释性工具的非传统用途
- **"更强模型可能差异化处理"的观察**虽不显著但富有启发：如果这个趋势在更大模型上被确认，将意味着不确定性感知是一种涌现能力

## 局限性

- **错误预测作为不确定性代理是粗糙的**：错误可能源于分布偏移而非认知不确定性，正确预测也可能是"猜对"的低置信结果
- **仅使用 Tuned Lens 一种分析工具**：其他探针方法（如 probing classifiers、Logit Lens、CKA 分析）可能揭示不同层面的信息
- **Workshop paper 规模受限**：详细的统计检验、效果量分析以及对更大规模模型（70B+）的验证缺失
- **未区分 aleatoric 和 epistemic uncertainty**：将所有错误归为 epistemic uncertainty 过于简化

## 相关工作与启发

- **vs Logit Lens 系列工作**：Logit Lens 主要用于可视化模型逐层"思考过程"，本文将其用于比较两类输入的动态差异，是方法论上的延伸
- **vs probing-based 不确定性检测**：probing 方法在特定层训练线性分类器区分确定/不确定，本文从动态视角质疑这类方法的上限——如果动态模式相同，probing 可能只是在利用最终层附近的置信度差异
- **对幻觉检测的启示**：基于中间层的幻觉检测方法可能需要更复杂的特征（非简单统计量），或者需要多层联合特征而非单层特征

## 评分

- 新颖性: ⭐⭐⭐⭐ 可解释性+不确定性的交叉视角，否定性发现本身有价值
- 实验充分度: ⭐⭐⭐ 5 模型×11 数据集覆盖广，但分析手段单一、缺乏统计检验
- 写作质量: ⭐⭐⭐⭐ 简洁清晰，逻辑连贯
- 价值: ⭐⭐⭐⭐ 为不确定性检测方向提供了重要的负面证据，避免社区走弯路

<!-- RELATED:START -->

## 相关论文

- [Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](../../NeurIPS2025/interpretability/improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](../../NeurIPS2025/interpretability/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [Layer by layer, module by module: Choose both for optimal OOD probing of ViT](../../ICLR2026/interpretability/layer_by_layer_module_by_module_choose_both_for_optimal_ood_probing_of_vit.md)
- [Using Certifying Constraint Solvers for Generating Step-wise Explanations](../../AAAI2026/interpretability/using_certifying_constraint_solvers_for_generating_step-wise_explanations.md)
- [Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)

<!-- RELATED:END -->
