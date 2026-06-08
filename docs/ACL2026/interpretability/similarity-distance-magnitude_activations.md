---
title: >-
  [论文解读] Similarity-Distance-Magnitude Activations
description: >-
  [ACL 2026][可解释性][激活函数] 本文提出 SDM（Similarity-Distance-Magnitude）激活函数作为 softmax 的更鲁棒替代，通过将正确预测的深度匹配（Similarity）、到训练分布的距离（Distance）和决策边界距离（Magnitude）三个认知维度解耦并整…
tags:
  - "ACL 2026"
  - "可解释性"
  - "激活函数"
  - "softmax替代"
  - "选择性分类"
  - "分布外检测"
  - "预测不确定性"
---

# Similarity-Distance-Magnitude Activations

**会议**: ACL 2026  
**arXiv**: [2509.12760](https://arxiv.org/abs/2509.12760)  
**代码**: 无  
**领域**: 可解释性 / 不确定性估计  
**关键词**: 激活函数, softmax替代, 选择性分类, 分布外检测, 预测不确定性

## 一句话总结

本文提出 SDM（Similarity-Distance-Magnitude）激活函数作为 softmax 的更鲁棒替代，通过将正确预测的深度匹配（Similarity）、到训练分布的距离（Distance）和决策边界距离（Magnitude）三个认知维度解耦并整合为新的激活 $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$，并在此基础上构建 SDM 估计器进行选择性分类，在协变量偏移和分布外输入下比现有校准方法更鲁棒。

## 研究背景与动机

**领域现状**：神经网络语言模型的参数不可辨识性（多组参数可产生相同输出分布）使得直接解释参数极为困难。softmax 是最常用的最终输出层激活函数，将 logits 转化为概率分布。现有的不确定性量化方法涵盖贝叶斯（如变分推断）、频率学派（如保形预测）和经验方法（如温度缩放），但 LLM 中高置信错误和幻觉的普遍性表明这些方法存在根本性不足。

**现有痛点**：softmax 仅捕捉 Magnitude（决策边界距离）一个维度的信息——通过 logits 的相对大小反映分类置信度。但它忽略了两个关键的认知信号：(1) 模型预测是否与训练集中的正确预测模式一致（Similarity）；(2) 输入是否在训练分布的覆盖范围内（Distance）。这导致模型在面对分布外输入时仍然输出高置信度预测。

**核心矛盾**：有效的预测不确定性需要分解认知不确定性的来源，但 softmax 的单一温度参数 $\tau$ 无法实现实例级的多维不确定性表征——$\tau$ 是全局超参数，实例间的差异仅由 logits 大小决定。

**本文目标**：设计一个新的激活函数，将 Similarity、Distance、Magnitude 三个维度的认知不确定性信号显式分解并整合，提供更可靠的选择性分类基础。

**切入角度**：利用神经网络作为隐式的基于实例的度量学习器的能力，通过 exemplar adaptor（1-D CNN 适配器）在冻结的预训练 LM 隐藏状态之上构建紧凑的表示空间，从中提取 Similarity 和 Distance 信号。

**核心 idea**：将 softmax 的固定底数 $e$ 替换为数据驱动的底数 $(2+q)$（依赖 Similarity），将固定温度 $\tau$ 替换为实例级的 Distance $d$——使激活函数的输出直接编码三个维度的认知不确定性。

## 方法详解

### 整体框架

SDM 系统包含三层：(1) 冻结的预训练 LM 提供隐藏状态 $\mathbf{h}$；(2) exemplar adaptor（1-D CNN + 线性层）将 $\mathbf{h}$ 映射为紧凑表示 $\mathbf{h}'$ 和新 logits $\mathbf{z}'$；(3) SDM 激活层利用 $\mathbf{h}'$ 计算 Similarity $q$ 和 Distance $d$，与 $\mathbf{z}'$ 结合输出校准的概率分布。在此之上，SDM 估计器通过数据驱动的经验 CDF 分区构建高可靠性区域用于选择性分类。

### 关键设计

**1. Similarity（$q$）：用最近邻是否"既同类又预测对"度量可靠性。**

softmax 的高置信无法区分"模型真见过这种样本"和"模型只是外推得很自信"，Similarity 正是为补这一刀而设。在 exemplar adaptor 的表示空间 $\mathbf{h}'$ 中，把训练集按 $L^2$ 距离排序，从最近邻起逐个检查并累计连续满足三条件的匹配数：(a) 训练样本预测与当前实例相同 $\hat{y} = \hat{y}^{tr}_{(i)}$，(b) 该训练样本预测正确 $\hat{y}^{tr}_{(i)} = y^{tr}_{(i)}$，(c) 匹配链不能断裂。结果 $q \in \{0, \ldots, |D_{tr}|\}$，一旦最近邻就破坏条件则 $q=0$，直接给出分布外信号。它与普通 KNN 的区别在于同时调用了真实标签和模型预测——只有当最近的一圈邻居既标签相同又被模型判对，才认为模型在该局部区域具备可靠判别力。

**2. Distance（$d$）：把到训练分布的距离归一化成保守的不确定性。**

Similarity 回答"邻居像不像"，Distance 回答"还在不在分布内"。先算测试点到训练集最近邻的 $L^2$ 距离 $d_{\text{nearest}}$，再用校准集 $D_{ca}$ 各类别的经验 CDF 归一化：$d = \min[1 - \text{eCDF}^{y_1}_{ca}(d_{\text{nearest}}), \ldots, 1 - \text{eCDF}^{y_C}_{ca}(d_{\text{nearest}})]$。这里取所有类别 CDF 的最小值是刻意的保守——只要相对任何一个类别这个距离都算异常大，就触发高不确定性；当 $d_{\text{nearest}}$ 超过标注数据见过的最大距离时 $d=0$，SDM 直接退化为均匀分布，表达彻底的不确定。

**3. SDM 激活与高可靠性区域：把三维信号写进概率，再切出可信子集。**

三个维度最终在激活函数里合流：$\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$，等于把 softmax 的固定底数 $e$ 换成数据驱动的 $(2+q)$、固定温度换成实例级的 $d$，对应损失则改用换底公式 $\log_{(2+q)}$；当 $q=e-2, d=1$ 时精确退化为标准 softmax。在此之上还要自动框出一块可信区域：先算重缩放值 $q' = \min(q, (2+q)^{\text{sdm}(\mathbf{z}')_{\hat{y}}})$，再在 $q' > 0$ 的子集上逐步抬高阈值 $q'_{\min}$，直到所有类别的保形阈值 $\psi_c$ 都达到目标置信水平 $\alpha$（如 0.95）；凡满足 $q' \geq q'_{\min}$ 且 $\text{sdm}(\mathbf{z}')_{\hat{y}} \geq \psi_{\hat{y}}$ 的预测才进入高可靠性区域。渐进收紧带来理论有保证的选择性分类，而一旦找不到有限的 $q'_{\min}$，这个 $\infty$ 本身就是"模型或数据不足以支撑可靠估计"的明确警报。

### 损失函数 / 训练策略

使用 SDM 损失训练 exemplar adaptor（1-D CNN + 线性层），冻结底层 LM 参数。首轮训练以标准 softmax 初始化（$q=e-2, d=1$），后续每轮重新计算 $q$ 和 $d$。停止条件为校准集上最低的类别平衡损失。重复 $J=10$ 次随机划分和参数初始化，选全局最优。CNN 使用 $M=1000$ 个滤波器，每轮训练 200 个 epoch。

## 实验关键数据

### 主实验

**情感分类（Sentiment，同分布）选择性分类表现（$\alpha=0.95$）**

| 模型+估计器 | 类别条件 y=0 | y=1 | 预测条件 $\hat{y}$=0 | $\hat{y}$=1 | 接纳比例 |
|-----------|---------|-----|------------|------------|---------|
| phi3.5 softmax | 0.98 | 0.86 (<α) | 0.88 (<α) | 0.98 | 0.98 |
| phi3.5 tempScaling | 0.99 | 0.91 (<α) | 0.93 (<α) | 0.99 | 0.90 |
| phi3.5+sdm sdmHR | **1.00** | **0.99** | **0.99** | **1.00** | 0.68 |
| Mixtral8x7B softmax | 0.98 | 0.88 (<α) | 0.89 (<α) | 0.98 | 1.00 |
| Mixtral8x7B+sdm sdmHR | **0.99** | **0.98** | **0.99** | **0.98** | 0.74 |

**情感分类 OOD（SentimentOOD，分布外）**

| 模型+估计器 | 类别条件 y=0 | y=1 | 接纳比例 | 说明 |
|-----------|---------|-----|---------|------|
| phi3.5 softmax | 1.00 | 0.54 (<α) | 0.96 | 过度自信，大量错误 |
| phi3.5 APS | 1.00 | 0.59 (<α) | 0.77 | 仍然不达标 |
| phi3.5+sdm sdmHR | **1.00** | **1.00** | **0.01** | 几乎全部拒绝 OOD |

### 消融实验

| 组件 | 效果 | 说明 |
|------|------|------|
| softmax（无适配器） | 类别条件准确率不达标 | 缺乏 Similarity 和 Distance |
| softmax（有适配器） | ID 达标但 OOD 失败 | 有更好表示但无距离感知 |
| softmax($d \cdot \mathbf{z}'$) | 过度保守（ID 接纳率低） | 仅用 Distance 做温度，缺 Similarity |
| sdm$_\alpha$（简单阈值） | 预测条件达标但类别条件不保证 | 缺少高可靠性区域约束 |
| **sdmHR（完整估计器）** | **两维条件均达标** | Similarity+Distance+Magnitude 三维协同 |

### 关键发现

- 在同分布数据上，没有适配器的 softmax/tempScaling/APS/RAPS 估计器普遍出现过度自信，类别条件准确率低于目标 $\alpha=0.95$
- 在分布外数据上差异更加戏剧化——phi3.5+sdm 的 sdmHR 估计器将 SentimentOOD 的接纳率降至约 1%（几乎全部拒绝），而 softmax 仍接纳 96% 的 OOD 数据且 y=1 类准确率仅 0.54
- 当 Alg. 1 返回 $q'_{\min} = \infty$ 时，提供了模型/数据不足以支撑可靠估计的实用指标
- Factcheck 任务上，softmax 和 APS 在协变量偏移的测试集上类别条件准确率严重不达标，sdmHR 则适当收紧接纳范围维持可靠性

## 亮点与洞察

- Similarity 的定义非常巧妙——不仅要求最近邻标签相同，还要求模型对这些最近邻的预测也正确，且必须是连续的。这比传统 KNN 多了一个"模型在该区域是否可靠"的判断维度
- SDM 的数学形式优雅——将 softmax 的底数和温度从固定常数推广为数据驱动的实例级变量，当 $q=e-2, d=1$ 时精确退化为标准 softmax
- 高可靠性区域的概念对多阶段 LLM 管道有直接价值——自动化预测进入高可靠性区域，其余分流到更昂贵的工具或人工审核

## 局限与展望

- exemplar adaptor 需要维护完整的训练集用于 Similarity 和 Distance 计算，大规模数据集上的存储和检索效率是问题
- 仅在二分类任务上验证（情感分析、事实检查），多类别和更复杂的 NLP 任务需进一步测试
- $q$ 的计算需要遍历训练集按距离排序，实时推理延迟需要优化（可能通过近似最近邻搜索）
- 假设 exemplar adaptor 能够有效地在冻结 LM 之上学习判别性表示，对于某些任务这一假设可能不成立

## 相关工作与启发

- **vs Temperature Scaling**: 温度缩放是单参数全局校准，SDM 通过 $q$ 和 $d$ 提供实例级多维校准，在 OOD 场景下差异巨大
- **vs 保形预测（APS/RAPS）**: 保形方法的边际覆盖保证在选择性分类（仅取集合大小=1）时不直接适用，SDM 通过高可靠性区域的特殊构造提供了类别条件覆盖
- **vs VBLL**: 变分贝叶斯最后层在 OOD 上优于 softmax/tempScaling，但仍不如 SDM 在极端 OOD 场景下的鲁棒性
- **vs 基于 exemplar 的方法**: SDM 将 exemplar 匹配从后验解释工具提升为激活函数的核心组成部分

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 softmax 的底数和温度从常数推广为数据驱动变量，三维认知不确定性分解开创性
- 实验充分度: ⭐⭐⭐⭐ 系统的 ID/OOD/远 OOD 对比和多估计器消融，但任务范围偏窄（仅二分类）
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，从 softmax 到 SDM 的推广路径清晰，符号体系一致
- 价值: ⭐⭐⭐⭐ 为 LLM 部署中的不确定性量化提供了理论基础更强的方案，高可靠性区域概念有广泛应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[ACL 2026\] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](../../ICLR2026/interpretability/lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)
- [\[ICML 2026\] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](../../ICML2026/interpretability/disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](../../NeurIPS2025/interpretability/explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)

</div>

<!-- RELATED:END -->
