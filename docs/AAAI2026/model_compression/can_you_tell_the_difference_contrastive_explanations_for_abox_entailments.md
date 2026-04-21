---
title: >-
  [论文解读] Can You Tell the Difference? Contrastive Explanations for ABox Entailments
description: >-
  [AAAI 2026][模型压缩][对比解释] 提出对比式ABox解释（Contrastive ABox Explanations）的形式化框架，用于回答"为什么a是C的实例而b不是"的问题，在描述逻辑知识库中同时考虑正向蕴涵和缺失蕴涵，并分析不同描述逻辑和优化准则下的计算复杂度。
tags:
  - AAAI 2026
  - 模型压缩
  - 对比解释
  - 描述逻辑
  - ABox推理
  - 知识表示
  - 可解释性
---

# Can You Tell the Difference? Contrastive Explanations for ABox Entailments

**会议**: AAAI 2026  
**arXiv**: [2511.11281](https://arxiv.org/abs/2511.11281)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 对比解释, 描述逻辑, ABox推理, 知识表示, 可解释性

## 一句话总结

提出对比式ABox解释（Contrastive ABox Explanations）的形式化框架，用于回答"为什么a是C的实例而b不是"的问题，在描述逻辑知识库中同时考虑正向蕴涵和缺失蕴涵，并分析不同描述逻辑和优化准则下的计算复杂度。

## 研究背景与动机

知识表示系统的一个核心优势是其透明性和可解释性。在基于描述逻辑（Description Logic, DL）的本体中，所有推理都基于本体和数据中的显式声明。然而随着本体复杂度增长，推理结果往往不易理解。

现有工作分别解决两类问题：
- **为什么蕴涵**（Why）：通过 justification（子集最小的蕴涵集）解释
- **为什么不蕴涵**（Why Not）：通过 abduction（确定缺失信息）解释

但在实际查询中，用户更常提出**对比性问题**："为什么Alice被面试了而Bob没有？"——这要求同时考虑正向和缺失蕴涵，聚焦于两个个体间的相关共性和差异。

**动机示例**：招聘场景中，Alice被面试因为她在期刊发表论文且领导研究组，但分别解释"为什么Alice被面试"和"为什么Bob没有"时可能关注不同的资格原因（资助 vs 领导组），而对比解释能精准定位："Alice的论文发表在期刊而Bob的不是，且只有Alice有资助"。

## 方法详解

### 整体框架

形式化定义**对比ABox解释问题（CP）**为元组 $P = \langle \mathcal{K}, C, a, b \rangle$，其中：
- $\mathcal{K}$ 是知识库（TBox + ABox）
- $C$ 是目标概念
- $a$ 是**事实个体**（fact）：$\mathcal{K} \models C(a)$
- $b$ 是**反事实个体**（foil）：$\mathcal{K} \not\models C(b)$

对比解释（CE）是五元组 $\langle q_{com}(\vec{x}), q_{diff}(\vec{x}), \vec{c}, \vec{d}, \mathcal{C} \rangle$：

| 组件 | 含义 | 作用 |
|------|------|------|
| $q_{com}(\vec{x})$ | 共性模式 | a和b共有的ABox断言模式 |
| $q_{diff}(\vec{x})$ | 差异模式 | 仅a满足、b缺失的断言模式 |
| $\vec{c}$ | 事实证据 | 对a的变量实例化 |
| $\vec{d}$ | 反事实证据 | 对b的变量实例化（可含fresh个体） |
| $\mathcal{C}$ | 冲突集 | 使反事实假设与KB不矛盾需移除的断言 |

### 关键设计

**1. ABox 模式（ABox Pattern）**

使用参数化的ABox模式 $q(\vec{x})$ 抽象掉具体个体名，使得同一模式可分别对fact和foil实例化，从而在结构层面对比两者差异。

**形式化约束条件**（5个核心条件）：
- **C1**: $\mathcal{T}, q(\vec{c}) \models C(a)$ 且 $\mathcal{T}, q(\vec{d}) \models C(b)$（模式在两端都有解释力）
- **C2**: $\mathcal{K} \models q(\vec{c})$（fact证据被KB蕴涵）
- **C3**: $\mathcal{K} \models q_{com}(\vec{d})$（foil的共性部分被KB蕴涵）
- **C4**: $q(\vec{c})$ 是满足C1+C2的 $\subseteq$-最小集（避免无关断言）
- **C5**: $\mathcal{C}$ 是使 $\mathcal{T}, (\mathcal{A} \setminus \mathcal{C}) \cup q(\vec{d}) \not\models \bot$ 的 $\subseteq$-最小集（最小冲突）

**2. 句法CE vs 语义CE**

| 类型 | 约束 | 特点 |
|------|------|------|
| 句法CE | $q_{com}(\vec{c}), q_{diff}(\vec{c}), q_{com}(\vec{d}) \subseteq \mathcal{A}$ | 只引用ABox中显式存在的断言 |
| 语义CE | 无上述限制 | 可引用KB蕴涵的隐式信息 |

关键引理：语义CE可通过构造扩展ABox $\mathcal{A}_e$（添加所有蕴涵断言）归约为句法CE。

**3. 优化准则**

三种优化方向：
- **差异最小化（diff-min）**：使差异模式尽可能小
- **冲突最小化（conf-min）**：使冲突集尽可能小
- **共性最大化（com-max）**：使共性模式尽可能大

每种准则可按子集关系（$\subseteq$）或基数（$\leq$）衡量。

### 损失函数 / 训练策略

本文是理论工作，核心在于**计算复杂度分析**而非训练：

**复杂度总结表：不同优化准则下验证CE最优性的计算复杂度**

| 优化准则 | fresh个体 | $\mathcal{EL}_\bot$ | $\mathcal{ALC}$/$\mathcal{ALCI}$ |
|---------|-----------|-----|------|
| diff-min ($\subseteq$/$\leq$) | — | P / coNP-complete | ExpTime-complete |
| conf-min | 有 | ExpTime-complete | coNExpTime-complete |
| conf-min | 无 | coNP-complete | ExpTime-complete |
| com-max | — | open / coNP-complete | ExpTime-complete |

关键结论：轻量级DL（$\mathcal{EL}_\bot$）下差异最小化可在多项式时间内完成，但在表达力更强的DL下复杂度急剧升高。

## 实验关键数据

### 主实验

**实验设置**：在现实知识库上对生成的对比解释问题进行评估。

**表1：不同知识库上的计算性能**

| 知识库 | ABox规模 | TBox规模 | 平均CE计算时间 |
|-------|---------|---------|--------------|
| 小规模本体 | ~100断言 | ~50 GCI | <1s |
| 中等规模 | ~1000断言 | ~200 GCI | 数秒 |
| 大规模本体 | ~10000断言 | ~500 GCI | 分钟级 |

实验验证了形式化方法在现实规模知识库上的可行性，尤其在轻量级DL下表现良好。

**表2：CE质量对比**

| 方法 | 差异大小 | 共性大小 | 冲突数 | 解释可读性 |
|------|---------|---------|-------|-----------|
| 独立justification+abduction | 大 | 无 | 不考虑 | 低 |
| 差异最小CE | 最小 | 较大 | 可能有 | 高 |
| 共性最大CE | 较小 | 最大 | 可能有 | 最高 |

### 消融实验

- **有无fresh个体**：允许fresh个体使conf-min复杂度从coNP升至ExpTime
- **句法vs语义CE**：语义CE可通过引理5归约，但扩展ABox规模可能导致句法CE求解变慢
- **DL表达力影响**：从$\mathcal{EL}_\bot$到$\mathcal{ALCI}$，复杂度至少升一个量级

### 关键发现

1. 对比解释天然聚焦于相关差异，避免独立解释时的信息冗余
2. 在轻量级DL下diff-min可多项式求解，实用性强
3. 冲突集的引入使框架能处理反事实场景（"如果Bob也领导组的话..."），但增加复杂度
4. 句法CE与语义CE的归约关系为实现提供了统一途径

## 亮点与洞察

- **形式化优雅**：五元组定义精确捕获了对比解释的共性/差异/冲突三个维度
- **复杂度地图完整**：跨5个维度（变体、偏好度量、最优类型、DL、概念类型）的系统分析
- **实际应用场景丰富**：概念学习中解释正负例区别、医疗领域对比患者历史
- **冲突集设计**：允许与KB矛盾的解释同时标注矛盾来源，兼顾表达力与诚实性

## 局限与展望

1. 目前仅实现了一种CE变体的计算方法，其余变体（如com-max）的实现有待开发
2. com-max在$\mathcal{EL}_\bot$下的$\subseteq$-版本复杂度仍未确定（open problem）
3. 实验仅涉及生成的对比问题，缺少真实用户研究来评估解释质量
4. 大规模知识库上的效率仍需优化（分钟级对交互式应用过慢）
5. 未考虑用户偏好——不同用户可能偏好不同粒度的解释

## 相关工作与启发

- **Justification（Schlobach 2003, Horridge 2011）**：解释正向蕴涵的经典方法，本文将其扩展为对比设定
- **ABox Abduction（Koopmann 2021）**：解释缺失蕴涵，本文将其与justification统一
- **对比解释在ML中的应用（Dhurandhar 2018, Miller 2021）**：机器学习中对比/反事实解释的思路，本文将其形式化到DL推理中
- **对可解释AI的启发**：对比式解释范式可扩展到神经符号系统的可解释性设计

## 评分

- 新颖性: ⭐⭐⭐⭐ (首次形式化DL中的对比ABox解释)
- 实验充分度: ⭐⭐⭐ (理论为主，实验验证有限)
- 写作质量: ⭐⭐⭐⭐ (形式化严谨，示例生动)
- 价值: ⭐⭐⭐ (理论贡献扎实，但与模型压缩关联较弱)

<!-- RELATED:START -->

## 相关论文

- [SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism](../../NeurIPS2025/model_compression/shap_meets_tensor_networks_provably_tractable_explanations_with_parallelism.md)
- [TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](../../ICLR2026/model_compression/titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)
- [Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)
- [Uni-LoRA: One Vector is All You Need](../../NeurIPS2025/model_compression/uni-lora_one_vector_is_all_you_need.md)

<!-- RELATED:END -->
