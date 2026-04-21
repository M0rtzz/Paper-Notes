---
title: >-
  [论文解读] A Topological Rewriting of Tarski's Mereogeometry
description: >-
  [AAAI 2026][Tarski几何] 在 Coq 定理证明器中扩展 λ-MM 库，将基于 Leśniewski 部分学（mereology）的 Tarski 固体几何重写为具备完整拓扑结构的形式化系统，证明部分学类对应正则开集、满足 Kuratowski 内部公理且具有 Hausdorff（T2）性质，从而为定性空间推理提供了统一的部分学-几何-拓扑理论框架。
tags:
  - AAAI 2026
  - Tarski几何
  - 部分学
  - 拓扑空间
  - Coq定理证明
  - 正则开集
  - 定性空间推理
---

# A Topological Rewriting of Tarski's Mereogeometry

**会议**: AAAI 2026  
**arXiv**: [2511.12727](https://arxiv.org/abs/2511.12727)  
**代码**: [λ-MM Coq库](https://www.univ-smb.fr/listic/technologies/logiciels/modeles-decidables-pour-la-specification-d-ontologies/extending-tarskis-mereo-geometry-with-topological-definitions/)  
**领域**: 定性空间推理 / 形式化验证  
**关键词**: Tarski几何, 部分学, 拓扑空间, Coq定理证明, 正则开集, 定性空间推理

## 一句话总结

在 Coq 定理证明器中扩展 λ-MM 库，将基于 Leśniewski 部分学（mereology）的 Tarski 固体几何重写为具备完整拓扑结构的形式化系统，证明部分学类对应正则开集、满足 Kuratowski 内部公理且具有 Hausdorff（T2）性质，从而为定性空间推理提供了统一的部分学-几何-拓扑理论框架。

## 研究背景与动机

**领域现状**：定性空间推理（QSR）中，部分学与拓扑的结合是 RCC8、接触代数等空间理论的基础。这些理论追随 Whitehead 无点几何的思想，用"区域"取代"点"作为基本实体。

**现有痛点**：现有方法面临多重困难：（1）边界的本体论地位模糊（空间对象还是抽象实体？）；（2）可判定性问题悬而未决；（3）推理机制不足导致空间表示实用性受限；（4）将部分-整体关系简单映射到一阶逻辑导致表达力不足。

**核心矛盾**：基于 Goodman 风格部分学和伪拓扑的定性空间模型缺乏真正的欧几里得几何和完整的拓扑空间，因此无法支持高级几何推理。Tarski 的固体几何虽然定义了固体、球、内点等概念，但缺少与现代拓扑学的形式化桥接。

**本文切入角度**：利用已有的 λ-MM Coq 库（基于归纳类型理论实现的 Leśniewski 部分学），在其上构建代数拓扑关系，证明部分学结构本身蕴含拓扑，并将 Tarski 几何形式化为该拓扑的子空间。

**核心 idea**：证明部分学类（mereological classes）等价于正则开集，从而在不引入集合论的前提下，从部分学内部构建出完整的拓扑空间。

## 方法详解

### 整体框架

论文分两大部分推进：
- **第一部分**：在 λ-MM 库中添加 meet/join 算子和拓扑定义（内部、闭包、边界），证明部分学的个体名集合构成拓扑空间 $\langle N, \mathbb{T}_{MM} \rangle$，满足 Kuratowski 内部公理
- **第二部分**：将 Tarski 几何的固体、球、内点等概念定义为上述拓扑的子空间，证明 Tarski 的三条公设，并扩展理论使其具备 T2（Hausdorff）性质

### 关键设计

1. **部分学→拓扑的桥接（正则开集对应）**:

    - 功能：证明 λ-MM 中的部分学类（m-classes）与正则开集之间存在一一对应
    - 核心思路：部分学的个体名（Individual names）被解释为拓扑的开集，部分-整体关系 $pt$ 对应集合的包含关系 $\subseteq$，部分学的 $b\_sum$/$b\_product$/$compl$ 分别对应拓扑的并/交/补。内部算子定义为 $\eta\;P\;(interior\;Q) \triangleq \eta\;Q\;Q \wedge \eta\;P\;Q$，即一个个体的内部就是它自身（因为个体名都是开集）
    - 设计动机：无需引入外部集合论语义，完全在名目系统（nominalist system）内部实现拓扑构造

2. **Kuratowski 公理的 Coq 形式化证明**:

    - 功能：在 Coq 中创建 Topological type class，将 Kuratowski 内部公理的四条性质编码为类型类字段，并提供机器验证的证明
    - 核心思路：四条公理分别是——全空间保持 $int(\mathbb{T}_{MM}) = \mathbb{T}_{MM}$、内包含性 $int(A) \subseteq A$、幂等性 $int(int(A)) = int(A)$、交集保持 $int(A \cap B) = int(A) \cap int(B)$。证明使用 CoqHammer（Vampire 和 Eprover 自动推理器）半自动完成
    - 设计动机：机器验证的证明比手工证明更可靠，且 type class 的泛型性使得更换底层数据不影响结构

3. **Tarski 几何作为拓扑子空间**:

    - 功能：将内点（saturated interior point）重新定义在球（ball）的子基上，证明 Tarski 的三条公设，并证明 T2（Hausdorff）分离公理
    - 核心思路：Tarski 将"点"定义为共心球的类（内涵类），固体定义为球的部分学和。论文证明这些定义在子空间拓扑中与正则开集一致，从而将 Tarski 的 5 条公理简化为 3 条并补充了 T2 性质
    - 设计动机：T2 性质保证点的唯一分离，是许多空间推理算法的前提条件；公理简化提高了系统的实用性

### 形式化系统 / 证明策略

所有证明在 Coq 8.x 中完成，使用 CoqHammer 调用 Vampire 和 Eprover 自动定理证明器进行半自动化证明。λ-MM 库基于归纳构造演算（CIC）并限制为经典逻辑。形式化代码已开源。

## 实验关键数据

### 形式化成果

| 成果 | 内容 | 说明 |
|------|------|------|
| Kuratowski 内部公理 | 4条全部证明 | 机器验证在Coq中 |
| Tarski 公设 | 证明了3条（原5条简化后） | 公理数减少40% |
| T2 (Hausdorff) 性质 | 首次在该框架中证明 | 保证点的唯一分离 |
| 闭包/边界/正则性 | 全部形式化定义+证明 | 完整的拓扑工具箱 |

### 理论贡献对比

| 方面 | RCC8 / 接触代数 | λ-MM (原始) | 本文扩展 |
|------|----------------|-------------|---------|
| 拓扑空间 | 伪拓扑 | 无 | 完整Kuratowski拓扑 |
| 几何 | 无 | Tarski几何(部分) | Tarski几何+T2 |
| 形式化验证 | 无 | Coq部分学 | Coq部分学+拓扑+几何 |
| 边界定义 | 模糊 | 无 | 精确形式化 |
| 可判定性 | 部分片段 | 是(归纳类型) | 继承 |

### 关键发现

- 部分学类与正则开集的等价性意味着部分学"天然"蕴含拓扑结构，无需外部引入
- 通过子空间构造简化 Tarski 的公理系统（5→3），降低了形式化的复杂度
- T2 性质的证明说明该系统具备足够的分离能力，可支持精细的空间推理

## 亮点与洞察

- **部分学即拓扑**的深刻洞察：这不是简单的"给部分学加个拓扑层"，而是证明部分学的代数结构本身就是一个拓扑空间，统一了两个传统上分离的理论
- **形式化验证的价值**：在空间推理领域，手工证明的错误历史悠久，Coq 机器验证提供了无可争议的正确性保证
- **Type class 的泛型设计**：拓扑结构以参数化 type class 实现，更换底层数据（如不同的部分学变体）不需要重写证明

## 局限与展望

- **高度理论化**：论文几乎没有计算实验或应用演示，对实际空间推理系统的影响需要进一步验证
- **可判定性未充分讨论**：虽然归纳类型提供了构造性基础，但扩展后的理论的可判定片段未明确刻画
- **与 RCC8 的关系**：论文声称优于 RCC8 等接触代数方法，但缺少直接的表达力对比或具体的推理任务基准
- **Coq 代码的可用性**：虽然开源，但形式化证明代码的学习门槛很高，限制了推广

## 相关工作与启发

- **vs RCC8**: RCC8 基于接触关系构建空间推理，本文基于部分学构建并证明其包含更多拓扑信息（如边界、闭包），表达力更强但实用工具不如 RCC8 成熟
- **vs Whitehead 无点几何**: 继承了无点几何用区域替代点的思想，但在 Coq 中提供了首个完整的形式化实现
- **vs 传统 Tarski 几何形式化**: 之前的 Tarski 几何 Coq 形式化（如 GeoCoq）基于点几何，本文是首个基于无点部分学的形式化

## 评分

- 新颖性: ⭐⭐⭐⭐ 在部分学和拓扑之间建立形式化等价是原创性贡献
- 实验充分度: ⭐⭐ 纯理论工作，无计算实验或应用评估
- 写作质量: ⭐⭐⭐ 数学严谨但阅读门槛极高，非该领域专家难以理解
- 价值: ⭐⭐⭐ 对定性空间推理的理论基础有重要贡献，但短期实际应用有限

<!-- RELATED:START -->

## 相关论文

- [On Topological Descriptors for Graph Products](../../NeurIPS2025/others/on_topological_descriptors_for_graph_products.md)
- [Unlocking Speech Instruction Data Potential with Query Rewriting](../../ACL2025/others/unlocking_speech_instruction_data_potential_with_query_rewriting.md)
- [GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](../../ACL2025/others/genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)
- [Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)

<!-- RELATED:END -->
