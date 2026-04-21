---
title: >-
  [论文解读] CompKe: Complex Question Answering under Knowledge Editing
description: >-
  [ACL 2025][知识编辑] 提出CompKe基准——包含11,924个复杂问题——用于评估知识编辑方法在涉及**一对多关系、逻辑操作（交集/并集）和条件确认**的复杂推理场景下的表现，揭示现有方法在复杂问答上的显著不足。
tags:
  - ACL 2025
  - 知识编辑
  - 复杂问答
  - 知识图谱
  - 多跳推理
  - LLM评估
---

# CompKe: Complex Question Answering under Knowledge Editing

**会议**: ACL 2025  
**arXiv**: [2506.00829](https://arxiv.org/abs/2506.00829)  
**代码**: [GitHub](https://github.com/kzjkzj666/CompKE)  
**领域**: NLP理解  
**关键词**: 知识编辑, 复杂问答, 知识图谱, 多跳推理, LLM评估

## 一句话总结

提出CompKe基准——包含11,924个复杂问题——用于评估知识编辑方法在涉及**一对多关系、逻辑操作（交集/并集）和条件确认**的复杂推理场景下的表现，揭示现有方法在复杂问答上的显著不足。

## 研究背景与动机

**领域现状**：知识编辑(Knowledge Editing, KE)旨在高效更新LLM的知识而不做全量微调。现有基准如ZsRE、COUNTERFACT主要测试背诵式记忆，MQuAKE引入了多跳问答，但仍然很有限。

**现有痛点**：
   - **线性问题结构**：MQuAKE等基准的问题只有顺序的step-by-step推理链
   - **仅限一对一关系**：事实三元组只涉及一对一关系，不反映现实世界的一对多关系（如"公司的主要股东有哪些"）
   - **编辑操作有限**：只支持替换操作，遗漏了添加和删除

**核心矛盾**：多跳问答不能充分评估编辑后的模型在**需要逻辑组合（交集、并集）和条件过滤**的真实场景中的表现。

**本文目标**：设计一个更贴近现实的复杂问答基准，包含多样的推理结构、一对多关系和增/删/改编辑类型。

**切入角度**：将复杂问题形式化为图结构推理 $Q = (\mathbf{S}, \mathbf{L})$，其中 $\mathbf{S}$ 是实体集合序列，$\mathbf{L}$ 是推理链接（知识映射 + 条件确认 + 逻辑操作）。

**核心 idea**：从一对多关系和逻辑组合的角度，填补知识编辑评估在复杂推理上的空白。

## 方法详解

### 整体框架

从Wikidata提取关系模板和实体三元组 → 构造复杂问题推理结构 → 引入反事实编辑 → 过滤冲突 → 转化为自然语言问题。

### 关键设计

1. **复杂问题的图结构定义**：

    - 复杂问题 $Q = (\mathbf{S}, \mathbf{L})$
    - $\mathbf{S} = \{S_1, S_2, \dots\}$：中间实体集序列，每个 $S_i$ 本身是一个实体集合（支持一对多）
    - $\mathbf{L} = \{L_1, L_2, \dots\}$：推理链接，分为**知识相关链接**和**逻辑链接**
    - 设计动机：突破线性推理链限制，支持更复杂的推理拓扑

2. **推理链接类型**：

    - **知识映射(Knowledge Mapping)**：给定 $S_i$，通过关系 $r$ 找到所有邻接实体 $S_j = \bigcup_{s \in S_i} A_r(s)$
    - **条件确认(Condition Confirmation)**：在 $S_i$ 中筛选满足特定关系约束的实体
    - **交集(Intersection)**：$S_j = \bigcap_{k=1}^n S_k$
    - **并集(Union)**：$S_j = \bigcup_{k=1}^n S_k$

3. **知识编辑的形式化**：

    - 编辑定义为 $e = (s, r, \mathcal{O} \to \mathcal{O}')$，支持一对多的对象集合变更
    - 分解为三种基本操作：添加 $\mathcal{O}_{\text{add}} = \mathcal{O}' \setminus \mathcal{O}$、删除 $\mathcal{O}_{\text{del}} = \mathcal{O} \setminus \mathcal{O}'$、保留 $\mathcal{O}_{\text{ret}} = \mathcal{O} \cap \mathcal{O}'$

4. **数据集构建流程**（6步）：

    - Step 1: 从Wikidata选取37个关系模板
    - Step 2: 按访问频次采样事实三元组，用GPT-3.5过滤模型无法回忆的知识
    - Step 3: 从种子问题中抽象推理结构模板，再用真实事实实例化
    - Step 4: 随机选择知识映射引入反事实编辑
    - Step 5: 检测并过滤冲突编辑
    - Step 6: 用GPT-4o-mini为每个问题生成3个自然语言变体

### 评估指标

- **Aug (Augment Accuracy)**：新增实体被正确加入答案的比例
- **Ret (Retain Accuracy)**：原有实体被正确保留的比例
- **Acc = (Aug + Ret) / 2**

## 实验关键数据

### 主实验

| 模型 | 方法 | 1-edit Acc | 100-edit Acc | 3000-edit Acc |
|------|------|-----------|-------------|--------------|
| Qwen2.5-3B | ROME | 15.26 | 4.60 | 1.21 |
| Qwen2.5-3B | MEMIT | **22.43** | 7.27 | 2.64 |
| Qwen2.5-3B | MeLLo | 3.83 | 3.23 | 1.35 |
| Qwen2.5-7B | MEMIT | **28.56** | **24.46** | 1.97 |
| Qwen2.5-7B | MeLLo | 15.58 | 13.84 | **10.79** |
| LLaMA-3.1-8B | MEMIT | 19.06 | 17.14 | 17.12 |
| LLaMA-3.1-8B | MeLLo | 16.00 | 13.51 | 11.58 |
| GPT-3.5-turbo | MeLLo | **47.05** | **40.60** | **35.60** |
| GPT-4o-mini | PokeMQA | **39.47** | **38.39** | **31.69** |

### 消融/对比实验

- 与MQuAKE-T和MQuAKE-CF-3k对比：CompKe上MeLLo和PokeMQA的准确率显著更低，证明CompKe更具挑战性
- 参数方法(ROME/MEMIT)在batch编辑数 $k \geq 100$ 时急剧崩溃
- 记忆方法(MeLLo/PokeMQA)则缓慢下降

### 关键发现

- **小模型上参数方法优于记忆方法**：Qwen2.5-3B上MEMIT(22.43) >> MeLLo(3.83)，因为小模型指令遵循能力弱，记忆方法依赖的分解推理难以执行
- **大模型上记忆方法更稳健**：GPT-3.5-turbo上MeLLo(47.05)大幅领先
- **参数方法的过拟合问题**：MEMIT在小模型上看似高分，实际是因为模型不加区分地输出新注入的知识（过拟合），Aug虚高
- **MeLLo的遗漏现象**：分解问题时经常跳过逻辑交集步骤，因为原始prompt示例中不包含此类操作
- **批量编辑的灾难性影响**：参数方法在 $k \geq 100$ 时几乎崩溃，输出不连贯

## 亮点与洞察

- **填补了知识编辑评估的重要空白**：首次系统引入一对多关系、逻辑操作和增/删编辑类型
- **揭示了方法-模型大小的交互效应**：参数方法适合小模型，记忆方法需要强推理能力
- **过拟合和遗漏现象**的深入分析为未来方法设计提供了方向
- **数据集设计严谨**：6步构建流程有冲突检测、质量过滤

## 局限与展望

1. **编辑的真实性**：反事实编辑是随机引入的，与真实世界知识变更的分布可能不一致
2. **关系类型受限**：只涉及一对一和一对多关系，未覆盖多对多和多对一
3. **评估方法有限**：只测试了4种KE方法，未覆盖更新的approach（如GRACE、IKE等在CompKe上的表现）
4. **可扩展方向**：可以引入时序编辑链、多轮持续编辑等更复杂场景

## 相关工作与启发

- **CompKe vs MQuAKE**：MQuAKE只支持线性多跳、一对一替换；CompKe覆盖图结构推理、逻辑操作、增删改操作
- **对知识编辑方法的启示**：（1）参数方法需要解决batch编辑的catastrophic forgetting；（2）记忆方法需要更健壮的问题分解机制
- **启发**：prompt examples的覆盖面直接决定分解质量——MeLLo漏掉交集就是因为示例不够

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 复杂推理+KE的结合是新颖且有意义的评估视角
- **实验充分度**: ⭐⭐⭐⭐ — 5个模型×4种方法×4种batch size，加上详细的case analysis
- **写作质量**: ⭐⭐⭐⭐ — 形式化严谨，Example驱动的解释清晰
- **价值**: ⭐⭐⭐⭐⭐ — 对KE社区有重要推动作用，benchmark有长期使用价值

<!-- RELATED:START -->

## 相关论文

- [SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)
- [Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)
- [Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)

<!-- RELATED:END -->
