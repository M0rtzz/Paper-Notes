---
title: >-
  [论文解读] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning
description: >-
  [ACL 2025][知识注入] 提出四层知识注入框架（记忆→检索→推理→关联），构建 DeepKnowledge 合成测试平台，系统性揭示了知识注入各层级的关键因素：重复学习实现记忆、表达多样性实现检索、显式推理模式实现深度推理和关联，为 LLM 知识更新提供了完整的方法-层级映射。
tags:
  - ACL 2025
  - 知识注入
  - 四层框架
  - 记忆-检索-推理-关联
  - 持续预训练
  - DeepKnowledge
  - 知识类型
---

# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning

**会议**: ACL 2025  
**arXiv**: [2504.00472](https://arxiv.org/abs/2504.00472)  
**代码**: 未公开  
**作者**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  
**机构**: 中国科学院软件研究所, 中国科学院大学, a-m-team  
**领域**: 知识注入 / LLM知识管理  
**关键词**: 知识注入, 四层框架, 记忆-检索-推理-关联, 持续预训练, DeepKnowledge, 知识类型

## 一句话总结

提出四层知识注入框架（记忆→检索→推理→关联），构建 DeepKnowledge 合成测试平台，系统性揭示了知识注入各层级的关键因素：重复学习实现记忆、表达多样性实现检索、显式推理模式实现深度推理和关联，为 LLM 知识更新提供了完整的方法-层级映射。

## 研究背景与动机

**领域现状**：
   - LLM 从海量预训练数据中捕获知识，但静态性导致知识过时
   - 持续预训练（CPT）是更新领域知识的常用策略
   - 现有知识注入研究停留在浅层——主要关注知识**记忆**（文本补全）和**检索**（改写问题回答）

**核心问题**：
   - 知识注入不是二元过程，而是从 0 到 1 的连续过程——现有工作缺乏系统性的层级定义
   - 浅层知识（仅记忆和检索）无法支撑推理任务，导致在需要深度推理的场景中 LLM 表现不佳
   - 不同知识类型（全新 vs 增量 vs 更新）对注入效果的影响尚未系统研究

**研究目标**：建立知识注入层级与注入方法之间的系统映射，指导实践中的高效知识注入

## 方法详解

### 四层知识注入框架

| 层级 | 名称 | 定义 | 能力要求 |
|------|------|------|----------|
| Level 1 | **知识记忆 (Memorization)** | 以原始形式回忆和复述注入知识 | 文本补全 |
| Level 2 | **知识检索 (Retrieval)** | 在不同语义等价表述下正确提取知识 | 改写问答 |
| Level 3 | **知识推理 (Reasoning)** | 将注入知识应用于推理任务 | 多步推理 |
| Level 4 | **知识关联 (Association)** | 联合注入知识和已有知识进行推理 | 跨知识推理 |

### DeepKnowledge 测试平台构建

#### 知识获取

**已有知识过滤**：
- 来源：WikiFactDiff + MQuAKE
- 三重过滤标准：唯一性（主-关系对结果唯一）、非递归性（主语≠宾语）、可链式推理
- 手动选择 16 组关键推理关系
- 3-shot 测试保留模型能正确回忆的事实 → 26,477 条有效知识

**合成知识生成**：
- 用 LLM 生成虚构实体名（如 "FrankTown"）
- 为虚构实体分配与真实知识相同的关系类型
- 生成 109,860 条合成知识

#### 四层测试用例生成

1. **记忆测试**：从训练语料中移除宾语，形成完形填空题
2. **检索测试**：用 LLM 将记忆测试重写为 10 个语义等价问题
3. **推理测试**：定义两种基本推理规则
    - **组合 (Combination)**：多跳知识聚合
    - **比较 (Comparison)**：知识大小比较
    - n 步推理 = 采样 n 个规则 + 填充知识 + GPT-4 翻译为自然语言问题
4. **关联测试**：类似推理测试，但问题必须同时包含新注入知识和已有知识

### 知识类型

| 类型 | 定义 | 示例 |
|------|------|------|
| **Novel** | 关于新实体的全新信息 | 新提出的科学理论 |
| **Incremental** | 对已有实体的补充信息 | 已知作者的新著作 |
| **Updated** | 替换已有实体的过时信息 | 球队换了新教练 |

### 注入场景

| 场景 | 描述 |
|------|------|
| Duplicate | 同一知识重复多次（无修改） |
| Vanilla Paraphrase | LLM 改写知识表述 |
| Style-enhanced Paraphrase | 带风格变化的改写 |
| Single-step Implicit Reasoning | 改写知识 + 单步推理问答 |
| Single-step Explicit Reasoning | 改写知识 + 单步推理问题 + 详细推理过程 + 答案 |

所有场景均保证每条知识注入 **20 次**，消除数据量差异影响。

### 训练设置

- **模型**：LLaMA 3-8B
- **方法**：持续预训练（CPT），避免 SFT 引发幻觉
- **数据配比**：训练数据与通用指令 1:1 混合
- **学习率**：3e-5

## 实验

### 关键发现 1：重复学习 → 记忆

在 Duplicate 场景下，0-shot 记忆分数随重复次数增加稳定提高，约在 **95 分**处趋于饱和。但：
- 3-shot 设置下记忆分数显著低于 0-shot → 记忆知识不稳定，易受上下文干扰
- Duplicate 下检索和推理分数极低 → 记忆知识是孤立的，缺乏与其他知识的连接

### 关键发现 2：表达多样性 → 检索

知识检索分数在不同注入场景下的表现：

| 注入场景 | 检索分数趋势 |
|----------|-------------|
| Duplicate | 始终约 20，无提升 |
| Vanilla Paraphrase | 显著提升 |
| Style-enhanced Paraphrase | **进一步大幅提升（最优）** |

- 表达多样性是从记忆到检索的关键桥梁
- 风格增强的改写比普通改写效果更好，说明表达的**异质性**（而非仅仅不同表述）是关键

### 关键发现 3：显式推理模式 → 深度推理 (Table 1)

在 2 步和 3 步推理任务上的注入效果：

| 注入场景 | Novel 2步(3S-CoT) | Novel 3步(3S-CoT) |
|----------|-------------------|-------------------|
| Duplicate | 3.3 | 3.7 |
| Style-enhanced Paraphrase | 31.3 | 24.7 |
| Single-step Implicit Reason | 34.3 | 31.7 |
| Single-step Explicit Reason | **41.0** | **49.3** |

**关键结论**：
- 隐式推理提升了零样本多步推理（28.7→41.7）
- 显式推理在 3-shot CoT 下表现最佳（49.3 vs 31.7）
- 仅用单步显式推理训练即可泛化到多步推理和新实体 ← **最重要发现**

### 关键发现 4：LLM 擅长浅关联，深关联需显式推理 (Table 2)

| 注入场景 | 浅关联2步(3S-CoT) | 深关联3步(3S-CoT) |
|----------|-------------------|-------------------|
| Duplicate | 7.7 | 6.0 |
| Style-enhanced Paraphrase | 41.0 | 33.3 |
| Single-step Explicit Reason | **48.3** | **57.3** |
| 基线（无注入的旧知识） | 64.0 | 55.3 |

- 改写注入即可使浅层关联分数达到约 45（接近基线的 64）
- 但深层关联（3 步）需要显式推理注入才能恢复到基线水平

### 消融：知识类型的影响

| 知识类型 | 推理表现 | 原因分析 |
|----------|---------|----------|
| Novel | 较低 | 新实体缺乏已有推理框架 |
| Updated | **较高** | 实体已有推理框架可复用 |
| Incremental | 中等 | 介于两者之间 |

**洞察**：Updated 知识比 Novel 知识更容易达到推理级别注入，因为模型已有相关实体的推理路径。

### 消融：通用指令配比 (Table 3)

| 训练配比 (知识:指令) | Novel 3步推理(3S-CoT) |
|---------------------|----------------------|
| 2:1 | 6.3 |
| 1:1 | **49.3** |
| 1:2 | 54.7 |

通用指令数据对知识推理至关重要——配比从 2:1 到 1:1，3 步推理分数从 6.3 飙升至 49.3。

### 消融：表达多样性阈值

增加同一知识的改写变体数（2→5），检索分数持续提升至 **4 个变体时趋于饱和**。超过 4 个变体后进一步增加多样性不再有改善 → 存在最优多样性阈值。

### 错误分析

复杂推理任务的主要错误来源：
- **Novel 知识**：50%+ 错误源于**错误的问题分解路径**
- **Updated 知识**：**错误的知识回忆**是主要错误来源（新旧知识冲突导致幻觉）

## 亮点与洞察

1. **首个系统化的四层知识注入框架**：将模糊的"知识更新"分解为记忆→检索→推理→关联四个清晰层级，为后续研究提供了统一的评估标准
2. **方法-层级的精确映射**：
    - 记忆 ← 重复训练
    - 检索 ← 多样化表达
    - 推理 ← 显式推理模式
    - 关联 ← 显式推理 + 新旧知识桥接
3. **单步训练泛化到多步的发现**：仅用单步显式推理训练数据即可使模型在 2-3 步推理上取得显著提升，说明推理能力可以泛化
4. **知识类型的实践指导**：Updated 知识更容易达到深层注入（可利用已有推理路径），Novel 知识需要更多显式推理训练
5. **通用指令数据的关键作用**：1:1 混合通用指令是知识推理的必要条件，纯知识训练会导致推理能力丧失

## 局限性

1. **仅使用 LLaMA3-8B**：实验限于单一模型，不同规模和架构上的结果可能不同
2. **仅探索 CPT 方法**：知识注入方式仅限于持续预训练，未评估 LoRA 微调、knowledge editing、RAG 等替代方法
3. **推理操作类型有限**：仅定义了组合和比较两种原子推理操作，未涵盖归纳、类比、反事实等更丰富的推理类型
4. **合成知识的局限性**：虚构实体知识可能无法完全反映真实世界知识注入的复杂性（如常识推理、隐含知识等）
5. **每条知识固定 20 次注入**：实际场景中知识出现频次差异巨大，固定频次可能掩盖长尾知识的注入难度

## 相关工作

- **知识记忆与检索**：Carlini et al. (2021) 训练数据提取、Physics of LM Part 3.1 (Allen-Zhu & Li) 知识存储与提取、MQuAKE (Zhong et al., 2023) 多跳知识编辑
- **知识注入方法**：持续预训练 (Zhang et al., 2023; Jang et al., 2022)、知识编辑 (Zhang et al., 2024)
- **知识推理**：Physics of LM Part 3.2 (Allen-Zhu & Li, 2023) 知识操作、Grokked Transformers (Wang et al., 2024a) 隐式推理
- **知识与幻觉**：Gekhman et al. (2024) SFT 导致幻觉、WikiFactDiff (Khodja et al., 2024) 时序知识差分
- **知识增强 LLM**：Adapting LLMs via Reading Comprehension (Cheng et al., 2023)

## 评分

⭐⭐⭐⭐ — 框架清晰、实验系统全面、发现有实践指导价值（尤其是显式推理→多步泛化和通用指令配比），但仅用单一模型+CPT方法的实验广度不足，合成知识的生态效度也有限。

<!-- RELATED:START -->

## 相关论文

- [A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)
- [Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)
- [ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)

<!-- RELATED:END -->
