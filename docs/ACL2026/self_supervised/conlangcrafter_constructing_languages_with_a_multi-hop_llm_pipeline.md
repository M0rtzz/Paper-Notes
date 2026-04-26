---
title: >-
  [论文解读] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
description: >-
  [ACL 2026][自监督学习][构造语言] 本文提出 ConlangCrafter，一个基于 LLM 的多跳管道，将构造语言（conlang）设计分解为音系、语法、词汇三个模块化阶段，通过随机性注入保证类型学多样性、通过自精炼循环保证内部一致性，并提出了一个包含类型学多样性分析和翻译一致性评估的自动评估框架。
tags:
  - ACL 2026
  - 自监督学习
  - 构造语言
  - 多跳推理
  - 类型学多样性
  - 自精炼
  - 元语言推理
---

# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline

**会议**: ACL 2026  
**arXiv**: [2508.06094](https://arxiv.org/abs/2508.06094)  
**代码**: [项目页面](https://conlangcrafter.github.io)  
**领域**: 计算语言学 / 创意生成  
**关键词**: 构造语言, 多跳推理, 类型学多样性, 自精炼, 元语言推理

## 一句话总结

本文提出 ConlangCrafter，一个基于 LLM 的多跳管道，将构造语言（conlang）设计分解为音系、语法、词汇三个模块化阶段，通过随机性注入保证类型学多样性、通过自精炼循环保证内部一致性，并提出了一个包含类型学多样性分析和翻译一致性评估的自动评估框架。

## 研究背景与动机

**领域现状**：构造语言（conlangs）如世界语和精灵语在艺术、哲学和国际交流中扮演重要角色。基础模型已在文本、图像等领域实现了革命性的创意生成。

**现有痛点**：(1) 构造语言的创建极其耗时——设计者可能花费数年甚至数十年才能达到自然语言的范围和复杂性；(2) LLM 在单次提示下难以生成内部一致的复杂语言系统；(3) LLM 倾向于生成缺乏类型学多样性的输出（Hopkins and Renda 2023），产生的语言过于相似；(4) 缺乏评估计算构造语言质量的自动化框架——没有 ground-truth。

**核心矛盾**：LLM 具备元语言推理能力，但直接生成完整语言描述会导致内部矛盾和多样性不足——语言的各层次（音系、语法、词汇）相互依赖，需要分阶段构建。

**本文目标**：(1) 研究 LLM 能否生成内部一致且类型学多样的语言系统；(2) 提出可扩展的自动评估指标；(3) 探索计算构造语言在创意辅助、游戏生成等方面的应用。

**切入角度**：借鉴语言类型学和语言记录实践，将语言描述分为音系→语法→词汇三层，每层通过多步提示构建，利用 RNG 注入类型学多样性，利用自精炼保证一致性。

**核心 idea**：将构造语言生成建模为多跳推理任务——每个语言层次是一个推理步骤，通过维护可动态更新的"语言草图"记忆库来积累和一致化语言知识。

## 方法详解

### 整体框架

两阶段管道：Stage A（语言草图引导）—— 按音系→语法→词汇的顺序，逐层提示 LLM 生成语言描述并存入语言草图 S；Stage B（建构性翻译）—— 给定 S，翻译新文本到构造语言中，过程中可动态扩展词汇和语法规则。核心组件：LLM M（推理模型如 DeepSeek-R1）+ 语言草图 S（自由文本记忆库）+ 可选用户约束 c。

### 关键设计

1. **多跳语言草图生成**:

    - 功能：分阶段构建内部一致的语言描述
    - 核心思路：将语言分为音系、语法（形态-句法）、词汇三层，按依赖关系顺序生成（音系先于语法以提供词形，语法先于词汇）。每层通过多个子步骤提示 LLM，结果存入语言草图 S。这种结构类似于其他复杂推理任务的多跳方法
    - 设计动机：单次提示无法生成足够详细且一致的语言系统——分层分步构建将复杂任务分解为可处理的子问题

2. **随机性注入（Randomness Injection）**:

    - 功能：确保生成的语言在类型学上具有多样性
    - 核心思路：在音系和语法阶段开始时，让 LLM 生成包含 10 个语言类型学特征的检查清单，每个特征有 5 个选项。然后用外部随机数生成器（RNG）为每个特征随机选择一个选项，LLM 据此实例化语言描述
    - 设计动机：LLM 本身倾向于生成类似的输出，将多样性控制交给外部 RNG 可以利用 LLM 的类型学知识同时保证输出多样性

3. **自精炼循环（Self-Refinement）**:

    - 功能：检测并修复语言描述和翻译中的内部矛盾
    - 核心思路：利用"评估比生成容易"的观察——用同一个 LLM 实现批评者（识别错误和歧义）和编辑者（根据错误列表修改）两个角色，迭代执行直到无进一步问题或达到最大迭代次数
    - 设计动机：语言草图中的矛盾会传播到后续阶段，翻译必须遵守构造的语法，因此一致性检查至关重要

### 损失函数 / 训练策略

不涉及模型训练。使用推理时链式思维扩展的大型推理模型（DeepSeek-R1、Gemini 2.5 Flash/Pro）。评估使用 OpenAI o3 作为判断 LLM，避免生成和评估使用同一模型产生偏差。

## 实验关键数据

### 主实验

**类型学多样性评分（Dmean，越高越好）**

| 方法 | Dmean |
|------|-------|
| 自然语言（WALS 数据库，1874 种） | ~0.55 |
| ConlangCrafter (DeepSeek-R1) | 最高 |
| ConlangCrafter (Gemini 2.5 Pro) | 高 |
| ConlangCrafter (Gemini 2.5 Flash) | 高 |
| 单阶段基线 | 低 |

**翻译一致性评分（Nc,t/Nt,t，越高越好）**

| 方法 | 一致性率 |
|------|---------|
| ConlangCrafter (DeepSeek-R1) | 最高 |
| ConlangCrafter (Gemini 2.5 Pro) | 高 |
| 单阶段基线 | 明显更低 |

### 消融实验

| 配置 | 多样性 | 一致性 |
|------|--------|--------|
| 完整 ConlangCrafter | 高 | 高 |
| 去掉随机性注入 | 低（多样性显著下降） | 高 |
| 去掉自精炼 | 高 | 低（一致性显著下降） |
| 单阶段基线 | 低 | 低 |

### 关键发现

- 多跳管道比单阶段方法在类型学多样性和翻译一致性上都显著更好
- 随机性注入是保证多样性的关键——去掉后生成的语言在 t-SNE 可视化中聚集成团
- 自精炼循环对一致性至关重要——没有它翻译中会出现大量语法违反
- 人工专家评估与自动评估中度一致，支持了自动评估框架的有效性
- DeepSeek-R1 在一致性上表现最佳，Gemini 2.5 在多样性上有竞争力

## 亮点与洞察

- "计算构造语言"是一个全新的范式——将 LLM 的"幻觉"转化为创意特性而非缺陷
- 多跳推理+记忆库+自精炼的架构对任何需要构建复杂一致系统的 LLM 任务都有借鉴意义
- 类型学特征检查清单+RNG 的多样性控制策略可迁移到其他需要多样性的生成任务

## 局限与展望

- 语言草图未覆盖语义学、语用学和正字法等更多语言层面
- 自动评估基于 LLM-as-judge，在这种高度专业化任务上仍有局限
- 实验仅使用 10 个测试句子和约 20 种语言，规模相对有限
- 未来可扩展到低资源语言的辅助记录和教育应用

## 相关工作与启发

- **vs 低资源翻译**: 低资源翻译中幻觉是有害的，但构造语言翻译中"幻觉"是必要的创意——目标语言本就不存在
- **vs 过程化世界生成**: ConlangCrafter 可直接用于开放世界游戏中的社会/语言程序化生成
- **vs 链式思维推理**: 多跳管道本质上是一种结构化的思维链，每层对应一个推理步骤

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创了"计算构造语言"这一全新研究范式
- 实验充分度: ⭐⭐⭐⭐ 自动+人工评估，消融实验充分，但样本量有限
- 写作质量: ⭐⭐⭐⭐ 背景动机清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 对创意AI和语言学研究有启发，实际应用前景广泛

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [\[ECCV 2024\] Adaptive Multi-head Contrastive Learning](../../ECCV2024/self_supervised/adaptive_multihead_contrastive_learning.md)
- [\[ACL 2025\] Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](../../ACL2025/self_supervised/llm_back_gen_treebank.md)
- [\[CVPR 2026\] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](../../CVPR2026/self_supervised/teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](../../ICML2025/self_supervised/mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)

<!-- RELATED:END -->
