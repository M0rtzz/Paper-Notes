---
title: >-
  [论文解读] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems
description: >-
  [ACL 2025][LLM Agent][方言偏差] 提出一个双agent协作框架(方言Agent + 隐私政策Agent)，通过将非标准英语方言翻译为标准美式英语(SAE)并进行迭代验证，在不需要重训练或方言特定微调的前提下，显著降低隐私政策问答中的方言偏差并提升整体性能。
tags:
  - ACL 2025
  - LLM Agent
  - 方言偏差
  - 隐私政策QA
  - 多agent协作
  - 语言多样性
  - 公平性
---

# A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems

**会议**: ACL 2025  
**arXiv**: [2506.02998](https://arxiv.org/abs/2506.02998)  
**代码**: 无  
**领域**: LLM Agent / NLP公平性  
**关键词**: 方言偏差, 隐私政策QA, 多agent协作, 语言多样性, 公平性

## 一句话总结

提出一个双agent协作框架(方言Agent + 隐私政策Agent)，通过将非标准英语方言翻译为标准美式英语(SAE)并进行迭代验证，在不需要重训练或方言特定微调的前提下，显著降低隐私政策问答中的方言偏差并提升整体性能。

## 研究背景与动机

**领域现状**: 隐私政策是告知用户数据收集和使用方式的重要文档，但其复杂的法律语言限制了可及性。隐私政策QA系统旨在提供简洁的查询驱动答案。

**现有痛点**: 现有隐私政策QA系统对语言多样性几乎视而不见，非标准方言(如非裔美式英语AAVE、牙买加英语等)的用户获得的回答质量显著低于标准美式英语用户。边缘化社区在隐私数据收集和滥用中已受到不成比例的伤害。

**核心矛盾**: LLM虽然通用能力强，但在特定方言输入时表现退化——如果所有群体无法平等地提问以保护其信息，这些群体就面临风险。

**本文目标**: 如何在不收集大量方言训练数据的前提下，让LLM在隐私政策QA中对多种英语方言保持一致的性能表现？

**切入角度**: 受人类中心设计(Human-Centered Design)原则启发，设计两个专业化agent的"结构化提示协作"——方言Agent负责方言理解与翻译，隐私政策Agent负责领域专业回答，二者通过迭代反馈达成共识。

**核心 idea**: 用方言Agent翻译+验证 + 隐私政策Agent回答 + 迭代协商的多agent框架，在零训练条件下消除隐私QA的方言偏差。

## 方法详解

### 整体框架

框架包含两个agent和四个步骤：Step 1: 方言Agent接收非标准方言查询，利用方言背景知识将其翻译为SAE → Step 2a: 隐私政策Agent用翻译后的查询和政策文本生成初始回答 → Step 2b: 方言Agent评估回答是否忠实于原始方言查询的意图 → Step 2c: 若不一致则反馈给隐私政策Agent重新修正，最多迭代2轮后返回最终答案。

### 关键设计

1. **方言Agent(Dialect Agent)**
    - 功能：翻译方言查询为SAE + 验证最终回答是否符合用户原始意图
    - 核心思路：通过prompt注入目标方言的语音、语法、词汇和文化简述，使agent理解方言特征；负责翻译输入和评审输出两个阶段
    - 设计动机：直接用SAE prompt处理方言输入会丢失细微语义差异；方言背景知识是消除偏差的关键信息

2. **隐私政策Agent(Privacy Policy Agent)**
    - 功能：基于标准化查询和隐私政策文本生成专业回答
    - 核心思路：被prompt为领域专家，熟悉典型隐私政策结构和术语，输出答案和依据理由
    - 设计动机：隐私政策QA需要专业的法律/技术文本理解能力，需要独立角色承担

3. **迭代协商机制(Iterative Refinement)**
    - 功能：确保回答准确性和方言忠实性
    - 核心思路：方言Agent评审隐私政策Agent的回答，若发现不准确则提供反馈，隐私政策Agent据此修正，最多循环2轮
    - 设计动机：单次翻译无法完全捕捉用户语言的细微差别，迭代反馈可修正微妙的理解偏差

### 损失函数 / 训练策略

本方法完全基于推理时的prompt工程，不涉及任何训练或微调。在few-shot设置中，每个agent使用8个覆盖多种方言、问题类型和政策场景的示例。使用Multi-VALUE框架将标准英语查询转换为50种英语方言进行系统性评测。

## 实验关键数据

### 主实验

| 模型+设置 | SAE | AAVE | 牙买加 | 原住民 | 威尔士 | 西南英 | 平均 | 最大差 |
|----------|-----|------|--------|--------|--------|--------|------|--------|
| GPT-4o-mini Zero | .394 | .344 | .332 | .329 | .312 | .301 | .335 | .093 |
| GPT-4o-mini Few | .605 | .573 | .562 | .555 | .547 | .547 | .565 | .058 |
| **GPT-4o-mini Multi-agent-zero** | **.601** | **.588** | **.578** | **.587** | **.592** | **.576** | **.587** | **.025** |
| GPT-4o-mini Multi-agent-few | .611 | .595 | .596 | .602 | .592 | .594 | .598 | .019 |
| Llama 3.1 Zero | .469 | .349 | .370 | .325 | .356 | .336 | .368 | .144 |
| **Llama 3.1 Multi-agent-zero** | **.549** | **.527** | **.520** | **.524** | **.523** | **.526** | **.528** | **.029** |
| DeepSeek-R1 Multi-agent-zero | .582 | .579 | .583 | .579 | .566 | .573 | .577 | .017 |

### 消融实验

| 消融实验 | PrivacyQA Initial | Final | PolicyQA Initial | Final |
|---------|-------------------|-------|-----------------|-------|
| Zero-shot | .53 | .59 (+.06) | .43 | .45 (+.02) |
| Few-shot | .58 | .61 (+.03) | .47 | .48 (+.01) |

| 方言信息消融 | 初始F1 | 最终F1 |
|-------------|--------|--------|
| 有方言信息 | .5772 | .5966 |
| 无方言信息 | .5210 | .5894 |

### 关键发现

- Zero-shot多agent框架匹配甚至超越few-shot基线——GPT-4o-mini zero-shot multi-agent(.587) > few-shot baseline(.565)
- 方言间性能差异(Max Diff)降低最高达82%(GPT-4o-mini从.093降至.019)
- 迭代协商带来一致性提升（初始→最终：+.02到+.06），验证单次翻译不够充分
- 方言背景信息在初始阶段帮助更大(+.056)，但最终结果差异缩小(+.007)
- 多agent框架不仅提升非标准方言性能，SAE性能也略有提升——验证协作设计对所有用户有益

## 亮点与洞察

- 首次系统研究方言偏差在隐私政策QA领域的影响，具有重要社会意义
- 方法完全不需要训练或方言特定数据——仅通过prompt工程实现方言鲁棒性，可移植性极强
- Zero-shot框架超越few-shot基线的结果非常惊人，说明结构化agent协作的价值可能大于few-shot示例
- 从EPIC的声明切入(边缘化社区在数据收集中受到不成比例的伤害)，研究动机有强社会价值

## 局限与展望

- Multi-VALUE是基于规则的方言转换框架，可能无法完全代表真实方言用户的表达方式
- 迭代协商增加推理开销(最多3次LLM调用)，延迟敏感场景的适用性有限
- 仅在英语方言中验证，跨语言方言(如中文方言、阿拉伯语方言)的适用性未知
- 方言背景知识的准确性和全面性依赖于prompt设计，可能引入新的偏差
- 测试的5种方言仅是50种中性能最差的子集，完整评估有限

## 相关工作与启发

- 与DADA/TADA等需要方言感知训练的方法不同，本框架通过推理时agent协作避免训练需求
- 与LongAgent等多agent系统的联系：都用agent间迭代通信提升可靠性，但本文聚焦公平性
- 启发：多agent框架不仅是解决复杂任务的工具，也是解决公平性问题的有效范式——通过角色分工注入被忽视的背景知识

## 评分

- **新颖性**: ⭐⭐⭐⭐ (方言偏差+多agent的组合是新颖的应用创新)
- **实验充分度**: ⭐⭐⭐⭐ (3个模型×2个数据集×6种方言，消融充分)
- **写作质量**: ⭐⭐⭐⭐ (动机和社会意义阐述有力，方法描述清晰)
- **价值**: ⭐⭐⭐⭐ (对NLP公平性研究和隐私保护领域都有启发)

<!-- RELATED:START -->

## 相关论文

- [Unveiling Privacy Risks in LLM Agent Memory](mextra_agent_memory_privacy.md)
- [iAgent: LLM Agent as a Shield between User and Recommender Systems](iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)
- [Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)

<!-- RELATED:END -->
