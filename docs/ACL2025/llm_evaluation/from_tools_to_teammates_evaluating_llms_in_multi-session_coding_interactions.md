---
title: >-
  [论文解读] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions
description: >-
  [ACL 2025][multi-session dialogue] 提出 MemoryCode 合成多会话数据集评估 LLM 在长期交互中追踪和执行编码指令的能力，发现即使 GPT-4o 在提供完整对话历史时准确率也下降 67%，揭示了当前 LLM 在前瞻性记忆和信息整合上的根本局限。
tags:
  - ACL 2025
  - multi-session dialogue
  - long-term memory
  - coding instructions
  - prospective memory
  - LLM evaluation
---

# From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions

**会议**: ACL 2025  
**arXiv**: [2502.13791](https://arxiv.org/abs/2502.13791)  
**代码**: https://github.com/Cohere-Labs-Community/MemoryCode  
**领域**: LLM/NLP  
**关键词**: multi-session dialogue, long-term memory, coding instructions, prospective memory, LLM evaluation

## 一句话总结
提出 MemoryCode 合成多会话数据集评估 LLM 在长期交互中追踪和执行编码指令的能力，发现即使 GPT-4o 在提供完整对话历史时准确率也下降 67%，揭示了当前 LLM 在前瞻性记忆和信息整合上的根本局限。

## 研究背景与动机

**领域现状**：LLM 在单次任务解决上表现出色，被广泛用于工作环境。但从"工具"转变为"队友"需要在多次交互中保持和利用信息的能力。

**现有痛点**：现有多轮/多会话评估要么聚焦简单信息检索（Needle-in-Haystack），要么缺少实际任务要求。大多数工作在提示检索时才测试记忆，不测试前瞻性记忆（spontaneous retrieval）。

**核心矛盾**：LLM 能否像新同事一样，在日常工作交互中积累的信息用于未来任务（不被显式提示去回忆）？

**本文目标** 测试 LLM 在跨会话的长期交互中，自发检索并执行之前收到的编码规范的能力。

**切入角度**：设计导师-新人入职场景，指令简单（如"函数名以 g_ 开头"），但分散在大量无关信息中，且可能被更新。

**核心 idea**：LLM 在隔离指令场景下表现完美，但在真实多会话对话历史中准确率暴跌——问题不在于执行能力，而在于检索和整合能力。

## 方法详解

### 整体框架
手动设计种子（指令、干扰信息、人设、公司名）-> 模板化组合 -> LLM 生成多会话对话历史 -> 在不同层级评估：仅指令 / 单会话 / 完整历史。

### 关键设计

1. **四类种子**

    - **指令（51条）**：简单编码规范（如"函数名以 g_ 开头"），16 条可更新最多 8 次
    - **干扰信息（80条）**：公司政策、非代码相关指令
    - **人设**：导师 6 种 × 新人 5 种
    - **名字**：随机虚构公司/人名
    - 设计动机：种子的组合保证数据多样性

2. **可控对话历史生成**

    - 参数化模板：会话数（1-100）、含指令的会话比例（50-70%）、每会话指令数（1-3）、更新比例等
    - Command R+ 生成自然对话
    - 设计动机：精确控制复杂度维度

3. **三层评估**

    - **Level 1 — 仅指令**：直接给指令 + 任务，测试执行能力
    - **Level 2 — 单会话**：给含指令和干扰的完整会话，测试会内检索
    - **Level 3 — 完整历史**：给所有会话串联，测试跨会话检索+整合
    - 设计动机：逐步增加难度，分离"不会做"和"找不到"

4. **测试函数设计**

    - 用正则表达式检测指令是否被遵循（如函数名是否以 g_ 开头）
    - 不评估代码功能正确性，仅评估规范遵从
    - 设计动机：将检索能力与编程能力解耦

### 数据集规模
- 360 个对话历史，每种会话数 30 个
- "短"历史（<15 会话，54%），"长"历史（>15 会话，46%）
- 最长历史 63K tokens

## 实验关键数据

### 主实验 — 不同层级的准确率

| 模型 | Level 1 (仅指令) | Level 2 (单会话) | Level 3 (完整历史) | 下降 |
|------|-----------------|-----------------|-------------------|------|
| GPT-4o | ~95% | ~85% | **~28%** | **-67%** |
| DeepSeek-R1 | ~93% | ~80% | ~35% | -58% |
| Claude-3.5 | ~92% | ~82% | ~30% | -62% |
| Command R+ | ~88% | ~75% | ~25% | -63% |
| Llama-3.1-70B | ~85% | ~65% | ~20% | -65% |

### 会话数量对性能的影响

| 会话数 | GPT-4o 准确率 | 小模型准确率 |
|--------|-------------|------------|
| 1-5 | ~70% | ~50% |
| 10-20 | ~35% | ~20% |
| 50-100 | **~15%** | **~5%** |

### 消融 — 指令更新的影响

| 配置 | 准确率 | 说明 |
|------|--------|------|
| 无更新 | 较高 | 仅需检索 |
| 有更新（取最新） | **显著下降** | 需整合+覆盖旧信息 |

### 关键发现
- **所有模型在 Level 1 都能做对**：指令本身很简单，不是能力问题
- **Level 3 准确率暴跌**：GPT-4o 从 95% 降到 28%，证明问题在于检索而非执行
- **会话越多性能越差**：100 个会话时几乎所有模型都失败
- **指令更新是额外挑战**：模型不仅要找到指令，还要找到最新版本
- **DeepSeek-R1（推理模型）略优但仍然差**：推理能力无法弥补检索缺陷
- **干扰信息（尤其是类似指令的干扰）严重影响性能**

## 亮点与洞察
- **三层评估设计**精确地分离了"执行能力"和"检索能力"——Level 1→3 的性能断崖式下降清晰地证明瓶颈在检索。
- **入职导师场景**非常贴近真实工作场景——这正是 LLM 作为编程助手的典型使用方式，发现的局限直接影响产品设计。
- **前瞻性记忆**（不被提示就自发想起之前的指令并应用）是人类合作的基本能力，但 LLM 几乎不具备。

## 局限与展望
- 合成数据可能不完全反映真实工作场景的复杂性
- 仅测试编码规范，未测试更复杂的知识（如架构决策）
- 未探索外部记忆机制（如 RAG、记忆模块）的缓解效果
- 改进方向：带记忆模块的 LLM 评估、自适应信息压缩、渐进式指令学习

## 相关工作与启发
- **vs Needle-in-Haystack**：NIAH 显式提示检索，MemoryCode 需要隐式/前瞻性记忆
- **vs MMMT-IF**：MMMT-IF 测试指令遵循但不涉及多会话更新
- **vs LoCoMo**：LoCoMo 测试事实检索，MemoryCode 测试指令应用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个测试多会话前瞻性记忆+指令整合的 benchmark
- 实验充分度: ⭐⭐⭐⭐ 多模型 × 多会话长度 × 三层评估
- 写作质量: ⭐⭐⭐⭐⭐ 场景设计贴近真实，实验逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 作为协作工具的产品设计有直接启示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ICML 2025\] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](../../ICML2025/llm_evaluation/evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)
- [\[ACL 2025\] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)

<!-- RELATED:END -->
