---
title: >-
  [论文解读] Agentic Knowledgeable Self-Awareness
description: >-
  [ACL 2025][LLM Agent][agent planning] 本文提出 KnowSelf，一种数据驱动方法，通过在 agent 自探索轨迹上标注特殊 token 来标识不同思维情境（快速思考/慢速思考/知识思考），经两阶段训练（SFT + RPO）使 agent 模型学会自主判断何时需要调用外部知识，以最小知识消耗代价达到最优规划效果。
tags:
  - ACL 2025
  - LLM Agent
  - agent planning
  - self-awareness
  - special tokens
  - knowledge utilization
  - two-stage training
---

# Agentic Knowledgeable Self-Awareness

**会议**: ACL 2025  
**arXiv**: [2504.03553](https://arxiv.org/abs/2504.03553)  
**代码**: https://github.com/zjunlp/KnowSelf  
**领域**: LLM Agent / 智能体规划  
**关键词**: agent planning, self-awareness, special tokens, knowledge utilization, two-stage training

## 一句话总结
本文提出 KnowSelf，一种数据驱动方法，通过在 agent 自探索轨迹上标注特殊 token 来标识不同思维情境（快速思考/慢速思考/知识思考），经两阶段训练（SFT + RPO）使 agent 模型学会自主判断何时需要调用外部知识，以最小知识消耗代价达到最优规划效果。

## 研究背景与动机

LLM-based agent 在 ALFWorld、WebShop 等交互式规划任务上取得了显著进展，但现有方法普遍采用"大水漫灌"式的训练策略——不加区分地向模型注入黄金轨迹、外部反馈和领域知识。这种做法忽略了人类决策中的一个基本认知原则：**情境自我意识（situational self-awareness）**——即在决策时动态评估情境需求并策略性地调配认知资源的能力。

换言之，人类在做决策时会自动判断：这个步骤是简单的、可以快速直觉完成的，还是复杂的、需要停下来仔细思考的，还是超出自身能力、需要查阅外部资料的。现有 agent 训练方法缺乏这种自适应能力——要么完全不给知识（能力受限），要么每步都灌入知识（代价高昂、引入噪声）。

核心矛盾：**知识利用的粒度控制**——如何让 agent 模型像人一样"知道自己什么时候需要什么"？

切入角度：提出 **agentic knowledgeable self-awareness** 范式，将问题转化为让模型学会生成特殊 token 来标识当前所处的思维情境，从而自主调控知识利用策略。

## 方法详解

### 整体框架
KnowSelf 的 pipeline 分为三个阶段：
1. **知识系统构建**：生成步级轨迹对，通过比较分析生成并整合领域知识库
2. **训练数据构建**：让模型自探索生成轨迹，用启发式准则标注特殊 token，构造三种思维模式的训练数据
3. **两阶段训练**：Stage 1 用 SFT 学习特殊 token 生成；Stage 2 用 RPO 强化偏好对齐

### 关键设计

1. **三种思维情境与特殊 Token（Situation Classification）**:

    - 功能：将 agent 的每个决策步骤分类为三种思维模式
    - 核心思路：定义三种情境及对应的特殊 token：
        - **快速思考（Fast Thinking）** `[FAST]`：当前步骤简单，模型可直接凭已有能力快速决策，不需要额外停顿或知识
        - **慢速思考（Slow Thinking）** `[SLOW]`：当前步骤较难，模型需要仔细反思后再行动（类似 System 2 思维），但不需要外部知识
        - **知识思考（Knowledgeable Thinking）** `[KNOW]`：当前步骤超出模型能力，需要检索并利用外部知识库中的信息来辅助决策
    - 设计动机：这种分类直接对应人类认知的 dual process theory（快/慢系统），加上知识检索维度。通过特殊 token 实现了轻量级的行为切换，避免了复杂的路由网络或额外分类器

2. **启发式情境判断准则（Heuristic Situation Judgement）**:

    - 功能：自动标注训练数据中每一步应使用哪种思维模式
    - 核心思路：让模型先自主探索任务生成轨迹，构造"步级轨迹对"——同一任务状态下的成功/失败动作对。标注逻辑如下：
        - 如果模型自主决策成功 → 标注为 `[FAST]`
        - 如果模型决策失败，但经过反思（reflection）后能自行纠正 → 标注为 `[SLOW]`
        - 如果模型决策失败，反思后仍无法纠正，但给予外部知识后能成功 → 标注为 `[KNOW]`
    - 设计动机：通过失败驱动的标注，精确捕获了模型的能力边界——只有在模型真正"不会"的地方才引入知识，避免无谓的知识注入

3. **知识系统构建（Knowledge System Construction）**:

    - 功能：为 agent 构建可查询的领域知识库
    - 核心思路：参考 AutoManual 方法，使用 GPT-4o（2024-08-06）通过步级轨迹对比分析生成知识条目，然后进行知识整合去重。ALFWorld 限制为 24 条知识，WebShop 限制为 10 条
    - 设计动机：精炼的知识库确保每条知识都是高价值的，避免知识冗余导致的检索噪声。知识条目限制也鼓励模型更依赖自身能力而非外部知识

4. **两阶段训练（Two-Stage Training）**:

    - 功能：分阶段让模型学会生成特殊 token 并优化行为
    - 核心思路：
        - **Stage 1 — SFT（监督微调）**：在标注了特殊 token 的混合数据上进行标准因果语言建模训练。数据包含三种思维模式的样本，模型学会在适当时机生成 `[FAST]`/`[SLOW]`/`[KNOW]`。训练参数：Llama-3.1-8B 为例，lr=2e-5, batch_size=8, 3 epochs, max_seq_len=3072
        - **Stage 2 — RPO（Rejection-sampling Policy Optimization）**：收集 Stage 1 模型的失败轨迹作为负样本，配对对应的黄金轨迹作为正样本，进行偏好优化。$\beta=0.5$, lr=5e-7, 1 epoch
    - 设计动机：Stage 1 让模型学会"基本形式"（何时生成什么 token），Stage 2 通过对比训练进一步校准模型的决策质量，特别是减少关键步骤的失误率

### 推理流程
推理时，模型在每步决策前自动生成特殊 token：
- 生成 `[FAST]` → 直接输出动作
- 生成 `[SLOW]` → 先生成内部反思，再输出动作
- 生成 `[KNOW]` → 用 DeepSeek-V3 从知识库中检索相关知识，将知识注入上下文后再输出动作

## 实验关键数据

### 主实验

| 任务 | 模型 | 方法 | 成功率 | 外部知识使用步数 |
|------|------|------|--------|-----------------|
| ALFWorld | Llama-3.1-8B | ReAct | ~30% | 0 |
| ALFWorld | Llama-3.1-8B | Reflexion | ~45% | 0 |
| ALFWorld | Llama-3.1-8B | ETO | ~55% | 0 |
| ALFWorld | Llama-3.1-8B | KnowAgent | ~60% | 全步 |
| ALFWorld | Llama-3.1-8B | WKM | ~62% | 全步 |
| ALFWorld | Llama-3.1-8B | **KnowSelf** | **~70%** | **仅需少量步** |
| WebShop | Llama-3.1-8B | ReAct | ~25% | 0 |
| WebShop | Llama-3.1-8B | ETO | ~40% | 0 |
| WebShop | Llama-3.1-8B | **KnowSelf** | **~50%** | **仅需少量步** |

### 消融实验

| 配置 | ALFWorld成功率 | 说明 |
|------|--------------|------|
| KnowSelf（完整） | ~70% | 最优 |
| 去除 [KNOW]（无知识） | ~58% | 证明知识检索的必要性 |
| 去除 [SLOW]（无反思） | ~63% | 慢速思考有明显贡献 |
| 全步注入知识（无自主调控） | ~62% | 知识过量反而有害 |
| 仅 Stage 1（无 RPO） | ~64% | RPO 提升约6个点 |

### 关键发现
- KnowSelf 的**知识使用效率远高于基线**：在约20-30%的步骤使用外部知识就超越了全步使用知识的方法
- 全步注入知识反而比选择性使用效果差，验证了"大水漫灌"策略的弊端
- RPO（Stage 2）主要减少了关键决策点的失误，对简单步骤影响不大
- KnowSelf 在 Llama-3.1-8B 和 Qwen-2.5-7B 上都有效，展示了方法的通用性
- 特殊 token 的生成分布与任务难度高度相关：简单任务 `[FAST]` 多，难任务 `[KNOW]` 多

## 亮点与洞察
- **特殊 token 作为思维模式开关**：极其轻量的设计，仅通过 3 个特殊 token 实现了复杂的自适应行为切换，避免了额外模块的引入
- **失败驱动的标注策略**：只在模型真正失败的地方引入额外计算/知识，完美捕获了能力边界
- **双过程理论的实践**：快速思考/慢速思考的设计与认知科学的 System 1/System 2 理论高度契合
- **知识检索的按需调控**：不是"用不用知识"的二元选择，而是"何时用、用多少"的精细调控

## 局限与展望
- 知识系统依赖 GPT-4o 构建，成本和知识质量受限于辅助模型能力
- 仅在 ALFWorld 和 WebShop 两个任务上验证，需更多真实世界 agent 任务的验证
- 启发式标注准则可能非最优——模型失败不一定意味着需要知识（可能需要更好的推理策略）
- 知识库大小（24/10条）人工设定，自适应知识库规模可能更好
- 特殊 token 的三分类可能过于粗粒度，更细的情境划分（如"需要哪类知识"）可能有帮助

## 相关工作与启发
- **vs KnowAgent (Zhu et al., 2024)**: KnowAgent 全步使用知识图谱辅助规划，不区分情境；KnowSelf 按需使用知识，效率更高效果更好
- **vs WKM (World Knowledge Model)**: WKM 也引入外部知识，但同样是全步注入；KnowSelf 证明了选择性注入的优越性
- **vs ETO (Exploratory Training Optimization)**: ETO 通过探索优化但不使用外部知识；KnowSelf 在 ETO 基础上增加了知识维度和情境意识
- **vs Self-RAG**: Self-RAG 用特殊 token 控制检索时机，但针对 QA 任务；KnowSelf 将类似理念拓展到 agent 规划，且增加了慢速思考维度

## 评分
- 新颖性: ⭐⭐⭐⭐ 将认知科学的情境自我意识引入agent规划，特殊token设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 多个基线对比和消融，但仅两个任务场景略显不足
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，概念类比直观，但公式较少
- 价值: ⭐⭐⭐⭐ 提出了有实用价值的轻量级知识调控范式，开源代码和数据

<!-- RELATED:START -->

## 相关论文

- [Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [SMART: Self-Aware Agent for Tool Overuse Mitigation](smart_self-aware_agent_for_tool_overuse_mitigation.md)
- [REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)

<!-- RELATED:END -->
