---
title: >-
  [论文解读] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering
description: >-
  [ACL 2026][LLM Agent][表格问答] 提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "表格问答"
  - "多Agent框架"
  - "多推理路径"
  - "模型无关"
  - "LLM效率"
---

# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

**会议**: ACL 2026  
**arXiv**: [2602.09642](https://arxiv.org/abs/2602.09642)  
**代码**: [GitHub](https://github.com/AIDASLab/MATA)  
**领域**: LLM Agent  
**关键词**: 表格问答, 多Agent框架, 多推理路径, 模型无关, LLM效率

## 一句话总结
提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。

## 研究背景与动机

**领域现状**：LLM 显著推动了表格问答（TableQA）的发展，使自然语言与结构化表格的交互成为可能。现有方法通常利用 CoT、PoT（Program-of-Thought）或 text2SQL 等推理策略生成答案。

**现有痛点**：(1) 多数高性能方法依赖闭源 LLM（GPT-4o 等），在隐私敏感或成本受限场景下不适用，且在开源小模型上的可靠性未被充分验证；(2) 为提高答案可靠性，现有框架频繁调用 LLM 推理（如 Self-Consistency），导致计算成本高昂甚至因 over-prompting 反而降低准确率；(3) 大多数框架仅利用 CoT+PoT 两种推理路径，未能充分利用 CoT、PoT 和 text2SQL 三种互补推理策略的多样性。

**核心矛盾**：推理多样性与推理效率之间的权衡——增加推理路径可提升准确率，但每条路径都需要 LLM 推理开销，盲目执行所有路径既浪费又可能引入噪声。

**本文目标**：构建模型无关的 TableQA 框架，在多种开源/闭源 LLM 上都能保持高准确率，同时通过智能调度最小化 LLM 调用次数。

**切入角度**：推理多样性不需要固定的推理预算——通过轻量级控制器决定哪些推理分支是必要的、何时可以提前终止验证。

**核心 idea**：用轻量工具模型（Scheduler、Confidence Checker、Format Matcher）协调多个 LLM Agent 的推理路径选择和答案验证，实现推理多样性与效率的最优平衡。

## 方法详解

### 整体框架
MATA 接收表格 T 和问题 Q，通过三阶段流程产出最终答案：(1) Agent 选择阶段：Scheduler 决定 PoT 和 text2SQL 的执行优先级，同时 CoT Agent 并行执行；(2) 代码生成与调试阶段：PoT/text2SQL Agent 生成代码并由 Debug Agent 迭代修复；(3) 最终答案决策阶段：Confidence Checker 评估候选答案置信度，必要时调用 Judge Agent 仲裁。

### 关键设计

1. **调度器 (Scheduler)**:

    - 功能：根据表格特征和问题语义决定优先执行 PoT 还是 text2SQL
    - 核心思路：基于 MobileBERT + 两层 MLP 构建（仅 24.65M 参数），输入表格的元特征（大小、schema、数据类型）和问题语义，输出 PoT 和 text2SQL 的概率。优先执行概率更高的路径，若其答案与 CoT 一致则跳过另一路径，直接进入答案选择
    - 设计动机：不同推理路径的优势取决于底层模型特性和问题类型，智能调度可避免不必要的 LLM 调用。训练数据来自 WikiTQ/TabMWP/TabFact 上三个 LLM 的推理结果标注

2. **置信度检查器 (Confidence Checker)**:

    - 功能：为每个候选答案计算置信度分数，决定是否需要额外的 Judge Agent 仲裁
    - 核心思路：基于 DeBERTaV3-large 微调（~435M 参数），输入表格、问题和候选答案，输出各推理路径的置信度分数。若最高置信度超过阈值 $\theta=0.1$，直接选择该答案；否则调用 Judge Agent 综合判断
    - 设计动机：避免每次都调用昂贵的 LLM Judge，大部分情况下轻量模型即可完成高质量的答案筛选

3. **代码生成与调试循环 (Code Generation & Debugging)**:

    - 功能：迭代修复 PoT/text2SQL 生成的代码中的语法和逻辑错误
    - 核心思路：PoT/text2SQL Agent 生成代码并执行，若出错则由对应的 Debug Agent（PDA/SDA）修复，最多迭代 $N=3$ 轮。引入提前终止条件：若新代码与前一版高度相似且执行结果相同则停止
    - 设计动机：代码推理天然存在语法错误的倾向，而文本推理（CoT）的迭代修复收益甚微，因此只对代码路径进行调试以平衡成本和效果

### 损失函数 / 训练策略
Scheduler 和 Confidence Checker 分别在 173,664 条样本上训练。Scheduler 训练标签为 PoT 或 text2SQL 路径是否正确，CC 训练标签为三条路径各自的正确性。所有 LLM Agent 共享同一骨干模型，仅通过角色提示区分。

## 实验关键数据

### 主实验

| 基准 | 指标 | MATA | MixSC | SynTQA | TabLaP |
|------|------|------|-------|--------|--------|
| Penguins (平均) | EM | 0.881 | 0.626 | 0.810 | 0.524 |
| Penguins (平均) | F1 | 0.881 | 0.637 | 0.811 | 0.544 |
| TableBench (平均) | EM | 0.451 | 0.286 | 0.322 | 0.260 |
| TableBench (平均) | F1 | 0.482 | 0.331 | 0.362 | 0.307 |

### 消融实验

| 配置 | Penguins EM | TableBench EM | 说明 |
|------|------------|--------------|------|
| MATA (完整) | 0.881 | 0.451 | 完整框架 |
| w/o Scheduler | ~0.86 | ~0.43 | 执行所有路径，LLM 调用增加 |
| w/o CC (仅 JA) | ~0.85 | ~0.42 | 每次都调用 Judge Agent |
| w/o Debug | ~0.82 | ~0.38 | 不进行代码调试 |

### 关键发现
- MATA 在小模型（3B-7B）上的提升最为显著：qwen2.5-3b 从 TabLaP 的 0.163 EM 提升至 0.291，mistral-7b 从 0.036 提升至 0.294
- 在大模型上，MATA 仍然保持优势但差距缩小，因为大模型本身推理能力更强
- Scheduler 有效减少了约 30-40% 的 LLM 调用，同时保持甚至提升了准确率

## 亮点与洞察
- 轻量工具模型（总计不到 1B 参数）配合 LLM Agent 的设计非常实用——Scheduler 和 CC 作为"守门人"，避免了不必要的昂贵推理调用
- 三种推理路径的互补性被充分验证：CoT 擅长模糊/直觉性问题，PoT 擅长数值计算，text2SQL 在结构化查询上更精确
- 模型无关设计使框架可直接迁移到任何新 LLM，这在开源模型快速迭代的当下非常有价值

## 局限与展望
- Scheduler 和 CC 的训练依赖特定数据集（WikiTQ/TabMWP/TabFact），可能在领域差异较大的表格上泛化受限
- 当前仅支持单表推理，多表关联问答尚未涉及
- Debug 循环的最大迭代次数 N=3 是经验值，不同复杂度的任务可能需要自适应调整

## 相关工作与启发
- **vs MixSC**: MixSC 仅整合 CoT 和 Python 两种路径并用自一致性投票，缺少 text2SQL 和智能调度，MATA 的平均 EM 高出 25.5%
- **vs SynTQA**: SynTQA 集成 text2SQL 和 E2E TQA 但不支持模型切换，MATA 的模型无关设计使其在小模型上优势巨大
- **vs TabLaP**: TabLaP 依赖多个不同 LLM 协同且仅支持特定模型，MATA 统一用同一骨干实现更优效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 轻量工具+多Agent协调的架构设计新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 10个LLM、两个基准、三种指标，覆盖面极广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法描述详细
- 价值: ⭐⭐⭐⭐ 模型无关框架对工业部署有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ACL 2026\] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
