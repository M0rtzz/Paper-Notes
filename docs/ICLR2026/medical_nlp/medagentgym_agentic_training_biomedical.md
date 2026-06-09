---
title: >-
  [论文解读] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science
description: >-
  [ICLR 2026 Oral][医疗NLP][biomedical data science] 构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth…
tags:
  - "ICLR 2026 Oral"
  - "医疗NLP"
  - "biomedical data science"
  - "agentic training"
  - "code-centric reasoning"
  - "reinforcement-learning"
  - "Med-Copilot"
  - "LLM agent"
---

# MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science

**会议**: ICLR 2026 Oral  
**arXiv**: [2506.04405](https://arxiv.org/abs/2506.04405)  
**代码**: 有  
**领域**: 医疗NLP
**关键词**: biomedical data science, agentic training, code-centric reasoning, reinforcement-learning, Med-Copilot, LLM agent

## 一句话总结
构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，系统基准评估 29 个 LLM 揭示商业/开源差距，并通过高效多线程轨迹采样 + 离线/在线 RL 训练出 Med-Copilot，分别获得 +43.02%/+45.28% 提升，达到与 GPT-4o 竞争的性能。

## 研究背景与动机
**领域现状**：生物医学数据科学涵盖基因组分析、临床数据处理、医学图像分析、药物发现等多个子领域，每个任务需要复杂的编程和领域推理能力。LLM 作为编码助手已在通用编程领域展示潜力，但在生物医学编码任务上的系统评估和训练基础设施缺乏。

**现有痛点**：(1) 现有医学 AI benchmarks（如 MedQA、PubMedQA）是静态的选择题/问答评估，不支持交互式代码执行和迭代调试；(2) 没有统一平台涵盖多种生物医学数据科学场景（基因组、临床、影像、药物等都是各自独立的 benchmark）；(3) 开源 LLM 与闭源模型（GPT-4o 等）在生物医学编码任务上差距显著，需要有效的训练方法缩小差距。

**核心矛盾**：训练一个能写生物医学分析代码的 Agent 需要大规模可交互的任务环境，但构建这种环境成本极高（需要真实数据、ground truth、安全沙盒、反馈机制）。

**本文目标** 同时解决环境构建和 Agent 训练两个问题：提供大规模训练环境 + RL 训练 pipeline。

**切入角度**：将 12 个真实生物医学场景统一为"输入数据+任务描述→执行代码→验证输出"的标准化格式，支持交互式反馈和自动化评分。

**核心 idea**：大规模可交互的统一训练环境 + RL 训练 pipeline = 缩小开源模型与闭源 LLM 在生物医学编码上的差距。

## 方法详解

### 整体框架
MedAgentGym 包含三个核心组件：(1) **任务库**：72,413 个任务实例，每个任务包含数据文件、任务描述、可执行沙盒、ground truth 答案和评分函数；(2) **交互引擎**：Agent 通过多轮对话与沙盒交互——提交代码、获取执行结果/错误信息、迭代修正；(3) **训练 pipeline**：高效多线程轨迹生成 + 离线/在线 RL 训练。

### 关键设计

**1. 12 场景 × 129 类别任务体系：把割裂的生物医学 benchmark 收进一个统一接口**

过去基因组、临床、影像、药物各有各的 benchmark，Agent 没法在它们之间迁移。MedAgentGym 把 12 个真实场景——基因组学（RNA-seq 分析、基因表达聚类）、临床数据科学（EHR 预测、生存分析）、医学影像（病理切片分类、X-ray 检测）、药物发现（分子属性预测、ADMET 分析）等——统一成同一套标准化接口：输入是数据文件路径加元信息，任务指令用自然语言描述分析目标，Ground Truth 是精确数字答案或分类标签，再配一个评分函数 $\text{score}(\hat{y}, y) \in [0, 1]$。所有任务长得一样，Agent 才有可能在不同类型任务间迁移和泛化，而不是被困在单一领域里。

**2. 可执行沙盒 + 交互反馈：让 Agent 像真人一样边跑边调**

生物医学分析代码单次生成的正确率很低，很多任务必须靠调试才能跑通。MedAgentGym 给每个任务配一个隔离的 Python 执行环境（预装 pandas、scikit-learn、biopython 等库），Agent 提交代码后能拿到真实的 stdout/stderr 反馈。整个交互最多 $K$ 轮：每轮 Agent 生成代码 $c_t$，沙盒执行返回输出和错误 $(o_t, e_t)$，Agent 据此决定继续修正还是提交最终答案，最终形成一条轨迹 $\tau = [(c_1, o_1, e_1), ..., (c_K, o_K, e_K)]$。正是这条「跑错→看报错→改」的反馈回路，让 Agent 能从错误里学到东西，而不是一次性赌对。

**3. 多线程轨迹生成 + 离线/在线 RL：用可验证的评分当奖励训练 Agent**

有了沙盒和 ground truth 评分，训练信号天然可验证。MedAgentGym 并行采样多个任务的交互轨迹，喂给两条 RL 路线。离线 RL 先从 GPT-4o-mini、Claude-3.5-Sonnet、DeepSeek-V2.5 等多个 LLM 采集大批轨迹 $\{(\tau_i, r_i)\}$，以 ground truth 评分 $r = \text{score}(\hat{y}, y)$ 作奖励，挑出 $r > \theta$ 的成功轨迹当正样本，用 DPO 或 rejection sampling 训练——好处是直接复用已有轨迹、数据效率高。在线 RL 则让 Agent 亲自与环境交互采实时轨迹，用 PPO/GRPO 优化策略 $\pi_\theta$，奖励同样是 $R(\tau) = \text{score}(\hat{y}_\tau, y)$——好处是能持续探索、不断改进。两者互补，共同把开源小模型推向商业模型的水平。

### 训练策略
- 基础模型：Llama-3.1-8B-Instruct 作为 Med-Copilot 的骨干
- 离线阶段：从 GPT-4o-mini、Claude-3.5-Sonnet、DeepSeek-V2.5 等采集轨迹
- 在线阶段：Med-Copilot 自身与环境交互，每轮更新策略

## 实验关键数据

### 主实验：29 LLM 基准评估

| 模型类别 | 代表模型 | 平均 Score | 排名 |
|---------|---------|----------|------|
| 闭源商业 | GPT-4o | ~0.55 | Top-1 |
| 闭源商业 | Claude-3.5-Sonnet | ~0.50 | Top-3 |
| 开源基础 | Llama-3.1-8B-Instruct | ~0.32 | 中下 |
| Med-Copilot (离线 RL) | Llama-3.1-8B + Offline RL | ~0.46 (+43.02%) | 接近 GPT-4o |
| Med-Copilot (在线 RL) | Llama-3.1-8B + Online RL | ~0.46 (+45.28%) | 竞争 GPT-4o |

### 消融实验

| 配置 | 提升幅度 | 说明 |
|------|---------|------|
| 离线 RL only | +43.02% | 从多模型轨迹学习 |
| 在线 RL only | +45.28% | 自主探索，略优于离线 |
| 多轮交互 vs 单次 | 显著提升 | 证明交互反馈的价值 |
| 任务难度分层 | 简单任务提升大 | 困难任务仍有改进空间 |

### 关键发现
- 商业 LLM 与开源 LLM 在生物医学编码任务上存在显著差距（~20 分），但 RL 训练可显著缩小此差距
- 在线 RL 略优于离线 RL，但两者都大幅超越 SFT 基线
- 多轮交互（调试循环）对性能至关重要——单次代码生成的成功率远低于多轮迭代
- 不同生物医学场景的难度差异大：基础统计分析较简单，复杂的基因组管线分析较难

## 亮点与洞察
- **训练 + 评估一体化**：MedAgentGym 不仅是 benchmark（评估 29 个 LLM），更是训练环境（RL 训练管道直接可用）——这在医学 AI 领域是首创
- **实际缩小差距的证明**：8B 参数的开源模型通过 RL 训练达到 GPT-4o 水平——这对隐私敏感的医学场景极具实际价值（本地部署 vs API 调用）
- **规模化设计**：72K 任务 + 多线程轨迹采样 + 标准化接口——真正可扩展的训练基础设施
- **代码中心而非问答中心**：不同于 MedQA 等选择题 benchmark，MedAgentGym 要求写真实可执行的分析代码——更接近实际科研场景

## 局限与展望
- 任务以编码为中心，临床推理、诊断决策等知识密集型能力评估不足
- Ground truth 需要预定义的标准答案，不适合开放式探索性研究任务
- 当前仅用 8B 模型训练 Med-Copilot，更大模型（70B+）的扩展结果未报告
- 评分函数主要基于精确匹配或数值误差，未评估代码质量、可读性、效率等软指标
- 未评估模型在训练任务分布外的迁移能力

## 相关工作与启发
- **vs MedQA/PubMedQA**: 这些是静态 QA benchmark，无代码执行和交互反馈；MedAgentGym 支持多轮代码交互
- **vs SWE-bench**: SWE-bench 聚焦软件工程（修 bug），MedAgentGym 聚焦生物医学数据分析——任务性质不同
- **vs AgentBench**: AgentBench 覆盖多种 Agent 任务但不聚焦医学；MedAgentGym 提供深度的生物医学场景覆盖
- **vs AIME**: AIME 等评估医学推理，MedAgentGym 评估的是医学编程实践

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个统一的生物医学 Agent 训练环境，问题定义有价值，但 RL 训练方法本身不是新的
- 实验充分度: ⭐⭐⭐⭐⭐ 72K 任务 + 29 LLM 系统评估 + 离线/在线 RL 对比 + Med-Copilot 验证
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，任务分类和实验组织合理
- 价值: ⭐⭐⭐⭐⭐ 为生物医学 AI Agent 研究提供了关键基础设施，开源环境有长期社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](../../ACL2026/medical_nlp/ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_nlp/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](../../NeurIPS2025/medical_nlp/cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)
- [\[AAAI 2026\] Real-Time Trust Verification for Safe Agentic Actions Using TrustBench](../../AAAI2026/medical_nlp/real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)

</div>

<!-- RELATED:END -->
