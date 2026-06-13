---
title: >-
  [论文解读] DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking
description: >-
  [ACL 2025][模型压缩][RAG] 提出面向复杂工程方案设计的新基准 SolutionBench 和新系统 SolutionRAG，通过树搜索探索+双视角思维（设计-审查交替）在 RAG 框架下逐步生成满足多约束的可靠工程方案，在 8 个工程领域达到 SOTA。 - 任务定义： 复杂工程方案设计要求系统针对含多个现实…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "RAG"
  - "tree search"
  - "engineering solution design"
  - "bi-point thinking"
  - "benchmark"
---

# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking

**会议**: ACL 2025  
**arXiv**: [2502.20730](https://arxiv.org/abs/2502.20730)  
**代码**: [GitHub](https://github.com/Li-Z-Q/DeepSolution)  
**arXiv**: [2502.20730](https://arxiv.org/abs/2502.20730)  
**代码**: [https://github.com/Li-Z-Q/DeepSolution](https://github.com/Li-Z-Q/DeepSolution)  
**领域**: 模型压缩  
**关键词**: RAG, tree search, engineering solution design, bi-point thinking, benchmark  
**作者**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun (中科院软件所 & 通义实验室)  

## 一句话总结

提出面向复杂工程方案设计的新基准 SolutionBench 和新系统 SolutionRAG，通过树搜索探索+双视角思维（设计-审查交替）在 RAG 框架下逐步生成满足多约束的可靠工程方案，在 8 个工程领域达到 SOTA。

## 研究背景与动机

- **任务定义**: 复杂工程方案设计要求系统针对含多个现实约束的工程需求（如：在年降雨 3000mm、膨胀土、高频地震区设计安全高效的医院建设方案），生成完整且可行的解决方案
- **现有差距**: 以往 RAG 研究主要关注 Multi-hop QA 和 Long-form QA，其答案是实体片段或拼接段落；而工程方案设计需要灵活的改进过程和满足所有约束的完整方案，二者存在本质不同
- **核心挑战**: (1) 从次优方案到可靠方案的改进路径灵活、无固定推理模式；(2) 需求含多个现实约束，单次生成难以保证全部满足

## 方法详解

### 1. SolutionBench 基准构建

构造流程：从权威工程期刊收集技术报告 → 用 GPT-4o 按人工设计的模板提取结构化内容 → 人工校验与去重 → 整合为 8 个工程领域的数据集与知识库。

每条数据包含 5 个字段：
- **Requirement**: 来自真实场景的复杂工程需求
- **Solution**: 行业专家设计的标准方案
- **Analytical Knowledge**: 分析需求时使用的专业知识
- **Technical Knowledge**: 解决需求时使用的技术知识
- **Explanation**: 专家的方案设计过程说明

覆盖 8 个领域：环境、矿业、交通、航空航天、电信、建筑、水利、农业，共约 950 条数据、约 6000 条知识。

### 2. SolutionRAG 系统

核心思想是在 **双视角思维树 (Bi-point Thinking Tree)** 上进行树搜索推理：

**(a) 双视角思维树结构**
- **Solution Node**: 存放针对需求设计的方案（浅层可靠性低、深层可靠性高）
- **Comment Node**: 存放对某个方案的审查意见（指出方案的不足之处）
- 两类节点交替连接：方案节点 → 评论节点 → 更优方案节点 → ...

**(b) 节点扩展 — Design & Review**
- **Design**: 给定需求 q、上层评论 c 和历史方案 s，先用 LLM 采样生成 H 个改进提案 → 为每个提案从知识库检索相关知识 → 结合这些信息生成更优方案
- **Review**: 给定需求 q 和当前方案 s，同样生成 H 个审查方向 → 检索知识 → 生成审查评论

**(c) 节点评估与剪枝**
- 利用 LLM 的 logits 对方案节点和评论节点打分
    - 方案评分：拼接方案+评论+后缀 "According to the comment, above solution is reliable"，取平均 logits 作为可靠性分数
    - 评论评分：拼接旧方案+评论+新方案+后缀 "Comparing the new solution and old solution, the comment is helpful"，取平均 logits 作为有用性分数
- 每层只保留得分最高的 W 个节点，实现效率与性能的平衡

**超参数设置**: 树最大深度 L=5，每节点子节点数 H=2，保留节点数 W=1；基座模型 Qwen2.5-7B-Instruct，检索模型 NV-Embed-v2，检索数 R=10。

## 实验关键数据

### 表 1: SolutionBench 数据统计

| 工程领域 | 数据条数 | 知识条数 |
|:---------|:---------|:---------|
| 环境 | 119 | 554 |
| 矿业 | 117 | 543 |
| 交通 | 124 | 870 |
| 航空航天 | 115 | 802 |
| 电信 | 116 | 840 |
| 建筑 | 118 | 858 |
| 水利 | 119 | 802 |
| 农业 | 122 | 868 |

### 表 2: 主实验结果（Analytical Score / Technical Score）

| 方法 | 环境 | 矿业 | 交通 | 航空航天 | 电信 | 建筑 | 水利 | 农业 |
|:-----|:-----|:-----|:-----|:---------|:-----|:-----|:-----|:-----|
| o1-2024-12-17 | 60.5/48.3 | 51.9/37.5 | 57.3/44.7 | 57.8/47.6 | 63.5/52.3 | 61.2/52.0 | 59.9/50.4 | 62.9/52.2 |
| Naive-RAG | 64.8/62.2 | 57.2/40.1 | 62.7/54.9 | 67.7/65.4 | 67.4/66.8 | 66.2/63.3 | 66.0/57.5 | 65.7/63.0 |
| Self-RAG | 64.2/63.6 | 56.1/41.6 | 62.9/56.5 | 68.8/69.9 | 67.6/66.9 | 66.7/65.9 | 64.8/58.6 | 65.1/61.1 |
| **SolutionRAG** | **66.4/67.9** | **59.7/50.5** | **64.1/58.5** | **69.9/72.7** | **68.8/69.0** | **67.9/68.0** | **66.0/60.7** | **66.9/65.2** |

- SolutionRAG 在全部 8 个领域均达到 SOTA
- 矿业领域 TS 较 Naive-RAG 提升 +10.4，较 Self-RAG 提升 +8.9

### 表 3: 消融实验（Overall AS/TS）

| 配置 | Overall AS | Overall TS |
|:-----|:-----------|:-----------|
| SolutionRAG (完整) | 66.2 | 64.1 |
| w/o 树结构（退化为单链） | 62.7 | 61.7 |
| w/o 双视角思维（仅方案，无审查） | 62.9 | 61.5 |

树搜索和双视角思维对性能贡献相当，均显著正向。

## 亮点

- 新颖地定义了「复杂工程方案设计」任务，填补 RAG 在工程领域的研究空白
- 双视角思维（设计+审查交替）是对 self-refine 的结构化升级，通过审查节点显式约束方案的完备性
- 树搜索 + logits 评分剪枝机制简洁高效，仅用 7B 模型即可超越 o1 等大模型
- 构建了覆盖 8 个工程领域、含专家标注的高质量基准 SolutionBench

## 局限性

- 仅利用现有 LLM 能力，未引入强化学习训练，方案质量受基座模型上限制约
- 受 GPU 资源限制，未充分探索树的宽度 H 和深度 L 等超参数空间
- 评估依赖 GPT-4o 打分，可能存在评分偏差
- 论文领域标注为 model_compression，实际为 RAG/工程设计方向，分类有偏差

## 与相关工作的对比

| 对比维度 | 已有方法 | SolutionRAG |
|:---------|:---------|:------------|
| 任务类型 | Multi-hop QA / Long-form QA | 复杂工程方案设计（多约束+完整方案） |
| 推理结构 | 单链迭代（Self-RAG/RQ-RAG） | 双视角思维树（方案-审查交替） |
| 约束满足 | 无显式保障机制 | 审查节点显式检测约束缺失 |
| MCTS-based RAG | 缺乏工程约束保障机制 | 通过 bi-point thinking 保障方案可靠性 |

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次定义工程方案设计任务+构建专用基准，双视角思维树是有趣的结构创新
- **实验充分度**: ⭐⭐⭐⭐ — 8 领域全面测试+消融+树深度分析+剪枝有效性验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示直观，任务定义明确
- **实用价值**: ⭐⭐⭐ — 基准和系统有实际工程参考意义，但 GPT-4o 评估成本较高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[ACL 2025\] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)

</div>

<!-- RELATED:END -->
