---
title: >-
  [论文解读] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion
description: >-
  [ACL 2026 Findings][代码智能][Shapley 值] 提出 RepoShapley，一种基于 Shapley 值的联盟感知上下文过滤框架，通过估计检索代码片段在组合中的交互贡献来决定保留/丢弃，显著提升仓库级代码补全质量。 领域现状：仓库级代码补全需要解析跨文件依赖（如项目 API、共享契约）…
tags:
  - "ACL 2026 Findings"
  - "代码智能"
  - "Shapley 值"
  - "上下文过滤"
  - "仓库级代码补全"
  - "检索增强生成"
  - "联盟博弈"
---

# RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion

**会议**: ACL 2026 Findings  
**arXiv**: [2601.03378](https://arxiv.org/abs/2601.03378)  
**代码**: [github](https://github.com/yuhuo03/RepoShapley)  
**领域**: 信息检索 / 代码补全  
**关键词**: Shapley 值, 上下文过滤, 仓库级代码补全, 检索增强生成, 联盟博弈

## 一句话总结

提出 RepoShapley，一种基于 Shapley 值的联盟感知上下文过滤框架，通过估计检索代码片段在组合中的交互贡献来决定保留/丢弃，显著提升仓库级代码补全质量。

## 研究背景与动机

**领域现状**：仓库级代码补全需要解析跨文件依赖（如项目 API、共享契约），检索增强生成（RAG）通过注入跨文件证据来增强代码 LM。

**现有痛点**：检索到的代码片段效用具有交互依赖性——某些片段单独无用但与互补上下文配对时变得关键；某些片段看似相关但与冲突证据共存时反而降低生成质量。现有方法（如 CODEFILTER）独立评分每个片段，无法捕捉这种组合效应。

**核心矛盾**：固定上下文预算下，独立评分的片段效用与实际多片段组合消费时的效用存在系统性偏差。

**本文目标**：设计联盟感知的上下文过滤机制，用 Shapley 边际贡献信号监督片段选择。

**切入角度**：将上下文选择建模为合作博弈——每个检索片段为一个玩家，任意子集为一个联盟，用 Shapley 值量化片段在所有可能组合中的平均边际贡献。

**核心 idea**：通过轻量级代理博弈近似 Shapley 值 + 有界后验证选择最优联盟，再将验证结果蒸馏为离散控制 token 实现在线推理。

## 方法详解

### 整体框架

两阶段流程：(1) ChunkShapley 离线标注：单片段探测 → 代理博弈 → 精确 Shapley 值 → 有界后验证生成 keep/drop 标签；(2) RepoShapley 在线推理：将验证标签蒸馏为控制 token（`<KEEP>`/`<DROP>`/`<NEED>`/`<DONE>`），使单一模型同时完成检索触发、片段选择和代码生成。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["检索候选片段（top-K，K≤10）"] --> OFF
    subgraph OFF["ChunkShapley 离线标注"]
        direction TB
        B["单片段探测与代理博弈<br/>逐片段似然增益 → 代理效用 v_sur"] --> C["精确 Shapley 值与后验证<br/>枚举 2^K 子集 + 冻结生成器选最优联盟 S*"]
    end
    OFF --> D["keep/drop 标签 + 检索触发标签"]
    subgraph ON["RepoShapley 蒸馏与在线推理"]
        direction TB
        E["双格式蒸馏训练<br/>Format-1 选择 + Format-2 生成，共享参数"] --> F["单模型预测控制 token<br/>NEED/DONE 触发检索，KEEP/DROP 选片段"]
    end
    D --> E
    F --> G["仓库级代码补全输出"]
```

### 关键设计

1. **单片段探测与代理博弈（surrogate game）**：对每个候选片段 $cc_i$ 计算单独 teacher-forced log-likelihood 增益 $\Delta_i = \ell(X_{in}, \{cc_i\}) - \ell(X_{in})$，得到符号 $y_i = \text{sign}(\Delta_i)$ 和权重 $\omega_i = |\Delta_i|$。定义代理效用 $v_{sur}(S) = \sigma(\beta \sum_{i \in S} \omega_i y_i) - \sigma(0)$，其中 logistic（sigmoid）的饱和性自然捕捉冗余效应，负投票（$y_i=-1$）建模冲突。这一步用 $O(K)$ 次探测就把组合效应压成一维加权投票，为下一步精确枚举铺路。

2. **精确 Shapley 值与后验证**：由于代理效用 $v_{sur}$ 为闭式解，可在 $K \leq 10$ 的检索集上精确枚举 $2^K$ 个子集计算 Shapley 值。构建候选池 $\mathcal{C}$（Shapley 前缀 + $\Delta$ 前缀 + top-L 片段的 size-2/3 组合），用冻结生成器解码选择 ES/EM 最优联盟 $S^\star$。

3. **双格式蒸馏训练**：Format-1 监督选择（预测每个片段的 keep/drop token 序列）；Format-2 监督生成（仅用保留片段做 FIM 补全）。两种格式共享参数，使模型在统一自回归接口中学习选择与生成。

### 损失函数 / 训练策略

训练损失包含检索控制损失 $\mathcal{L}_R$（预测 `<NEED>`/`<DONE>`）和片段选择损失 $\mathcal{L}_S$（预测 keep/drop 序列），均使用标准交叉熵。推理时通过阈值 $t_c$ 决定是否触发检索。

## 实验关键数据

### 主实验

| 方法 | RepoEval Line EM | RepoEval API EM | CCLongEval Chunk ES | CCEval Line EM |
|---|---|---|---|---|
| No-Retrieve (SC-1B) | 43.14 | 38.03 | 47.29 | 18.72 |
| Full-Retrieve | 52.27 | 44.18 | 55.93 | 22.38 |
| RepoFormer | 54.71 | 45.73 | 57.69 | 25.42 |
| CODEFILTER | 57.19 | 48.37 | 59.91 | 27.81 |
| **RepoShapley** | **61.34** (+4.15) | **53.62** (+5.25) | **64.39** (+4.48) | **32.26** (+4.45) |

*StarCoder-Base-1B 上的代码补全性能*

### 消融实验

RepoShapley 在 11 个评测指标上全面超越所有对比方法（包括 No-Retrieve、Full-Retrieve、RepoFormer、CODEFILTER）。在 StarCoder-Base-7B 和 CodeLlama-13B 上也呈现一致优势，证明方法的 backbone 无关性。

### 关键发现

- 联盟感知监督比独立评分提升 4-5 个百分点
- Shapley 前缀选择优于单纯按 $\Delta$ 排序，验证了交互效应的重要性
- 检索触发控制有效减少不必要检索，同时不损失性能
- 代理博弈的 $\beta$ 参数控制饱和尺度，过大或过小都会退化

## 亮点与洞察

- **从独立到联盟的范式转换**：将上下文过滤从"逐个评分"升级为"组合博弈"，是 RAG 控制的重要思路演进
- **计算可行性的巧妙设计**：用轻量代理博弈避免指数级生成器评估，再通过有界后验证兜底精度，很好地平衡了效率与效果
- **蒸馏到控制 token**：将离线的组合推理能力压缩为在线的单 token 预测，工程上优雅

## 局限与展望

- 检索集大小限制为 $K \leq 10$，更大检索集需要采样近似
- 代理博弈的 sigmoid 假设可能对某些代码结构不适用
- 后验证需要目标序列 $Y$，只能用于离线标注，不能在线更新
- 未来可探索自适应 $\beta$ 调节和更丰富的交互建模

## 相关工作与启发

- Data Shapley (Ghorbani & Zou, 2019) 的数据价值评估思想在 RAG 场景的延伸
- 与 SHAP 的区别：本文是前向监督（构建训练标签），而非后向解释
- 控制 token 蒸馏思路可推广到其他需要动态上下文选择的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将 Shapley 值引入 RAG 上下文控制，联盟博弈视角非常新颖
- **实验充分度**: ⭐⭐⭐⭐ 多 benchmark、多 backbone 验证充分，消融分析详细
- **写作质量**: ⭐⭐⭐⭐ 数学形式化清晰，方法动机解释充分
- **价值**: ⭐⭐⭐⭐⭐ 对 RAG 场景的上下文控制提出了系统性的解决方案，具有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](../../ICML2026/code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](../../ACL2025/code_intelligence/feabench_repo_code_gen.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](../../ICLR2026/code_intelligence/improving_code_localization_with_repository_memory.md)

</div>

<!-- RELATED:END -->
