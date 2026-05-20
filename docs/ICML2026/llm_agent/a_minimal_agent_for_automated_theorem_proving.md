---
title: >-
  [论文解读] A Minimal Agent for Automated Theorem Proving
description: >-
  [ICML 2026][LLM Agent][定理证明] 本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "定理证明"
  - "Lean 4"
  - "智能体架构"
  - "迭代细化"
  - "自管理记忆"
---

# A Minimal Agent for Automated Theorem Proving

**会议**: ICML 2026  
**arXiv**: [2602.24273](https://arxiv.org/abs/2602.24273)  
**代码**: https://github.com/Axiomatic-AI/ax-prover-base  
**领域**: LLM Agent / 形式数学  
**关键词**: 定理证明, Lean 4, 智能体架构, 迭代细化, 自管理记忆

## 一句话总结
本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件，在不微调的前沿 LLM（Claude Opus）上达到甚至超越 Hilbert/Seed-Prover 等专用系统，成本却低出 100 倍。

## 研究背景与动机

**领域现状**：AI 定理证明近年突破频出（AlphaProof、Hilbert、Seed-Prover），但多数依赖大规模合成数据微调或 RL，复杂度和成本极高。同时前沿通用 LLM 在形式数学能力上也在快速提升，但系统设计与模型进步对最终性能的贡献难以分离。

**现有痛点**：（1）复杂架构难复现；（2）与 Lean/Mathlib 版本耦合紧，升级需重训；（3）GPU 集群或 API 成本高；（4）迭代反馈、记忆、工具搜索各自贡献多少未量化。

**核心矛盾**：人们普遍假设强证明器需要复杂设计，但是否成立？简化是否会让性能崩溃？

**本文目标**：找到"最小必要模块组合"，用极简架构达成竞争力性能，并给出清晰消融基线。

**切入角度**：从 ReAct 框架出发，把系统拆为三个可替换模块：Proposer、Reviewer、Memory。自底向上逐层堆叠，量化边际收益。

**核心 idea**：迭代反馈 >> 记忆 >> 工具搜索；"编译反馈 + 自我反思笔记本"已能匹敌最复杂系统，工具搜索只是锦上添花。

## 方法详解

### 整体框架

核心循环：

```
while not_proved and iters < N:
  proposal = Proposer(theorem, file_context, memory)
  feedback = Compiler(proposal)
  if not_proved:
    memory.update(proposal, feedback, reasoning)
```

- **Proposer**：通用 LLM（Claude Opus）或 ReAct 智能体，可选挂载 LeanSearch / Tavily 工具。
- **Reviewer**：Lean 4 编译器 + LLM 审查者双层验证，防止伪证。
- **Memory**：三种策略——无记忆、历史记忆（前 N 次完整尝试）、自管理笔记本（LLM 维护精炼洞察）。

### 关键设计

1. **受限工具调用的 Proposer**:

    - 功能：让模型可检索 Mathlib 定理与网络信息，但限制调用次数防止信息噪声主导提议过程。
    - 核心思路：在每次提议前最多并行调用一轮 LeanSearch（向量检索定理）和 Tavily（网络搜索）。Web 搜索允许的理由是核心难度不在逻辑而在"写出能编译的 Lean 代码"。
    - 设计动机：工具有帮助但非决定性；过度调用会让上下文膨胀，反而拉低质量。

2. **自管理记忆（Self-Managed Context）**:

    - 功能：LLM 自主维护"实验室笔记本"，记录关键技术洞察和过往错误的避免方式。
    - 核心思路：每次迭代后让 Proposer 反思本次尝试，更新笔记本——保留重要洞察、删除过时条目。后续迭代优先读取笔记本而非完整历史。相比历史记忆，上下文精简约 50%、成本降 20%、方差减半。
    - 设计动机：模仿数学家工作方式——记忆要点而非完整流水账；让 LLM 自己判断信息价值，避免硬编码启发式。

3. **多层审查防伪证**:

    - 功能：阻止借助 sorry/admit/元编程技巧伪装"证明完成"。
    - 核心思路：第一层 Lean 编译器验证代码可编译且无 sorry/admit/suggestion；第二层提取剩余 goal，确保确实没有 unclosed subgoal；第三层 LLM 审查者验证定理陈述未被篡改、是否存在过度一般化导致的循环论证。
    - 设计动机：Lean 系统的可信性最后一道防线；多层设计成本低但安全性提升显著。

### 训练策略
无训练；直接使用现成 LLM 推理。

## 实验关键数据

### 消融研究（PutnamBench 100 题子集）

| 配置 | Pass@1 (%) | Pass@20 (%) | 平均成本 | 说明 |
|------|-----------|-----------|--------|------|
| 单发 LLM（Claude Opus） | 2.0 | 5.0 | – | Baseline |
| + 迭代反馈（1 次重试） | 8.5 | 18.0 | $0.30/题 | **收益最大单一改动** |
| + 历史记忆（5 次） | 15.2 | 31.0 | $0.80/题 | 有效但上下文膨胀 |
| + 自管理记忆（5 次） | 16.3 | 33.2 | $0.64/题 | **最优权衡** |
| + 工具搜索 | 17.8 | 35.5 | $0.72/题 | 边际收益 ~8% |

### 模型对比（完整系统，50 iter）

| 模型 | Pass@1 | Pass@50 | 相对成本 |
|------|--------|---------|--------|
| Claude Sonnet 4.5（10k thinking） | 28.5% | 51.3% | 0.8x |
| Claude Opus 4.5（10k thinking） | 38.2% | 60.7% | 1.0x |
| Claude Opus 4.5（32k thinking） | 45.1% | **68.3%** | 1.8x |
| Gemini 3 Flash（high） | 9.2% | 25.1% | 0.3x |
| Gemini 3 Pro（high） | 12.5% | 28.7% | 0.6x |

### 主基准（Opus 32k, 50 iter）

| 基准 | AxProverBase | SOTA 竞品 | 备注 |
|------|-------------|-----------|------|
| PutnamBench (pass@1) | **54.7%** | Hilbert 55.9% | 成本少 100x |
| FATE-M (pass@1) | **98.0%** | REAL-Prover 56.7% | 大幅超越 |
| FATE-H (pass@1) | **66.0%** | REAL-Prover 0% | 首个 >60% |
| FATE-X (pass@1) | 24.0% | Seed-Prover 33% | 难度极高 |
| LeanCat (pass@1) | **59.0%** | Opus 单发 8.25% | 迭代收益显著 |

### 关键发现
- **迭代反馈一招制胜**：仅加反馈环就让 pass@1 从 2% 跳到 8.5%（4.25 倍），超过其他改动的累计效果。
- **自管理记忆优于历史记忆**：相同成本下性能更好、稳定性更高，体现"精选记忆 > 全记忆"的价值。
- **模型能力被框架放大**：Opus 32k thinking 比 10k thinking pass@50 高 7.6 个百分点，更强模型在该框架下增益更大。
- **工具搜索价值有限**：竞赛数学环境下 web 搜索几乎不帮助，LeanSearch 略有助益但非关键。
- **跨域泛化**：从竞赛数学到抽象代数（FATE-M）再到范畴论（LeanCat），简单架构都适用。

## 亮点与洞察
- **极简主义的力量**：证明无需大规模训练或复杂搜索，只需"编译反馈 + 自反思 + 强模型"就能匹敌 SOTA。
- **自我反思的有效性**：让 LLM 维护自己的笔记本胜过固定启发式信息检索，提示 AI 系统的"元认知"价值。
- **消融设计严谨**：自底向上层层堆叠，每层有明确贡献量化，为后续改进提供方向。
- **成本-性能新视角**：$12.6/题 的成本对比 Hilbert 的成百上千美元，显著降低普及门槛。

## 局限与展望
- 在最难题集 FATE-X 上 24% 表明系统对深层数学直觉仍有瓶颈。
- 仅评估单一模型族（Claude），不同架构表现未测。
- Lean 4 特定，对 Coq/Isabelle 的迁移性需要验证。
- 自管理记忆依赖模型自省能力，对弱模型可能失效。
- 改进方向：增强语义+符号的混合检索；融入专用几何/代数求解器；先草图后形式化的双阶段范式。

## 相关工作与启发
- **vs Seed-Prover / Goedel-Prover**：后者依赖大规模合成数据 + RL；本文证明只用通用 LLM 也能竞争。
- **vs AlphaProof**：AlphaProof 用树搜索 + 复杂启发式；本文用线性迭代程序，简洁但仍有竞争力。
- **启发**：迭代反馈 + 自反思 + 轻工具的范式可迁移到程序合成、科学验证等其他复杂推理任务。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 单个模块新颖性不高，但"极简即强"的实验结论本身具有启发性。
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 5 个基准、多模型、详细消融，覆盖广泛。
- 写作质量: ⭐⭐⭐⭐⭐ 架构清晰，伪代码完整，结果表述精确。
- 价值: ⭐⭐⭐⭐⭐ 降低形式数学 AI 门槛，对开源社区影响显著。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ICML 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](../../ICML2025/llm_agent/theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[ACL 2025\] An Empirical Study on LLM-based Agents for Automated Bug Fixing](../../ACL2025/llm_agent/an_empirical_study_on_llm-based_agents_for_automated_bug_fixing.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)

</div>

<!-- RELATED:END -->
