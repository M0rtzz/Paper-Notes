---
title: >-
  [论文解读] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation
description: >-
  [ICML 2025][代码智能][过程监督] 提出 ORPS（Outcome-Refining Process Supervision），通过将代码执行反馈与 LLM 自我批评结合，在树状搜索框架中统一过程奖励与结果奖励，无需训练 PRM 即可在代码生成中实现 26.9% 的正确率提升和 42.2% 的效率提升。
tags:
  - "ICML 2025"
  - "代码智能"
  - "过程监督"
  - "结果监督"
  - "代码生成"
  - "奖励模型"
  - "树搜索"
---

# Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation

**会议**: ICML 2025  
**arXiv**: [2412.15118](https://arxiv.org/abs/2412.15118)  
**代码**: [github.com/zhuohaoyu/ORPS](https://github.com/zhuohaoyu/ORPS)  
**领域**: 代码智能  
**关键词**: 过程监督, 结果监督, 代码生成, 奖励模型, 树搜索

## 一句话总结

提出 ORPS（Outcome-Refining Process Supervision），通过将代码执行反馈与 LLM 自我批评结合，在树状搜索框架中统一过程奖励与结果奖励，无需训练 PRM 即可在代码生成中实现 26.9% 的正确率提升和 42.2% 的效率提升。

## 研究背景与动机

LLM 在代码生成上已取得显著进展，但面对复杂编程任务（如动态规划、并行计算优化）仍存在困难。现有监督范式存在两大瓶颈：

**结果监督（Outcome Supervision）**：仅评估最终输出，忽略推理过程。一个通过所有测试但用 O(n²) 的暴力解法会被标记为"正确"，无法引导模型走向 O(n log n) 的最优解。

**过程监督（Process Supervision）**：依赖训练好的 Process Reward Model (PRM)，存在三大问题：(a) 需要昂贵的人工标注; (b) PRM 本身可能幻觉或被 reward hacking; (c) 针对代码的 PRM 极其稀缺，数学领域的 PRM 不适用于编程逻辑。

**核心洞察**：代码天然具备可验证性——可以执行并获得客观反馈。现有的执行反馈方法（Reflexion、LDB、Self-Repair）主要聚焦局部代码修复，缺少对不同算法策略的全局探索能力。作者提出的关键问题是：**当 LLM 的内在推理能力可以被可验证的执行结果有效引导时，是否还需要单独训练 PRM？**

## 方法详解

### 整体框架

ORPS 将推理、代码实现和执行验证融合到一个**树结构搜索过程**中。框架核心思想是：将"关于结果改进的推理"本身作为需要监督的过程。与逐步修复代码的方法不同，ORPS 在更高抽象层面引导 LLM 思考整体解题策略。

搜索过程中，每个节点的状态定义为六元组 $s_t = (\mathcal{R}_t, C_t, F_t, \omega_t, K_t, \rho_t)$，分别对应：推理链、代码实现、执行反馈、结果奖励分数、自我批评推理、过程奖励分数。

搜索分三个阶段循环进行：候选生成 → 执行与性能分析 → 自我批评与过程打分，通过 beam search 保留 Top-K 最优路径。

### 关键设计

1. **候选生成（Candidate Generation）**：对搜索树中每个节点，LLM 基于历史推理链 $\mathcal{R}_{t-1}$、代码 $C_{t-1}$ 和执行反馈 $F_{t-1}$ 生成 N 个候选方案。每个候选包含推理更新 $r_t^{(j)}$（如"调整循环终止条件避免 off-by-one 错误"）和对应代码 $c_t^{(j)}$。这种设计确保了**算法多样性**——可同时保留贪心和动态规划两种方案，直到执行反馈明确哪个更优。

2. **执行与性能分析结果奖励（Execution & Profiling Outcome Rewards）**：对每个候选代码在单元测试上执行并收集性能指标。结果奖励 $\omega_t^{(j)} = \sum_{k=1}^{M} \beta_k \cdot \text{normalize}(m_k^{(j)})$，综合了**动态分析指标**（正确性、执行时间、CPU 指令数、缺页次数）和**静态分析指标**（代码长度、AST 节点数、圈复杂度、认知复杂度）。这确保暴力实现即使通过全部测试也会因复杂度指标低而被激励优化。

3. **自我批评与过程打分（Self-Critic & Process Rewarding）**：同一个 LLM 同时扮演"程序员"和"评审员"。执行代码后，模型生成文本性批评 $k_t^{(j)}$ 和数值过程奖励 $\rho_t^{(j)}$。这种混合打分机制防止了 reward hacking——模型无法在不真正提升可验证执行结果的情况下膨胀奖励。若 $\rho$ 高但 $\omega$ 低，则标志着过程-结果不匹配，这是传统过程监督的关键失败模式。

### 损失函数 / 训练策略

ORPS 是**纯推理时框架**，无需训练。Beam search 使用加权步分数选择 Top-K 后继状态：

$$q_t = \alpha \rho_t + \beta \omega_t, \quad \alpha + \beta = 1$$

其中 $\alpha = \beta = 0.5$（默认设置）。该公式统一了传统监督范式：$\beta = 0$ 退化为纯过程监督；$\alpha = 0$ 退化为结果监督（类似树上的 Best-of-N）。

搜索超参数：搜索深度 T=5，beam 宽度 K=3，扩展因子 N=20。LLM 调用次数为 $2 \times N \times (K \times T + 1)$。

**双向反馈协同**：执行结果通过识别预期行为与实际行为的差异来锚定推理过程（如测试失败揭示递归算法中有缺陷的基例）；过程奖励引导探索朝向算法更优的实现（如识别出记忆化可将 O(2^n) 暴力解转化为 O(n) 动态规划解）。

## 实验关键数据

### 主实验

在 3 个基准（LBPP 竞赛级、HumanEval、MBPP）和 5 个模型上评估。以下选取代表性结果：

| 模型/方法 | LBPP Pass@1↑ | LBPP Time↓ | HumanEval Pass@1↑ | MBPP Pass@1↑ |
|-----------|-------------|-----------|-------------------|-------------|
| Qwen-7B CoT | 40.1% | 118.6% | 72.6% | 79.0% |
| Qwen-7B Reflexion | 37.7% | 111.2% | 75.6% | 79.0% |
| Qwen-7B LDB (w/T) | 35.8% | 187.8% | 87.8% | 66.9% |
| Qwen-7B BoN | 53.1% | 117.9% | 77.4% | 82.9% |
| **Qwen-7B ORPS** | **59.9%** | **84.1%** | **79.9%** | 76.7% |
| **Qwen-7B ORPS (w/T)** | **77.8%** | **82.4%** | **96.3%** | **94.9%** |
| Qwen-14B CoT | 53.7% | 119.2% | 82.9% | 84.0% |
| **Qwen-14B ORPS (w/T)** | **85.8%** | **64.2%** | **97.0%** | **95.3%** |
| GPT-4o-Mini CoT | 50.0% | 124.5% | 79.9% | 78.6% |
| **GPT-4o-Mini ORPS (w/T)** | **88.9%** | **61.6%** | **97.6%** | **95.7%** |

**关键发现**：Qwen-7B + ORPS (w/T) 在 LBPP 上达到 77.8%，超过 Qwen-14B CoT 的 53.7%，说明**推理空间比模型规模更重要**。

### 消融实验

在 LBPP 上使用 Qwen-7B 进行消融（Table 3）：

| 配置 | Pass@1 | Tests% | Valid% | Time% | 说明 |
|------|--------|--------|--------|-------|------|
| ORPS（完整） | 59.9% | 75.7% | 92.0% | 84.1% | 基线 |
| − 执行反馈 | 43.8% | 56.4% | 72.8% | 200.5% | Pass@1 下降 16.1%，证明执行反馈至关重要 |
| − 推理过程 | 55.6% | 74.5% | 94.4% | 124.5% | Pass@1 下降 4.3%，推理对效率提升更关键 |

**PRM 分析**（Table 4，LBPP + Qwen-7B）：

| 方法 | Pass@1 | 粒度 | 需训练 |
|------|--------|------|--------|
| 训练 PRM（结果级） | 37.0% | Outcome | ✓ |
| 训练 PRM（行级） | 32.1% | Line | ✓ |
| **ORPS（推理时，结果级）** | **59.9%** | Outcome | ✗ |
| ORPS（推理时，行级） | 38.3% | Line | ✗ |

**计算成本分析**（Table 5，LBPP + Qwen-7B）：

| 方法 | 20次调用 | 50次调用 | 100次调用 |
|------|---------|---------|----------|
| Reflexion | 37.0% | 40.7% | 39.5% |
| LDB | 37.0% | 36.4% | 37.0% |
| REx | 43.2% | 53.7% | 54.3% |
| PRM-GPT | 44.4% | 37.0% | 35.8% |
| **ORPS** | **48.4%** | **55.6%** | **64.2%** |

ORPS 随计算预算增加持续提升，而 REx 出现收益递减，PRM 甚至下降。

### 关键发现

1. **推理空间 > 模型规模**：ORPS 使小模型（Qwen-7B）在 LBPP 上超越大模型（Qwen-14B）的 CoT 基线
2. **无需 PRM**：结合执行反馈的自我批评比训练好的 PRM（包括人工标注的高质量 PRM）效果更好
3. **全局策略探索 > 局部修复**：LDB 在 LBPP 上使用 Qwen-7B 仅 35.8%，ORPS 达 77.8%（使用相同测试用例）
4. **CodeContests 额外验证**（附录）：ORPS 在 100 次调用下达到 20.61% Pass@1，远超 REx 的 13.33% 和 Reflexion 的 8.48%
5. **单指标优化陷阱**：仅优化单一指标（如执行速度）会导致其他维度大幅退化，验证了多目标联合优化的必要性

## 亮点与洞察

- **哲学上的优雅**：将"关于结果改进的推理"本身视为过程来监督，巧妙地统一了两种监督范式为连续谱上的端点
- **实用性强**：纯推理时方法，无需训练 PRM，无需额外标注数据，即插即用
- **深层洞察**：LLM 不能可靠地自我纠正（Huang et al. 2023），但当自我批评被可验证的执行结果锚定时，就能产生高质量的过程奖励——"锚定"是关键
- **代码独特优势的利用**：代码的可执行性提供了数学推理所没有的客观验证信号，这使得代码领域特别适合此类方法

## 局限与展望

1. **MBPP 上表现有限**：在简单任务上 ORPS 相对 BoN 改进不大，甚至略有劣势，说明复杂搜索在简单问题上可能过度开销
2. **自生成测试用例质量**：使用 gold 测试用例（w/T）和自生成测试用例的性能差距巨大（LBPP 上 77.8% vs 59.9%），测试用例质量是瓶颈
3. **推理成本**：LLM 调用次数为 $2 \times N \times (K \times T + 1)$，默认配置下约 320 次调用/问题，对大规模应用仍较昂贵
4. **仅限代码领域**：方法高度依赖可执行验证，难以直接迁移到不可验证的推理任务
5. **未探索与 RL 训练的结合**：ORPS 产生的搜索轨迹可作为高质量训练数据用于 RLHF/DPO，但论文未涉及

## 相关工作与启发

- **Reflexion / LDB / Self-Repair**：局部代码修复方法，无法探索不同算法策略
- **Math-Shepherd / AlphaMath**：数学领域的过程监督方法，依赖训练 PRM
- **REx**：将代码修复建模为 exploration-exploitation 权衡，与 ORPS 最相关但仅做局部探索
- **Test-time Scaling**（Snell et al. 2024）：ORPS 可视为代码领域高效利用推理时计算的策略
- **启发**：可尝试将 ORPS 框架扩展到有形式化验证器的领域（如定理证明、硬件设计）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 统一过程/结果监督的视角有启发性，但树搜索+执行反馈的组合并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个模型、3+1个基准、详尽消融、PRM训练对比、计算成本控制实验、案例分析
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、形式化严谨、实验组织有条理
- 价值: ⭐⭐⭐⭐ — 实际证明推理时方法可超越训练 PRM，对社区有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](../../ACL2026/code_intelligence/recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ICML 2025\] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](adaptivestep_automatically_dividing_reasoning_step_through_model_confidence.md)
- [\[ICML 2025\] EpiCoder: Encompassing Diversity and Complexity in Code Generation](epicoder_encompassing_diversity_and_complexity_in_code_generation.md)
- [\[ICML 2025\] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](../../ACL2026/code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)

</div>

<!-- RELATED:END -->
