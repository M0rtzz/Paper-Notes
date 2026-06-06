---
title: >-
  [论文解读] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information
description: >-
  [ICML 2026][多智能体][LLM聚合] 本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票…
tags:
  - "ICML 2026"
  - "多智能体"
  - "LLM聚合"
  - "多智能体推理"
  - "信息聚合"
  - "贝叶斯最优"
  - "无监督标注"
---

# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information

**会议**: ICML 2026  
**arXiv**: [2510.01499](https://arxiv.org/abs/2510.01499)  
**代码**: 无  
**领域**: 多智能体  
**关键词**: LLM聚合, 多智能体推理, 信息聚合, 贝叶斯最优, 无监督标注  

## 一句话总结

本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票，并在 UltraFeedback、MMLU 和医疗健康数据集上验证了一致的提升。

## 研究背景与动机

**领域现状**：多智能体 LLM 推理（如 LLM debate、LLM council）已广泛使用，在聚合多个模型回答时，绝大多数工作直接采用多数投票 (Majority Voting, MV) 作为标准聚合策略。

**现有痛点**：MV 是一种"零阶"方法，仅依赖原始回答的频次，完全忽略了不同 LLM 之间的能力异质性（有的模型准确率 90%、有的只有 60%）和回答之间的相关性。这意味着一个弱模型和一个强模型在投票中拥有相同权重，且当多个弱模型犯相同错误时容易误导最终结果。

**核心矛盾**：要利用模型准确率（一阶信息）来加权需要大量带标签数据来估计准确率，但在自动标注、预测市场等无监督场景中根本没有标签；而经典的 Surprisingly Popular (SP) 方法虽然利用了二阶相关性信息且不需要标签，但在 LLM 场景下反而不如 MV——因为 LLM 不像人类群体那样存在系统性偏差，SP 所利用的信号反而起了反效果。

**本文目标**：设计不依赖标签、利用高阶信息的聚合算法，在理论上可证地优于 MV。

**切入角度**：作者观察到随机打乱选项顺序后，联合分布具备对称结构（所有错误选项等概率），这种结构允许推导出闭式最优解。同时 SP 失败的原因在于其"放大偏差"的方向恰好与 LLM 场景相反，反转 SP 的计算方向即可纠正。

**核心 idea**：用准确率的逆 sigmoid 函数作为最优权重实现贝叶斯最优聚合；在没有标签时，将 SP 的预测方向反转（从"反事实"视角计算分数）来利用二阶信息超越 MV。

## 方法详解

### 整体框架

给定 $N$ 个 LLM 对 $K$ 选项的选择题的回答 $a_1, \ldots, a_N$，系统首先对每个问题随机打乱选项顺序（预处理步骤），确保联合分布具备对称性。然后根据可用的信息层级选择不同的聚合器：有准确率时用 OW，无标签时用 ISP 或通过 ISP 估计准确率后再用 OW（OW-I / OW-L）。最终通过逆置换映射回原始选项顺序输出结果。

### 关键设计

1. **Optimal Weight (OW) — 一阶最优聚合**:

    - 功能：在已知各 LLM 准确率 $x_i$ 时，计算贝叶斯最优的加权投票
    - 核心思路：为第 $i$ 个 agent 赋予权重 $\omega_i = \sigma_K^{-1}(x_i)$，其中 $\sigma_K(x) = e^x / (K-1+e^x)$ 是广义 sigmoid 函数。聚合公式为 $f_{OW} = \arg\max_s \sum_i \sigma_K^{-1}(x_i) \cdot \mathbb{1}\{a_i = s\}$。当 $K=2$ 时退化为 logistic 函数的逆，与 Bradley-Terry 模型建立了理论联系
    - 设计动机：在随机打乱引出的对称信息结构下，这组权重恰好是后验概率最大化的充要解，是所有聚合器（不限于线性加权）中的贝叶斯最优。当所有 agent 同质时，OW 退化为 MV，说明 MV 仅在 self-consistency 采样等场景下才最优

2. **Inverse Surprising Popularity (ISP) — 二阶无标签聚合**:

    - 功能：仅利用 agent 间回答的条件概率 $\mathbb{P}(A_i|A_j)$ 来聚合，不需要任何标签
    - 核心思路：与经典 SP 计算每个 agent "预测别人会怎么回答"不同，ISP 计算的是"如果别的 agent 给了反事实回答，我会怎么预测"。具体地，ISP 分数为 $S_{ISP}(s,i) = \frac{1}{N-1}\sum_{j \neq i} \frac{1}{K-1}\sum_{a \neq a_j} \mathbb{P}(A_i=s|A_j=a)$，然后选择优势函数 $Adv_{ISP}(s) = \sum_i \mathbb{1}\{a_i=s\} - \sum_i S_{ISP}(s,i)$ 最大的选项
    - 设计动机：经典 SP 在 LLM 场景下不如 MV，因为 LLM 没有人类群体那种系统性低估正确答案的偏差。ISP 反转了预测方向——用反事实条件概率代替真实条件概率——使预测分数偏向错误方向从而放大正确答案的优势。理论证明 $\mathbb{E}[Adv_{ISP}(s^*)] \geq \mathbb{E}[Adv_{MV}(s^*)] \geq \mathbb{E}[Adv_{SP}(s^*)]$

3. **OW-L / OW-I — 从二阶信息估计一阶权重**:

    - 功能：在无标签场景下近似获得 OW 所需的准确率参数
    - 核心思路：OW-L 通过最小化经验条件概率与理论条件概率之间的均方误差来反解准确率 $\hat{x}_1, \ldots, \hat{x}_N$；OW-I 则用 ISP 的聚合结果作为伪标签，直接统计每个 agent 与伪标签的一致率作为准确率估计。两者估计出准确率后都代入 OW 的权重公式 $\sigma_K^{-1}(\hat{x}_i)$ 进行聚合
    - 设计动机：OW 虽然是贝叶斯最优，但需要标签来估计准确率。这两种方法将二阶信息"桥接"到一阶最优框架，实验表明两者在实际数据上表现高度一致，且均优于直接使用 ISP

## 实验关键数据

### 主实验（模拟数据）

| 方法 | $K=2$ | $K=4$ | $K=6$ | $K=8$ | $K=10$ |
|------|-------|-------|-------|-------|--------|
| MV | 85.13% | 92.64% | 94.22% | 94.85% | 95.54% |
| SP | 79.94% | 90.52% | 92.68% | 93.66% | 94.40% |
| Single Best | 90.34% | 89.94% | 90.31% | 89.95% | 90.05% |
| **ISP (本文)** | **90.48%** | **94.45%** | **95.78%** | **96.23%** | **96.49%** |
| OPT (clairvoyant) | 91.37% | 94.94% | 96.05% | 96.46% | 96.81% |

### 真实数据集实验（4 个强模型）

| 方法 | UltraFeedback | MMLU | ARMMAN |
|------|--------------|------|--------|
| MV | 72.21% | 89.32% | 85.24% |
| ISP | 73.26% | 90.01% | 85.78% |
| **OW-L** | **73.66%** | **90.37%** | **85.78%** |
| **OW-I** | **73.66%** | **90.37%** | **85.78%** |
| Single Best (oracle) | 73.14% | 91.02% | 85.32% |

### 关键发现

- ISP 在所有 $K$ 值下均优于 MV，且两者差距随 $K$ 增大而缩小（$\Theta(1/K)$），与理论预测一致
- 在 16 种模型组合中，OW-L 在 97.92% 的情况下优于 MV，绝对提升最高达 14.20%；MV 在所有组合中从未取得最佳
- 假设检验 t-statistic 分别为 12.53（UltraFeedback）、23.39（MMLU）和 3.22（ARMMAN），p-value 均 < 0.001，提升在统计上显著
- 在"强干扰项"子集（MMLU-hard，至少两个模型选了相同错误选项）上，OW-L/OW-I 比 MV 提升超过 7%（17.23% → 24.79%），说明高阶信息在困难场景下更有价值

## 亮点与洞察

- **逆 sigmoid 权重的贝叶斯最优性**：看似简单的加权方案 $\omega_i = \sigma_K^{-1}(x_i)$ 实际上是所有可能聚合器中的最优解（不限于线性），这为 Bradley-Terry 模型在 RLHF 中的使用提供了理论背书。这个结论非常优雅且可直接应用
- **反转 SP 的反直觉设计**：经典 SP 在人类群体中有效但在 LLM 中失败，作者深入分析了原因（LLM 缺少人类的系统性偏差），然后将方向反转得到 ISP，这种"诊断失败原因→针对性修改"的思路值得借鉴
- **二阶信息桥接一阶最优**：OW-L/OW-I 将"无标签可用的二阶信息"转化为"需要标签的一阶最优权重"，这种间接利用信息层级的思路可以迁移到其他无监督聚合场景

## 局限与展望

- 理论分析依赖条件独立假设（给定正确答案后各 LLM 独立），虽然实验表明在违反假设时仍有效，但缺乏正式的鲁棒性界
- 所有模型对同一问题使用相同的全局权重，未考虑不同问题类型上模型能力的差异（如某模型擅长数学但不擅长语言理解），prompt-specific 权重是明确的改进方向
- 仅处理封闭选择题（$K$ 个选项），对开放式生成任务的扩展尚不明确
- 位置偏差的理论假设（LLM 不受选项顺序影响）在弱模型上不完全成立，虽然实验中不需要去偏就有效，但对弱模型的理论保证有待加强

## 相关工作与启发

- **Surprising Popularity (Prelec et al., 2017)**：经典的基于二阶信息的聚合方法，但作者证明在 LLM 场景下 SP 劣于 MV，ISP 是对 SP 的针对性改进
- **Bradley-Terry 模型**：Corollary 3.2 建立了 OW 与 BT 模型的联系，为 RLHF 中 BT 模型的有效性提供了理论支持
- **Self-Consistency (Wang et al., 2022)**：Corollary 3.3 证明了当 agent 同质时 MV 即最优，即 self-consistency 场景下无需更复杂的聚合
- **启发**：该框架可直接作为多 LLM 系统中 MV 的 drop-in 替代品，且计算开销仅为 CPU 级别的几秒钟，适用于 API 调用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](../../ACL2026/multi_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](../../ACL2025/multi_agent/beyond_frameworks_multi_agent_collaboration.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)

</div>

<!-- RELATED:END -->
