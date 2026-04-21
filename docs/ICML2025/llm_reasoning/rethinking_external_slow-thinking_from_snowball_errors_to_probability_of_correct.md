---
title: >-
  [论文解读] Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning
description: >-
  [ICML 2025][LLM推理][Test-time scaling] 本文从信息论视角系统分析了 LLM 推理中的"雪球误差"现象，建立了雪球误差与推理正确概率之间的理论联系，证明了外部慢思考方法（如 BoN、MCTS）本质上是通过扩展搜索宽度来缓解误差累积，并在理论和实验上证明了方法效果主要取决于总推理代价和奖励函数可靠性，而非搜索框架本身。
tags:
  - ICML 2025
  - LLM推理
  - Test-time scaling
  - 雪球误差
  - information theory
  - Best-of-N
  - MCTS
---

# Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning

**会议**: ICML 2025  
**arXiv**: [2501.15602](https://arxiv.org/abs/2501.15602)  
**代码**: [ZyGan1999/Snowball-Errors-and-Probability](https://github.com/ZyGan1999/Snowball-Errors-and-Probability)  
**领域**: LLM推理  
**关键词**: Test-time scaling, 雪球误差, information theory, Best-of-N, MCTS

## 一句话总结
本文从信息论视角系统分析了 LLM 推理中的"雪球误差"现象，建立了雪球误差与推理正确概率之间的理论联系，证明了外部慢思考方法（如 BoN、MCTS）本质上是通过扩展搜索宽度来缓解误差累积，并在理论和实验上证明了方法效果主要取决于总推理代价和奖励函数可靠性，而非搜索框架本身。

## 研究背景与动机
**领域现状**：Test-time scaling（慢思考）已被证明能增强 LLM 的多步推理能力，OpenAI o1、DeepSeek R1、QwQ 等模型展示了通过延长推理时间提升质量的可行性。

**现有痛点**：尽管外部慢思考方法被广泛使用，但其有效性的底层机制仍然知之甚少，导致设计更先进高效策略时缺乏理论指导。

**核心矛盾**：复杂的慢思考技术（如 MCTS）在实际应用中往往需要巨大计算资源才能取得有限成功，因为难以优化设计选择和超参数，经常导致次优性能。简单方法（如 BoN）在相当的计算预算下能否匹配复杂方法？

**本文目标**：从理论角度解释外部慢思考方法的工作机制，建立统一的分析框架。

**切入角度**：利用信息论中的互信息和 Fano 不等式，将 LLM 推理中的雪球误差与推理错误概率建立数学联系。

**核心 idea**：外部慢思考的效果并非由搜索框架决定，而是由搜索范围和奖励模型的可靠性决定；扩大搜索范围或提升模型内在推理能力才是长期改进的方向。

## 方法详解

### 整体框架
论文提出一个基于信息论的系统框架，分四步展开：(1) 定义和量化推理中的雪球误差；(2) 建立雪球误差与推理错误概率的理论联系；(3) 分析外部慢思考在实际场景中的正确推理概率；(4) 对比不同慢思考策略（BoN vs. MCTS）的理论效果和计算成本。

### 关键设计

1. **雪球误差的信息论建模**:

    - 功能：用互信息（MI）量化 LLM 隐式推理序列 $\bm{t}$ 和可观测响应序列 $\bm{r}$ 之间的信息差距
    - 核心思路：将推理过程类比为柏拉图洞穴寓言——LLM 的输出只是内部推理的"影子"。每步推理都存在信息损失 $\text{InfoLoss}(r_l) = H(t_l | r_l)$，雪球误差即为累积信息损失 $H_{<l}(\bm{t}|\bm{r}) = \sum_i^{l-1} H(t_i | r_i)$
    - 设计动机：传统方法从 token 级别分析误差累积，但推理任务的误差发生在句子级别，更难刻画。通过信息论可提供统一的数学框架

2. **从雪球误差到推理错误概率的理论桥梁**:

    - 功能：证明推理错误概率的下界与累积雪球误差正相关
    - 核心思路：基于 Fano 不等式推导出定理 3.3——推理步骤 $l$ 处的错误概率下界为 $P(e_l) \geq \log^{-1}(|\mathcal{T}_l|-1)[H_{<l}(\bm{t}|\bm{r})/(l-1) - H_b(e_l)]$
    - 设计动机：建立雪球误差与实际推理错误之间的定量联系，为后续分析慢思考方法提供理论基础
    - 关键发现：当雪球误差超线性增长时，推理错误概率的下界随推理长度增加而上升

3. **τ-正确推理概率的实际建模**:

    - 功能：定义 τ-正确步骤和 τ-正确推理的概念，建模实际场景中的推理正确概率
    - 核心思路：假设每步生成正确步骤的概率服从指数衰减 $\Pr[|φ(r_l) - φ(r_l^*)| \leq τ] = \min(λ_τ e^{-l}, 1)$，推导出完整推理正确概率的上界为 $λ_τ^L e^{-L(L+1)/2}$
    - 设计动机：将理论分析与实际推理质量评估（如 reward model 打分）对接

4. **宽度扩展方法的统一分析**:

    - 功能：统一分析 beam search 类宽度扩展方法对推理正确概率的影响
    - 核心思路：将外部慢思考分解为"生成"和"选择"两步，总概率 $\Pr[\psi(\mathcal{R}) \leq \tau] = \Pr(\tau_{\text{generate}}) \times \Pr(\tau_{\text{select}})$
    - 关键定理（4.6）：宽度扩展方法的推理正确概率上界为 $\epsilon_b^L k^L \lambda_\tau^L e^{-L(L+1)/2}$，其中 $k$ 为每层采样数，$\epsilon_b$ 为选择正确性概率
    - 设计动机：揭示扩展搜索宽度可以提升生成概率，但选择负担也随之增加，两者存在权衡

5. **BoN vs. MCTS 理论对比**:

    - 功能：在理论框架下严格对比 BoN 和 MCTS 的推理正确概率和计算成本
    - 核心结论：BoN 达到与 MCTS 相当正确率所需的 N 值：最佳情况 $O(b)$，最坏情况 $O(b^{L/2})$
    - 计算成本对比：最佳情况下 BoN 和 MCTS 计算成本渐近等价 $O(bL)$；最坏情况下 BoN 为 $O(Lb^{L/2})$ vs. MCTS 为 $O(b^L)$，当 $L$ 增大时 BoN 甚至更便宜
    - 核心洞察：两种方法在相当正确率下计算成本相近，框架设计不是决定因素

### 损失函数 / 训练策略
本文是理论分析工作，不涉及训练。

## 实验关键数据

### 主实验（雪球误差验证）

| 模型 | 数据集 | MI 衰减趋势 | Reward 趋势 | 说明 |
|------|--------|------------|-------------|------|
| Llama-3.1-8B-Instruct | GSM8k | 近指数衰减 | 随长度下降 | 验证了雪球误差存在 |
| Qwen2.5-7B-Instruct | GSM8k | 近指数衰减 | 随长度下降 | 在不同模型上一致 |
| Skywork-o1-Open-8B | GSM8k | 近指数衰减 | 随长度下降 | o1 类模型也存在 |

### BoN vs. MCTS 实验对比

| 任务 | MCTS 准确率 | BoN (N=N̄_res) | BoN (N=N̄_call) | 说明 |
|------|------------|---------------|----------------|------|
| GSM8k (ORM Max) | ~84% | ~82% | ~85% | BoN 在合理 N 范围匹配MCTS |
| GSM8k (ORM Vote) | ~84% | ~83% | ~85% | 投票策略略好 |
| PrOntoQA (Self-Consistency) | ~95% | ~90% | ~90% | 二分类任务无reward model无法提升 |
| PrOntoQA (ORM Max) | ~95% | ~94% | ~96% | 有 RM 引导后匹配 MCTS |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| BoN + Self-Consistency | 有上限 | 二分类任务中增大N无用 |
| BoN + ORM Vote | 随N提升 | 需要可靠的奖励模型 |
| BoN + ORM Max | 随N提升 | 与 MCTS 在相当成本下效果接近 |
| 不同模型大小 | MI衰减一致 | 雪球误差是普遍现象 |
| 不同难度级别 | 更难问题MI衰减更快 | 难题的雪球效应更严重 |

### 关键发现
- 互信息以近指数速率衰减，远快于线性衰减，验证了雪球误差的存在
- BoN 在配备可靠奖励模型时，可以用与 MCTS 相当的计算成本达到相似准确率
- 慢思考方法的效果主要取决于总推理代价和奖励函数可靠性，而非搜索框架设计
- 奖励模型的选择准确性 $\epsilon_b$ 需满足 $\epsilon_b > 1/k$ 才能保证慢思考有收益
- 更难的推理问题中 MI 衰减更快，雪球效应更严重

## 亮点与洞察
- 用柏拉图洞穴寓言类比 LLM 推理中输出与真实推理的关系，直觉性强
- 首次从信息论角度给出雪球误差的形式化定义和推理错误概率下界
- 理论证明了 BoN 和 MCTS 在理想条件下的推理正确概率和计算成本是可比的，具有实际指导意义
- 结论非常实用：与其设计复杂搜索框架，不如优化奖励函数或提升模型基础推理能力
- 理论分析具有普适性，不限于特定慢思考方法

## 局限与展望
- 核心命题 4.3（推理错误概率指数衰减假设）是假设而非严格证明，虽然论文在附录中说明了更宽松假设下结论仍成立
- 实验验证主要在 GSM8k 和 PrOntoQA 上，推理步骤较短；更长推理链（如数学竞赛题）的验证不足
- MI 的估计方法依赖代理指标，直接测量隐式推理与输出的互信息本身存在困难
- 未讨论内部慢思考（如 o1 的训练策略）与外部慢思考的关系
- 理论框架假设各步推理错误概率独立，实际中可能存在依赖

## 相关工作与启发
- 信息论在 LLM 分析中的应用：Ton et al. (2024) 用信息论量化推理误差，本文在此基础上进一步建立了与慢思考方法的联系
- BoN vs. MCTS 的实践启示：在算力有限的场景下，简单的 BoN + 可靠 RM 可能比复杂的 MCTS 更实用
- 对 reward model 重要性的强调值得关注：RM 的可靠性是慢思考方法成败的关键

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails](../../ACL2025/llm_reasoning/thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)
- [Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](../../ACL2025/llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [Controlling Thinking Speed in Reasoning Models](../../NeurIPS2025/llm_reasoning/controlling_thinking_speed_in_reasoning_models.md)
- [Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](../../ACL2025/llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](../../NeurIPS2025/llm_reasoning/does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)

<!-- RELATED:END -->
