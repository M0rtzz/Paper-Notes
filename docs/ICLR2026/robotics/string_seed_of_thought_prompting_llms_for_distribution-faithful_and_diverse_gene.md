---
title: >-
  [论文解读] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation
description: >-
  [ICLR 2026][机器人][提示学习] 本文提出 String Seed of Thought（SSoT），一种简洁的提示方法，通过指示 LLM 先生成随机字符串再从中提取随机性来选择答案，显著提升了概率指令跟随（PIF）的分布忠实度和开放式任务（DAG）的响应多样性，理论证明了 TV 距离随字符串长度指数衰减，实验表明推理型 LLM 的表现接近伪随机数生成器。
tags:
  - ICLR 2026
  - 机器人
  - 提示学习
  - probabilistic instruction following
  - diversity
  - LLM reasoning
  - randomness
---

# String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation

**会议**: ICLR 2026  
**arXiv**: [2510.21150](https://arxiv.org/abs/2510.21150)  
**代码**: 无  
**领域**: 机器人  
**关键词**: prompting, probabilistic instruction following, diversity, LLM reasoning, randomness

## 一句话总结

本文提出 String Seed of Thought（SSoT），一种简洁的提示方法，通过指示 LLM 先生成随机字符串再从中提取随机性来选择答案，显著提升了概率指令跟随（PIF）的分布忠实度和开放式任务（DAG）的响应多样性，理论证明了 TV 距离随字符串长度指数衰减，实验表明推理型 LLM 的表现接近伪随机数生成器。

## 背景与动机

1. **LLM 在概率性选择上存在系统性偏差**：LLM 擅长确定性单答案任务，但在需要按特定分布选择答案时表现不佳。例如让 LLM 模拟公平抛硬币，结果往往严重偏斜，而非接近 50-50。

2. **多种真实应用需要概率行为**：人类行为模拟、内容多样化、博弈论中的混合策略（如猜拳的纳什均衡）等场景，都要求 LLM 的经验分布与目标分布对齐，而非找到单一最优答案。

3. **响应多样性对 test-time scaling 至关重要**：生成大量候选方案再选择最优解是 test-time scaling 的核心策略，但 LLM 的输出往往坍缩到有限答案集中，限制了候选方案的多样性。

4. **现有去偏方法效果有限**：调高温度、few-shot 示例、prompt 集成等方法虽能部分缓解偏差，但在有偏分布任务上效果不稳定，且多数方法需要针对每个任务单独调整。

5. **LLM 能描述分布但不能采样**：已有研究表明 LLM 能准确描述一个概率分布，但让它们实际从该分布中采样时准确率明显滞后，存在"知道但做不到"的鸿沟。

6. **推理型 LLM 的长 CoT 提供新机遇**：deepseek-r1、QwQ-32B 等推理模型具有超长思维链，这为在推理过程中生成足够的随机性熵源提供了可能性。

## 方法详解

### 核心思想

SSoT 的设计极其简洁——只需在 prompt 中添加一条两阶段指令：

1. **生成随机字符串**：指示 LLM 输出一个随机字符串，利用这个与任务无关的简单指令产生足够的熵
2. **从字符串中提取随机性**：指示 LLM 操作该字符串（如求和取模、哈希等）来推导最终答案

对于 PIF 任务，核心指令为"生成随机字符串，操作它以从目标分布中采样"；对于 DAG 任务，指令为"生成随机字符串，操作它以生成一个多样化的响应"。

### 关键设计原理

- **偏差隔离**：直接从 prompt 选择答案会受到选项位置、标签频率等训练偏差影响；SSoT 将随机性生成与答案选择解耦，"生成随机字符串"这一与任务无关的指令不易受选择偏差影响
- **完全可并行化**：每次生成是独立的，无需维护历史记录，不同于顺序采样方法
- **统一 prompt 框架**：同一个 prompt 框架适用于所有 PIF/DAG 任务，LLM 自主选择最优策略

### 理论分析

**定理 4.1（2-universal 哈希函数界）**：假设 LLM 生成的字符串中每个字符的条件概率有界（$\delta \leq P(x_i|\{x_j\}_{j<i}) \leq 1-(A-1)\delta$），使用 2-universal 哈希函数提取随机性时，TV 距离满足：

$$d_{TV} \leq \frac{\sqrt{M}}{2\delta''} 2^{-\frac{n}{2}\log_2 \frac{1}{(1-(A-1)\delta)^2+(A-1)\delta^2}} + \sqrt{\frac{\ln((2^M-2)/\delta')}{K\phi(\pi_{P_X})}}$$

第一项随字符串长度 $n$ 指数衰减，第二项是有限样本误差。

**定理 4.2（Sum-Mod 策略界）**：当 LLM 使用求和取模策略（将字符 ASCII 码求和后对 $M$ 取模）时，TV 距离同样随字符串长度指数衰减，只要各字符的边际分布不严重偏离均匀分布。

### LLM 自主策略选择

分析 CoT 发现 LLM 会根据任务复杂度自主选择策略：
- **Sum-Mod**：对均匀分布任务使用简单的 ASCII 码求和取模
- **Rolling Hash**：对有偏分布任务自动切换到更复杂的多项式哈希（$\sum_i B^i \text{ord}(c_i)$ 后取模）
- **DAG 任务**：创造性类别自动采用模板+局部采样策略，其他类别用列表+全局采样

## 实验结果

### PIF 性能：5 个前沿 LLM 的系统评估

| 模型 | 方法 | 2-choice | Biased 2-choice | 3-choice | Biased 3-choice | Biased 9-choice |
|------|------|:-:|:-:|:-:|:-:|:-:|
| deepseek-v3 | Baseline | 5.97 | 111.45 | 136.03 | 117.28 | 297.33 |
| deepseek-v3 | **SSoT** | **2.91** (↓51%) | **3.54** (↓97%) | **15.33** (↓89%) | **15.65** (↓87%) | **44.90** (↓85%) |
| deepseek-r1 | Baseline | 36.09 | 69.58 | 106.30 | 49.53 | 138.21 |
| deepseek-r1 | **SSoT** | **3.03** (↓92%) | **1.51** (↓98%) | **4.98** (↓95%) | **4.30** (↓91%) | **18.06** (↓87%) |
| QwQ-32B | **SSoT** | 3.39 | **2.47** (↓98%) | **1.82** (↓98%) | **1.30** (↓99%) | **11.48** (↓96%) |
| PRNG（理想） | — | 1.85 | 1.93 | 3.36 | 2.85 | 13.72 |

（JS 散度 ×10³，越低越好）

**关键发现**：deepseek-r1 和 QwQ-32B 使用 SSoT 后的 JS 散度接近伪随机数生成器（PRNG），特别是 QwQ-32B 在 Biased 3-choice 上 JS 散度仅 1.30，甚至优于 PRNG 的 2.85。

### DAG 性能与对抗博弈

| 方法 | NoveltyBench Overall (Distinct / Utility) |
|------|:-:|
| Baseline | 4.70 / 5.17 |
| Paraphrase | 5.63 / 5.57 |
| T=1.0 | 5.57 / 6.03 |
| **SSoT** | **6.19** / 5.92 |

SSoT 在 Distinct 分数上最高（6.19），且在 Creativity 类别上同时提升了 Distinct 和 Utility。WildChat 数据集上 SSoT 的 Distinct 从 3.39 提升到 5.25（+55%）。

**猜拳对抗实验**：SSoT 使 LLM 在面对 10 个"黑带"猜拳机器人时平均得分接近零（理想的混合策略均衡），而 Baseline 和 Simple prompt 均被机器人系统性击败。

### CoT Scaling 分析

使用 budget forcing 控制推理长度发现：
- 随着 thinking token 增加，生成整数的均匀性显著改善（JS 散度持续下降）
- 即使在 T=0（完全确定性解码）下，更长的推理链也能生成更高复杂度的字符串（Lempel-Ziv 复杂度和 zlib 压缩率均随推理长度增长）

## 亮点与创新

- **极致简洁**：仅需在 prompt 中添加一条指令即可大幅改善概率行为，无需训练或外部工具
- **理论与实践统一**：严格证明了 TV 距离的收敛保证，且实验结果与理论预测高度吻合
- **LLM 自主策略选择**：揭示了 LLM 能根据任务复杂度自动发明合适的随机性提取策略（Sum-Mod vs Rolling Hash）
- **推理长度 scaling law**：首次证明 PIF 性能随 CoT 长度 scaling，为推理模型的概率能力提供了新的理解维度

## 局限性

- **依赖模型推理能力**：8B 以下小模型可能无法正确执行取模/哈希等算术操作，导致效果不佳
- **偏差传播风险**：若 LLM 生成的随机字符串具有强位置偏差且采用"懒惰"策略（仅用首字符），输出分布将有偏
- **不适用于单答案任务**：SSoT 专为多有效答案或概率需求场景设计，应用于数学/事实检索等单答案任务可能分散模型注意力
- **推理开销增加**：生成随机字符串和执行字符串操作会增加 CoT 长度和推理成本

## 相关工作对比

### vs. Prompt Ensemble（提示集成）
Prompt Ensemble 使用 50 个改写 prompt + 随机化选项顺序来减少位置偏差。在均匀分布 PIF 上表现良好，但在有偏分布上明显退化——因为仅消除位置偏差不足以实现精确的分布对齐。SSoT 在均匀和有偏设置上均接近 PRNG 理想性能，展现出对分布偏斜的强鲁棒性。

### vs. Few-shot Examples（少样本示例）
Few-shot 方法提供 k 个按目标分布采样的示例（k=3/10/50），期望 LLM 通过上下文学习校准输出分布。但实验表明 few-shot 在 action 数量增加时效果迅速下降（特别是 biased 设置），而 SSoT 在 2 到 64 个选项范围内保持一致的低 JS 散度，可扩展性远优于 few-shot。

### vs. Sequential Sampling（顺序采样）
Sequential Sampling 将历史选择记录加入 prompt 中，期望 LLM 根据已选分布调整后续选择。该方法破坏了生成间的独立性，无法并行化，且在长序列后 prompt 膨胀严重。SSoT 每次生成完全独立，天然支持并行采样。

## 评分

- ⭐⭐⭐⭐⭐ 新颖性：将"先生成随机字符串再提取随机性"作为 prompt 策略极具创意，开辟了 LLM 概率行为研究新方向
- ⭐⭐⭐⭐ 技术质量：理论分析严谨（两个定理），实验覆盖 5 个模型、多种任务设置和对抗场景
- ⭐⭐⭐⭐ 实用性：零成本即可部署，适用于游戏、模拟、内容多样化等多种场景
- ⭐⭐⭐⭐ 写作质量：结构清晰，理论-实验-分析层层递进，CoT 策略分析尤为精彩

<!-- RELATED:START -->

## 相关论文

- [When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)
- [Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)

<!-- RELATED:END -->
