---
title: >-
  [论文解读] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization
description: >-
  [ICML 2025 (Workshop on ML4Wireless)][大语言模型] 提出基于 LLM 上下文学习（In-context Learning）的基站功率控制算法，通过自然语言任务描述和经验池驱动的示例选择，在不更新模型参数的条件下达到接近传统深度强化学习的性能。
tags:
  - ICML 2025 (Workshop on ML4Wireless)
  - 大语言模型
  - 上下文学习
  - 无线网络优化
  - 功率控制
  - 经验池
---

# Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization

**会议**: ICML 2025 (Workshop on ML4Wireless)  
**arXiv**: [2408.00214](https://arxiv.org/abs/2408.00214)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 大语言模型, 上下文学习, 无线网络优化, 功率控制, 经验池

## 一句话总结

提出基于 LLM 上下文学习（In-context Learning）的基站功率控制算法，通过自然语言任务描述和经验池驱动的示例选择，在不更新模型参数的条件下达到接近传统深度强化学习的性能。

## 研究背景与动机

6G 网络日益复杂，传统网络优化方法面临两大瓶颈：

**凸优化方法**：需要针对每个任务做专门的问题建模，将目标函数或约束转化为凸形式，泛化性差

**机器学习方法**（如强化学习）：虽然对问题建模要求较低，但模型训练和超参数调优过程冗长，需要大量迭代

LLM 的出现为网络优化提供了新的思路。In-context learning 具有两个核心优势：
- **免训练**：依赖 LLM 推理能力，避免参数更新的计算开销
- **自然语言接口**：运营商可以用人类语言直接描述优化任务

然而，既有 LLM 网络优化研究大多是静态场景或简单反馈机制，缺乏对动态环境下经验积累和示例选择的系统设计。本文以基站（BS）发射功率控制为案例，提出了一套完整的 LLM 上下文学习优化框架。

## 方法详解

### 整体框架

整体流程包含四个核心模块：

1. **自然语言任务描述**（Task Description）：用结构化语言描述优化目标、环境定义和回复规则
2. **经验池**（Experience Pool）：收集 LLM 的历史决策经验 $E = \{s, a, r(s, a)\}$
3. **示例选择**（Example Selection）：根据当前状态从经验池中挑选最有参考价值的示例
4. **LLM 推理**：将任务描述 + 选定示例 + 当前状态组合为 prompt，由 LLM 输出功率决策

In-context learning 的核心形式化定义为：

$$D_{task} \times \mathcal{E}_t \times s_t \times \mathcal{LLM} \Rightarrow a_t$$

其中 $D_{task}$ 是任务描述，$\mathcal{E}_t$ 是 $t$ 时刻的示例集合，$s_t$ 是当前环境状态，$a_t$ 是 LLM 输出的决策。

### 关键设计

#### 1. 自然语言任务描述模板

任务描述分三个层次：

- **Task_goal**：指定"基站功率控制的决策任务"，目标是"在 4 个功率等级中选择"
- **Task_definition**：引入需要考虑的环境状态（如用户数量），并嵌入示例集 $\mathcal{E}_t$
- **Rules**：设定回复规则，如"根据以上示例，从 level 1-4 中选择"

这种模板设计具有通用性，可扩展到其他网络优化任务。

#### 2. 经验池与示例设计

每个示例定义为三元组 $E = \{s, a, r(s, a)\}$，其中 $s$ 是环境状态，$a$ 是决策动作，$r$ 是奖励值。奖励函数借鉴强化学习思想：

$$r = P_{target} - P_b - \beta$$

- $P_{target}$：目标功耗
- $P_b$：BS 实际总功耗（$P_b = \sum_{k=1}^{K_b} p_{b,k}$）
- $\beta$：惩罚项，仅在违反最小数据速率约束 $C_{min}$ 时施加

每次 LLM 做出决策后，实际实施结果 $(s_t, a_t, r_t)$ 会被追加到经验池中，形成持续积累的知识库。

#### 3. 状态基示例选择（离散状态）

对于离散状态空间（如用户数量），直接匹配经验池中状态相同的示例：

$$\mathcal{E}_{relevant} = \{E\{s, a, r(s,a)\} \mid s = s_{target}, E \in \mathcal{E}_{pool}\}$$

从匹配结果中选择：
- **推荐示例**：奖励值最高的经验
- **反面示例**：奖励低或违反约束的经验

两类示例共同帮助 LLM 理解"什么是好决策，什么是坏决策"。

#### 4. 排名基示例选择（连续状态）

对于连续状态空间（如用户到基站的平均距离），无法精确匹配状态值，因此定义综合评价指标：

$$\mathcal{L}(E, s_{target}) = r(s, a) - \tau \|s - s_{target}\|$$

该指标同时考虑两个维度：
- $r(s, a)$：示例本身的奖励质量
- $\|s - s_{target}\|$：示例状态与目标状态的 $L_2$ 距离
- $\tau$：距离权重的超参数

按 $\mathcal{L}$ 值排序后选取 top-k 示例，平衡了"高质量"和"高相关性"两个要求。

#### 5. Epsilon-Greedy 探索策略

为平衡探索与利用，引入经典的 $\epsilon$-greedy 策略：

$$a = \begin{cases} \text{随机动作}, & \text{if } rand < \epsilon \\ \text{LLM 决策}, & \text{otherwise} \end{cases}$$

随机探索不断产生新示例，使经验池持续丰富，LLM 得以从更优的示例中学习。

### 损失函数 / 训练策略

本方法的核心特点是**无需训练**——LLM 参数完全不更新。其"学习"过程通过以下机制实现：

- **隐式微调理论**：根据 Dai et al. 的分析，ICL 可等价为 $\tilde{f}_{ICL}(\mathbf{q}) = \mathbf{q}(W_{ZSL} + \Delta W_{ICL})$，即 LLM 在前向计算中通过注意力机制产生元梯度，实现隐式的权重更新
- **经验池增长**：随交互轮次增加，经验池不断积累更多高质量示例
- **示例筛选优化**：更好的示例 → 更好的决策 → 更高奖励的新经验 → 正向循环

计算复杂度方面，示例选择为线性复杂度（遍历经验池计算匹配度），无需反向传播。

## 实验关键数据

### 实验设置

- 3 个相邻小基站（SBS），用户数在 5-15 之间随机变化
- SBS 覆盖范围 20 米，信道模型为 3GPP 城市网络
- 两种场景：Case I（离散状态 - 用户数）、Case II（连续状态 - 平均用户距离）
- 4 个功率等级可选

### 主实验

| 方法 | 离散状态奖励 | 连续状态奖励 | 是否需要训练 |
|------|-------------|-------------|-------------|
| 穷举搜索（最优） | 最优基线 | 最优基线 | 否 |
| DRL（传统基线） | 接近最优 | 接近最优 | 是（大量迭代） |
| GPT-4 (ICL) | 接近 DRL | 接近 DRL | **否** |
| Llama3-70b (ICL) | 接近 DRL | 接近 DRL | **否** |
| Llama3-8b (ICL) | 接近 DRL | 接近 DRL | **否** |
| GPT-3.5 (ICL) | 低于 DRL | 低于 DRL | 否 |
| Feedback-based | 显著低于 DRL | 显著低于 DRL | 否 |

### 消融实验

| 配置 | 平均奖励变化 | 说明 |
|------|-------------|------|
| 完整方法（ICL + 经验池 + 示例选择 + 探索） | 基线 | 所有组件协同工作 |
| 移除经验池 | 大幅下降 | 无历史经验参考，决策质量差 |
| 移除示例选择策略 | 显著下降 | 随机示例无法有效指导 LLM |
| 移除 ε-greedy 探索 | 明显下降 | 经验池多样性不足 |
| Feedback-based 方法 | 显著低于完整方法 | 仅靠反馈无法捕捉动态环境复杂性 |

### 关键发现

1. **模型能力决定性能**：GPT-4 和 Llama3（SOTA 模型）表现远优于 GPT-3.5（早期模型），说明 ICL 性能与 LLM 基础能力强相关
2. **示例数量效应**：增加 prompt 中的示例数量可持续提升奖励，但边际收益递减
3. **状态空间扩展**：更大的状态空间需要更多示例才能达到同等性能，但总趋势仍在持续改善
4. **约束适应性**：在不同最小数据速率约束下，GPT-4 和 Llama3 能自适应调整策略，功耗和服务质量表现稳健
5. **动态环境优势**：相比 feedback-based 方法只能处理静态优化，本方法的经验池机制能有效应对动态场景

## 亮点与洞察

1. **范式创新**：首次系统性地将 LLM ICL 应用于无线网络动态优化，证明了"用语言做网络优化"的可行性
2. **经验池设计精巧**：将强化学习的经验回放思想迁移到 ICL 场景，(s, a, r) 三元组设计简洁有效
3. **双模式示例选择**：针对离散/连续状态分别设计精确匹配和排名选择策略，实用性强
4. **零训练开销**：在网络动态变化频繁的场景下，免训练特性意味着极快的部署和适应速度
5. **可解释性**：LLM 可以为其决策提供自然语言解释，有助于运营商理解复杂网络行为

## 局限与展望

1. **场景简化**：仅考虑 3 个 SBS、4 个功率等级的小规模场景，大规模网络的可扩展性未验证
2. **推理延迟**：LLM 推理时间可能无法满足毫秒级网络控制的实时性要求
3. **API 成本**：使用 GPT-4 等商业模型的 API 调用成本在大规模部署时可能过高
4. **连续动作空间**：功率被离散化为 4 级，无法实现精细的连续功率控制
5. **多目标优化**：目前只考虑功率最小化 + 数据速率约束，更复杂的多目标场景未涉及
6. **示例池膨胀**：长期运行后经验池可能无限增长，需要设计淘汰机制

## 相关工作与启发

- **LLM for Telecom**：与 Lin et al. (6G edge intelligence)、Qiu et al. (LLM wireless design) 等工作形成呼应，但本文首次提出经验池驱动的动态 ICL 框架
- **ICL 理论**：基于 Dai et al. 的隐式微调理论解释 ICL 机制，将 attention 机制视为元梯度计算器
- **跨领域启发**：经验池 + 示例选择的 ICL 范式可推广到其他网络优化任务（波束成形、RIS 相位优化等），甚至更广泛的组合优化问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 ICL 用于动态网络优化是有意义的新方向，经验池设计较新颖
- 实验充分度: ⭐⭐⭐ — 消融实验完整，但场景规模小，缺少与更多基线的对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，任务定义和算法描述规范
- 价值: ⭐⭐⭐ — Workshop 论文，概念验证阶段，距离实际部署还有较大距离

<!-- RELATED:START -->

## 相关论文

- [ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](../../ACL2025/signal_comm/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)
- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](../../NeurIPS2025/signal_comm/the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)
- [Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)
- [WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications](../../ACL2025/signal_comm/wirelessmathbench_a_mathematical_modeling_benchmark_for_llms_in_wireless_communi.md)

<!-- RELATED:END -->
