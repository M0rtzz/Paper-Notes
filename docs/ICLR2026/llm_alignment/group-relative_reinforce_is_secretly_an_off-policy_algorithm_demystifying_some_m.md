---
title: >-
  [论文解读] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends
description: >-
  [ICLR 2026][LLM对齐][GRPO] 通过构造 KL 正则化代理目标并推导 pairwise consistency condition，从第一原理证明 group-relative REINFORCE（GRPO）天然是 off-policy 算法…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "GRPO"
  - "off-policy RL"
  - "importance sampling"
  - "clipping"
  - "REINFORCE"
  - "策略优化"
---

# Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends

**会议**: ICLR 2026  
**arXiv**: [2509.24203](https://arxiv.org/abs/2509.24203)  
**代码**: [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k)  
**领域**: LLM对齐 / RL  
**关键词**: GRPO, off-policy RL, importance sampling, clipping, REINFORCE, 策略优化

## 一句话总结

通过构造 KL 正则化代理目标并推导 pairwise consistency condition，从第一原理证明 group-relative REINFORCE（GRPO）天然是 off-policy 算法；进而通过组件隔离实验发现 clipping 才是训练稳定性的关键而 importance sampling 完全可以去掉，并在此统一框架下重新解释了 Kimi OPMD、Meta AsymRE 等多个看似独立的算法。

## 研究背景与动机

**领域现状**：GRPO 及其变体（DAPO、GiGPO）已成为 LLM RL 训练的主流算法。DeepSeek-R1 用 GRPO 训练推理模型取得了突破性成果，Kimi 团队提出了 OPMD，Meta 提出了 AsymRE——这些方法各自给出了不同的理论 justification，但实际上它们之间的内在联系并不清楚。

**现有痛点**：GRPO 的成功被归因于多种因素——group-relative advantage 降低方差、importance sampling 校正分布偏移、clipping 稳定训练——但每个组件的真实贡献从未被系统性地隔离和验证。更关键的是，GRPO 在理论上被视为 on-policy 算法（需要从当前策略采样以得到无偏梯度估计），但工程实践中几乎总是在 off-policy 数据上运行（rollout 生成和模型训练速度不匹配、数据来自旧版策略、奖励反馈延迟），这种理论与实践的脱节缺乏严格解释。

**核心矛盾**：经典策略梯度理论要求训练数据来自当前策略 $\pi_\theta$，off-policy 校正依赖 importance sampling 权重 $\pi_\theta(y|x)/\pi_b(y|x)$，但这些权重在 LLM 语境下会随序列长度指数爆炸。现有做法用 token-wise 比率代替 response-wise 比率，引入了偏差但没有严格的理论保证。

**本文要解决**：(1) 为 GRPO 提供不依赖采样分布假设的理论推导；(2) 系统性地隔离 IS、clipping 等组件的作用；(3) 在统一框架下解释 GRPO、OPMD、AsymRE 的内在联系。

**切入角度**：作者观察到如果从 KL 正则化代理目标出发，推导其最优解满足的 pairwise consistency condition，然后构造强制满足该条件的均方代理损失，对这个代理损失在当前参数处取一步梯度——恰好就是 GRPO 的更新公式。整个推导过程完全不需要指定数据来自哪个策略。

**核心idea**：GRPO 是 off-policy 算法；clipping 是稳定性的唯一关键组件而 IS 几乎无用；用两条增强原则（正则化策略更新 + 主动塑造数据分布）可以统一理解和改进一系列 RL 算法。

## 方法详解

### 整体框架

本文的理论框架分三步：首先定义以上一轮策略 $\pi_{\theta_t}$ 为锚点的 KL 正则化代理目标 $J(\theta; \pi_{\theta_t}) = \mathbb{E}[r(x,y)] - \tau \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\theta_t})$，推导其最优策略满足的 pairwise consistency condition；然后用有限样本构造强制该条件的均方代理损失；最后证明在当前参数 $\theta_t$ 处对该损失取一步梯度，结果恰好等价于 group-relative REINFORCE 的更新公式——而整个推导不需要任何关于训练数据分布的假设，因此天然支持 off-policy。

基于这个 off-policy 解释，作者提出两条增强原则来应对任意数据分布：(1) 正则化策略更新步（如 clipping、KL 惩罚），防止在次优数据分布下策略更新过大导致崩溃；(2) 主动塑造训练数据分布（如样本加权、低奖励样本丢弃），引导策略更新方向。这两条原则统一解释了 GRPO、OPMD、AsymRE 以及各种数据加权启发式方法。

### 关键设计

1. **三步推导：从代理目标到 REINFORCE**:

    - 功能：证明 group-relative REINFORCE 天然具有 off-policy 解释
    - 核心思路：KL 正则化代理目标的最优解满足 $\pi^*(y|x) \propto \pi_{\theta_t}(y|x) \exp(r(x,y)/\tau)$，由此推出任意两个响应 $y_1, y_2$ 之间的 pairwise consistency condition：$r_1 - \tau(\log\pi(y_1|x) - \log\pi_{\theta_t}(y_1|x)) = r_2 - \tau(\log\pi(y_2|x) - \log\pi_{\theta_t}(y_2|x))$。构造强制该条件的均方损失 $\hat{L} = \frac{1}{K^2}\sum_{i<j}(a_i - a_j)^2$，在 $\theta = \theta_t$ 处取梯度后，log-probability 差项归零，结果化简为 $\frac{1}{K}\sum_i (r_i - \bar{r}) \nabla_\theta \log\pi_\theta(y_i|x)$——正是 GRPO 的更新公式
    - 设计动机：传统策略梯度理论要求 on-policy 采样，限制了 GRPO 在异步训练中的理论正当性。这个推导绕过了采样分布假设，直接从最优性条件出发，为 off-policy 使用提供了坚实的理论基础

2. **REC 系列：隔离 IS 与 Clipping 的作用**:

    - 功能：精确识别 GRPO 中每个组件对训练稳定性的贡献
    - 核心思路：设计一系列 REINFORCE-with-Clipping（REC）变体进行消融。REC-OneSide-IS 保留 IS 权重和单侧 clipping 但去掉 advantage normalization，REC-OneSide-NoIS 在此基础上进一步去掉 IS 权重，只保留 clipping mask $M_i^t = \mathbb{1}(A_i > 0, \rho_i^t \leq 1+\epsilon_{\text{high}}) + \mathbb{1}(A_i < 0, \rho_i^t \geq 1-\epsilon_{\text{low}})$。同时测试扩大 clipping 范围从标准的 $(0.2, 0.2)$ 到激进的 $(0.6, 2.0)$
    - 设计动机：社区普遍认为 IS 是 off-policy 校正的核心机制，但实验表明去掉 IS 后性能几乎不变（奖励曲线完全重叠），而去掉 clipping 后训练立即崩溃。这意味着 clipping 实际上是一种隐式的信赖域约束——限制了每步策略更新的幅度，防止在有限样本覆盖不足时策略偏离到次优区域

3. **统一解释 OPMD 和 AsymRE**:

    - 功能：揭示三个看似独立的算法共享相同的底层结构
    - 核心思路：Kimi OPMD 的损失函数可以分解为 REINFORCE loss + 均方正则化 $\frac{\beta}{2K}\sum_i(\log\pi_\theta(y_i|x) - \log\pi_{\text{old}}(y_i|x))^2$，其中 $\beta = \tau$。Meta AsymRE 的 baseline 偏移 $\bar{r} - \beta$ 等价于 REINFORCE loss + KL 正则化 $\frac{\beta}{K}\sum_i \log\frac{\pi_{\text{old}}(y_i|x)}{\pi_\theta(y_i|x)}$，在大样本极限下收敛到 $\beta \cdot D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)$。两者都是"正则化策略更新"原则的具体实例，只是正则化形式不同
    - 设计动机：OPMD 原论文的推导从 KL 正则化目标的 pointwise consistency condition 出发（和本文部分重叠但在 step 2 分道），AsymRE 原论文用多臂赌博机分析 justify baseline 偏移。本文的统一视角更清晰——它们都只是 REINFORCE + 某种正则化，对应第一条增强原则

### 数据加权方法（RED 系列）

作者进一步将 pairwise 代理损失中的均匀权重推广为一般权重 $\sum_{i<j} w_{i,j}(a_i - a_j)^2$，推导出加权 REINFORCE 更新公式。基于此提出两种方法：

- **RED-Drop**：丢弃部分低奖励负样本，只用子集 $\mathcal{S} \subseteq [K]$ 训练。动机是负梯度增加 entropy collapse 风险（与 Kimi-Researcher 博客建议一致），在 off-policy 框架下有理论 justification
- **RED-Weight**：用奖励相关的权重 $w_i$ 对每个样本的梯度项加权。可以分解为 pairwise 加权 REINFORCE + 一个模仿高奖励响应的正则化项，呼应了 offline RL 文献中"对高奖励轨迹正则化比保守模仿所有轨迹更有效"的发现

### 损失函数 / 训练策略

核心损失函数：标准 REINFORCE loss $-\frac{1}{K}\sum_i(r_i - \bar{r})\log\pi_\theta(y_i|x)$ + 可选正则化项（clipping mask / KL 惩罚 / 均方正则化），不同搭配对应不同算法。训练采用 Trinity-RFT 框架，通过 `sync_interval`（模型同步频率）和 `sync_offset`（rollout 与训练的延迟）两个参数精确控制 off-policy 程度。

## 实验关键数据

### 主实验：IS vs Clipping 消融（GSM8k, Qwen2.5-1.5B-Instruct）

| 算法 | Clipping 范围 | IS | On-Policy 奖励 | Mixed 奖励 | Offline 奖励 |
|------|-------------|-----|---------------|-----------|-------------|
| GRPO | (0.2, 0.2) | ✓ | 收敛正常 | 收敛正常 | 收敛正常 |
| REC-OneSide-IS | (0.2, 0.2) | ✓ | ≈ GRPO | ≈ GRPO | ≈ GRPO |
| REC-OneSide-NoIS | (0.2, 0.2) | ✗ | ≈ GRPO | ≈ GRPO | ≈ GRPO |
| REC-OneSide-NoIS | (0.6, 2.0) | ✗ | **加速收敛** | **加速收敛** | 速率↑但不稳定 |
| REINFORCE（无 clipping） | — | ✗ | **训练崩溃** | **训练崩溃** | **训练崩溃** |

核心结论：去掉 IS 后三种设置下奖励曲线完全重叠，证明 IS 非必要；去掉 clipping 则立即崩溃，证明 clipping 是唯一关键组件。扩大 clipping 范围在 on-policy 和 mixed 设置下加速收敛，但在纯 offline 设置下出现速率-稳定性 trade-off。

### 消融与扩展实验

| 实验设置 | 任务/模型 | 关键发现 |
|---------|----------|---------|
| REC 系列 | ToolACE / Llama-3.2-3B | IS 非必要；clipping 仍然是稳定性关键；结论跨模型和任务一致 |
| RED-Drop | GSM8k / Qwen2.5-1.5B | 丢弃低奖励样本在 on/off-policy 均有效，性能与 REC 扩大范围相当 |
| RED-Weight | Guru-Math / Qwen2.5-7B | 加权方法在大规模任务上超过 GRPO，且 KL 偏移相近，规模效应正面 |
| RED-Weight | MATH / Llama-3.1-8B | 跨模型验证有效性，在更难的数学任务上仍有提升 |
| OPMD 复现 | GSM8k / Qwen2.5-1.5B | 均方正则化和 clipping 有互补效果，但单独使用 clipping 已足够 |
| AsymRE 复现 | GSM8k / Qwen2.5-1.5B | Baseline 偏移（KL 正则化）有效但不如 clipping 鲁棒 |
| Offline 压力测试 | GSM8k | 只用初始策略采样的离线数据训练，暴露了扩大 clipping 范围的稳定性极限 |

### 关键发现

- **IS 完全可以去掉**：在所有测试的模型（1.5B/3B/7B/8B）、任务（GSM8k/MATH/ToolACE/Guru-Math）和 off-policy 程度（on-policy/mixed/offline）下，去掉 IS 后性能无显著变化。这意味着工程实现可以省去存储和计算旧策略概率的开销
- **Clipping 是唯一不可或缺的组件**：相当于隐式信赖域约束，限制了 $\pi_\theta/\pi_{\text{old}}$ 的变化幅度。没有它，策略更新在有限样本下方向不可控
- **扩大 clipping 不对称范围可加速训练**：允许正 advantage 的策略比率更大增长（$\epsilon_{\text{high}} = 2.0$），同时适度放松负 advantage 的下限（$\epsilon_{\text{low}} = 0.6$），直觉上是鼓励学好的同时允许遗忘坏的
- **3-arm bandit 反例**：vanilla REINFORCE 在行为策略 $\pi_b = [0.3, 0.6, 0.1]$、奖励 $r = [0, 0.8, 1]$ 时会收敛到次优动作 $a_2$ 而非最优 $a_3$，因为 $\pi_b(a_2)(r(a_2) - \mu_r) > \pi_b(a_3)(r(a_3) - \mu_r)$——说明不加正则化/数据塑造时 off-policy REINFORCE 必然失败

## 亮点与洞察

- **理论推导的优雅性**：三步推导（代理目标 → consistency condition → 均方损失 → 一步梯度 = REINFORCE）结构清晰，每步都有明确的物理直觉。特别是在 $\theta_t$ 处取梯度时 log-probability 差项自然归零的技巧，使得最终公式干净地回到 REINFORCE——这不是偶然的巧合而是深层结构的反映
- **"IS 无用"的反直觉发现**：IS 通常被认为是 off-policy RL 的基础设施，但在 LLM 微调的语境下（策略变化通常不大，token-wise IS 本身就是偏的），IS 权重接近 1，校正效果可忽略。真正起作用的是 clipping 带来的隐式信赖域。这个洞察直接简化了工程实现
- **统一框架的解释力**：将 GRPO（clipping 正则化）、OPMD（均方正则化）、AsymRE（KL 正则化）统一为 REINFORCE + 不同正则化形式，把三条独立的研究线索串成了一个故事。同时，RED-Drop 和 RED-Weight 对应第二条原则（数据分布塑造），完成了理论闭环

## 局限与展望

- **缺乏收敛保证**：off-policy 解释提供了正当性但没有给出 policy improvement 或收敛的形式化保证，需要未来工作在特定数据分布假设下建立
- **纯 offline 设置的 trade-off 未解决**：扩大 clipping 范围在 offline 设置下可能导致不稳定，作者指出这是一个开放问题，可能需要自适应 clipping 策略（根据训练进度、off-policy 程度动态调整范围）
- **单/多轮 RL 的差异**：主要分析基于 one-step RL（单轮 prompt-response），多步推广在附录但缺乏实验验证。对于需要多轮环境交互的 agentic RL，结论的迁移性有待检验
- **仅验证了数学推理和工具使用**：GSM8k、MATH、ToolACE 等任务的奖励信号都是明确的正确/错误，对于奖励更模糊的场景（如对话质量、创意写作）中组件贡献是否一致尚不清楚
- **Group size $K$ 的影响未充分分析**：pairwise consistency condition 基于有限样本对，$K$ 的选择如何影响代理损失对真实目标的近似质量？小 $K$ 下 group-relative baseline 的方差问题是否与 off-policy 程度交互？

## 相关工作与启发

- **vs PPO**：PPO 用 clipping 限制 on-policy 策略更新步长，本文证明同样的 clipping 机制在 off-policy 设置下也是关键的稳定器。区别在于 GRPO 的 clipping 作用于 group-relative advantage 而非原始 ratio，且 GRPO 不需要 value function
- **vs DPO**：DPO 是纯 offline 偏好优化（Bradley-Terry 模型推导），本文的 off-policy REINFORCE 保持了 online 学习能力但可以容忍旧数据。两者可以互补——DPO 用于冷启动，GRPO 用于持续学习
- **vs DAPO**：DAPO 在 GRPO 基础上增加了 token-level entropy bonus 和 dynamic sampling 等探索机制。本文的分析为 DAPO 的经验成功提供了理论基础——DAPO 的改进本质上都可以归入"正则化策略更新"和"塑造数据分布"两条原则
- **vs REBEL/CoPG**：这两个工作与本文共享 KL 正则化目标和 pairwise consistency condition（step 1-2 重叠），但它们选择直接最小化代理损失（多步梯度下降），而本文的贡献在于发现单步梯度就回到 REINFORCE——由此建立了理论和实践的桥梁

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从第一原理推导出 GRPO 的 off-policy 解释，组件隔离实验彻底推翻了"IS 是核心"的共识
- 实验充分度: ⭐⭐⭐⭐ 覆盖 5 个模型、4 个任务、3 种 off-policy 程度，消融全面；但缺少更多 agentic 和对话场景验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导极其清晰，三步结构层层递进；统一框架的呈现方式教科书级别
- 价值: ⭐⭐⭐⭐⭐ 对 LLM RL 社区有根本性指导——"去掉 IS、扩大 clipping 范围"是可立即落地的工程优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)

</div>

<!-- RELATED:END -->
