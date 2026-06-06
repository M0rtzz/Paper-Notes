---
title: >-
  [论文解读] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents
description: >-
  [ICML 2026][LLM Agent][主动推理] 针对 LLM agent 在多轮主动推理中"动作选择(AS)"与"信念跟踪(BT)"互相拖累、outcome-only RL 训练陷入低信息自锁(SeL)的失效模式，本文给出 POMDP 视角下的耦合梯度分析与"自锁区"形式化定义…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "主动推理"
  - "Agentic RL"
  - "信用分配"
  - "自锁失效"
  - "advantage reweighting"
---

# On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents

**会议**: ICML 2026  
**arXiv**: [2603.12109](https://arxiv.org/abs/2603.12109)  
**代码**: https://github.com/unimpor/T3 (有)  
**领域**: LLM推理 / Agent / 强化学习  
**关键词**: 主动推理, Agentic RL, 信用分配, 自锁失效, advantage reweighting

## 一句话总结
针对 LLM agent 在多轮主动推理中"动作选择(AS)"与"信念跟踪(BT)"互相拖累、outcome-only RL 训练陷入低信息自锁(SeL)的失效模式，本文给出 POMDP 视角下的耦合梯度分析与"自锁区"形式化定义，并提出 AReW：用环境/读出层即可获得的方向性 critique 对 stepwise advantage 做加性 reweighting，在 9 个主动推理任务上最高带来 60 分性能提升。

## 研究背景与动机

**领域现状**：outcome-based RL（PPO / GRPO / GSPO）已经成为训练 LLM agent 的事实标准，被用于 deep research、coding、工具调用、多轮医疗问诊等需要"边交互边推理"的任务。这类任务的共同特征是：开局信息不全，agent 必须主动 query 环境/用户来收集证据，再根据反馈更新对任务状态的内部判断。

**现有痛点**：作者在 PE（偏好估计）、MediQ（医疗诊断）、FloDial（故障排查）等基准上观察到一个反直觉现象——reward 曲线确实在涨，但"每轮信息量(AS proxy)"和"每轮信念增益(BT proxy)"几乎不动甚至倒退；agent 学到的不是"更会问问题、更会整合证据"，而是绕开交互、用启发式或先验直接出答案的捷径。在 MediQ 上把所有患者反馈替换成 "Unknown" 后，未 RL 训练模型掉点 10.75（41.25→30.50），RL 训练后仅掉 5.50（61.00→55.50），同时 belief consistency 从 78.7 升到 92.8——说明 RL 反而让 agent 变得"更固执、更不依赖交互"。

**核心矛盾**：agent 的能力可以解耦成 AS（决定能拿到什么观测）和 BT（决定观测能不能被吸收），但二者在 outcome reward 下是双向耦合的：弱 BT 让信息丰富的 action 拿不到 credit（AS 学不动），弱 AS 又让 BT 拿不到有用证据可学（BT 学不动）。这两个 channel 互锁后形成低信息低 BT 的吸引子——作者称为 information self-locking (SeL)。

**本文目标**：(i) 形式化 SeL 是什么、为什么在 outcome RL 下稳定存在；(ii) 给出一个**不引入校准 intermediate reward、不训练 dense reward model** 的轻量方案打破 SeL。

**切入角度**：在很多主动推理场景里，"这一步动作有没有引出新信息"以及"这一步信念更新有没有朝真值方向移动"这两件事，比"这一步该给多少 reward"要容易判断得多——前者只需要看环境反馈是否包含新证据、看 self-report 置信度是否变高，是定性的方向性 critique，可以用规则或廉价 LLM 标注，且不需要校准。

**核心 idea**：把这些 ±1/0 的方向性 critique 翻译成一个 likelihood-margin 辅助目标，其梯度恰好等价于在标准 policy gradient 的 advantage 上加一项常数偏置 $\hat{A}_t \leftarrow A_t + \lambda u_t$，从轨迹内部把 credit 从被负向 critique 的步骤搬到被正向 critique 的步骤上，outcome reward 不动，RL 主干不动。

## 方法详解

### 整体框架
作者把主动推理建模成 POMDP $(\mathcal{S},\mathcal{A},\mathcal{O},T,O,R,\gamma)$，latent state $s^\star$ 是隐藏的用户偏好/诊断/方案。Agent 的行为被分解成两个交错的 kernel：

- **AS kernel** $\pi_\omega^{\mathrm{as}}(a_t\mid b_t)$：基于当前内部信念 $b_t$ 选环境动作，决定下一步观测 $o_t \sim O(\cdot\mid s^\star, a_t)$；
- **BT kernel** $\pi_\omega^{\mathrm{bt}}(b_{t+1}\mid b_t, a_t, o_t)$：把新观测整合进信念。

轨迹是 (AS round, Update round) 交替序列。文章先在第 2、3 节用 oracle-belief 轨迹和 Bayesian update 把 SeL 形式化、证明在 SeL 区 outcome-gradient 对 AS/BT 的提升信号都按 $\eta \cdot (\text{当前能力})$ 一阶尺度衰减（Thm. 3.4），从而 agent 很难自发逃出 SeL；然后在第 4 节用 AReW 做 advantage reweighting 引入方向性 critique，把这个一阶项重新拉起来。

### 关键设计

1. **SeL 形式化与耦合梯度分解**：

    - 功能：给"为什么 outcome RL 学不动主动推理"一个可证伪的解释，并定义清楚 SeL 区。
    - 核心思路：定义 belief potential $\Psi(b) := b(s^\star)$，用 oracle-belief 轨迹 $\bar\tau \sim (\pi_\omega^{\mathrm{as}}, \mathsf{BayesUpd})$ 隔离 AS 自身的信息能力 $I_{\mathrm{AS}}(\omega) := \mathbb{E}[\Psi(\bar b_H) - \Psi(\bar b_0)]$；用 on-policy 轨迹的"吸收正分量"$\Delta\Psi_t^+ = \max(0, \Psi(b_{t+1})-\Psi(b_t))$ 求和定义 BT 容量 $C_{\mathrm{BT}}(\omega)$。SeL 区为 $\mathcal{R}_{\delta,\varepsilon} := \{\omega: I_{\mathrm{AS}}\le\delta,\, C_{\mathrm{BT}}\le\varepsilon\}$。把 $\nabla_\omega \log p_\omega(\tau)$ 拆成 AS-channel 与 BT-channel 两个梯度 $g_{\mathrm{as}}, g_{\mathrm{bt}}$，Thm. 3.4 证明在 SeL 区两 channel 的 one-sided drift 满足 $(\Delta_{\mathrm{as}}^+ I_{\mathrm{AS}},\, \Delta_{\mathrm{bt}}^+ C_{\mathrm{BT}})^\top \preceq \eta M (I_{\mathrm{AS}}, C_{\mathrm{BT}})^\top + o(\eta)$，其中矩阵 $M$ 的非零项乘的是另一 channel 的能力——所以 AS 弱时 BT 学不动、BT 弱时 AS 学不动，escape time 有显式下界。
    - 设计动机：以前关于 LLM agent RL 失效的解释多停在"reward 太稀疏"，本文把失败原因精确归到"AS-BT 双向掩蔽 credit"，这才能解释 reward 在涨但能力不涨的现象，也直接给出干预接口——只要给两个 channel 加方向性 critique 就行。

2. **AS / BT 方向性 critique**：

    - 功能：在不构造 calibrated dense reward 的前提下，为每一步行为提供 $z_t \in \{-1, 0, +1\}$ 的廉价方向信号。
    - 核心思路：**AS critique** $z_t^{\mathrm{as}}$ 取自"这一步动作有没有从环境/用户那里拿到新证据"，可由规则或 LLM-judge 给出（如 PE 中按 attribute pair 是否非支配判定，MediQ 中看新事实集合是否非空）；**BT critique** $z_t^{\mathrm{bt}} := \mathrm{Sign}(\hat\Psi_{t+1} - \hat\Psi_t)$，其中 $\hat\Psi_t$ 是 prompt agent 自己 report 出来的对 ground-truth 候选的置信度——这是 instrumentation 用的标量，不假设它等于 analytical belief $b_t$。Prop. 4.1 证明 AReW 的提升量为 $I_{\mathrm{AS}}(\hat{\mathcal{T}}_{\mathrm{as}}) - I_{\mathrm{AS}}(\mathcal{T}_{\mathrm{as}}) = \eta W(\omega) (2\,\mathrm{Acc}_{\mathrm{as}} - 1)$，所以只要加权准确率 $> 1/2$ 就有正向收益，不需要校准。
    - 设计动机：calibrated step reward 在 long-horizon agentic 任务里几乎不可得（要么需训 reward model 引入新偏差，要么需大量标注），而方向性 critique 只问"好/坏/不知道"，门槛低、对噪声鲁棒，又恰好对得上 SeL 的两个 channel。

3. **likelihood-margin 目标 → advantage reweighting**：

    - 功能：把方向性 critique 注入 policy-gradient，并保证 (i) 局部作用于被 critique 的步骤、(ii) outcome reward 与 RL 优化机制一律不动。
    - 核心思路：对轨迹 $\tau$ 令 $\mathcal{P}_\tau := \{t: z_t=+1\}$、$\mathcal{N}_\tau := \{t: z_t=-1\}$，定义轨迹内 likelihood-margin 辅助目标 $\hat{\mathcal{L}}(\omega;\tau) := \frac{1}{|\mathcal{P}_\tau|}\sum_{t\in\mathcal{P}_\tau}\log\pi_{\omega,t} - \frac{1}{|\mathcal{N}_\tau|}\sum_{t\in\mathcal{N}_\tau}\log\pi_{\omega,t}$。对它求梯度恰好得到 $\sum_t u_t \nabla_\omega \log\pi_{\omega,t}$，其中 $u_t = +1/|\mathcal{P}_\tau|, -1/|\mathcal{N}_\tau|, 0$ 对应 $z_t$ 三种取值，且 $\sum_t u_t = 0$（centering 性质，纯 margin 无均值漂移）。最终 augmented surrogate $\hat{\mathcal{L}}_{\mathrm{aug}} := \mathcal{J}_{\mathrm{RL}} + \lambda \mathbb{E}_\tau[\hat{\mathcal{L}}]$，等价于把 advantage 改成 $\hat A_t := A_t + \lambda u_t$，其余的 critic、ratio、KL 项全部保持原样——这也是为什么作者强调 AReW 是 RL 算法无关的（PPO / GRPO / GSPO 都能挂）。
    - 设计动机：以加性 advantage shaping 这种极简形式注入，既不会污染 outcome reward 的无偏估计（centering 保证），又把信用从负向步骤搬到正向步骤，正好打断 Thm. 3.4 里"另一 channel 弱→当前 channel drift 被压低"的链路。

### 损失函数 / 训练策略
最终训练目标即 Eq. (2) 的 $\hat{\mathcal{L}}_{\mathrm{aug}}(\omega) = \mathcal{J}_{\mathrm{RL}}(\omega) + \lambda\,\mathbb{E}_\tau[\hat{\mathcal{L}}(\omega;\tau)]$，落到实现就是逐 step 把 advantage 替换成 $A_t + \lambda u_t$。AS critique 用规则/feedback parser 给出，BT critique 让 agent 在每个 Update Round 里以特定 prompt 输出对 $s^\star$ 候选的置信度 $\hat\Psi_t$，再用相邻两轮差分取符号即可。$\lambda$ 与稀疏 reward 量级匹配，作者实证对超参不敏感。

## 实验关键数据

### 主实验
9 个主动推理任务，跨 4 个领域（偏好估计、医疗诊断、故障排查、客服 $\tau^2$-bench），主干为 Qwen-2.5-7B-Instruct 与 LLaMA-3.1-8B-Instruct。下表是 Qwen-2.5-7B-Instruct + PPO 上的核心 7 个任务（数字是 test set 平均 outcome reward）：

| 任务 | o4-mini (直推) | Vanilla PPO | AReW–AS only | AReW–AS+BT | AS+BT 相对 Vanilla |
|------|----------------|-------------|--------------|------------|-------------------|
| PE-G S=2 | 17.11 | 24.00 | 46.00 | 49.33 | +25.3 |
| PE-G S=3 | 21.15 | 18.33 | 32.00 | **80.33** | **+62.0** |
| PE-F D=8 | 8.42 | 30.52 | 39.62 | 47.89 | +17.4 |
| PE-F D=6 | 12.47 | 32.03 | 42.10 | 44.47 | +12.4 |
| MediQ | 74.67 | 50.50 | 57.25 | 61.25 | +10.8 |
| FloDial-Easy | 35.00 | 37.33 | 43.67 | 41.00 | +3.7 |
| FloDial-Hard | 26.33 | 21.33 | 36.00 | 42.33 | +21.0 |

LLaMA-3.1-8B-Instruct 上规律完全一致，PE-G S=3 同样从 11.00 跳到 77.67（+66.7），PE-F D=6 从 6.00 升到 54.65（+48.7）。28 个 (task, RL algo) 配置里 AReW 赢 27 个；as+bt 在 14 对比较里有 11 次显著优于 as-only，验证两 channel 同时干预的必要性。

### 消融 / 鲁棒性

| 配置 | PE-G S=3 | FloDial-Hard | 说明 |
|------|----------|--------------|------|
| Vanilla PPO | 18.3 | 21.3 | outcome-only baseline |
| AReW (α=0, 干净 critique) | 80.3 | 36.0 | 上限 |
| AReW (α=0.1, 翻转 10%) | 40.0 | 30.3 | 仍优于 vanilla |
| AReW (α=0.2) | 65.0 | 29.0 | 仍优于 vanilla |
| AReW (α=0.3) | 31.3 | 27.6 | 仍优于 vanilla |
| AReW (α=0.4) | 22.3 | 30.6 | 临界附近 |
| AReW (α=0.5) | 30.3 | 23.3 | 随机 critique 退化到接近 baseline |

这与 Prop. 4.1 给出的 $2\,\mathrm{Acc} - 1$ 标度一致：只要加权准确率不到 0.5 才崩。

### 关键发现
- **耦合是真的**：PE-G/MediQ 上固定动作序列、只替换 BT（用规则或前沿模型做 belief update），AS 与 reward 的相关性显著上升，说明弱 BT 在掩盖 AS 的贡献——直接验证了 Obs. 2 与 Thm. 3.4 的假设。
- **as-only 也能涨 BT**：纯 AS critique 不仅拉高 AS proxy，BT proxy 也跟着涨，说明 AReW 通过 AS 端拉升信息流就能间接帮 BT 学习——再次印证两个 channel 的内禀耦合。
- **RL 算法无关**：把 AReW 挂到 GRPO/GSPO 上效果与 PPO 一致；group-based 多 rollout 本身解不了 SeL，因为耦合是结构性的而非采样方差问题。
- **客服 $\tau^2$-bench 上额外发现**：vanilla PPO 训练后 tool execution error 与 token 用量都飙升、turn 数被压缩；AReW 把交互节奏拉回来，error rate 显著下降——SeL 在真实工具调用场景同样出现。

## 亮点与洞察
- **失效模式的精准命名**：把"reward 涨能力不涨"这一长期被笼统归因为"sparse reward"或"shortcut learning"的现象，定位到 AS-BT 双向信用掩蔽，并给出形式化定义与 escape-time 下界，是少见的"先把病讲清楚再开药"的工作。
- **方向性 critique 替代 calibrated reward**：Prop. 4.1 把"critique 质量"刻画为加权准确率 $>1/2$，等于告诉社区：在 long-horizon agentic RL 里别再死磕 process reward model，能拿到"好/坏"判断就足够。这条结论的工程价值远超本文实验范围，可直接迁移到 RAG-agent、coding agent、deep research agent 等任何"环境反馈本身就携带方向信号"的场景。
- **零侵入实现**：AReW 实际改动只是 advantage 加一项常数偏置，centering 性质保证 critic / KL / clipping 都不用改——这是它能 plug 到 PPO/GRPO/GSPO 上都涨点的真正原因。把同样的轨迹内 likelihood-margin 思路用到 process supervision 数据上，可以作为 PRM-free 替代品的一个起点。

## 局限与展望
- **依赖能产生方向反馈的环境**：方向性 critique 的前提是"是否引出新证据"或"自报置信度变化"这类信号能被廉价获取，对于 reward 极其稀疏且环境无反馈接口的任务（如纯文本数学证明）并不直接适用。
- **BT critique 依赖 self-report 的置信度**：$\hat\Psi_t$ 来自 prompt 出来的概率，本身受 LLM 校准能力影响；如果模型严重 over/under-confident，$z_t^{\mathrm{bt}}$ 的符号会失真。文中虽证明对噪声鲁棒，但极端 mis-calibration 下的退化曲线没充分展开。
- **理论假设较多**：Thm. 3.4 用到 Lipschitz reward-belief coupling、action-invariant harmful belief drift、BT-capability-limited absorption、conservative belief propagation 等多个 regularity 条件，真实 LLM 上能否严格满足只能靠现象级证据回填——属于"对的方向但精度有限"的分析。
- **可扩展方向**：把 SeL 框架推广到多 agent 协作（AS/BT 跨 agent 解耦）、把方向性 critique 用 LLM-judge 在线产生（避免规则手工写）、与 process reward model 做互补（高置信用 PRM、低置信退回 directional critique）都是自然延伸。

## 相关工作与启发
- **vs Process Reward Model (PRM)**：PRM 路线（如 Math-Shepherd, Let's verify step by step）追求 step-level 校准 reward；AReW 反其道而行，只要方向性、加性、零校准，论证只要加权准确率 $>1/2$ 就稳赢。两者并不冲突：PRM 给绝对值、AReW 给方向，可以叠加。
- **vs vanilla GRPO/GSPO**：group sampling 通过多 rollout 减方差，但本文证明 SeL 是结构性耦合而非采样问题——这解释了为什么 GRPO 在 agentic 任务上的提升不如在 math 上明显，也指出后续 agentic RL 算法应内置 channel-level 信用分配。
- **vs information-seeking RL (Active RAG / InfoSeek)**：之前的工作多用辅助 reward 鼓励 query diversity 或 information gain，本文不再造 reward，而是从"梯度信号被另一 channel 压低"这一更底层的视角解释为什么这些辅助 reward 经常不稳定。
- **vs Belief-augmented LLM agents**：和把显式 belief 模块塞进 LLM agent 的工作相比，AReW 不改架构、不加模块，只改训练目标——更易落地，但也回避了"BT 到底该怎么显式表示"这个长期开放问题。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 给 agentic RL 失效模式起了一个精确名字并形式化，advantage shaping 的极简注入方式在工程上很有迁移潜力。
- 实验充分度: ⭐⭐⭐⭐ 4 领域 9 任务、2 模型主干、3 种 RL 算法、噪声鲁棒、$\tau^2$-bench 真实工具调用都覆盖，唯一缺憾是缺更大模型尺度（30B+）的验证。
- 写作质量: ⭐⭐⭐⭐ 故事线"观察→理论→方法→验证"非常清晰，theorem-proposition 与实验图表对应紧密；定理假设偏多，可读性对非 RL 背景读者稍硬。
- 价值: ⭐⭐⭐⭐⭐ "方向性 critique + advantage reweighting"很可能成为 agentic RL 的默认 trick，对 deep research / coding agent 训练有立竿见影的工程意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ICML 2026\] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)

</div>

<!-- RELATED:END -->
