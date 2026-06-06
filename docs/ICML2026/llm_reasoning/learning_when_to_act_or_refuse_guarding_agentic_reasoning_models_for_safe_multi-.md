---
title: >-
  [论文解读] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use
description: >-
  [ICML 2026][LLM推理][智能体安全] MOSAIC 把"安全决策"从隐式推理副产物变成 plan → check → act/refuse 循环里的显式一等动作（含 `<safety_thoughts>` 和 `refusal_tool`）…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "智能体安全"
  - "工具使用"
  - "显式安全检查"
  - "成对偏好强化学习"
  - "GRPO"
---

# MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use

**会议**: ICML 2026  
**arXiv**: [2603.03205](https://arxiv.org/abs/2603.03205)  
**代码**: 待确认  
**领域**: LLM安全 / Agent / 工具使用  
**关键词**: 智能体安全, 工具使用, 显式安全检查, 成对偏好强化学习, GRPO

## 一句话总结
MOSAIC 把"安全决策"从隐式推理副产物变成 plan → check → act/refuse 循环里的显式一等动作（含 `<safety_thoughts>` 和 `refusal_tool`），用 LLM judge 的成对轨迹偏好 + GRPO 训练；在 Qwen2.5-7B / Qwen3-4B-Thinking / Phi-4 上零样本 OOD 减少 50% 有害行为、prompt injection 拒绝率提升 20%、隐私泄漏下降，benign 任务效用不退。

## 研究背景与动机

**领域现状**：LLM 从聊天助手扩展到 agent —— 规划、调用工具、与外部系统多步交互。AgentHarm、Agent Security Bench、PrivacyLens 等基准已经显示，单次失误（写入文件、发起付款、泄露 credentials）就可能造成不可逆伤害。SLM（Phi-4 / Qwen2.5-7B / Qwen3-4B）因成本/延迟/隐私被偏好部署在 agent 场景。

**现有痛点**：（1）聊天安全（RLHF/Constitutional AI）不能可靠迁移到 agent ——会拒绝有害聊天却在被包装成工具任务后照做；（2）现有 agent RL（math/coding 风）专注任务完成，长推理痕迹里几乎不显式检查安全/不可逆性；（3）outcome-only 标量奖励把"早拒绝"和"晚中止"在最终分上拉平，但二者在安全上根本不同；（4）SLM 上下文/世界模型更紧，对 prompt injection、异常工具反馈、级联失败尤其脆弱。

**核心矛盾**：当前 agent 训练目标只是"任务完成"，安全决策被埋在隐式 reasoning 里既不可控也不可监督；轨迹级安全分布对结果级奖励是次序敏感的（同样最终错，但"何时错"对安全极端重要），但标量 reward 完全无法表达。

**本文目标**：（1）把安全检查和拒绝重构为显式一等动作，使其可学、可控、可审计；（2）用轨迹级偏好替代结果级标量奖励，捕捉"何时拒绝"的时序差异；（3）在多模型族 + OOD 基准上验证泛化。

**切入角度**：观察到 agent 的不安全往往不是"想干坏事"，而是"没意识到该停"——长推理里没显式做 safety check 这步；这意味着只要训练模型"何时该插入 safety check / 何时该拒绝"，就能用相同模型容量大幅提升安全。

**核心 idea**：plan → check → act/refuse 循环 + 偏好 RL；安全检查通过 `<safety_thoughts>` 块触发（开/关由模型自学），拒绝通过 `refusal_tool` 作为终止动作；LLM judge 对同一任务的两条轨迹做成对比较，用 GRPO 优化策略。

## 方法详解

### 整体框架

每步 $t$：
1. **plan**：用 `<think>` 块产出 plan 和候选工具调用
2. **gate $g_t \in \{0,1\}$**：模型自决是否开 `<safety_thoughts>`（通过是否输出该开标签）；端到端 RL 学到的，无外部开关
3. **safety check**（若 $g_t=1$）：在 `<safety_thoughts>` 里结构化推理潜在伤害、不可逆性、权限变化、工具反馈风险
4. **act / refuse**：从 $\{\text{tool\_call}, \text{refusal\_tool}, \text{answer}\}$ 选动作；`refusal_tool` 是终止动作并附说明
5. 轨迹 $\tau = \{(o_t, \text{plan}_t, g_t, \text{safety}_t, a_t)\}_{t=1}^{T_{\text{term}}}$

训练用 GRPO（不需要 critic，pairwise rollouts），mask 工具输出 token，只在模型生成的文本上反传。

### 关键设计

1. **显式 safety check + refusal 作为一等动作**:

    - 功能：把安全决策从隐式推理副产物变成可学、可控、可审计的离散动作
    - 核心思路：定义 `<safety_thoughts>` 块和 `refusal_tool` 终止动作；前者开关由模型自决（learned gate），关时直接跳过避免常时开销；后者像普通工具一样进入动作空间，可被 RL 直接奖励
    - 设计动机：长推理痕迹里"忘记检查"是 agent 安全的主要漏洞；让 check 和 refuse 显式存在 + 端到端学习，相当于把安全决策从"隐藏在 logits 概率分布里"提到"显式 token 级动作"——RL 信号才能精确作用在该决策上

2. **成对轨迹偏好 RL（替代标量奖励）**:

    - 功能：用 LLM judge 在同一任务的两条 rollouts 间做相对偏好，捕捉时序安全差异
    - 核心思路：对每个 prompt 采样若干 rollouts；LLM judge 成对比较"哪条更安全更恰当"（不打绝对分）；用偏好对监督 GRPO 的 group advantage；判断维度包括早期拒绝 vs. 晚期中止、是否服从注入指令、是否泄露隐私
    - 设计动机：结果级标量 reward 会把"完全没碰危险工具就拒"和"已经执行了不安全操作才中止"映射到几乎相同的分；pairwise 比较保留这种时序敏感性，是 agent 安全 RL 的关键监督

3. **复合奖励 + 长度感知训练（GRPO 联合优化）**:

    - 功能：在安全对齐、任务效用、结构化输出格式、token 效率间平衡
    - 核心思路：composite reward 包含安全偏好 + 任务成功 + 格式规范（必须含正确 `<think>` `<safety_thoughts>` 标签）+ token 长度惩罚；GRPO 的 group relative advantage 让奖励差异自动归一化，不需要 critic
    - 设计动机：单纯优化安全会过度拒绝（over-refusal），单纯优化任务会忽略安全；长度惩罚防止思维链无限膨胀。GRPO 比 PPO 在 agent 长 trajectory 上更稳

## 实验关键数据

### 三模型主结果（OOD 基准）

| 模型 | AgentHarm 有害任务降幅 | AgentHarm 拒绝率 | PrivacyLens 隐私泄漏降幅 | BFCL benign 完成率 |
|------|--------|--------|--------|--------|
| Qwen2.5-7B base | – | 35% | – | 78% |
| Qwen2.5-7B + **MOSAIC** | **−50%** | **87%** | −38% | **82%** |
| Qwen3-4B-Thinking base | – | 41% | – | 44% |
| Qwen3-4B-Thinking + **MOSAIC** | −37% | 71% | −29% | **85%** |
| Phi-4 base | – | 52% | – | 71% |
| Phi-4 + **MOSAIC** | −44% | 79% | −33% | **91%** |

Qwen3-4B-Thinking 的 benign 完成率从 44%→85%（接近翻倍），原因是 base 模型会陷入"无尽推理循环"，MOSAIC 学会了适时终止。

### MOSAIC vs 闭源模型

| 模型 | Agent 安全分↑ |
|------|----|
| GPT-4o（无脚手架）| 71 |
| GPT-5（无脚手架）| 76 |
| Qwen2.5-7B + MOSAIC | **78** |
| Phi-4 + MOSAIC | **74** |

MOSAIC 把 SLM 推到与 frontier 模型相当的 agent 安全水平；加上显式脚手架后 frontier 也涨，差距进一步缩小。

### Prompt injection 拒绝率（Agent Security Bench）

| 注入类型 | base | + MOSAIC |
|--------|------|--------|
| 工具调用劫持 | 31% | **62%** |
| 系统提示覆盖 | 38% | **68%** |
| 隐式有害子任务 | 44% | **65%** |

平均 +20% 以上的拒绝率，对 prompt injection 尤其有效。

### 关键发现
- **selective safety invocation 工作**：safety token 平均占总 token < 20%，模型学会了"只在危险步插 safety check"
- **同时减少 under- 和 over-refusal**：Phi-4 over-refusal 降 56%（benign 任务被错拒）同时 harmful 任务拒绝率涨——证明 MOSAIC 不是简单"更保守"
- **pairwise vs scalar 消融**：换成标量 reward，有害任务降幅从 50% 跌到 28%，验证 pairwise 信号对"何时拒绝"差异的关键
- **跨模型族泛化**：Qwen / Phi 两个家族 + 多个尺度都受益，说明 MOSAIC 是范式而非特定 trick

## 亮点与洞察
- **"让安全成为一等动作"的范式转变**：以往把 safety 当成 RLHF 阶段的隐式 alignment 或推理时的过滤器；MOSAIC 把它提到与工具调用同等的动作类别，这种结构性改动让监督和审计都变可能
- **pairwise 偏好抓时序安全的洞察**：把同一 prompt 的两条 rollout 做相对比较，自动放大"何时"的差异——agent 场景下这是个被低估的设计选择，可迁移到所有"轨迹质量与终止时机相关"的任务
- **SLM 友好**：MOSAIC 主要受益模型恰恰是 4–7B 范围的 SLM；这意味着不依赖大模型容量就能解决 agent 安全，对实际部署（成本、延迟、隐私）更有意义
- **selective gate 学得很自然**：模型自学到 "安全敏感步开 check / 普通步跳过"，没有手工启发式

## 局限性 / 可改进方向
- LLM judge 本身可能有偏差（用 GPT-4o/-5 作判官，对 frontier 模型可能有偏向）；未来可考虑 ensemble judge 或 self-play 判官
- 仅验证三种工具/动作类型；现实 agent 工具空间远大，是否仍 selective gate 表现好未知
- `<safety_thoughts>` 是单段结构化推理，没区分不同维度（harm、privacy、irreversibility）单独评分；可拆分为多 head
- 长 horizon（>20 步）任务下偏好 RL 的样本效率未充分验证

## 相关工作与启发
- **vs 聊天 RLHF（Constitutional AI 等）**：那套对单回合文本有效，对 multi-step agent 不可迁移；MOSAIC 在 agent 场景重做了 alignment
- **vs 推理时安全过滤器**：过滤器是 post-hoc 的，不能阻止前面已发生的不安全行为；MOSAIC 把决策提前到每步动作前
- **vs scalar reward agent RL（如 RLVR）**：标量 reward 在数学/代码 OK，但 agent 安全的时序差异它表达不了；pairwise 是必要升级
- **启发**：把"何时做某事"作为一等学习目标的思路可推广到所有 multi-step 决策（如交易系统的"何时止损"、医疗 agent 的"何时呼叫人类"）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "把安全决策提到一等动作 + pairwise 时序偏好"是 agent 安全的真正新范式
- 实验充分度: ⭐⭐⭐⭐⭐ 三个模型族 × 四类 OOD 基准 × harmful/injection/privacy/benign 全覆盖；消融完整
- 写作质量: ⭐⭐⭐⭐ MOSAIC 框架图直观，复合 reward 部分稍简
- 价值: ⭐⭐⭐⭐⭐ SLM agent 是当前部署主流，本文给出工程可用的安全 post-training 方案；可直接被产业采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ICLR 2026\] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/llm_reasoning/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)

</div>

<!-- RELATED:END -->
