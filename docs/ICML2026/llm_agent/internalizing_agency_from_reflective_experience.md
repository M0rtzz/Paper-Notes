---
title: >-
  [论文解读] Internalizing Agency from Reflective Experience
description: >-
  [ICML 2026][LLM Agent][agentic LLM] 本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "agentic LLM"
  - "反思经验"
  - "回溯探索"
  - "经验蒸馏"
  - "Pass@k"
---

# Internalizing Agency from Reflective Experience

**会议**: ICML 2026  
**arXiv**: [2603.16843](https://arxiv.org/abs/2603.16843)  
**代码**: 未公开  
**领域**: LLM Agent / 长程交互训练  
**关键词**: agentic LLM, 反思经验, 回溯探索, 经验蒸馏, Pass@k

## 一句话总结
本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力，在 CodeContests、WebShop、ALFWorld 等长程任务上把 Pass@128 拉高最多 14%，远胜 GRPO 等 outcome-driven RL。

## 研究背景与动机

**领域现状**：LLM 从被动回答转向自主 agent，常用的 post-training 方法是 RL with verifiable rewards（RLVR / GRPO）——多轮采样、终局给一个 scalar 奖励、用策略梯度推高成功轨迹的概率。

**现有痛点**：在长程交互场景里，终局 scalar 奖励信息密度极低。多数 rollout 拿不到奖励，更新被少数已经能成功的样本主导，模型学到的只是把「已经会做的」做得更稳；环境其实在每一步都给了丰富反馈（错误信息、状态转移、编译错误），但全被压缩成 0/1 信号扔掉。结果就是 distribution sharpening：Pass@1 涨、Pass@1024 不动甚至掉。

**核心矛盾**：分布锐化 vs agency 内化是两件事。要真的扩大模型能解的问题集合，模型得学会「我现在轨迹失败了，错在哪一步，怎么修」，而 outcome-driven 训练只教它「这条轨迹整体好」。

**本文目标**：把「识别关键决策点 → 在该点回滚 → 用环境反馈做有针对性的修正」这一恢复程序内化进模型权重，而不是在推理时靠 best-of-$k$ 重试或 Tree-of-Thoughts 外挂搜索。

**切入角度**：与其只奖励整条成功轨迹，不如显式制造失败案例、找出错误位置、监督修正动作。环境反馈不再被压成 scalar，而被结构化成自然语言的「诊断 + 修复指令」（experience summary）作为训练监督。

**核心 idea**：用反思生成「失败→回滚→修正→成功」的实验性轨迹，再 SFT 蒸馏 post-rollback 的修正动作，从而把 recovery agency 写进权重。

## 方法详解

### 整体框架
LEAFE 两阶段：(1) Tree-Based Experience Generation with Rollback：在 base policy $\pi_\theta$ 上 rollout 轨迹，每若干步触发一次反思，反思模块判断当前轨迹是否在偏离，若是则产出回滚点 $\tau$ 和经验摘要（包含失败诊断 + 修复建议），从 $\tau$ 重启并 branch 出一个或多个修正动作；(2) Experience Distillation：从所有「最终成功」的修正轨迹中提取「在 $\tau$ 处的修正动作」作为目标 token，做 SFT，使模型在推理时即使不显式提供 experience，也能在类似失败信号下输出对应修正。

### 关键设计

1. **基于树的反思回滚式经验生成**:

    - 功能：把单轨迹 rollout 扩展成一棵「失败检测 + 反事实修正」搜索树，让一条失败 trace 衍生出多条「修正后成功」trace 当训练数据。
    - 核心思路：在 ReAct 范式下，每个时间步 $t$ 的状态为 $h_t=(o_0,a_0,\ldots,o_t)$，动作 $a_t\sim\pi_\theta(\cdot|h_t,q)$。每隔若干步插入一次反思 prompt，模型自身判断要不要回滚；若要回滚就输出 (i) 回滚点 $\tau$，(ii) 自然语言 experience summary $e$（含「错在哪+怎么改」）；然后用 $\pi_\theta(\cdot|h_\tau,q,e)$ 采样新动作开 branch。一条原始 rollout 可衍生多个 (failure→rollback→fix→success) 三元组。
    - 设计动机：scalar reward 看不到「哪一步出错」，但 LLM 本身有阅读环境反馈、定位错误位置的能力，把这种 in-context 能力外化成显式回滚 + 修正的训练信号，比 GRPO 的 group-relative reward 信息密度高得多。

2. **经验到策略的蒸馏**:

    - 功能：把第一阶段生成的修正动作蒸馏进模型权重，使部署时无需 experience prompt 也能自然做出修正。
    - 核心思路：对每条 (failure→rollback→fix→success) 轨迹，从回滚点 $\tau$ 起截取「带 experience 提示生成的修正后子轨迹」，构造 SFT 数据 $(h_\tau, a^{\rm fix}_\tau,\ldots,o_T)$，**训练时不喂 experience，只让模型在 $h_\tau$ 后直接产出修正动作序列**。这样在测试时，模型遇到类似失败模式就能自己内生地切换到修正模式，而不需要外部 reflection prompt。
    - 设计动机：要么模型每次推理都跑昂贵的反思 + 重试（部署成本高），要么把这个能力存到权重里。蒸馏的关键是「条件化在没有 experience 的 $h_\tau$ 上去预测修正动作」，强迫模型从环境反馈自身推断要做什么修正。

3. **对比 GRPO：稀疏奖励 vs 决策级监督**:

    - 功能：在同等交互预算下提供 decision-level 的 reflect→revise 监督，而非 episode-level 的标量评分。
    - 核心思路：GRPO 把同一 prompt 的 $G$ 条 trace 算 group-relative advantage $\hat{A}_i=(r_i-\bar{r})/\sigma_r$ 后做 policy gradient，本质是给某条 trace 整体加权；LEAFE 直接给「回滚后应该输出什么」这种逐 token 监督。前者鼓励 exploit 已知成功模式，后者把行为分布往新区域推。
    - 设计动机：在长程任务上 GRPO 容易 sharpening 到 base 模型 long tail 里已经会的少数模式，Pass@k 大 k 不动；显式纠错监督能扩大覆盖。

### 损失函数 / 训练策略
Stage 1 完全用 base policy 自采样 + 反思生成数据，无梯度更新；Stage 2 标准 SFT 交叉熵 $\mathcal{L}=-\sum_t \log \pi_\theta(a^{\rm fix}_t|h_t)$，损失只算修正后的 action token，不算环境反馈 token。实验里反思频率、回滚预算等超参在 appendix 给出。

## 实验关键数据

### 主实验
5 个 agentic benchmark：CodeContests（程序合成 + 执行反馈）、WebShop（购物 agent）、ALFWorld（家务 agent）、ScienceWorld（科学探索）、Sokoban（推箱子）。所有方法在固定交互预算下评测。

| 任务 | 指标 | Base | GRPO | Early Exp. | LEAFE | 相对最强 baseline |
|--------|------|------|----------|---|---|---|
| CodeContests | Pass@1 | base | 略涨 | 略涨 | 显著涨 | 提升 |
| 长程任务平均 | Pass@128 | base | ≈base | + | +14% | +14% |
| 通用 | Pass@1 | base | + | + | ++ | 一致领先 |

### 消融实验

| 配置 | Pass@1 | Pass@128 | 说明 |
|------|---------|---------|------|
| Base | 低 | 低 | 不做 post-training |
| GRPO (outcome RL) | 中-高 | ≈Base | 典型 sharpening |
| Early Experience (无回滚) | 中 | 中 | 只蒸馏成功轨迹，无 recovery 信号 |
| LEAFE w/o rollback | 中 | 中 | 去掉树式 branch，退化为线性 SFT |
| LEAFE w/o experience summary | 中-高 | 中-高 | 仅有「修正动作」没诊断说明 |
| **完整 LEAFE** | **高** | **高 (+14%)** | 完整框架 |

### 关键发现
- 大 $k$ 才能暴露真正差异：Pass@1 上 GRPO 也能涨，但 Pass@128 上 GRPO 几乎不动，LEAFE 一骑绝尘——说明 RLVR 真的只是把已有支持集里的高频模式锐化，没有扩展覆盖。
- 经验摘要（experience summary）和回滚是协同的：单独去掉任一项都会让 Pass@128 大幅下降，说明「诊断 + 修正动作」共同构成了 decision-level 监督。
- LEAFE 训出的模型即使在推理时不被提示「请反思」，也会在内部自发触发修正——验证了 agency 真被 internalize 到了权重里。
- 数据效率高于 Early Experience：在相同 SFT 样本量下，LEAFE 的回滚式构造把「失败 trace」也利用起来，每条 failure 平均产出多条成功子轨迹。

## 亮点与洞察
- 「distribution sharpening vs agency internalization」是个非常清晰的概念切分，能帮社区跳出「Pass@1 涨就好」的评估陷阱。
- 把环境反馈结构化成「自然语言诊断 + 修正建议」是个可复用的 pattern：在 tool use、code agent、web agent 等任何「环境会喊错误」的场景都适用。
- 蒸馏时**不喂** experience 但训练时**喂**了——这种「训练辅助、推理自洽」的设计让模型必须从环境信号自身推导出修正逻辑，比单纯把 experience 当 prompt 喂泛化更好。

## 局限与展望
- 反思触发频率、回滚预算都是超参，对不同任务可能要重调；自动决定何时反思仍是开放问题。
- 反思模块依赖 base policy 自己的 self-assessment 能力，若 base 太弱（如 7B 以下小模型）可能根本认不出失败。
- 实验主要在「环境反馈丰富 + 有 verifier」的场景；在反馈稀疏（如开放对话）或反馈延迟的场景需进一步验证。
- 没有和 best-of-$N$ + 自反思（如 Reflexion）做直接计算开销对比，部署阶段两者真实成本差异需更系统化测算。

## 相关工作与启发
- **vs GRPO / DeepSeek-R1 风格 RLVR**：本文同样在 LLM 上做 post-training，但用结构化反思替代 scalar reward；GRPO 大 $k$ 不涨、本文涨。
- **vs Early Experience**：Early Experience 也用 reflective trajectory 但不做回滚式分叉，等于只蒸馏成功 trace 没利用 failure 信号；本文进一步用失败 trace 当数据源。
- **vs Reflexion / Tree-of-Thoughts**：那些方法把反思 / 树搜索留在推理时，每次都要多轮调用；本文把这种 agency 内化进权重，推理时单 pass 即可。

## 评分
- 新颖性: ⭐⭐⭐⭐ 「反思生成数据 + 蒸馏到权重」组合不算原创，但 explicit rollback + decision-level supervision 的框架化和 Pass@k 视角是清晰贡献
- 实验充分度: ⭐⭐⭐⭐ 5 个 agentic benchmark 覆盖 coding/web/家务/科学/推箱子，但缺与 Reflexion 等 inference-time 方法直接成本对比
- 写作质量: ⭐⭐⭐⭐ 「sharpening vs internalization」叙事清晰，Pass@k 曲线极有说服力
- 价值: ⭐⭐⭐⭐ 给 agentic LLM post-training 提供了和 RLVR 互补的新范式，部署成本低，方法可复用到 tool / code agent 多个场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](../../ACL2026/llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ACL 2025\] R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](../../ACL2025/llm_agent/r2d2_reflective_agentic_memory.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)

</div>

<!-- RELATED:END -->
