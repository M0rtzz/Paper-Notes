---
title: >-
  [论文解读] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents
description: >-
  [ICML 2026][LLM安全][Reasoning-Level DoS] OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "Reasoning-Level DoS"
  - "红队攻击"
  - "工具调用劫持"
  - "推理 payload 优化"
  - "ICL 遗传搜索"
---

# OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents

**会议**: ICML 2026  
**arXiv**: [2605.08876](https://arxiv.org/abs/2605.08876)  
**代码**: https://github.com/llm2409/OTora  
**领域**: LLM Agent 安全 / 红队攻击  
**关键词**: Reasoning-Level DoS, 红队攻击, 工具调用劫持, 推理 payload 优化, ICL 遗传搜索  

## 一句话总结
OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源，再在该资源里投放经 ICL 遗传搜索优化的「思考型 payload」）让 LLM agent 进入持续多轮的过度推理状态，在 WebShop / Email / OS 三类 agent 上实现 10× 推理 token 膨胀和数量级延迟攻击，且最终任务准确率几乎不变。

## 研究背景与动机

**领域现状**：现有 LLM 安全研究大致分三类——(i) 越狱攻击让模型输出违规内容；(ii) agent 行为劫持让 agent 调错工具或泄露数据；(iii) 过度思考研究观察到误导性输入可让推理模型多用 token。系统侧另有传统 DoS / 应用层 DoS。

**现有痛点**：上述工作都聚焦「正确性」或「行为偏移」，**漏掉了一个根本失效模式——agent 输出仍然正确，但因被诱导做了大量无意义推理而违反延迟/SLA/成本预算，从可用性角度上构成拒绝服务**。这在真实 LLM agent 部署中尤其致命，因为工业系统普遍有严格 timeout 和 cost budget。

**核心矛盾**：传统 DoS 攻击的特征（流量洪泛、错误输出）容易被检测；而推理级 DoS 的「输出正确 + 延迟暴增」让所有基于输出正确性或安全策略的检测系统都失灵——你拦不住，因为它没做错任何事。

**本文目标**：(a) 正式定义 R-DoS 威胁模型；(b) 构造一个**自动化**、**统一**、**白/黑盒兼容**的红队框架来稳定实例化 R-DoS；(c) 量化它对真实 agent 系统的影响并讨论防御。

**切入角度**：agent 普遍把「工具返回」和「环境观察」当作可信输入。因此攻击可以分成两阶段：先诱导 agent 主动去 fetch 一个攻击者控制的 URL（外部访问触发），然后让攻击者预放置在该 URL 里的 payload 自动让 agent 陷入计算密集型推理。两阶段分离的原因是：注入到指令或第三方环境的通道太窄太嘈杂，没法稳定投放长 payload；但一旦 fetch 成立，攻击者就完全控制了内容，能可靠地投递任意复杂的 payload。

**核心 idea**：「窄通道触发 + 宽通道投放」分阶段红队 + 「持续策略段」让单次劫持转化成多轮持续过度思考。

## 方法详解

### 整体框架
OTora 是一个两阶段的端到端管线（Algorithm 1）：**Stage I** 在用户指令或环境观察里注入一个对抗后缀 $s$，让 victim agent $\mathcal{M}$ 自然把「访问 attacker.test」纳入其 ReAct 行动计划；**Stage II** 在已被访问的 attacker.test 上预放置经多目标遗传搜索优化过的推理 payload $r$，让 agent 在拿到 fetched content 后多轮、持续地陷入过度思考，但保持任务最终输出仍然正确。整个流水线对白盒和黑盒 agent 都可用，黑盒下用 API top-$k$ logprobs 或代理模型替代梯度。

### 关键设计

1. **Stage I：注意力感知的插入点打分 + 动态目标共进化**:

    - 功能：在 agent 响应序列中找到最适合插入对抗后缀的位置，并同时让「目标 token 串」自己沿响应分布演化以更容易被触发。
    - 核心思路：定义位置打分函数 $r_j(t) = \tfrac{1}{|t|+1}(\alpha M_j(t) + \beta P_j(t) + \lambda A_j(t,s))$，三项分别是「该位置前缀对目标 token 串的匹配数」「匹配 token 在分布 $\mathcal{P}$ 下的平均概率」「生成时分配给后缀 $s$ 的平均 attention」。第三项是为了避免「看似匹配但其实是上下文先验贡献」的伪信号。然后做**动态目标共进化**：不固定目标短语，而是用 $\mathcal{P}$ 的高概率 token + 辅助 LLM 生成一组语义等价候选 $\mathcal{T}$，选 $t^\star = \arg\max_{t^{(k)}}\max_j r_j(t^{(k)})$；再做加权区间调度选 top-$\ell$ 不重叠插入点。最后梯度下降（白盒）或 log-prob 搜索（黑盒）优化 $s$ 最大化 $\sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$。
    - 设计动机：固定目标短语限制了搜索空间，且 agent 自己的语言风格未必匹配人工设定的短语；共进化让目标随 agent 的响应分布漂移，触发率显著提升。attention 项把伪匹配（看似有目标 token 但其实跟 $s$ 无关）滤掉，让优化收敛更稳定。

2. **Stage II：Agent-aware 持久化 payload + ICL 引导遗传搜索**:

    - 功能：让一次劫持产生**跨多轮**的推理膨胀，而不是只让单步响应变长。
    - 核心思路：把 payload 分解成两段——**局部 sink 段**（在被劫持的那一步抛出一个复杂数学/逻辑题让 agent 当场算），**持久策略段**（注入到 agent 历史的元指令，引导后续每个 Thought 都用更繁琐的推理风格）。这是利用 ReAct agent 的关键性质：每个新 Thought-Action 都条件在整段交互历史上，一旦持久段被写入历史，未来回合自然复读 → 单次劫持变多轮放大。优化目标是多目标分 $\mathrm{Score}(r) = w_1 S_{\text{RTI}} + w_2 S_{\text{FID}} + w_3 S_{\text{STAB}}$，分别衡量推理 token 膨胀、最终任务保真度、跨种子稳定性（用 -Var 表达），三者等权 $w_i=1.0$。优化后端是黑盒遗传搜索，每代用 ICL-capable 模型 $\mathcal{M}_{\text{ICL}}$ 在 agent 上下文条件下 mutate 顶级 payload。
    - 设计动机：传统过度思考攻击只增加单步 token，对多轮 agent 影响有限；「局部 + 持久」二段结构和「上下文感知的 ICL 变异」让攻击形成持续作用的策略，而不是一次性 cost。多目标分确保不能用「明显垃圾 payload」蒙混过关——必须同时保证任务保真和跨种子稳定。

3. **白/黑盒统一接口与 fidelity 评估**:

    - 功能：让同一框架既能在白盒（有 logits 和 attention）下用梯度优化，也能在 GPT-3.5/Gemini 这种黑盒 API 下退化为概率搜索。
    - 核心思路：把第 1 项算法的 attention $A_j$ 在黑盒下置零（$\lambda=0$）或用代理模型 attribution 近似；第 3 项 gradient-based 优化换成 log-prob 反馈下的离散搜索。fidelity 评估上引入 $\mathrm{ASR}_S$（目标 token 序列出现率）、$\mathrm{ASR}_H$（agent 实际产生有效 tool 调用率）、Hit（Stage II 内容是否被执行）、Accuracy（任务正确率），多指标解耦触发可靠性和延迟放大。
    - 设计动机：现实部署中很多 victim 都是黑盒 API，攻击器必须能优雅退化；同时为了避免「攻击成功率统计夸大」，把「触发可靠」和「持续放大」拆开评估，用乘积 $\mathrm{ASR}_H \times \mathrm{Hit}$ 估计端到端有效率。

### 损失函数 / 训练策略
Stage I 目标是 $\max_s \sum_{j\in\mathcal{J}}\log p(t^\star\mid x,o,s,z_{[:j]})$，白盒用类 GCG 的离散梯度搜索，黑盒用 log-prob 搜索。Stage II 是黑盒多目标遗传搜索，无梯度，每代评估 + ICL mutate；权重 $w_1=w_2=w_3=1$。

## 实验关键数据

### 主实验
基准 agent：WebShop（购物 agent）+ InjecAgent 的 Email/OS（系统 agent）；backbone 涵盖 LLaMA-70B、GPT-OSS-120B、Gemini-1.5-Flash、GPT-3.5-Turbo。摘要中给出的核心结论：

| 指标 | OTora 结果 |
|---|---|
| Reasoning Token Inflation | 最高 10× |
| End-to-end latency 放大 | 数量级 slowdown |
| 任务 accuracy 变化 | 接近 baseline（preservation of correctness） |
| Stage I trigger ASR_H | 在 Gemini-1.5-Flash WebShop 黑盒下显著优于 SNES 等基线 |

Stage I 黑盒实验（Table 1 节选）以 ASR_S/ASR_H 与 Iters 为指标，OTora 的插入感知打分 + 共进化方案在多个 (model, agent) 组合里都拿到 best，且平均迭代次数下降。

### 消融实验

| 配置 | 影响 |
|---|---|
| 移除 attention 项（$\lambda=0$） | 优化稳定性下降，黑盒下基本退化为纯似然搜索 |
| 固定目标短语（无共进化） | 触发成功率明显下降 |
| 仅 sink 段（无持久策略段） | RTI 单步增加但多轮膨胀消失，端到端 slowdown 弱很多 |
| Score 改为单目标 RTI | FID 急剧下降——产出经常破坏任务正确性 |
| ICL-guided 变异 → 随机 mutation | 遗传搜索收敛慢、payload 质量差 |

### 关键发现
- 「持久策略段」是把单次劫持转成跨轮放大的关键设计，去掉之后整体 slowdown 量级即降。
- attention 感知打分对优化稳定性贡献显著，尤其在白盒可获取 attention 的设置下。
- 防御侧（budgeted reasoning、relevance filtering、runtime monitoring）能部分缓解但无法根治 R-DoS，尤其对 low-and-slow 缓慢渗透模式无效。

## 亮点与洞察
- 第一次正式形式化「推理级 DoS」威胁模型，把传统 DoS 思维迁移到了 LLM agent 的「reasoning budget」维度，为安全研究开了新口子。
- 「窄通道触发 + 宽通道投放」两阶段分解是处理一切 prompt injection 通道限制的通用模式，可直接迁移到其他需要长 payload 的攻击或防御研究。
- 「持久策略段 + ReAct 历史复用」揭示了 ReAct agent 的一个本质漏洞：历史拼接架构天然放大了任何能注入历史的攻击。

## 局限与展望
- 评估集中在 WebShop / Email / OS 三类 agent，对更复杂的多智能体协作系统（如 AutoGen）、强工具链（MCP）的迁移性需要验证。
- 持久策略段依赖 ReAct 历史拼接，对截断历史或采用纯 state-based memory 的 agent 框架可能效果减弱。
- 黑盒攻击假设有 top-$k$ logprob 可用，闭源 API 越来越关闭这个接口，限制了实际可行性。
- 论文给出的防御讨论较初步，未系统设计专门针对 R-DoS 的检测器（如 reasoning anomaly detection）。

## 相关工作与启发
- **vs 越狱攻击（GCG / Pliny）**：越狱以「让模型输出违规内容」为目标，操作输入-输出层；OTora 保持输出正确性，攻击运行时 budget，触发的是完全不同的检测面。
- **vs Agent 行为劫持（如 InjecAgent）**：行为劫持改变行动（误调工具、泄露数据），靠安全过滤即可拦；OTora 不改行动，仅改推理路径长度，所有安全过滤都直接绕过。
- **vs Overthinking 攻击**：传统 overthinking 在单步 QA 模型上膨胀 token；OTora 把它升级到 multi-turn agent 上并提出「持久策略段」让攻击跨轮放大。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义 R-DoS 威胁模型并系统化求解，开新方向
- 实验充分度: ⭐⭐⭐⭐ 多 agent / 多 backbone / 白黑盒覆盖广，但深度防御实验偏少
- 写作质量: ⭐⭐⭐⭐ 算法盒子和定义清晰，但符号略密集，初读需要反复对照
- 价值: ⭐⭐⭐⭐⭐ 揭示真实部署 LLM agent 的可用性安全盲区，对系统设计有警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)
- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](../../ICLR2026/llm_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)

</div>

<!-- RELATED:END -->
