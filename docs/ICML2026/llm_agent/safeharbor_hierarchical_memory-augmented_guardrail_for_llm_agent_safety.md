---
title: >-
  [论文解读] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety
description: >-
  [ICML 2026][LLM Agent][Guardrail] SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 6…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Guardrail"
  - "Agent Safety"
  - "Hierarchical Memory"
  - "对比学习"
  - "Over-Refusal"
---

# SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety

**会议**: ICML 2026  
**arXiv**: [2605.05704](https://arxiv.org/abs/2605.05704)  
**代码**: [ljj-cyber/SafeHarbor](https://github.com/ljj-cyber/SafeHarbor)  
**领域**: AI 安全 / LLM Agent  
**关键词**: Guardrail, Agent Safety, Hierarchical Memory, 对比学习, Over-Refusal

## 一句话总结
SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 63.6%，显著缓解 over-refusal 问题。

## 研究背景与动机
**领域现状**：LLM Agent 能调用工具、执行真实操作（写文件、发邮件、调 API），但攻击面也从「输出有害文本」扩大到「执行有害动作」。主流防御要么是 (i) 用辅助 LLM 监控运行时（GuardAgent、ShieldAgent），要么是 (ii) fine-tune 安全模型（AgentAlign、Llama-Guard-3），要么是 (iii) 静态规则匹配。

**现有痛点**：以上方案都把安全边界视为「全局固定的线性切分」—— 一旦想严防恶意 prompt 就连带封禁了相似但合法的 benign 复杂工作流，导致严重 over-refusal；而引入辅助 Agent 又会带来 prohibitive latency（例如 ShieldAgent 要实时跑代码生成）。

**核心矛盾**：safety strictness 与 utility on benign tasks 之间存在尖锐 trade-off；越严越易过拒，越宽越易被绕过 —— 根本原因是「边界本身不随上下文动态调整」。

**本文目标**：在不重训 base model、不增加重型 Agent 代理的前提下，给 LLM Agent 装一个「能随每个 query 动态重构安全边界」的防御层，同时把延迟控制在可接受范围。

**切入角度**：把安全规则看作「按语义分簇的局部边界」而不是全局阈值；通过检索式动态规则注入 + 训练一个轻量 Safety Projector 把语义空间几何化，让边界由 query 自身的位置决定。

**核心 idea**：用一棵自组织的「分层记忆树」存放对抗式生成的禁则与豁免对，配合一个由对比损失训练的双中心 MLP Projector 提供 harmful/benign 双分数，最后用「快速路径 + 模糊区 LLM 判官」的门控决定是否触发完整安全验证。

## 方法详解

### 整体框架
SafeHarbor 把 query $x$ 处理分三个阶段：(I) **对抗规则生成** —— 离线把种子有害轨迹通过 mutation 生成多样对抗变体，再用 LLM rule generator 产出对比型规则对 $\Pi_i=\{R_{\text{harm}},E_{\text{benign}}\}$；(II) **双知识存储** —— 把规则有结构地组织进二层记忆树 $\mathcal{M}$（上层是 routing pivot，下层 leaf 存细粒度规则对），同时训练一个 Safety Projector $f_\theta:\mathcal{X}\to\mathbb{R}^d$，引入两个可学习的 prototype $\mathbf{w}_B,\mathbf{w}_H$；(III) **在线检索与评分** —— 用 dual-score gating，多数 benign query 走 fast path 直接放行，模糊/高风险 query 才检索相关规则 + 调 LLM judge。形式化的轨迹目标是 $\tau^*\in\mathcal{T}_{\text{refuse}}$ 若 $x\in\mathcal{T}_{\text{harm}}$，否则 $\tau^*\in\mathcal{T}_{\text{exec}}$。

### 关键设计

1. **对抗规则生成 + 信息熵驱动的记忆树演化**:

    - 功能：自动把零散有害样例扩展为覆盖三类社工范式的规则库，并按信息增益决定「新建 cluster / 新建 leaf / 合并精修」。
    - 核心思路：对每条种子轨迹 $\tau_h$，generator 轮转使用 Goal Decomposition（拆解恶意意图）、Privilege Escalation（伪装高优先级 debug 请求）、Contextual Reframing（包装在教育/假设场景下）三种 mutation 生成多样变体；然后把 $z_h=f_\theta(\tau_h)$ 与现有 cluster 中心算 cosine 距离。判定逻辑用 Shannon 熵：$p_i=\exp(\text{Sim}(z_i,c)/\gamma)/\sum_j\exp(\text{Sim}(z_j,c)/\gamma)$，$H(C)=-\sum p_i\log_2 p_i$，信息增益 $\Delta I(z_h,C^*)=H(C^*\cup\{z_h\})-H(C^*)$。若到最近 cluster 的相似度 $<\tau_{\text{sim}}$ 则建新 cluster；若 $\Delta I>\tau_{\text{gain}}$ 则在原 cluster 下新建 leaf；否则合并并 refine 最近 leaf 的规则对。
    - 设计动机：传统静态规则库无法跟上对抗演化，而依赖单一相似度阈值又容易导致树结构爆炸或冗余合并；信息熵增益给出了「这个样本是否真的带来新分布」的统计标准，避免规则膨胀也防止漏报。

2. **几何感知的双中心对比 Safety Projector**:

    - 功能：把 query 映射到一个由 $\mathbf{w}_B,\mathbf{w}_H$ 两个可学习 prototype 锚定的几何空间，让距离直接反映 harmful 程度。
    - 核心思路：projector 是个 2 层 MLP，输出 $z'=\text{MLP}(z)$，分别算 $d_B=\|z'-\mathbf{w}_B\|_2, d_H=\|z'-\mathbf{w}_H\|_2$，最终风险分数 $s(x)=\exp(-d_H)/[\exp(-d_H)+\exp(-d_B)]$。训练时同时优化二分类 CE $\mathcal{L}_{cls}$ 与 margin-based center-wise 对比损失 $\mathcal{L}_{con}=\frac{1}{|\mathcal{B}|}\sum_z \max(0,\Delta+\|z'-\mathbf{w}_y\|_2-\|z'-\mathbf{w}_{\neg y}\|_2)$，总损失 $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$。
    - 设计动机：纯 CE 会让 score 极端化（要么 0 要么 1），抹平模糊样本的差异；引入 margin 对比损失迫使 latent space 既可分又紧凑，让距离真正度量「语义风险等级」，从而支撑后续 dual-score 门控。

3. **双分数门控：fast path + LLM judge**:

    - 功能：在保持精度的同时把 90% 以上的普通流量从重型 LLM 推理里释放出来。
    - 核心思路：每个 query 同时算 (i) projector 给的有害概率 $S_{\text{harm}}$，与 (ii) 与全局 benign DB 最近邻样本 $\mathbf{b}_{ret}$ 的相似度 $S_{\text{benign}}=1-\|\mathbf{z}_q-\mathbf{b}_{ret}\|_2^2/2$。若 $S_{\text{harm}}<\tau_{\text{low}}$ 且 $S_{\text{benign}}>\tau_{\text{high}}$ 则走 fast path 直接放行；否则进入「中心化规则检索」—— 先选 top-$k$ cluster，再在 cluster 内找最相似 leaf，把该 leaf 的禁则 $R_{\text{harm}}$ 与豁免 $E_{\text{benign}}$ 拼成 prompt 交给 base LLM in-context 判断。
    - 设计动机：现实中绝大多数 agent 请求是平凡 benign，对所有 query 都做规则检索 + LLM 判官是浪费；fast path 只放行「双重证据都说 benign」的 query，模糊区才付出复杂验证成本，把延迟集中花在最值得的地方。

### 损失函数 / 训练策略
仅训练 projector：$\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$，base LLM 完全 frozen；记忆树是 training-free 离线构建 + 在线 self-evolution。整个系统 plug-and-play，可挂在任意 frozen LLM agent 前。

## 实验关键数据

### 主实验
基于 GPT-4o 与多个 base LLM，在 benign request 与 harmful request 上同时评测「Score / Full pass / Refusal / Non-Refusal」。

| Model | Method | Harmful Refusal ↑ | Benign Score ↑ | 评价 |
|-------|--------|------------------|---------------|-----|
| GPT-4o | No Defense | 58.0% | 44.2% | over-permissive |
| GPT-4o | Rule Traverse | 100.0% | 12.1% | 严重 over-refusal |
| GPT-4o | **SafeHarbor** | **93%+** | **63.6%** | 最佳 trade-off |

SafeHarbor 是表中唯一同时把「harmful refusal > 93%」与「benign utility > 60%」两端都做到的方案。

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 完整 SafeHarbor | 93%+ refusal / 63.6% benign | 主结果 |
| 去掉 $\mathcal{L}_{con}$ 对比损失 | benign score 下滑 | margin 对比是几何区分关键 |
| 去掉 fast path | latency 显著上升 | fast path 是延迟优化核心 |
| 关闭记忆自演化（固定规则库） | 长期 attack 通过率上升 | 信息熵驱动的合并/分裂必要 |
| 仅用单分数 ($S_{\text{harm}}$ only) | over-refusal 回归 | benign similarity 是降低误杀关键 |
| 朴素 MoE/线性分类 | 边界模糊样本错判 | 双中心几何空间提供更强语义结构 |

### 关键发现
- 对抗式 rule generation 用三种 social engineering 范式轮转，保证规则库覆盖结构性（多步分解）、权威性（特权升级）与语义性（场景包装）三类攻击 —— 单一范式生成的规则容易被同类对抗 prompt 一锅端。
- 信息熵门 $\Delta I$ 比固定相似度阈值更能区分「新威胁」与「同类变体」—— 既避免规则爆炸，也避免漏掉真正的新攻击面。
- 双 prototype 的几何空间让 ambiguous query 的 score 真正落在 0.3~0.7 区间，给 fast path / LLM judge 的门控提供了有信息量的连续度量。

## 亮点与洞察
- 把「per-query 重构安全边界」做成了一个工程上可落地的轻量结构（projector + 记忆树），整套系统 training-free 即可挂载到 GPT-4o 这种 closed-source LLM 上。
- 对比型规则对 $\{R_{\text{harm}},E_{\text{benign}}\}$ 是缓解 over-refusal 的精妙设计 —— 同一 leaf 不仅说「禁什么」还说「合法相邻情形是什么」，迫使 LLM judge 明确豁免边界而不是一概拒绝。
- 信息熵驱动的记忆演化机制可移植到任何「需要不断纳入新模式但不能让索引爆炸」的检索增强系统（如 RAG 知识库、ToolBench）。
- Fast path 思想（用便宜的双分数把大部分流量挡在重型验证之外）应作为所有 LLM-as-a-Judge guardrail 的标配。

## 局限与展望
- 评估的「harmful score」依赖 LLM-based judge $\mathcal{M}_{\text{eval}}$，存在 judge 模型本身的偏差与上限。
- 三种 mutation 范式（Goal Decomp/Privilege/Contextual Reframing）是固定的； 面对未知类型的攻击（如多模态注入、long-horizon planning attack），覆盖性还需要后续工作进一步度量。
- 记忆树长期演化下的「漂移」与「遗忘」未充分讨论 —— 持续运行数月后会不会被对抗 prompt 灌入污染规则？
- Fast path 的两阈值 $\tau_{\text{low}},\tau_{\text{high}}$ 是经验设置，未给出自适应策略；在不同 domain 下重新校准的代价没量化。
- Benign DB 需要预先准备一个大且干净的合法 query 库，对小众场景可能不可得。

## 相关工作与启发
- **vs AgentAlign**: AgentAlign 通过 SFT 把安全约束烧进模型，需要重训且产生 retrain cost；SafeHarbor training-free 且与任意 frozen LLM 兼容。
- **vs Llama-Guard-3**: 后者是静态内容分类器，不感知 agent tool execution 上下文；SafeHarbor 直接定义 trajectory 级别的安全。
- **vs GuardAgent / ShieldAgent**: 前者每次都要在线生成代码再执行，延迟高且维护脆弱；SafeHarbor 用轻量 projector + 记忆检索绕开 code-gen，端到端延迟低得多。
- **vs A-Mem 等记忆机制**: A-Mem 关注时间感知的知识网络，本文提出的是「时间无关、约束驱动」的安全记忆，可对照参考 misevolution 问题（Shao et al. 2025）。
- 启发：双中心几何 + margin 对比的「prototype anchored embedding」思想可迁移到 RAG retrieval 的安全过滤、多模态内容审核等场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把信息熵记忆演化 + 对抗规则对引入 LLM agent guardrail
- 实验充分度: ⭐⭐⭐⭐ 多 LLM + 多攻击范式，但缺少 long-horizon 与多模态攻击的覆盖
- 写作质量: ⭐⭐⭐⭐ 三阶段框架图清晰，Algorithm 1 写得很标准
- 价值: ⭐⭐⭐⭐⭐ training-free 可直接挂在 GPT-4o 之类 closed LLM 上，工程落地性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](../../ACL2026/llm_agent/shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_agent/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)

</div>

<!-- RELATED:END -->
