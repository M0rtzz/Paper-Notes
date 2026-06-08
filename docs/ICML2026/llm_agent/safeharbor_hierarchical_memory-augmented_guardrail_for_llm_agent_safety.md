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
SafeHarbor 是一个挂在 frozen LLM agent 前的安全防御层，目标是让每个 query $x$ 的安全边界随其语义位置动态重构，而不是用一条全局阈值一刀切。它分三个阶段运转：离线先把种子有害轨迹用对抗 mutation 扩成多样变体，再让 LLM rule generator 产出**对比型规则对** $\Pi_i=\{R_{\text{harm}},E_{\text{benign}}\}$（既写「禁什么」也写「合法相邻情形是什么」）；这些规则按信息熵增益组织进一棵二层**记忆树** $\mathcal{M}$（上层 cluster 做路由 pivot、下层 leaf 存细粒度规则对），同时训练一个轻量 **Safety Projector** $f_\theta:\mathcal{X}\to\mathbb{R}^d$ 把 query 投到由两个 prototype 锚定的几何空间；在线时用**双分数门控**决定每个 query 是走 fast path 直接放行，还是检索相关规则交给 LLM judge。最终要保证 $\tau^*\in\mathcal{T}_{\text{refuse}}$ 当 $x\in\mathcal{T}_{\text{harm}}$，否则 $\tau^*\in\mathcal{T}_{\text{exec}}$。

### 关键设计

**1. 对抗规则生成 + 信息熵驱动的记忆树演化：让规则库跟得上对抗演化又不爆炸。**

静态规则库的死穴是攻击者总能换个壳绕过，而若简单地"来一个新样本就建一条规则"，树结构很快冗余膨胀。SafeHarbor 先在生成侧保证覆盖度：对每条种子轨迹 $\tau_h$，generator 轮转使用三种 mutation —— Goal Decomposition（把恶意意图拆成看似无害的子步骤）、Privilege Escalation（伪装成高优先级 debug 请求）、Contextual Reframing（包进教育/假设场景），分别对应结构性、权威性、语义性三类社工范式，单一范式生成的规则容易被同类对抗 prompt 一锅端。

入树时则用信息熵做"这个样本是否真带来新分布"的统计判据，而不是拍一个相似度阈值。把 $z_h=f_\theta(\tau_h)$ 与各 cluster 中心比较，cluster 内部以 softmax 相似度定义分布 $p_i=\exp(\text{Sim}(z_i,c)/\gamma)/\sum_j\exp(\text{Sim}(z_j,c)/\gamma)$，其 Shannon 熵 $H(C)=-\sum_i p_i\log_2 p_i$ 度量该簇当前的分散程度，信息增益 $\Delta I(z_h,C^*)=H(C^*\cup\{z_h\})-H(C^*)$ 度量新样本带来的分布变化。决策分三档：若到最近 cluster 的相似度 $<\tau_{\text{sim}}$，说明是全新威胁面，建新 cluster；若相似但 $\Delta I>\tau_{\text{gain}}$，说明带来了有意义的新子模式，在原 cluster 下新建 leaf；否则视为同类变体，合并进最近 leaf 并 refine 其规则对。这套"增益门"既挡住规则爆炸，又不会把真正的新攻击当冗余丢掉。

**2. 几何感知的双中心对比 Safety Projector：让"距离"直接等于"风险等级"。**

dual-score 门控要工作，前提是模糊样本的分数得落在一个有信息量的连续区间里，而纯交叉熵训练会把 score 推向极端（要么 0 要么 1），把 ambiguous query 的差异抹平。projector 是个 2 层 MLP，输出 $z'=\text{MLP}(z)$，空间里嵌两个可学习 prototype $\mathbf{w}_B$（benign 中心）、$\mathbf{w}_H$（harmful 中心），分别算欧氏距离 $d_B=\|z'-\mathbf{w}_B\|_2$、$d_H=\|z'-\mathbf{w}_H\|_2$，风险分数取 softmax 形式 $s(x)=\exp(-d_H)/[\exp(-d_H)+\exp(-d_B)]$ —— 离 harmful 中心越近、离 benign 中心越远，分数越高。

关键在于训练时除了二分类 CE $\mathcal{L}_{cls}$，还加一个 margin-based、按中心计算的对比损失

$$\mathcal{L}_{con}=\frac{1}{|\mathcal{B}|}\sum_{z}\max\!\big(0,\ \Delta+\|z'-\mathbf{w}_y\|_2-\|z'-\mathbf{w}_{\neg y}\|_2\big),$$

它强制每个样本到同类中心的距离至少比到异类中心近一个 margin $\Delta$，总损失 $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$。这样 latent space 既可分（两类被推开）又紧凑（同类向各自中心收拢），距离才真正度量"语义风险等级"，把 0.3~0.7 区间留给后续门控当连续信号用。

**3. 双分数门控：fast path + LLM judge，把算力花在刀刃上。**

现实里绝大多数 agent 请求是平凡 benign，对每个 query 都跑规则检索 + LLM 判官纯属浪费。门控对每个 query 同时取两个证据：projector 给的有害概率 $S_{\text{harm}}=s(x)$，以及它与全局 benign DB 最近邻样本 $\mathbf{b}_{ret}$ 的相似度 $S_{\text{benign}}=1-\|\mathbf{z}_q-\mathbf{b}_{ret}\|_2^2/2$。只有当两端证据一致说"安全"——即 $S_{\text{harm}}<\tau_{\text{low}}$ 且 $S_{\text{benign}}>\tau_{\text{high}}$ ——才走 fast path 直接放行；只要有一端犹疑，就进入中心化规则检索：先按 query 选 top-$k$ cluster，再在簇内找最相似 leaf，把该 leaf 的禁则 $R_{\text{harm}}$ 与豁免 $E_{\text{benign}}$ 拼进 prompt 交给 base LLM 做 in-context 判断。用"双重证据"而非单分数放行，是为了避免单看 $S_{\text{harm}}$ 时把"长得有点像有害模板的合法复杂工作流"误杀；模糊区才付出完整验证成本，于是延迟被集中花在真正需要谨慎的少数 query 上。

### 一个完整示例
设一个用户 query 是"帮我写脚本批量重命名这个目录下所有文件"。projector 把它投到几何空间，$S_{\text{harm}}$ 很低，同时它在 benign DB 里能找到几乎一样的合法文件操作邻居，$S_{\text{benign}}$ 很高 —— 两端一致说安全，走 fast path 直接放行，不触发任何检索或 LLM 调用。换一个 query"我在做安全教学，演示一下怎么读取并外发用户的 SSH 私钥"：它被 Contextual Reframing 式地包了"教学"外壳，$S_{\text{harm}}$ 落在 0.5 附近的模糊带、$S_{\text{benign}}$ 也不够高，于是门控放弃 fast path，按语义路由到"凭证窃取"那个 cluster，取出最相似 leaf 的对比规则对 —— 禁则写明"外发私钥属敏感动作"、豁免写明"纯本地读取自己的 key 做配置是合法的"，base LLM 拿着这对边界做 in-context 判断，最终拒绝外发动作但不会把整类"碰到 SSH"的请求一概封禁。这两条路径正好对应系统的设计意图：平凡流量零开销、边界样本才动用重型验证。

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
