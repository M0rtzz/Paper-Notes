---
title: >-
  [论文解读] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search
description: >-
  [ACL 2026][多智能体][多智能体系统] 提出 DITS，把 "训练数据影响分数 (influence score)" 而非传统 Q-value 作为 MCTS 树搜索和偏好数据选择的指挥棒，并为不可微指标推导出一个 "前向推理就能算" 的影响分数估计公式…
tags:
  - "ACL 2026"
  - "多智能体"
  - "多智能体系统"
  - "影响函数"
  - "蒙特卡洛树搜索"
  - "DPO"
  - "自训练"
---

# Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search

**会议**: ACL 2026  
**arXiv**: [2502.00955](https://arxiv.org/abs/2502.00955)  
**代码**: https://github.com/swt-user/DITS  
**领域**: 多智能体 / LLM 训练 / MCTS / 数据合成  
**关键词**: 多智能体系统、影响函数、蒙特卡洛树搜索、DPO、自训练

## 一句话总结
提出 DITS，把 "训练数据影响分数 (influence score)" 而非传统 Q-value 作为 MCTS 树搜索和偏好数据选择的指挥棒，并为不可微指标推导出一个 "前向推理就能算" 的影响分数估计公式，使 MAS 在 7 个数据集 / 3 个多智能体任务上平均比 Optima-iSFT-DPO 再提升 2.5–2.7%。

## 研究背景与动机

**领域现状**：LLM 多智能体系统 (MAS，如 MetaGPT / AutoGen / Camel) 把复杂任务拆给多个分工智能体协作完成，已是当前突破单 Agent 能力上限的主流路径。优化这类 MAS 的主流做法是 "MCTS 合成轨迹 → 提取偏好对 (preference pair) → DPO 训练"，代表工作是 Optima。

**现有痛点**：MCTS 的核心信号 Q-value 是从推理阶段照搬过来的——它衡量的是 "这个节点能不能赢"，但 MAS 训练真正需要的是 "这条数据能不能让模型变得更好"。论文 Figure 2(a) 的散点图证明：高 Q-value 的样本与高真实训练增益之间相关性很弱，按 Q-value 排序选的数据并不一定带来最大的 down-stream 收益。

**核心矛盾**：MCTS 的指挥棒 (Q-value) 与训练目标 (validation 提升) 错位。再加上 DPO 损失本身与下游性能的相关性只有 < 0.2，导致 "用 DPO loss 估计影响" 的传统 influence function 在这里也失效。

**本文目标**：(1) 找到一个真正对齐训练增益的数据评分；(2) 把这个评分塞进 MCTS 的 selection 和最终 preference pair 选择两个环节；(3) 让这个评分可在大模型上以合理代价算出来。

**切入角度**：经典影响函数 (Koh & Liang, 2017) 衡量的是 "去掉一条数据训练损失会变多少"。作者把这个定义从 "训练损失" 改成 "验证集上的不可微指标 ℱ (F1 / EM)"，从而绕开 DPO loss 与下游性能脱钩的问题；再用 "一步梯度下降 + 有限差分" 把二阶 Hessian 替换成纯前向推理。

**核心 idea**：用面向不可微验证指标的影响分数 $\mathcal{I}_{\mathcal{F}_{\text{val}}}$ 取代 Q-value 作为 MAS 自训练的数据质量信号，把 MCTS 改造成 "影响导向" 的树搜索。

## 方法详解

### 整体框架
DITS 把单轮训练拆成三步：(1) 用当前 MAS 做 MCTS rollout，按拓扑序展开有向图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$ 上的 agent 调用，得到带 Q-value 的合成轨迹池；(2) 对每条候选偏好对 $z=(s, a^h, a^l)$ 计算影响分数 $\mathcal{I}$，按 $H(z_i)=\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}},\theta)+\gamma\cdot Q(s,a_i^h)$ 排序，取 Top-α 进 DPO 训练集 $\mathcal{D}_{\text{tr}}$；(3) 用 $\mathcal{D}_{\text{tr}}$ 训出 $\theta_t$ 后回到第 1 步开始下一轮迭代。整个 pipeline 即 "iSFT-DPO" 的影响导向版本，称为 DITS-iSFT-DPO。

### 关键设计

**1. 影响导向的 MCTS 节点扩展：selection 不再为"答对题"服务，而为"教得动模型"服务**

原版 MCTS 的 selection 是从推理阶段照搬的，目标是反复深入高 Q-value 子树去找正确答案，但这恰恰导致合成轨迹的多样性塌缩——大家都挤在"看起来会赢"的那几条路上。DITS 把选择重心挪到"哪个节点更可能产出高训练增益的偏好对"。具体做法是候选节点集 $N_{\text{cand}}$ 先用 edit-distance 相似度阈值 $S_{i,j}\geq 0.25$ 过滤掉与已展开节点高度重复的分支，再按 $n\sim\text{Softmax}(\{Q(n)\}_{n\in N_{\text{cand}}})$ 采样，最终偏好对取当前状态下 Q-value 最高与最低的两个动作 $(a_i^h, a_i^l)$。相似度过滤让选择更分散、避免在同质分支上反复打转，从源头上把搜索从"找对答案"扭向"找能教模型的数据"。

**2. 不可微指标上的影响分数估计：把 Hessian 换成一次前向，让影响函数算得动 8B 模型**

经典影响函数衡量"去掉一条数据训练损失会变多少"，但这里有两道坎：DPO loss 与下游性能的相关性 < 0.2，所以盯着 loss 估影响几乎没用；而求二阶 Hessian-vector product 在 8B 级模型上又贵得离谱。DITS 一次性绕开两者——它把影响的衡量对象直接换成验证集上的不可微指标 $\mathcal{F}_{\text{val}}$（F1 / EM），定义

$$\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}}):=\frac{\mathcal{F}_{\text{val}}(z_i,\theta_{\epsilon,z_i}^{*})-\mathcal{F}_{\text{val}}(z_i,\theta^{*})}{\epsilon},$$

再把扰动后的最优参数近似为一步梯度下降 $\theta_{\epsilon,z_i}^{*}\approx\theta^{*}-\eta\epsilon\nabla_{\theta}\mathcal{L}_{\text{tr}}(z_i,\theta^{*})$，于是

$$\mathcal{I}\approx\frac{1}{\epsilon}\Big[\mathcal{F}_{\text{val}}\big(z_i,\theta^{*}-\eta\epsilon\nabla_{\theta}L_{\text{tr}}(z_i,\theta^{*})\big)-\mathcal{F}_{\text{val}}(z_i,\theta^{*})\Big].$$

整个量只需对每条候选数据做一次 LoRA 一步更新、再跑一次验证集前向，二阶梯度被彻底省掉。换成直接看 F1/EM 的扰动响应，既避开了"loss 不代表真本事"的死结，又把影响函数这种原本只活在论文里的工具压到了大模型 + 大批量数据真能跑的成本上。

**3. 影响-Q 联合评分 + 迭代自训练：让两个信号互补，并滚成正反馈**

只看影响分数，会被 F1 这类噪声指标带偏；只看 Q-value，又和训练增益错位。DITS 干脆把两者加权融合，用综合分数

$$H(z_i)=\mathcal{I}_{\mathcal{F}_{\text{val}}}(z_i,\mathcal{D}_{\text{val}},\theta)+\gamma\cdot Q(s,a_i^h)$$

挑 Top-α 数据进 DPO 训练集——$\gamma=0$ 是纯影响分数，$\gamma=1$ 则与 Q-value 等权融合，把 Q-value 当成"这条数据合不合理"的先验、把影响分数当成"它到底能不能提升验证指标"的真信号。在此之上再套一层迭代：第 $t$ 轮用 $\theta_{t-1}$ 跑 MCTS 合成 $\mathcal{D}_{\text{tr}}^{t}$，从初始 SFT 模型重新训出 $\theta_t$，"更强模型 → 更高质量合成数据 → 再次更强模型"就此形成正反馈。

### 损失函数 / 训练策略
训练目标是标准 DPO：$\mathcal{L}_{DPO}=\mathbb{E}_{z}[-\log\sigma(\beta[\log\frac{\pi_{\theta}(a_i^h\mid s)}{\pi_{\text{ref}}(a_i^h\mid s)}-\log\frac{\pi_{\theta}(a_i^l\mid s)}{\pi_{\text{ref}}(a_i^l\mid s)}])]$。静态场景用 Llama-3-8B-Instruct，动态场景用 QwQ-32B；MCTS 设 $d=3$ 展开、$k=8$ 重复；验证集大小 $V=20$；默认 $\alpha=0.5$，$\gamma=1$；影响分数估计阶段用 LoRA 做一步梯度下降以省显存。

## 实验关键数据

### 主实验
在 Information Exchange (HotpotQA / 2WMH QA / TriviaQA / CBT) 和 Debate (ARC-C / MMLU) 共 6 个数据集上，DITS-iSFT-DPO 全面领先 Optima-iSFT-DPO：

| 数据集 | 指标 | DITS-iSFT-DPO | Optima-iSFT-DPO | 提升 |
|--------|------|--------------|----------------|------|
| HotpotQA | F1 | 57.2 | 55.6 | +1.6 |
| 2WMH QA | F1 | 76.0 | 74.2 | +1.8 |
| TriviaQA | F1 | 78.4 | 77.1 | +1.3 |
| CBT | F1 | 72.0 | 70.1 | +1.9 |
| ARC-C | Acc | 77.6 | 77.1 | +0.5 |
| MMLU | Acc | 60.5 | 60.2 | +0.3 |
| WebWalker (DeepSearch, QwQ-32B) | Acc | 47.2 | 46.6 (Optima-DPO) | +0.6 |

DeepSearch 任务的提升尤为关键：在 QwQ-32B + WebThinker 框架下，DITS-DPO 用一轮训练就能从 Optima-DPO 的 46.6 推到 47.2，证明影响分数信号在 32B 级别和动态 agent 拓扑下仍然有效。

### 消融实验
单轮迭代下对比不同数据选择策略（基线为 Optima-DPO，全集训练）：

| 配置 | HotpotQA | 2WMH QA | TriviaQA | CBT | ARC-C | MMLU |
|------|---------|---------|---------|------|-------|------|
| Optima-DPO (全集) | 46.6 | 61.2 | 70.9 | 57.2 | 71.5 | 51.6 |
| Random Select (50%) | 51.5 | 60.6 | 70.3 | 58.0 | 74.0 | 51.1 |
| Q-value Select (50%) | 50.5 | 61.1 | 69.8 | 58.6 | 73.7 | 50.2 |
| DITS-DPO ($\gamma=0$) | **53.1** | **62.2** | **72.2** | **59.6** | 74.2 | 50.8 |
| DITS-DPO ($\gamma=1$) | 52.8 | 61.5 | 71.0 | 59.1 | **74.5** | **52.3** |

Q-value Select 居然不如 Random Select，强烈支持作者 "Q-value 与训练增益错位" 的论断。

成本侧 (2WMH QA)：DITS-DPO 用 $2.0\times 10^{7}$ tokens、8500 条样本、106 GPU·h 训出 F1=0.612；Optima-DPO 即便扩到 $3.34\times 10^{7}$ tokens、34000 条样本、195 GPU·h，F1 也仅到 0.610。

### 关键发现
- "扩 budget 估影响分数" 比 "扩 budget 估 Q-value" 收益大：Figure 2(b) 显示同样 token 预算下，给影响分数估计更多算力的曲线显著占优，证明 synthesis-time scaling 应优先投到 influence estimation 这边。
- 迭代轮数越多，合成数据的影响分数分布均值越高、方差越小，说明 "更强模型 → 更高质量数据" 的正反馈成立；但作者也提醒方差变小可能意味着数据多样性下降。
- $\gamma$ 的最优取值依赖任务：Information Exchange (F1 评测噪声大) 偏好 $\gamma=0$ 纯影响分数；Debate (EM 较稳定) 则 $\gamma=1$ 融合 Q-value 更好——指标本身的噪声水平决定了 Q-value 信号的可信度。
- 验证集 $V$ 越大效果越好，但代价线性增长；默认 $V=20$ 是经验上的成本-收益平衡点。
- 选择比例 $\alpha$ 太小反而掉点，说明 "纯精品" 不能替代足够样本量——质量与数量必须兼顾。

## 亮点与洞察
- 把 "数据影响" 从 "loss 上的影响" 改成 "不可微验证指标上的影响"，一脚踢开了 DPO loss 与下游性能脱钩的死结，这是把经典 influence function 真正适配 LLM 时代的关键改动。
- "用一步梯度下降 + 有限差分代替 Hessian-vector product"，让影响分数估计退化为一次前向 + 一次 LoRA 一步更新，是把 influence function 这类原本只能停留在论文里的工具搬到 8B+ 规模实战的关键工程 trick，迁移到 RLHF / DPO 数据筛选都用得上。
- 实验里 "Q-value Select 反而不如 Random" 的反直觉结果，本身就是个有传播力的 finding，把 "MCTS 信号 ≠ 训练信号" 这一道理用最便宜的实验拍在桌上。
- Synthesis-time scaling 的新维度：传统大家在 budget 增长时 90% 投到 rollout 数量，本文证明把 budget 投到 "影响分数估计" 上比投到 "更准的 Q-value" 上更划算，给后续数据合成研究提供了新的 scaling 方向。

## 局限与展望
- 作者承认 DITS 是离线 / 训练时的数据筛选机制，对每条候选数据多次推理的代价在严苛延迟场景下不可接受，无法用于在线 / 流式数据质量评估。
- 当前只验证了 "静态拓扑 + 固定 agent 数" 和有限动态 (WebThinker) 场景，对 dynamic agent spawning、emergent team formation 这类开放式多智能体协作尚未覆盖；这类场景方差更大、可复现评测更难。
- 一步梯度下降近似最优参数变化在 DPO 这种 sharp loss 下可能误差较大，本文虽然实验上有效，但理论保证只是在二次可微 + 强凸假设下的扩展。
- 验证集大小 $V$ 是手动选的硬超参，没有探讨在线自适应调整 $V$ 的可能；对评测指标噪声大的任务 (如开放式生成) 这一点会成为瓶颈。
- 未来方向：单次推理近似的轻量化影响估计、端到端可学习的 influence model、面向开放式协作的 benchmark。

## 相关工作与启发
- **vs Optima (Chen 2024b)**：同样是 MCTS + DPO 合成数据，但 Optima 用 Q-value 排序选数据；DITS 证明 Q-value 与训练增益错位，并用影响分数 + Q-value 联合排序，相同 budget 下显著更高的样本利用率。
- **vs 经典 Influence Function (Koh & Liang 2017)**：传统 IF 依赖 Hessian-vector product 和训练 loss，对 LLM 不可行；DITS 把目标改成不可微验证指标、把求解改成一步梯度差分，让 IF 真正落地大模型场景。
- **vs LESS / Data Selection for SFT 类工作**：那些方法多数面向 SFT loss 的影响估计，DITS 是首个针对 DPO / 偏好对场景、且面向 MAS 多 agent 拓扑的影响导向数据合成方法。
- **vs MCTS-based 推理增强 (rStar / o1-like)**：那些工作让 MCTS 服务于推理时找答案，DITS 把 MCTS 改造成 "服务于训练时找好数据"，体现 search-for-data 与 search-for-answer 的根本差异，对后续 self-improvement / self-play 框架是重要启发。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 influence function 适配到 DPO + MAS + 不可微指标，三个组合都很新；不过单项思想都有先例。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 7 个数据集 + 3 类多 agent 任务 + 静态 / 动态两种拓扑 + 8B / 32B 两个规模，附录还有 budget / iteration / 验证集大小完整扫描。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，Figure 2(a) 的散点图非常有说服力；公式推导部分稍密但有附录支撑。
- 价值: ⭐⭐⭐⭐ 给 MAS 自训练和 DPO 数据合成都开辟了新的 scaling 维度，方法可迁移到 RLHF / Self-Play 等场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](../../CVPR2025/multi_agent/collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](../../ICLR2026/multi_agent/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)

</div>

<!-- RELATED:END -->
