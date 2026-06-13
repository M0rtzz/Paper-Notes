---
title: >-
  [论文解读] GRPO is Secretly a Process Reward Model
description: >-
  [ICML 2026][LLM推理][GRPO] 本文从理论上证明 GRPO + ORM 在"组内轨迹共享前缀"的温和条件下**等价于**一个带有 Monte-Carlo PRM 的过程奖励 RL 目标，从而揭示出 vanilla GRPO 隐藏的一个 bug——前缀长度不均会让高奖励轨迹的大部分 token 拿到负 advantage——并提出 $\lambda$-GRPO 做一个 PRM-aware 归一化，在推理 benchmark 上稳定超过 GRPO 且训练快约 2 倍。
tags:
  - "ICML 2026"
  - "LLM推理"
  - "GRPO"
  - "过程奖励"
  - "优势归一化"
  - "数学推理"
  - "RL 训练加速"
---

# GRPO is Secretly a Process Reward Model

**会议**: ICML 2026  
**arXiv**: [2509.21154](https://arxiv.org/abs/2509.21154)  
**代码**: https://github.com/coli-saar/grpo-prm/ (有)  
**领域**: LLM 推理 / 强化学习  
**关键词**: GRPO, 过程奖励, 优势归一化, 数学推理, RL 训练加速

## 一句话总结
本文从理论上证明 GRPO + ORM 在"组内轨迹共享前缀"的温和条件下**等价于**一个带有 Monte-Carlo PRM 的过程奖励 RL 目标，从而揭示出 vanilla GRPO 隐藏的一个 bug——前缀长度不均会让高奖励轨迹的大部分 token 拿到负 advantage——并提出 $\lambda$-GRPO 做一个 PRM-aware 归一化，在推理 benchmark 上稳定超过 GRPO 且训练快约 2 倍。

## 研究背景与动机
**领域现状**：在 LLM 数学推理 RL 训练里，PRM（过程奖励模型）能给每个中间步骤打分，credit assignment 比 ORM（结果奖励）细得多，因此通常配合 PPO + GAE 使用。GRPO（DeepSeekMath）的卖点是**砍掉了 critic 和 GAE**，用组内 reward 的标准化作为 advantage——简单、省显存，因此被广泛应用（tool use、RLHF、math reasoning），但因为没了 GAE，几乎所有 GRPO 工作都只能用 ORM。

**现有痛点**：把 PRM 接进 GRPO 需要对算法做非平凡修改（如 TreeRPO、GroupPRM、TreeRL），增加了实现复杂度并放弃了 GRPO 简洁的卖点。此外，神经 PRM 训练贵（要 step-level 标注）且容易被 reward-hack。

**核心矛盾**：大家一直把"GRPO 用 ORM"和"PRM-aware RL"当作两件不同事来处理。但 GRPO 在 rollout 阶段会从同一个 prompt 抽多条 trajectory，这些 trajectory 天然形成一棵前缀共享树——这棵树本身就携带了 process-level 信息，只是从来没被显式利用。

**本文目标**：(1) 在数学上证明 vanilla GRPO 在共享前缀的假设下就是一个 PRM-aware 目标，量化它对应的 PRM 到底长什么样；(2) 用这把分析工具找 GRPO 目标的隐藏 bug；(3) 修复这个 bug 而不引入显式 PRM。

**切入角度**：作者注意到一个简单事实——一个 trajectory 的优势 $a_i$ 在 GRPO 里被均匀分配到它的所有 token；如果这条 trajectory 与组内多条**高分** trajectory 共享一段长前缀，那么这段前缀实际上是"好的"——但 vanilla GRPO 因为只用一条 trajectory 的整体 reward 做 advantage，会把这段前缀错算成"坏的"。从前缀树的视角推导，问题就豁然开朗。

**核心 idea**：把 GRPO 看成"在前缀树上做 MC-PRM 的 RL"，识别出归一化项里的不对称性，加上一个简单的 $\lambda$ 因子修正。

## 方法详解

### 整体框架
全文是一条"先证等价、再借等价诊断 bug、最后修 bug"的理论链。第一步在两个温和假设下，把同一 prompt 抽出的一组 trajectory 按共享前缀组织成一棵前缀树，证明 vanilla GRPO 的 loss 恒等于一个建立在这棵树上的 Monte-Carlo PRM 目标——也就是说 GRPO 暗地里一直在做 process-level credit assignment。第二步用这个 PRM 视角找出 vanilla GRPO 归一化里的不对称，第三步加一个 $\lambda$ 因子修复它，得到 $\lambda$-GRPO，一行代码即可落地。

### 关键设计

**1. 前缀树 $\mathcal B(\mathbb G)$：把 GRPO 翻译成一个 MC-PRM 目标**

要说清 GRPO 隐含的 process reward，先得把"哪些 token 属于同一个 process step"形式化。作者对组 $\mathbb G=\{y^{(1)},\dots,y^{(|\mathbb G|)}\}$ 定义 process set $\mathcal B(\mathbb G)=\{\lambda\subseteq\mathbb G\mid \exists n\geq 0,\forall y^{(i)},y^{(k)}\in\lambda: y_{:n}^{(i)}=y_{:n}^{(k)}\}$：每个 $\lambda$ 是一组共享同一段前缀的 trajectory，按 $\supseteq$ 关系它们自然形成一棵树，节点 $\lambda$ 对应一段 step、跨度 $[s(\lambda), e(\lambda))$。这段 step 的奖励就取组内这些轨迹的均值 outcome reward $r_\lambda = \frac{1}{|\lambda|}\sum_{y^{(i)}\in\lambda} r^{(i)}$，advantage 仍按组均值归一化。关键结论是：在 $\mu=1$（每批只更新一次）、DAPO 风格 token-level objective、忽略 clip 这三个温和假设下，这个 MC-PRM-aware loss $L_{\text{PRM}}(\mathbb G)$ 数值上恒等于 $L_{\text{GRPO}}(\mathbb G)$。这条等价性第一次给"GRPO 到底在干什么"赋予了 PRM 语义——想要 process reward 不必再训神经 PRM 或改算法，**只要 rollout 时让 trajectory 共享前缀**，MC-PRM 信号就免费送上门。作者还在 Section 3.2 实证这种前缀共享在真实 GRPO 训练里非常普遍，所以这个"隐含 PRM"几乎总是非平凡的。

**2. 缺陷诊断：advantage 与 step 频次错配，同时损害 exploitation 与 exploration**

有了 PRM 视角，vanilla GRPO 的一个系统性 bug 就显形了。考虑图 1 的 trajectory JKLNQU，假设它整体 reward 高于组均值，但它的前缀 JKL 又和多条低分轨迹共享。在 PRM-aware 视角下，JKL 这段 step 的 step-reward 等于"JKL 之下所有轨迹的均值 reward"，被那些低分轨迹拉低，于是 JKL 三个 token 拿到的是**负** advantage，只有 JKLNQU 独占的最后一个 token U 拿到正 advantage。可 vanilla GRPO 的 token-level loss 把整条轨迹当一体，所有 token 共享同一个 sample-level $a_i$，这恰恰违背了 PRM 视角"分段 advantage 应按 step 出现频次加权"的要求——具体地，loss 的分母 $\sum_{y^{(i)}}\text{len}(y^{(i)})$ 在 token 数与 step 频次失配时会注入系统性 bias。作者强调这个错配在**不同条件下会同时拖累 exploitation 和 exploration**：JKLNQU 是损害 exploitation 的典型——把已知的高分推理链压低；而在 reward 与 step 频次的另一种失衡下，错配又会反向扭曲探索信号、抑制 exploration。这条诊断把"GRPO 偶尔会把好轨迹搞砸、甚至反而降低正确推理链概率"的模糊直觉，转化成了一条能在白板上画出来、能形式化的 bug。

**3. $\lambda$-GRPO：一个 PRM-aware 归一化因子**

修法很轻：保持原 GRPO 的 sample-level advantage 不动，只把 token 累加时的分母从"全组 token 数"换成按前缀树节点频次重加权的归一化项，等价于给每个 token 乘上 $\lambda_t = 1/n_t$，其中 $n_t$ 是该 token 所属 process step 在组里出现的次数。这样高频共享的 step token 不会因为反复出现而被反复推动，恢复了"共享 step 不该被重复 penalize / 重复 reward"的对称性。它既保住了 GRPO 不需要 critic / GAE 的轻量优势，又用上了那份已经免费拿到的 MC-PRM 信号，一行 patch 即可塞进 TRL 的 GRPO trainer。作者实测它对每步训练时间几乎零开销，却能在多个推理 benchmark 上稳定优于 vanilla GRPO，且**收敛快约 2 倍**——说明把 bug 修掉后梯度信号确实更干净。

### 损失函数 / 训练策略
- Vanilla GRPO（在 $\mu=1$、DAPO token-level 假设下）：
  $L_{\text{GRPO}}(\mathbb G)=\frac{1}{\sum_{y^{(i)}}\text{len}(y^{(i)})}\sum_{y^{(i)}}\sum_t (P_{i,t}\cdot a_i - D_{i,t})$，其中 $a_i=(r^{(i)}-r_{\text{mean}}(\mathbb G))/r_{\text{std}}(\mathbb G)$。
- $\lambda$-GRPO：将分母替换为 PRM-aware 的归一化和（按 process step 频次加权），其余保持不变。
- 训练设置：与 DeepSeekMath GRPO 一致，$\mu=1$ 更新次数；在数学推理 SFT 数据上做 RL；TRL 框架；2× 训练加速来自更快达到峰值 validation acc。

## 实验关键数据

### 主实验

| 设置 | 训练时间 | 下游推理 acc | 收敛速度 |
|------|---------|-------------|---------|
| Vanilla GRPO | $1\times$ baseline | baseline | baseline |
| $\lambda$-GRPO | 几乎相同/步 | 全面 $>$ baseline | 达 peak 快约 $2\times$ |
| 显式 PRM (PPO+GAE) | 远慢 | 受 reward-hack 影响 | 慢 |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 共享前缀比例高的 group | $\lambda$-GRPO 提升明显 | 隐含 PRM 信号丰富 |
| 共享前缀稀薄（多样化 rollout） | $\lambda$-GRPO 退化为 GRPO | 与理论一致：trivial PRM 时无差异 |
| 去掉归一化重加权（保留树视角不改 loss） | 性能恢复到 GRPO | 证明性能增益来自 $\lambda$ 修正而非视角本身 |

### 关键发现
- **GRPO 的隐含 PRM 在真实训练里几乎总是非平凡**：作者用实证表明前缀共享在 group rollout 中频繁发生（图 1 的 JKLNQU 这种结构是常态而非特例），因此分析有现实意义。
- **bug 的方向是系统性反例**：vanilla GRPO 倾向于把高 reward 轨迹的"早期共享前缀"打成负 advantage，这是为什么 GRPO 训出来的模型有时**反而降低**正确推理 chain 出现概率（exploitation 受损）的根因之一；在另一类 reward / step 频次失衡下，同一错配又会损害 exploration——两个方向都源于同一个归一化偏差。
- **$\lambda$-GRPO 的收敛加速比性能提升更显著**：peak validation acc 早约 2× 步数到达，意味着实际 GPU 时间节省巨大；这一点对工业 RL pipeline 价值很高。
- **不需要任何额外标注或额外 forward**：与神经 PRM 相比，零标注成本；与显式 MC-PRM (VineRL) 相比，零额外 rollout。

## 亮点与洞察
- **"算法等价性"作为分析工具**：把 vanilla GRPO 改写成 PRM-aware 形式，然后在改写后的形式里诊断 bug，再改回 vanilla 框架修复——这是非常优雅的"理论先行"分析范式，值得任何 RL 算法分析借鉴。
- **前缀树视角免费的 MC-PRM**：揭示出"想要 process reward 不必再训 PRM，只要让 rollout 共享前缀"，对希望节省 PRM 标注的人来说是颠覆性结论。
- **bug 的具象化**：作者举的 JKLNQU 例子把抽象的归一化错配变成了一个能在白板上画出来的反例，可读性极强。
- **$\lambda$ 修正一行代码**：trick 可以直接 hot-patch 到 TRL/verl 等主流框架，落地成本几乎为零。

## 局限与展望
- 等价性证明依赖 $\mu=1$ 和 token-level (DAPO) loss 假设；在 sample-level GRPO 或 $\mu>1$ 多更新的设置下，前缀树视角不再严格成立，bug 是否仍按同方向出现需要额外讨论。
- 实验在数学推理 benchmark 上做，对 RLHF / tool use / agent 等其他 GRPO 主战场没系统验证。
- 隐含 PRM 的质量依赖前缀共享密度，长 trajectory 或多样化 sampling（高 temperature）下会变稀疏；可能需要主动鼓励前缀共享的 rollout 策略才能保证 $\lambda$-GRPO 持续受益。
- 缺乏与 TreeRPO / GroupPRM 等显式过程奖励 GRPO 变体的端到端对比，无法判断显式 vs 隐式哪个更好。

## 相关工作与启发
- **vs TreeRPO / TreeRL（feng2025, ji2025）**: 它们显式构造树结构 PRM 并改算法；本文证明 vanilla GRPO 自己就是这种树 PRM 的一个特例，且 $\lambda$-GRPO 用更少改动达到类似目标。
- **vs VinePPO / treeRL（MC-based PRM）**: VinePPO 用 MC rollout 估计 step value 但需要额外 forward；本文把 MC 估计藏在 group 内的共享前缀里，零额外开销。
- **vs DAPO**: DAPO 提出 token-level objective 解决 sample-level loss 训练不稳；本文把 DAPO 作为前提之一，正交地修复另一个归一化 bug。
- **启发**：任何"在 group / batch 内做相对评分"的算法（如 DPO 的成对比较、RLAIF 的多样本聚合）都可以套用"前缀/共享子结构 → 隐含 process signal"的视角去检查归一化是否存在系统偏差。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "GRPO 是隐式 PRM" 是一个非常漂亮且此前完全未被注意到的等价性结论
- 实验充分度: ⭐⭐⭐ 推理 benchmark 上的提升和加速实证清晰，但跨任务（RLHF/tool use）的系统验证缺失
- 写作质量: ⭐⭐⭐⭐⭐ 用图 1 的 JKLNQU 反例把抽象 bug 讲透，从假设、证明到修复全链条干净
- 价值: ⭐⭐⭐⭐⭐ 一行 patch 拿到 ~2× 训练加速 + 稳定性能增益，工业 RL pipeline 立即可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](reward_modeling_from_natural_language_human_feedback.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
