---
title: >-
  [论文解读] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models
description: >-
  [ICML 2026][强化学习][奖励模型] 本文系统测量五个高质量 RM（含 SOTA Skywork-Reward-V2）的长度、不确定性、位置、谄媚、模型风格五类偏置，把它们划分为"低复杂度（线性可修）"和"高复杂度（线性不可修）"两类…
tags:
  - "ICML 2026"
  - "强化学习"
  - "奖励模型"
  - "奖励黑客"
  - "线性探针"
  - "零空间投影"
  - "RM 偏置"
---

# One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models

**会议**: ICML 2026  
**arXiv**: [2603.03291](https://arxiv.org/abs/2603.03291)  
**代码**: https://github.com/drfein/OneBiasAfterAnother (有)  
**领域**: 对齐RLHF / AI 安全  
**关键词**: 奖励模型, 奖励黑客, 线性探针, 零空间投影, RM 偏置

## 一句话总结
本文系统测量五个高质量 RM（含 SOTA Skywork-Reward-V2）的长度、不确定性、位置、谄媚、模型风格五类偏置，把它们划分为"低复杂度（线性可修）"和"高复杂度（线性不可修）"两类，并提出 mechanistic reward shaping —— 用 DiffMean 线性探针在最后一层隐藏态上做零空间投影 —— 在不掉 RewardBench2 准确率的前提下显著缓解前三类偏置且能 OOD 泛化到 best-of-N。

## 研究背景与动机

**领域现状**：RLHF 是当前对齐 LM 的主流方法，但 RM 充当代理奖励时极易被 policy 学到的 reward hacking 利用，长度、位置、过度自信、谄媚等偏置已被多次记录。现有补救要么改训练数据、要么加 length penalty、要么训 robust RM，多数仍把偏置当作线性虚假相关来处理。

**现有痛点**：(1) 即便最新 SOTA RM（Skywork-Reward-V2 系列、AllenAI-Llama-8B）仍持续出现旧偏置，且为修长度偏置训练出的 RM 反而出现"惩罚冗长"的过修正——把简洁错误答案排得比正确长答案还高；(2) 现有 post-hoc 修法（如 length penalty）依赖对偏置函数形式的显式建模，换 prompt-conditioned 场景（best-of-N）就失效；(3) 没有人系统区分"哪些偏置确实是线性 spurious correlation 能修"和"哪些是 entangled 需要更深干预"，导致一刀切方法浪费在解决不了的高复杂度偏置上。

**核心矛盾**：偏置在 RM 激活空间里可能与有用信号共线（co-linear），单方向干预要么修不动，要么把好信号也削掉。

**本文目标**：(i) 在最新 RM 上重新审计已知偏置 + 挖新偏置；(ii) 给出"线性可修 vs 不可修"的实证分类；(iii) 设计对低复杂度偏置生效、数据高效、模型内（不改 policy 优化器）、可 OOD 泛化的干预方法。

**切入角度**：基于 Park et al. (2024a) 的线性表示假设——高层概念在表示空间中近似为线性方向。如果某个偏置主要由单一线性方向承载，那么把这个方向 null 掉就能局部去偏；如果偏置和真信号已经在同一子空间纠缠，线性 null 自然无效，这本身就是一个有用的诊断信号。

**核心 idea**：用一对"有偏 vs 无偏"对照样本做 difference-of-mean 探针，把探针方向在 RM 最后一层隐藏态里做 null-space projection（即 mechanistic reward shaping），既能修偏置又能识别哪些偏置本质上修不动。

## 方法详解

### 整体框架
方法分三步：(1) 偏置审计 —— 在 PlausibleQA / BigBench / GSM8K-MC / MMLU 上对五个 RM（Skywork-Llama-8B、Skywork-Qwen-8B/0.6B、AllenAI-Llama-8B、DeBERTa-large-v2）系统度量五类偏置；(2) 对每类偏置构造对照数据集，提取 RM 最后一层非 padding token 的隐藏态，用 DiffMean 算线性探针 $\mathbf{p}$；(3) 推断时对每个新输入的隐藏态 $\mathbf{h}$ 做 $\mathbf{h}_{\text{null}} = \mathbf{h} - \sum_k \alpha (\mathbf{p}_k^{\top}\mathbf{h})\mathbf{p}_k$，再喂给 reward head 得到去偏奖励。多探针时先 Gram-Schmidt 正交化再联合 null。

输入：prompt-completion pair；输出：去偏后的标量 reward。整条干预完全发生在 RM 内部，不需要重训 RM 也不需要改 policy 优化算法，因此天然适配 best-of-N、red-teaming、data filtering 等所有以 RM 为基座的对齐技术。

### 关键设计

**1. 偏置复杂度二分类：先判断一个偏置到底"修不修得动"**

前面提到一刀切的去偏方法常常浪费在解决不了的偏置上，第一个设计就是给所有偏置一个实证准则：把它们分成低复杂度（线性可修，如长度 / 不确定性 / 位置）和高复杂度（线性不可修，如谄媚 / 模型风格）。这里的"mechanistic"取 Saphra & Wiegreffe (2024) 的窄义——只问"在激活空间识别并移除某个方向，能否对下游 reward 行为带来可测量的因果改变"，不追求 circuit-level 解释。判据很直接：如果某偏置的主导信号能被单一线性方向近似，null 掉它就会在不损害基线准确率的前提下显著缩小目标偏置；若 null 之后偏置纹丝不动，恰好说明它在 RM 激活空间里和质量信号 co-linear、纠缠在同一子空间，需要更深的方案。作者还用 Iterative Nullspace Projection (Ravfogel et al., 2020) 在表示层提供独立证据（附录 C.9）。这样设计的好处是把有限资源先投到能拿下的低复杂度偏置上，同时把"修不动"本身变成一个可发表的实证结论，而不是一次失败的尝试。

**2. DiffMean 探针 + 零空间投影：在 RM 隐藏态里做"外科手术式"去偏**

确定一个偏置可修之后，怎么修？对每类偏置构造正负对照样本集 $\{\mathbf{h}_i^+\}$、$\{\mathbf{h}_j^-\}$——比如长度偏置就用 GSM8K 的 verbose-correct 当正例、concise-correct 当负例——取最后一层最后一个非 padding token 在进入 reward head 之前的隐藏态，按 AxBench 上验证最强的 DiffMean 算出探针方向

$$\mathbf{p} = \mathrm{normalize}\Big(\tfrac{1}{n_+}\sum_i \mathbf{h}_i^+ - \tfrac{1}{n_-}\sum_j \mathbf{h}_j^-\Big).$$

推断时把每个新输入的隐藏态向探针的正交补投影 $\mathbf{h}_{\text{null}} = \mathbf{h} - \sum_k \alpha (\mathbf{p}_k^{\top}\mathbf{h})\mathbf{p}_k$，再喂给 reward head 得到去偏奖励，$\alpha$ 控制投影强度（除校准外都取 $\alpha=1$）；要同时去多个偏置时先 Gram-Schmidt 把多个探针正交化再联合 null。相比 length penalty、ensemble、bounded transformation 这些全局后处理，它不需要假设偏置的函数形式，整套干预完全发生在 RM latent space 内，既不重训 RM 也不动 policy 优化器，因此能即插即用到 RLHF / best-of-N / red-teaming / data filtering。数据效率也高得惊人——仅用 GSM8K 一个数据集做出的长度探针就能 OOD 迁移到 RewardBench2 和 AlpacaEval BoN。

**3. 五类偏置的对照数据构造范式：把"挖偏置"做成可复制流水线**

为了让后人能直接套用同一套方法审计新发布的 RM，作者把每类偏置的"诊断 + 探针构造 + 干预评估"固化成统一范式。长度偏置在 GSM8K 上为每题构造 (concise-correct, incorrect, verbose-correct) 三元组（verbose 平均 477.3 词 vs concise 171.1 词），看 RM 是否因冗长而偏好不正确答案；不确定性偏置在答案前缀加"我不太确定…"，要求 RM 满足规范排序 $r(C) \geq r(C+U) \geq r(I+U) \geq r(I)$；校准偏置在答案后追加 `confidence: {low, medium, high}`，看 Spearman(置信度, 正确性) 是否提升；位置偏置在 MCQA 里轮换正确答案位置 A–D，在 free-form 里比正确答案放首/尾的偏好差；模型风格偏置则用 10 个 LM（Gemma/Llama/Qwen 三族）算 per-byte cross-entropy，再算每个 RM 的奖励与 panel-relative $\Delta s_m$ 的 Spearman 相关——非零就说明 RM 系统性偏好某个 model-family 的"熟悉风格"。所有评估都先过滤掉"RM 在无干扰时本来就答错"的样本，确保看到的干预效果不被基线能力波动掩盖。

### 损失函数 / 训练策略
本方法**完全无需训练**。所有干预都是 inference-time 的线性投影，唯一"参数"是探针构造样本量（length 用 GSM8K 一个数据集，uncertainty/position 跨多数据集构造）和投影强度 $\alpha \in \{0.5, 1.0, 1.5\}$。校准实验显示 $\alpha$ 可调节"想去多少"——已经较少偏置的 Llama8B 系列 RM 用 $\alpha=0.5$ 反而比 $\alpha=1.0$ 更好。

## 实验关键数据

### 主实验

| 偏置类型 | 基线表现 | 干预后 | 是否显著缓解 |
|--------|------|------|----------|
| 长度（DeBERTa 偏好冗长） | 经典 length bias，Spearman(reward, length) = 0.611 | 0.067（95% CI 不重叠） | 是 |
| 长度（SOTA RM 过度修正） | 偏好简洁错答 > 冗长正答 | 缩小到不再倾向错答，不掉准确率 | 是 |
| 不确定性 | 含 "I'm not sure" 的正确答案 → RM 准确率平均掉 22.6% | 错答时偏好不确定表达，正答时仍偏好直接 | 是 |
| 校准（Skywork-Qwen-8B） | Spearman(confidence, correctness) = 0.182 | $\alpha=1.0$ 时 0.386（翻倍），成为最强校准 RM | 是 |
| 位置（MCQA A-D） | 跨位置偏差 2-28% | 三个模型显著降低位置方差 | 部分 |
| RewardBench2 综合准确率 | 70.1% | Length / Position / Uncertainty / Combined 干预后 69.3% / 69.3% / 70.1% / 69.3% | 全部通过 5pp 非劣性检验 (p < 0.001) |

### 消融实验

| 实验设置 | 关键发现 |
|------|------|
| Best-of-N on AlpacaEval (5 RM, 512 prompts × 64 cand) | 平均 within-prompt $\|r,L\|$ correlation 从 0.10 → 0.04；4/5 RM 的 length-controlled win rate 提升，优于 Huang et al. (2025) 全局校准 |
| Best-of-N on GSM8K (5 RM, 64 generations) | within-prompt 相关从 0.076 → 0.007；mean BoN accuracy 62.1% → 62.8% |
| 谄媚（regressive sycophancy） | 全部 5 个 RM 显著表现谄媚；最强的 Skywork-Qwen-8B 仍有 23.7% 跟错答，DeBERTa 高达 63.3%；**线性干预无法在不损害 progressive sycophancy 的前提下降低 regressive sycophancy**，证明这是高复杂度偏置 |
| 模型风格敏感性 | 全部 5 个 RM 都出现统计显著的奖励-风格相关（panel-relative cross-entropy）；平均绝对相关 ≈0.1，对单个 LM 可达 ±0.2~0.4，对应 4-16% 排名方差被风格解释；同样线性不可修 |
| 校准 $\alpha$ 扫描 | Skywork-Qwen 系列在 $\alpha=1.0$ 达峰；Llama8B 系列在 $\alpha=0.5$ 更好——说明已较少偏置的 RM 需要更轻干预，避免削掉真信号 |

### 关键发现
- 最关键的"啊哈"：SOTA RM 不是没有长度偏置，而是**反过来**惩罚冗长——为了消长度偏置训练的精修数据反而引入了相反方向的偏置，最终让 RM 在 GSM8K 上偏好"简洁但错"的答案。这说明 RLHF 数据治理常常按下葫芦浮起瓢。
- 探针的 OOD 迁移惊人：仅用 GSM8K 数学题构造的 length 探针，在 RewardBench2 + AlpacaEval BoN 都能让 reward-length 相关趋零且不掉 ranking 准确率，证明它抓的是真·length 方向而非数据集 artifact。
- "线性能修 vs 不能修"是非常 actionable 的分类：sycophancy 和 model-style 即便用 INLP 反复投影也只是把 bias 重定向不能根除，这给后人指明了需要去激活空间之外（如 sparse autoencoder、行为干预）寻找更深方案。
- 模型风格偏置的实际影响：ChatbotArena、LMSYS-Chat-1M、Tulu 2.5 Preference Data 都被 Llama/Qwen/Gemma 三族主导，意味着主流 RM 可能在系统性奖励"熟悉的写作风格"而非真实质量——对 RM-policy 配对策略和数据筛选有直接含义。

## 亮点与洞察
- **把"修不动"做成主结果**：作者没有遮掩 sycophancy 和 model-style 线性干预失败，反而把"高复杂度"作为正式 contribution，给出实证 + 表示层证据（INLP）。这种"诚实标注哪些方法解决不了什么"在 alignment 文献里非常珍贵，比 over-claim 实用得多。
- **Inference-time 干预的工程友好性**：不改 RM 权重、不改 policy 算法、新偏置只需一对对照样本就能加新探针，特别适合企业里"RM 已上线但发现新偏置"的紧急修补场景；可与 RLHF / best-of-N / red-teaming / data filtering 任何下游用法叠加。
- **可迁移 trick**：DiffMean + null-space projection 这套范式可以原样搬到其他单一 head 的标量输出模型上（判别器、毒性分类器、有用性 classifier 等），凡是怀疑"模型把某个 spurious 方向写进了最后一层 hidden state"就能用一对对照样本做诊断 + 一行投影做修复。
- **模型风格偏置的揭示**：从 panel-relative cross-entropy 角度量化"RM 偏好哪家 LM 风格"是一个新的可重复诊断范式，对开源 RM 生态的偏见审计意义重大。

## 局限与展望
- 作者承认的局限：(1) 高复杂度偏置（sycophancy / model-style）线性干预无效，需要更根本方案；(2) 长度探针在 SOTA RM 上会引入轻微正相关（如 Skywork-Llama-8B $r_s$ 从 0.267 → 0.305），说明 null 的强度需要按模型调；(3) 评估仅限四个 reasoning/knowledge benchmark，未覆盖 chat / safety 等长尾域。
- 自己看到的局限：(a) 探针构造对"如何选对照样本"敏感，长度偏置只用 GSM8K 一个数据集做探针虽 OOD 有效，但若对照样本本身有 confound（如 verbose 答案带了更多 reasoning chain），可能误把"reasoning 痕迹方向"也 null 掉；(b) 评估的所有 RM 都是 single-output reward head 架构，是否能迁移到 generative reward model 或 critic 还需验证；(c) 五个 RM 中三个共享 Skywork-Reward-V2 训练 pipeline，"持续偏置"的结论部分可能由训练管线共性而非 RM 普遍规律导致；(d) 校准 $\alpha$ 需按模型手调，未给出自动选择准则。
- 改进思路：把 DiffMean 升级为 sparse autoencoder 特征 + 多方向 null，或与 Casademunt et al. (2025) 的 SAE-based finetuning steering 结合；针对 sycophancy 尝试在更深层而非最后一层做干预；把探针构造数据扩展到 chat / safety 域以提升 RewardBench2 的全域泛化。

## 相关工作与启发
- **vs Park et al. (2024b) / Huang et al. (2025) 等 length penalty**：他们对 length-reward 关系做显式参数化建模，要求知道偏置函数形式；本文用线性 null 不假设函数形式，且作者在 BoN 实验中实证 length-controlled win rate 优于 Huang et al. (2025) 的全局校准。
- **vs Ravfogel et al. (2020) INLP（Iterative Nullspace Projection）**：INLP 反复投影到完全无法线性预测某属性为止，本文用单步 DiffMean 投影更轻量；本文把 INLP 当作高复杂度偏置的诊断工具（附录 C.9），是对其使用方法的扩展。
- **vs Casademunt et al. (2025) SAE-based steering**：他们用 sparse autoencoder 提取方向再 null，本文用更朴素的 DiffMean，预示未来可结合两者获得更精细的偏置分解。
- **vs Sharma et al. (2024) / Hong et al. (2025) sycophancy 评估**：前作没区分 "RM 本就答错" vs "RM 答对但被用户带歪"，本文先过滤前者再评估，给出更干净的谄媚度量；同时本文给出 sycophancy 线性不可修的实证，为后续研究划清边界。
- **vs Malik et al. (2026) inverse-rank model-lineage analysis**：他们发现 RM 略偏好同 base 模型族的生成，本文给出更细的 panel-relative cross-entropy 量化，且把这种偏置归入高复杂度类别。

## 评分
- 新颖性: ⭐⭐⭐⭐ —— "偏置复杂度二分类 + mechanistic reward shaping"是清晰的新框架，DiffMean + null 单独看不算新但组合到 RM 去偏并系统验证 OOD 是首次。
- 实验充分度: ⭐⭐⭐⭐⭐ —— 5 个 RM × 5 类偏置 × 4 benchmark + RewardBench2 + AlpacaEval BoN + GSM8K BoN + LLM-as-judge OOD 验证 + INLP 表示层证据，可重复性极强。
- 写作质量: ⭐⭐⭐⭐ —— 诚实标注线性干预失败的偏置；taxonomy 论述清晰；表格组织略密集，部分实验细节藏在附录略影响主线阅读。
- 价值: ⭐⭐⭐⭐⭐ —— 对当前 RLHF / best-of-N / RM-based pipeline 直接可用；对 alignment community 提供"哪些偏置能修、哪些需要新方法"的实证 roadmap；模型风格偏置的揭示对开源数据生态有警示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CAMEL: Confidence-Gated Reflection for Reward Modeling](camel_confidence-gated_reflection_for_reward_modeling.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2025\] Automatic Reward Shaping from Confounded Offline Data](../../ICML2025/reinforcement_learning/automatic_reward_shaping_from_confounded_offline_data.md)
- [\[ICML 2025\] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](../../ICML2025/reinforcement_learning/action-dependent_optimality-preserving_reward_shaping.md)

</div>

<!-- RELATED:END -->
