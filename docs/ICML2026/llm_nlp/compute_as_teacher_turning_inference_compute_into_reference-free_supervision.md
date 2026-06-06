---
title: >-
  [论文解读] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision
description: >-
  [ICML 2026][LLM/NLP][GRPO] 本文提出 Compute as Teacher（CaT）：把 GRPO 已经在采样的 G 条 rollouts 通过冻结锚模型"合成"出一个伪参考答案…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "GRPO"
  - "自合成参考"
  - "自提议 rubric"
  - "非可验证奖励"
  - "HealthBench"
---

# Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision

**会议**: ICML 2026  
**arXiv**: [2509.14234](https://arxiv.org/abs/2509.14234)  
**代码**: 无（论文中未给出公开仓库）  
**领域**: LLM / NLP；RLHF 替代方案；无参考监督 RL  
**关键词**: GRPO、自合成参考、自提议 rubric、非可验证奖励、HealthBench

## 一句话总结
本文提出 Compute as Teacher（CaT）：把 GRPO 已经在采样的 G 条 rollouts 通过冻结锚模型"合成"出一个伪参考答案，再在非可验证领域用模型自己从该伪参考衍生的二元 rubric 给每条 rollout 打分作为 RL 奖励，从而在没有任何人工标注的情况下把推理算力直接变成监督信号，在 HealthBench 上相对基线最高提升 30%，并以 9× 更低的测试时算力匹配甚至超过 inference-time aggregation。

## 研究背景与动机

**领域现状**：当前大模型后训练主要靠两条路——有人类标注参考答案的 SFT（Ouyang et al. 2022），或是有程序化 verifier 的 RLVR（如 math/code 的 GRPO，Shao et al. 2024）。两条路都要求"参考答案存在且可获取"。

**现有痛点**：医学咨询、生活建议、开放对话、创意写作等任务里，答案天然是开放的、多解的、专家意见分歧的，根本写不出 ground truth，更没法写程序化 checker。常见 fallback 要么是花大价钱建标注流水线，要么直接让另一个 LLM 给 1–10 打分（LLM-as-judge），后者已被反复证明存在不一致、偏长、style bias、reward hacking 等问题（Zheng et al. 2023）。

**核心矛盾**：RL 训练需要一个"参考信号"才能算 advantage；但在非可验证领域里这个参考信号既不来自人类也无法由程序产生。这导致后训练在最有价值的开放领域反而最贫瘠。

**本文目标**：(i) 在没有任何人工参考的条件下，给非可验证任务的 RL 提供一个稳定、可用的奖励信号；(ii) 让这个奖励机制和现有 RLVR pipeline（GRPO）即插即用，不引入显著额外算力。

**切入角度**：GRPO 已经为每个 prompt 平行采样 G 条 rollouts 来估 advantage，这些 rollouts 恰好"在模型不确定的地方相互分歧"——一条可能算对了中间步骤、另一条对了最终答案、第三条做了正确的校验。**整组 rollouts 的信息量本质上大于任意单条**，而现在这部分信息只被用作方差归一化，被严重浪费了。

**核心 idea**：用"合成（synthesis）"把多条 rollouts 调和成一个伪参考答案 $s$，再让模型自己从 $s$ 提取若干个二元 rubric criteria 作为奖励——把"算力换监督"做成可即插即用的两阶段管线，分别对应 **reference estimation** 和 **reward derivation**。

## 方法详解

### 整体框架
输入：prompt $q$，当前策略 $\pi_t$，冻结锚 $\pi_0$（一般取初始策略），裁判 $\pi_J$（如 GPT-4o）。流程：

1. 用 $\pi_t$ 采样 $G$ 条 rollouts $o_{1:G}$（与 GRPO 复用）；
2. **Reference estimation**：用 $\pi_0$ 对 rollouts 做 synthesis，得到伪参考 $s \sim \pi_0(\cdot \mid p_{\text{syn}}, o_{1:G})$；
3. **Reward derivation**：可验证域直接对答案串匹配；非可验证域由 $\pi_0$ 从 $s$ 生成 $n\ge 5$ 条二元 rubric $\mathcal{R}=\{r_1,\dots,r_n\}$，再让 $\pi_J$ 对每条 rollout 逐条判 yes/no，奖励为通过比例 $R_{\text{rub}}(o;\mathcal{R}) = \frac{1}{n}\sum_j \mathbf{1}[\pi_J(o,r_j)=\text{yes}]$；
4. GRPO 用 $\hat A_i = (R_i - \bar R_G)/\sigma_G$ 更新 $\pi_t$。

亮点是整套机制**和 GRPO 原生采样完全对齐**：只多了 1 次 synthesis + 1 次 rubric 生成 + $n\times G$ 次极短的 yes/no 判定，可并行化，整体开销远小于 G 条 rollouts 本身。

### 关键设计

1. **Synthesis 作为 reference estimator**:

    - 功能：把 $G$ 条分歧的 rollouts 调和成一个伪参考 $s$，而不是从中"选"一条。
    - 核心思路：让冻结的初始策略 $\pi_0$（不是当前 $\pi_t$）在一个固定 prompt $p_{\text{syn}}$ 下读完所有 rollouts，生成新的回答。**关键设计是输入里故意不放原 prompt $q$**（消融见 Appx 6.4），目的是迫使模型完全依赖 rollouts 内部信息做调和而不是直接重答；**用冻结锚而不是当前策略**则解耦了"探索"与"估计"——$\pi_t$ 通过 RL 不断进步，$\pi_0$ 始终提供稳定的参考估计基线，避免参考随策略漂移导致目标移动。实证上 synthesis 在 5–15% 的题上和多数票不一致，且不一致时仍正确率 70–86%（Table 1），甚至在 ~1% 的题上能"全队都错时唯独合成对"——这是任何 selection 方法（majority vote、Self-BoN、min-PPL）原理上做不到的。
    - 设计动机：selection 至多能恢复"最好的那条 rollout"；synthesis 能跨 rollouts 拼接正确片段，生成分布之外的更优答案，从而把推理算力的潜力榨干。

2. **Self-proposed Rubrics（核心贡献）**:

    - 功能：在没有任何人工参考的非可验证领域，把"这个回答好不好"这种粗判定，拆成"该回答是否满足条件 $r_j$"这样的若干个二元细粒度判定。
    - 核心思路：$\mathcal{R} \sim \pi_0(\cdot \mid p_{\text{rub}}, s)$，由锚模型从伪参考 $s$ 中提炼出 $\ge 5$ 条二元、可审计、可重复判断的 criteria（如"建议咨询医生""提到了 lifestyle modification""回避了给确诊"），然后裁判模型 $\pi_J$ 对每条 rollout 独立判 yes/no，奖励 = 满足比例。整个 pipeline **从 inference compute → pseudo-reference → rubrics → reward**，全程无任何人类参考介入。
    - 设计动机：分解判定带来三大好处——(i) 可靠性：每条二元问题对 LLM 来说远比"打 1–10 分"稳定；(ii) 可审计：可以查到具体哪条 criterion 失败，方便 debug；(iii) 降 style bias：rubric 奖励的是"内容是否覆盖"而不是行文风格/长度，缓解 verbosity bias 和 reward hacking。

3. **Drop-in 兼容可验证域 + 算力摊销**:

    - 功能：同一框架在 math/code 等可验证域里退化为"对伪参考的答案匹配"，无需任何代码改动。
    - 核心思路：可验证域 reward 简化为 $R_{\text{ver}}(o;s)=\mathbf{1}[\texttt{answer}(o)=\texttt{answer}(s)]$，依然由 synthesis 提供 $s$；这一步等价于 TTRL 的 majority-vote pseudo-labeling，但 synthesis 可以走出 rollout 集合的支撑。算力摊销层面：训练完之后单次 forward 就能产生与 inference-time aggregation 同等甚至更好的回答，等于把"每次部署都付 G 倍算力"的成本一次性烧进了模型权重里。
    - 设计动机：非可验证才是真正难解的问题，但作者证明只要更换 reward derivation 一行，框架就能"插"进任何域，验证 CaT 不是 healthcare-specific trick，而是真正的统一范式。

### 损失函数 / 训练策略
- 基础：GRPO 的 clipped surrogate + KL 到 $\pi_0$ 正则；
- Group size $G=8$；
- 锚 $\pi_0$ 与初始策略相同，裁判 $\pi_J=$ GPT-4o；
- 算力开销：synthesis 约等于多 1 条 rollout；rubric 评分需 $n\times G$ 次极短的 yes/no 判定，可完全并行。

## 实验关键数据

### 主实验

| 模型 | 数据集 | Initial | CaT | Inference-time Synthesis | 相对提升 / 算力比 |
|------|--------|---------|-----|--------------------------|------------------|
| Gemma 3 4B | HealthBench | base | +up to 30% | < CaT | CaT 用 1× 测试算力 vs synth 9× |
| Qwen 3 4B | HealthBench | base | 显著超过 base | ≈ CaT | 9× 测试算力降至 1× |
| Llama 3.1 8B | HealthBench | base | 0.38 (vs SFT 0.28) | < CaT | 同上 |
| 三个模型 | MATH-500 | base | 最高 +33% | ≈ CaT | drop-in 即可与可验证基线持平 |

### 消融实验

| 配置 | HealthBench 关键现象 | 说明 |
|------|--------------------|------|
| CaT（self-proposed rubric） | 与 physician rubric 持平 | 两个模型上"自己写的标准 ≈ 医生写的标准" |
| Model-as-judge（1–10 打分） | 全模型显著低于 CaT | 粗粒度判定不稳定，reward 噪声大 |
| CaT-SFT（用伪参考做 SFT） | Llama 0.28 vs CaT 0.38 | RL 比 SFT 在小数据下泛化更好 |
| Synthesis vs Majority/Self-BoN/Min-PPL | HealthBench 全胜，MATH-500 持平 | 非可验证域 synthesis 优势最大 |
| Synthesis 输入 8 条 vs 1 条 | 0.85 vs 0.80（Qwen MATH） | 证明 synthesis 在做跨 rollout 推理而非"多采一条" |

### 关键发现
- **自提议 rubric 能逼平专家标注**：HealthBench 上两个模型中 self-proposed rubric 与人类医生设计的 rubric 几乎打平，证明"能写出像样答案"的模型同时具备"提炼有效评分维度"的能力。
- **Synthesis 是真正在做调和**：在 ~1% 题上"全队 rollouts 都错而 synthesis 对"，且与多数票不一致时正确率高达 82–86%，说明 synthesis 能产生 rollouts 分布外的更优答案。
- **算力一次烧进权重**：CaT 训练后单次前向就能匹配甚至超过 9× G-rollout 的 inference-time synthesis，把"每次部署都付 9 倍算力"的代价彻底摊销为"训练一次"。
- **Llama 在 synthesis 上收益小，但在 RL 上收益最大**：弱模型不擅长 meta-cognitive 调和，但 RL 能补；这暗示 CaT 对中弱模型更友好。
- **Entropy collapse 后训练饱和**：rollouts 收敛后 synthesis 调和空间消失，再训练边际收益变小，与 RL fine-tuning 常见的 entropy collapse 现象一致。

## 亮点与洞察
- **"算力即监督"的范式漂移**：以往无标注 RL（TTRL、Absolute Zero）只敢用 majority vote 这种选择性 aggregator，且只在可验证域可用；本文第一次把"生成式 aggregator + 二元 rubric"组合成在非可验证域可用的统一管线，本质上是把推理时的 best-of-N 收益翻译成了训练时的监督信号。
- **解耦 anchor 与 policy 是关键工程细节**：用冻结 $\pi_0$ 而不是 $\pi_t$ 做 synthesis，避免了"被自己骗自己"的正反馈漂移，让奖励信号锚定在一个稳定的参考分布上——这一点和 RLHF 里 KL-to-reference 思想一脉相承，但用法不同（这里是用 anchor 估目标而不是约束更新幅度）。
- **Rubric 是 reward 的"白盒接口"**：rubric 让 reward 可读、可审、可调试，这对工业部署里"为什么这条被罚分"的问题至关重要——相当于把奖励工程从黑箱 LLM judge 升级成结构化条件清单，可以人工 spot-check / curate。
- **可迁移设计**：synthesis-as-aggregator + rubric-as-reward 这一对范式可直接迁到 reasoning trace 评分、multi-turn dialog、agentic trajectory 等场景；任何"需要打分但说不清打分标准"的场景，都可以让模型先自己生成标准。

## 局限与展望
- **依赖基础模型能力**：弱模型生成的 rollouts 信息量不够、调和能力差，CaT 收益相应缩水；本质上 CaT 是"用算力放大基础能力"，对完全没掌握领域的模型无效。
- **Entropy collapse 后训练饱和**：rollouts 收敛后 synthesis 失去调和空间，训练进入瓶颈；作者建议未来引入 exploration reward 或更多样的采样策略来缓解。
- **裁判模型的能力依赖**：用 GPT-4o 做 $\pi_J$，对小团队来说成本与可复现性都是问题；用开源裁判替换后效果如何未在主文系统化研究（Appx 6.3 略提）。
- **Rubric 粒度仍粗**：当前 rubric 是二元 yes/no，未引入 partial credit / 层级化标准 / 置信度加权，未来可以引入更细粒度的 rubric 来提升 reward 信号分辨率。
- **自验证：模型自己生成的 rubric 是否在保护自己？** 文中没系统讨论 rubric 是否存在 "self-collusion"（即 rubric 偏向 anchor 自身的回答模式），这是值得后续深究的潜在 reward hacking 路径。

## 相关工作与启发
- **vs TTRL / Absolute Zero**：他们也做 reference-free RL，但只在可验证域用 majority vote / 自博弈；CaT 用 synthesis 走出 rollout 支撑，且通过 rubric 把非可验证域纳入 RL 框架。
- **vs Rubrics as Rewards (RaR, Gunjal et al. 2026)**：RaR 也用 rubric 打分，但 rubric 是从**人工参考**构造的；CaT 把 rubric 也变成自生成，彻底去掉人类标注依赖。
- **vs Test-time scaling (majority vote, best-of-N)**：那条路是"每次部署都付 G 倍算力"；CaT 把同样的算力一次性摊销到训练，部署时回到 1 倍。
- **vs LLM-as-judge (Zheng et al. 2023)**：1–10 打分粗粒度且偏 style；CaT 用二元 rubric + 可审计 criteria，降噪+降偏。
- **vs Constitutional AI / Self-Instruct**：那些方法针对特定能力（无害化、指令遵循），CaT 提供一个更普适、领域无关的 reference-free RL 框架。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"算力即监督 + 自生成 rubric"打通成一套即插即用的统一管线，是 reference-free RL 在非可验证域上的关键一步。
- 实验充分度: ⭐⭐⭐⭐ 覆盖三个模型家族 + 两个域，含与人类专家 rubric 的对比和 selection baseline 全套消融；但仅在 4–8B 规模、单领域（healthcare）做了非可验证验证。
- 写作质量: ⭐⭐⭐⭐⭐ 行文清晰，"why it works" intuition 段精炼，算法块、图表与结论紧密呼应。
- 价值: ⭐⭐⭐⭐⭐ 提供了一个可立刻嵌入工业 RLHF pipeline 的范式（GRPO 兼容、不需要 verifier、不需要人类标注），对开放领域后训练价值极高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](../../NeurIPS2025/llm_nlp/wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)
- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](../../ACL2025/llm_nlp/skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](../../NeurIPS2025/llm_nlp/dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)

</div>

<!-- RELATED:END -->
