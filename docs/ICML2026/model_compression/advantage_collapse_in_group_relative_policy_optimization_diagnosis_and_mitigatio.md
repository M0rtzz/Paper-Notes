---
title: >-
  [论文解读] Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation
description: >-
  [ICML 2026][模型压缩][GRPO] 这篇论文指出 GRPO 在二值可验证奖励下会因为组内奖励全同而失去梯度信号，提出 ACR 指标实时诊断这种“优势坍塌”，并用 AVSPO 注入虚拟奖励样本恢复组内方差，从而在多个 Qwen2.5 数学推理模型上稳定提升 4-6 个百分点。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "GRPO"
  - "RLVR"
  - "优势坍塌"
  - "训练诊断"
  - "虚拟样本"
---

# Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation

**会议**: ICML 2026  
**arXiv**: [2605.21125](https://arxiv.org/abs/2605.21125)  
**代码**: https://github.com/hexixiang/Advantage-Collapse-Rate  
**领域**: 强化学习 / LLM 推理  
**关键词**: GRPO, RLVR, 优势坍塌, 训练诊断, 虚拟样本

## 一句话总结
这篇论文指出 GRPO 在二值可验证奖励下会因为组内奖励全同而失去梯度信号，提出 ACR 指标实时诊断这种“优势坍塌”，并用 AVSPO 注入虚拟奖励样本恢复组内方差，从而在多个 Qwen2.5 数学推理模型上稳定提升 4-6 个百分点。

## 研究背景与动机
**领域现状**：LLM 数学推理的后训练越来越依赖 RLVR，也就是用自动验证器给最终答案打二值奖励，再用强化学习优化模型。GRPO 是这一范式里的代表算法，它不训练 critic，而是在同一个问题的多条采样回复之间做相对比较来估计 advantage，因此比 PPO 这类 actor-critic 方法更省显存，也更容易扩展到长推理任务。

**现有痛点**：GRPO 的优势估计依赖组内奖励的均值和标准差。当一个问题采样出的 $G$ 条回复全错或全对时，组内奖励方差为 0，所有样本的 advantage 都会变成 0。问题在于，这种 batch 已经消耗了昂贵的 LLM rollout，却不会对策略更新提供有效梯度；训练日志里的 loss、平均 reward 甚至 accuracy 也未必能及时暴露这个浪费。

**核心矛盾**：二值 verifiable reward 简洁可靠，但它也最容易产生“全 0 / 全 1”的同质奖励。GRPO 为了省掉 critic，正好把学习信号押在组内相对差异上，于是奖励越稀疏、题目越容易或越困难，越可能出现计算已经做完但梯度为零的情况。

**本文目标**：作者要解决两个子问题。第一，训练时如何量化有多少组已经进入无效梯度状态；第二，发现这些组之后，能否不重新采样、不额外调用模型，就让这些本来被浪费的样本重新产生学习信号。

**切入角度**：论文没有先改奖励模型或采样策略，而是回到 GRPO 的 advantage 公式本身。只要监控组内 reward 标准差，就能知道该组是否会产生有效梯度；如果用很便宜的虚拟 reward 改变归一化统计量，也许就能恢复非零 advantage。

**核心 idea**：用 ACR 直接测量 GRPO batch 中“奖励方差近零”的比例，并在坍塌组里只向归一化统计量注入虚拟奖励，从而把无效 rollout 转化为可更新样本。

## 方法详解

### 整体框架
论文要解决的是 GRPO 在二值奖励下“算了一整 batch 却没有梯度”的浪费，做法是给 GRPO 加一个诊断器再加一个轻量干预器。诊断器 ACR 回答“当前 batch 里有多少计算白做了”，干预器 AVSPO 则在这些白做的组里补一个可控的归一化参照，让真实样本重新拿到有方向的 advantage。关键在于虚拟样本只是几个数值，不是新的文本输出，不参与策略梯度，只改变 reward 的 mean/std，因此不引入任何额外的 LLM 前向开销。

具体地，训练仍沿用 GRPO 主干：对每个问题 $q$ 旧策略采样 $G$ 条回复，验证器给出二值奖励 $r_i \in \{0,1\}$，普通 GRPO 用 $\hat{A}_i=(r_i-\mu_R)/(\sigma_R+\epsilon)$ 算组内 advantage 后进入 clipped objective。AVSPO 在中间插三步：先对每组算 reward 标准差，$\sigma_R<\tau$ 就判定该组发生 advantage collapse；再在 batch 级统计坍塌组比例得到 ACR，并用动态阈值决定是否介入；最后对触发介入的坍塌组构造 $K$ 个虚拟 reward，把真实和虚拟 reward 合并重算 $\mu_{R'}$、$\sigma_{R'}$，但只给真实样本计算新的 $\hat{A}'_i$ 去更新模型。

### 关键设计

**1. Advantage Collapse Rate：把“训练白干”变成可实时观测的数字。**

GRPO 的痛点是当一组 $G$ 条回复全对或全错时组内 reward 方差为 0，advantage 全归零，昂贵的 rollout 不产生任何梯度，而 loss、平均 reward 乃至 accuracy 都未必及时暴露这种浪费。ACR 的做法是对 batch 内 $N$ 个问题组逐一检查 $\sigma_{R_j}<\tau$，再求坍塌组占比 $ACR=\frac{1}{N}\sum_j \mathbb{I}(\sigma_{R_j}<\tau)$：接近 0 说明绝大多数组都有奖励差异，接近 1 则几乎所有 rollout 都卡在零梯度。它之所以好用，是因为完全复用了 GRPO 已经算出的 reward 统计量，不需要 critic、额外标注或额外推理，就把“训练停滞”从事后才看得到的 accuracy 现象，变成了训练中可监控的实时信号。

**2. 自适应虚拟样本 AVSPO：不重采样，只给坍塌组补一个统计参照。**

诊断出坍塌只是第一步，真正要做的是让这些被浪费的组重新产生学习信号，但又不能重新调用模型采样。AVSPO 的做法是：当 batch 的 ACR 超过动态阈值且某组坍塌时，构造 $K=\max(1,\min(G,\lceil G\cdot ACR^\alpha\rceil))$ 个虚拟 reward——真实组全对就让虚拟 reward 从接近 1 的值分层向下递减，全错就用小的正 anchor reward 拼出一组非零虚拟值，然后用合并集 $R'=R\cup V$ 重算均值和标准差，而 policy gradient 仍只作用在原来 $G$ 条真实回复上。这样做有效，是因为全错组和全对组都不是无信息组：全错意味着当前策略要远离这些失败轨迹，全对意味着成功轨迹值得继续强化。虚拟 reward 不伪造新答案，只是提供一个统计参照，让这些同质组里本该存在的方向信息不被归一化公式抹平。

**3. 动态触发与有界偏差控制：让干预按需发生且不会乱带方向。**

虚拟 reward 既然改了 advantage 的归一化，就必须控制“何时介入、介入多强”，否则阈值定太低会过度干预、放大方差，定太高又会错过早期坍塌。AVSPO 把触发阈值 $\tau_{adapt}$ 初始设为 0.5，并按训练是否还在改进来动态调整，使它更像一个“当前没进步才加力”的按需修复器，而不是常驻的奖励整形项；同时虚拟样本数量随 $ACR^\alpha$（默认 $\alpha=0.5$）缩放，坍塌越普遍才补越多。理论上在 $K\leq G$ 时，同质组里 AVSPO 产生的 uniform advantage 幅度满足 $|A^c(K)|\leq\sqrt{K/G}\leq 1$，偏差上界也随 ACR 收缩，这正回应了“虚拟 reward 会不会把策略带偏”的担忧——介入幅度天然有界，且越接近坍塌临界处补得越克制。

### 损失函数 / 训练策略
AVSPO 的训练目标仍是 GRPO 的 clipped surrogate，只是把 advantage 从 $\hat{A}_i$ 换成用增强 reward 集算出的 $\hat{A}'_i$；虚拟 reward 只进入 $\mu_{R'}$、$\sigma_{R'}$，不产生 $\nabla_\theta \log \pi_\theta$ 项。实验中 group size 为 8，训练温度 1.0，评估用 greedy decoding；AVSPO 特有超参为初始阈值 0.5、$\alpha=0.5$、阈值学习率 0.01、collapse 阈值 $10^{-6}$、anchor reward 0.1。

## 实验关键数据

### 主实验
论文在 6 个 Qwen2.5 系列模型上训练 500 steps，训练集是从 MATH training split 抽取的 Level3-500，评估覆盖 MATH-500、GSM8K、Minerva、OlympiadBench、AMC、AIME24 和 MMLU-Pro。核心结论是 AVSPO 同时降低 ACR 和提升平均准确率，且 baseline ACR 越高的模型受益越明显。

| 模型 | GRPO ACR | AVSPO ACR | GRPO 平均准确率 | AVSPO 平均准确率 | 提升 |
|------|----------|-----------|----------------|-----------------|------|
| Qwen2.5-0.5B | 0.45 | 0.18 | 16.5 | 21.0 | +4.5 |
| Qwen2.5-3B | 0.37 | 0.14 | 27.9 | 32.2 | +4.3 |
| Qwen2.5-3B-Instruct | 0.35 | 0.13 | 39.7 | 43.4 | +3.7 |
| Qwen2.5-14B | 0.28 | 0.11 | 49.9 | 54.5 | +4.6 |
| Qwen2.5-Math-1.5B | 0.40 | 0.15 | 33.5 | 39.6 | +6.1 |
| Qwen2.5-Math-7B | 0.33 | 0.14 | 42.2 | 45.9 | +3.7 |

和其他 baseline 相比，AVSPO 的平均表现也更强。作者报告它相对 DCPO 约 +2.9，优于 INTUITOR 和 RENT 的幅度更大。这个对比说明，单纯改 clipping、鼓励低熵或用自信度做 reward，都不如直接修复 batch-level reward diversity 来得稳定。

### 消融实验
虚拟样本构造方式是最关键的消融。随机采样、固定 partial credit、指数衰减都能降低 ACR，但分层 reward assignment 的效果最好，说明“只要有方差”还不够，虚拟 reward 的结构也会影响 advantage 的方向和稳定性。

| 配置 | ACR | MATH-500 | 说明 |
|------|-----|----------|------|
| GRPO 无增强 | 0.40 | 58.6 | 所有坍塌组 advantage 为 0 |
| 随机均匀虚拟 reward | 0.22 | 62.1 | 有效降低坍塌，但引入随机方差 |
| 固定 partial credit | 0.19 | 63.5 | 简单稳定，但 reward 层次不足 |
| 指数衰减 | 0.18 | 64.2 | 比固定值更细，但仍不如分层策略 |
| AVSPO 分层策略 | 0.15 | 67.2 | ACR 最低，准确率最高 |

机制隔离实验也很有说明力。只修复 all-wrong 组能把 all-wrong collapse 从 24.8% 降到 9.1%，准确率到 63.2；只修复 all-correct 组能把 all-correct collapse 从 15.2% 降到 4.2%，准确率到 60.8；完整 AVSPO 同时压到 8.7% 和 6.3%，MATH-500 达到 67.2。固定阈值对比中，最佳固定阈值需要 380 steps 才到 60% accuracy，而自适应阈值只需 295 steps，并取得 67.2 的最终准确率。

### 关键发现
- ACR 是强诊断信号：前 100 step 的 ACR 与最终 MATH-500 accuracy 的相关系数为 $r=-0.785$，线性拟合 $R^2=0.617$，也就是早期 ACR 能解释约 62% 的最终性能方差。
- 坍塌不是少数异常：普通 GRPO 在这些数学推理设置里有 28%-45% 的 batch 组发生完整 advantage collapse，足以成为训练效率瓶颈。
- 中等难度样本最适合 RLVR 训练：太容易导致全对，太难导致全错，都会提高 ACR；Level 3-4 难度让 reward diversity 更自然。
- 增强优于过滤：在 Qwen2.5-Math-7B 上，Filter-Drop 只利用 62.4% 样本，DAPO 成本约 1.8 倍；AVSPO 保持 100% 样本利用和 1.0 倍成本，同时 GSM8K/MATH 达到 69.7/74.1。

## 亮点与洞察
- 这篇论文最有价值的点是把 GRPO 的失败模式写成一个可测量的训练诊断，而不是只报告“RL 不稳定”。ACR 很简单，但它直接对应 advantage 公式里的零方差条件，因此解释力很强。
- AVSPO 的虚拟样本设计比较巧妙：它没有生成伪文本，也没有给模型引入新的 reward model，而是只改变 normalization 统计量。这让方法的工程开销接近为零，也避免了额外 rollout 成本。
- all-correct collapse 被纳入讨论很重要。很多人只会关注全错组没有学习信号，但全对组同样会让 GRPO 无法进一步强化成功轨迹；AVSPO 对这两种情况给出统一处理。
- ACR 对数据难度、温度、group size 的敏感性分析可以直接迁移到其他 RLVR pipeline。即使不采用 AVSPO，训练者也可以用 ACR 早停低效配置，或调整采样温度与题目难度。
- 论文的理论部分不是装饰性的。它说明虚拟样本在全错组上会降低失败集合概率，在全对组上会提高成功集合概率，并且 PPO clipping 会限制过度强化，这正好回应了“虚拟 reward 是否会乱带方向”的担忧。

## 局限与展望
- 实验主要集中在数学推理和二值 deterministic verifier。对于开放式偏好奖励、多级 reward 或 noisy verifier，ACR 的阈值和虚拟 reward 设计是否仍然合适，还需要重新验证。
- AVSPO 修复的是组内 reward 方差为零的问题，但它不能解决 verifier 本身错误、题目分布偏差或模型容量不足。作者也观察到在 AMC/AIME 这类竞赛级任务上收益更小，说明难题上瓶颈可能已经转向模型能力。
- 虚拟 reward 本质上改变了 advantage normalization，会引入有界偏差。虽然论文给出上界和收敛讨论，但在长程、多轮 tool-use agent 任务里，这种偏差是否会累积成策略偏好，还值得进一步做长期训练实验。
- 当前方法默认所有坍塌组都可以从统计层面修复。未来可以结合过程 reward、错误类型诊断或 curriculum 调度，只对“有学习价值”的坍塌组介入，避免在真正无信息或 verifier 不可靠的样本上制造信号。

## 相关工作与启发
- **vs GRPO**: GRPO 用组内相对 reward 省掉 critic，但在 reward 全同的时候完全没有梯度。AVSPO 保留 GRPO 的主体结构，只在坍塌组里重算 advantage 统计量，是对 GRPO failure mode 的局部修补。
- **vs PPO/GAE**: PPO 搭配 GAE 可以通过 value baseline 缓解部分方差问题，但需要 critic，显存和训练复杂度都更高。本文的目标是保持 GRPO 的无 critic 优势，因此选择了 reward-statistics 层面的干预。
- **vs PRM / dense reward**: 过程奖励模型能提供更细粒度监督，但要额外标注或训练 reward model。AVSPO 只使用最终答案 verifier，适合已经有 deterministic checker 的数学和代码任务。
- **vs DAPO / DCPO**: 这些方法从 clipping、动态采样或优化细节改善 GRPO。AVSPO 的切入点更低层，关注 batch 内 reward diversity，因此可以与系统级 GRPO recipe 互补。
- **启发**: 对所有基于 group comparison 的 RL 算法，都应监控“有效梯度比例”，而不只是看 reward 均值。类似 ACR 的指标也可以扩展到代码生成、工具调用、自动证明等二值验证任务中。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 诊断指标本身很简单，但把 advantage collapse 系统量化并用虚拟 reward 修复，是对 GRPO 训练机制的直接有效补充。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖多尺度模型、多个数学 benchmark、ACR 相关性、消融和成本对比，但开放式 RLHF 或非二值奖励场景还缺验证。
- 写作质量: ⭐⭐⭐⭐☆ 问题定义清楚，方法和实验围绕 collapse 展开，理论分析也能服务主论点；表格较多但主线不散。
- 价值: ⭐⭐⭐⭐⭐ 对正在做 RLVR/GRPO 训练的人很实用，ACR 即使单独作为诊断指标也值得接入训练日志。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](../../AAAI2026/model_compression/metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](../../ICML2025/model_compression/confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)

</div>

<!-- RELATED:END -->
