---
title: >-
  [论文解读] Position: AI Evaluations Should be Grounded on a Theory of Capability
description: >-
  [ICML 2026][图像生成][AI 评测] 作者主张"benchmark 分数 = 能力"是一种**隐式推断**而非直接测量，呼吁把 AI 评测显式建模成统计推断任务，并借鉴心理测量学（CTT/IRT/CDM/BNSM）四种能力理论作为模板，给出一张"Evaluation Card"让评测者自证假设。
tags:
  - "ICML 2026"
  - "图像生成"
  - "AI 评测"
  - "能力理论"
  - "心理测量学"
  - "IRT"
  - "扰动鲁棒性"
---

# Position: AI Evaluations Should be Grounded on a Theory of Capability

**会议**: ICML 2026  
**arXiv**: [2509.19590](https://arxiv.org/abs/2509.19590)  
**代码**: https://github.com/nathanaj99/ai_stat_test (有)  
**领域**: LLM 评测 / 心理测量学 / 立场论文  
**关键词**: AI 评测、能力理论、心理测量学、IRT、扰动鲁棒性

## 一句话总结
作者主张"benchmark 分数 = 能力"是一种**隐式推断**而非直接测量，呼吁把 AI 评测显式建模成统计推断任务，并借鉴心理测量学（CTT/IRT/CDM/BNSM）四种能力理论作为模板，给出一张"Evaluation Card"让评测者自证假设。

## 研究背景与动机

**领域现状**：当前 LLM 评测几乎清一色采用"在 benchmark 上跑、报平均准确率"的范式 —— MMLU、BBH、HELM 都默认 *score = capability*，排行榜（HuggingFace、Vellum）直接用聚合分数排名。

**现有痛点**：评测结果脆弱、不可复现已成共识 —— 同一模型在不同 prompt 措辞、不同温度、不同 system message 下分数可以差出十几个点，且不同 benchmark 之间的"capability"含义互不兼容。更糟的是，IRT-based 新方法（adaptive testing、tinyBenchmarks 等）已经在悄悄改写"能力"的定义，但没人显式声明。

**核心矛盾**：分数到能力之间隔着一层**未被声明的统计模型**。经典 ML 里（如疾病检测）能力定义明确（recall/precision），指标选择有据可依；而生成模型是 general-purpose 的，benchmark 上一个题目同时考事实记忆、语言能力、推理 —— 平均准确率究竟在估计什么 *latent ability* 是个未表态的建模选择。

**本文目标**：(1) 揭示当前评测都隐式承诺了某种能力理论；(2) 给出可选的能力理论清单（CTT、IRT、CDM、BNSM、RT）并指出它们对 AI 系统需要怎样改造；(3) 通过一个具体实验（输入扰动敏感性）证明不同能力理论会给出**系统性不同**的结论；(4) 提出 Evaluation Card 规范化评测者的建模决策。

**切入角度**：心理测量学已经为"如何从有限观察推断 latent ability"积累了 60 年方法论 —— Lord、Rasch 的工作就是为人类智力测验设计的。AI 评测本质是同一类问题，理应继承这套思维。

**核心 idea**：把 AI 评测重新定位为 **inference task**：先写出能力的生成模型 $\phi_i = f(\theta, \text{item}_i) + \text{noise}$，再讨论这个模型在 AI 场景下哪些假设站不住、需要怎么修正。

## 方法详解

本文是 position paper，不提新算法，要解决的是"benchmark 平均分到底在估计什么 latent ability"这个被默认掉的建模问题。它的做法是把评测显式写成统计推断，把 CTT/IRT/CDM/BNSM 四种能力理论纳入同一公式族对比，再用一个 prompt 扰动实验证明"选哪套理论"会系统性改变结论。

### 整体框架

论文分三层推进。第一层（Section 2）点破当前评测默认走的是 CTT（Classical Test Theory）：$\phi_i = \theta_i + \epsilon_i$、$\theta = \mathbb{E}_i[\theta_i]$，等价于假设"所有题目信息量相同、误差独立同分布"；而 tinyBenchmarks 这类 IRT-based 方法虽没明说，实际换了能力定义 —— $\theta$ 成了潜变量，两个准确率相同的模型可以有不同 ability，因为它们的错题落在了 discrimination 不同的题目上。第二层（Section 3）把五种理论纳入统一公式 $\phi_i = \theta_i + s(x_i) + r(h) + g(c) + \epsilon_i$，其中 $s(x_i)$ 是 phrasing 扰动、$r(h)$ 是温度/top-$p$ 等超参影响、$g(c)$ 是上下文影响，并指出人类心理测量学赖以成立的条件独立、mean-zero 噪声两大假设，在 AI 系统上被 Potemkin understanding、prompt sensitivity、温度依赖系统性违反。第三层（Section 4）拿 prompt 扰动当具体抓手，把四种理论各写成一个可执行推断算法，在 7 个开源 LLM × 8 个 benchmark 子任务上对比它们给出的"能力估计"差多远。

### 关键设计

**1. 统一的"能力 + 扰动"公式族：让理论差异显化为函数形式的选择**

不同评测论文各用各的隐式模型，却装作在比同一个"能力"，可比性无从谈起。本文的对策是把所有候选理论塞进同一框架，使它们的区别收敛成生成函数 $f(\theta, \text{item})$ 的不同选法。以 CTT 为例，把 $\phi_i = \theta_i + \epsilon_i$ 扩展为 $\phi_i = \theta_i + s(x_i) + \epsilon_i$，并提出 **Assumption 4.1（mean-zero perturbations）**：$\mathbb{E}_{x_i \sim \mathcal{P}_i}[s(x_i)] = 0$；Table 1 进一步给出 CTT/IRT/CDM/BNSM 在加入 $s(x_i)$ 后的统一表达式。一旦 $f$ 的形式被写死，能力定义随之锁定，"两篇论文是否在量同一件事"才有了可判定的答案。

**2. 两阶段采样诊断：证明单一 phrasing 的 benchmark 无法识别 $\theta_i$**

社区对 prompt sensitivity 的抱怨一直停留在 anecdotal 层面，本文把它升级成一个干净的可识别性命题。关键是把 benchmark item 的生成拆成两阶段：Stage 1 从 task space $\mathbb{P}$ 抽问题 $i$，Stage 2 从 phrasing 分布 $\mathcal{P}_i$ 抽具体措辞 $x_i$。Curator 通常只严格控制 Stage 1 的独立抽样，Stage 2 却往往只手写一个 $x_i$、且全部出自同一团队，于是 phrasing 之间带上结构性依赖、违反 Assumption 4.1，导致 $\theta_i$ 在 $\phi_i = \theta_i + s(x_i) + \epsilon_i$ 下根本不可识别 —— 不是模型脆弱，而是 benchmark 本身在统计上 underspecified。Proposition B.3 给出补救方向：用多次扰动 $\{x_{ij}\}_{j=1}^{m_i}$ 逼近真实 $\mathcal{P}_i$ 可减小偏差，bias $|\delta_i|$ 随近似分布 $\tilde{\mathcal{P}}_i$ 到真 $\mathcal{P}_i$ 的距离单调缩小。

**3. Evaluation Card：把建模决策强制公开**

既然没有 strictly better 的能力理论，能比的只剩"谁的假设更透明"。本文沿用 Datasheets for Datasets、Model Cards 的成功路径，给评测论文一份必填模板。Table 2 把 Evaluation Card 拆成四栏：(a) Meaning of Capability（CTT 报均值 / IRT 报 latent ability / CDM 报技能掌握度）；(b) Task Structure（是否假设 latent 技能、聚合用 DINA 还是 DINO）；(c) Sources of Systematic Variation（哪些 confounder 当噪声、哪些显式建模）；(d) Data Considerations（IRT 需校准的 item parameters、CDM 需要的 skill-to-item 先验图）。其精神是用"说清楚假设"替代"找最对的模型"，把评测者被迫做却从不声明的决策摆到台面上。

### 损失函数 / 训练策略

四种理论虽不涉及训练，却各对应一个推断算法，统一配 item-level bootstrap 出 uncertainty：CTT 用 clustered bootstrap，把 item 当 cluster、扰动当 within-cluster 观察；IRT 用 Fisher scoring / Newton-Raphson 求 $\hat{\theta}$ 的 MLE；CDM 用 logistic likelihood + Gaussian prior 的 MAP，并把 $\alpha \in \{0,1\}^K$ 放松为 $\alpha \in \mathbb{R}^K$；BNSM 用 Bayesian network 做后验推断。

## 实验关键数据

### 主实验

实验设置：7 个开源 instruction-tuned LLM（Llama-3.2、Qwen-2.5、Gemma 三家）× 2 个 benchmark（BBH、LMEntry）× 各 4 个子任务（AWFC/FA/ML/RW、CJ/MR/FF/S），用 mizrahi2024state 已发布的扰动版本作为 phrasing distribution proxy。

| 对比维度 | CTT（均值） | IRT（latent ability） | 关键差异 |
|---------|-------------|----------------------|----------|
| 模型排名一致性 | baseline | 与 CTT 总体一致 | 一致 |
| 模型间分离度 | 较小 | 显著放大 | IRT 在 AWFC 上把 Qwen-3.5B 拉到最高分（因为它在难题上表现好） |
| 样本复杂度 | 全量 | 自适应抽题 | IRT 用更少样本（结果中粗体数字）达到同样推断 |
| 同准确率不同 ability | 不可分辨 | 可分辨 | Qwen-3.5B 在 AWFC 和 S 上准确率相同，但 IRT 算出 S 的 ability 更低，因为 AWFC 题更难 |

### 消融 / 对比

| 配置 | 关键发现 | 解释 |
|------|---------|------|
| CTT vs IRT | 排名大体一致但分离度差异显著 | "高准确率 ≠ 高 ability"，难题加权信息更多 |
| CDM vs BNSM（Movie Recommendation 任务） | BNSM 把 Social Reasoning 评得明显更高 | CDM 只能用低 $Soc$ 解释 MR 上的差表现；BNSM 引入 World Knowledge ($W$) 作为额外 latent skill，把 MR/CJ 的错归因到低 $W$ 而非低 $Soc$ |
| 原 phrasing vs 扰动均值 | 差异 $D_i = \phi_i^{\text{orig}} - \bar{\phi}_i$ 在多个任务上显著非零 | 单一 phrasing 的 benchmark 是 biased estimator |

### 关键发现
- **理论选择直接改变结论**：同一组 raw answer 数据，套 CTT vs IRT 可能得出"两个模型能力一样" vs "差 2 个标准差"两种结论，这是建模决策的产物，不是测量误差。
- **CDM/BNSM 的 skill structure 假设权力极大**：MR 任务上一个 latent skill 的引入与否就能把 Social Reasoning 评分翻倍 —— Evaluation Card 第二栏（Task Structure）的重要性由此凸显。
- **没有 "correct" 理论**：作者刻意不推荐任何一个 —— 真正的 take-away 是"你选了哪个、为什么选、它的假设在 AI 场景成立吗"必须写在评测论文里。
- **扰动不是为了找真值，而是为了识别**：作者重新解释 prompt perturbation 这类工作 —— 它们的本质贡献是扩大 phrasing space 覆盖度、改善 $\theta_i$ 的可识别性，而非"测真实能力"。

## 亮点与洞察

- **把社区 anecdotal 抱怨升级成统计可识别性问题**：大家都吐槽 prompt sensitivity，但这篇用两阶段采样（task space + phrasing space）干净地证明"单一 phrasing 的 benchmark 在 $\phi = \theta + s(x) + \epsilon$ 下 $\theta$ 不可识别"，把工程问题变成了可证伪的数学问题。
- **指出 IRT-based 评测正在"暗中改定义"**：tinyBenchmarks、adaptive testing 这些被当作"省样本的工程优化"的工作，其实换了能力定义 —— 这个观察非常锐利，是整篇论文最有冲击力的一击。
- **AI vs 人类的"deliberation time"重新定义**：Section 3.3 把 reasoning-token 数 / FLOPs 当作 AI 版的 response time，套 log-normal 模型，给 agentic / o1-style 系统的 speed-accuracy tradeoff 评测留了一个可扩展的接口，思路可迁移到任何"算多算少"的推理模型。
- **Evaluation Card 这套范式**：直接复用 Datasheets/Model Cards 的成功路径，写论文/审稿/复现 AI 评测时都可以拿来当 checklist —— 比单纯呼吁"做更好评测"实用得多。

## 局限与展望

- **作者自承**：能力理论清单非穷尽（还有 Mokken scaling、BKT、PFA、SDT、DDM 等）；prompt perturbation 只是 confounder 之一，温度、上下文、解码策略尚未实证。
- **方法论局限**：DINA 模型里需要专家给 Q-matrix（skill-to-item mapping），BNSM 需要 skill 依赖图先验 —— 在通用 benchmark 上谁来定义这些先验本身是开放问题，等于把"主观性"从准确率换到了 prior 上。
- **实验规模**：只有 7 个开源 LLM × 8 个子任务，没碰前沿闭源模型；扰动数据集复用 mizrahi2024state，扰动质量未独立验证。
- **缺乏与下游有效性的连接**：作者在 Alternative View 2 承认"interpretability ≠ downstream validity"，但论文里没给出"显式能力理论估计的 ability 是否更能预测部署表现"的实证，这是个明显的下一步实验。
- **改进思路**：把 confounder taxonomy（temperature/top-$p$/system prompt/inference-time strategies）真正系统化、给每个常见 benchmark 提供"推荐 Evaluation Card 填法"作为社区基础设施。

## 相关工作与启发

- **vs Datasheets for Datasets / Model Cards (gebru2021datasheets)**：思路一致（强制公开建模决策），目标层不同 —— 那两套规范数据和模型，这篇规范的是**评测过程**本身。
- **vs tinyBenchmarks (polo2024tiny) / adaptive testing (zhuang2023static)**：这些 IRT-based 方法被本文重新解读 —— 它们的真正贡献不是"减少样本"，而是"换了一套能力定义"，本文给了它们一个理论解释框架。
- **vs robustness 一系列工作 (mizrahi2024state; sclar2023quantifying; zheng2023large)**：把 prompt perturbation 重新定位为"提升 $\theta_i$ 可识别性"，而不是"测鲁棒性"，这个视角转换让原本看起来零散的扰动技巧有了统一动机。
- **vs Mitchell (mitchell2024debates) / Hardt (hardt2025emerging) 的 benchmark science 呼吁**：本文是这股潮流的一次具体方法论实现，可看作"science of benchmarks"运动的一篇技术白皮书。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把心理测量学引入 LLM 评测不算第一个，但把它形式化成可对比的能力理论族、配实证 demo 是相对新的组合
- 实验充分度: ⭐⭐⭐ 7×8 规模较小且没碰闭源模型，主要价值在 proof-of-concept 而非 benchmark
- 写作质量: ⭐⭐⭐⭐⭐ 论证链清晰，公式与直觉穿插得当，Alternative Views 一节体现了 position paper 应有的辩论自觉
- 价值: ⭐⭐⭐⭐ Evaluation Card 有立刻被审稿规范采纳的潜力，对 LLM 评测社区有方向性影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)

</div>

<!-- RELATED:END -->
