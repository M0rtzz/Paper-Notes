---
title: >-
  [论文解读] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?
description: >-
  [ACL 2026][模型压缩][小模型教师] LightReasoner 用较弱的 Amateur 模型和较强的 Expert 模型之间的 token 分布差异来自动找出高价值推理步骤，再只对这些步骤做对比式自蒸馏，使数学推理模型在接近或超过 SFT 的同时显著减少采样、训练时间和调参 token。
tags:
  - "ACL 2026"
  - "模型压缩"
  - "小模型教师"
  - "推理蒸馏"
  - "KL散度"
  - "选择性微调"
  - "LoRA"
---

# LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?

**会议**: ACL 2026  
**arXiv**: [2510.07962](https://arxiv.org/abs/2510.07962)  
**代码**: https://github.com/HKUDS/LightReasoner  
**领域**: 模型压缩 / LLM 推理 / 高效微调  
**关键词**: 小模型教师, 推理蒸馏, KL散度, 选择性微调, LoRA

## 一句话总结
LightReasoner 用较弱的 Amateur 模型和较强的 Expert 模型之间的 token 分布差异来自动找出高价值推理步骤，再只对这些步骤做对比式自蒸馏，使数学推理模型在接近或超过 SFT 的同时显著减少采样、训练时间和调参 token。

## 研究背景与动机
**领域现状**：提升 LLM 数学推理能力的常见路线是 rejection-sampling SFT：先让模型生成多条推理轨迹，用答案或验证器筛出正确轨迹，再把整条轨迹作为监督数据进行微调。这类方法直接、有效，也和 Chain-of-Thought、RFT 等推理增强范式相容。

**现有痛点**：rejection SFT 的代价非常高。它需要完整生成候选解、用 ground truth 或外部验证器筛选，还会把整条推理链上的所有 token 一视同仁地优化。论文指出，很多 token 只是常规连接词或低信息量步骤，真正决定推理成败的往往是少数关键转折点，因此全轨迹训练会把计算资源浪费在低回报 token 上。

**核心矛盾**：强模型已经具备一部分潜在推理能力，但现有训练信号往往依赖外部答案或人工构造数据；另一方面，弱模型虽然能力不足，却能在同一前缀下暴露出“哪里会走偏”。本文的核心矛盾是：如何不用标签、不完整生成轨迹，也能识别强模型相对于弱模型真正有优势的推理时刻。

**本文目标**：作者希望构造一种 verifier-free 的推理增强框架，自动定位高价值 token，只在这些 token 上训练 Expert，并让训练信号不仅是 Expert 自己的 one-hot 输出，而是体现 Expert 相对 Amateur 的优势分布。

**切入角度**：作者观察 Expert 和 Amateur 在同一 prefix 下的 next-token 分布。如果两者高度一致，该 token 大概率只是常规步骤；如果 KL 散度突然升高，则可能对应算术操作、逻辑转折或中间结论等关键推理点。论文还给出统计：约 60% token 的 KL 落在 $[0.0, 0.1)$，只有约 20% token 超过 0.4；当 Expert 和 Amateur top-1 不一致时，平均 KL 为 1.99，而 top-1 一致时为 0.166。

**核心 idea**：用 Expert-Amateur 分布差异替代人工标签和全轨迹 SFT，把弱模型变成“反面参照物”，只蒸馏 Expert 相对 Amateur 最明显的推理优势。

## 方法详解
LightReasoner 可以理解为一种面向推理模型的选择性自蒸馏。它不让小模型直接给大模型生成答案，也不是在推理时做双模型对比解码，而是在训练数据构造阶段比较两者的 token 分布，并把高差异步骤转化为 soft supervision。

### 整体框架
输入是一批推理问题，论文主实验使用 GSM8K 训练集生成监督样本。对于每个问题，Expert 模型先按 CoT 方式生成短前缀推理轨迹，采样 rollout 长度限制为 128 token。对轨迹上的每个 prefix $s_t$，同时计算 Expert 分布 $\pi_E(\cdot\mid s_t)$ 和 Amateur 分布 $\pi_A(\cdot\mid s_t)$。

第一阶段是采样和筛选：如果 $D_{KL}(\pi_E\|\pi_A)>\beta$，该步骤被认为是 informative step。第二阶段是构造对比监督：在 Expert 高置信 token 的 mask 支撑集上计算 $\log \pi_E(a\mid s_t) / \pi_A(a\mid s_t)$，再归一化成 soft target $v_C$。第三阶段是微调：用 LoRA 训练同一个 Expert，使它的输出分布靠近 $v_C$，从而强化 Expert 已经比 Amateur 做得好的推理决策。

### 关键设计

**1. KL 驱动的信息步骤筛选：用强弱模型的分歧定位高价值 token**

一条推理轨迹里大部分 token 只是连接词或低信息步骤，把训练预算平摊到所有 token 上既浪费又会稀释关键信号。LightReasoner 的做法是在同一 prefix $s_t$ 下比较 Expert 与 Amateur 的 next-token 分布，用 KL 散度 $D_{KL}(\pi_E(\cdot\mid s_t)\|\pi_A(\cdot\mid s_t))$ 作为该步骤“是否值得训练”的代理：KL 越大，说明两个模型在这一步的选择差异越明显，往往对应算术操作、符号转换或逻辑跳转这类瓶颈步骤。主实验取阈值 $\beta=0.4$，只有超过该值的步骤才被标为 informative step 进入后续训练。相比固定前缀长度或人工规则，这个分歧信号能贴着每条轨迹自身的实际难点走——论文统计也佐证了它的判别力：约 60% token 的 KL 落在 $[0.0,0.1)$，Expert 与 Amateur top-1 一致时平均 KL 仅 0.166，而 top-1 不一致时平均 KL 高达 1.99。

**2. 对比式分布监督：训练标签编码的是“Expert 比 Amateur 强在哪”，而非 Expert 自己的 one-hot 输出**

如果直接拿 Expert 生成的 token 当硬标签，会丢掉整个分布的信息，还容易把 Expert 的偶然输出当成唯一真相。LightReasoner 改用对比 soft label：先用 $\alpha=0.2$ 砍掉 Expert 的低概率尾部，只保留满足 $\pi_E(a\mid s_t)\geq\alpha\max_b\pi_E(b\mid s_t)$ 的 token，再在这个支撑集上算对比得分 $v'_C(a\mid s_t)=\log\pi_E(a\mid s_t)/\pi_A(a\mid s_t)$，最后 softmax 归一化成 soft target $v_C$。这样标签强调的是 Expert 相对 Amateur 的优势 margin 而非绝对置信度，既保留了分布形状，又能弱化低置信噪声。消融也显示这一步最关键：去掉对比监督后平均分从 54.0 掉到 44.8。

**3. 短 rollout 与 LoRA 自蒸馏：把监督构造和微调都压到低成本**

完整生成长答案不仅贵，后期 token 还会受错误级联污染，产生假阳性的“高价值”步骤。LightReasoner 因此把采样 rollout 限制在前 128 token——论文认为早期推理步骤更稳定，越往后越容易被前面的错误带偏。微调阶段复用同一个 Expert，用 LoRA 训练 1000 steps、每步 16 个对比监督样本，损失就是让 Expert 输出去匹配 $v_C$。短 rollout、选择性 token、轻量 LoRA 三者叠加，使 LightReasoner 在采样问题数和调参 token 上都远低于 rejection SFT。

### 损失函数 / 训练策略
训练目标是让 Expert 输出分布匹配对比监督 $v_C$：$\mathcal{L}(s_t)=D_{KL}(v_C(\cdot\mid s_t)\|\pi_E(\cdot\mid s_t))$。由于 $v_C$ 对当前训练参数是常量，该目标等价于 $-\sum_a v_C(a\mid s_t)\log\pi_E(a\mid s_t)$。实验中 Expert 包括 Qwen2.5-Math-1.5B/7B、Instruct 版本和 DeepSeek-R1-Distill-Qwen-1.5B，Amateur 固定为 Qwen2.5-0.5B。

## 实验关键数据

### 主实验
主结果使用 zero-shot pass@1 或文中说明的对应评估设置，覆盖 7 个数学推理基准。下表摘取 AVG 和若干代表性模型，说明 LightReasoner 在多数模型上能超过或接近 rejection SFT。

| Expert 模型 | 方法 | GSM8K | MATH | SVAMP | ASDiv | MMLU STEM | AVG |
|--------|------|------|------|-------|-------|-----------|-----|
| Qwen2.5-Math-1.5B | Baseline | 42.5 | 34.2 | 68.8 | 68.1 | 49.8 | 42.4 |
| Qwen2.5-Math-1.5B | SFT | 69.2 | 57.1 | 64.1 | 70.2 | 47.7 | 50.1 |
| Qwen2.5-Math-1.5B | LightR | 70.6 | 59.3 | 76.0 | 79.8 | 54.9 | 54.2 |
| DeepSeek-R1-Distill-Qwen-1.5B | Baseline | 75.2 | 54.2 | 79.9 | 84.9 | 22.3 | 50.3 |
| DeepSeek-R1-Distill-Qwen-1.5B | SFT | 78.2 | 60.3 | 81.5 | 87.4 | 26.2 | 53.3 |
| DeepSeek-R1-Distill-Qwen-1.5B | LightR | 79.5 | 60.2 | 83.5 | 87.5 | 26.2 | 55.9 |
| Qwen2.5-Math-7B | Baseline | 57.5 | 51.8 | 67.9 | 72.7 | 69.8 | 50.0 |
| Qwen2.5-Math-7B | SFT | 64.4 | 63.3 | 76.2 | 76.6 | 68.5 | 54.5 |
| Qwen2.5-Math-7B | LightR | 67.9 | 57.8 | 77.2 | 80.6 | 70.5 | 54.7 |

### 消融实验
论文的消融在 Qwen2.5-Math-1.5B 上逐步移除 step selection 和 contrastive supervision。完整 LightReasoner 平均 54.0，高于 rejection SFT 的 50.6；去掉 contrast 后平均降到 44.8，说明对比监督比单纯筛 token 更关键。

| 配置 | GSM8K | MATH | SVAMP | ASDiv | Minerva Math | Olympiad Bench | AVG |
|------|------|------|-------|-------|--------------|----------------|-----|
| Baseline | 42.5 | 34.2 | 68.8 | 68.1 | 9.9 | 23.7 | 41.2 |
| Rejection SFT | 69.2 | 57.1 | 64.1 | 70.2 | 15.1 | 27.6 | 50.6 |
| GT Supervision | 43.4 | 34.8 | 70.4 | 69.7 | 10.2 | 19.8 | 41.4 |
| Full LightReasoner | 70.6 | 59.3 | 76.0 | 79.8 | 11.4 | 27.1 | 54.0 |
| 无 step selection, 有 contrast | 67.6 | 58.8 | 78.7 | 80.5 | 11.0 | 26.4 | 53.8 |
| 有 step selection, 无 contrast | 62.0 | 53.1 | 56.6 | 61.0 | 10.7 | 25.5 | 44.8 |
| 二者都移除 | 55.5 | 50.2 | 50.0 | 65.4 | 10.4 | 24.0 | 42.6 |

### 关键发现
- 效率表显示，Qwen2.5-Math-1.5B 上 SFT 需要 4.0h、3952 个问题和 1.77M tuned tokens，而 LightReasoner 只需要 0.5h、1000 个问题和 0.02M tuned tokens，平均增益反而从 +7.7% 提高到 +11.8%。
- Qwen2.5-Math-7B 上，SFT 为 9.5h、6029 个问题、2.20M tokens，LightReasoner 为 0.75h、1000 个问题、0.02M tokens，平均增益相近或略高。
- 整体口径上，论文报告最多 28.1% 的准确率提升，同时节省约 90% 时间、80% sampled problems 和 99% tuned tokens。
- 机制分析表明，Expert-Amateur 能力差越合适，对比信号越有效；若 Amateur 接近或强于 Expert，收益会减弱甚至退化。

## 亮点与洞察
- LightReasoner 的巧妙之处在于把“弱模型”从传统蒸馏中的学生，反转成识别强模型优势的参照物。它不是让小模型教大模型答案，而是让小模型暴露自己不会的地方，从而提醒大模型哪些 token 最值得强化。
- 方法把 contrastive decoding 的思想从推理时搬到训练时。这样保留了 Expert-Amateur 对比的优势，同时避免了每次推理都运行两套模型带来的延迟和显存开销。
- 选择性 token 训练的证据比较充分：KL 分布、top-1 分歧、消融表和效率表都指向同一个结论，即推理能力不是均匀分布在整条轨迹上，而是集中在少数高杠杆决策点。
- 对模型压缩和高效微调的启发是，压缩不一定只意味着把大模型知识迁移到小模型；也可以利用小模型的失败模式反向提高大模型训练效率。

## 局限与展望
- 论文主要评估数学推理，包括 GSM8K、MATH、SVAMP、ASDiv、Minerva Math、Olympiad Bench 和 MMLU STEM；代码推理、工具调用、开放式规划等领域是否同样有效仍需验证。
- Expert-Amateur 配对依赖合适的能力差。能力差太小会导致对比信号不足，能力差为负甚至可能误导 Expert，因此自动选择 Amateur 或动态调整配对是后续关键问题。
- $\alpha$ mask 和 $\beta$ filtering 都是额外超参，虽然论文给出默认值 $\alpha=0.2$、$\beta=0.4$，但不同任务、不同模型族可能需要重新调参。
- 实验覆盖小到中等规模开源模型，尚未证明在更大闭源模型或强推理模型上的可扩展性。

## 相关工作与启发
- **vs rejection SFT / RFT**: SFT 依赖完整轨迹和答案验证，本文只用短 prefix 的分布差异构造监督，优势是成本低且不依赖 ground-truth，劣势是需要可访问两个模型的 logits。
- **vs Contrastive Decoding**: CD 在推理时同时运行 Expert 和 Amateur，本文把对比信号蒸馏进 Expert，优势是推理阶段不增加双模型开销，但训练前需要额外采样和分布计算。
- **vs RHO-1 / selective token training**: RHO-1 等方法关注 token 学习价值，LightReasoner 的不同点是用同族模型的领域能力差来定义 token 价值，不需要外部 reference scorer。
- **对后续工作的启发**: 可以把 Expert-Amateur KL 用作通用“学习价值探针”，用于代码修复、工具规划或多模态推理中的局部监督构造。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用弱模型失败模式反向增强强模型，视角很有辨识度。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 5 个 Expert 和 7 个数学基准，消融清楚，但跨领域验证不足。
- 写作质量: ⭐⭐⭐⭐ 方法动机和效率论证清晰，部分表格在缓存文本中排版较密。
- 价值: ⭐⭐⭐⭐⭐ 对高效推理微调、标签稀缺场景和选择性训练都有直接参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](tlora_task-aware_low_rank_adaptation_of_large_language_models.md)

</div>

<!-- RELATED:END -->
