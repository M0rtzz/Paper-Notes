---
title: >-
  [论文解读] Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding
description: >-
  [ACL2026][信息检索/RAG][摘要事实性] 本文提出 ConSUM，在摘要生成候选中同时考察候选对源文档的事实一致性和候选之间的共识，用 MBR decoding 结合 FENICE/FIZZ 等事实性指标重排序，在 CNN/DailyMail、XSum 和人工评估中提升摘要事实可靠性。
tags:
  - "ACL2026"
  - "信息检索/RAG"
  - "摘要事实性"
  - "MBR decoding"
  - "伪参考摘要"
  - "reference-free metric"
  - "reranking"
---

# Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding

**会议**: ACL2026  
**arXiv**: [2605.29336](https://arxiv.org/abs/2605.29336)  
**代码**: https://github.com/naist-nlp/ConSUM  
**领域**: 摘要生成 / 事实一致性 / 重排序  
**关键词**: 摘要事实性, MBR decoding, 伪参考摘要, reference-free metric, reranking

## 一句话总结
本文提出 ConSUM，在摘要生成候选中同时考察候选对源文档的事实一致性和候选之间的共识，用 MBR decoding 结合 FENICE/FIZZ 等事实性指标重排序，在 CNN/DailyMail、XSum 和人工评估中提升摘要事实可靠性。

## 研究背景与动机
**领域现状**：自动摘要系统通常先由生成模型产出一个或多个候选摘要，再用 ROUGE、BERTScore、事实性评估器或 reranker 选择更好的输出。由于测试时没有人工 gold summary 可用，很多 reference-free reranking 方法只能把源文档当作唯一依据，判断候选摘要是否忠实于输入。

**现有痛点**：只依赖源文档的 reference-free 指标并不稳定。一方面，源文档很长，评估器可能只粗粒度地判断候选和源文档是否相关，漏掉细小但关键的事实错误；另一方面，单一指标容易把 reranking 推向该指标自身的偏好，例如更长摘要、更容易被某个事实抽取器识别的摘要，未必是真正更好的摘要。

**核心矛盾**：摘要事实性需要同时满足两个条件：它必须和源文档一致，也应当落在生成模型自身认为可信的语义区域中。前者对应 consistency，后者对应 consensus。过去的 reranking 往往只优化其中一个信号，导致选择结果容易被 metric bias 或单个异常候选牵着走。

**本文目标**：作者希望在没有人工参考摘要的测试场景中，仍然构造一个可用的“参考信号”。具体来说，系统要从同一模型采样出的候选与伪参考中选择最终摘要，既利用 source-based factuality metric，也利用候选和伪参考之间的 NLI 式一致性。

**切入角度**：Minimum Bayes Risk decoding 在机器翻译中常用于从候选池中选择期望效用最高的输出。本文把这一思想移到摘要事实性上：如果一个候选和多个同源伪参考都保持一致，它更可能代表模型分布中的稳定事实；再加上源文档一致性检查，就能过滤掉“看起来流畅但偏离事实”的候选。

**核心 idea**：用“候选间共识”补足“源文档一致性”，通过 MBR 分数和 reference-free factuality 分数的加权组合来选择更可靠的摘要。

## 方法详解
ConSUM 不是重新训练一个摘要模型，而是在解码之后做候选选择。它把摘要生成看成三步：先生成候选摘要和伪参考摘要，再分别计算 consensus score 与 consistency score，最后把两个分数归一化后加权融合，选出最终摘要。

### 整体框架
输入是一篇源文档 $s$ 和一个已有的摘要生成模型。系统首先从模型中采样两组文本：候选集合 $\mathcal{Y}$ 用作可被选择的最终输出，伪参考集合 $\mathcal{R}$ 用作近似 gold reference 的内部参照。两组文本可以来自不同采样策略，因此候选池负责提供多样输出，伪参考池负责估计模型对事实内容的共识。

随后，每个候选 $y_i$ 会得到两个分数。第一个是 consistency score，即候选和源文档之间的事实一致性，本文用 FENICE 或 FIZZ 这样的 reference-free factuality metric 计算。第二个是 consensus score，即候选和伪参考集合之间的平均效用，本文用 MENLI 作为 utility function，衡量候选与每个伪参考之间的 NLI 式事实一致性。

最后，ConSUM 对两个分数做 z-score 标准化，并用权重 $w$ 融合：$S_{fin}=wZ(S_{sen})+(1-w)Z(S_{sis})$。当 $w=0$ 时退化为只看源文档一致性，当 $w=1$ 时退化为只看 MBR 共识。实验中作者选择 $w=0.75$ 作为统一默认设置，因为它在 CNN/DM 上最优，并且在 XSum 上仍具竞争力。

### 关键设计
1. **候选集合与伪参考集合分离**:

    - 功能：把可被选中的摘要候选 $\mathcal{Y}$ 和用于估计共识的伪参考 $\mathcal{R}$ 区分开。
    - 核心思路：两者都由同一摘要模型在源文档条件下采样，但可以采用不同 decoding 策略。候选集合强调多样性，伪参考集合强调对模型真实分布的覆盖。
    - 设计动机：如果简单令 $\mathcal{Y}=\mathcal{R}$，MBR 容易被候选池自身的采样偏差影响；分离后可以独立调候选生成和共识估计，降低把异常候选当作共识的风险。

2. **用 MENLI 做事实共识型 MBR**:

    - 功能：衡量候选摘要与伪参考摘要之间是否在事实层面一致。
    - 核心思路：对每个候选 $y_i$，计算它与所有伪参考 $r_j$ 的 MENLI utility，然后取平均：$S_{sen}(y_i,\mathcal{R})=\frac{1}{|\mathcal{R}|}\sum_j u(y_i,r_j)$。
    - 设计动机：ROUGE 或 BERTScore 更偏词面/语义相似，而本文关注 factual agreement。用 NLI-based MENLI 做 MBR utility，可以让“共识”更接近事实一致性，而不是只选择最常见措辞。

3. **MBR 与 source consistency 的加权融合**:

    - 功能：同时利用模型内部分布信号和源文档事实信号。
    - 核心思路：用 FENICE/FIZZ 得到 $S_{sis}$，用 MBR 得到 $S_{sen}$，分别标准化后加权组合，再选择 $\arg\max_y S_{fin}(y,s,\mathcal{R})$。
    - 设计动机：MBR 可能偏好更长或更容易被 MENLI 认可的摘要，reference-free 指标又可能漏掉细粒度错误。二者组合可以降低单一指标 reward hacking，把候选推向既代表共识又忠实源文档的位置。

### 损失函数 / 训练策略
本文没有训练新的生成模型，也没有学习一个监督 reranker。关键“训练策略”体现在推理期的候选采样和超参数选择上。PLM 使用 epsilon sampling 或 diverse beam search，LLM 使用 nucleus sampling；伪参考固定为 64 个 epsilon sampling 样本。作者在 $w\in\{0,0.25,0.5,0.75,1.0\}$ 上做权重敏感性实验，并将 $w=0.75$ 作为最终系统默认值。统计显著性用 paired-bootstrap resampling，迭代 10,000 次，并做 Bonferroni correction。

## 实验关键数据

### 主实验
| 数据集 / 评估 | 指标或设置 | 本文关键结果 | 对照 | 结论 |
|--------|------|------|----------|------|
| CNN/DM | FIZZ 分数，epsilon setting | FENICE-0.75 将 Fi 从 39.36 提到 52.44 | Baseline 39.36 | 事实性显著提升 |
| XSum | FIZZ 分数，epsilon setting | FENICE-0.75 将 Fi 从 16.91 提到 27.79 | Baseline 16.91 | 对高幻觉抽象摘要更有效 |
| CNN/DM | MENLI-Entailment | 从 4.46 提到 10.44 | Baseline 4.46 | 共识信号改善 entailment |
| XSum | MENLI-Entailment | 从 -31.15 提到 -20.36 | Baseline -31.15 | 负分场景下仍有明显改善 |
| Human Eval / CNN/DM | Overall | FENICE-0.75 为 4.63 | Baseline 4.56, MBR-1.0 为 4.57, Gold 为 3.92 | 人工评价最偏好 FENICE-0.75 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FENICE, $w=0.75$ | CNN/DM 81.05, XSum 77.52 | 两个数据集都较稳，是最终默认配置的重要依据 |
| FIZZ, $w=0.75$ | CNN/DM 71.08, XSum 55.03 | 相比 $w=0$ 的 14.15 / 17.37，大幅受益于 MBR 共识 |
| SimCLS, $w=1.0$ | CNN/DM 65.35, XSum 90.91 | reference-free 部分反而伤害 SimCLS，最终系统排除 SimCLS |
| MBR-only, $w=1.0$ | FENICE: CNN/DM 68.86, XSum 39.70 | 只靠 MBR 不够稳，说明仍需要源文档一致性约束 |
| Human Eval / Factuality | FENICE-0.75 为 4.87 | 高于 MBR-1.0 的 4.74、FIZZ-0.75 的 4.77、Baseline 的 4.79 |

### 关键发现
- 最有效的设置不是只看 reference-free factuality，也不是只看 MBR，而是二者融合；$w=0.75$ 表明共识信号占主导，但仍需要源文档一致性兜底。
- XSum 的收益尤其说明问题：该数据集摘要更抽象、幻觉更频繁，ConSUM 通过伪参考共识过滤掉更明显的事实偏差。
- Oracle 分数仍远高于当前方法，作者指出许多指标上 oracle 可以超过 ConSUM 最优分数两倍以上，说明候选池中存在更好摘要，但选择器还没有充分识别它们。
- FENICE 与 FIZZ 虽然都是 atomic fact 风格指标，但粒度差异很大：FENICE 通常抽 3-6 个 ACU，FIZZ 会抽取更多事实单元，因此二者和最终事实分数的关系并不完全一致。

## 亮点与洞察
- 这篇论文最好的点是把“没有 gold reference”转化为“从模型分布里构造伪参考”。它没有假设伪参考一定正确，而是把多个伪参考的平均一致性当作抗噪信号，思路很实用。
- 候选和伪参考分离是一个细节但很关键。它提醒我们：用采样集合做自评价时，候选多样性和参考代表性是两个目标，混在一起会让评估信号变脏。
- 论文没有盲目追求单个 factuality metric 的最优，而是显式讨论 metric bias。对于任何“用评估器指导生成”的任务，这都是可迁移的经验：最好把目标指标和约束指标拆开建模。
- 人工评估中 gold reference 得分不高，是一个有趣但合理的信号。CNN/DM 的参考摘要质量本身存在争议，因此“超过 gold”不必解读为模型完全更好，而应理解为该 benchmark 的 reference 噪声确实会影响自动评价。

## 局限与展望
- MBR 的计算复杂度是 $O(n^2)$，候选数和伪参考数增加会快速放大计算成本。本文只探索了候选/伪参考数量等少量设置，utility function 和伪参考生成策略还没有系统展开。
- FENICE 和 FIZZ 在大规模处理时也有计算瓶颈，导致作者无法更充分地搜索权重、指标组合和采样策略。
- 实验只覆盖 CNN/DM 与 XSum 两个英文新闻摘要数据集。两者之间的最优权重差异已经很明显，说明方法迁移到长文档、对话摘要、医学/法律摘要或多语言摘要时可能需要重新校准。
- 未来可以考虑更高效的 MBR 近似、学习式权重选择、按源文档/候选特征动态调整 $w$，以及把伪参考共识扩展到多模型 ensemble，而不是只从同一模型采样。

## 相关工作与启发
- **vs source-only reranking**: 传统 reference-free reranking 主要比较候选和源文档，本文额外引入候选间共识，优势是可以捕捉源文档评估器漏掉的异常事实，但代价是更高采样和 pairwise scoring 成本。
- **vs MBR decoding in NMT**: 机器翻译中的 MBR 常用候选和伪参考的相似度选译文，本文把 utility 换成 MENLI，并把任务目标从语义相似改成事实一致性，更贴合摘要幻觉问题。
- **vs SimCLS / BERTScore-style consensus**: 这些方法更强调语义或质量相似度，ConSUM 的 MENLI utility 更关注 entailment 与 contradiction，因此更适合 factuality-sensitive summarization。
- **启发**：在没有人工参考答案的生成任务中，可以用“模型自采样共识 + 外部一致性约束”构造弱监督选择器。这一范式也可能用于问答、报告生成、代码解释和开放式信息抽取。

## 评分
- 新颖性: ⭐⭐⭐⭐ MBR 和 factuality metric 本身不是新概念，但把候选共识与源文档一致性系统结合到摘要事实性 reranking 上很清晰。
- 实验充分度: ⭐⭐⭐⭐ 自动指标、权重分析、人工评估和 oracle 分析比较完整，但数据域仍局限于英文新闻摘要。
- 写作质量: ⭐⭐⭐⭐ 方法动机明确，图表和附录充分；不足是主表较大，读者需要在大量 metric 中抓主线。
- 价值: ⭐⭐⭐⭐ 对事实性摘要、无参考 reranking 和采样式 decoding 都有直接参考价值，尤其适合已有生成模型的推理期增强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[ACL 2025\] Reranking-based Generation for Unbiased Perspective Summarization](../../ACL2025/information_retrieval/reranking-based_generation_for_unbiased_perspective_summarization.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[ICML 2026\] CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels](../../ICML2026/information_retrieval/care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois.md)

</div>

<!-- RELATED:END -->
