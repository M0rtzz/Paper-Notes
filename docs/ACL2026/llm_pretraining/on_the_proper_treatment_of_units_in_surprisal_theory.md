---
title: >-
  [论文解读] On the Proper Treatment of Units in Surprisal Theory
description: >-
  [ACL2026][预训练][惊奇理论] 这篇论文指出 surprisal theory 中“下一个单位”的单位选择一直被预训练语言模型 tokenizer 悄悄决定，因而提出一个把模型 token、语言学单位和实验 ROI 明确分离的有限状态转导框架…
tags:
  - "ACL2026"
  - "预训练"
  - "惊奇理论"
  - "单位库存"
  - "有限状态转导"
  - "眼动阅读时间"
  - "GPT-2"
---

# On the Proper Treatment of Units in Surprisal Theory

**会议**: ACL2026  
**arXiv**: [2604.28147](https://arxiv.org/abs/2604.28147)  
**代码**: https://github.com/samuki/units-surprisal  
**领域**: LLM预训练 / 心理语言学 / 语言模型评估  
**关键词**: 惊奇理论、单位库存、有限状态转导、眼动阅读时间、GPT-2

## 一句话总结
这篇论文指出 surprisal theory 中“下一个单位”的单位选择一直被预训练语言模型 tokenizer 悄悄决定，因而提出一个把模型 token、语言学单位和实验 ROI 明确分离的有限状态转导框架，并在 MECO 眼动数据上验证不同单位库存会改变 surprisal 对阅读时间的预测问题本身。

## 研究背景与动机
**领域现状**：Surprisal theory 用 $-\log p(u_t \mid u_{<t})$ 解释人类语言理解中的处理负担：越不可预测的语言单位，通常越难处理、阅读时间越长。早期研究往往自己训练 PCFG、n-gram 或小型语言模型，因此可以把模型的基本 alphabet 设成实验所需的词、PTB token 或其他语言学单位。

**现有痛点**：大规模预训练语言模型普及后，研究者通常直接继承 GPT-2、LLaMA 等模型的 BPE/token alphabet。问题是模型 token 并不等于语言学上的 word、morpheme、phoneme，也不一定对齐眼动实验中的区域边界。于是很多论文必须事后把 token surprisal 拼成 word surprisal，或者用 leading/trailing whitespace 之类规则处理边界，这些做法常常把“科学问题中想分析什么单位”和“模型内部怎么分词”混在一起。

**核心矛盾**：Surprisal theory 真正需要定义的是人类理解中的单位 $U$，而预训练 LM 提供的是模型 alphabet $\Sigma$ 上的概率分布。两者不一致时，如果直接把 $\Sigma$ 当成 $U$，就把 tokenizer 这种工程细节误当成心理语言学假设；如果用简单空白字符规则转换，又会在标点、缩写、数字、句首词等上下文依赖场景中产生不一致。

**本文目标**：作者要解决三个子问题：第一，形式化区分模型 alphabet、实验单位库存和 ROI；第二，给出从 token-level LM 得到 arbitrary unit-level surprisal 的通用计算方式；第三，用真实眼动数据展示单位选择不仅影响数值大小，还会改变回归观测、控制变量和显著性解释。

**切入角度**：论文从一个很朴素但重要的观察出发：tokenization should be an implementation detail, not a scientific primitive。也就是说，研究者应该先根据理论问题选择 units，再把语言模型转换到这个 unit inventory 上，而不是反过来被 tokenizer 限定。

**核心 idea**：把 unit parser $\rho: \Sigma^* \to U^*$ 表示为可组合的有限状态转导器，并通过 pushforward / transduced language model 在任意合理单位库存上计算 next-unit surprisal。

## 方法详解
这篇论文的方法不是训练一个新模型，而是给 surprisal 分析建立一套形式化和可执行的“单位处理协议”。它的核心动作是：先把预训练 LM 看成定义在模型 alphabet $\Sigma$ 上的分布 $p_\Sigma$，再用一个 unit parser $\rho$ 把 $\Sigma$ 字符串映射到研究者想要的单位序列 $U^*$，最后在 $U$ 上重新定义 next-unit probability 和 surprisal。

### 整体框架
整体流程可以分成四步。

第一步，研究者先选择 unit inventory $U$。这个 $U$ 可以是 GPT-2 token、字符、空白切分词、PTB-style contextual words，也可以是更细的音素或更粗的 discourse units。重要的是，$U$ 是科学建模选择，不由 LM tokenizer 自动决定。

第二步，定义 unit parser $\rho: \Sigma^* \to U^*$。给定模型 alphabet 上的字符串，$\rho$ 负责输出单位序列。对中文分词这类可能歧义的情况，论文先给出 stochastic parser 的一般形式；主体中为了可计算性，假设 $\rho$ 是确定性的 total function。

第三步，把 $p_\Sigma$ 推到单位空间。理想情况下，单位序列 $\mathbf{u}$ 的概率是 $p_U(\mathbf{u}) = \sum_{\boldsymbol{\sigma} \in \rho^{-1}(\mathbf{u})} p_\Sigma(\boldsymbol{\sigma})$。这一步揭示了一个关键点：realization $\rho^{-1}$ 一般是 relation，不应该被强行当作 function，因为同一个单位可能有多种字符串实现。

第四步，用转导后的 LM 计算 next-unit surprisal。作者把每个单位先映射为底层字符串再加一个分隔符 sep，使 $h(u)=\xi\,\text{sep}$ 成为 prefix-free 编码；再令 $f=h\circ\rho$，用有限状态转导器把 $\Sigma^*$ 转成带 sep 的有限 alphabet $\Delta^*$。这样，单位概率可以通过 $\Delta$ 上的 prefix probability 比值恢复：$p_U(u\mid\mathbf{u}) = \overrightarrow{p}_\Delta(h(\mathbf{u})h(u)) / \overrightarrow{p}_\Delta(h(\mathbf{u}))$。

### 关键设计

**1. 把单位库存 $U$ 与模型 alphabet $\Sigma$ 解耦：让实验单位和模型 token 各归各位。**

以往很多工作默认 token 就是分析单位，或者用后处理把 token surprisal 汇总成 word surprisal，结果 BPE 的压缩规则、空白符归属、标点处理都直接渗进了心理语言学结论。本文的做法是让语言模型 $p_\Sigma$ 照旧在 $\Sigma^*$ 上给概率，但把 surprisal 的目标事件定义在 $U^*$ 上，二者之间由 unit parser $\rho$ 连接。这样词、字符、PTB token、morpheme 乃至 ROI 对齐单位都能成为合法的分析对象，而 tokenizer 退回到纯计算实现的角色，不再越权决定理论单位。

**2. 把 realization 视为 relation 而非 function：修复同一个词在不同上下文被迫拆成不同单位的毛病。**

已有方法常假设 $\rho^{-1}$ 是 monoid homomorphism，把 alphabet 分成 boundary symbols 与 continuation symbols，这会引出 unit inconsistency：句首的 `Hale` 可能实现为 `bosHale`，空白后的 `Hale` 实现为 `\u2423Hale`，一旦 realization 必须是函数，二者就被迫当成不同单位。本文允许同一单位与多个字符串实现相关联，于是 `Hale` 在哪都还是同一个 unit。背后的道理很直白：一个语言单位的身份不该由它处不处在句首、前面有没有空格来决定。relation 形式既贴近这一直觉，也才能一致地处理 `1,000`、`end, he said`、`don't`、`cat's` 这类上下文依赖的标点情形。

**3. 用 regular unit inventory 和有限状态转导实现可计算转换：把可能无限的词形集合压回有限 alphabet 上算。**

自然语言的词形集合可能无限，直接把每个词塞进 LM alphabet 并不现实，但许多音系、形态和 tokenization 规则其实都能用正则约束表达。于是本文假设单位本身是某个有限 alphabet $\Xi$ 上的 regular language（$U\subseteq\Xi^*$），给每个单位追加 sep 得到 prefix-free 的编码 $h(u)$，再用 finite transducer $f=h\circ\rho$ 把模型字符串转成 sep 标注的字符串。计算时直接复用已有的 transduced LM 算法，对所有映射到目标输出的源字符串概率做边际化，单位概率即可恢复。这个设计把理论上的自由度（任意合理单位）和工程上的可行性（有限状态机可算）接到了一起。

### 损失函数 / 训练策略
本文不训练新的神经语言模型。实验中的 source LM 是 GPT-2 Small，token inventory 直接读取 GPT-2 token surprisal；其他 inventory 通过相应 transducer 与 GPT-2 组合来估计 contextual surprisal。

在人类阅读时间建模上，论文使用 log-normal generalized additive mixed model (GAMM)。baseline model 只包含单位长度、unigram surprisal 及其前两位 spillover；target model 在 baseline 基础上增加 contextual surprisal 及其前两位 spillover。预测贡献用 held-out log-likelihood 改善 $\Delta_{\text{llh}}$ 衡量，正值表示 contextual surprisal 在控制长度和 unigram frequency 后仍能解释额外阅读时间方差。

Unigram surprisal 也从同一个 LM 分布估计，而不是调用外部词频资源。作者从 GPT-2 采样文本，把样本通过 transducer 转到目标单位空间，并在单位边界处估计 marginal next-unit probability；这样 frequency 控制变量与 contextual surprisal 来自同一个模型分布，避免 Speer 等词频资源把标点形式混在一起的问题。

## 实验关键数据

### 主实验
论文在 MECO English 眼动语料上评估四类单位库存：GPT-2 tokens、characters、acontextual words，以及 PTB-style contextual words。其中 acontextual words 又区分 leading whitespace 和 trailing whitespace 归属。数据来自 46 名读者阅读 12 段 Wikipedia 短文的 raw fixation；阅读时间指标包括 first fixation (FF)、gaze duration (GD) 和 total reading time (TRT)。评估采用按 trial 的 12-fold leave-one-out cross-validation，并用 1000 次 trial-level bootstrap 给出置信区间。

| 单位库存 | FF $\Delta_{\text{llh}}$ | GD $\Delta_{\text{llh}}$ | TRT $\Delta_{\text{llh}}$ | 结论 |
|--------|--------------------------|--------------------------|---------------------------|------|
| Characters | 0.11, p=0.145 | 0.09, p=0.185 | 0.10, p=0.171 | 三个指标均不显著 |
| GPT-2 tokens | 0.55*, p=0.013 | 1.52**, p<0.001 | 2.56**, p<0.001 | token surprisal 有稳定增益 |
| Acontextual words (leading) | 0.28, p=0.065 | 1.41**, p<0.001 | 3.00**, p<0.001 | 早期注视不显著，后期指标显著 |
| Acontextual words (trailing) | 0.63**, p=0.004 | 1.68**, p<0.001 | 2.91**, p<0.001 | 三个指标均显著 |
| Contextual words | 0.81**, p=0.003 | 2.13**, p<0.001 | 3.24**, p<0.001 | PTB-style 单位在三个指标上均显著 |

表中数值是 per-observation held-out log-likelihood improvement，单位为 $10^{-3}$ nats。作者特别强调：不同 inventory 的观测数量、单位粒度、长度控制和 unigram surprisal 都不同，所以这些绝对数值不能直接拿来做“哪个单位最好”的排序；它们主要说明在每个 inventory 内，contextual surprisal 是否相对 baseline 提供额外预测力。

### 消融实验
本文没有传统神经网络 ablation，最接近消融的是改变单位库存、空白归属和转导器复杂度，看同一个 GPT-2 Small 在不同单位定义下形成怎样的回归问题。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GPT-2 tokens | 2,478 units；40,589 GAMM observations | 使用模型原生 token，最贴近 LM 实现，但语言学解释较弱 |
| Acontextual leading | 2,095 units；39,290 observations；212.8 symbols/s | 空白归到后一个词，FST 快，FF 不显著 |
| Acontextual trailing | 2,095 units；39,767 observations；203.6 symbols/s | 空白归到前一个词，FF/GD/TRT 都显著 |
| Contextual words | 2,264 units；33,472 observations；12.0 symbols/s | PTB-style 规则更语言学化，但 FST 大很多，计算慢一个数量级以上 |
| Characters | 13,226 units；48,834 observations | 粒度最细，很多单字符不被单独注视，三个阅读时间指标均不显著 |

### 关键发现
- Word-like inventories 在 gaze duration 和 total reading time 上都显著提升 held-out log-likelihood，说明在控制长度、unigram surprisal 和 spillover 后，contextual surprisal 仍然能解释后期阅读时间。
- Character inventory 的 $\Delta_{\text{llh}}$ 很小且不显著，原因不一定是字符 surprisal 没有认知意义，而是眼动 fixation 很少天然落在单个字符单位上；当单位和观测 ROI 不匹配时，统计问题会变形。
- Leading/trailing whitespace 不是无关紧要的实现细节。Acontextual leading 在 first fixation 上不显著，而 trailing 显著，说明 delimiter 归属会改变单位长度、spilling controls 和 fixation attribution。
- Contextual words 的结果最符合“语言学单位”直觉，但计算开销也最大：contextual FST 的 contextual surprisal throughput 只有 12.0 symbols/s，而两个 acontextual FST 约 200 symbols/s。
- 论文的最重要实验结论不是“PTB token 一定最好”，而是“单位选择必须作为实验设计的一部分报告并解释”。不同单位会诱导不同数据表，跨单位比较绝不能只看 $\Delta_{\text{llh}}$ 大小。

## 亮点与洞察
- 论文把一个常被当作 preprocessing 的问题提升为理论建模问题：surprisal 的 $u_t$ 到底是什么。这个视角很有价值，因为许多 LLM 心理语言学实验的争议其实来自单位和 ROI 的隐式错配。
- “realization 是 relation”是一个小但很关键的形式化修正。它优雅解释了句首词、空白符、标点和数字内部符号为什么不能靠固定 boundary partition 解决。
- 用 sep 让 unit 编码 prefix-free 是整套方法可计算的关键。它把“一个单位结束了”这个事件显式纳入概率，而不是只把字符或 token surprisal 简单求和。
- 实验设计没有试图证明某个单位绝对更好，而是展示单位选择会改变观测、控制变量和显著性。这种谨慎解释比单纯刷指标更适合方法论论文。
- 这套框架可以迁移到很多 NLP/认知建模场景：比如 phoneme-level EEG/MEG surprisal、morpheme-level reading、sentence/discourse ROI、甚至不同 tokenization 下的 LM 可解释性分析。

## 局限与展望
- 实证范围较窄：实验只覆盖英语 MECO、GPT-2 Small 和 GAMM。不同语言的“word”概念差异很大，中文、日文、形态丰富语言或无空格文字系统可能需要完全不同的 parser 和 empirical validation。
- 当前框架假设 unit parser 确定且可由 rational / finite-state transducer 表示。真正歧义的分词、需要句法栈或上下文无关结构的单位转换，不在本文实现范围内。
- 计算成本仍然明显。尤其是 contextual words 和 unigram surprisal 估计，需要对许多源字符串边际化，实验中用 beam-search 近似和并行采样才能跑完。
- 依赖 raw fixation 数据。很多公开阅读时间语料已经按词聚合，无法重新分配 fixation 到新单位；self-paced reading 数据也天然绑定实验展示单位。
- ROI 部分更多是概念澄清，实验主要还是 word/character/token 粒度。未来可以直接验证 discourse units、clause-level ROI、parafoveal preview 字符窗口等更复杂 ROI。

## 相关工作与启发
- **vs Oh and Schuler (2024) / Pimentel and Meister (2024)**: 这些工作给出了从 token LM 计算 word-level surprisal 的 formalism，但依赖 whitespace/boundary partition 和函数式 realization。本文指出这会造成 unit inconsistency，并用 relation + transducer 框架统一 leading/trailing 与 contextual segmentation。
- **vs Nair and Resnik (2023) / Beinborn and Pinter (2023)**: 这些研究关注 subword tokenization 的认知合理性，常把模型 token 作为分析对象。本文并不否定 token-level 分析，而是强调 token 只是一个可选 inventory，适合研究模型自身，却不应默认代表人类处理单位。
- **vs Wilcox et al. (2023), Shain et al. (2024), Goodkind and Bicknell (2018)**: 这些工作验证 surprisal 对阅读时间的预测力。本文贡献不在于提出新的 linking function，而是在同一 linking pipeline 前加上单位选择和 ROI 兼容性这层理论约束。
- **vs Snæbjarnarson et al. (2026) / Vieira et al. (2025)**: 本文直接借用 transduced LM 和 token-to-character 转换工具，把它们用于心理语言学中的 unit-level surprisal。启发是：有限状态方法可以成为 LLM 与语言学单位之间的“概率接口”。
- **对后续研究的启发**: 做 surprisal 分析时，论文应显式报告 unit inventory、ROI 定义、空白/标点归属、unigram surprisal 来源和是否跨单位比较。否则同一个模型的结果可能因为预处理规则不同而不可复现。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 论文没有提出新 LM，但把 tokenizer/单位问题形式化为 surprisal theory 的核心建模选择，问题抓得很准。
- 实验充分度: ⭐⭐⭐⭐☆ 实验覆盖多个单位库存、三个眼动指标和严格交叉验证；不足是只在英语 MECO + GPT-2 Small 上验证。
- 写作质量: ⭐⭐⭐⭐⭐ 论证层次清楚，从 unit inconsistency 到 finite-state solution 再到眼动实验，概念推进很稳。
- 价值: ⭐⭐⭐⭐⭐ 对所有用 LLM surprisal 做心理语言学或认知建模的工作都有方法论提醒，尤其适合作为实验报告规范参考。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[ACL 2026\] Demystifying Data Organization for Enhanced LLM Training](demystifying_data_organization_for_enhanced_llm_training.md)
- [\[ACL 2026\] Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering](is_a_document_educational_or_just_wikipedia-style_--_pitfalls_of_classifier-base.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)

</div>

<!-- RELATED:END -->
