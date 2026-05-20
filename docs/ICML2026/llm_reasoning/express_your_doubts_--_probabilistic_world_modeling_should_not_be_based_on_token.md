---
title: >-
  [论文解读] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs
description: >-
  [ICML 2026 (Position Paper)][LLM推理][token logprob] 这是一篇 position paper，主张：**用 LLM 的 token softmax 概率（logprob）当成"世界事件概率"是理论上错的**——因为 distribution estimation…
tags:
  - "ICML 2026 (Position Paper)"
  - "LLM推理"
  - "token logprob"
  - "世界建模"
  - "二阶预测"
  - "calibration"
  - "surface-form competition"
---

# Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs

**会议**: ICML 2026 (Position Paper)  
**arXiv**: [2505.02072](https://arxiv.org/abs/2505.02072)  
**代码**: 无（position paper）  
**领域**: NLP / LLM 概率语义 / 校准  
**关键词**: token logprob、世界建模、二阶预测、calibration、surface-form competition

## 一句话总结
这是一篇 position paper，主张：**用 LLM 的 token softmax 概率（logprob）当成"世界事件概率"是理论上错的**——因为 distribution estimation、response prediction 和 target distribution estimation 是三个不同任务，对应不同 ideal 输出分布；获取世界概率的正确做法是**二阶预测**——让 LLM 在输出里**显式写出**它对事件的概率（数值或语言修饰词），而不是去算"它说 X 的概率"。

## 研究背景与动机

**领域现状**：早期语言模型（Shannon、Bahl）定义就是"字符串上的分布"，logprob 就是这个分布。但现代 LLM 已经发生两个根本转变：(1) **任务从 distribution estimation 转向 response prediction**（输出"正确回答"而不是"模仿数据分布"）；(2) **学习目标从词到世界**（事实/事件而非语言模式）。可大量后续工作还在用 token softmax 当"模型置信度"或"事件概率"。

**现有痛点**：把 softmax logprob 当成事件概率会得到系统性错误的结论。比如 Yona et al. 2024 测 LLM 是否"诚实表达不确定性"，把"生成某答案的概率"当成"置信度"——发现 gap 巨大就归咎 LLM "不诚实"；Hu & Levy 2023 比较 prompt (a) "The keys to the cabinet" 和 prompt (b) "What word is most likely to follow?" 的 next-word 概率，认为理想 LLM 应当给出相同分布——但其实一个是 completion（distribution estimation）、一个是 instruction-following（prediction），目标分布本就该不同。这些"calibration failure"很多是**category error**，不是模型缺陷。

**核心矛盾**：language modeling 的传统定义（数据分布）和 prediction（最大化准确率）会产生**矛盾的最优分布**。极端例子：一个有偏硬币 $p(H)>0.5$，传统 LM 应该按比例输出 H/T；但用作 predictor 应该 always 输出 H（这样 accuracy 最高，分布完全坍缩到众数）。如果要事件概率 $p(H)$，两者都不对——前者带 reporting bias，后者只剩 mode。

**本文目标**：(1) 形式化区分三个任务；(2) 把 LLM 训练（pretrain / SFT / RLHF）和推理（naïve completion / zero-shot / few-shot / 二阶预测）映射到三个任务；(3) 用 Bayesian agent 框架解释为什么不同 prompt 形式应产生不同 ideal 输出分布；(4) 系统盘点文献里的概念混淆，给出可行的改进方向。

**切入角度**：作者用 belief-desire-intention (BDI) agent 框架——文本由一个"持有信念 $b$、带有欲望 $d$（一个从 belief 到字符串的函数）"的 agent 生成；prompt $\mathbf{w^x}$ 设置 ground，决定了 agent 的 belief 和 desire 分布。换 prompt 等价于换 agent 的状态，所以理想输出分布也跟着换。

**核心 idea**：**二阶预测**——既然要事件概率 $p(y|x)$，就让 LLM 直接把概率作为输出的一部分写出来（如 "我 70% 确定 X"），而不是去 softmax token "X"；这样 unbiased belief 加上"如实报告概率"就能让输出分布与世界分布一致。

## 方法详解

### 整体框架
论文不提算法，而是建一个**任务-训练-推理-用例**四层映射的概念框架：

- **任务层**：source distribution estimation / target distribution estimation / response prediction。
- **训练层**：T1 pretrain（distribution estimation on $p_{LM}$）、T2a SFT（指令-回应配对，目标分布 $p_{\text{res}}$）、T2b RLHF（最大化奖励，分布往众数靠）。
- **推理层**：I1 softmax probabilities（细分为 naïve description $p_{\text{ND}}$、zero-shot $p_{\text{ZS}}$、few-shot $p_{\text{FS}}$）；I2 二阶预测（让模型显式输出概率）。
- **用例层**：word completion（要 source distribution）、world responses（要 prediction）、world modeling（要 target distribution = 事件分布）。

主要论证靠"用 Bayesian agent 推三个 case"。

### 关键设计

1. **三任务的形式化区分**：

    - 功能：用统一符号把"估计分布"和"预测响应"分开，避免文献里互换使用。
    - 核心思路：distribution estimator $p_\theta:\mathcal{X}\to\Delta^{|\mathcal{Y}|-1}$ 学习 $p_S(y|x)$；predictor $f_\theta:\mathcal{X}\to\mathcal{Y}$ 学习单一最优响应；target distribution estimation 是估计另一个分布 $p_T\ne p_S$（带 transfer learning 性质）。当输出空间是 $\Delta^{|\mathcal{Y}|-1}$（即"概率分布本身"）时，target estimation 退化成在"分布空间"上的预测——作者称之为 **second-order prediction**。
    - 设计动机：很多文献的混淆根源就是没区分 $p_\theta$ 和 $f_\theta$，比如把 SFT 后的 LLM 当 distribution estimator 用其 token logprob，但它已经被训成 predictor（mode-seeking）了。

2. **BDI 三案例分析**：

    - 功能：解释相同事件 $y$ 在不同 prompt 下的 ideal LM 概率为何不同。
    - 核心思路：
        - **Case 1（多种表达同一结果）**：prompt 无指令、agent 可任意措辞，输出概率反映**语言使用频率**，与事件分布的关系取决于 reporting bias。
        - **Case 2（agent 观察到结果）**：prompt 含指令、agent 看到事件，理想下 $p_{LM}(\mathbf{w^y}|\mathbf{w^x})=p_E(y|x)$，但前提是 agent always 如实报告——实际不一定（如默认色 "banana is yellow"）。
        - **Case 3（agent 未观察、需预测）**：写为 $p(\mathbf{w^y}|\mathbf{w^x})=\sum_{f\in D^y}p(f|\mathbf{w^x})\cdot p(f(b^y)=y)$；要让它等于 $p(y|x)$ 必须同时满足"信念正确" + "如实报告"，但 prediction 任务下 agent 应把质量全压在 mode，所以正确 belief + accuracy-optimizing 不能共存。
    - 设计动机：用 agent 抽象统一不同推理设置——pretrain 对应 Case 1（数据频率）、SFT/RLHF 对应 Case 3（预测最优响应），同一 LLM 在不同 prompt 下扮演不同 agent，自然给出不同分布。

3. **二阶预测作为 escape hatch**：

    - 功能：在 Case 3 下打破 "prediction vs 概率" 的死锁。
    - 核心思路：当 prompt 明确要求 agent **报告概率本身**（而非选项），agent 的"理性选择"恰好就是把 belief 中的概率写出来——因为"输出 belief 中的概率值"既是 truthful 也最大化 brier-like 评分；所以输出分布与世界分布对齐变得可能。具体做法：在 prompt 里说 "输出格式：A: 概率p, B: 概率q, ..." 或允许 verbal hedging（"likely a but maybe b"），由外部 parser 验证或读语义。
    - 设计动机：softmax logprob 受 surface-form competition（"heads" vs "the coin landed on heads" 抢概率）和 tokenizer artefact 影响；二阶预测把"事件"直接搬到输出空间，绕开词面竞争，且天然支持任意粒度（数字 / 模糊词 / 区间）。

### 损失函数 / 训练策略
本文是 position paper，**不提任何新训练目标**。但给出方向建议：未来如果要让二阶预测真正可靠，需要 (a) 训练数据里包含"事件 + 显式概率标注"的语料；(b) 用 calibration-aware 评分（Brier / log score）做 fine-tune，而不是 cross-entropy on token；(c) 区分 "what to say" 和 "how confident"，可能要双 head 结构。

## 实验关键数据

本文**无实验**，靠概念分析。但作者用文献复现的反例支撑论点：

| 文献 | 实验结论 | 作者重新解读 |
|------|---------|-------------|
| Hu & Levy 2023 | prompt (a) 和 prompt (b) 的 next-word 概率不一致 → LLM 不会 meta-linguistic 任务 | 两个 prompt 触发不同任务（estimation vs prediction），分布本就不该一致；不是 LLM 缺陷 |
| Yona et al. 2024 | LLM 生成"自信"答案的概率不反映正确率 → unfaithful hedging | 生成概率 ≠ 置信度；category error |
| Liu et al. 2023 | 内部 probing 概率与 output 概率有 gap | 同上，response prediction 必然 mode-seeking |
| Gupta et al. 2025 | LLM 即使被告知公平硬币也偏向 heads | 验证 case 1 的 reporting bias 真实存在 |
| Paik et al. 2021 | LLM 把香蕉描述成绿色多于黄色 | 也是 reporting bias（黄香蕉默认不提颜色） |

这些"calibration failure"在传统视角下都是 LLM 的缺陷；在本文框架下大多是研究者用错了概率。

### 关键发现
- **"calibration"在 instruction-tuned LM 上几乎必然失败**，因为目标分布就是 mode-seeking 的，与 "match 正确率" 这一校准定义在数学上不兼容。
- **二阶预测在简单设置下可工作**（如"硬币偏向 0.7"，让 LLM 显式说"我估 70%"），但在大输出空间（千类、连续）上语义解析很难自动化。
- **surface-form competition** 是一个被严重低估的 confound——LLM 给 "heads" 0.3、给 "the coin landed on heads" 0.2，加起来才是真正的事件概率，但研究者常只算前者。

## 亮点与洞察
- **三任务的形式化区分** 是这篇文章最有价值的概念贡献：一旦想清楚"我要的是 source 频率、prediction mode、还是 event 概率"，很多 calibration 论文的设计选择就一目了然。
- **BDI 框架统一了不同 prompt 形式** 这个抽象很优雅——把 "prompt 改写产生不同分布" 这一现象，归结为 "ground 改变改了 agent 的 belief/desire"，比单纯说 "prompt 影响 prior" 更深刻。
- **二阶预测**不是新概念（probabilistic supervised learning 早就有），但作者把它放到 LLM 语境下重新激活，给"如何让 LLM 输出事件概率"这一目标提供了一条理论合理的路。

## 局限与展望
- 论文几乎纯概念，**没有任何对比实验**证明二阶预测在实际任务上确实优于 softmax logprob；这削弱了说服力，至少应该在一两个 toy benchmark 上演示。
- 二阶预测的实际执行非常困难：(a) LLM 是否真有 well-calibrated 的 belief？现有证据并不支持；(b) 输出格式（数字 / 自然语言）的解析和评估如何标准化？论文没给方案。
- 对 RLHF 训练目标的批评偏定性，没分析 KL 正则项如何具体影响输出分布的尖锐度。
- 文章一直说 token logprob 不行，但忽略了一些实际场景（如句子级生成似然用于 ranking、检索增强中的文档打分），这些场景里 logprob 仍然有用，没必要全盘否定。

## 相关工作与启发
- **vs Holtzman 2021 (surface-form competition)**：Holtzman 指出 logprob 受表达方式影响、提出 PMI 修正；本文把它纳入更大的"任务错配"框架，认为 PMI 只是局部补丁，二阶预测才是根本方案。
- **vs Kadavath 2022 (Language Models Mostly Know What They Know)**：Kadavath 用 "P(True)" 让 LLM 自评——这其实就是二阶预测的雏形，本文给了它一个理论合法的位置。
- **vs Farquhar 2024 (semantic uncertainty)**：Farquhar 试图把"相同语义不同表面"的概率合起来；本文论证这种合并仍治标，根本问题是 task type 不对，要直接问概率。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三任务区分 + BDI 框架对 LLM 概率语义的整理很有原创性
- 实验充分度: ⭐⭐ position paper，零实验；说服力靠论证质量，不靠数据
- 写作质量: ⭐⭐⭐⭐ 形式化干净，三个 case 推导清晰，文献盘点全面
- 价值: ⭐⭐⭐⭐ 对所有做 LLM calibration / probing / hallucination 研究的人都是必读的"地基"清算

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)
- [\[ICLR 2026\] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](../../ICLR2026/llm_reasoning/cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[NeurIPS 2025\] Reasoning Models Better Express Their Confidence](../../NeurIPS2025/llm_reasoning/reasoning_models_better_express_their_confidence.md)

</div>

<!-- RELATED:END -->
