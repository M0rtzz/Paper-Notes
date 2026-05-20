---
title: >-
  [论文解读] Optimizing Language Models for Crosslingual Knowledge Consistency
description: >-
  [ICML 2026][多语言/翻译][跨语言一致性] 本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**…
tags:
  - "ICML 2026"
  - "多语言/翻译"
  - "跨语言一致性"
  - "DCO"
  - "DPO 变体"
  - "product of experts"
  - "RankC"
---

# Optimizing Language Models for Crosslingual Knowledge Consistency

**会议**: ICML 2026  
**arXiv**: [2603.04678](https://arxiv.org/abs/2603.04678)  
**代码**: [github.com/Betswish/ConsistencyRL](https://github.com/Betswish/ConsistencyRL)  
**领域**: 强化学习 / 多语言 LLM / 偏好对齐  
**关键词**: 跨语言一致性, DCO, DPO 变体, product of experts, RankC

## 一句话总结
本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**，证明其最优策略呈 product-of-experts 形式并在 $\gamma_1\gamma_2=\beta^2$ 时保证跨语言偏好一致；据此推导出无需 reward model、无需 online 采样的 **DCO（Direct Consistency Optimization）** 算法，在 9 个 LLM、3 个多语言 QA 基准、26 种语言上同时提升跨语言一致性（RankC）与回答准确率。

## 研究背景与动机

**领域现状**：现代 LLM（Llama、Qwen、Aya、Gemma 等）号称多语言，但**对同一个问题用不同语言提问会得到冲突答案**——qi-etal-2023-cross 提出 RankC 指标后，跨语言知识不一致（Crosslingual Consistency, CLC）成为多语言 LLM 评估的标配。

**现有痛点**：(1) interpretability-based 干预方法（向量编辑、表征对齐）只在小数据/特定模型上验证，难规模化。(2) wang-etal-2025 的 CALM 把多语言 majority voting 选出的"赢家"做 DPO，但需要 >2 种语言，bilingual 场景失效；且加入低资源语言后 majority voting 反而失真。(3) 没有理论上保证"最优策略一定一致"的目标函数。

**核心矛盾**：DPO 的 Bradley-Terry 偏好建模天然是"单一语言下 winner vs loser"，**没有一种直接的方式表达"跨语言下 winner/loser 排序应当一致"这种二阶约束**；强行用偏好对（pair）来推也容易破坏在 post-train 语言上的原始准确率。

**本文目标**：1) 给出 CLC 的形式化定义（preference rank 在不同语言间不变）；2) 设计一个能直接驱动 RL 收敛到 consistent 策略的 reward；3) 推导无需 online sampling 与 reward model 的高效算法；4) 在多模型/多基准上验证。

**切入角度**：作者跳出"找一个跨语言 winner"的范式，转而用"语言 A 中回答的 reward = 把该回答翻译到语言 B 后在原模型下的 log-likelihood"——这种"用 partner 语言的似然给本语言打分"的设计，**让最优策略的对偶形式天然包含跨语言对称性**。

**核心 idea**：通过定义 reward $r_{\text{align}}$ 用对方语言的 $\log\pi_{\text{ref}}(\tau(\mathbf y)|\tau(\mathbf x))$，KL-正则 RL 的最优策略就是一个**跨语言 product of experts**；只要满足超参约束 $\gamma_1\gamma_2=\beta^2$，最优策略必然在两种语言间保持偏好排序一致，且能用 DPO 风格无 reward model 求解。

## 方法详解

### 整体框架
方法三层叠加：(1) **CLC 形式化（Def 1）**：模型 $\pi^\star$ 在 $L_1, L_2$ 一致 ⟺ 对任意翻译等价的回答对 $(\mathbf y_w^1, \mathbf y_l^1) \sim (\mathbf y_w^2, \mathbf y_l^2)$，偏好顺序在两语言间一致。(2) **结构化 reward 与最优策略**：定义 piecewise reward Eq.8，求解 KL-正则 RL，得到 product-of-experts 形式的最优策略 Eq.9，证明 $\gamma_1\gamma_2 = \beta^2$ 是一致性的充分条件（Lemma 1）。(3) **DCO 算法**：把 reward 匹配 (Eq.10) 转写为 DPO 风格的差分目标，避免 reward model 和 online sampling，用平行 prompt/response 数据集 $\mathcal D_\|$ 直接训练。

### 关键设计

1. **结构化 reward 与跨语言对偶 (Eqs. 7-9)**:

    - 功能：用一个 reward 表达"我希望本语言对答案的偏好和对方语言保持一致"。
    - 核心思路：定义 piecewise reward
      $r_{\text{align}}(\mathbf x, \mathbf y) = \gamma_i \log\pi_{\text{ref}}(\tau^j(\mathbf y)|\tau^j(\mathbf x))$ 当 $\mathbf x, \mathbf y \in L_i$（$j\ne i$）。
      根据 Rafailov 的 KL-正则 RL 最优策略公式，得到
      $\pi^\star(\mathbf y^1|\mathbf x^1) \propto \pi_{\text{ref}}(\mathbf y^1|\mathbf x^1) \cdot \pi_{\text{ref}}^{\gamma_1/\beta}(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$，这是一个**product of experts**——本语言原始 likelihood 与对方语言翻译版 likelihood 相乘。
      根据 rearrangement inequality，最大化 reward 等价于让 $\{\pi_\theta(\mathbf y|\mathbf x)\}_y$ 与 $\{r_{\text{align}}(\mathbf x, \mathbf y)\}_y$ 单调对齐，恰好对应 Def 1 的一致性。
    - 设计动机：作者要的是一个**形式化保证**："这个 reward 的最优解一定一致"，而不是经验上 plausible 的启发式。Product-of-experts 既保留 base model 知识（避免性能崩），又强行注入跨语言对偶约束。

2. **超参约束 $\gamma_1\gamma_2 = \beta^2$ 与 NN 语言推广 (Lemma 1)**:

    - 功能：在多种 $\beta, \gamma_1, \gamma_2$ 组合中选出唯一能保证一致性的子集。
    - 核心思路：把 Eq.9a 两边取 $\beta/\gamma_1$ 次方，可改写成 $(\pi^\star(\mathbf y^1|\mathbf x^1))^{\beta/\gamma_1} \propto \pi^\star(\tau^2(\mathbf y^1)|\tau^2(\mathbf x^1))$，由 $x \mapsto cx^{\beta/\gamma_1}$ 是 monotone 增函数得到偏好排序在两语言下一致。$\gamma_1, \gamma_2$ 分别控制对应语言对 $\pi_{\text{ref}}$ 的偏离强度（小 $\gamma$ = 更靠近原模型），$\beta$ 控制整体 KL 偏离。NN 语言时引入 $N^2 - N$ 个 $\gamma_{ij}$ 控制两两对齐强度，给出一致性需满足的对应约束（详见附录 E）。
    - 设计动机：实践中需要能"调"哪个语言被对齐得更紧（如低资源语言可能希望接近原模型避免被高资源语言"拉跑"），$\gamma_{ij}$ 这套设计给出可控旋钮。同时 $\gamma_1\gamma_2=\beta^2$ 约束让算法实现简单（只需选一组合法值，比如 $\gamma_1=\gamma_2=\beta$）。

3. **DCO 算法：免 reward model、免 online sampling (Eq. 10)**:

    - 功能：把上面的 RL 目标变成可以直接对模型参数 $\theta$ 做梯度下降的离线目标。
    - 核心思路：仿照 DPO，用 $\hat r_\theta(\mathbf x, \mathbf y) = \beta\log\frac{\pi_\theta(\mathbf y|\mathbf x)}{\pi_{\text{ref}}(\mathbf y|\mathbf x)}$ 重参数化 reward，让 $\hat r_\theta$ 的差分匹配 $r_{\text{align}}$ 的差分：
      $L(\theta) = \mathbb E\big[\|d_\theta^1 - \gamma_1\log\frac{\pi_{\text{ref}}(\mathbf y_w^2|\mathbf x^2)}{\pi_{\text{ref}}(\mathbf y_l^2|\mathbf x^2)}\| + \|d_\theta^2 - \gamma_2\log\frac{\pi_{\text{ref}}(\mathbf y_w^1|\mathbf x^1)}{\pi_{\text{ref}}(\mathbf y_l^1|\mathbf x^1)}\|\big]$，
      其中 $d_\theta^i = \hat r_\theta(\mathbf x^i, \mathbf y_w^i) - \hat r_\theta(\mathbf x^i, \mathbf y_l^i)$；这样：(a) winner/loser 标签**不需要 ground truth**，随机配对就行；(b) 不用真训 reward model；(c) 完全离线，只跑平行 prompt-response 数据 $\mathcal D_\|$，每个样本是 $(\mathbf x^1, \mathbf y^1, \mathbf x^2, \mathbf y^2)$ 翻译对。Lemma 2 证明最优 $\hat r_\theta^\star$ 收敛到 $r_{\text{align}}$ 加一个与 $\mathbf y$ 无关的常数 $c(\mathbf x)$（对策略不构成影响）。
    - 设计动机：和 DPO 一样用"差分形式消掉 partition function $Z(\mathbf x)$"的把戏，但目标从"匹配人类偏好"换成"匹配跨语言一致 reward"；这让训练 pipeline 与现有 DPO 框架完全兼容，工程落地极轻量。

### 损失函数 / 训练策略
9 个 LLM（Qwen2.5-7B/14B、Qwen3-8B/14B、Aya-Expanse-8B、Llama3.1-8B、Llama3.2-3B、Gemma3-4B/12B），3 个平行 QA 数据集（MMMLU 14 语言、XCSQA 16 语言、BMLAMA 17 语言），共 26 种语言。使用 DCO loss（Eq. 10），平行 prompt-response 配对训练。

## 实验关键数据

### 主实验
MMMLU 多语言联合训练（clc_all = 全语言对平均 RankC；a_en / a_¬en = 英/非英准确率），相对 base 模型增量：

| 模型 | 方法 | $\Delta$clc_all | $\Delta$a_en | $\Delta$a_¬en |
|------|------|-----------------|--------------|----------------|
| Qwen2.5-14B | Base = 68.6 / 72.5 / 58.1 | — | — | — |
| Qwen2.5-14B | + SFT* | +0.6 | +1.5 | +6.7 |
| Qwen3-14B | + SFT* | -0.2 | +0.1 | +0.5 |
| Aya-Expanse-8B | + SFT* | +3.5 | +0.7 | — |
| Llama3.1-8B | + SFT* | — | — | — |

(论文 Table 1 完整版还含 +DPO、+CALM、**+DCO** 行——DCO 在 clc_all 上一致优于其他方法，准确率不降甚至略升；具体数值见原文 Table 1。)

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| DCO vs SFT* | DCO 在 RankC 上显著高 | SFT 只优化 gold answer，不解决跨语言排序问题 |
| DCO vs DPO* | DCO 跨语言一致性更高，且不依赖 gold label | DCO 用平行对而非偏好对 |
| DCO vs CALM | CALM 在低资源语言加入后退化，DCO 稳定 | majority voting 不是关键 |
| DCO + DPO 组合 | 当 gold label 可用时，DCO 与 DPO 互补 | 不同目标解决不同子问题 |
| Bilingual 训练 | DCO 同样有效，CALM 失效 | DCO 不需 ≥3 种语言 |
| OOD generalization | 在未见 domain 上仍提升 RankC | 学到的是一致性结构，不是特定知识 |
| $\gamma_1 \ne \gamma_2$ 控制语言偏向 | 可定向偏向某语言保持原性能 | 工程可控 |

### 关键发现
- **DCO 提升一致性时不破坏单语言准确率**——这是相比 DPO 最关键的优势：DPO 容易在 post-train 语言上为了对齐偏好牺牲性能。
- $\gamma_1/\gamma_2$ 的不对称设置可实现"方向性对齐"：让高资源语言更紧地拉低资源语言的同时，保持高资源语言原始性能。
- 跨域（cross-domain）泛化好：在 MMMLU 上训的一致性模式能迁移到 XCSQA、BMLAMA。

## 亮点与洞察
- 把 CLC 一致性转写为 "另一语言的 likelihood 当 reward"，得到的 product-of-experts 形式既有数学美感（rearrangement inequality 直接给一致性证明）又有工程优势（与 DPO pipeline 兼容）。
- **不需要 gold label 也能训**：随机配对 winner/loser 仍然有效，因为差分形式只关心"两个回答 reward 差异是否跨语言一致"，这把数据需求降到只要平行翻译。
- $\gamma_1\gamma_2 = \beta^2$ 这种 elegant 的代数约束把"哪些超参组合保证一致"的迷雾清扫了，是少见的"理论给出 hyperparameter 选取指南"的工作。

## 局限与展望
- 评测依赖**翻译映射 $\tau$ 的存在**——文中限定在 factual QA 这种"答案有限且客观可翻译"的场景；对开放生成（创作、摘要）等没有明确答案空间的任务，"一致性"定义本身就模糊，DCO 不直接适用。
- 平行数据集（MMMLU/XCSQA/BMLAMA）的翻译质量影响训练；低资源语言翻译噪声大可能让 reward 失真。
- Lemma 1 的 $\gamma_1\gamma_2=\beta^2$ 是充分条件而非必要条件，是否存在更宽松的合法区域未探究。
- 与 chain-of-thought 推理交互如何，未讨论；CoT 中跨语言一致性更难因为中间过程也要对齐。
- 计算开销：每个样本需要在两种语言下分别 forward，比单语 DPO 翻倍。

## 相关工作与启发
- **vs DPO (Rafailov et al. 2023)**：DCO 把 DPO 的"匹配人类偏好"换成"匹配跨语言一致 reward"，沿用差分技巧消 partition function，但目标完全不同——一个是 align preferences，一个是 align across languages。
- **vs CALM (wang-etal-2025)**：CALM 需 ≥3 种语言做 majority voting 找"winner"再 DPO，DCO 用平行对直接训，bilingual 也行，低资源加入也不退化。
- **vs 表征干预方法 (Lu, Wang, Liu)**：DCO 不需要白盒访问 hidden states，纯用 likelihood 信号，规模化容易。
- **vs RankC 评估 (qi-etal-2023-cross)**：本文是首批"把 RankC 当 RL 训练目标"而非只评估的工作之一。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把跨语言一致性转成 RL reward 并推出 DPO 风格离线算法是真正的新构造。
- 实验充分度: ⭐⭐⭐⭐ 9 模型 × 3 数据集 × 26 语言覆盖广，OOD/bilingual/控制实验都有。
- 写作质量: ⭐⭐⭐⭐ 推导严谨，Lemma 1/2 给出完整证明，notation 复杂但逻辑清晰。
- 价值: ⭐⭐⭐⭐ 对多语言 LLM 部署有直接价值，且可与 DPO 互补。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](../../ACL2026/multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](../../ACL2026/multilingual_mt/language_models_entangle_language_and_culture.md)
- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](../../ACL2026/multilingual_mt/llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](../../ACL2026/multilingual_mt/multilingual_language_models_encode_script_over_linguistic_structure.md)

</div>

<!-- RELATED:END -->
