---
title: >-
  [论文解读] Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan
description: >-
  [ACL 2026][可解释性][中世纪奥克语] 针对中世纪奥克语这种低资源历史语言，作者搭了一套 mBERT + 混合分词 + 域适应 MLM 的可解释框架，把"原拉丁中性名词在奥克语里到底是男性还是女性"这个问题拆成词形线索 vs. 句法上下文两路证据来量化…
tags:
  - "ACL 2026"
  - "可解释性"
  - "中世纪奥克语"
  - "语法性别"
  - "拉丁中性词"
  - "可解释 NLP"
  - "混合分词"
---

# Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan

**会议**: ACL 2026  
**arXiv**: [2605.09156](https://arxiv.org/abs/2605.09156)  
**代码**: https://github.com/ahan-2000/Lost-in-Translation- (有)  
**领域**: 计算语言学 / 历史语言学 / 低资源 NLP  
**关键词**: 中世纪奥克语、语法性别、拉丁中性词、可解释 NLP、混合分词

## 一句话总结
针对中世纪奥克语这种低资源历史语言，作者搭了一套 mBERT + 混合分词 + 域适应 MLM 的可解释框架，把"原拉丁中性名词在奥克语里到底是男性还是女性"这个问题拆成词形线索 vs. 句法上下文两路证据来量化，发现后缀形态贡献最大单一信号、上下文（尤其冠词与形容词）能把宏 F1 从 0.665 推到 0.929。

## 研究背景与动机

**领域现状**：罗曼语族从拉丁三性（男 / 女 / 中）演化成两性（男 / 女）是历史语言学的经典问题，但绝大多数计算研究集中在法语、西语等高资源语言，奥克语虽是 UNESCO 列入"濒危"的浪漫语，相关 NLP 工作极其稀少。

**现有痛点**：① 中世纪奥克语正字法极度不稳定，同一词条往往有多种拼写，标准 WordPiece/BPE 分词器要么 OOV 率高、要么把有意义的形态线索拆碎；② 已有性别预测工作要么纯规则（不可迁移）、要么只看孤立词形（忽视一致性），没有量化"词形 vs. 上下文"各自贡献多少；③ 拉丁中性名词在奥克语里到底归男归女缺乏系统的可解释分析。

**核心矛盾**：性别信息其实分布在两个层面 —— 词内形态（后缀 `-um/-ia/-la`）和句子级一致性（冠词 `lo/la`、形容词词尾）—— 但现有方法没把两路证据分开看，导致既无法解释模型为何成功，也无法在词形模糊（如 `psalmista`）时知道上下文到底救了多少。

**本文目标**：(RQ1) 仅靠词级形态特征能预测多少奥克语性别？(RQ2) 加上句子上下文后增益多少、由哪些词性提供？

**切入角度**：把"性别分配"看成一个可量化的双源问题——词内信号 + 上下文信号，分别建模、对比，再用消融、SHAP、PoS occlusion 三种可解释工具把每一路证据拆开看。

**核心 idea**：用 mBERT + 混合分词（语料 BPE + word-level 回退）+ 域适应 MLM 作为统一骨干，分别构造 word-only / context / masked-context 三种输入，对比它们在性别预测上的 Macro-F1 与 log-prob 增量，从而量化形态与上下文各自的贡献。

## 方法详解

### 整体框架

整个 pipeline 由三段组成：

1. **预备阶段（嵌入与分词选择）**：在拉丁-奥克语对上用 FastText / mBERT / ByT5 跑三个 probe（冻结性别预测、Latin→Occitan 变体检索、聚类），mBERT 全胜，被选为骨干；再比较 mBERT WordPiece、纯 BPE 与 Hybrid（语料 BPE + word-level 回退），Hybrid 是唯一同时实现 0% OOV 且 masked recovery 达 25.23% 的方案；用 Hybrid 词表对 mBERT 做 10 epoch 域适应 MLM，验证 PPL 从 942.85 降到 9.52。
2. **词级性别预测（RQ1）**：从拉丁 / 奥克语词对里抽取初始子串、后缀 1-4gram、音节数 $S(w)$、CV 模板 $P(w)$、长度比、重音位置代理等形态-音系特征，再可选地拼接 FastText / mBERT / ByT5 的 mean-pool 表示，喂给一组 13 种分类器（LR / RF / XGB / FFN / BiLSTM / BiLSTM+Attn / 2×BiLSTM+MHSA）做 10 折 lemma-grouped CV。
3. **上下文性别预测（RQ2）**：从约 130k token 的未标注奥克语语料里把名词 token 对齐到拉丁-奥克语 lemma 词典（精确匹配优先，否则用 $\textsc{Sim}=0.3\,\textsc{CosSim}+0.7\,\textsc{LevSim}\ge 0.85$ 的模糊匹配），得到 (sentence, noun position, Latin lemma, Latin gender, gold Occitan gender) 五元组；然后用同一个 mBERT + MLP head，分别在 word-only / context（noun-conditioned attention）/ masked-context（把目标词替换为 `[MASK]`）三种表示上训练评测。

### 关键设计

1. **Hybrid 分词器（BPE + word-level 回退）**:

    - 功能：在中世纪奥克语这种高拼写噪声语料里同时保证零 OOV 与有意义的子词切分。
    - 核心思路：先在奥克语语料上训练一个小词表 BPE（按 $V_{t+1}=V_t\cup\{ab\},\ (a,b)=\arg\max_{(x,y)} f(x,y)$ 迭代合并最高频对），再加一条 word-level 回退规则——任何 BPE 切不开的整词原样保留。BPE 子词能捕捉到 `primpcipat` 里 `mp` 这种辅音簇变体、`secretament` 里词尾 `t`（对应 Old Occitan 副词 `-t` 的脱落）等历史变异规律。
    - 设计动机：标准 mBERT WordPiece 虽 0% OOV 但 masked recovery 只有 15.78%；纯 BPE（vocab=600/800）反而引入 2.63–2.86% OOV、recovery 仅 3–5%；Hybrid 兼得两者优点，是后续 mBERT 表现最好的关键基础。

2. **2×BiLSTM + MHSA 形态分类头**:

    - 功能：以孤立 lemma 的多源特征（拉丁/奥克语 n-gram + 句法-音系特征 + mBERT embedding）作输入，输出二分类性别概率。
    - 核心思路：双层 BiLSTM 捕捉子词序列的顺序依赖，再叠 8-head 自注意力让模型在序列中找到对性别最敏感的子词位置；训练目标是 label smoothing CE 或 focal loss + class weight 以抗 2:1 不平衡。在 mBERT embedding 上取得最佳 lemma 级 Macro-F1 = 0.8224。
    - 设计动机：树模型与浅 LSTM 都只能拿到 ~0.71–0.78 F1；性别信号既在后缀又可能在词中某些音位组合上，需要既能建模序列又能定位关键位置的结构。

3. **noun-conditioned attention 上下文头**:

    - 功能：在不破坏目标名词表示的前提下，让模型把整句一致性信息读进来。
    - 核心思路：用 mBERT 编码整句得到 $H=(h_1,\dots,h_T)$，把目标名词位置 $i$ 的隐状态 $h_i$ 作 query，全句作 key/value 做多头注意力 $\mathrm{Attn}(h_i, H, H)$，再与拉丁 lemma embedding $e(L)$、拉丁性别 one-hot $\mathrm{onehot}(G_L)$ 拼接送入共享 MLP $p(y\mid r)=\mathrm{softmax}(f_\phi(r))$。同时设计 masked-context 变体——把名词位替成 `[MASK]` 再读 $h_i^{\text{mask}}$，专门测"剥掉名词后单凭上下文能恢复多少性别"。
    - 设计动机：在奥克语里 `la torista` 的冠词 `la` 才是消歧关键；naive 用整句 `[CLS]` 池化会稀释信号，noun-conditioned attention 让模型显式定位"我现在要决定这个名词的性别"，并把消歧权重交给可解释的注意力分布。

### 损失函数 / 训练策略
- 词级实验用 lemma-grouped 10-fold CV 防变体泄漏，Optuna 贝叶斯优化超参，最佳头是 2×BiLSTM+MHSA，CE + label smoothing 0.1，训练 100 epoch；不平衡用 focal loss + class weights 处理。
- 上下文实验用 group K-fold（3 折，按 lemma 分组），AdamW，warmup 0.06 + 线性衰减，grad clip 0.5，dropout 0.1，固定随机种子 13。

## 实验关键数据

### 主实验

| 设置 (mBERT) | Accuracy | Macro F1 |
|---|---|---|
| Word-only | $0.808 \pm 0.154$ | $0.665 \pm 0.108$ |
| Context model (noun attention) | $0.979 \pm 0.012$ | $0.929 \pm 0.034$ |
| Context model (noun masked) | $0.977 \pm 0.008$ | $0.902 \pm 0.097$ |

加上上下文后 Macro-F1 从 0.665 跳到 0.929（+26.4），masked-context 仍达 0.902，说明上下文里的一致性信号本身就能恢复绝大多数性别。lemma 级最佳为 mBERT + 2×BiLSTM+MHSA，Macro-F1 = $0.8224 \pm 0.0385$，配对 bootstrap 显示对 ByT5 优势 $\Delta = +0.0395$，95% CI $[+0.0250, +0.0543]$，$p<10^{-6}$。

### 消融实验

| 移除特征块 | F1 (mBERT) | $\Delta$ | % drop |
|---|---|---|---|
| Latin n-grams | 0.8092 | 0.0132 | 1.61% |
| Meta-features | 0.8168 | 0.0056 | 0.68% |
| Occitan n-grams | 0.8169 | 0.0055 | 0.67% |
| Syllable counts | 0.8194 | 0.0030 | 0.37% |
| VC patterns | 0.8220 | 0.0004 | 0.05% |
| Stress patterns | 0.8239 | -0.0015 | -0.18% |

附加 PoS-conditioned occlusion 实验：NOUN 的 mean $\Delta=+0.0026$、DET $+0.0010$、ADJ $+0.0003$（均 $p<10^{-4}$ 显著为正），CCONJ/ADP/VERB 反而是负贡献，PUNCT/PRON 不显著，定量证实"性别信息主要靠名词 + 冠词 + 形容词三类一致性载体"。

### 关键发现
- 后缀形态是最强单源信号：去掉拉丁 n-gram 在三种 embedding 上都掉 1.6–1.8 个 F1 点，远高于 VC 模板或重音代理；重音代理甚至轻微伤害模型（去掉后 F1 反升 0.001-0.002），暗示作者的启发式重音规则有噪声。
- 上下文带来巨大增量：context vs. word-only 的 $\Delta_1^{\text{prob}}=+0.283$、$\Delta_2^{\log p}=+0.340$（95% CI 均严格大于 0），说明上下文不仅提升准确率，还稳定推高了对真实类别的置信度。
- 拉丁元信息是上下文的"放大器"：去掉拉丁 lemma + 拉丁性别后，masked-context 增益从 ~0.28 跌到 ~0.09–0.11（约 3× 缩水），说明跨语对齐特征和上下文是互补放大关系而非冗余。

## 亮点与洞察
- 把"形态 vs. 上下文"做成可对比的三档输入（word-only / context / masked-context），masked-context 设计尤为精巧——它强迫模型在不看名词本身的前提下从句法一致性恢复性别，从而干净地剥离两路信号的贡献。
- Hybrid 分词器同时拿到 0% OOV 和 25.23% masked recovery，反衬出"低资源历史语料里 OOV 与子词质量不是非此即彼"，这个组合策略可以直接迁移到拉丁、古希腊、古典阿拉伯等存在拼写不稳定的语言。
- PoS-conditioned occlusion + 符号翻转置换检验给出了每种词性对性别预测的统计学显著贡献，这个轻量级 attribution 工具值得迁移到任何"一致性驱动"的语言学任务（如格、数、人称预测）。

## 局限与展望
- 语料体量小（~130k token）且 2:1 不平衡，对女性类的泛化能力受限；focal loss + class weight 只能缓解不能根治。
- 关键超参用启发式定：模糊匹配阈值 $\tau=0.85$、重音代理规则、$\alpha=0.3$ 等都缺乏系统调参；消融已显示重音代理引入轻微噪声。
- PoS occlusion 依赖自动标注，但 PoS tagger 整体 71% 准确率（NOUN 70%、ADJ 最低），attribution 结果会被打标错误污染。
- 句首/句尾名词以及一致性词稀疏的句子表现明显差，错误分析显示这是最大误差源；后续可加 boundary-aware 注意力 mask 或额外句法监督。
- 整套结论只在"拉丁中性 → 奥克语男 / 女"这一历史转折上验证，能否推广到其他罗曼语（如普罗旺斯方言、加泰罗尼亚语）尚未测试；也未直接回答"历史上为什么发生这个重新分配"，需要 diachronic 平行数据。

## 相关工作与启发
- **vs Williams et al. (2019) 信息论量化德语/捷克语性别**：他们也把性别拆成 form/meaning/inflection 三源做信息论分解，本文则面向低资源历史语料、加入 BERT-style 上下文 attribution；本文优势是显式量化 word vs. context 增量并提供 PoS 级解释，劣势是没有形式化的 mutual information 估计。
- **vs Cucerzan & Yarowsky (2003) 最小监督上下文性别归纳**：他们用 co-occurrence with gendered articles 推未见词性别，本文是 neural 版本（mBERT + attention），并提供了可量化的 masked vs. unmasked 对比；启发是"冠词/形容词作为弱标签"这条思路在历史语料里依然成立、效果还很强。
- **vs 法语规则式性别预测（Lyster 2006）**：规则方法可解释但完全语言相关，本文的混合分词 + 域适应 mBERT 范式更具迁移性，且通过 SHAP/occlusion 提供与规则等价的"哪个后缀最重要"的可视化。

## 评分
- 新颖性: ⭐⭐⭐ 没有炫技模型，但"用 NLP 量化历史语言学问题 + 可解释双源对比"对中世纪奥克语属空白填补
- 实验充分度: ⭐⭐⭐⭐ 嵌入 / 分词 / 特征三层消融 + paired bootstrap + 符号翻转检验 + SHAP + occlusion 一应俱全
- 写作质量: ⭐⭐⭐⭐ 研究问题清晰、结论与数据严格对齐、附录完整
- 价值: ⭐⭐⭐ 对历史语言学和低资源 NLP 双方向都有借鉴；模型本身不算先进但方法学框架可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ACL 2025\] CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](../../ACL2025/interpretability/cleme2_gec_evaluation.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](../../ICML2026/interpretability/optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](../../NeurIPS2025/interpretability/cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)

</div>

<!-- RELATED:END -->
