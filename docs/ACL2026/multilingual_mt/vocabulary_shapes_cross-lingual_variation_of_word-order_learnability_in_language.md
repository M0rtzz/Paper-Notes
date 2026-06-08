---
title: >-
  [论文解读] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models
description: >-
  [ACL2026][多语言/翻译][词序可学习性] 本文用 Mallows 模型生成 10 种欧洲语言的连续词序扰动谱，训练小型自回归 LM 后发现：词序越不规则越难学，但跨语言差异主要由词表覆盖、句长和形态复杂度解释，而不是简单的自由/固定词序标签。
tags:
  - "ACL2026"
  - "多语言/翻译"
  - "词序可学习性"
  - "多语言建模"
  - "词表结构"
  - "Mallows排列"
  - "语言类型学"
---

# Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models

**会议**: ACL2026  
**arXiv**: [2603.19427](https://arxiv.org/abs/2603.19427)  
**代码**: https://gitlab.gwdg.de/huds/projects/shuffle/-/tree/v1.0.2  
**领域**: multilingual_mt  
**关键词**: 词序可学习性, 多语言建模, 词表结构, Mallows排列, 语言类型学

## 一句话总结
本文用 Mallows 模型生成 10 种欧洲语言的连续词序扰动谱，训练小型自回归 LM 后发现：词序越不规则越难学，但跨语言差异主要由词表覆盖、句长和形态复杂度解释，而不是简单的自由/固定词序标签。

## 研究背景与动机
**领域现状**：语言类型学长期关注不同语言为何有不同词序，例如英语强依赖 SVO 位置，而捷克语、芬兰语等可以通过格标记和形态变化允许更自由的词序。NLP 里也有一类 synthetic language 或 word-order perturbation 实验，用打乱词序来研究语言模型是否有自然语言归纳偏置。

**现有痛点**：已有实验往往把词序、形态和 tokenizer 混在一起。很多工作在 subword 层面打乱，结果会把一个词切开的子词也拆散，等于同时破坏词序和形态结构。另一些工作用离散扰动强度，难以比较“轻微局部打乱”和“完全随机打乱”的连续变化。再加上语言选择常偏英语，跨语言差异很难解释。

**核心矛盾**：自由词序语言通常形态更丰富，固定词序语言通常更依赖位置，两者天然相关。若直接比较“自由 vs 固定”，无法判断模型学不好的原因到底是词序本身、形态复杂度、词表统计，还是 tokenizer 对不同语言的编码效率。

**本文目标**：作者希望构造一个更干净的实验：只改变词序规律，尽量保留原语言的词表、形态和全局熵；然后观察模型 surprisal 如何随扰动强度变化，并进一步解释为什么不同语言对词序扰动的鲁棒性不同。

**切入角度**：论文用 Mallows permutation model 给每个句长采样一个确定性排列。控制参数 $\theta$ 能连续覆盖原始词序、局部打乱、完全随机、局部反序和完全反转。这样每种语言都得到一条完整的词序规律谱，而不是几个离散打乱版本。

**核心 idea**：用 word-level deterministic shuffling 隔离词序因素，再用词表统计与 PLS 回归解释模型 surprisal 的跨语言变化。

## 方法详解

### 整体框架
整套流程从 Europarl 多语言平行语料出发。作者选择 10 种欧洲语言，其中 5 种通常归为固定词序，5 种通常归为自由词序，并覆盖 analytic、fusional、agglutinative 等形态类型。每种语言先做统一清洗：小写化、去标点、移除超过 80 词的句子，再切分为 650,000 句训练、5,000 句验证、5,000 句测试。

随后，作者对每种语言构造一系列合成词序变体。核心是：对每个句长 $n$，只采样一个排列 $\pi^{(n)}$，再把这个排列应用到该语言所有长度为 $n$ 的句子。这样变换是确定性的，新增描述长度很小，不会像随机逐句打乱那样显著抬高模型无关熵。

最后，作者为每个语言变体从头训练相同的小型自回归 Transformer，即 PicoLM 风格 50.5M 参数 decoder-only LM。默认使用 ByteLevel-BPE，词表大小 $|V|=16,000$；在词表实验中扫描 $|V|=258, 1000, 8000, 16000, 32000, 64000$。可学习性用测试集平均 surprisal 衡量，surprisal 越低，表示模型越能捕捉该语言变体的概率结构。

### 关键设计
**1. Mallows 连续词序扰动谱：用一个参数把词序从原样连续推到完全随机再到反转。**

以往实验大多只比"原文 vs 随机打乱"两个离散点，没法回答"轻微局部打乱"和"完全乱序"之间到底发生了什么。作者改用 Mallows 排列模型，按排列 $\pi$ 与原始词序 $\pi_0$ 的距离赋概率 $P(\pi) \propto \exp(-\theta d(\pi,\pi_0))$，距离 $d$ 取 Kendall's $\tau$，也就是把当前排列还原成原序所需的相邻交换次数。

控制参数 $\theta$ 就此成了一根连续旋钮：$\theta \to \infty$ 贴近原始词序，$\theta=0$ 时所有排列等概率（最乱），$\theta \to -\infty$ 逼近完全反转。这样每种语言都能得到一整条 $S(\theta)$ 可学习性曲线，而不是几个孤立的打乱版本；正负 $\theta$ 的对称性还让"句子反转"和"同等强度的正向局部打乱"可以被直接对照，从而单独检验模型是否对违反类型学的反序格外敏感。

**2. 词级确定性打乱：只动词序，不碰词内部的形态结构。**

如果先 tokenize 再在 subword 层面打乱，一个捷克词可能被切成两个子词、散落到句子两端——这时模型学不好，到底是因为词序乱了还是形态单位被拆了，根本分不清。作者因此把扰动严格限制在词级：先把 word 定义为 whitespace 分隔的正字法单位，在词级别做排列，再交给 tokenizer 处理打乱后的文本。

关键的"确定性"体现在：对同一语言、同一句长 $n$，只采样一个排列 $\pi^{(n)}$，并把它套用到该语言所有长度为 $n$ 的句子，而不是逐句重新随机。这样变换的新增描述长度很小，几乎不抬高模型无关熵，词表和形态分布也基本保持原样，扰动带来的 surprisal 变化才能干净地归因到词序本身。

**3. 用词表结构解释跨语言差异：找比"自由/固定词序"更细的自变量。**

"自由词序语言更抗打乱"这种粗标签解释力很弱，因为自由词序通常又伴随更丰富的形态，两者纠缠在一起。作者于是从每种语言抽出一组更细的统计量——词表覆盖率、subword 覆盖率、coverage integral、word-subword coverage similarity、平均词数与 subword 数、fertility、平均词长、unique word types 等。

由于这些指标高度共线、而语言样本只有 10 种，普通线性回归会很不稳，作者改用 multivariate PLS 回归，把相关特征压成少数潜变量去预测每种语言在 28 个 $\theta$ 值上的整条 surprisal 曲线。PLS 既稳住了小样本高共线的拟合，又能把"词表覆盖分量"和"形态复杂度分量"分开解读——后续实验正是靠这一分离，得出"词表覆盖解释平常难度、形态复杂度解释强打乱后的可恢复性"的结论。

### 损失函数 / 训练策略
本文没有提出新训练损失，而是把标准自回归语言建模作为测量工具。模型目标是预测下一个 subword，评估指标是 $S(\theta)=1/N \sum_i -\log p(w_i|w_{<i})$。作者关心的不是绝对性能，而是相对原始词序的 surprisal 增量 $\Delta S(\theta)=S(\theta)-S_{orig}$，因为 deterministic shuffling 基本保持模型无关熵，$\Delta S$ 可以解释为模型对词序扰动的敏感性。

## 实验关键数据

### 主实验
第一组结果研究词序扰动本身。模型在所有语言上都对完全不规则词序敏感，但对句子反转的额外敏感度很弱；自由/固定词序二分也不能解释不规则词序下的鲁棒性。

| 实验问题 | 关键指标 / 现象 | 结果 | 解释 |
|----------|----------------|------|------|
| 词序越不规则是否越难学 | $\Delta S$ 随 $\theta$ 变化 | $\theta=0$ 附近 surprisal 增量最大 | 自回归 LM 有明显 locality bias，完全随机词序最难压缩 |
| 反转是否比同强度正向打乱更难 | median asymmetry $\Delta S^{+/-}$ | 0.096，约为不规则词序影响的 6% | 反序违背类型学相关性，但模型主要感知局部性而非语言学合法性 |
| 反转差异是否显著 | Wilcoxon signed-rank | $p=0.0098$，显著但效应小 | 统计上有差异，实际幅度有限 |
| 自由/固定词序能否解释 $\Delta S_{irreg}$ | Wilcoxon-Mann-Whitney | $p=0.55$ | 粗粒度类型标签无法区分不规则扰动鲁棒性 |
| subword vs word 打乱 | irregular order 下 surprisal | subword 打乱整体更高，Balto-Slavic 和 Uralic 增幅更大 | 打破形态单位会扭曲跨语言比较 |

### 消融实验
第二组结果把解释变量换成词表结构和形态代理指标，显示词表比二分词序标签更有解释力。

| 分析配置 | 指标 | 数值 / 现象 | 说明 |
|----------|------|-------------|------|
| PLS 两个潜变量预测完整 $S(\theta)$ 曲线 | overall explained variance | $R^2=0.97$ | 词表和形态指标几乎可重构跨语言 surprisal 曲线 |
| leave-one-language-out | per-language $R^2$ | Hungarian 0.93，Portuguese / Latvian 0.99 | 对未见语言也有较强预测力 |
| 逐个 $\theta$ 切片，两组件 | mean $\bar{R}^2$ | 0.79，范围 0.66-0.86 | 解释力在不同扰动强度下稳定 |
| 只用 vocabulary component | mean $\bar{R}^2$ | 0.65，范围 0.26-0.76 | 词表覆盖主要解释原序和反序表现 |
| 第二组件 | 主要载荷 | unique word types、word length、fertility | 强不规则词序下需要形态复杂度解释 |
| 扫描词表大小 | 分离阈值 | $|V|>8000$ 后原始 surprisal 开始区分自由/固定词序 | tokenizer 规模调节跨语言可学习性差异 |

### 关键发现
- 完全不规则词序会系统性提高 surprisal，但模型对“反序”并没有表现出强烈额外惩罚。这说明自回归 LM 更在意局部预测结构是否被破坏，而不是句子是否符合人类类型学偏好。
- 自由词序语言并不天然更抗打乱。自由和固定词序组在 $\Delta S_{irreg}$ 上高度重叠，只有 Romance 与 Finnic 等极端簇同时在原始 surprisal 和扰动增量上分开。
- 词表 coverage 能把自由/固定词序语言聚成更连续的结构。自由词序语言常有更慢增长的词和 subword 覆盖曲线，说明更多低频形式参与建模。
- 形态复杂度代理指标主要解释 $\theta=0$ 附近的强不规则区。换句话说，词表统计解释“平常学起来难不难”，形态结构解释“被强打乱后还剩多少可恢复线索”。
- 词表大小本身会改变结论。大于 8K 词表时，原始 surprisal 才明显区分组别；但在不规则词序增量上，随着词表继续扩大，不同语言又出现收敛或分化趋势。

## 亮点与洞察
- 最巧妙的设计是 deterministic word-level shuffling。它既能连续控制词序规律，又避免随机打乱增加全局熵，是比常见 shuffle baseline 更干净的语言学实验工具。
- 论文把“语言是否难学”从粗标签推进到可量化的词表结构。对于多语言 LM，这提醒我们不能只看语系或形态类型，tokenizer 形成的 coverage 和 fertility 可能更直接影响模型学习难度。
- 反转实验很有启发：模型对违反自然语言类型学相关性并不特别敏感，说明小型自回归 LM 的归纳偏置可能更偏局部可预测性，而不是人类语言合法性。
- PLS 的使用很合适。样本只有 10 种语言，变量高度相关，PLS 比堆一堆单变量相关更稳，也能把 vocabulary component 和 morphological complexity component 分开解释。
- 对 MT 和 multilingual NLP 的启发是：低资源或形态复杂语言的困难可能部分来自词表覆盖和 subword 稀疏性，而不只是语料规模或任务数据不足。

## 局限与展望
- 语料局限于 Europarl 和欧洲语言，虽然可比性强，但覆盖的类型学空间仍窄。更广泛的非欧洲语系、非 SVO 语言和口语/网络文本可能呈现不同规律。
- 模型是 50.5M 参数小型 autoregressive LM，结论不一定直接外推到大规模多语言 LLM。大模型的容量和 tokenizer 规模可能改变词表覆盖与 learnability 的关系。
- 论文用 LM surprisal 研究 computational learnability，不等价于人类学习难度。后续需要和眼动、阅读时间或二语习得数据做对照。
- 自由/固定词序仍被用作初始分组，但作者也承认类型学是连续的。未来可以用 subject-object entropy、dependency locality 等连续指标替代二分标签。
- 评估看全局 surprisal，尚未细分哪些 token 或构式贡献最大。进一步分析 determiner-adjective-noun、格标记、动词位置等语言特定结构，会让解释更可操作。

## 相关工作与启发
- **vs Kallini et al. 的 artificial language / shuffled LM 工作**: 他们证明打乱语言会伤害模型学习，本文进一步把扰动做成跨语言连续谱，并避免 subword-level 形态破坏。
- **vs Cotterell et al. / Mielke et al. 的语言建模难度研究**: 早期工作争论形态复杂度还是简单统计更能解释 LM 难度。本文给出折中图景：coverage 和句长解释大部分曲线，形态复杂度在强不规则扰动下尤其重要。
- **vs Arnett and Bergen 的 tokenizer 效率解释**: 本文支持 tokenizer/词表结构是跨语言差异的重要中介，但把它放进词序扰动实验中，说明词表不仅影响原始 LM 难度，也影响对打乱的鲁棒性。
- **vs 传统类型学自由/固定词序标签**: 粗标签对解释 $\Delta S_{irreg}$ 不够，本文主张用连续词表统计和形态代理指标表达语言结构。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 方法不是新模型，但实验控制很漂亮，把词序、词表和形态的纠缠拆得更清楚。
- 实验充分度: ⭐⭐⭐⭐☆ 训练了大量小模型并做多种扰动、词表大小与回归分析，但语言范围和模型规模有限。
- 写作质量: ⭐⭐⭐⭐⭐ 论文逻辑清楚，语言学动机、实验控制和统计解释衔接自然。
- 价值: ⭐⭐⭐⭐☆ 对多语言 LM、tokenizer 设计和语言类型学实验很有参考价值，工程直接收益相对间接。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](../../ACL2025/multilingual_mt/cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](../../ACL2025/multilingual_mt/crosslingual_pitfalls.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](../../ACL2025/multilingual_mt/dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)

</div>

<!-- RELATED:END -->
