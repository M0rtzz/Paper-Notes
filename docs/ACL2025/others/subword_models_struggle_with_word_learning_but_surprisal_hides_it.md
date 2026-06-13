---
title: >-
  [论文解读] Subword Models Struggle with Word Learning, but Surprisal Hides It
description: >-
  [ACL 2025][词汇学习] 本文通过心理语言学中的词汇决策任务（lexical decision），揭示了子词（BPE）语言模型在孤立词汇识别上远不如字符级模型，而常用的 surprisal 指标因引入句法上下文掩盖了这一缺陷。 人类习得语言时，先学会识别单词，再理解语法。然而，现有的将 LM 作为语言习得模型的研究大…
tags:
  - "ACL 2025"
  - "词汇学习"
  - "子词分词"
  - "字符级模型"
  - "词汇决策"
  - "surprisal"
---

# Subword Models Struggle with Word Learning, but Surprisal Hides It

**会议**: ACL 2025  
**arXiv**: [2502.12835](https://arxiv.org/abs/2502.12835)  
**代码**: 有  
**领域**: NLP / 认知语言学  
**关键词**: 词汇学习, 子词分词, 字符级模型, 词汇决策, surprisal

## 一句话总结

本文通过心理语言学中的词汇决策任务（lexical decision），揭示了子词（BPE）语言模型在孤立词汇识别上远不如字符级模型，而常用的 surprisal 指标因引入句法上下文掩盖了这一缺陷。

## 研究背景与动机

人类习得语言时，先学会识别单词，再理解语法。然而，现有的将 LM 作为语言习得模型的研究大多聚焦于句法层面，对隐含的"词汇学习"过程关注不足。已有的词汇学习研究主要通过 **surprisal（负对数概率）** 来衡量模型是否"学会"了某个词，但 surprisal 本质上衡量的是"在给定上下文中词的预期程度"，与模型训练目标直接对应，无法真正揭示模型是否具备独立的词汇知识。

此外，BPE 等子词分词方法会将单词切分为语言学上不合理的子单元，从认知科学角度看并不可信。而字符级模型避免了这种先验切分，理论上应更接近人类的词汇发现过程。

本文的核心问题是：**LM 是否"知道"哪些字符串是合法单词？** 这比"LM 能否预测词在上下文中的出现"更基本。

## 方法详解

### 整体框架

作者设计了三种实验范式来探测模型的词汇知识，从无上下文到有上下文递进：

1. **词汇决策（Lexical Decision）** —— 无上下文
2. **Surprisal** —— 合理上下文
3. **Anti-Surprisal** —— 不合理上下文

### 关键设计

1. **词汇决策任务**
    - 功能：给定一个真词/非词对（如 sending / monding），判断哪个是真词
    - 核心思路：仅在空格符（最中性的起始 token）后计算两个词的平均 surprisal，比较大小
    - 设计动机：模拟心理语言学中的强制选择词汇决策，剥离句法/语义上下文的干扰
    - 用 wuggy 工具生成 1000 对高频词和 1000 对低频词的最小对

2. **Surprisal 实验**
    - 功能：在合理句法上下文中测量真词 vs 非词的 surprisal
    - 核心思路：从 OpenSubtitles 中采样包含目标词的句子，将目标词替换为匹配的非词
    - 设计动机：测试"当句法上下文可用时"模型能否更好区分词与非词

3. **Anti-Surprisal 实验**
    - 功能：在不匹配的上下文中插入真词/非词
    - 核心思路：选择不包含目标词的句子，随机在位置 ≥3 处插入
    - 设计动机：提供词汇上下文但不提供语义/句法线索，测试纯粹的"其他词存在"是否有助于判断

4. **学习轨迹分析**
    - 对数间隔保存 19 个中间 checkpoint
    - 同时在 BLiMP（句法基准）和词汇决策任务上评估
    - 对比字符模型和子词模型中"词汇学习"与"句法学习"的时间关系

### 模型配置

| 模型 | 分词方式 | 参数量 | 训练数据 |
|------|----------|--------|----------|
| Llama (×3) | 字符/BPE | 0.49M-30M | BabyLM 10M |
| GPT-2 (×2) | 字符/BPE | 85-97.5M | 100M words |
| Pythia (×6) | BPE | 14M-1.4B | 825GB |

## 实验关键数据

### 主实验——词汇决策 vs Surprisal（Table 1 摘要）

| 模型 | 分词 | 词汇决策(高频/低频) | Surprisal(高频/低频) | Anti-Surprisal(高频/低频) |
|------|------|---------------------|---------------------|-------------------------|
| Llama-0.49M | 字符 | 97.6/83.0 | 98.2/84.3 | 98.0/83.1 |
| Llama-21.9M | 字符 | 99.0/93.3 | 99.8/94.7 | 99.0/92.5 |
| GPT-2 | 字符 | 98.7/97.3 | 99.8/99.4 | 98.0/96.3 |
| Llama-30M | BPE | 83.6/68.6 | 92.7/81.1 | 83.7/76.1 |
| Pythia-1.4B | BPE | 87.8/81.6 | 97.9/97.9 | 76.5/84.7 |
| GPT-2 | BPE | 35.6/79.1 | 99.0/99.2 | 84.7/86.9 |

### 关键发现

1. **字符级模型在词汇决策上接近完美**（97-99%），而即使最大的 BPE 模型也只有约 88%
2. **Surprisal 掩盖了差距**：在有上下文的情况下，BPE 模型追赶上来（>90%），但这依赖于句法信号
3. **Anti-Surprisal 揭示 BPE 的纠结**：BPE 模型在不合理上下文中反而偏好非词，说明其词汇和句法知识不可分离
4. **学习轨迹差异显著**：
    - 字符模型：词汇学习先于句法学习，两条曲线清晰分离
    - BPE 模型：词汇和句法学习轨迹高度相关、同时发生，呈S形曲线
5. BPE 模型存在高频/低频词的持续性能差距，无法通过增大模型弥合

### 消融实验

- 不同模型架构（Llama/GPT-2/Pythia）结果一致
- 模型规模扩大带来的收益在 BPE 模型上有限
- 训练数据量差异（10M vs 825GB）不改变字符 vs BPE 的整体趋势

## 亮点与洞察

1. **方法论创新**：将心理语言学的词汇决策范式引入 LM 评估，填补了"独立于句法的词汇探测"空白
2. **深刻洞察**：surprisal 作为评估词汇学习的指标存在根本性问题——它与训练目标直接对应，无法真正探测抽象的词汇知识
3. **认知启示**：字符模型的学习轨迹（先词汇后句法）更符合人类儿童的语言习得顺序
4. **实证发现**：BPE 分词的先验切分实际上"跳过"了词汇发现阶段，导致词汇和句法学习纠缠在一起

## 局限与展望

- 仅在英语上实验，不同书写系统/音素系统可能有不同模式
- 字符级模型仅在小规模上测试，缺乏大规模字符模型的验证
- 未涵盖词汇学习的语义/指称维度（如物体命名）
- 未探究形态学感知分词器（morphology-aware tokenizers）的表现

## 相关工作与启发

- Chang & Bergen (2022) 的 surprisal 阈值方法虽直觉，但将频繁功能词识别为"最早学会"的词，与儿童实际产出矛盾
- Le Godais et al. (2017) 早已在字符 LSTM 上观察到 ~95% 的词汇决策准确率
- 对分词方法的选择在 BabyLM / 语言习得模拟中应更加谨慎

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 词汇决策范式在 Transformer LM 评估中首次系统使用，视角新颖
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多架构、多分词方式、学习轨迹分析全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰，动机与实验紧密衔接，图表精美
- **价值**: ⭐⭐⭐⭐ — 对理解 LM 内部词汇表征有重要意义，对 BabyLM 社区尤为关键

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ACL 2025\] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [\[ACL 2025\] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](aligned_but_blind_implicit_bias.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/others/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)

</div>

<!-- RELATED:END -->
