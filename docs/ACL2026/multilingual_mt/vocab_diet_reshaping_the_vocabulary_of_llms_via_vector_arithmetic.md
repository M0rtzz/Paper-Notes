---
title: >-
  [论文解读] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic
description: >-
  [ACL 2026][多语言/翻译][组合式词表] 本文发现 LLM 在嵌入空间中将词形变化（如 walk→walked）编码为线性方向，基于此提出组合式词表设计：用基础词+变换向量的加法组合替代为每个表面形式分配独立 token，在冻结预训练骨干的前提下仅训练小型适配模块…
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "组合式词表"
  - "向量算术"
  - "形态学变换"
  - "词表压缩"
  - "多语言覆盖"
---

# Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic

**会议**: ACL 2026  
**arXiv**: [2510.17001](https://arxiv.org/abs/2510.17001)  
**代码**: [GitHub](https://vocabdiet.github.io)  
**领域**: 多语言翻译  
**关键词**: 组合式词表, 向量算术, 形态学变换, 词表压缩, 多语言覆盖

## 一句话总结

本文发现 LLM 在嵌入空间中将词形变化（如 walk→walked）编码为线性方向，基于此提出组合式词表设计：用基础词+变换向量的加法组合替代为每个表面形式分配独立 token，在冻结预训练骨干的前提下仅训练小型适配模块，释放 10-40% 的词表槽位用于多语言扩展，同时几乎不影响下游性能。

## 研究背景与动机

**领域现状**：现代 LLM 普遍使用 BPE 分词算法，词表规模已超过 100K token。词表设计本质是资源分配问题：每个分配给某种语言或领域的槽位，都以其他覆盖范围为代价。近期研究发现词表在不同语言间分配严重失衡，负面影响模型成本和性能。

**现有痛点**：(1) **冗余分配**：标准分词将形态相关的词形（walk, walks, walking, walked）视为独立 token，每个占据一个词表槽位。以 GPT-4 词表为例，24.6K 英语全词 token 去除大小写和屈折/派生变体后仅剩 14.3K 基础形式——42% 的冗余。(2) **多语言覆盖不足**：大量词表槽位被高资源语言的表面变体占据，低资源语言覆盖严重不足。(3) **OOV 问题**：现有 14.3K 基础形式和变换可以组合出 98K 目前词表外的词汇，但标准词表无法利用这一结构。

**核心矛盾**：词表大小受内存和计算约束限制，但标准分词忽略了 LLM 嵌入空间中已经存在的线性形态结构——模型内部已将词形变化编码为简单的向量偏移，却在词表层面仍为每个变体单独学习嵌入。

**本文目标**：(1) 验证 LLM 是否能理解"基础词+变换向量"的组合式嵌入；(2) 构建组合式词表，释放冗余槽位；(3) 在后训练适配和从头预训练两种场景下验证可行性。

**切入角度**：从 word2vec 时代的向量算术（king - man + woman = queen）出发，系统验证 LLM 嵌入空间中形态变换的线性结构，并将其从分析工具提升为实用的词表设计方案。

**核心 idea**：用组合式词表替代扁平式词表——每个表面形式 $w$ 由基础词 $b_w$ 和一组变换 $T(w)$ 组合而成：$e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$，在输入和输出端同时应用，释放词表空间用于多语言扩展。

## 方法详解

### 整体框架

分为三个阶段：(1) 分析验证——使用 Patchscopes 探针验证 LLM 能否正确解读组合式嵌入；(2) 后训练适配——在已训练模型上用知识蒸馏微调变换向量并加 LoRA 适配器；(3) 从头预训练——验证组合式词表作为新模型设计选择的可行性。输入端将表面形式替换为基础词+变换向量的和，输出端将大 unembedding 矩阵拆分为基础词和变换两个独立投影。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["表面形式<br/>walk / walks / walking / walked"] --> REP
    subgraph REP["组合式词表表示（基础词 + 变换向量之和）"]
        direction TB
        B["输入端：e_w = e_b + Σ e_t"]
        C["输出端：unembedding 拆成基础词路 + 变换路"]
    end
    REP --> D["Patchscopes 验证<br/>模型天生能读懂加法嵌入吗：屈折成功 · 派生较弱"]
    D --> E["后训练适配<br/>两阶段知识蒸馏 + 末 8 层 LoRA"]
    REP --> F["从头预训练<br/>分解式预测 p(w|h)=p(b|h)·p(T|b,h)"]
    E --> G["释放 10-40% 词表槽位 → 多语言扩展"]
    F --> G
```

### 关键设计

**1. 组合式词表表示：把每个表面形式拆成「基础词 + 变换向量」之和**

标准分词的浪费在于 walk / walks / walking / walked 各占一个词表槽位，可它们在嵌入空间里其实只差一个固定方向。组合式词表正面利用这一点：定义基础词词表 $V_b \subset V_{orig}$（规范词形加少量辅助 token）和变换词表 $V_t$（时态、数等形态操作），任何词 $w$ 都由它的基础词 $b_w$ 和一组变换 $T(w)$ 拼出来。输入端把嵌入写成加法 $e_w = e_{b_w} + \sum_{t_i \in T(w)} e_{t_i}$；输出端则把原来一整张大 unembedding 矩阵拆成基础词和变换两路独立投影，再把打分相加 $\text{logit}(w) = h \cdot u_{b_w} + \sum_{t_i \in T(w)} h \cdot u_{t_i}$。

变换向量并非随机初始化，而是直接从模型已学到的偏移里估计：对某个变换 $t$，取所有适用该变换的词，平均它们与各自基础词的嵌入差，$o_t = \frac{1}{|R(t)|}\sum_{w \in R(t)} (o_w - o_{b(w)})$。这样一个变换向量被所有同类词形共享，冗余的独立嵌入被压成共享组合表示；更妙的是，词表外的词（如从未单独成 token 的 "walkable"）也能用同一套加法即时拼出，词表内外的处理被统一起来。

**2. Patchscopes 验证框架：先确认模型「天生」就能读懂这种加法嵌入**

在把组合式词表真正投入训练前，必须回答一个前置问题——模型内部是否真把词形变化编码成了线性方向？如果不是，加法嵌入就只是一厢情愿。作者用 Patchscopes 探针来验证：对每个可组合的词 $w$，把它原本的 token 嵌入替换成组合表示 $e_w$，再用 Patchscopes 提示让模型生成文本描述，看输出是否还原成目标词。

验证分两个位置进行：直接在嵌入层（embed）读，以及在早期若干层完成去分词（detok）后再读。覆盖五种语言（英语、阿拉伯语、德语、俄语、西班牙语）和多个骨干（Llama-3-8B、Qwen2.5-7B、OLMo-2-7B、ALLaM-7B、EuroLLM-9B）。只有当模型确实能把"walk + 过去时"这类组合解读回 "walked"，后面的词表压缩才站得住脚——而实验也正是在这一步分出了屈折变换（成功）和派生变换（较弱）的边界。

**3. 两阶段知识蒸馏微调：在冻结骨干上几乎零成本地装上组合式词表**

验证通过后，问题变成怎样在一个已经训练好的模型上换装组合式词表而不伤性能。直接同时优化输入端和输出端的变换向量会互相干扰、训练不稳，所以作者把它拆成两阶段蒸馏：第一阶段冻结输出 unembedding，只训输入变换向量，以原始模型的预测作软目标；第二阶段反过来冻结输入嵌入，只训输出变换向量，并以第一阶段的模型为蒸馏目标。

为了让中间表示尽量不动，LoRA 只加在最后 $k=8$ 层（$r=256$），其余参数全冻——因为组合式表示主要改变的是输入/输出映射，中间层无需重学。整套适配只引入不到 0.001% 的额外参数（即那些变换嵌入），训练数据也只用了 FineWeb-Edu 约 5M tokens，却能释放约 10K 词表槽位、下游平均仅掉 1.0 个点。

### 损失函数 / 训练策略

使用知识蒸馏损失（KL 散度），以原始模型预测为软目标。后训练适配引入不到 0.001% 额外参数（变换嵌入）。从头预训练场景使用分解式预测：$p(w|h) = p(b_w|h) \cdot p(T(w)|b_w; h)$，先预测基础词再条件预测变换。

## 实验关键数据

### 主实验 — 英语后训练适配（Llama-3.1-8B）

| 任务类别 | 任务 | 原始模型 | 组合式词表 | 差异 |
|---------|------|---------|-----------|------|
| 知识 | MMLU | 65.2 | 64.9 | -0.3 |
| 知识 | ARC | 53.6 | 52.5 | -1.1 |
| 阅读理解 | BoolQ | 83.2 | 83.3 | +0.1 |
| 阅读理解 | TriviaQA | 66.5 | 63.3 | -3.3 |
| 常识 | HellaSwag | 60.6 | 59.5 | -1.1 |
| 常识 | Winogrande | 78.1 | 78.6 | +0.5 |
| **平均** | | **66.9** | **65.9** | **-1.0** |

### 从头预训练实验（nanoGPT-124M）

| 语言 | 词表压缩 | BPB (基线) | BPB (组合式) | bytes/tok 变化 |
|------|---------|-----------|-------------|---------------|
| 英语 | 41.6% | 1.08 | 1.09 | — |
| 西班牙语 | 41.8% | 1.00 | 1.11 | 4.77→4.92 |

### 关键发现

- 英语屈折变换（复数、时态等）在嵌入层即可被正确解读：复数名词 92%、过去时 71%、现在分词 83%
- 早期层去分词后准确率进一步提升：复数 96%、过去时 81%、现在分词 93%
- 关键区分：屈折变换（inflection）效果好，派生变换（derivation）效果差——derivation 在词表中很少作为单 token 出现，模型学到的线性结构较弱
- 多语言结果惊人：俄语大小写 97%、形容词屈折 100%（样本量小），表明非英语语言的线性结构可能更强
- 后训练适配仅损失平均 1.0 个点的下游性能，同时释放约 10K 词表槽位
- 从头预训练可释放 41% 词表槽位，BPB 仅增加 0.01（英语）
- 词表槽位重新分配后，四种语言的 bytes-per-token 平均提升 9.3%

## 亮点与洞察

- 揭示了一个深刻的现象：LLM 嵌入空间中的线性形态结构不仅存在于词表内词汇，对词表外词汇同样有效——模型从未见过 "walkable" 作为单个输入向量，但 "walk + -able" 的组合在早期层也能被正确解读（虽然派生类较弱）
- 词表大小与形态线性结构呈反比关系：词表越小，模型被迫更多地利用线性组合编码词形变化，线性结构越强。这为小词表的合理性提供了理论支撑
- 实用价值突出：释放的 10K 槽位可为每种目标语言分配 2.5K 专用 BPE token，仅此即可显著改善多语言分词效率（平均 +9.3% bytes-per-token）

## 局限与展望

- 派生变换（-able, un-, re- 等）的组合准确率低，当前方案只能处理屈折和大小写变换
- 依赖 UniMorph 形态学数据库构建分解映射，对无标注的语言不适用
- 后训练适配在阅读理解类任务（TriviaQA -3.3, SQuAD -2.1）上损失相对较大，可能因为这些任务更依赖精确的词形匹配
- 从头预训练仅在 124M 参数的小模型上验证，更大规模模型的效果待探索
- 输出端的分解预测增加了计算步骤，虽然实测仅减速 0.8%，但在极端延迟敏感场景仍需关注

## 相关工作与启发

- **vs 词表扩展方法（如 Nakash et al., 2025）**: 这些工作通过直接添加新 token 来改善特定语言覆盖，但受制于词表总大小上限；Vocab Diet 通过压缩已有冗余来"腾出空间"，与扩展方法互补
- **vs Park et al. (2024) 的线性探针分析**: 他们分析了 LLM 嵌入空间的线性结构但仅作为分析工具；Vocab Diet 首次将这种结构用于端到端语言建模的实际系统
- **vs BPE 分词改进（如 Tao et al., 2024）**: 这些工作主张扩大词表以提升性能，但受限于计算成本；Vocab Diet 提供了一种在固定词表预算下提升效率的正交思路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从嵌入空间的线性形态结构出发重新设计词表，视角独到且开创性
- 实验充分度: ⭐⭐⭐⭐ 五语言五模型的验证全面，但从头预训练仅在 124M 模型上做
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条清晰，从验证→后训练→预训练层层递进，图表精美
- 价值: ⭐⭐⭐⭐⭐ 对多语言 LLM 的词表设计有深远影响，既有理论洞察又有实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)
- [\[ACL 2026\] What Factors Affect LLMs and RLLMs in Financial Question Answering?](what_factors_affect_llms_and_rllms_in_financial_question_answering.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](../../ACL2025/multilingual_mt/dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)

</div>

<!-- RELATED:END -->
