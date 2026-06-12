---
title: >-
  [论文解读] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties
description: >-
  [NeurIPS 2025][LLM安全][语言鲁棒性] 提出Trans-EnV框架，结合语言学专家知识和LLM变换能力，将标准美式英语（SAE）数据集自动转换为38种英语变体（18种方言+20种ESL英语），揭示LLM在非标准英语上最高46.3%的性能下降，凸显了语言公平性问题。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "语言鲁棒性"
  - "英语方言"
  - "公平性"
  - "LLM评测"
  - "ESL英语"
---

# Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties

**会议**: NeurIPS 2025  
**arXiv**: [2505.20875](https://arxiv.org/abs/2505.20875)  
**代码**: [GitHub](https://github.com/jiyounglee-0523/TransEnV)  
**领域**: AI安全  
**关键词**: 语言鲁棒性, 英语方言, 公平性, LLM评测, ESL英语

## 一句话总结

提出Trans-EnV框架，结合语言学专家知识和LLM变换能力，将标准美式英语（SAE）数据集自动转换为38种英语变体（18种方言+20种ESL英语），揭示LLM在非标准英语上最高46.3%的性能下降，凸显了语言公平性问题。

## 研究背景与动机

当前LLM的评测基准几乎全部使用标准美式英语（SAE），但英语在全球范围内存在丰富的变体——包括地区方言（如苏格兰英语、爱尔兰英语）和非英语母语者使用的ESL英语（如阿拉伯母语者的英语）。已有研究表明LLM在非标准英语上表现显著下降，更严重的是，模型对非主流变体的回应更具刻板印象和居高临下的语气。

然而，对LLM语言鲁棒性的系统评估面临三个障碍：

**现有多变体数据集规模小、覆盖面窄**：如Multi-VALUE仅涉及数种方言和有限任务

**人工标注方法不可扩展**：为数十种变体逐一人工创建数据集资源消耗巨大

**纯LLM生成不可靠**：LLM在再现弱势英语变体时存在已知偏差，不能单独依赖

**规则方法不够灵活**：无法捕捉词汇选择和语用层面的变化

作者的方案是将专家语言学资源（确保正确性）与LLM（确保可扩展性）结合，构建一个可自动将SAE句子转换为目标变体的框架。

## 方法详解

### 整体框架

Trans-EnV分三步：（1）数据收集——从语言学文献和语料库中获取变体及其语言特征；（2）转换指南生成——为每个特征创建包含"资格检查"和"应用步骤"的转换操作说明；（3）句子转换——使用LLM依据指南逐步将SAE句子转换为目标变体，并用语义一致性检查器把关质量。

### 关键设计

1. **基于eWAVE的方言特征提取**：利用eWAVE数据库（84位语言学家基于175篇论文编纂，涵盖77个英语变体的235个语言特征），对77个变体进行KNN聚类以区分英语方言和克里奥尔语等非英语变体，筛选出18种方言。每种方言选取eWAVE中标注为最高存在度（highest level of presence）的特征作为其特征集 $\mathcal{L}_{v_i}$。

2. **ESL英语的双维度特征**：

    - **CEFR维度**（语言能力）：基于English Grammar Profile的1,222个"能力描述"（can-do descriptors），模拟目标CEFR水平时移除高于该水平的能力特征。如模拟A级时移除所有B级和C级的can-do特征。
   
    - **母语L1维度**（迁移效应）：使用三个ESL语料库（CLC-FCE、ICLE、EFCamDat），通过统计t检验（$p < 0.05$）从相同CEFR水平的学习者写作中提取L1特有的语法错误模式。覆盖10种母语（阿拉伯语、中文、法语、德语等），每种母语在每个CEFR水平平均提取10个特征。母语特征与CEFR特征经验证不冲突后合并使用。

3. **转换指南与执行机制**：每个语言特征 $l_j$ 配套一个转换指南 $g_j$，包含两步：（a）**Qualification**——判断该特征是否适用于当前句子（如"she/her用于无生命指代"需要句中存在无生命指代对象和相应代词）；（b）**Application**——具体变换操作（如找到无生命指代并将其代词替换为she/her）。使用GPT-4进行one-shot生成。句子转换时随机打乱特征顺序逐一应用，每次变换后由语义检查器 $S$（LLaMA-3.3-70B-Instruct）验证语义一致性。对ESL英语还增加词汇简化步骤，允许最多15%的高级词汇保留以符合真实ESL文本的词汇分布。

### 损失函数 / 训练策略

- 特征转换器 $T$ 使用Gemma-2-27B-Instruct，语义检查器 $S$ 使用LLaMA-3.3-70B-Instruct
- 语义检查器 $S$ 经人工评估达到83.6% precision、97.0% recall、89.8% F1
- 人工评估显示除LLaMA-3.1-8B外，所有模型作为 $T$ 的有效转换率均超90%
- 变换覆盖率：整体82-95%的样本至少被修改一个特征，每样本平均2个特征

## 实验关键数据

### 主实验（方言对LLM性能影响）

| 模型 | SAE原始 | 方言平均 | 最差方言 | 最大下降 |
|------|--------|---------|---------|---------|
| Qwen2.5-72B | 82.2 | 80.5 | NFE: 76.1 | -7.4% |
| DeepSeek-R1-70B | 80.8 | 79.6 | NFE: 75.2 | -6.9% |
| LLaMA-3.3-70B | 76.4 | 74.9 | IrE: 71.3 | -6.7% |
| o4-mini | 88.5 | 87.1 | NFE: 83.4 | -5.8% |
| GPT-4o-mini | 74.6 | 73.1 | WeE: 69.8 | -6.4% |

### ESL英语性能下降幅度

| 配置 | MMLU下降 | ARC下降 | GSM8K下降 | HellaSwag下降 |
|------|---------|--------|----------|-------------|
| CEFR B（中级） | ~3-8% | ~2-6% | ~5-15% | ~3-10% |
| CEFR A（基础） | ~5-15% | ~4-12% | ~10-46.3% | ~5-15% |
| 推理任务平均 | 22.6% | - | - | - |
| 知识任务平均 | 10.9% | - | - | - |

### 关键发现

- **ESL英语远比方言更具挑战性**：ESL最大下降46.3%，方言最大下降12.5%，ESL英语的语法偏离程度更大
- **推理任务比知识任务更敏感**：方言上推理任务平均下降4.7% vs 知识任务1.3%；ESL上差距更大（22.6% vs 10.9%）
- **强推理模型更鲁棒**：DeepSeek-R1和o4-mini等专注推理的模型在方言上表现更稳健
- **语言距离与性能下降正相关**：通过eWAVE特征向量的欧氏距离和DLIFLC语言难度分级，证实了语言距离越大性能下降越严重的趋势
- **数据量决定鲁棒性**：表现最差的方言（纽芬兰英语、威尔士英语）使用者少，表现最好的（澳大利亚英语）使用者多；ESL中法语/意大利语（训练数据多）优于阿拉伯语/土耳其语（训练数据少）

## 亮点与洞察

- **专家知识+LLM的黄金组合**：eWAVE和English Grammar Profile提供语言学保证，LLM实现可扩展执行，两者互补克服各自弱点
- **揭示公平性鸿沟**：英语变体间的性能差距不仅是技术问题，更是全球用户能否公平获得AI服务的社会公平问题
- **方法论启发**：Qualification+Application两步转换范式可推广到其他语言变换任务，如简化、风格迁移等

## 局限与展望

- 仅覆盖英语变体，向其他语言（如西班牙语、阿拉伯语变体）扩展需要对应的语言学资源
- 评测限于QA任务，面向文本生成、对话等开放式任务的评估虽有初步探索但不够深入
- 部分结构简单或高度专业的样本无法转换（如纯数学符号），约5-30%样本未被修改
- 转换特征的随机排序可能引入不必要的变异性
- 未探索多变体数据增强是否能提升LLM对非标准英语的鲁棒性

## 相关工作与启发

- Multi-VALUE和DIVA等先前多变体数据集为本工作提供了动机，但受限于规模和覆盖范围
- eWAVE数据库是宝贵的语言学资源，在NLP中的利用远未充分
- 本工作与dialect bias研究（如Value的方言偏见检测）形成互补——Trans-EnV关注任务性能下降，后者关注生成内容的偏见

## 评分

- **新颖性**: ⭐⭐⭐⭐ 系统性地结合语言学专家资源与LLM变换的框架设计新颖，38种变体的覆盖范围是前所未有的
- **实验充分度**: ⭐⭐⭐⭐⭐ 6个数据集、38种变体、7个模型，加上人工评估、语言距离分析和开放式任务实验
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，语言学背景介绍充分，但部分内容分散在附录中
- **价值**: ⭐⭐⭐⭐⭐ 为LLM的语言公平性研究提供了标准化工具和大规模实证证据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2025\] Truth Knows No Language: Evaluating Truthfulness Beyond English](../../ACL2025/llm_safety/truth_knows_no_language_evaluating_truthfulness_beyond_english.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](../../ACL2025/llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](../../ACL2026/llm_safety/evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)

</div>

<!-- RELATED:END -->
