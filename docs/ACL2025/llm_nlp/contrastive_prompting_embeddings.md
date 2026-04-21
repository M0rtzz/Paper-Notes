---
title: >-
  [论文解读] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering
description: >-
  [ACL 2025][LLM/NLP][sentence embedding] 提出对比提示（Contrastive Prompting, CP）方法，通过构造辅助提示编码句子的非核心信息，在推理时将正常提示与辅助提示的隐层激活值做"语义减法"，过滤停用词等无关语义，使 LLM 句子嵌入更聚焦核心语义，即插即用地一致提升 PromptEOL/CoT/Knowledge 等多种提示方法在 STS 和分类任务上的表现。
tags:
  - ACL 2025
  - LLM/NLP
  - sentence embedding
  - 提示学习
  - activation steering
  - inference-time
  - LLM
---

# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering

**会议**: ACL 2025  
**arXiv**: [2505.12831](https://arxiv.org/abs/2505.12831)  
**代码**: [GitHub](https://github.com/zifengcheng/CP)  
**领域**: llm_nlp  
**关键词**: sentence embedding, contrastive prompting, activation steering, inference-time, LLM  

## 一句话总结

提出对比提示（Contrastive Prompting, CP）方法，通过构造辅助提示编码句子的非核心信息，在推理时将正常提示与辅助提示的隐层激活值做"语义减法"，过滤停用词等无关语义，使 LLM 句子嵌入更聚焦核心语义，即插即用地一致提升 PromptEOL/CoT/Knowledge 等多种提示方法在 STS 和分类任务上的表现。

## 研究背景与动机

**领域现状**: 从 LLM 直接提取零样本句子嵌入（无需微调或额外数据）是实用方向，现有方法通过 prompt 工程将句子语义压缩到最后一个 token 的隐状态，如 PromptEOL（"This sentence: '[TEXT]' means in one word:"）、MetaEOL（多元 meta-task 提示）、Pretended CoT（思维链提示）、Knowledge（知识增强提示）等。

**现有痛点**: 即使精心设计 prompt，最后一个 token 仍编码了大量非核心信息——实验表明即使用 Knowledge 提示强调"主语和动作"，解码概率最高的仍是停用词"a"而非语义关键词。Prompt 工程本质上只能**间接**改变表示，无法**直接**过滤非核心信息。

**核心矛盾**: 现有方法都是通过改变前缀文本间接影响最后 token 的表示，缺乏一种直接在隐层空间剥离非核心语义的机制。

**本文切入点**: 受 activation steering 启发，但不依赖监督正负样本对——用一个辅助提示（"这句话的无关信息是..."）自适应捕获每个句子的非核心信息激活，再用正常提示的激活减去辅助提示的激活，实现逐句自适应的"语义减法"。

## 方法详解

### 整体框架

三步流程：(1) 将文本包裹在辅助提示中前向传播到第 $\ell$ 层，提取最后 token 的 contextualized value vector $\mathbf{v}^{\text{aux},(\ell)}$；(2) 将文本包裹在正常提示中前向传播到第 $\ell$ 层，计算对比向量 $\Delta\mathbf{v}^\ell = \mathbf{v}^{\text{nor},(\ell)} - \mathbf{v}^{\text{aux},(\ell)}$ 并替换最后 token 的 value vector；(3) 对替换后的向量做范数调整，继续前向传播到中间层提取句子嵌入。

### 关键设计

1. **辅助提示构造（Auxiliary Prompt）**: 设计模板 "The irrelevant information of this sentence: '[TEXT]' means in one word:" 引导 LLM 关注句子中的非核心信息并将其编码到最后 token。辅助提示仅需传播到低层（第 5\~7 层），计算开销极小。论文还探索了 "redundant information"、"background"、"descriptive term" 等变体，结果表明只要语义指向"非核心信息"，效果均稳定提升，方法对辅助提示措辞不敏感。

2. **对比激活导向（Contrastive Activation Steering）**: 在第 $\ell$ 层多头注意力处，提取正常提示和辅助提示最后 token 的 contextualized value vector，计算语义激活向量 $\Delta\mathbf{v}^\ell = \mathbf{v}_{N_\text{nor}}^{\text{nor},(\ell)} - \mathbf{v}_{N_\text{aux}}^{\text{aux},(\ell)}$。该向量是**逐句自适应**的（不同句子生成不同的对比向量），不需要额外监督数据的正负样本对。仅干预最后一个 token 的 value vector，保持其他 token 不变。

3. **范数调整与中间层嵌入（Norm Adjustment + Intermediate Embedding）**: 干预后向量范数可能显著变化，提出两种调整策略——**Norm Scaling (NS)**: $\hat{\mathbf{v}} = \alpha \cdot \Delta\mathbf{v}^\ell$，通过缩放因子 $\alpha$ 控制干预强度（最优值 2\~3）；**Norm Recovering (NR)**: $\hat{\mathbf{v}} = \Delta\mathbf{v}^\ell \cdot \frac{\|\mathbf{v}^{\text{nor}}\|_2}{\|\Delta\mathbf{v}^\ell\|_2}$，恢复原始范数以维持模型稳定性。此外采用中间层（而非最后一层）输出作为嵌入，进一步提升质量并节省计算。

### 即插即用特性

CP 是纯推理时干预，可与 PromptEOL、Pretended CoT、Knowledge、MetaEOL 等任意提示方法**无缝组合**，无需修改模型参数或训练流程。对于多提示方法（如 CK = CoT + Knowledge 平均），辅助提示只需传播一次即可同时优化所有正常提示。

## 实验关键数据

### STS 基准（LLaMA2-7B，7 任务平均 Spearman×100）

| 方法 | Avg. (原始) | +CP-NS | +CP-NR | 提升幅度 |
|------|------------|--------|--------|---------|
| PromptEOL | 70.03 | **75.27** | 75.20 | **+5.24** |
| Pretended CoT | 76.86 | **77.45** | 77.45 | +0.59 |
| Knowledge | 77.14 | **77.56** | 77.40 | +0.42 |
| CK (CoT+Know) | 78.23 | **78.68** | 78.60 | +0.45 |

### 跨模型泛化（Pretended CoT + CP-NS）

| Backbone | Avg. (原始) | +CP-NS | 提升 |
|----------|------------|--------|------|
| LLaMA2-7B | 76.86 | **77.45** | +0.59 |
| LLaMA2-13B | 73.34 | **73.91** | +0.57 |
| LLaMA3.1-8B | 74.07 | **75.22** | +1.15 |

### 下游分类任务（LLaMA2-7B，PromptEOL + CP-NS）

| 任务 | 原始 | +CP-NS | 变化 |
|------|------|--------|------|
| SUBJ | 96.32 | **96.97** | +0.65 |
| TREC | 95.40 | **97.00** | +1.60 |
| MRPC | 75.19 | **77.51** | +2.32 |
| SST2 | 95.00 | **95.94** | +0.94 |
| 7 任务 Avg. | 90.94 | **91.73** | +0.79 |

### 计算开销（前向传播层数，LLaMA2-7B 共 32 层）

| 方法 | 无 CP | 有 CP | 额外开销 |
|------|-------|-------|---------|
| PromptEOL | 27 层 (1×) | 31 层 (1.15×) | +15% |
| Knowledge | 31 层 (1.15×) | 37 层 (1.37×) | +19% |
| CK (双提示) | 54 层 (2×) | 60 层 (2.22×) | +11% |

### 关键消融发现

- **干预位置**: 注意力头 > Transformer 层输出 > FFN 输出，注意力头最优（STS-B dev 82.61 vs 81.93）
- **干预层**: PromptEOL 最优第 5 层，CoT/Knowledge 最优第 7 层
- **缩放因子 $\alpha$**: PromptEOL 最优 2，CoT/Knowledge 最优 3，过大过小均降性能
- **解码概率验证**: CP 后 top-1 预测 token 从停用词（"It", "Don"）变为语义关键词（"Dec", "Throw"）

## 亮点与洞察

- "语义减法"思路极简优雅——辅助提示捕获噪声激活，正常提示减去噪声 = 纯语义信号
- 逐句自适应生成对比向量，比传统 activation steering（需全局正负样本对）更灵活
- PromptEOL 提升最大（+5.24），因为其原始提示最简单、非核心信息最多，CP 的增益与基线方法的"弱点"成正比
- 辅助提示只传播到低层（5\~7 层），额外计算开销仅 11%\~19%

## 局限与展望

- 方法设计假设辅助提示能捕获"非核心信息"，但对何为"核心"无理论保证，效果依赖辅助提示与正常提示的语义互补性
- 仅在英语 STS 和分类任务上评估，跨语言效果未验证
- 辅助提示设计虽然鲁棒但仍有一定主观性，指向"情感"或"实体"的提示反而降性能
- 干预层和缩放因子需在验证集上网格搜索，不同 prompt 的最优超参不同
- 未与微调方法（SimCSE-LLM 等）在同等规模下对比
- 未探索将 CP 与微调方法结合的可能——CP 作为推理时干预是否能在微调模型上继续叠加增益是开放问题

## 评分

- **新颖性**: ⭐⭐⭐⭐ 辅助提示 + 激活对比的"语义减法"思路简洁新颖，将 activation steering 从需要监督数据推广到无监督逐句自适应
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 种基线方法、3 种 LLM backbone、STS + 分类双维度评估、干预位置/层/缩放因子/辅助提示全面消融
- **写作质量**: ⭐⭐⭐⭐ 动机通过解码概率实验直观展示，方法三步流程清晰，图表丰富
- **实用价值**: ⭐⭐⭐⭐ 即插即用无需训练，开销极小（+15%），可直接集成到生产系统的句子嵌入管道

<!-- RELATED:START -->

## 相关论文

- [Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)
- [Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments](explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)

<!-- RELATED:END -->
