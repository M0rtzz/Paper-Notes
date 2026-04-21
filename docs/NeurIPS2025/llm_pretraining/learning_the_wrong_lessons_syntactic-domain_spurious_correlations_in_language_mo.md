---
title: >-
  [论文解读] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models
description: >-
  [NeurIPS 2025][spurious correlations] 揭示 LLM 学会了句法模板（PoS n-gram）与领域之间的虚假关联，导致跨域性能骤降，甚至可利用此关联绕过安全拒绝机制（refusal bypass），在 OLMo-2 上将拒绝率从 40% 降至 2.5%。
tags:
  - NeurIPS 2025
  - spurious correlations
  - syntactic templates
  - LLM 安全
  - 越狱攻击
  - 指令微调
  - 领域泛化
---

# Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.21155](https://arxiv.org/abs/2509.21155)  
**代码**: 待确认  
**领域**: llm_nlp / ai_safety  
**关键词**: spurious correlations, syntactic templates, LLM 安全, 越狱攻击, 指令微调, 领域泛化

## 一句话总结

揭示 LLM 学会了句法模板（PoS n-gram）与领域之间的虚假关联，导致跨域性能骤降，甚至可利用此关联绕过安全拒绝机制（refusal bypass），在 OLMo-2 上将拒绝率从 40% 降至 2.5%。

## 研究背景与动机

LLM 被广泛部署于医疗、金融等领域，其可靠性要求模型真正理解指令的**语义**和**领域**知识。然而近期研究发现，训练数据中大量存在的**句法模板**（syntactic templates，即频繁出现的 PoS tag n-gram）会被模型学习并在输出中重复。

这引出一个关键问题：**LLM 是否真正利用语义和领域知识，还是仅在利用浅层句法模式？**

作者通过一个典型例子阐明：当问 "Where is Paris located?" 时模型正确回答 "France"。但当把句子的语义完全打破（如 "Quickly sit Paris clouded?"），只要保持相同的 PoS 模板 (ADV VB NNP VBD)，模型**仍然回答 "France"**。这说明模型过度依赖句法-领域关联而非语义理解。

更危险的是，这种依赖可被利用来绕过模型的安全拒绝机制。

## 方法详解

### 整体框架

作者形式化了句法模板（syntactic template）、领域（domain）和语义（semantics）三者的交互关系，提出了**句法-领域虚假关联**的定义和检测框架。

### 关键定义

每个样本表示为三元组 $x = (d, t, e)$，其中 $d$ 为领域、$t$ 为句法模板（PoS tag 序列）、$e$ 为实体（语义内容）。

**虚假关联定义**：句法模板 $\tau$ 在领域 $d$ 中是虚假预测的，当其条件频率显著大于边际频率：

$$P(\tau | d) \gg P(\tau)$$

### 五种提示扰动

为量化模型对句法的依赖程度，设计五种提示变体：

1. **Exact**：原始训练时的精确模板
2. **Synonym**：语义保持 + 句法保持（替换同义词）
3. **Antonym**：语义破坏 + 句法保持（替换反义词）
4. **Disfluent**：语义破坏 + 句法保持（插入随机同 PoS 词）
5. **Paraphrase**：语义保持 + 句法改变（重新措辞）

分为三类：
- $\mathcal{P}_{\text{Semantic Preserving}}$: Exact, Synonym
- $\mathcal{P}_{\text{Semantic Breaking}}$: Antonym, Disfluent
- $\mathcal{P}_{\text{Utility}}$: Paraphrase

### 句法-领域依赖量化

定义领域风险：

$$R_{M_\theta}(d) = \mathbb{E}_{(p,t,e) \sim X_d}\left(\mathbb{E}_{p^- \sim \mathcal{P}_{\text{SB}}} M(e|p^-) + \mathbb{E}_{p^+ \sim \mathcal{P}_{\text{SP}}} M(e|p^+)\right)$$

两个判定条件：(1) 域内语义保持提示上高性能；(2) 域内/跨域风险之间存在大 gap。

### 检测框架（3 步）

1. **模板提取**：从训练数据（如 FlanV2）中提取 PoS 模板
2. **测试集构建**：对每个实体-模板配对生成 5 种扰动，共 $n \times m \times 4$ 个提示
3. **相关性度量**：将模型正确预测的模板划为"域内"，其余为"跨域"，比较性能 gap

### 行为分类学（Taxonomy）

定义 6 种指令执行行为模式：
- **正确行为**：Exact/Synonym/Paraphrase 高，Antonym/Disfluent 低
- **实体记忆**：所有设置、所有域均高
- **提示记忆**：仅 Exact 高
- **词-领域关联**：Exact/Synonym 域内高，Antonym/Disfluent 域内低
- **句法-领域关联**：Antonym/Disfluent 域内也高（关键区分点）

### 安全影响：拒绝绕过

利用句法-领域关联绕过安全机制：将有害请求的 PoS 模板替换为"安全"领域（如 chain-of-thought）的模板作为前缀/后缀。

## 实验关键数据

### 主实验：合成数据训练 OLMo-2

| 模型 | 设置 | Exact | Synonym | Antonym | Disfluent | Paraphrase |
|------|------|-------|---------|---------|-----------|------------|
| OLMo-2-1B Instruct | 域内 | 0.93 | 0.91 | 0.90 | 0.24 | 0.53 |
| | 跨域 | 0.42 | 0.40 | 0.41 | 0.25 | 0.44 |
| | **Δ** | **↓0.51** | **↓0.51** | **↓0.49** | ↑0.01 | ↓0.09 |
| OLMo-2-13B Instruct | 域内 | 0.94 | 0.93 | 0.93 | 0.13 | 0.84 |
| | 跨域 | 0.40 | 0.42 | 0.56 | 0.24 | 0.50 |
| | **Δ** | **↓0.54** | **↓0.51** | **↓0.37** | ↑0.11 | ↓0.34 |

关键发现：跨域性能下降约 **0.40–0.60**，且模型规模和指令微调均无法缓解。Antonym 设置域内高达 0.93 证明句法可覆盖语义。

### 真实模型验证（FlanV2 Sentiment140）

| 模型 | Synonym 域内→跨域 | Δ |
|------|-------------------|---|
| OLMo-2-7B | 0.85→0.48 | ↓0.37 |
| GPT-4o-mini | 1.00→0.44 | **↓0.56** |
| GPT-4o | 0.69→0.36 | ↓0.33 |

开源和闭源模型均存在句法-领域虚假关联。

### 安全实验：拒绝绕过

| 模板 | 位置 | Exact 拒绝率 | Synonym | Antonym |
|------|------|-------------|---------|---------|
| 基线 | — | 0.400 | 0.400 | 0.400 |
| CoT | 前缀 | **0.025** | 0.357 | 0.195 |
| CoT | 后缀 | 0.129 | 0.382 | 0.259 |
| Math | 前缀 | 0.481 | 0.251 | 0.189 |

使用 chain-of-thought 模板作为前缀，OLMo-2-7B-Instruct 的拒绝率从 40% **骤降至 2.5%**。

### 关键发现

1. 句法可覆盖语义：Antonym（反义词）设置域内性能与 Synonym 相当
2. 模型规模无法解决：1B→13B 跨域下降幅度相似
3. Llama-4-Maverick 表现不同：跨域下降小，但表现为实体记忆而非句法-领域关联
4. 安全影响严重：CoT 模板前缀可将拒绝率降低 37.5 个百分点

## 亮点与洞察

1. **揭示新型虚假关联**：句法-领域关联是此前未被识别的 LLM 失败模式，与 CV 中背景-标签关联类似但更隐蔽
2. **严谨的形式化框架**：三元组表示 + 五种扰动 + 六种行为分类学，为后续研究提供标准化评估方案
3. **从理论到安全**：不仅是学术发现，直接转化为实际安全漏洞（jailbreak），影响力远超一般分析工作
4. **覆盖开源+闭源**：在 OLMo-2、Llama-4、GPT-4o 上均验证了现象的普遍性
5. **可操作的建议**：(1) 测试句法-领域关联；(2) 确保领域内句法多样性

## 局限性

1. **闭源模型推测性**：无法确认 GPT-4o/Llama-4 是否在 FlanV2 上训练，跨域下降可能有其他原因
2. **未覆盖推理模型**：CoT 推理模型可能有不同的句法依赖模式
3. **模板粒度有限**：仅使用 PoS tag 级别的模板，更细粒度的句法结构未探索
4. **合成数据简化假设**：假设领域间模板集无重叠，实际中可能有显著重叠
5. **防御方案未充分探索**：仅提出"增加句法多样性"的建议，缺乏具体实现和验证

## 相关工作与启发

- 与 NLP 中经典虚假关联工作（McCoy et al. 2019）的区别：本文关注 PoS 模板而非词汇层面，且覆盖了多轮训练后的大模型
- 与 Shaib et al. 2024 的联系：后者发现 LLM 学习并重复句法模板，本文进一步证明这些模板与领域形成虚假关联
- 对数据策划的启示：指令微调数据集（如 Flan）应确保每个领域使用多样化的模板格式
- 对 jailbreak 研究的影响：提供了一种基于训练数据结构的系统性攻击方法

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首次系统性揭示句法-领域虚假关联，形式化框架完整
- **实用性**: ⭐⭐⭐⭐ — 提供检测框架和安全启示，但防御方案有待完善
- **实验严谨度**: ⭐⭐⭐⭐ — 合成+真实数据双重验证，多模型覆盖，但闭源模型结论有局限
- **写作质量**: ⭐⭐⭐⭐ — 例子直观，框架清晰，安全案例有冲击力
- **推荐阅读指数**: ⭐⭐⭐⭐⭐ — LLM 安全/鲁棒性/指令微调研究者必读

<!-- RELATED:START -->

## 相关论文

- [Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](retrospective_incontext_learning_for_temporal_credit_assignm.md)
- [Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)
- [Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)

<!-- RELATED:END -->
