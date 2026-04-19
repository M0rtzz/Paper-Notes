---
title: >-
  [论文解读] SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence
description: >-
  [LLM/NLP] SelfElicit 发现 LLM 深层注意力分数天然具有定位上下文中关键证据的能力（即使模型回答错误时也是如此），据此提出一种推理时的上下文增强方法：仅需生成一个额外 token 即可自动识别并高亮关键证据句，引导模型给出更准确的回答。
tags:
  - LLM/NLP
---

# SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2502.08767](https://arxiv.org/abs/2502.08767) |
| 代码 | [ZhiningLiu1998/SelfElicit](https://github.com/ZhiningLiu1998/SelfElicit) |
| 领域 | 上下文增强、QA |
| 关键词 | 注意力分析、证据定位、上下文高亮、推理时增强、无训练 |

## 一句话总结

SelfElicit 发现 LLM 深层注意力分数天然具有定位上下文中关键证据的能力（即使模型回答错误时也是如此），据此提出一种推理时的上下文增强方法：仅需生成一个额外 token 即可自动识别并高亮关键证据句，引导模型给出更准确的回答。

## 研究背景与动机

- **上下文证据利用不充分**：尽管为 LLM 提供包含证据的上下文可以显著提升回答质量，但近期研究发现 LLM 在上下文包含噪声和无关信息时难以充分利用关键证据，即使证据就在输入中也可能给出错误答案。
- **现有方法的不足**：改进的 prompting 和 decoding 方法将整个上下文视为单一实体处理，忽略了并非所有上下文信息都同等重要的事实。
- **核心发现**：通过分析多个 LM 家族在生成首个 token 时各层的注意力分布，作者发现**深层注意力对证据句的关注度显著高于非证据句**（高达 6 倍），且这一规律在模型回答正确和错误时都成立。这表明 LM 内部已经具备证据定位能力，只是未被有效利用。

## 方法详解

### 整体框架

SelfElicit 分两步：(1) **证据发现**：利用 LM 深层注意力分数自动定位上下文中的关键证据句；(2) **证据高亮**：在原始上下文中用文本标记高亮证据句，并修改 prompt 模板引导模型关注高亮内容后重新生成答案。

### 关键设计

1. **句子级注意力聚合**：对于输入序列中的 $m$ 个上下文句子，计算每层 $\ell$ 的句子级注意力 $\bar{a}_i^{(\ell)}$（对句子内所有 token 的平均注意力），提供每个句子在各层的相对重要性。
2. **证据阅读层选择**：选取后 50% 的层作为"证据阅读层" $\mathcal{L}_{ER}$，聚合这些层的句子级注意力得到证据分数 $e_i = \frac{1}{|\mathcal{L}_{ER}|}\sum_{\ell \in \mathcal{L}_{ER}} \bar{a}_i^{(\ell)}$。
3. **阈值化证据选择**：引入阈值参数 $\alpha \in [0,1]$（默认 0.5），选择证据分数超过最大值 $\alpha$ 倍的句子：$\mathcal{S}_{SE} = \{s_i | e_i \geq \alpha \cdot \max(\mathbf{e})\}$。
4. **文本标记高亮**：在选中的证据句前后插入 `<start_important>` 和 `<end_important>` 标记，同时更新 prompt 模板引导模型关注高亮信息。

### 损失函数

SelfElicit 是纯推理时方法，**无需训练**，不涉及损失函数。唯一的额外计算开销是生成一个 token 以获取注意力分数。

## 实验

### 主实验：6 个 LM × 4 个 QA 任务

| 模型 | 方法 | HotpotQA EM | NewsQA EM | TQA EM | NQ EM | 推理时间(ms) |
|------|------|------------|----------|--------|------|-----------|
| Llama-3.1-8B | Base | 58.9 | 64.3 | 72.8 | 59.7 | 224.1 |
| | CoT | 60.4 | 64.9 | 74.4 | 59.6 | 224.8 |
| | FullElicit | 60.7 | 65.9 | 72.8 | 61.1 | 226.3 |
| | PromptElicit | 66.3 | 62.8 | 76.0 | 61.8 | 1672.0 |
| | **SelfElicit** | **68.5** | **66.9** | **79.4** | **64.0** | **264.1** |
| Llama-3.1-70B | Base | 71.8 | 66.7 | 78.0 | 59.3 | 1389.8 |
| | **SelfElicit** | — | — | — | — | — |

SelfElicit 在所有模型-数据集组合上均取得最佳或接近最佳的 EM 和 Token F1，同时推理时间开销极小（仅增加约 18% vs Base）。

### 消融实验：设计选择分析

| 消融项 | 影响 |
|--------|------|
| 证据阅读层选择 | 后 50% 层一致最优，前 50% 层效果差 |
| 阈值 α 选择 | α=0.5 在所有模型和任务上稳定表现良好，鲁棒性强 |
| Token 级 vs 句子级高亮 | 句子级语义更完整、效果更好 |
| 高亮方式（标记 vs 加粗 vs 删除非证据） | 文本标记方式最优 |

### 关键发现

- **深层注意力天然定位证据**：跨越 Llama、Mistral、Qwen 等多个模型家族，深层注意力对证据句的关注度一致显著高于非证据句，即使模型回答错误时也是如此
- **效率极高**：仅需额外生成 1 个 token 获取注意力分数，相比 PromptElicit（需要 LLM 先提取证据再回答）快约 6 倍
- **对噪声鲁棒**：在上下文严重受噪时（大量无关段落），SelfElicit 仍能稳定定位证据并提升性能
- **证据发现精度**：在 HotpotQA 上，SelfElicit 的证据发现准确率（recall of supporting facts）在多数模型上超过 70%

## 论文亮点

- 核心发现极具启发性：LM 深层注意力天然具备证据定位能力，与回答正确与否无关
- 方法极简且高效：无训练、无迭代、仅 1 个额外 token 开销
- 泛化性强：在 6 个模型家族、4 个 QA 数据集上一致有效
- 对高亮方式的系统性消融分析为后续工作提供了有价值的设计指导

## 局限性

- 阈值 $\alpha$ 虽然对性能影响较小但仍需预设，未实现完全自适应
- 主要验证在开放式 QA 上，对其他 NLG 任务（摘要、对话）的效果未探索
- 证据高亮依赖句子分割质量，对结构不规则的文本可能效果下降
- 假设上下文中确实包含相关证据，对完全无证据的场景（如需外部检索）不适用

## 相关工作

- **上下文增强 QA**：Context-Aware Decoding (Shi et al., 2024) 将整个上下文视为单一实体增强
- **KV Cache 压缩与注意力分析**：H2O (Li et al., 2024a) 利用注意力模式进行 KV cache 压缩
- **检索增强生成（RAG）**：RAG 方法关注如何向 LLM 提供相关证据，SelfElicit 则关注如何更好地利用已有证据

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Improve Language Model and Brain Alignment via Associative Memory](improve_language_model_and_brain_alignment_via_associative_memory.md)
- [InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model](infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)
- [SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](songcomposer_llm_lyric_melody_generation.md)
- [A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)

<!-- RELATED:END -->
