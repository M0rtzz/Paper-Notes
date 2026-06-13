---
title: >-
  [论文解读] Retrofitting Large Language Models with Dynamic Tokenization
description: >-
  [ACL 2025][预训练][动态分词] 本文提出对已有语言模型进行动态分词改造（dynamic tokenization），通过受 BPE 启发的子词合并算法动态决定 token 边界，结合预训练的嵌入预测超网络在线计算合并后 token 的嵌入向量，在 encoder 模型上实现平均 >20% 的序列长度缩减且性能仅下降不到 2%，在 decoder 模型上也实现了最高 17% 的序列缩减。
tags:
  - "ACL 2025"
  - "预训练"
  - "动态分词"
  - "子词合并"
  - "嵌入预测超网络"
  - "推理加速"
  - "跨语言公平"
---

# Retrofitting Large Language Models with Dynamic Tokenization

**会议**: ACL 2025  
**arXiv**: [2411.18553](https://arxiv.org/abs/2411.18553)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 动态分词、子词合并、嵌入预测超网络、推理加速、跨语言公平

## 一句话总结

本文提出对已有语言模型进行动态分词改造（dynamic tokenization），通过受 BPE 启发的子词合并算法动态决定 token 边界，结合预训练的嵌入预测超网络在线计算合并后 token 的嵌入向量，在 encoder 模型上实现平均 >20% 的序列长度缩减且性能仅下降不到 2%，在 decoder 模型上也实现了最高 17% 的序列缩减。

## 研究背景与动机

**领域现状**：当前的语言模型普遍使用固定的、静态子词分词器（如 BPE、WordPiece、SentencePiece）。分词表在预训练前确定，之后不再改变。这种设计虽然简单统一，但在多语言场景中带来了显著的效率和能力差异。

**现有痛点**：静态分词器的词表通常偏向英语，导致其他语言（尤其是形态丰富的语言）的相同语义内容需要更多 token 来表示，造成推理速度慢、上下文窗口利用率低、以及跨语言能力不均衡。例如，同样一句话在德语中可能需要比英语多 50% 的 token。

**核心矛盾**：想要根据不同语言/领域动态调整 token 粒度，但模型的嵌入层是固定大小的查找表，只能处理预定义词表中的 token。动态改变 token 边界意味着会产生训练时从未见过的新 token，模型无法直接处理。

**本文目标**：设计一种方法，使得已有的、已经预训练好的语言模型能够不经过重新训练就接受动态粒度的 token 输入，从而实现更短的序列长度和更公平的跨语言表现。

**切入角度**：作者观察到 BPE 算法本身就是通过频率驱动的子词合并来构建词表的。如果在推理时也按照类似的频率合并策略将相邻子词合并为更大的单元，只需要一种方式来计算这些合并后 token 的嵌入，就可以在不修改模型参数的情况下实现动态分词。

**核心 idea**：训练一个轻量级的嵌入预测超网络（hypernetwork），输入是被合并子词的原始嵌入序列，输出是合并后 token 的嵌入向量；推理时先做动态合并再用超网络算嵌入，然后正常过模型。

## 方法详解

### 整体框架

给定一段输入文本，先用原始静态分词器得到子词序列。然后在batch级别统计子词二元组（bigram）频率，将频率最高的二元组合并，重复此过程直到达到目标序列缩减率。对于合并产生的新 token，使用预训练好的超网络从其成分子词的嵌入中预测出合并后的嵌入。最终将缩短的序列送入原始模型进行正常推理。

### 关键设计

1. **批次级子词合并算法（Batch-Level Subword Merging）**:

    - 功能：在推理时动态决定哪些相邻子词应该合并为更大的 token
    - 核心思路：在一个 batch 内的所有序列上统计所有相邻子词二元组的频率。选择频率最高的二元组作为合并操作，将 batch 中所有该二元组的出现全部合并为一个新 token。迭代执行此过程直到达到预设的合并次数或目标序列缩减比例。这个过程完全是在推理时动态进行的，不需要预定义的固定合并规则
    - 设计动机：频率高的二元组更可能是有意义的语言单元（类似 BPE 的核心假设），在 batch 级别统计而非全局统计使得合并策略能适应当前输入的分布，实现了真正的动态性

2. **嵌入预测超网络（Embedding-Prediction Hypernetwork）**:

    - 功能：为合并产生的新 token 计算嵌入向量
    - 核心思路：超网络是一个小型 Transformer 架构，输入为被合并的 $k$ 个子词的原始嵌入（从模型的嵌入查找表中获取），输出为合并后 token 的嵌入向量（维度与原始嵌入相同）。超网络在训练时使用对比学习/重建目标：让预测的合并嵌入接近该 token 序列在模型中间层产生的上下文化表示。超网络参数量很小（约模型的 1%），预训练后冻结
    - 设计动机：单纯取平均或拼接子词嵌入会丢失顺序和组合信息；超网络可以学习到更复杂的合并模式，例如 "un" + "happy" 合并后的语义不是简单的均值。使用对比学习目标使预测嵌入与模型内部的上下文表示保持一致

3. **Decoder 模型的两种应用模式（Prefilling & ANN Generation）**:

    - 功能：将动态分词扩展到自回归生成模型
    - 核心思路：(a) Prefilling 模式——仅在 prompt 编码阶段做动态合并，缩短 prefill 序列从而加速 KV cache 的初始化，生成阶段仍逐 token 解码；(b) ANN 模式——用近似最近邻索引维护一个百万级别的动态词表，生成时直接从中选取合并后的 token 进行解码，实现生成阶段的加速
    - 设计动机：Decoder 模型的生成是自回归的，不能简单地事后合并。Prefilling 模式是低风险策略（不改变生成质量），ANN 模式更激进但可实现更大加速

### 训练策略

超网络的训练独立于主模型。使用大量文本数据，对每段文本做多种可能的子词合并，然后训练超网络去预测合并后 token 的嵌入，监督信号来自主模型中间层对相应位置的上下文表示。训练目标为 MSE 损失加对比损失的混合。主模型参数始终冻结不变。

## 实验关键数据

### 主实验

Encoder 模型（XLM-R）在 XNLI 14 语言上的平均结果：

| 配置 | 平均序列缩减率 | 平均准确率下降 | 推理加速 |
|------|--------------|--------------|---------|
| 原始静态分词 | 0% | - | 1.0x |
| 动态分词 (word级边界) | 22.3% | 1.7% | ~1.25x |
| 动态分词 (aggressive) | 30.1% | 3.2% | ~1.40x |

Decoder 模型（Mistral-7B）的结果：

| 模式 | 序列缩减率 | 性能变化 | 说明 |
|------|----------|---------|------|
| Prefilling 动态分词 | 最高 40% (相对word级) | 几乎无损 | 仅加速 prefill |
| ANN 生成 (100万词表) | 17% | 轻微下降 | 支持超大动态词表 |

### 消融实验

| 配置 | XNLI 平均准确率 | 说明 |
|------|----------------|------|
| 完整方法 (超网络预测嵌入) | 82.1% | 配合动态合并 |
| 用平均池化替代超网络 | 79.8% | 简单均值不够好 |
| 固定合并规则 (非动态) | 80.5% | 动态优于固定 |
| 无合并 (原始分词) | 83.6% | 原始基线 |

### 关键发现

- 非英语语言从动态分词中获益更多：德语、土耳其语等形态丰富语言的序列缩减率可达 30%+，而英语仅约 15%；这有效缩小了跨语言的 token 数量差距
- 超网络预测嵌入显著优于简单的平均池化（+2.3%），验证了组合语义需要非线性建模
- Prefilling 模式几乎无损，是工程上最安全的部署方案
- 动态合并的收益具有边际递减效应——前 10% 的序列缩减几乎无损，之后每增加 5% 缩减约引入 0.5% 的性能下降

## 亮点与洞察

- **"改造而非重训"的设计哲学非常实用**：不需要重新预训练模型，只需训练一个轻量超网络就能赋予已有模型动态分词能力。这种即插即用的特性大大降低了部署门槛
- **从 BPE 训练灵感到推理时动态合并的类比非常优雅**：BPE 在训练时做静态合并定义词表，本文在推理时做动态合并适应输入，二者形成完美的镜像关系
- **跨语言公平性的视角值得关注**：很多 NLP 效率研究只关注英语场景，本文明确优化了多语言公平性，对全球化部署有直接意义

## 局限与展望

- 批次级统计依赖 batch 内的文本分布，如果 batch 非常小或文本多样性极高，合并策略可能不稳定
- 超网络在合并长度 >4 时性能开始下降，对于某些需要大粒度合并的场景可能受限
- 目前仅验证了 XLM-R 和 Mistral-7B，对于更大规模模型（如 70B+）的效果未知
- ANN 生成模式的工程复杂度较高，实际部署还需优化

## 相关工作与启发

- **vs BPE Dropout (Provilkov et al., 2020)**: BPE Dropout 在训练时引入分词随机性增强鲁棒性，但仍是预定义的静态策略；本文在推理时动态决定分词，更灵活
- **vs Charformer (Tay et al., 2021)**: Charformer 从字符级动态学习 token 表示，但需要从头训练；本文的关键优势是可以改造任何已有模型
- **vs Byte-level Models (Yu et al., 2023)**: 字节级模型完全跳过分词但序列极长；本文在子词级操作实现了效率与灵活性的平衡

## 评分

- 新颖性: ⭐⭐⭐⭐ 动态分词+超网络嵌入预测的组合新颖，"改造已有模型"的定位独特
- 实验充分度: ⭐⭐⭐⭐ 覆盖 encoder/decoder 两种架构和 14 种语言，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述清晰直观，图示和流程讲解到位
- 价值: ⭐⭐⭐⭐ 对多语言效率优化和部署有直接帮助，即插即用的特性增加了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)

</div>

<!-- RELATED:END -->
