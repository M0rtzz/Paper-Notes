---
title: >-
  [论文解读] GeAR: Generation Augmented Retrieval
description: >-
  [ACL 2025][document retrieval] GeAR 在传统 bi-encoder 检索框架上引入融合编码器和文本解码器，通过生成任务增强检索模型对文档内部细粒度语义的理解能力，同时不增加全局检索的计算开销。
tags:
  - "ACL 2025"
  - "document retrieval"
  - "fine-grained retrieval"
  - "generation-augmented"
  - "bi-encoder"
  - "对比学习"
---

# GeAR: Generation Augmented Retrieval

**会议**: ACL 2025  
**arXiv**: [2501.02772](https://arxiv.org/abs/2501.02772)  
**代码**: [有](https://github.com/microsoft/LMOps)  
**领域**: 其他  
**关键词**: document retrieval, fine-grained retrieval, generation-augmented, bi-encoder, contrastive learning

## 一句话总结

GeAR 在传统 bi-encoder 检索框架上引入融合编码器和文本解码器，通过生成任务增强检索模型对文档内部细粒度语义的理解能力，同时不增加全局检索的计算开销。

## 研究背景与动机

当前文档检索的主流方法是 bi-encoder：将查询和文档分别编码为向量，通过余弦相似度计算匹配分数。这种方法存在几个根本性问题：

**信息表达瓶颈**：查询与文档之间复杂的语义关系被压缩成一个标量相似度，难以提供可解释的检索结果

**粗粒度匹配**：对于长文档（256-512+ tokens），bi-encoder 只能给出整体匹配分数，无法定位文档中与查询最相关的具体段落或句子

**下游任务需求**：许多 NLP 任务（句子选择、搜索结果高亮、细粒度引用、needle-in-a-haystack）需要检索模型具备局部细粒度理解能力，而传统方法只优化全局语义对齐

作者提出一个核心问题：**如何让检索器同时具备全局理解和局部理解的双重能力？**

实现这个目标面临两大挑战：（1）缺乏足够的（查询-文档-局部信息）三元组训练数据；（2）模型架构、训练目标和训练方法的有效组合尚未被充分探索。

## 方法详解

### 整体框架

GeAR 的架构受多模态表示学习（如 ALBEF、BLIP）启发，将文档和查询视为两种"模态"，包含三个核心组件：

- **Bi-Encoder**：独立编码查询和文档，用于全局文档检索（零额外开销）
- **Fusion Encoder**：通过交叉注意力将文档嵌入与查询嵌入深度融合
- **Text Decoder**：基于融合表示，生成文档中与查询相关的细粒度信息

### 关键设计

1. **Bi-Encoder（全局检索）**：初始化两个独立编码器 $E_d(\cdot)$ 和 $E_q(\cdot)$，分别编码文档和查询，使用 mean pooling 获取文本嵌入。推理时直接计算双编码器相似度，与经典检索方法完全一致，不引入额外计算成本。设计动机是保持检索效率的同时增加局部理解能力。

2. **Fusion Encoder（局部信息交互）**：与查询编码器共享大部分参数，但在每层添加轻量级可学习交叉注意力模块。文档嵌入通过交叉注意力与查询嵌入融合。核心思想是让文档中每个 token 都能感知查询意图，从而产生有意义的注意力权重分布。这些交叉注意力权重可以直接用于排序文档内的句子，实现局部信息检索。

3. **Text Decoder（生成任务增强）**：接收融合编码器的输出表示，使用单向因果注意力自回归生成与查询相关的细粒度文本。在 QA 场景中生成答案，在相关信息检索场景中生成文档内最相关的句子。引入特殊 `[Decode]` token 标记生成起始位置。设计动机是通过生成任务显式建模查询与文档细粒度语义片段之间的对齐关系。

4. **数据构建**：定义了两种检索场景：

    - 问答检索（QAR）：从 PAQ 数据集抽取 30M 训练数据
    - 相关信息检索（RIR）：构建 LLM 辅助的数据合成流水线，从高质量 Wikipedia 文档中采样句子并改写为查询，经去重和相关性过滤后获得 5.8M 三元组

### 损失函数 / 训练策略

GeAR 的总损失是两部分的加权和：

$$\mathcal{L}_{\text{GeAR}} = \mathcal{L}_{\text{CL}} + \alpha \cdot \mathcal{L}_{\text{LM}}$$

- **对比学习损失 $\mathcal{L}_{\text{CL}}$**：优化查询与文档之间的语义相似度。引入动量 Bi-Encoder（类似 MoCo/BLIP）提供更丰富的软标签监督信号。
- **语言建模损失 $\mathcal{L}_{\text{LM}}$**：通过交叉熵损失优化生成任务，最大化真实文本的生成概率。这是增强局部信息检索能力的关键。
- 权重 $\alpha = 0.25$ 效果最优，即生成任务作为辅助任务，权重较低。

训练使用 bert-base-uncased 初始化编码器，解码器 110M 参数随机初始化，在 16 张 AMD MI200 GPU 上训练 10 个 epoch。

## 实验关键数据

### 主实验——全局文档检索（表格）

| 方法 | SQuAD R@5 | NQ R@5 | TriviaQA R@5 | PAQ R@5 | RIR R@5 |
|------|-----------|--------|-------------|---------|---------|
| BGE (预训练) | 0.829 | 0.674 | 0.690 | 0.752 | 0.451 |
| GTE (预训练) | 0.866 | **0.767** | **0.726** | 0.836 | 0.528 |
| BGERT (重训) | 0.841 | 0.751 | 0.640 | 0.901 | 0.953 |
| **GeAR** | **0.887** | 0.762 | 0.664 | **0.952** | **0.964** |

GeAR 在 SQuAD、PAQ、RIR 上达到 SOTA，相比 BGERT 平均 Recall@5 相对提升 3.5%。

### 主实验——局部信息检索（表格）

| 方法 | SQuAD R@1 | NQ R@1 | TriviaQA R@1 | PAQ R@1 | RIR R@3 |
|------|-----------|--------|-------------|---------|---------|
| BGE-Reranker-L | 0.751 | 0.670 | 0.464 | 0.704 | 0.891 |
| **GeAR** | **0.814** | **0.761** | **0.510** | **0.884** | **0.897** |

GeAR 在局部信息检索上相比次优的 BGE-Reranker-L 平均相对提升 12.9%。

### 消融实验（表格）

| LM 损失权重 α | 全局检索 Avg Recall | 局部检索 Avg Recall |
|-------------|-------------------|-------------------|
| 0 | 0.844 | 0.663 |
| 0.25 | **0.846** | 0.781 |
| 0.5 | 0.844 | **0.785** |
| 1.0 | 0.838 | 0.784 |

生成任务对检索性能的影响呈倒 U 型：适度的 LM 损失权重显著提升局部检索（从 0.663 到 0.785），权重过高则会削弱全局检索。

### 关键发现

1. 生成任务不仅增强局部信息检索，还对全局文档检索有微小但一致的正面影响
2. GeAR 的局部检索不需要额外对文档分块重编码，直接利用交叉注意力权重排序句子
3. 融合编码器中间层（第 10 层附近）的交叉注意力权重比最高层更适合局部检索
4. 尽管解码器只有 110M 参数，GeAR 在域内数据上的生成性能令人惊讶地可与 Llama 3.2 3B 相比

## 亮点与洞察

- **核心创新**：将生成任务作为辅助手段增强检索模型的细粒度理解，而非用于替代检索——这种设计哲学非常实用
- **零额外推理开销**：全局检索阶段完全等价于普通 bi-encoder，融合编码器和解码器仅在需要局部信息时启用
- **可视化分析**突破了传统检索的"黑箱"特性：通过高亮交叉注意力权重和生成相关文本，让检索结果从数值变为可解释的

## 局限与展望

1. 上下文长度限制在 512 tokens，未探索长文本检索场景
2. 合成数据的多样性和质量可能不如大规模真实检索数据
3. 解码器仅 110M 参数，无法完成复杂生成任务；未来可探索扩大解码器规模
4. 未在 BEIR 等标准检索基准上评估，泛化性有待进一步验证

## 相关工作与启发

- 关键灵感来自多模态表示学习（CLIP、ALBEF、BLIP），将文档与查询视为两种"模态"进行对齐
- 与 ColBERT 的延迟交互不同，GeAR 通过生成任务而非多向量匹配来捕捉细粒度语义
- 为 RAG 系统中"检索后定位"提供了新思路：检索器自身即可完成局部信息定位，无需额外的 reranker 或 reader

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 用生成任务增强检索的全局-局部双能力是一个新颖且自洽的思路
- **实验充分度**: ⭐⭐⭐⭐ — 两个场景、五个数据集、多种基线和消融，但缺少 BEIR 等标准基准
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，可视化分析直观，但部分数学符号可以更简洁
- **价值**: ⭐⭐⭐⭐ — 对检索系统的可解释性和细粒度能力有实际推动作用，尤其适用于 RAG 场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)

</div>

<!-- RELATED:END -->
