---
title: >-
  [论文解读] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment
description: >-
  [CVPR 2026][信息检索/RAG][多模态检索] 提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10…
tags:
  - "CVPR 2026"
  - "信息检索/RAG"
  - "多模态检索"
  - "查询对齐"
  - "强化学习"
  - "密集检索"
  - "查询重写"
---

# BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment

**会议**: CVPR 2026  
**arXiv**: [2604.07201](https://arxiv.org/abs/2604.07201)  
**代码**: [GitHub](https://github.com/mm-bright/multimodal-reasoning-retrieval)  
**领域**: 多模态检索 / 强化学习  
**关键词**: 多模态检索, 查询对齐, 强化学习, 密集检索, 查询重写

## 一句话总结
提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。

## 研究背景与动机
**领域现状**: 密集检索在纯文本场景已很成熟（BEIR 59.0 nDCG@10），多模态编码器（CLIP、Nomic-Vision、VLM2Vec）也在发展，但在推理密集型多模态检索上表现不佳。

**现有痛点**: MM-BRIGHT 基准揭示了反直觉现象——最佳多模态检索器 Nomic-Vision（27.6）甚至不如最佳纯文本检索器（32.2）。现有方法集中在改进检索器端（更大编码器、对比训练、LLM 重排），但都接受噪声查询作为固定输入。

**核心矛盾**: 瓶颈不在检索器而在查询——原始多模态查询纠缠了图像描述、对话噪声和检索意图，系统性地恶化了嵌入相似度。没有任何视觉编码能力可以补偿查询质量差的问题。

**本文要解决**: 在检索之前重构查询，使其从"噪声多模态输入"变为"检索优化的纯文本查询"。

**切入角度**: 查询端对齐（而非检索器端改进），用 RL 直接优化下游检索质量。

**核心idea**: 多模态检索中的模态鸿沟本质上是查询表示问题而非模型能力问题。FORGE 用 RL 学会"桥接"用户多模态表达和检索器需要的输入。

## 方法详解

### 整体框架

BRIDGE 的判断是：推理密集型多模态检索做不好，瓶颈不在检索器而在查询——原始多模态查询把图像描述、对话噪声和真实检索意图缠在一起，再强的视觉编码也补不回来。于是它把流程拆成三段：GPT-4o 先把查询图像转成文本描述 $\delta(q_v)$，FORGE 再把噪声查询对 $(q_t, \delta(q_v))$ 蒸馏成一条紧凑的检索字符串 $\hat{q}$，最后 LENS 编码 $\hat{q}$ 从纯文本语料库召回，整条链推理时完全在文本空间运行：

$$\hat{\mathcal{D}}_k = \text{LENS}(\text{FORGE}(q_t, \text{GPT-4o}(q_v)), \mathcal{C}, k)$$

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["噪声多模态查询<br/>文本问题 q_t + 查询图像 q_v"] --> B["视觉接地（Visual Captioning）<br/>GPT-4o 离线生成图像描述 δ(q_v)"]
    B --> C["FORGE 查询对齐<br/>Qwen2.5-7B + GRPO，以检索 nDCG 为奖励<br/>蒸馏出检索优化字符串 q̂"]
    C --> D["LENS 推理检索器<br/>Qwen3-Embedding-4B 双编码器<br/>按 cosine 相似度召回 top-k"]
    D --> E["纯文本语料库召回结果<br/>推理时全程在文本空间"]
```

### 关键设计

**1. 视觉接地（Visual Captioning）：把视觉内容翻译成纯文本，让文本模型也能用上图像信息**

纯文本检索器本身碰不到像素，是整条链能在文本空间运行的前提。BRIDGE 让 GPT-4o 为查询图像生成密集、领域感知的描述 $\delta(q_v)$，捕捉对象类型、空间关系、标签等，并离线一次性生成、缓存复用。视觉内容一旦被翻译成自然语言，后续 FORGE 与 LENS 就能在纯文本空间里完整处理，推理时无需任何多模态编码器。

**2. FORGE：用 RL 直接拿检索结果当奖励来重写查询**

监督式查询改写只会模仿参考改写，并不知道改完检索效果好不好。FORGE 基于 Qwen2.5-7B-Instruct 微调，输入拼接的文本问题 $q_t$ + 图像描述 $\delta(q_v)$，输出不超过 200 词的检索优化字符串 $\hat{q}$，训练用 GRPO，奖励直接就是下游检索质量 $r(\hat{q}, d^+) = \text{nDCG@}k(\text{LENS}(\hat{q}, \mathcal{C}), \{d^+\})$。每步采样 $G=8$ 个候选查询，算各自的检索奖励再做 GRPO 梯度更新。因为优化目标是"召回对不对"而非"像不像参考"，模型可以自由探索最有利于检索的表达方式，这也是它能胜过启发式改写的原因。

**3. LENS：配一个推理能力强的检索器去接住 FORGE 的结构化查询**

FORGE 产出的是意图丰富、结构化的查询，普通编码器接不住。LENS 是基于 Qwen3-Embedding-4B 的双编码器密集检索器，在数学、科学、医学、法律、软件工程等推理密集型数据上微调，用 InfoNCE 损失 + 批内负样本 + $M=7$ 个硬负样本训练，检索时按 cosine 相似度打分 $\text{score}(\hat{q}, d_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{d_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{d_i}\|}$。查询重写和检索器双方都对齐到"推理意图"上，整条链才咬合得上。

### 损失函数 / 训练策略
- FORGE: GRPO 训练，lr=$1\times10^{-6}$，max 256 tokens，3 epochs
- LENS: 对比学习，lr=$1\times10^{-5}$，batch 512，$\tau=0.02$，3 epochs
- 4× H100 80GB 训练

## 实验关键数据

### 主实验（MM-BRIGHT，2803 查询，29 领域）

| 方法 | nDCG@10 | 类型 |
|------|---------|------|
| CLIP | 10.8 | 多模态编码器 |
| Nomic-Vision | 27.6 | 多模态编码器（最佳） |
| Stella-400M (text) | 32.2 | 纯文本检索器（最佳） |
| BRIDGE (FORGE+LENS) | **29.7** | 查询对齐系统 |
| FORGE + Nomic-Vision | **33.3** | 插件模式 |

### 消融实验

| 配置 | nDCG@10 | 说明 |
|------|---------|------|
| LENS only (原始查询) | 较低 | 噪声查询限制检索器 |
| FORGE + 通用检索器 | 中等 | FORGE 对齐有效但检索器也重要 |
| FORGE + LENS | **29.7** | 最优组合 |
| FORGE + Nomic-Vision | **33.3** | 证明 FORGE 是通用插件 |
| GPT-4o 查询改写 (非 RL) | 较低 | RL 训练优于启发式改写 |

### 关键发现
- FORGE 作为插件将 Nomic-Vision 从 27.6 提升到 33.3（+5.7），首次让多模态系统超越最佳纯文本检索器
- 在 29 个领域中 BRIDGE 全面超越所有多模态编码器基线
- 推理时无需多模态编码器——完全在文本空间操作，轻量、模块化、可扩展
- 验证了核心论点：多模态检索的瓶颈是查询表示而非模型能力

## 亮点与洞察
- 核心洞察极其深刻——"修复查询而非增强检索器"颠覆了常规思路
- FORGE 的 RL 训练方式让查询优化直接面向检索结果，避免了中间监督的误差
- 作为即插即用模块与任意检索器兼容，实用性极强
- 证明了在某些场景下"理解图像内容"不如"理解检索意图"重要

## 局限与展望
- 依赖 GPT-4o 进行图像描述，引入较高的 API 成本和延迟
- FORGE 基于 7B 模型，推理开销大于直接编码查询
- 视觉描述可能丢失细粒度视觉信息（如精确的 UI 布局）
- 可探索轻量化 FORGE 或端到端多模态查询编码器

## 相关工作与启发
- DeepRetrieval 开创了 RL 查询生成，FORGE 将其扩展到多模态场景
- HyDE、Query2Doc 等查询扩展方法用伪文档产生，FORGE 用 RL 奖励指导
- MM-BRIGHT 基准揭示了多模态检索的根本挑战，本文给出了第一个有效回应
- 启示：在很多 AI 系统中，"输入质量"可能比"模型能力"更是瓶颈

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "查询而非检索器是瓶颈"的洞察深刻，RL 训练查询对齐方案新颖
- 实验充分度: ⭐⭐⭐⭐ 29 领域全面评测，插件模式验证通用性，但缺少更多检索器组合
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，系统设计逻辑完整
- 价值: ⭐⭐⭐⭐⭐ 为多模态检索提供了新范式，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](../../ACL2026/information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[ACL 2026\] Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](../../ACL2026/information_retrieval/verbal-r3_verbal_reranker_as_the_missing_bridge_between_retrieval_and_reasoning.md)
- [\[ICLR 2026\] LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](../../ICLR2026/information_retrieval/lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
