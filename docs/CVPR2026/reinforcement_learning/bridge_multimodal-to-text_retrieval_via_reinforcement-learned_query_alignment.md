---
title: >-
  [论文解读] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment
description: >-
  [CVPR 2026][多模态检索] 提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。
tags:
  - CVPR 2026
  - 多模态检索
  - 查询对齐
  - 强化学习
  - 密集检索
  - 查询重写
---

# BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment

**会议**: CVPR 2026  
**arXiv**: [2604.07201](https://arxiv.org/abs/2604.07201)  
**代码**: [GitHub](https://github.com/mm-bright/multimodal-reasoning-retrieval) (有)  
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
三阶段管线: (1) GPT-4o 将查询图像转为文本描述 $\delta(q_v)$ → (2) FORGE 将噪声查询对 $(q_t, \delta(q_v))$ 蒸馏为紧凑检索字符串 $\hat{q}$ → (3) LENS 编码 $\hat{q}$ 并从文本语料库检索。

$$\hat{\mathcal{D}}_k = \text{LENS}(\text{FORGE}(q_t, \text{GPT-4o}(q_v)), \mathcal{C}, k)$$

### 关键设计

1. **FORGE（Focused Retrieval Query Generator）**:

    - 基于 Qwen2.5-7B-Instruct 微调
    - 输入: 拼接的文本问题 + 图像描述；输出: 不超过 200 词的检索优化搜索字符串
    - 用 GRPO 强化学习训练，奖励函数为下游检索质量：
    $r(\hat{q}, d^+) = \text{nDCG@}k(\text{LENS}(\hat{q}, \mathcal{C}), \{d^+\})$
    - 训练循环: 采样 $G=8$ 个候选查询 → 计算检索奖励 → GRPO 梯度更新
    - **设计动机**: 与监督式查询重写不同，RL 直接优化检索结果而非模仿参考改写，让模型自由探索最优查询策略

2. **LENS（Language-Enhanced Neural Search）**:

    - 基于 Qwen3-Embedding-4B 的双编码器密集检索器
    - 在推理密集型检索数据上微调（数学、科学、医学、法律、软件工程）
    - InfoNCE 损失 + 批内负样本 + $M=7$ 个硬负样本
    - cosine 相似度检索: $\text{score}(\hat{q}, d_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{d_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{d_i}\|}$
    - **设计动机**: FORGE 产生的是意图丰富的结构化查询，需要推理能力强的检索器来匹配

3. **Visual Captioning（视觉描述）**:

    - GPT-4o 生成密集领域感知描述，捕捉对象类型、空间关系、标签
    - 离线一次性生成并缓存
    - **设计动机**: 将视觉内容接地为自然语言，让纯文本模型可处理

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

## 相关论文

- [Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](msrl_scaling_generative_multimodal_reward_modeling.md)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](../../ICLR2026/reinforcement_learning/longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)
- [Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)

<!-- RELATED:END -->
