---
title: >-
  [论文解读] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP
description: >-
  [ACL 2026][音译] 本文系统综述了音译（transliteration）在跨语言 NLP 中的演变角色，提出五大动机分类（命名实体/OOV处理、代码混合、跨文字相似性利用、英语中心迁移、统一预处理），比较了六种整合方式的优劣，并在现代 LLM 语境下讨论了音译是否仍然必要。
tags:
  - ACL 2026
  - 音译
  - 跨文字迁移
  - 罗马化
  - 脚本壁垒
  - 多语言语言模型
---

# Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP

**会议**: ACL 2026  
**arXiv**: [2604.18722](https://arxiv.org/abs/2604.18722)  
**代码**: 无  
**领域**: 多语言NLP / 综述  
**关键词**: 音译, 跨文字迁移, 罗马化, 脚本壁垒, 多语言语言模型

## 一句话总结

本文系统综述了音译（transliteration）在跨语言 NLP 中的演变角色，提出五大动机分类（命名实体/OOV处理、代码混合、跨文字相似性利用、英语中心迁移、统一预处理），比较了六种整合方式的优劣，并在现代 LLM 语境下讨论了音译是否仍然必要。

## 研究背景与动机

**领域现状**：跨语言迁移是多语言 NLP 的核心驱动力，但"脚本壁垒"——不同书写系统之间缺乏词汇重叠——严重阻碍了语言间的知识迁移。音译作为将一种书写系统转换为另一种的技术，已成为缓解跨脚本不兼容性的实用方案。

**现有痛点**：(1) 音译研究分散在不同动机和方法论之间，缺乏统一的分类框架；(2) 何时音译有益、何时有害（如中文等表意文字转拉丁化会丢失语义信息）尚无系统性总结；(3) 在 LLM 时代，大规模预训练是否已经使音译变得不必要需要重新审视。

**核心矛盾**：音译通过增加词汇重叠来促进迁移，但同时可能丢失语义、语态等重要信息。需要明确在什么条件下音译利大于弊。

**本文目标**：提供一个全面的分类框架，指导研究者根据语言、任务和资源条件选择最适合的音译策略。

**切入角度**：按动机（why）和方法（how）两个维度构建分类体系，追溯从统计MT时代到LLM时代的技术演变。

**核心 idea**：音译不仅是一种预处理技术，更是连接不同书写系统间知识迁移的桥梁。其有效性取决于语言亲缘性、脚本类型和下游任务的交互作用。

## 方法详解

### 整体框架

综述覆盖 50+ 篇论文，按以下结构组织：(1) 五大动机分类——为什么使用音译；(2) 六种整合方式——如何将音译融入NLP流水线；(3) 条件分析——何时音译有益；(4) LLM 时代的定位——现代大模型是否需要音译。

### 关键设计

1. **五大动机分类体系**:

    - 功能：系统化理解音译在 NLP 中的作用
    - 核心思路：(1) 命名实体和 OOV 处理——最早期的应用；(2) 代码混合文本处理——处理同一文本中多脚本混用；(3) 跨脚本语言相似性利用——相关语言用不同脚本书写时（如印地语/乌尔都语）；(4) 英语中心迁移——利用英语为中心的预训练模型；(5) 统一预处理——减少多语言模型的词汇表大小
    - 设计动机：现有文献按方法或任务分类，忽视了动机的时间演变。按动机分类揭示了从"修补"到"系统设计"的范式转变

2. **六种整合方式的比较**:

    - 功能：指导实践者选择最适合的音译策略
    - 核心思路：(1) 完全替换为音译语料；(2) 音译增强（保留原始+增加音译版本）；(3) 词汇表增强；(4) 嵌入拼接/融合；(5) 提示策略（ICL 中加入音译）；(6) 多编码器/多集成架构
    - 设计动机：不同整合方式在复杂度、效果和适用条件上差异巨大——简单替换可能最有效但也最有风险

3. **LLM 时代的重新审视**:

    - 功能：评估现代大规模预训练是否使音译变得多余
    - 核心思路：即使在 LLM 时代，(1) tokenizer 对低资源脚本的覆盖仍然不足（一个词可能被切成 10+ 个 token）；(2) 罗马化可以显著改善推理效率（减少 token 数）；(3) 对于训练数据极少的脚本，音译仍然是最实用的方案
    - 设计动机：避免"LLM 解决一切"的误解——tokenization 瓶颈在多语言 LLM 中仍然存在

### 损失函数 / 训练策略

综述论文，不涉及训练。

## 实验关键数据

### 主实验

综述不包含自身实验，但总结了以下关键发现：

| 条件 | 音译效果 | 原因 |
|------|---------|------|
| 相关语言+不同脚本 | 强正面 | 最大化词汇重叠增益 |
| 表意文字（中文等） | 负面 | 音译丢失语义信息 |
| 代码混合文本 | 正面 | 统一脚本减少分布不匹配 |
| LLM tokenizer 覆盖差 | 正面 | 减少 token 碎片化 |

### 关键发现

- 音译最有效的场景是相关语言使用不同脚本（如天城文印地语→拉丁化后与乌尔都语对齐）
- 罗马化在 LLM 推理效率方面有实际价值——将低资源语言罗马化可减少 token 数量 2-5 倍
- 音译的主要风险是信息丢失——尤其是声调语言和表意文字
- 音译+原始文本的双语料增强通常比纯替换更安全

## 亮点与洞察

- 按动机的时间演变组织综述是一个有启发性的视角——从"补丁"到"系统设计"的演变揭示了领域的成熟过程
- 关于 LLM 时代音译仍然必要的论证非常有说服力——tokenization 瓶颈是一个容易被忽视但影响实际的问题
- 具体的实践建议是综述的重要价值——研究者可以直接根据语言、任务和资源条件查表选策略

## 局限与展望

- 综述范围侧重于罗马化/拉丁化，非拉丁目标脚本的讨论较少
- 对音译质量（工具选择、错误影响）的讨论不够深入
- 缺少定量的 meta-analysis

## 相关工作与启发

- **vs 多语言预训练综述**: 关注模型架构和训练策略，本文关注数据/输入层面的音译干预
- **vs 代码混合综述**: 覆盖代码混合的完整现象，本文聚焦跨脚本转换这一正交维度

## 评分

- 新颖性: ⭐⭐⭐ 综述本身的组织框架有价值，但不涉及新方法
- 实验充分度: ⭐⭐⭐ 综述论文无自身实验
- 写作质量: ⭐⭐⭐⭐ 分类体系清晰，表格总结有用
- 价值: ⭐⭐⭐⭐ 对多语言NLP研究者有实用参考价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](../../NeurIPS2025/recommender/mmpb_its_time_for_multi-modal_personalization.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)

<!-- RELATED:END -->
