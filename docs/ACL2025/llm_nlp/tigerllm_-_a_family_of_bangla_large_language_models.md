---
title: >-
  [论文解读] TigerLLM - A Family of Bangla Large Language Models
description: >-
  [ACL 2025][LLM/NLP][低资源语言] 针对孟加拉语（全球第5大语言）的 LLM 严重不足问题，构建高质量教科书语料 Bangla-TextBook（10M token）和原生指令数据 Bangla-Instruct（100K），训练的 TigerLLM 家族在六项基准上超越所有开源替代方案并胜过 GPT-3.5。
tags:
  - ACL 2025
  - LLM/NLP
  - 低资源语言
  - 孟加拉语LLM
  - 高质量数据
  - 持续预训练
  - 模型蒸馏
---

# TigerLLM - A Family of Bangla Large Language Models

**会议**: ACL 2025  
**arXiv**: [2503.10995](https://arxiv.org/abs/2503.10995)  
**代码**: [github.com/mraihan-gmu/TigerLLM](https://github.com/mraihan-gmu/TigerLLM/tree/main/)  
**领域**: LLM/NLP  
**关键词**: 低资源语言, 孟加拉语LLM, 高质量数据, 持续预训练, 模型蒸馏

## 一句话总结

针对孟加拉语（全球第5大语言）的 LLM 严重不足问题，构建高质量教科书语料 Bangla-TextBook（10M token）和原生指令数据 Bangla-Instruct（100K），训练的 TigerLLM 家族在六项基准上超越所有开源替代方案并胜过 GPT-3.5。

## 研究背景与动机

孟加拉语拥有约2.37亿母语使用者，是全球第5大语言，但在 NLP 领域严重被低估：

### 现有孟加拉语 LLM 的问题

**训练过程不规范**：
   - titu-Gemma、Bong-LLaMA 等模型缺乏技术文档和学术论文
   - 微调后性能反而低于基座模型（如 titu-LLM 在 MMLU-bn 上仅 0.06，远低于 Gemma-2 基座的 0.35）
   - 结果不可复现

**数据质量低下**：
   - 多数项目依赖翻译版的 Alpaca-Instruct 和 OpenOrca
   - 这些数据集由早期 GPT-3.5 生成，其孟加拉语支持能力有限
   - 使用 Google Translate 翻译后质量进一步下降

**训练语料问题**：
   - 主要依赖 OSCAR 和 Common Crawl，质量控制不足
   - 缺乏高质量的教育类孟加拉语内容

## 方法详解

### 整体框架

TigerLLM 的开发包含三个核心贡献：

1. **Bangla-TextBook 语料**：来自孟加拉国国家课程与教科书委员会的6-12年级教科书
2. **Bangla-Instruct 数据集**：使用 GPT-4o 和 Claude-3.5-Sonnet 生成的原生孟加拉语指令数据
3. **TigerLLM 模型家族**：基于 LLaMA 3.2 (1B) 和 Gemma 2 (9B) 持续预训练并微调

### 关键设计

**Bangla-TextBook 语料构建**：
- 来源：孟加拉国国家课程与教科书委员会出版的163本开源教科书
- 年级范围：6-12年级，覆盖多个学科领域
- 规模：9,897,623 token，697,903 句子
- 核心理念：数据质量优于数量（Gunasekar et al., 2023 "Textbooks Are All You Need" 的启发）

**Bangla-Instruct 生成流水线**（四阶段）：

1. **种子与指令生成**：

    - 500个种子任务由50名来自孟加拉国主要大学的本科/研究生志愿者创建
    - 覆盖5个学科方向和10个类别
    - 每轮采样 k=8 个种子，用 Claude 生成新指令候选

2. **任务类型分类**：

    - GPT-4o 将每条指令分为开放式、分类、生成三类
    - 确定最低回答长度阈值

3. **回答起草**：

    - Claude 根据指令和类型生成全面回答
    - 保留内部一致性评分最高的版本

4. **多阶段过滤**：

    - GPT-4o 应用四维过滤：语言（ℒ）、文化（𝒞）、质量（𝒬）、新颖性（𝒩）
    - 约63%的 (指令, 回答) 对通过过滤
    - 复杂度分布：40%基础、40%中级、20%高级
    - 通过验证的对加入种子池，循环直到达到100K高质量对

**模型选择与演化**：
- 候选基座模型：LLaMA 3.2 (1B, 3B)、Gemma 2 (2B, 9B)、Pangea (7B)
- 经选择后确定 LLaMA 3.2 (1B) 和 Gemma 2 (9B) 作为最终基座
- Pangea 因孟加拉语性能过低被淘汰

### 损失函数 / 训练策略

**持续预训练**：
- 硬件：8 × NVIDIA A100 (40GB)，512GB RAM
- 使用 Bangla-TextBook 语料
- 训练约120小时（梯度检查点启用）
- 多次试验以经验性确定超参数

**微调**：
- 硬件：单块 NVIDIA A100 (40GB)，Google Colab
- **不使用 LoRA，采用全参微调**以获得更好学习效果
- 使用 Flash Attention 加速
- 关键参数：最大序列长度 2048，batch size 8，梯度累积步数 4，训练 3 epoch
- 学习率 5×10⁻⁵，权重衰减 0.02，10%预热步数
- 训练约96小时

## 实验关键数据

### 主实验

**六项孟加拉语基准上的表现（Pass@1）**：

| 模型 | MMLU-bn | PangBench | BanglaQuaD | mHumanEval | BEnQA | BanglaRQA |
|------|---------|-----------|------------|------------|-------|-----------|
| GPT-3.5 | 0.55 | 0.55 | 0.50 | 0.56 | 0.50 | 0.49 |
| GPT-4o-mini | 0.67 | 0.62 | 0.65 | 0.56 | 0.60 | 0.60 |
| Gemma 2 (27B) | 0.35 | 0.51 | 0.43 | 0.64 | 0.50 | 0.56 |
| LLaMA 3.2 (11B) | 0.22 | 0.19 | 0.21 | 0.15 | 0.18 | 0.20 |
| Titu-LLM | 0.06 | 0.19 | 0.08 | 0.02 | 0.17 | 0.21 |
| Bong-LLaMA | 0.05 | 0.12 | 0.08 | 0.02 | 0.15 | 0.13 |
| **TigerLLM (1B)** | **0.61** | **0.55** | **0.68** | **0.61** | **0.59** | **0.62** |
| **TigerLLM (9B)** | **0.72** | **0.68** | **0.70** | **0.63** | **0.65** | **0.68** |

核心发现：
- TigerLLM (9B) 在所有指标上超越 GPT-3.5 和 GPT-4o-mini（除编码外）
- TigerLLM (1B)（仅1B参数！）在多数任务上超越 GPT-3.5 和所有开源替代方案
- 现有微调模型（Titu-LLM、Bong-LLaMA）的结果不可复现，性能远低于基座模型

### 消融实验

**数据质量 vs. 数量的验证**：
- TigerLLM 仅使用 10M token 预训练 + 100K 指令微调
- 相比之下 titu-Gemma 使用 4.4B token、titu-LLaMA 使用 37B token
- TigerLLM 以极小数据量实现远超大规模方案的效果
- 验证了"高质量数据优于海量低质数据"的假说

**预训练与微调的损失曲线**：
- 持续预训练：损失稳定下降，模型有效吸收孟加拉语知识
- 微调：损失快速收敛，3 epoch 内达到良好效果

### 关键发现

1. **数据质量压倒性地重要于数据量**：10M token 的教科书语料 > 37B token 的网络数据
2. **原生语言指令优于翻译指令**：Bangla-Instruct（原生生成）远优于翻译 Alpaca/OpenOrca
3. **全参微调优于 LoRA**：在资源允许的情况下，全参微调带来更好效果
4. **小模型的潜力**：1B 模型通过高质量数据可超越 11B-27B 基座模型
5. **既有孟加拉语 LLM 的系统性问题**：训练不当导致微调后反而退化

## 亮点与洞察

1. **"教科书就是你所需要的"在低资源语言中的验证**：将 Phi-1 的理念成功应用于孟加拉语，证明了高质量策划数据的普适价值
2. **自指令生成的多文化扩展**：500个人工种子任务确保了文化真实性，避免了翻译数据的文化失真
3. **完整开源**：语料、指令数据、模型全部开源，具有极高的可复现性和社区价值
4. **务实的算力方案**：整个训练流程仅需 8×A100（预训练）+ 1×A100（微调），适合资源有限的团队
5. **系统性诊断现有问题**：深入分析了其他孟加拉语 LLM 失败的根本原因

## 局限与展望

1. **语料领域偏窄**：仅来自6-12年级教科书，缺少新闻、文学、技术文档等领域
2. **模型规模受限**：仅1B和9B，未探索更大规模是否能进一步提升
3. **指令类型有限**：100K 指令仅覆盖部分任务类型，可能无法涵盖真实使用场景的复杂性
4. **缺乏深度定性分析**：未展示模型的错误模式和失败案例
5. **评估基准有限**：孟加拉语评估基准本身不够全面，可能低估或高估某些能力

## 相关工作与启发

- **Phi-1/Textbooks Are All You Need (Gunasekar et al., 2023)**：高质量小数据优于低质量大数据的理念，直接启发了 Bangla-TextBook
- **BanglaBERT (Sami et al., 2022)**：孟加拉语 BERT，证明了单语言专门化的有效性
- **Self-Instruct (Wang et al., 2023)**：指令数据生成方法论，本文用 GPT-4o + Claude 升级了教师模型
- **BLOOM/Aya**：覆盖多语言的开放模型，但在低资源语言上仍有显著性能差距
- 启发：低资源语言 LLM 的关键瓶颈是数据质量而非模型规模

## 评分

- **创新性**：⭐⭐⭐ — 方法层面是已有技术的组合应用，但问题定义和数据工程有价值
- **实用性**：⭐⭐⭐⭐⭐ — 为2.37亿孟加拉语使用者提供了首个高质量开源 LLM
- **实验充分性**：⭐⭐⭐ — 覆盖6项基准但缺少消融和深度分析
- **写作质量**：⭐⭐⭐⭐ — 问题动机清晰，数据工程流程描述详细

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)

<!-- RELATED:END -->
