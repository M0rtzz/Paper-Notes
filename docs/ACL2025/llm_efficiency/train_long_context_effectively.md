---
title: >-
  [论文解读] How to Train Long-Context Language Models (Effectively)
description: >-
  [ACL 2025][LLM效率][长上下文训练] 本文系统研究如何通过持续预训练和 SFT 有效训练长上下文语言模型，提出数据配比、训练长度缩放、评估协议等一系列关键设计，最终训练出的 ProLong-8B 仅用 Llama-3.1 **5%** 的长上下文训练数据即在 128K 长度上达到同规模 SOTA。
tags:
  - ACL 2025
  - LLM效率
  - 长上下文训练
  - 持续预训练
  - 数据配比
  - 监督微调
  - 位置外推
---

# How to Train Long-Context Language Models (Effectively)

**会议**: ACL 2025  
**arXiv**: [2410.02660](https://arxiv.org/abs/2410.02660)  
**领域**: LLM 效率  
**关键词**: 长上下文训练, 持续预训练, 数据配比, 监督微调, 位置外推

## 一句话总结

本文系统研究如何通过持续预训练和 SFT 有效训练长上下文语言模型，提出数据配比、训练长度缩放、评估协议等一系列关键设计，最终训练出的 ProLong-8B 仅用 Llama-3.1 **5%** 的长上下文训练数据即在 128K 长度上达到同规模 SOTA。

## 研究背景与动机

- **应用需求迫切**：长上下文 LM 已解锁书籍摘要、多示例 ICL、长文档 RAG 等新应用，但将预训练模型适配到 128K+ 上下文面临基础设施与数据双重挑战
- **无训练外推不可靠**：修改 RoPE 频率基数等训练免费方法（YaRN、Dynamic NTK）无法可靠通过简单的 Needle-in-a-Haystack（NIAH）任务，仍需在数十亿长文档 token 上继续训练
- **设计决策不透明**：前沿开源模型（如 Llama-3.1）采用"长上下文持续训练 → SFT"范式，但数据配比、序列长度选择、SFT 数据类型等关键决策对社区公开不足
- **评估方法存在缺陷**：现有评估手段（困惑度、NIAH）不可靠——NIAH 对强模型已饱和（Llama-3.1-8B/70B 均 100 分），困惑度与下游任务表现不一致（100% 长数据持续改善 PPL 但严重损害下游性能）

## 方法详解

### 整体框架

采用三阶段方案：**(1)** 建立基于 HELMET 的可靠评估协议指导模型开发；**(2)** 基于 Llama-3-8B-Instruct 进行两阶段长上下文持续预训练（64K → 512K）；**(3)** 在短指令数据 UltraChat 上进行 SFT。最终得到 ProLong-8B，上下文窗口支持 512K token，训练总量仅 40B token。

### 关键设计

**设计 1：可靠评估协议——SFT 后评估 + 多任务基准**

采用 HELMET 评估套件覆盖 6 类下游任务（Recall、RAG、Re-ranking、ICL、QA、Summarization），而非依赖困惑度或 NIAH。核心发现：必须在 SFT **之后**评估——部分长上下文性能改进（如 RAG、Re-ranking）仅在 SFT 后才显现；同时跟踪短上下文基准（HellaSwag、MMLU、ARC-c、WinoGrande、GSM8K），确保性能不退化。评估协议对比如下：

| 评估方式 | NIAH | Recall | RAG | Re-rank | 能否区分强弱模型 |
|---------|------|--------|-----|---------|---------------|
| NIAH only | 100 | - | - | - | ✗（强模型已饱和）|
| PPL only | ↓ | - | - | - | ✗（与下游不一致）|
| **HELMET (SFT后)** | 100 | 99.4 | 56.3 | 37.0 | **✓（多维区分）** |

**设计 2：数据配比策略——60% 长 + 40% 高质量短混合**

长数据来源消融发现：代码仓库（将同一 repo 所有文件拼接为单文档，98.8B 长 token）和书籍（33.2B 长 token）是最优长数据源，1:1 混合效果最佳。短长比例消融发现：60% 长数据 + 40% 短数据为最优比例，100% 长数据反而严重损害下游长上下文任务。短数据混合设计 ProLong ShortMix，保留数学推理能力：

| 短数据组件 | 占比 | 作用 |
|-----------|------|------|
| FineWeb | 25% | 通用网页文本 |
| FineWeb-Edu | 25% | 教育性网页文本 |
| Wikipedia | 10% | 百科知识 |
| Tulu-v2 | 10% | 指令数据 |
| StackExchange | 10% | 技术问答 |
| ArXiv | 10% | 学术论文 |
| OpenWebMath | 10% | 数学推理保留 |

**设计 3：两阶段训练长度缩放 + 纯短 SFT**

训练采用课程学习策略：Stage 1 在 64K 长度训练 20B token（RoPE base = 8×10⁶，2.2K H100 小时），Stage 2 在 512K 长度训练 20B token（RoPE base = 1.28×10⁸，12.2K H100 小时）。关键发现：超越评估长度的训练显著提升性能（512K 训练在 64K 评估上 Re-rank 32.9 vs. 28.0）。SFT 阶段，仅使用短上下文指令数据 UltraChat（平均 1.2K token）即可获得最强长上下文性能，加入合成长 SFT 数据（即使仅 1%）反而降低性能。其他设计：禁用跨文档注意力（提升性能 + 训练吞吐）、从 Instruct 而非 Base 初始化（显著保留短上下文能力）。

## 实验关键数据

### 主实验：HELMET 128K 评估

| 模型 | 参数量 | Max Len | Recall | RAG | ICL | Re-rank | QA | Summ. | Avg. |
|------|--------|---------|--------|-----|-----|---------|-----|-------|------|
| **ProLong** | 8B | 512K | **98.8** | **63.2** | **86.5** | **22.5** | 43.9 | 29.2 | **49.4** |
| Llama-3.1 | 8B | 128K | 95.2 | 59.5 | 83.9 | 14.0 | 43.2 | 27.0 | 46.5 |
| MegaBeam-Mistral | 7B | 512K | 89.6 | 57.0 | 86.2 | 14.7 | 37.3 | 28.9 | 45.4 |
| Llama-3.1 | 70B | 128K | 90.7 | 56.2 | 81.4 | 24.5 | 56.3 | 31.6 | 49.7 |

ProLong-8B 仅用 40B token（Llama-3.1 长上下文训练预算的 5%）即超越 Llama-3.1-8B-Instruct，在除 Summarization 外的所有类别领先，甚至在 Avg. 上接近 70B 的 Llama-3.1。

### 消融分析

**长数据源对比**（60% 长 + 40% ShortMix，5B token 训练）：

| 长数据源 | Recall | RAG | Re-rank | ICL | QA | Summ. | Avg. |
|---------|--------|-----|---------|-----|-----|-------|------|
| CommonCrawl | 84.1 | 53.3 | 28.1 | 67.5 | 35.2 | 37.0 | 50.9 |
| Books | 94.9 | 53.9 | 30.7 | 72.2 | 33.2 | 37.7 | 53.8 |
| Code Repos | 99.2 | 53.8 | 29.0 | 61.2 | 34.7 | 36.2 | 52.3 |
| **Books/Repos 1:1** | **96.0** | **54.9** | 29.4 | **73.9** | **35.7** | **37.9** | **54.6** |

**短数据源对比**：

| 短数据源 (40%) | Long-Ctx Avg. | HellaSwag | MMLU | GSM8K | Short Avg. |
|----------------|---------------|-----------|------|-------|------------|
| SlimPajama | 52.9 | 81.2 | 63.0 | 41.9 | 64.2 |
| FineWeb-Edu | 53.0 | 81.0 | 62.6 | 39.4 | 63.0 |
| DCLM-Baseline | 52.0 | 82.0 | 65.6 | 39.4 | 64.8 |
| **ProLong ShortMix** | **54.6** | 81.6 | 65.3 | **46.6** | **65.5** |

**合成 SFT 数据比例影响**：

| 合成数据占比 | RAG | Re-rank | ICL | QA | Summ. | Avg. |
|-------------|-----|---------|-----|-----|-------|------|
| **0%（纯 UltraChat）** | **58.1** | **38.5** | 80.3 | **49.7** | **42.1** | **55.7** |
| 1% | 57.0 | 38.3 | 80.8 | 45.3 | 41.5 | 54.1 |
| 10% | 55.5 | 36.1 | 80.6 | 41.7 | 39.4 | 53.9 |
| 50% | 48.8 | 18.8 | 70.5 | 42.3 | 33.3 | 43.3 |

### 关键发现

1. **纯长数据训练有害**：100% 长数据虽持续改善 PPL，但严重损害下游长上下文任务表现（SFT 后 RAG、Re-ranking 大幅下降）
2. **超长训练长度有益**：512K 训练 vs. 64K 训练，在 64K 评估上 Recall 98.5 vs. 95.0，Re-rank 32.9 vs. 28.0
3. **短 SFT 数据即可**：0% 合成长 SFT 数据 Avg. 55.7，加入 50% 后骤降至 43.3
4. **短上下文性能保留**：ProLong ShortMix 短上下文均值 65.5，接近原始 Llama-3-8B 的 66.0

## 亮点与不足

**亮点**：

- 挑战"长上下文训练用全长数据"的直觉，首次系统证明混合高质量短数据的关键性
- 首次证明训练序列长度超过评估长度的好处，并给出基于依赖距离的理论解释
- SFT 仅需短指令数据的发现大幅简化长上下文模型训练流程
- 评估方法论贡献：揭示 PPL 和 NIAH 的不可靠性，推动社区采用 HELMET 等多任务评估
- 仅 40B token 训练（5% 数据预算）即超越 Llama-3.1-8B，极高的数据效率

**不足**：

- 实验仅基于 Llama-3-8B（~8B 参数），更大规模下结论是否成立未验证
- 未探索 RLHF/偏好优化对长上下文 SFT 的影响
- 合成长 SFT 数据无效可能与生成器质量有关，需更强模型验证
- 512K 训练计算成本显著（12.2K vs. 2.2K H100 小时），计算最优方案有待探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 系统性消融实验，多个反直觉发现（纯长数据有害、短 SFT 更优）
- **实用性**: ⭐⭐⭐⭐⭐ — 完整可复现的训练方案（ProLong recipe），直接可用
- **实验充分度**: ⭐⭐⭐⭐⭐ — 海量消融覆盖评估协议、数据配比、长度缩放、SFT 策略
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，Takeaway Box 设计精巧，图表信息密度高

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)

<!-- RELATED:END -->
