---
title: >-
  [论文解读] EXECUTE: A Multilingual Benchmark for LLM Token Understanding
description: >-
  [ACL 2025][多语言/翻译][multilingual] 扩展字符理解基准 CUTE 到 8 种语言和多种文字系统，提出 EXECUTE 框架，发现 LLM 在不同语言的字符/词/子字符级别表现差异巨大，且意外发现 LLM 对越不熟悉的语言反而在 token 理解任务上表现越好。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "multilingual"
  - "tokenization"
  - "character understanding"
  - "benchmark"
  - "writing system"
---

# EXECUTE: A Multilingual Benchmark for LLM Token Understanding

**会议**: ACL 2025  
**arXiv**: [2505.17784](https://arxiv.org/abs/2505.17784)  
**代码**: [https://github.com/Leukas/EXECUTE](https://github.com/Leukas/EXECUTE)  
**领域**: 多语言翻译  
**关键词**: multilingual, tokenization, character understanding, benchmark, writing system

## 一句话总结
扩展字符理解基准 CUTE 到 8 种语言和多种文字系统，提出 EXECUTE 框架，发现 LLM 在不同语言的字符/词/子字符级别表现差异巨大，且意外发现 LLM 对越不熟悉的语言反而在 token 理解任务上表现越好。

## 研究背景与动机

**领域现状**：CUTE 基准证明 LLM 在英语字符操作上表现差。但不同语言的文字系统差异（字母表 vs 辅音标 vs 表意文字）可能导致不同的挑战。

**现有痛点**：CUTE 仅覆盖英语和俄语，未考虑书写系统差异；多语言 LLM 对不同语言的 token 分配严重不均匀。

**核心矛盾**：LLM 的 token 理解挑战是否与语言的字符-词-token (CWT) 统计特性相关？

**本文目标** 构建跨文字系统的多语言 token 理解基准。

**切入角度**：统一框架覆盖字母表（英/俄）、辅音标（阿姆哈拉/印地）、阿布吉德（阿拉伯）、表意文字（中文）、混合（日文）、特征文字（韩文）。

**核心 idea**：LLM 的 token 理解困难取决于语言的 CWT 统计——字符/词比越高，字符级任务越难；token/词比越高，词级任务越难。

## 方法详解

### 整体框架
选择 8 种语言（覆盖所有已知书写系统）-> 用翻译模型生成各语言数据 -> 保留 CUTE 的组合和操作任务（去除相似性任务）-> 增加 CJK 子字符任务 -> 评估 5 个多语言 LLM。

### 关键设计

1. **CWT 统计框架**

    - 字符/词比 (c/w)：中文 1.51（极低），英语 4.04，俄语 5.06（高）
    - Token/词比 (t/w)：阿姆哈拉语 7.69（极高，字节级编码），中文 1.25（低）
    - 字符/Token 比 (c/t)：英语 3.05（高），中文 1.20（低）
    - 设计动机：这三个比值预测了 LLM 在不同粒度任务上的表现

2. **简化的可扩展框架**

    - 去除需要静态嵌入和母语者的相似性任务
    - 添加新语言只需要英语到目标语言的翻译模型
    - 设计动机：大幅降低扩展门槛

3. **CJK 子字符任务**

    - 测试 LLM 对中文/日文汉字的部首和笔画理解
    - 设计动机：表意文字有独特的子字符结构

## 实验关键数据

### 主实验 -- 不同语言的平均准确率

| 语言 | 字符级任务 | 词级任务 | 子字符任务 |
|------|-----------|---------|-----------|
| 英语 (c/w=4.04) | ~45% | ~85% | N/A |
| 中文 (c/w=1.51) | ~75% | ~60%* | ~25% |
| 阿拉伯语 | ~40% | ~70% | N/A |
| 阿姆哈拉语 (t/w=7.69) | ~55% | ~45% | N/A |
| 韩语 | ~50% | ~75% | ~30% |
*中文词级任务涉及单字词

### 关键发现矩阵

| 发现 | 说明 |
|------|------|
| CWT 统计预测性能 | c/w 越低（如中文），字符级任务越容易 |
| 越不熟悉的语言表现越好 | 低资源语言的字符操作反而更好（因为字节级 tokenization 保留了更多字符信息） |
| 子字符任务极难 | CJK 部首理解准确率仅 ~25% |
| 不同语言挑战在不同粒度 | 英语在字符级，中文在词级，阿姆哈拉语在词级 |

### 关键发现
- **CWT 统计是强预测因子**：字符/词比越高的语言字符级任务越难，符合直觉
- **反直觉发现：不熟悉的语言表现更好**——可能因为低资源语言被编码为字节级 token，保留了更多字符结构
- **子字符理解是 LLM 的盲区**：部首/笔画任务准确率极低
- **同一文字系统的语言表现高度相关**（如英语和俄语同为字母表）

## 亮点与洞察
- **CWT 统计框架**将语言学特征与 LLM 行为联系起来，为理解多语言 LLM 提供了量化工具
- **"越不熟悉越好"的反直觉发现**揭示了 tokenization 策略对下游能力的深层影响
- **可扩展框架设计**使得添加新语言几乎零门槛

## 局限与展望
- 每种语言仅作为一种文字系统的代表
- 翻译质量可能影响结果
- 未探索微调对 token 理解的改善
- 改进方向：字符级 LLM 对比、tokenization 策略优化

## 相关工作与启发
- **vs CUTE**：EXECUTE 扩展到 8 种语言和多种文字系统
- **vs 拼写纠错研究**：这些工作混合了字符级和语义知识，EXECUTE 纯粹测试 token 结构理解

## 评分
- 新颖性: ⭐⭐⭐⭐ CWT 框架和跨文字系统分析有创新
- 实验充分度: ⭐⭐⭐⭐ 8 语言 × 5 模型 × 多任务类型
- 写作质量: ⭐⭐⭐⭐ 分析细致
- 价值: ⭐⭐⭐⭐ 对多语言 LLM 开发有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)
- [\[ACL 2025\] Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)
- [\[ACL 2025\] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)

</div>

<!-- RELATED:END -->
