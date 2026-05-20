---
title: >-
  [论文解读] Explain the Flag: Contextualizing Hate Speech Beyond Censorship
description: >-
  [ACL 2026][社会计算][仇恨言论检测] 本文提出一种混合方法，结合 LLM 和三种语言（英/法/希腊语）的人工策展词汇表来检测和解释仇恨言论——术语管道通过词汇匹配+LLM 语义消歧检测固有贬损用语，无术语管道用 LLM 检测群体针对性内容，两者融合生成有据可查的解释。
tags:
  - "ACL 2026"
  - "社会计算"
  - "仇恨言论检测"
  - "可解释性"
  - "多语言词汇表"
  - "上下文化解释"
  - "混合系统"
---

# Explain the Flag: Contextualizing Hate Speech Beyond Censorship

**会议**: ACL 2026  
**arXiv**: [2604.14970](https://arxiv.org/abs/2604.14970)  
**代码**: [GitHub](https://github.com/ails-lab/detoex)  
**领域**: 社会计算 / 仇恨言论  
**关键词**: 仇恨言论检测, 可解释性, 多语言词汇表, 上下文化解释, 混合系统

## 一句话总结
本文提出一种混合方法，结合 LLM 和三种语言（英/法/希腊语）的人工策展词汇表来检测和解释仇恨言论——术语管道通过词汇匹配+LLM 语义消歧检测固有贬损用语，无术语管道用 LLM 检测群体针对性内容，两者融合生成有据可查的解释。

## 研究背景与动机

**领域现状**：自动化仇恨言论检测系统广泛用于在线平台审核，但大多聚焦于审查或删除，缺乏透明度和解释性——用户被标记但不知为何被标记。

**现有痛点**：（1）纯删除方式缺乏透明度，限制了用户理解为什么其语言有害；（2）审核决策可能显得武断或有偏见；（3）仇恨言论有两种形态——固有贬损用语（如侮辱性称呼）和群体针对性内容（即使无侮辱词也可能有害）——需要不同的检测策略；（4）低资源语言（如希腊语）缺乏相关资源。

**核心矛盾**：审核需要在"阻止有害内容"和"解释为何有害"之间取得平衡——纯 LLM 方法缺乏稳定的术语知识，纯词汇方法缺乏上下文理解。

**本文目标**：构建一个能检测和解释仇恨言论的混合系统，覆盖英/法/希腊语。

**切入角度**：双管道设计——术语管道利用策展词汇表做精确匹配+LLM 消歧，无术语管道用 LLM 做上下文感知的群体针对检测。

**核心 idea**：策展词汇表（含义解释+身份特征标注）+ LLM 上下文推理 → 有据可查的解释。

## 方法详解

### 整体框架
双管道并行：（1）术语管道：词形还原+字符串匹配检测潜在贬损术语 → LLM 在上下文中消歧（贬义/非贬义用法）→ 输出解释；（2）无术语管道：LLM 直接判断文本是否针对群体/个人的身份特征攻击 → 输出解释。两管道融合：任一标记则标记，两者都标记则 LLM 融合去重输出统一解释。

### 关键设计

1. **多语言策展词汇表**:

    - 功能：为 LLM 提供可靠的术语知识基础
    - 核心思路：从 Wiktionary 提取带"derogatory/offensive/vulgarities"标签的术语，经五步流程构建：初始收集（11,310 英/3,749 法/965 希腊）→ 过滤（保留针对群体的固有贬损用语）→ 分类（标注身份特征）→ 丰富描述（LLM 生成包含争议/非争议用法的连续文本）→ 人工验证。最终得到 3,904 英/1,644 法/288 希腊条目
    - 设计动机：LLM 可能不了解罕见或文化特定的贬损用语，策展词汇表提供了可靠的外部知识来弥补 LLM 的知识盲区

2. **LLM 语义消歧**:

    - 功能：判断检测到的术语在当前上下文中是否为贬义用法
    - 核心思路：LLM 接收源文本和词汇表中该术语的含义描述（包括争议和非争议用法），输出是否为贬义使用的判断+解释。这处理了多义词（如"bitch"可指母狗/骂人）和回收用语（被目标群体回收使用的情况）
    - 设计动机：许多贬损术语有非贬义含义，简单匹配会产生大量误报——需要 LLM 的上下文理解来消歧

3. **双管道融合与解释生成**:

    - 功能：综合两种检测策略的结果，生成有据可查的统一解释
    - 核心思路：仅当两管道都认为无仇恨言论时才判为安全。一个管道检出则使用该管道的解释。两个都检出则由 LLM 融合两个解释，去除冗余，生成连贯的统一解释
    - 设计动机：两管道互补——术语管道检测固有贬损用语但可能遗漏无侮辱词的群体攻击，无术语管道检测上下文攻击但可能遗漏罕见术语

### 损失函数 / 训练策略
混合系统不涉及训练。使用 Claude Sonnet 3.7 作为大模型，Llama 系列作为轻量开源替代。

## 实验关键数据

### 主实验

| 语言 | 模型 | Precision | Recall | F1 (Safe) |
|------|------|-----------|--------|-----------|
| 英语 | Claude (混合) | 0.92 | 0.89 | 0.90 |
| 英语 | Llama (混合) | 0.82 | 0.82 | 0.82 |
| 法语 | Claude (混合) | 0.96 | 0.91 | 0.93 |
| 希腊语 | Claude (混合) | - | - | 高于基线 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅无术语管道 (LLM-only) | 较低 | 遗漏罕见/文化特定术语 |
| 仅术语管道 | 较低 | 遗漏无侮辱词的群体攻击 |
| 混合系统 | 最优 | 两管道互补 |

### 关键发现
- 混合系统一致优于纯 LLM 基线，证明策展词汇表对 LLM 有增强作用
- 人工评估显示解释质量高——用户能理解为什么内容被标记
- Claude 显著优于 Llama 系列，但 Llama 在低资源部署（单 GPU）中有实用价值
- 词汇表在希腊语（低资源语言）上的增益尤其显著

## 亮点与洞察
- **从审查到解释**的理念转变有重要的社会价值——解释为什么有害比简单删除更能促进用户理解和行为改变
- 策展词汇表+LLM 的混合模式是一个可推广的范式——在任何需要"精确领域知识+上下文理解"的任务中都适用
- 多语言词汇表的构建方法论（Wiktionary + LLM 过滤 + 人工验证）是可复用的资源构建流程

## 局限与展望
- 词汇表需要持续维护以覆盖新出现的贬损用语
- 仅在推文（短文本）上评估，长文本场景可能不同
- 回收用语（如 LGBTQ 社区回收的术语）的处理仍有挑战——缺少用户身份信息时难以判断
- 解释的自动评估指标有限，主要依赖人工评估

## 相关工作与启发
- **vs 纯 LLM 检测**: 缺乏稳定的术语知识，可能漏检罕见侮辱
- **vs 纯词汇方法**: 缺乏上下文理解，误报率高
- **vs Menis Mastromichalakis et al. (2025)**: 他们做可解释仇恨言论但不涉及多语言词汇表

## 评分
- 新颖性: ⭐⭐⭐ 双管道混合方法不算全新，但多语言词汇表是有价值的资源贡献
- 实验充分度: ⭐⭐⭐⭐ 三语言覆盖、人工评估检测和解释质量、多模型对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，社会动机充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](../../ACL2025/social_computing/implihatevid_video_hate.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](../../ACL2025/social_computing/silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)

</div>

<!-- RELATED:END -->
