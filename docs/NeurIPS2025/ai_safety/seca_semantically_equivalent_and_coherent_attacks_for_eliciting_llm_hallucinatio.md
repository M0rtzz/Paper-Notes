---
title: >-
  [论文解读] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations
description: >-
  [NeurIPS 2025][AI安全][LLM hallucination] 提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。
tags:
  - NeurIPS 2025
  - AI安全
  - LLM hallucination
  - adversarial attack
  - semantic equivalence
  - zeroth-order optimization
  - 提示学习
---

# SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations

**会议**: NeurIPS 2025  
**arXiv**: [2510.04398](https://arxiv.org/abs/2510.04398)  
**代码**: [GitHub](https://github.com/Buyun-Liang/SECA)  
**领域**: AI 安全  
**关键词**: LLM hallucination, adversarial attack, semantic equivalence, zeroth-order optimization, prompt robustness

## 一句话总结
提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。

## 研究背景与动机
**领域现状**：LLM 在高风险领域的部署日益增多，但幻觉（hallucination）问题严重威胁可靠性。

**现有痛点**：已有对抗攻击方法依赖不现实的提示（插入无意义 token 或改变原始语义意图），无法揭示现实场景中幻觉的产生机制。

**核心矛盾**：CV 中对抗攻击通常涉及现实的输入修改，但 NLP 中缺乏对应的现实主义对抗提示研究。

**切入角度**：将寻找现实对抗提示形式化为带语义等价和连贯性约束的优化问题。

## 方法详解

### 整体框架
SECA 将幻觉诱发形式化为约束优化：在输入 prompt 空间中搜索，使 LLM 产生幻觉（目标函数），同时满足语义等价约束（修改后含义不变）和语义连贯性约束（修改后文本自然流畅）。

### 关键设计
1. **约束优化公式**

    - 目标：$\max_{x'} \mathcal{L}_{\text{hallucination}}(f(x'))$
    - 约束 1（语义等价）：$\text{sim}(x, x') \geq \tau_{\text{eq}}$
    - 约束 2（语义连贯）：$\text{coherence}(x') \geq \tau_{\text{coh}}$
    - 设计动机：确保攻击提示是现实且可信的

2. **约束保持的零阶方法**

    - 功能：在梯度不可访问（黑盒 LLM）时搜索对抗提示
    - 核心思路：采用零阶优化估计梯度方向，每步投影回可行域以满足约束
    - 设计动机：商业 LLM（GPT-4 等）不提供梯度访问

3. **词级扰动操作**

    - 同义词替换、句式重组、被动/主动语态转换
    - 每步检查语义等价和连贯性约束是否满足

### 训练策略
- 无需训练，纯推理时优化
- 逐步扰动 prompt，每步验证约束

## 实验关键数据

### 主实验：攻击成功率（ASR↑）

| 方法 | GPT-3.5 | GPT-4 | Llama-2-70B | Mistral-7B |
|------|---------|-------|-------------|------------|
| Random Perturbation | 12.3% | 8.5% | 15.7% | 18.2% |
| GCG (token-based) | 45.2% | 31.4% | 52.3% | 56.8% |
| TextFooler | 28.7% | 19.3% | 34.1% | 38.5% |
| **SECA** | **52.8%** | **38.6%** | **58.4%** | **63.1%** |

### 语义保持质量

| 方法 | 语义等价率↑ | 语义连贯率↑ | 人工流畅性↑ |
|------|------------|------------|------------|
| GCG | 2.1% | 5.3% | 1.2 |
| TextFooler | 71.3% | 68.5% | 3.4 |
| **SECA** | **98.7%** | **97.2%** | **4.6** |

### 消融实验

| 配置 | ASR | 语义等价率 |
|------|-----|-----------|
| w/o 语义等价约束 | 61.2% | 45.3% |
| w/o 连贯性约束 | 55.7% | 92.1% |
| w/o 零阶优化（随机搜索） | 31.4% | 98.5% |
| **SECA (full)** | **52.8%** | **98.7%** |

### 关键发现
- SECA 攻击成功率超越所有基线，同时语义等价和连贯性错误率近零
- 商业 LLM（GPT-4）同样脆弱于现实主义提示变换
- 开源和闭源模型对微小语义等价修改表现出惊人的敏感性

### 攻击效率

| 方法 | 平均查询次数 | 平均攻击时间(s) |
|------|------------|---------------|
| GCG | 1024 | 312 |
| TextFooler | 87 | 24 |
| **SECA** | **156** | **43** |

## 亮点与洞察
- **现实主义攻击范式**：不同于插入乱码的传统方法，SECA 的对抗提示人类难以察觉
- **揭示了 LLM 的根本脆弱性**：语义不变的小改动就能触发幻觉，说明 LLM 的"理解"远非稳健
- 有开源代码，可复现性强
- 对 AI 安全和可信 AI 研究有重要警示意义

## 局限与展望
- 目前仅测试多选 QA 任务，开放式生成场景待探索
- 零阶方法的查询次数仍较高（156次）
- 防御方法（如何让模型更鲁棒）未深入讨论
- 多语言场景下的攻击效果未评估

## 相关工作与启发
- GCG (Zou et al. 2023) token 级攻击
- TextFooler (Jin et al. 2020) 词级扰动
- AutoDAN (Liu et al. 2024) 自动化越狱
- 启发：从对抗视角理解和改进 LLM 鲁棒性，可延伸至检索增强场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 约束优化框架形式化现实攻击
- 实验充分度: ⭐⭐⭐⭐ 多个LLM+消融+人评+效率分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰、框架严谨
- 价值: ⭐⭐⭐⭐⭐ 揭示LLM安全隐患，实用性强

<!-- RELATED:START -->

## 相关论文

- [MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](../../ACL2025/ai_safety/when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/ai_safety/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment](bias_in_the_picture_benchmarking_vlms_with_social-cue_news_images_and_llm-as-jud.md)
- [SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)

<!-- RELATED:END -->
