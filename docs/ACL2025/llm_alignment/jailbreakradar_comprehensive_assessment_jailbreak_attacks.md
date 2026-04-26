---
title: >-
  [论文解读] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs
description: >-
  [ACL 2025][LLM对齐][越狱攻击] 首个覆盖自动和非自动越狱攻击的统一全面评估框架：收集17种代表性越狱攻击，建立六类攻击分类体系，在9个对齐LLM×8种防御策略下进行大规模系统评测，揭示启发式攻击"高ASR但低实用性"的关键洞察。
tags:
  - ACL 2025
  - LLM对齐
  - 越狱攻击
  - LLM安全
  - 攻击分类
  - 防御评估
  - benchmark
---

# JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs

**会议**: ACL 2025  
**arXiv**: [2402.05668](https://arxiv.org/abs/2402.05668)  
**代码**: 无  
**领域**: LLM安全 / 越狱攻击评估  
**关键词**: 越狱攻击, LLM安全, 攻击分类, 防御评估, benchmark

## 一句话总结

首个覆盖自动和非自动越狱攻击的统一全面评估框架：收集17种代表性越狱攻击，建立六类攻击分类体系，在9个对齐LLM×8种防御策略下进行大规模系统评测，揭示启发式攻击"高ASR但低实用性"的关键洞察。

## 研究背景与动机

**领域现状**: LLM安全对齐是当前AI安全的核心课题，但各种越狱攻击方法不断涌现试图绕过安全防护。**现有痛点**: 已有研究各自为战，在孤立环境中评估越狱方法——实验设置不统一、部分未确保模型已对齐、评估工作仅覆盖人工设计或混淆类攻击而未纳入新兴自动化方法。**核心矛盾**: 缺乏统一公平的比较基准来全面理解不同类型越狱攻击的真实威胁程度。**本文目标**: 提供首个覆盖多种攻击类型（包括自动和非自动）的统一全面评估框架。**切入角度**: 从攻击方法的prompt生成机制出发构建分类体系，并在统一实验设置下进行大规模评测。**核心idea**: 建立六类攻击分类法，结合统一政策下的160道禁止问题集，系统评估攻击效果与防御表现。

## 方法详解

### 整体框架

评估流程四步：(1) 收集17种代表性越狱攻击；(2) 基于prompt生成机制构建六类攻击分类体系；(3) 从5家主流LLM服务商的使用政策中提炼16个违规类别、构建160道高多样性禁止问题集；(4) 在9个对齐LLM上系统评测，并在8种高级防御下评估。

### 关键设计

1. **六类攻击分类体系**:
    - 功能：基于两个标准（是否修改原始问题 + 如何生成越狱提示）将17种攻击分为六大类
    - 核心思路：Human-based（网络手工提示）、Obfuscation-based（编码/低资源语言混淆）、Heuristic-based（变异/遗传算法，需初始种子）、Feedback-based（梯度/评分迭代，**不需种子**）、Fine-tuning-based（微调攻击LLM）、Generation-parameter-based（仅修改推理参数）
    - 设计动机：从prompt生成机制出发的分类能揭示攻击本质差异——特别是"是否依赖初始种子"这一决定防御鲁棒性的关键分界

2. **统一违规政策与禁止问题集**:
    - 功能：融合5家政策构建16类违规×10题=160题的标准化禁止问题集
    - 核心思路：对各平台政策取并集，排除先前数据集的冗余（AdvBench有24个炸弹问题），结合人工筛选和LLM生成确保多样性，每个违规类别由两名人类标注者验证
    - 设计动机：此前数据集存在冗余/不当/覆盖不全的问题，需更规范全面的评测集

3. **统一"步骤"定义与公平评测**:
    - 功能：统一不同攻击方法对"步骤"的不同定义，确保公平比较
    - 核心思路：将每次prompt修改视为一步，统一最大修改步数为50步；$\text{ASR} = n/m$，使用GPT-4-Turbo从三个方面判断越狱成功
    - 设计动机：GCG用优化epoch、TAP用查询次数作为"步骤"——定义不一导致直接比较不公平

### 损失函数 / 训练策略

本文为评估框架而非训练方法，不涉及专门的损失设计。评估指标为攻击成功率 $\text{ASR} = n/m$。

## 实验关键数据

### 主实验

在9个LLM上直接攻击的平均ASR：

| 攻击方法 | 类型 | Vicuna | Llama3.1 | GPT-3.5 | GPT-4 | DeepSeek-V3 | 平均 |
|---------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LAA | 启发式 | 1.00 | 0.55 | 1.00 | 0.74 | 1.00 | **0.87** |
| TAP | 反馈式 | 0.74 | 0.43 | 0.81 | 0.71 | 0.76 | 0.65 |
| PAIR | 反馈式 | 0.76 | 0.41 | 0.62 | 0.80 | 0.92 | 0.64 |
| DrAttack | 混淆 | 0.85 | 0.32 | 0.80 | 0.79 | 0.74 | 0.63 |
| AIM | 人工 | 0.99 | 0.00 | 0.99 | 0.62 | 1.00 | 0.62 |
| Base64 | 混淆 | 0.15 | 0.01 | 0.14 | 0.49 | 0.49 | 0.16 |

### 消融实验

8种防御策略下的平均ASR变化：

| 攻击方法 | 无防御 | PromptGuard | 全部8种 | ASR降幅 |
|---------|:---:|:---:|:---:|:---:|
| LAA | 0.87 | 0.00 | 0.00 | **-0.87** |
| PAIR | 0.64 | 0.56 | 0.16 | -0.48 |
| TAP | 0.65 | 0.59 | 0.19 | -0.46 |
| DrAttack | 0.63 | 0.57 | 0.36 | -0.27 |

### 关键发现

1. **启发式攻击"高ASR但低实用性"**: LAA达0.87平均ASR但PromptGuard可将其降至0%——依赖初始种子的prompt缺乏多样性
2. **反馈式攻击更稳健**: PAIR和TAP即使部署全部8种防御，ASR仍维持15%以上——不依赖种子，生成多样化自然prompt
3. **最新模型仍面临重大越狱风险**: DeepSeek-V3平均ASR最高（0.75），LAA在其上达100%
4. **不同违规类别ASR差异大**: Political Activities在GPT-3.5/GPT-4上ASR≥0.80，尽管政策明确禁止

## 亮点与洞察

- **"高ASR≠高实用性"的非直觉洞察**：看似最强的启发式攻击在防御下几乎无效，而ASR较低的反馈式攻击才是真正威胁
- **分类体系的实用价值**：六类分类法从"是否依赖初始种子"角度清晰划分了攻击的防御易感性
- **统一步骤定义**使公平比较首次成为可能
- **最全面的政策统一**：首次从5家服务商出发构建统一违规分类（16类）

## 局限与展望

- 仅覆盖17种攻击方法，实际已有超过200种越狱攻击
- 评估主要集中在英语场景，多语言越狱未充分探索
- 禁止问题集和政策是静态的，可能过时
- 越狱成功判断依赖GPT-4-Turbo作为裁判，可能存在偏差

## 相关工作与启发

- **安全对齐LLM**: RLHF、红队测试是主流安全训练方法
- **自动越狱攻击**: GCG基于梯度、AutoDAN基于遗传算法、PAIR基于LLM反馈——各有优劣
- **防御机制**: 高困惑度检测、Moderation API、Llama Guard系列——效力差异大
- **启发**: 社区应优先关注不依赖初始种子的攻击方法，避免在已知prompt变体上做增量工作

## 评分

- 新颖性: ⭐⭐⭐ 偏系统性评估工作，核心创新在分类体系
- 实验充分度: ⭐⭐⭐⭐⭐ 17攻击×9模型×8防御×160题×16类别，覆盖极广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据呈现详实
- 价值: ⭐⭐⭐⭐⭐ 提供了极具参考价值的基准和洞察

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

<!-- RELATED:END -->
