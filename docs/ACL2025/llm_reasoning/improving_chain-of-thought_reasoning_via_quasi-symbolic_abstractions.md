---
title: >-
  [论文解读] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions
description: >-
  [ACL 2025][LLM推理][Chain-of-Thought] 本文提出QuaSAR（Quasi-Symbolic Abstract Reasoning），一种CoT变体方法，通过引导LLM先对问题进行符号化抽象（提取变量/谓词）、再用半形式化表示重构问题、最后基于准符号推理链求解，在GPT-4o上相比CoT提高最多8%准确率，并显著增强了对对抗性变体（选项打乱、数值替换）的鲁棒性。
tags:
  - ACL 2025
  - LLM推理
  - Chain-of-Thought
  - 准符号推理
  - 抽象表示
  - 上下文学习
  - 鲁棒性
---

# Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions

**会议**: ACL 2025  
**arXiv**: [2502.12616](https://arxiv.org/abs/2502.12616)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: Chain-of-Thought, 准符号推理, 抽象表示, 上下文学习, 鲁棒性

## 一句话总结
本文提出QuaSAR（Quasi-Symbolic Abstract Reasoning），一种CoT变体方法，通过引导LLM先对问题进行符号化抽象（提取变量/谓词）、再用半形式化表示重构问题、最后基于准符号推理链求解，在GPT-4o上相比CoT提高最多8%准确率，并显著增强了对对抗性变体（选项打乱、数值替换）的鲁棒性。

## 研究背景与动机
Chain-of-Thought（CoT）是目前LLM推理的主流策略，通过将复杂问题分解为中间步骤来提升性能。然而，CoT生成的解释容易受到**内容偏差（content bias）**的影响——模型可能基于表面线索而非逻辑关系进行推理，导致：
- 选项顺序改变时答案改变（MMLU-Redux）
- 数值替换后表现大幅下降（GSM-Symbolic）
- 推理过程不忠实于真实逻辑链

为解决这一问题，一类方法提出使用逻辑形式化（如将自然语言翻译为逻辑程序）配合外部符号求解器。但**完全符号化**面临效率瓶颈：自然语言到形式语言的完整翻译复杂、易出错且灵活性差。

QuaSAR的核心idea是寻找一个折中点：不做完全形式化，而是让LLM只将**关键变量和谓词**符号化，保持自然语言和符号表示的共存。这种"准符号抽象"（quasi-symbolic abstraction）可以将具体世界知识与符号推理解耦，减少内容偏差，同时避免完全形式化的瓶颈。

这一思路有科学哲学基础——Kitcher（1981）的统一解释理论认为，好的解释应当通过将具体实体替换为抽象符号来形成可复用的论证模式。

## 方法详解

### 整体框架
QuaSAR将推理过程结构化为四元组 $(\mathcal{Q}, \mathcal{S}, \mathcal{R}, \mathcal{A})$，其中 $\mathcal{S} = (s_1, s_2, s_3, s_4)$ 是四步结构化指令链。相比标准CoT的三元组 $(\mathcal{Q}, \mathcal{R}, \mathcal{A})$，QuaSAR额外引入符号化变换层。核心是一个单步prompting流程（不需要外部求解器），降低了成本。

### 关键设计

1. **Step 1 - 抽象（Abstraction）**

    - 指导LLM分析问题并提取关键信息：识别相关的符号谓词、变量（数值型或文本型）和常量
    - 这是问题求解的第一步——将具体问题抽象为结构化表示
    - 例如：将"Lisa有5个苹果"抽象为 `apples(Lisa) = 5`

2. **Step 2 - 形式化（Formalisation）**

    - 用符号-自然语言混合形式重新表述原始问题
    - 关键在于"准形式化"——只翻译必要的部分，保留自然语言中对求解有意义的上下文
    - 目的是最小化歧义和内容效应，同时不丢失重要信息

3. **Step 3 - 解释（Explanation）**

    - 基于准符号结构进行逐步推理
    - 推理轨迹使用符号化表示，显式展示步骤间的逻辑连接
    - 减少因上下文知识或隐式逻辑关系引起的错误

4. **Step 4 - 答案（Answering）**

    - LLM按固定模式生成最终答案："The answer is: [number]"
    - 确保推理链有明确结论，并便于自动化评估

### 两种应用模式

1. **QuaSAR for ICL**：直接作为in-context learning策略指导大模型推理
2. **QuaSAR for Demonstrations**：用高性能LLM（如GPT-4o）生成QuaSAR格式的demonstration，作为训练数据微调小模型
    - 质量过滤：先用exact match筛选正确answer，再检验引用精度（约50%被过滤掉）

### 训练策略
- 使用标准语言模型目标训练小模型：$\max_\theta \mathbb{E}_{(q, \alpha, y) \sim \mathcal{D}} \log p_\theta(Y \mid \alpha, Q) p_\theta(\alpha \mid Q)$
- 其中 $\alpha = \alpha_1 \cdot \alpha_2 \cdot \alpha_3 \cdot \alpha_4$ 是四步推理轨迹的拼接

## 实验关键数据

### 主实验（QuaSAR as ICL）

| 模型 | 方法 | AQuA | GSM8K | SVAMP | MMLU-Redux | GPQA | DROP |
|------|------|------|-------|-------|-----------|------|------|
| GPT-4o | Baseline | 72.8 | 94.0 | 90.4 | 79.7 | 46.5 | 83.4 |
| GPT-4o | + CoT | 84.3 | 94.5 | 90.3 | 88.1 | 50.2 | 84.2 |
| GPT-4o | + **QuaSAR** | **87.4** | **96.5** | **97.0** | **90.2** | **55.4** | **88.9** |
| Llama-3-70B | + CoT | 74.0 | 86.1 | 84.6 | 82.0 | 41.9 | 80.2 |
| Llama-3-70B | + **QuaSAR** | **79.1** | **88.2** | **84.9** | **85.7** | **49.2** | **88.0** |

### QuaSAR作为微调Demonstration（括号内为微调后结果）

| 模型 | AQuA | GSM8K | SVAMP | MMLU-Redux |
|------|------|-------|-------|-----------|
| Llama-3-8B + CoT | 69.6(72.2) | 80.4(82.6) | 76.3(78.8) | 64.5(65.9) |
| Llama-3-8B + QuaSAR | 67.2(**78.4**) | 77.2(**83.0**) | 77.3(**82.6**) | 63.0(**67.2**) |

### 鲁棒性测试

| 模型 | 任务 | Baseline | CoT | QuaSAR |
|------|------|---------|-----|--------|
| GPT-4o | MMLU-Redux (选项打乱) | 78.6(-1.2) | 86.8(-1.2) | **90.3(0.0)** |
| GPT-4o | GSM-Symbolic (数值替换) | 89.7(-4.3) | 90.8(-4.7) | **95.3(-1.2)** |
| Llama-3-8B | MMLU-Redux (选项打乱) | 27.0(-3.2) | 30.4(-1.2) | **37.3(-0.3)** |

### 消融实验（去除各步骤的影响）

| 配置 | 平均准确率变化 | 说明 |
|------|-------------|------|
| w/o Step 1 (抽象) | -1.8 | 重要但非决定性 |
| w/o Step 2 (形式化) | **-3.5** | 影响最大 |
| w/o Step 3 (解释) | **-3.4** | 影响最大 |
| w/o Step 4 (答案) | -2.5 | 对选择题影响较大 |
| 随机打乱步骤顺序 | ~-4.0 | 步骤顺序也很重要 |

### 关键发现
- **GPT-4o上QuaSAR vs CoT提升显著**：AQuA +3.1, SVAMP +6.7, GPQA +5.2
- **QuaSAR显著增强鲁棒性**：选项打乱时性能几乎无下降（0.0 vs CoT的-1.2），数值替换时仅降1.2（vs CoT的-4.7）
- **小模型直接用QuaSAR ICL效果有限**：Llama-3-8B等小模型跟不上四步指令的复杂性
- **但用QuaSAR生成demonstration微调小模型很有效**：Llama-3-8B微调后在AQuA上从72.2(CoT)提升至78.4(QuaSAR)
- **Step 2（形式化）和Step 3（解释）最关键**：去除任一都造成>3.4的下降
- **训练数据效率高**：QuaSAR demonstration只需CoT demonstration 25-50%的数据量就能达到相同或更好效果

## 亮点与洞察
- **"准符号化"是一个精妙的折中方案**：既获得了符号推理的精确性和鲁棒性，又避免了完全形式化的瓶颈，这个设计哲学值得借鉴
- **鲁棒性提升是最有说服力的结果**：选项打乱时性能零下降直接证明了内容偏差被有效消除
- **科学哲学视角的理论动机**：Kitcher的统一解释理论为方法提供了优雅的理论支撑
- **两种应用模式（ICL + Demonstration微调）**使方法的适用范围很广

## 局限与展望
- 方法仅在英语任务上验证，多语言泛化未知
- 对于不需要符号化的纯自然语言理解任务（如情感分析），QuaSAR的额外步骤可能是overhead
- 小模型直接用QuaSAR ICL效果不稳定，说明方法对模型能力有下限要求
- 未与o1类模型的长CoT推理进行对比
- 约50%的QuaSAR demonstration被质量过滤掉，生成效率有优化空间
- **研究idea**：QuaSAR的形式化步骤可以与外部验证器结合——先用QuaSAR生成准符号推理链，再用符号验证器自动检查逻辑一致性，实现"可验证的CoT"

## 相关工作与启发
- Faithful CoT（Lyu et al., 2023）：完全符号化+外部求解器的路线，QuaSAR证明了"部分符号化"就足够
- CoMAT（Leang et al., 2024）：另一种结合符号的CoT方法，QuaSAR在GPT-4o上平均高6.8%
- FLAIRE（Arakelyan et al., 2024）：基于逻辑形式化的推理方法
- 本文最重要的insight：**不需要完全形式化，只需要符号化关键变量和谓词**就能获得符号推理的好处，这是实用性和理论优雅性的良好平衡

## 评分
- 新颖性: ⭐⭐⭐⭐ "准符号抽象"介于纯NL和纯符号之间的定位很独特，有科学哲学理论支撑
- 实验充分度: ⭐⭐⭐⭐ 数学+NL推理多任务、多模型、ICL+微调两种模式、鲁棒性测试、全面消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论动机清晰、方法描述精确、实验分析有深度
- 价值: ⭐⭐⭐⭐ 提供了一种通用且实用的CoT增强策略，鲁棒性提升有实际部署意义

<!-- RELATED:START -->

## 相关论文

- [CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
