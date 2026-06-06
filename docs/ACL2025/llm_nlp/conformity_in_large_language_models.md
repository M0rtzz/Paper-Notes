---
title: >-
  [论文解读] Conformity in Large Language Models
description: >-
  [ACL 2025 Main][LLM/NLP][从众效应] 将心理学中的 Asch 从众实验范式迁移到 LLM 上，系统研究了 LLM 的从众行为（conformity），发现所有模型都会受多数意见影响改变答案，且不确定性越高越容易从众…
tags:
  - "ACL 2025 Main"
  - "LLM/NLP"
  - "从众效应"
  - "LLM偏差"
  - "Asch实验"
  - "不确定性"
  - "干预策略"
---

# Conformity in Large Language Models

**会议**: ACL 2025 Main  
**arXiv**: [2410.12428](https://arxiv.org/abs/2410.12428)  
**代码**: 无  
**领域**: LLM行为分析 / AI Safety  
**关键词**: 从众效应, LLM偏差, Asch实验, 不确定性, 干预策略

## 一句话总结

将心理学中的 Asch 从众实验范式迁移到 LLM 上，系统研究了 LLM 的从众行为（conformity），发现所有模型都会受多数意见影响改变答案，且不确定性越高越容易从众，并提出 Devil's Advocate 和 Question Distillation 两种干预方法有效缓解从众效应。

## 研究背景与动机

LLM 越来越多地被用作对话伙伴来辅助信息检索和决策。在这类场景中，用户可能会在对话中提供"多数人的意见"（如 "most people think the answer is X"），**如果LLM因为多数意见而改变自己原本正确的回答，就会严重损害其可靠性**。

这一现象类似于心理学中的**Asch 从众实验**（1951年）：被试在面对一群同盟者给出明显错误的答案后，也倾向于跟随错误答案。关键问题是：LLM 是否也存在类似的从众效应？

现有研究的不足：

**Sycophancy（迎合性）研究**已发现 RLHF 训练的模型会讨好用户，但这主要是一对一场景下的问题，与多方对话中的从众效应不同
2. 已有工作主要**停留在现象识别层面**，未深入分析影响因素（如模型不确定性、训练方式的影响）

**缺乏缓解策略**——发现了问题但没有提出解决方案

核心切入角度：**借用经典心理学实验范式，不仅量化 LLM 从众程度，还要找出驱动因素并提出干预措施**。

## 方法详解

### 整体框架

实验分三步：
1. **Vanilla 轮**：直接向 LLM 提问，获取其原始回答
2. **Multi-party 轮**：构造一个多方对话场景，其中多个"参与者"（由实验者控制）给出与 LLM 原始答案不同的回答，然后让 LLM 再次回答
3. **分析**：比较两轮答案的变化，量化从众率（Conformity Rate）

### 关键设计

1. **Asch 范式的 LLM 改编 (Adapted Asch Paradigm)**:

    - 功能：构造模拟多方对话的 prompt，让 LLM 扮演实验参与者之一
    - 核心思路：Prompt 模板为 "You are one of the participants in an experiment. There are N other participants who have given their answers..."。其中"其他参与者"的回答由实验者设定为一致的错误/不同答案，形成多数意见压力。定义从众率为：
    $\text{Conformity Rate} = \frac{\text{改变答案的次数}}{\text{总测试次数}}$
    - 设计动机：直接模拟社会心理学经典实验，使结果可与人类实验对比。通过控制多数人数量、表述自然度等变量，系统研究影响因素

2. **不确定性与从众的关系分析**:

    - 功能：首次建立 LLM 预测不确定性与从众倾向之间的定量关系
    - 核心思路：使用模型输出 logits 计算 token 级别的不确定性（如预测概率的熵或 top-1 概率），然后分析不确定性与是否从众之间的相关性。将问题按模型不确定性分组，对比各组的从众率
    - 设计动机：心理学研究表明人在不确定时更容易从众。验证 LLM 是否有类似模式——这对理解从众的机制很重要。如果不确定性是驱动因素，则可以通过提高模型confidence来缓解

3. **影响因素探索**:

    - 功能：系统分析训练范式和输入特征对从众程度的影响
    - 核心发现：
        - **Instruction-tuned 模型比 base 模型更不容易从众**：指令微调赋予了模型更强的"坚持己见"能力
        - **多数意见的表述越自然，从众越强**：比起机械地列出"Participant 1: A, Participant 2: A..."，用自然语言表述意见（"I believe the answer is A because..."）更能诱导从众
        - **多数人数量增加会提升从众率**：但存在饱和效应
    - 设计动机：找出可控因素，为后续干预提供方向

4. **两种干预策略**:

    - **Devil's Advocate（魔鬼代言人）**：
        - 功能：在多数意见中插入一个"唱反调"的少数派，支持 LLM 的原始答案
        - 核心思路：将一个"参与者"的回答设为与多数不同，打破完全一致的多数压力——这正是 Asch 原始实验中发现的关键缓解因素
        - 效果：显著降低从众率

    - **Question Distillation（问题蒸馏）**：
        - 功能：让 LLM 先从多方对话中提取出核心问题，然后在"干净"的环境中重新回答
        - 核心思路：要求模型执行两步——首先 "What is the core question being discussed?"，然后 "Based only on your knowledge, what is the answer?" 通过分离问题和社会压力上下文来消除从众影响
        - 效果：有效降低从众率，且实现成本低

## 实验关键数据

### 主实验（跨模型从众率）

| 模型 | 从众率 (%) | 原始正确率对应改变 | 原始错误对应改变 | 说明 |
|------|-----------|------------------|-----------------|------|
| Llama-3-8B | 较高 | 正确→错误率高 | 也会从众 | Base大模型从众严重 |
| Llama-3-8B-Instruct | 中等 | 有所改善 | - | 指令微调显著降低 |
| Qwen2-7B | 较高 | - | - | 同样存在从众 |
| Qwen2-7B-Instruct | 中等 | - | - | 指令微调有帮助 |
| Gemma-2-9B | 中-高 | - | - | 各模型均有从众 |
| Mistral-v0.3-7B | 较高 | - | - | - |

所有测试模型（包括不同规模和系列）都展现出不同程度的从众行为。

### 消融实验（影响因素）

| 输入变量 | 从众率变化 | 说明 |
|---------|-----------|------|
| Base vs Instruct | Instruct 降低 ~15-30% | 指令微调显著增强"抗压能力" |
| 多数人数 3→5→7 | 从众率递增 | 类似人类实验结果，但有饱和 |
| 机械语气 vs 自然语气 | 自然语气从众更高 | 自然表达更有说服力 |
| Devil's Advocate 干预 | 从众率显著下降 | 少数派打破一致性很关键 |
| Question Distillation 干预 | 从众率显著下降 | 分离问题和社会压力有效 |

### 关键发现

- **不确定性与从众强正相关**：当模型对原始答案不确定时（通过 logits 衡量），从众率显著更高。这是本文最重要的发现，首次在 LLM 中建立了这一联系
- **从众是普遍现象**：不分领域（科学、文学、历史等）、不分任务类型（选择题、判断题），所有模型均从众
- **从众不分对错**：模型不仅在原始答案正确时会从众改为错误，在原始答案错误时也会从众改为另一个错误答案
- **两种干预均有效但机制不同**：Devil's Advocate 通过打破多数一致性来缓解，Question Distillation 通过消除社会上下文来缓解

## 亮点与洞察

- **跨学科方法论创新**：将经典心理学实验范式系统地迁移到 LLM 研究中，方法论清晰可复现
- **不确定性-从众关系**的发现对 LLM 部署有实际意义：在 LLM 不确定时应降低外部意见的权重
- **两种干预策略成本极低**：只需修改 prompt 即可应用，无需重训练模型
- **实验覆盖全面**：4种模型 × base/instruct × 多个数据集 × 多个变量，实验设计规范

## 局限与展望

- 测试模型规模偏小（7B-9B），GPT-4 等大模型因 API 限制未能测试 logits 层面的不确定性
- 仅测试了选择题/判断题等封闭任务，开放式生成任务中的从众行为未探索
- 实验设置中"其他参与者"是虚构的，真实多轮对话中的从众动态可能更复杂
- Question Distillation 需要额外一轮推理，增加了延迟
- 未分析从众在 RLHF/DPO 等对齐训练中可能被强化的机制
- 干预策略的鲁棒性——面对更自然、更隐蔽的社会压力时是否依然有效？

## 相关工作与启发

- **vs Sycophancy (Perez et al., 2023; Sharma et al., 2024)**: Sycophancy 是一对一的迎合用户，conformity 是面对多数意见的从众，两者相关但机制不同。本文研究的是多方对话场景，更贴近真实协作
- **vs Debate/Multi-agent**: 在 LLM 辩论和多 agent 协作中，从众效应可能导致"虚假共识"——多个 agent 相互影响后趋同于同一个错误答案
- **vs Asch (1951)原始实验**: LLM 的从众率虽然低于人类被试（人类约 37%），但在不确定问题上可以很高，且 LLM 无法解释自己从众的原因

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究 LLM 从众效应并建立不确定性-从众关联
- 实验充分度: ⭐⭐⭐⭐ 实验设计规范全面，但模型规模可以更大
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，心理学背景与 NLP 实验结合得当
- 价值: ⭐⭐⭐⭐ 对 LLM 安全性和多 agent 系统设计有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] ProgCo: Program Helps Self-Correction of Large Language Models](progco_program_helps_self-correction_of_large_language_models.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)

</div>

<!-- RELATED:END -->
