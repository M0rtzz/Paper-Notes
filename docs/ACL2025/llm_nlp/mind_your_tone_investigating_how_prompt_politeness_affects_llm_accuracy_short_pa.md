---
title: >-
  [论文解读] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy
description: >-
  [ACL 2025 (Findings, Short Paper)][LLM/NLP][提示语气] 本文系统研究了提示语的礼貌程度对LLM回答准确率的影响，通过构建5种语气梯度（从"非常礼貌"到"非常粗鲁"）的250条多选题提示并在ChatGPT 4o上测试…
tags:
  - "ACL 2025 (Findings, Short Paper)"
  - "LLM/NLP"
  - "提示语气"
  - "礼貌度"
  - "LLM准确率"
  - "人机交互"
  - "提示工程"
---

# Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy

**会议**: ACL 2025 (Findings, Short Paper)  
**arXiv**: [2510.04950](https://arxiv.org/abs/2510.04950)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 提示语气、礼貌度、LLM准确率、人机交互、提示工程

## 一句话总结

本文系统研究了提示语的礼貌程度对LLM回答准确率的影响，通过构建5种语气梯度（从"非常礼貌"到"非常粗鲁"）的250条多选题提示并在ChatGPT 4o上测试，发现与直觉相反——粗鲁提示的准确率（84.8%）显著高于礼貌提示（80.8%）。

## 研究背景与动机

**领域现状**：提示工程（Prompt Engineering）已成为提升LLM性能的关键手段，研究者们已广泛探索了提示格式、指令结构、few-shot示例等因素对模型输出的影响。然而在语用学层面，提示语的"语气"和"礼貌度"对模型表现的影响几乎未被系统研究。

**现有痛点**：此前有少量研究（如Yin et al. 2024等）探索了情感因素对LLM的影响，部分研究发现粗鲁表达会导致模型表现下降。但这些结论可能随着模型迭代而过时，且缺乏对不同礼貌度梯度的细粒度分析。不同LLM版本在对齐训练策略上的差异，可能导致语气偏好完全不同。

**核心矛盾**：一方面，RLHF等对齐训练鼓励模型对礼貌用户给出更有帮助的回答；另一方面，粗鲁/直接的表达方式通常在指令层面更精确、更少歧义，可能反而有助于任务完成。礼貌表达引入的冗余信息（如"请您百忙之中帮忙"）可能分散模型的注意力。

**本文目标**：量化不同语气等级对LLM多选问答准确率的影响，并检验这种差异是否具有统计显著性。

**切入角度**：设计受控实验——保持问题内容不变，仅改变语气包装（Very Polite → Polite → Neutral → Rude → Very Rude），从而隔离语气变量的独立影响。

**核心 idea**：通过配对样本t检验（paired sample t-test）验证语气对LLM准确率的因果效应，发现新一代LLM在粗鲁提示下表现更好。

## 方法详解

### 整体框架

实验采用经典的受控变量设计：首先构建50道基础多选题（涵盖数学、科学、历史三个领域），然后将每道题改写为5种语气变体（Very Polite / Polite / Neutral / Rude / Very Rude），最终得到50×5=250条提示。所有提示输入ChatGPT 4o，记录回答正确率，最后用配对样本t检验评估差异的统计显著性。

### 关键设计

1. **五级语气梯度构建**:

    - 功能：为同一问题生成语气不同但内容等价的提示变体
    - 核心思路：Very Polite 使用敬语和感谢表达（如"I would be incredibly grateful if you could..."），Polite 使用"please"和"could you"，Neutral 直接提问不加修饰，Rude 使用命令式和不耐烦的语气，Very Rude 使用侮辱性和攻击性表达（如"Are you stupid? Just answer this..."）。每个变体保留完全相同的核心问题内容和选项
    - 设计动机：5级梯度比二分法（礼貌vs粗鲁）能揭示更细粒度的趋势，如是否存在某个"甜点"语气

2. **多领域基础题集**:

    - 功能：确保结论不受特定领域偏差影响
    - 核心思路：从数学、科学、历史三个领域各采样约17题，每题为标准多选问答格式，答案明确无歧义
    - 设计动机：不同领域问题的语义复杂度不同，多领域覆盖增强结论的泛化性

3. **配对样本t检验**:

    - 功能：统计检验不同语气对之间的准确率差异是否显著
    - 核心思路：对每一对语气条件（如VP vs VR），在50个问题上分别计算准确率，然后做配对t检验。由于同一基础问题在不同语气下构成天然配对，满足配对检验的前提
    - 设计动机：消除问题难度差异带来的混淆变量，直接比较同一问题在不同语气下的表现

### 损失函数 / 训练策略

本文为实验研究，不涉及模型训练或损失函数。核心分析方法为配对样本t检验，原假设为"语气条件之间准确率无差异"。

## 实验关键数据

### 主实验

| 语气条件 | 准确率 | 与Neutral差值 |
|----------|--------|--------------|
| Very Polite | 80.8% | -1.2% |
| Polite | 81.2% | -0.8% |
| Neutral | 82.0% | 基准 |
| Rude | 83.6% | +1.6% |
| Very Rude | 84.8% | +2.8% |

### 统计显著性检验

| 对比条件 | t值 | p值 | 显著性 |
|----------|-----|-----|--------|
| Very Polite vs Very Rude | - | <0.05 | 显著 |
| Polite vs Rude | - | <0.05 | 显著 |
| Neutral vs Very Rude | - | <0.05 | 显著 |

### 关键发现

- 准确率随礼貌度降低单调递增，从Very Polite的80.8%到Very Rude的84.8%，呈现明显的线性趋势
- 与Yin et al. (2024)等早期研究的结论相反——那些研究在较旧的模型上发现粗鲁降低表现
- 配对t检验证实差异具有统计显著性（p<0.05），排除了随机波动的可能
- 可能的解释：粗鲁提示通常措辞更简短直接，减少了冗余信息干扰；新一代模型的对齐训练可能削弱了对语气的敏感度

## 亮点与洞察

- **反直觉发现**是最大亮点：挑战了"礼貌提示=更好结果"的朴素假设，说明LLM的RLHF对齐并不意味着对礼貌表达有更高回应质量。这一发现对提示工程策略有直接指导意义
- **实验设计简洁有效**：通过配对设计巧妙控制了混淆变量，用最小的实验规模得到了可信结论。这种"改写同一问题"的方法可以迁移到研究其他语用因素（如正式度、情感色彩、委婉程度）
- **暗示了对齐训练的局限**：RLHF可能使模型学会了"不因语气生成攻击性回复"，但并未让模型在礼貌请求下投入更多"推理能力"

## 局限与展望

- **仅测试了ChatGPT 4o一个模型**：结论能否推广到开源模型（LLaMA、Mistral等）或不同RLHF策略的模型未知
- **数据规模较小**：50题×5语气，总共250条样本，统计功效有限；增大到500+题可获得更稳健结论
- **仅多选问答任务**：开放式生成、推理链、代码生成等任务中语气影响可能完全不同
- **语气改写的质量难以控制**：不同语气变体的语义等价性难以完美保证，粗鲁表达可能无意中简化了问题表述
- 未来可扩展到多语言、多模型、多任务的大规模研究

## 相关工作与启发

- **vs Yin et al. (2024)等早期研究**: 他们发现粗鲁表达降低LLM表现，本文在更新的GPT-4o上得到相反结论，说明对齐策略的迭代可能改变了模型的语气敏感度
- **vs EmotionPrompt (Li et al. 2023)**: 他们研究情感短语（如"这对我的职业生涯很重要"）对LLM的影响，发现正面情感有帮助。本文关注的是语气而非情感意图，维度不同但互补
- 这篇短文虽然规模不大，但揭示的现象值得更深入研究——如果粗鲁语气确实有效，那prompt设计中是否应该追求"简洁直接"而非"礼貌周到"？

## 评分

- 新颖性: ⭐⭐⭐⭐ 研究角度新颖，结论反直觉，但属于实验观察而非方法创新
- 实验充分度: ⭐⭐⭐ 作为短文尚可，但规模太小（50题/单模型），缺乏消融和跨模型验证
- 写作质量: ⭐⭐⭐⭐ 短文结构清晰，论证逻辑明确
- 价值: ⭐⭐⭐ 揭示了有趣现象，但需要更大规模复现才能指导实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [\[ACL 2025\] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](../../ICLR2026/llm_nlp/how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)

</div>

<!-- RELATED:END -->
