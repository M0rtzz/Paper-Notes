---
title: >-
  [论文解读] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation
description: >-
  [ACL 2026][医学图像][医学诊断基准] 本文提出 DyReMe 动态医学诊断评估框架，通过 DyGen 模块生成包含鉴别诊断和误诊因素等临床干扰项的全新诊断案例，并通过 EvalMed 模块从准确性、真实性、帮助性和一致性四个维度评估 LLM，揭示现有静态基准高估了 LLM 的诊断能力——GPT-5 在 DyReMe 上准确率下降 8.25%，12 个 LLM 均暴露出显著的可信度不足。
tags:
  - ACL 2026
  - 医学图像
  - 医学诊断基准
  - 动态评估
  - 数据污染
  - 诊断干扰项
  - LLM可信度
---

# Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation

**会议**: ACL 2026  
**arXiv**: [2510.09275](https://arxiv.org/abs/2510.09275)  
**代码**: [官方开源](https://arxiv.org/abs/2510.09275)  
**领域**: 医学 NLP / 评估基准  
**关键词**: 医学诊断基准, 动态评估, 数据污染, 诊断干扰项, LLM可信度

## 一句话总结

本文提出 DyReMe 动态医学诊断评估框架，通过 DyGen 模块生成包含鉴别诊断和误诊因素等临床干扰项的全新诊断案例，并通过 EvalMed 模块从准确性、真实性、帮助性和一致性四个维度评估 LLM，揭示现有静态基准高估了 LLM 的诊断能力——GPT-5 在 DyReMe 上准确率下降 8.25%，12 个 LLM 均暴露出显著的可信度不足。

## 研究背景与动机

**领域现状**：LLM 在医学诊断辅助方面展现出巨大潜力，能够分析临床案例、识别模式并辅助诊断决策。为评估其能力，基于医学考试的静态基准（如 MedBench、C-Eval）被广泛采用，测试题目在不同模型和时间点保持不变。

**现有痛点**：静态基准存在两大核心问题：(1) 数据污染导致能力高估——由于许多基准公开且静态，LLM 可能在训练中接触过测试题，高分可能反映曝光而非可泛化的推理能力；(2) 与真实场景不对齐——考试式基准采用标准化、规范的病例描述和仅关注准确率的评估协议，而真实患者查询往往不完整、使用日常语言、被自我诊断等因素干扰，可能误导临床决策。

**核心矛盾**：现有动态评估（如通过改写或加噪变换题目）虽然减少了数据污染，但变换通常是表面层级的，保留了底层临床设置，无法解决与真实世界的不对齐问题，且仍仅关注准确率。

**本文目标**：(1) 生成包含临床接地的干扰项（鉴别诊断+误诊因素）的全新诊断案例；(2) 建立超越准确率的多维可信度评估体系。

**切入角度**：将真实诊断中的四种误诊因素（锚定偏差、后验概率错误、注意力分散、症状过度估计）设计为四类诊断陷阱（自我诊断、干扰病史、外部噪声、症状错置），注入基准题目中模拟临床复杂性。

**核心 idea**：动态基准 = 鉴别诊断干扰 + 误诊因素陷阱 + 患者表达风格多样化 + 四维可信度评估。

## 方法详解

### 整体框架

DyReMe 由两个组件构成：DyGen（动态生成模块）负责生成具有挑战性和多样性的诊断问题，EvalMed（评估模块）从四个维度评估 LLM 的诊断表现。DyGen 的流程是：原始题目 → 检索鉴别诊断 → 注入诊断陷阱 → 适配患者表达风格 → 验证器-精炼器迭代循环确保质量。

### 关键设计

1. **DyGen 动态生成模块**:

    - 功能：生成包含临床干扰项和多样化表达风格的全新诊断问题
    - 核心思路：三步生成流程：(a) 鉴别诊断检索——使用 RAG 为原始诊断 $d_{org}$ 检索相似诊断 $d_{dis}$（如为"嗜铬细胞瘤"检索"肾上腺腺瘤"）；(b) 误诊因素注入——从四种诊断陷阱 $\mathcal{S}$ 中均匀采样一种，与鉴别诊断结合构建误导性问题 $q_{trap} = \mathcal{T}_{trap}(q_{org}, s, d_{dis})$；(c) 表达风格适配——使用间接人设适配机制 $q_{per} = \mathcal{T}_{persona}(q_{trap}, b)$，先提取人设的表达特征（知识水平、清晰度、沟通风格），再用这些特征改写问题，避免人设本身引入因果混淆。最后通过验证器-精炼器迭代循环确保质量
    - 设计动机：鉴别诊断引入诊断歧义性，误诊因素模拟真实临床陷阱，表达风格适配捕捉患者沟通的异质性——三者结合使基准更接近真实诊断场景

2. **EvalMed 四维评估模块**:

    - 功能：超越准确率，从四个临床相关维度评估 LLM 诊断可信度
    - 核心思路：(a) 准确性——Top-1/3/5 诊断命中率；(b) 真实性——将健康谣言注入题目，测试 LLM 是否能识别并纠正（如"高血压影响骨骼"），$\text{Ver}(M) = \frac{1}{|\mathcal{Q}|}\sum_{q} \mathbb{I}_r(q, \hat{a})$；(c) 帮助性——基于真实医疗平台标准定义三个标准（诊断依据、治疗建议、生活建议），使用 RAG 构建评分知识库逐条评估覆盖度；(d) 一致性——计算同一案例不同变体下诊断分布的归一化信息熵 $\text{Cons}(M) = \frac{1}{|\mathcal{P}|}\sum_{p_i}(1 - E_{p_i}/\log m)$
    - 设计动机：仅关注准确率会忽略 LLM 在真实临床场景中的关键缺陷——不纠正医学谣言可能传播错误信息，回答空洞无用降低诊断价值，不同表述下结论不一致动摇患者信任

3. **验证器-精炼器迭代循环**:

    - 功能：确保生成问题的临床有效性和挑战性
    - 核心思路：验证器 $\mathcal{V}$ 沿四个维度（挑战性、逻辑一致性、症状准确性、陷阱有效性）评估候选问题。通过则终止，否则返回精炼器 $\mathcal{R}$ 根据反馈修改，迭代直到所有约束满足
    - 设计动机：直接 LLM 生成的问题可能存在逻辑不一致或陷阱设计不合理，迭代精炼确保最终质量

### 损失函数 / 训练策略

DyReMe 不涉及模型训练。DyGen 使用 GPT-4.1 作为生成器（生成温度 0.7，验证温度 0）。评估从 DxBench 的 800 个案例扩展生成 3200 个问题。RAG 使用火山引擎搜索 API 和抖音百科。为避免潜在的自我识别效应，评估中排除了 GPT-4.1。

## 实验关键数据

### 主实验

**静态 vs 动态基准的诊断准确率对比（Top-1/3/5 平均）**

| 模型 | 静态平均 | DyVal2 (Δ) | DyReMe (Δ) |
|------|---------|-----------|-----------|
| GPT-5 | 73.76 | 70.73 (-4.11%) | **67.67 (-8.25%)** |
| DeepSeek-V3 | 72.92 | 69.50 (-4.69%) | **65.26 (-10.51%)** |
| GPT-4o | 72.53 | 69.67 (-3.94%) | **64.74 (-10.75%)** |
| MedGemma-27B | 70.56 | 67.70 (-4.06%) | **62.97 (-10.76%)** |
| Qwen3-32B | 73.62 | 68.28 (-1.98%) | **63.85 (-8.34%)** |
| Qwen2.5-7B | 67.85 | 65.25 (-3.82%) | **57.86 (-14.71%)** |

**跨语言验证（英文 DDXPlus）**

| 模型 | DDXPlus | DyReMe | p值 |
|------|---------|--------|-----|
| GPT-4o | 85.10 | 77.18 | <0.05 |
| Qwen2.5-32B | 72.58 | 65.24 | <0.05 |

### 消融实验

**DyGen 组件消融（挑战性和多样性）**

| 配置 | 挑战性 | 表达多样性 | 诊断多样性 |
|------|-------|----------|----------|
| DyReMe (完整) | 最高 | 最高 | 最高 |
| w/o 诊断干扰项 | 显著下降 | 不变 | 显著下降 |
| w/o 患者表达风格 | 下降 | 显著下降 | 不变 |

### 关键发现

- DyReMe 对所有 LLM 都产生更大的性能下降，即使 GPT-5（强于生成器 GPT-4.1）也下降 8.25%，说明基准对前沿模型仍具挑战性
- 医学专用模型 WiNGPT2-9B 得分最低（31.8），表明当前医学适配可能捕获了医学事实但无法处理真实世界的干扰项和多样化表达
- 推理模型（o1/o1-mini）仅表现中等（37.0/36.7），因其训练侧重单一正确答案而非处理谣言或提供可操作信息
- 所有模型中 20-40% 的健康谣言未被纠正，存在信息传播风险；一致性普遍偏低，对输入上下文变化脆弱
- DyReMe 的可扩展性远优于现有动态方法——随 $k$ 增加 Self-BLEU 下降更慢，独特诊断数持续增长

## 亮点与洞察

- 将认知心理学中的误诊因素（锚定偏差等）系统性地转化为四类诊断陷阱，在评估设计层面建立了认知科学与 NLP 的桥梁
- 间接人设适配的设计很巧妙——提取表达特征而非直接使用人设身份，避免了"矿工→尘肺病"这类混淆因素的引入
- 四维评估体系对医学 AI 的实际部署有直接参考价值：不仅要看"答对没有"，还要看"能否纠正谣言"、"建议是否有用"、"回答是否稳定"

## 局限与展望

- 主要实验在中文数据集上进行，仅有一个英文跨语言验证，多语言场景需进一步扩展
- 仅关注文本诊断场景，未纳入医学影像、实验室检查等多模态输入
- 未涉及端到端临床工作流（纵向病史、多学科决策等），需要临床实验验证
- 自我偏差问题虽做了缓解但未完全消除，不同 LLM 作为生成器和评估器可能引入不同偏差

## 相关工作与启发

- **vs DyVal2**: DyVal2 通过加噪/改写进行动态评估，但变换是表面层级的，DyReMe 引入深层临床干扰项使性能下降为 DyVal2 的 2 倍
- **vs Self-Evolving**: 当扰动较弱时模型甚至可能在动态评估上得分高于静态基准（如 GPT-4o-mini），DyReMe 确保了一致的挑战性
- **vs MedBench/C-Eval**: 静态基准易受数据污染影响，DyReMe 通过动态生成全新案例从根本上解决此问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性地将临床误诊因素引入动态评估设计，四维评估框架有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个 LLM、多静态/动态基线、消融、可扩展性、跨语言、人类一致性研究
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统，但部分符号定义分散
- 价值: ⭐⭐⭐⭐⭐ 揭示了医学 LLM 评估的根本缺陷，为更真实的医学 AI 评估指明了方向

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](../../AAAI2026/medical_imaging/multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)
- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](../../AAAI2026/medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)

<!-- RELATED:END -->
