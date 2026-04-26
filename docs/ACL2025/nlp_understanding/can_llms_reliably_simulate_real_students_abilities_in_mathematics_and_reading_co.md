---
title: >-
  [论文解读] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?
description: >-
  [ACL 2025][NLP理解][LLM评估] 利用项目反应理论(IRT)将11个LLM与真实学生放在同一能力量表上评估，发现在无引导情况下强模型远超学生平均水平，而"扮演某年级学生"的提示虽能改变表现，但**没有任何模型-提示组合**能在所有学科和年级上可靠模拟平均学生。
tags:
  - ACL 2025
  - NLP理解
  - LLM评估
  - 模拟学生
  - 项目反应理论(IRT)
  - 教育评估
  - 提示学习
---

# Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?

**会议**: ACL 2025  
**arXiv**: [2507.08232](https://arxiv.org/abs/2507.08232)  
**代码**: [GitHub](https://github.com/kvadityasrivatsa/IRT-for-LLMs-as-Students)  
**领域**: NLP理解  
**关键词**: LLM评估, 模拟学生, 项目反应理论(IRT), 教育评估, Persona Prompting

## 一句话总结

利用项目反应理论(IRT)将11个LLM与真实学生放在同一能力量表上评估，发现在无引导情况下强模型远超学生平均水平，而"扮演某年级学生"的提示虽能改变表现，但**没有任何模型-提示组合**能在所有学科和年级上可靠模拟平均学生。

## 研究背景与动机

**领域现状**：LLM越来越多地被用作智能辅导系统(ITS)中的"代理学生"和考试预测试的合成考生。如果LLM能可靠模拟真实学生，就可以低成本替代昂贵的真人试测。

**现有痛点**：目前评估LLM是否像真实学生的方法主要依赖**专家主观判断**，缺乏定量框架。Liu et al. (2025)和Grohs et al. (2024)虽然使用了LLM生成合成考生数据，但只关注聚合统计量，不考察LLM潜在能力是否与特定年级对齐。

**核心矛盾**：LLM作为代理学生的**可行性未经量化验证**——强模型可能远超学生表现，弱模型可能只是偶然对齐，persona prompting的效果高度不确定。

**本文目标**：(RQ1) 在标准提示下LLM与各年级真实学生的表现差距有多大？(RQ2) 要求LLM"扮演某年级学生"是否能改善对齐？

**切入角度**：采用来自NAEP（美国国家教育进步评估）的真实学生作答数据，使用Rasch模型（IRT的一种形式）将LLM和学生放在**同一能力量表**上。

**核心 idea**：用IRT框架定量评估LLM是否能作为K-12学生的可靠代理。

## 方法详解

### 整体框架

收集NAEP数据集（489道多选题，涵盖4/8/12年级的数学和阅读），让11个LLM回答同一组题目，再用Rasch模型估计LLM的能力参数 $\theta_i$，与真实学生的能力分布对比。

### 关键设计

1. **Rasch模型能力估计**：

    - 正确回答概率：$P(X_{ij}=1) = \frac{e^{\theta_i - b_j}}{1 + e^{\theta_i - b_j}}$
    - 其中 $\theta_i$ 是考生（LLM）能力，$b_j$ 是题目难度
    - 题目难度由学生正确率估计：$b_j \approx \log\frac{1-p_j}{p_j}$
    - 设计动机：Rasch模型简洁可解释，可将LLM和学生放在同一能力标度上

2. **百分位数对齐度量**：

    - 假设学生能力服从标准正态分布 $\theta \sim \mathcal{N}(0,1)$，平均学生的百分位为50
    - LLM的百分位：$\text{Percentile Rank} = \Phi(\theta_i) \times 100$
    - 越接近50说明越接近平均学生

3. **四种提示策略**：

    - **Unenforced**：标准零样本提示，不加任何角色指令
    - **GradeEnforcedMinimal**：加一句"请作为X年级普通学生回答"
    - **GradeEnforcedBasicCoT**：引导模型思考该年级学生会怎么选择
    - **GradeEnforcedFullCoT**：两步推理——先判断该年级学生是否可能答对，再据此选答案

### 模型选择

选取11个多样化LLM：LLaMA2-13B/70B、LLaMA3.1-8B/70B、Mistral-7B、Qwen2.5-7B、Qwen2.5-Math、GPT-3.5-Turbo、o3-Mini、SocraticLM、LearnLM-1.5-Pro，涵盖开源/闭源、不同规模、通用/领域微调。

## 实验关键数据

### 主实验（Unenforced vs. GradeEnforced百分位对比）

| 模型 | 数学4年级$P_U$ | 数学4年级$P_E$ | 阅读4年级$P_U$ | 阅读4年级$P_E$ |
|------|------------|------------|------------|------------|
| LLaMA3.1-70B | 99.6 | 97.6 | 99.9 | 99.9 |
| o3-Mini | 98.9 | 98.3 | 99.3 | 99.9 |
| Qwen2.5-7B | 99.6 | 18.5 | 98.2 | 5.2 |
| GPT-3.5-Turbo | 89.0 | 44.7 | 99.7 | 61.8 |
| Mistral-7B | 63.7 | 58.9 | 94.3 | 67.9 |
| LLaMA2-13B | 63.7 | 66.1 | 99.7 | 95.5 |
| 目标百分位 | **50** | **50** | **50** | **50** |

- 理想值为50。大多数强模型远超50（如o3-Mini在各设定下百分位均>86）
- 平均偏差：Unenforced下数学约32.9-40.5个百分位点，GradeEnforced下约28.8-30.2

### 准确率数据

| 模型 | 数学G4 | 数学G8 | 数学G12 | 阅读G4 | 阅读G8 | 阅读G12 |
|------|--------|--------|---------|--------|--------|---------|
| LLaMA3.1-70B | 93.9% | 91.5% | 80.0% | 98.0% | 93.1% | 83.6% |
| SocraticLM | 92.7% | 94.3% | 81.7% | 70.3% | 62.5% | 58.2% |
| Mistral-7B | 65.9% | 54.7% | 53.3% | 89.1% | 86.1% | 83.6% |

### 关键发现

- **无引导时强模型全面碾压学生**：GSM8K高分模型（o3-Mini、Qwen2.5-Math等）在所有年级百分位>95，完全无法模拟真实学生
- **弱模型可能偶然对齐**：LLaMA2-13B、Mistral-7B因能力较弱反而接近部分年级的平均水平
- **Grade prompting效果不一致**：GradeEnforcedFullCoT变化最大，但并非总是最优；Qwen2.5-7B在4年级阅读从98.2猛降到5.2（过度矫正）
- **教育微调模型不占优**：SocraticLM和LearnLM-1.5-Pro在grade alignment上并不比通用模型（如Mistral-7B）更好
- **没有任何模型-提示组合在所有年级和学科上都对齐**

## 亮点与洞察

- **IRT框架提供了可解释的定量对齐度量**，比传统的accuracy比较更有信息量
- **揭示了一个重要现实**：当前LLM作为"代理学生"用于教育评估是不可靠的，需要专门的训练策略
- **提出三条proxy student选择准则**：年级对齐（$\theta$在正态带内）、发展顺序（能力随年级单调递增）、提示稳定性
- **数据集可持续更新**：NAEP定期发布新题目，数据集可以动态扩展

## 局限与展望

1. **仅限文本类多选题**：排除了图表题和开放式题目，不能完全反映学生评估场景
2. **缺少跨年级数据**：NAEP只报告同年级学生在同年级题目的表现，无法验证学生在不同年级题目上的表现
3. **仅用提示方法**：未探索微调、in-context learning等更深层的对齐策略
4. **样本量有限**：489道题、2个学科、3个年级，覆盖面不够广
5. **persona prompting设计空间未充分探索**：只测试了三种grade enforcement提示

## 相关工作与启发

- **与Benedetto et al. (2024)的关系**：后者发现GPT-4可通过一句prompting模拟不同水平考生，但未使用IRT量化对齐度——本文补上了定量分析
- **与Zelikman et al. (2023)比较**：后者在K-12阅读中模拟学生但只做聚合相关性分析，不做年级对齐检验
- **启发**：对于需要LLM模拟人类行为（不仅是教育领域）的应用，都应先建立定量对齐框架再使用

## 评分

- **新颖性**: ⭐⭐⭐⭐ — IRT框架用于LLM-学生对齐评估是新颖的切入点
- **实验充分度**: ⭐⭐⭐⭐ — 11个模型×4种提示×3年级×2学科，评估全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，RQ驱动的分析逻辑严谨
- **价值**: ⭐⭐⭐⭐ — 对教育AI领域有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)
- [\[ACL 2025\] Déjà Vu? Decoding Repeated Reading from Eye Movements](deja_vu_decoding_repeated_reading_from_eye_movements.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)

<!-- RELATED:END -->
