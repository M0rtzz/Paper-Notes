---
title: >-
  [论文解读] Correlated Errors in Large Language Models
description: >-
  [ICML 2025][LLM相关性] 本文通过对超过350个LLM的大规模实证分析，发现不同LLM之间存在高度相关的错误模式——在两个模型都出错时约60%的情况下会选择同一个错误答案，且越准确的模型相关性越高；进而研究了这种相关性对LLM-as-Judge评估和招聘市场的下游影响。
tags:
  - ICML 2025
  - LLM相关性
  - 算法同质化
  - 错误一致性
  - LLM-as-Judge
  - 劳动力市场
---

# Correlated Errors in Large Language Models

**会议**: ICML 2025  
**arXiv**: [2506.07962](https://arxiv.org/abs/2506.07962)  
**代码**: https://github.com/nikhgarg/llm_correlated_errors_public/ (有)  
**领域**: LLM/NLP  
**关键词**: LLM相关性, 算法同质化, 错误一致性, LLM-as-Judge, 劳动力市场

## 一句话总结
本文通过对超过350个LLM的大规模实证分析，发现不同LLM之间存在高度相关的错误模式——在两个模型都出错时约60%的情况下会选择同一个错误答案，且越准确的模型相关性越高；进而研究了这种相关性对LLM-as-Judge评估和招聘市场的下游影响。

## 研究背景与动机
**领域现状**：LLM越来越多地被部署在多智能体/多模型的高风险场景中（如评估、招聘），而一个隐含假设是"使用不同的模型能带来多样性和鲁棒性"。  

**现有痛点**：缺乏大规模实证数据来验证不同LLM是否真正在行为上存在有意义的差异。多个企业使用看似不同的模型是否真能避免系统性排斥（systemic exclusion）？  

**核心矛盾**：直觉上，不同架构/不同厂商的模型应当有不同的错误分布；但如果训练数据和优化目标趋同，模型可能在错误模式上趋于一致。  

**本文目标**：量化LLM之间的错误相关性，找到影响相关性的因素，评估相关性在实际应用场景中的影响。  

**切入角度**：利用两个大型排行榜（HuggingFace Open LLM Leaderboard、Stanford HELM）的多选题回答数据，以及自建的简历筛选数据集，系统性地分析模型对错误答案的一致性。  

**核心 idea**：LLM的错误并非随机且独立的——越准确的模型在错误上越趋同，这对依赖模型多样性的应用场景构成风险。

## 方法详解

### 整体框架
本文的研究框架分为三大部分：(1) 量化LLM错误相关性的程度；(2) 回归分析解释相关性的来源；(3) 在两个下游任务中分析相关性的实际影响。使用了三个数据集：HuggingFace（349个LLM，12,032道选择题）、HELM（71个LLM，14,042道选择题）、简历筛选（20个LLM，1,800个简历-职位对）。

### 关键设计

1. **错误一致率（Agreement Rate When Both Wrong）**: 

    - 功能：度量两个模型在都答错的情况下选择同一个错误答案的概率
    - 核心思路：通过条件化"两个模型都错"来消除准确率的混淆因素——两个准确的模型自然会在正确答案上高度一致，但这不代表真正的同质化
    - 设计动机：随机猜测的基线agreement rate在HELM上为1/3（3个错误选项），在HuggingFace上为0.127（3-10个选项）
    - 和之前方法的区别：与Goel et al. (2025)相比，本文不依赖模型输出概率分布，更适用于只有最终答案的黑盒场景

2. **回归分析（Regression Analysis of Error Agreement）**: 

    - 功能：用线性回归分析哪些因素驱动了模型间的错误相关性
    - 核心思路：将每对模型的agreement rate作为因变量，以是否同一厂商、是否同一架构、各自准确率及其交互项作为自变量
    - 设计动机：区分"表面相似"（同公司/同架构）和"深层趋同"（高准确率模型的内在相似性）
    - 关键发现：即使控制了厂商和架构因素，更准确的模型pair依然有更高的错误相关性

3. **简历筛选数据集（Resume-Job Evaluation）**: 

    - 功能：构建了30个职位描述 × 60份简历的1,800个配对，用20个LLM评分，并有450对人工标注作为ground truth
    - 核心思路：通过残差相关性（residual correlation）来度量——残差 = 模型评分 - 人工评分
    - 设计动机：从选择题扩展到更接近实际部署场景的开放式评估

### 损失函数 / 训练策略
本文为实证分析工作，不涉及模型训练。回归分析使用标准OLS回归，所有数值变量标准化处理。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 平均Agreement Rate | 随机基线 | 倍率 |
|--------|------|-------------------|----------|------|
| HuggingFace | Agreement when both wrong | 0.423 | 0.127 | 3.3× |
| HELM | Agreement when both wrong | 0.600 | 0.333 | 1.8× |
| Resumes | Residual correlation | 高度相关 | 0 | - |

| 回归因素 | HuggingFace系数 | HELM系数 | Resumes系数 |
|----------|----------------|----------|------------|
| Same Company | 0.066** | 0.022** | 0.021 |
| Same Architecture | 0.076** | - | - |
| Acc.1 | 0.014** | 0.055** | 0.015** |
| Acc.2 | 0.013** | 0.054** | 0.028** |
| Acc.1 × Acc.2 | 0.023** | 0.026** | 0.043** |
| R² | 0.340 | 0.613 | 0.415 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 移除Same Company变量 | R²略降 | 厂商是重要但非唯一因素 |
| 仅保留准确率变量 | R²仍显著 | 准确率本身是错误趋同的独立驱动力 |
| 极端例子 | 0.9987 agreement | google/text-unicorn vs writer/palmyra-x-v3几乎完全一致 |

### 关键发现
- 几乎所有模型对的agreement rate都高于随机基线（HuggingFace 100%，HELM 97.5%）
- 更准确的模型有更高的错误相关性，即使架构和厂商不同——说明模型在"变好"的同时也在"变同"
- LLM-as-Judge中，judge模型系统性地高估比自己弱的模型的准确率（共享错误答案），同厂商bias更严重
- 在劳动力市场模拟中，即使用20个不同LLM筛简历仍有约20%的systemic exclusion
- 算法单一化对申请者福利的影响是双面的：增加了systemic exclusion，但被录用者有更多选择权

## 亮点与洞察
- 揭示了反直觉的现象：使用更好/更新的模型反而可能使生态系统更同质化
- 实验规模空前：350+模型、26,000+题目的系统性分析
- 将"算法单一化"概念用真实LLM和真实简历数据进行了实证验证
- 对LLM-as-Judge范式提出了重要警告：judge偏向于与自己相似的模型

## 局限与展望
- 仅评估了多选题和数值评分场景，缺乏对开放式生成任务的分析
- 错误相关性度量对所有错误答案一视同仁，没有考虑"接近正确"的错误
- 简历评估中的ground truth来自有限的人工标注（450对），主观性较强
- 没有探讨如何主动降低模型相关性
- 劳动力市场模拟较为简化，未考虑真实市场中的动态反馈

## 相关工作与启发
- Goel et al. (2025)同期独立提出了类似的模型相似性度量，但使用了概率分布信息
- Wu et al. (2024)从生成多样性角度发现LLM内部存在monoculture问题
- Peng & Garg (2024a)在理论层面分析了算法单一化对劳动力市场的影响，本文提供了实证验证
- 启发：未来排行榜应不仅评估单个模型的准确率，还应持续监测模型间的相关性

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)
- [CodeMEnv: Benchmarking Large Language Models on Code Migration](../../ACL2025/llm_evaluation/codemenv_benchmarking_large_language_models_on_code_migration.md)
- [Can Large Language Models Master Complex Card Games?](../../NeurIPS2025/llm_evaluation/can_large_language_models_master_complex_card_games.md)
- [Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](../../ACL2025/llm_evaluation/batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [AD-LLM: Benchmarking Large Language Models for Anomaly Detection](../../ACL2025/llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)

<!-- RELATED:END -->
