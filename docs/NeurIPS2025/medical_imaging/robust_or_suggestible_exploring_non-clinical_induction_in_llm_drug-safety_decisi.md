---
title: >-
  [论文解读] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions
description: >-
  [NeurIPS 2025][医学图像][LLM公平性] 通过基于Persona的评估框架发现，ChatGPT-4o和Bio-Medical-Llama-3-8B在药物不良事件预测中会受到临床无关的社会人口属性（教育、保险、住房等）系统性影响，展现出显式和隐式两种偏差模式。
tags:
  - NeurIPS 2025
  - 医学图像
  - LLM公平性
  - 药物安全
  - 不良事件预测
  - 社会人口偏差
  - 角色偏差
---

# Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions

**会议**: NeurIPS 2025  
**arXiv**: [2510.13931](https://arxiv.org/abs/2510.13931)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: LLM公平性, 药物安全, 不良事件预测, 社会人口偏差, 角色偏差

## 一句话总结

通过基于Persona的评估框架发现，ChatGPT-4o和Bio-Medical-Llama-3-8B在药物不良事件预测中会受到临床无关的社会人口属性（教育、保险、住房等）系统性影响，展现出显式和隐式两种偏差模式。

## 研究背景与动机

LLM在生物医学领域的应用日益广泛，但其在**药物安全预测**方面的可靠性和公平性尚未得到系统审计。本文关注三个核心问题：

**社会人口偏差在药物安全中的空白**：已有研究记录了LLM在法律、教育、急诊医学等领域的偏差，但在药物警戒（pharmacovigilance）任务中，是否存在因患者社会人口属性导致的预测偏差完全未知

**临床无关属性的不当影响**：药物不良事件（AE）预测应仅依赖年龄、性别、体重、用药、疾病等临床变量。教育水平、婚姻状况、保险类型、宗教等属性与AE发生无临床关联，但LLM是否会受其影响？

**用户角色的分层部署**：商业AI系统通常对全科医生（GP）、专科医生和患者提供差异化服务。不同角色框架下，模型行为是否一致？

本文的核心假设是：如果在完全相同的临床资料下，仅改变社会人口属性就导致预测变化，则说明模型存在偏差。

## 方法详解

### 整体框架

构建了一个**Persona×Role**的评估矩阵：
- 25个Persona覆盖7个社会人口维度
- 3个用户角色（GP、专科、患者）
- 在1000条FAERS肿瘤学AE报告上评估

每条记录包含6个结构化变量：年龄、性别、体重、用药、疾病、不良事件。通过提示模板先建立"基线假设（答案为Yes）"，再注入角色+人设要求模型"基于更新上下文重新评估"。

### 关键设计

1. **Drug-Safety Decisions数据集（DSD）**：从FDA不良事件报告系统（FAERS）2024 Q4中构建。合并DEMO、DRUG、INDI、REAC四张表，仅保留肿瘤相关适应症、年龄≥18、无缺失值的记录，每条仅保留首个不良事件Preferred Term。最终选取前1000条作为轻量级可复现子集。

2. **25个社会人口Persona**：覆盖7个维度——教育水平（4级：低于高中到研究生）、婚姻状况（4类）、就业状况（3类）、保险类型（3类）、家庭语言（3种：阿拉伯语/西班牙语/英语）、住房稳定性（4级：无家可归到房主）、宗教信仰（4类）。这些属性与AE发生无临床因果关系，因此任何预测变异都意味着偏差。

3. **双偏差分析框架**：

    - **显式偏差（Explicit Bias）**：模型在推理过程中明确引用Persona属性（如"大学毕业生认知功能更好，跌倒风险更低"）
    - **隐式偏差（Implicit Bias）**：预测结果随Persona变化而变化，但推理中未提及Persona属性

   通过分析模型生成的解释文本，量化显式偏差出现频率；通过排除显式偏差案例后的准确率变化，评估隐式偏差的存在。

### 损失函数 / 训练策略

本文为评估框架，不涉及训练。两个被测模型：
- **ChatGPT-4o**：OpenAI通用模型，通过API访问
- **Bio-Medical-Llama-3-8B**：在生物医学语料上适配的Llama-3 8B变体，本地推理（NVIDIA 3090 Ti）

## 实验关键数据

### 主实验：社会人口属性对预测准确率的影响

| 维度 | Persona | ChatGPT-4o GP(%) | ChatGPT-4o Patient(%) | Bio-Med GP(%) | Bio-Med Patient(%) |
|---|---|---|---|---|---|
| 教育 | 低于高中 | 59.8 | 63.5 | **73.7** | 73.8 |
| 教育 | 研究生 | 54.8 | 50.8 | 45.9 | **43.4** |
| 住房 | 无家可归 | **76.3** | 72.2 | 73.0 | 72.0 |
| 住房 | 房主 | **51.8** | 57.1 | 50.1 | 47.7 |
| 保险 | 无保险 | 58.1 | 60.6 | **66.9** | 67.9 |
| 保险 | 私人保险 | 49.6 | 54.5 | 49.1 | **45.3** |
| 语言 | 阿拉伯语 | 68.6 | **80.4** | 60.6 | 59.5 |
| 语言 | 英语 | 67.0 | 76.8 | **40.8** | 37.0 |

### 消融实验：显式偏差频率与影响

| 维度 | ChatGPT-4o引用率(Specialist) | Bio-Med引用率(Specialist) | 排除后准确率变化 |
|---|---|---|---|
| 住房-临时住所 | **51.8%** | 32.0% | 排除后准确率提升 |
| 住房-无家可归 | 45.8% | 11.6% | 排除后准确率提升 |
| 宗教-Religious | **53.6%** | 3.9% | 排除后准确率提升 |
| 保险-私人保险 | 42.7% | 10.8% | 排除后准确率提升 |
| 就业-全职 | 43.8% | 8.1% | 排除后准确率提升 |
| 教育-研究生 | 47.1% | 9.5% | 排除后准确率提升 |

### 关键发现

1. **反直觉的准确率倒置**：弱势群体（低教育、无家可归、无保险）反而获得更高预测准确率，优势群体（研究生、房主、私人保险）反而更低
2. **ChatGPT-4o显式偏差严重**：在推理中引用社会人口属性的频率高达50%+（如"临时住所"51.8%、"宗教"53.6%），远高于Bio-Medical-Llama（通常<15%）
3. **荒谬的因果推理**：模型生成了如"大学毕业生认知能力更强所以跌倒风险更低"、"阿拉伯语使用者更容易腹痛"等无临床根据的推理
4. **用户角色影响显著**：Patient角色在ChatGPT-4o中一致获得最高准确率，角色间差异在多个维度达到统计显著（p<0.001）
5. **排除显式偏差后准确率提升**：证明身份引用的推理确实降低了预测质量
6. **隐式偏差同样存在**：即使不提及Persona属性，预测结果仍随属性变化，说明偏差嵌入在模型行为深处

## 亮点与洞察

- 首次在**药物安全预测**领域系统性审计LLM的社会人口偏差
- **显式vs隐式偏差**的概念框架具有通用性，可推广到其他高风险AI应用
- 揭示了一个深层矛盾：模型"知道"不该有偏差的领域，但仍然系统性地产生偏差
- Persona框架设计简洁优雅，可低成本复现和扩展

## 局限与展望

- 数据集仅1000条肿瘤学记录，规模小且领域窄
- 仅测试2个模型（ChatGPT-4o和Bio-Medical-Llama-3-8B）
- FAERS数据本身存在报告偏差、漏报和潜在重复
- 基线假设固定为"Yes"可能引入方向性偏差
- 记录选取基于顺序非随机采样，可能引入时间偏差
- 未探索反事实提示（counterfactual prompting）或校准等缓解策略

## 相关工作与启发

- **Omar et al. (2025)**：在急诊科场景评估LLM的社会人口偏差，本文扩展到药物警戒领域
- **Gupta et al. (2024)**：Persona分配LLM的隐式推理偏差
- **Pfohl et al. (2024)**：LLM健康公平性偏差检测工具箱
- 启发：在所有涉及临床决策的LLM应用中，公平性审计应成为部署前的必要环节

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在药物安全预测领域揭示Persona偏差，显式/隐式偏差分析有价值
- 实验充分度: ⭐⭐⭐ 数据量小、模型少，但实验设计严谨（25 Persona × 3角色 × 7维度）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格丰富，定性示例有说服力
- 价值: ⭐⭐⭐⭐ 对LLM在高风险医疗场景的公平部署具有警示意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](gflownets_for_learning_better_drug-drug_interaction_representations.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[NeurIPS 2025\] Exploring and Leveraging Class Vectors for Classifier Editing](exploring_and_leveraging_class_vectors_for_classifier_editing.md)
- [\[ACL 2026\] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](../../ACL2026/medical_imaging/faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)

<!-- RELATED:END -->
