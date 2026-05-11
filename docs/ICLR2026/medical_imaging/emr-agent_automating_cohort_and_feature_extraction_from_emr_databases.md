---
title: >-
  [论文解读] EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases
description: >-
  [ICLR 2026][医学图像][电子病历] 提出EMR-AGENT，首个基于LLM Agent的电子病历（EMR）自动化预处理框架，通过动态SQL交互替代手工规则编写，实现跨数据库的队列选择、特征提取和代码映射，在MIMIC-III/eICU/SICdb上表现优异并具强泛化能力。
tags:
  - ICLR 2026
  - 医学图像
  - 电子病历
  - LLM Agent
  - 队列选择
  - 特征提取
  - 代码映射
---

# EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases

**会议**: ICLR 2026  
**arXiv**: [2510.00549](https://arxiv.org/abs/2510.00549)  
**代码**: [有](https://github.com/AITRICS/EMR-AGENT)  
**领域**: 医学图像  
**关键词**: 电子病历, LLM Agent, 队列选择, 特征提取, 代码映射

## 一句话总结

提出EMR-AGENT，首个基于LLM Agent的电子病历（EMR）自动化预处理框架，通过动态SQL交互替代手工规则编写，实现跨数据库的队列选择、特征提取和代码映射，在MIMIC-III/eICU/SICdb上表现优异并具强泛化能力。

## 研究背景与动机

临床预测模型依赖从EMR中提取的结构化数据，但这一过程仍被硬编码、数据库特定的流水线主导，涉及队列定义、特征选择和代码映射三大步骤。这带来两大核心挑战：

**挑战1：跨机构语义和结构异质性**。不同医院EMR系统差异极大。例如"心率"在MIMIC-III中是`itemid=211`，在SICdb中是`HeartRateECG`，在eICU中是列名`heartrate`。这导致模型在跨数据库部署时的可比性和泛化性严重受限。

**挑战2：同一数据库内的不一致性**。同一临床概念可能有多种测量方式（如心率可通过传感器、听诊、触诊获得），导致多种代码映射。队列选择标准的模糊性（如"首次ICU入院"的不同解读）也造成不同研究选出不同患者群体。

现有解决方案（YAIB、ACES、BlendedICU等）要么依赖硬编码规则缺乏灵活性，要么依赖预定义输入格式限制泛化性。**核心问题**：能否用AI agent替代人工规则编写，实现自动化EMR预处理？

## 方法详解

### 整体框架

EMR-AGENT由两个LLM Agent组成，共享一个**Schema链接与指南生成**前端模块：

- **CFSA（队列与特征选择Agent）**：自动化患者队列选择和临床变量提取
- **CMA（代码映射Agent）**：标准化不同EMR系统的临床特征代码

**核心理念**：将SQL视为**探索和决策工具**，而非仅作为最终查询输出。Agent通过迭代观察查询结果、推理schema和文档来完成预处理任务。

### 关键设计

**1. Schema链接与指南生成**

与传统仅依赖schema信息的链接不同，该模块利用多种知识源（数据库手册、评估备忘录）增强schema理解：

- 为CFSA：明确每个schema组件的角色，标识缺失信息，规划SQL观测
- 为CMA：定义每个表和列的角色以实现准确的候选列表

**2. CFSA的三阶段交互循环**

- **SQL充分性评估**：判断当前schema和指南是否足以生成目标SQL。不足则生成观察SQL获取额外数据
- **数据充分性检查**：评估观察SQL返回的数据是否改善了schema理解
- **Schema更新**：将新获取的数据整合到schema和指南中

然后进入**SQL生成**和**错误反馈**循环：
- 语法错误 → 直接重新生成
- Schema不匹配（语法正确但语义错误）→ 回到Schema链接步骤
- 正确结果 → 完成提取

**3. CMA的候选匹配**

- **特征定位**：先在schema中直接搜索目标特征列名
- **候选列表**：若直接搜索失败，从schema确定可能包含特征的表和列，执行DISTINCT查询获取所有候选组合
- **目标与候选匹配**：批量比较用户请求特征与候选，计算相似度分数（0-100），双阈值策略（先80后90）

### 损失函数 / 训练策略

EMR-AGENT是基于推理的Agent框架，不涉及训练，而是依赖：

- **问题分解策略**：将复杂EMR预处理任务分解为可管理的子问题
- **温度调度**：CFSA允许最多10次观察，前5次温度0，之后每次+0.1增加探索性
- **错误反馈最多5次重试**
- **LLM骨干**：Claude-3.5-Sonnet作为主力模型

## 实验关键数据

### 主实验

**队列与特征选择（F1/Accuracy）**

| 方法 | MIMIC-III F1 | eICU F1 | SICdb F1 |
|------|:---:|:---:|:---:|
| **EMR-AGENT** | **0.940** | **0.929** | **0.814** |
| ICL(PLUQ) | 0.749 | 0.132 | 0.407 |
| DinSQL | 0.726 | 0.000 | 0.071 |
| REACT | 0.308 | 0.524 | 0.503 |
| ICL(SeqSQL) | 0.040 | 0.000 | 0.040 |

EMR-AGENT在所有数据库上大幅领先。eICU和SICdb上基线方法几乎完全失败（F1<0.53），而EMR-AGENT保持>0.81。

**代码映射（F1/Balanced Accuracy）**

| 方法 | MIMIC-III F1 | eICU F1 | SICdb F1 |
|------|:---:|:---:|:---:|
| **EMR-AGENT** | **0.516** | **0.648** | **0.536** |
| ICL(PLUQ) | 0.022 | 0.125 | 0.119 |
| REACT | 0.214 | 0.067 | 0.218 |

代码映射更具挑战性，但EMR-AGENT仍大幅领先（提升0.3-0.5 F1）。

### 消融实验

**CFSA组件消融**

| 组件 | MIMIC-III F1 | SICdb F1 |
|------|:---:|:---:|
| 完整系统 | 0.940 | 0.814 |
| 去掉SQL观察 | 0.916 | 0.795 |
| 去掉错误反馈 | 0.688 | 0.617 |
| 去掉全部DB交互 | 0.677 | 0.570 |
| 去掉SchemaGuideline | 0.827 | 0.792 |

**DB交互**是最关键组件。去掉Documents+Modules后CFSA在eICU上F1降为0，CMA全面崩溃。

**不同LLM骨干（SICdb）**

| LLM | CFSA F1 | CMA F1 |
|------|:---:|:---:|
| Claude-3.5-Sonnet | **0.81** | 0.54 |
| Claude-3.7-Sonnet | 0.80 | **0.63** |
| Claude-3.5-haiku | 0.74 | 0.44 |
| Qwen2.5-72B | 0.22 | 0.31 |
| Llama-3.1-70B | 0.18 | 0.14 |

开源模型（Qwen/Llama）性能远低于Claude系列，说明Agent能力强烈依赖骨干LLM的推理质量。

### 关键发现

1. 动态数据库交互（SQL观察+错误反馈）是性能的最大贡献者
2. 外部知识（数据库手册+评估备忘录）对CMA尤为关键
3. 在未见过的数据库（SICdb，晚于LLM训练数据截止日期）上仍有良好泛化
4. 代码映射固有困难（同一特征多种编码），F1~0.5-0.65已是显著进步

## 亮点与洞察

- **范式创新**：从手工编写规则到AI Agent动态交互的EMR预处理范式转变
- **SQL作为探索工具**：不同于Text-to-SQL的单次查询，Agent将SQL用于迭代观察、验证和决策
- **Schema Guideline方法**：结合多知识源的上下文感知schema理解，超越传统schema linking
- **配套基准PreCISE-EMR**：首个标准化EMR预处理评估协议，与临床专家合作构建
- **实际价值巨大**：医疗ML的数据预处理是真实瓶颈，自动化可极大提升效率

## 局限与展望

- 代码映射F1仍有提升空间（0.5-0.65），特别是同义临床概念的消歧
- 强烈依赖Claude系列LLM，开源模型效果差距大
- 仅评估ICU数据库（MIMIC-III/eICU/SICdb），未涉及门诊或专科EMR
- 56个特征限于生命体征和实验室结果，未覆盖用药、诊断代码、影像报告等
- 计算成本（多次LLM调用+SQL交互）未详细分析
- 需用户获取PhysioNet凭据才能复现，增加了准入门槛

## 相关工作与启发

- **vs YAIB/BlendedICU**：后者硬编码规则，新数据库需人工适配；EMR-AGENT自动适应
- **vs ACES/Clairvoyance**：后者依赖固定输入格式，EMR-AGENT直接与原始数据库交互
- **vs Text-to-SQL**（PLUQ/EHRSQL）：后者假设用户熟悉schema、只做单次查询；EMR-AGENT处理多轮探索和schema不确定性
- **vs EHRAgent**：后者做孤立的图表查询，EMR-AGENT做结构化预处理流水线
- **启发**：Agent驱动的数据获取可能成为医疗AI的新基础设施层

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★★ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era](../../CVPR2025/medical_imaging/nyxus_a_next_generation_image_feature_extraction_library_for_the_big_data_and_ai.md)
- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](../../CVPR2026/medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)

</div>

<!-- RELATED:END -->
