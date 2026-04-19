---
title: >-
  [论文解读] Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives
description: >-
  [ACL 2025][LLM/NLP][Chain-of-Thought] 借鉴 Bloom 分类学和知识空间理论，提出 Re-TASK 框架将 LLM 任务从"能力项-技能-知识"三层视角进行重新审视，并设计 Re-TASK prompting 策略通过针对性的知识注入和技能适配来增强 CoT 在领域任务上的表现，在法律任务上最高提升 45%。
tags:
  - ACL 2025
  - LLM/NLP
  - Chain-of-Thought
  - 能力分解
  - 知识注入
  - 技能适配
  - Bloom分类学
  - 知识空间理论
  - 领域任务
---

# Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives

**会议**: ACL 2025  
**arXiv**: [2408.06904](https://arxiv.org/abs/2408.06904)  
**作者**: Zhihu Wang (Huawei), Shiwan Zhao (Nankai Univ.), Yu Wang (Xi'an Jiaotong Univ.) 等
**代码**: [GitHub](https://github.com/Uylee/Re-TASK)  
**领域**: llm_nlp  
**关键词**: Chain-of-Thought, 能力分解, 知识注入, 技能适配, Bloom分类学, 知识空间理论, 领域任务

## 一句话总结

借鉴 Bloom 分类学和知识空间理论，提出 Re-TASK 框架将 LLM 任务从"能力项-技能-知识"三层视角进行重新审视，并设计 Re-TASK prompting 策略通过针对性的知识注入和技能适配来增强 CoT 在领域任务上的表现，在法律任务上最高提升 45%。

## 研究背景与动机

### 问题背景
Chain-of-Thought（CoT）已成为 LLM 解决复杂问题的核心范式，通过将复杂任务分解为子任务序列来实现分而治之。然而在领域特定任务（法律、金融、STEM）上，CoT 经常失败——LLM 既难以准确分解任务，也无法有效执行子任务，根本原因在于缺乏领域知识和专业能力。

### 已有工作不足
- **CoT 只提供工作流视角**：关注"如何分解步骤"，忽略了每一步所依赖的具体能力、知识和技能
- **RAG 只解决知识缺失**：检索增强生成可以注入知识，但模型可能缺乏将检索到的知识有效应用的技能适配能力
- **Few-shot ICL 示例缺乏针对性**：随机选取的 few-shot 示例未必能覆盖任务所需的关键能力项
- **已有评测框架如 KoLA 停留在评估层面**，未提出系统化的能力增强方法

### 核心动机
从教育学理论出发，提出一种全新的**学习链（Chain-of-Learning）**视角：任务的成功完成依赖于顺序掌握多个能力项，每个能力项进一步由知识和技能构成。通过识别和强化这些能力项，可以系统性地提升 CoT 性能。

## 方法详解

### 整体框架：Re-TASK 理论模型

Re-TASK 框架将 LLM 任务分解为四个层次概念：

1. **任务（Task）**：从输入 $x$ 到输出 $y$ 的映射 $\mathbf{T}(ctx; I; x) = y$，其中 $I$ 为指令，$ctx$ 为可选上下文
2. **能力项（Capability Item）**：完成任务所需的具体演示或练习，引导 LLM 将特定技能应用于相关知识，实现知识-技能适配
3. **知识（Knowledge）**：包含事实性、概念性和程序性三类领域知识（对应 Bloom 分类学的知识维度）
4. **技能（Skill）**：对应 Bloom 分类学的认知过程维度，包括记忆/检索、理解、应用等

任务 $\mathbf{T}$ 的完成需要顺序掌握多个能力项 $C_{ij}$，其中 $i$ 表示子任务编号，$j$ 表示该子任务关联的能力项索引。$C_{01}$ 代表整体程序性知识，$C_{02}$ 代表对该知识的应用（类似 CoT 过程）。

### 关键设计 1：能力项构建

三类核心能力项的识别与构建：

- **知识检索（Knowledge Retrieval）**：识别任务相关知识点并从外部源检索，或回忆 LLM 内部存储的知识。知识本身被视为一种特殊能力项（技能默认为"回忆/检索"）
- **概念知识实例化（Instances of Conceptual Knowledge）**：通过具体示例阐释概念性知识，强化理解
- **程序性知识执行（Execution of Procedural Knowledge）**：演示如何按有序步骤执行程序性知识，如法律量刑推理流程

构建流程：先用大模型进行任务分解获得整体程序性知识 $C_{01}$，再生成 CoT 演示作为知识应用能力项 $C_{02}$，最后为每个子任务生成相关知识 $C_{i1}$ 和应用演示 $C_{i2}$。

### 关键设计 2：Re-TASK Prompting 策略

设计了两个版本的 prompting 策略：

**Re-TASK (Lite)**：仅包含整体任务的能力项
- 将整体程序性知识 $C_{01}$（知识注入）和其应用演示 $C_{02}$（技能适配）放入 prompt
- 相当于"单个演示 + 结构化知识"，token 开销与 One-shot CoT 相当

**Re-TASK (Full)**：包含所有可用能力项
- 按依赖关系（学习链）排列各子任务的能力项 $C_{ij}$
- 对每个子任务，先放知识回忆类能力项，再放理解/应用类能力项
- 最后放整体程序性知识 $C_{01}$ 和应用 $C_{02}$

### 关键设计 3：CoT 失败归因分析

框架将 CoT 失败归因为两类能力缺陷：
- **知识不足**：LLM 缺乏领域专有数据或数据时效性问题导致知识缺失
- **技能适配不足**：即使知识可用，LLM 也无法有效应用——这解释了为什么单纯 RAG 注入知识效果有限，还需要技能适配的演示

## 实验关键数据

### 法律领域（量刑预测任务，CAIL 数据集）

| 方法 | Llama3-Chinese-8B | Yi-1.5-9B | Qwen1.5-7B | 平均提升 |
|------|-------------------|-----------|------------|---------|
| Zero-shot CoT | 54.00 | 40.00 | 33.50 | - |
| Zero-shot CoT + SC | 54.50 | 40.50 | 33.50 | +0.33 |
| One-shot CoT | 53.67 | 66.50 | 36.17 | +9.61 |
| Three-shot CoT | 56.33 | 70.17 | 38.50 | +12.50 |
| Step-Back | 72.50 | 72.50 | 36.50 | +18.00 |
| Re-TASK (+K0, 仅知识) | 60.50 | 57.50 | 44.00 | +11.50 |
| **Re-TASK (Lite)** | **78.50** | **85.00** | **45.50** | **+27.17** |

法律领域改善最显著，Re-TASK (Lite) 在 Yi-1.5-9B 上提升 45 个百分点，平均提升 27.17%，远超所有基线。

### 金融领域（FinanceIQ 考试任务）

| 方法 | Llama3-Chinese-8B | Yi-1.5-9B | Qwen1.5-7B | 平均提升 |
|------|-------------------|-----------|------------|---------|
| Zero-shot CoT | 36.52 | 53.93 | 43.82 | - |
| Three-shot CoT | 34.27 | 63.82 | 46.07 | +3.30 |
| Step-Back | 30.90 | 66.85 | 44.38 | +2.62 |
| Re-TASK (Lite) | 38.20 | 61.80 | 49.44 | +5.06 |
| **Re-TASK (Full)** | **52.81** | **73.60** | **51.69** | **+14.61** |

金融领域 Re-TASK (Full) 平均提升 14.61%，远超 Three-shot CoT 的 3.30%，验证了子任务级别能力项的额外价值。

### STEM 领域（MMLU 数学/物理/生物）

| 领域 | 方法 | Llama3-8B | Mistral-7B | Qwen1.5-7B | 平均提升 |
|------|------|-----------|------------|------------|---------|
| 数学 | Zero-shot CoT | 40.58 | 24.28 | 36.96 | - |
| 数学 | Re-TASK (Lite) | 51.81 | 28.99 | 43.84 | +7.61 |
| 物理 | Zero-shot CoT | 57.84 | 37.25 | 42.16 | - |
| 物理 | Re-TASK (Lite) | 60.78 | 44.12 | 50.98 | +6.21 |
| 生物 | Zero-shot CoT | 76.39 | 57.64 | 59.72 | - |
| 生物 | Re-TASK (Lite) | 88.19 | 79.17 | 81.25 | +18.29 |

生物领域提升最大（+18.29%），Step-Back 在 STEM 上表现反而显著下降（-15.28%），说明小模型自身生成的高层原则质量不可靠。

## 关键发现

1. **知识注入必要但不充分**：Re-TASK (+K0) 仅注入知识，平均提升 11.50%；加上技能适配演示后（Re-TASK Lite），提升跃至 27.17%，说明技能适配是关键瓶颈
2. **能力项比随机示例更有效**：在相同 token 预算下，Re-TASK (Lite) 使用 1 个能力项演示即优于 One-shot CoT 的随机演示；Re-TASK (Full) 使用 3 个能力项组合远超 Three-shot CoT（14.61% vs 3.30%）
3. **模型规模增大不能替代能力增强**：Qwen1.5 从 7B→14B→32B 扩展中，Zero-shot CoT 和 Re-TASK (Lite) 同步提升，说明 Re-TASK 的增益在大模型上依然有效
4. **领域知识密集度决定提升幅度**：法律（+27.17%）> 生物（+18.29%）> 数学（+7.61%），知识越密集、越专业的领域，Re-TASK 提升越显著
5. **Step-Back 在小模型上不稳定**：小模型难以生成有效的高层"退后原则"，导致 STEM 上准确率大幅下降

## 亮点与洞察

- **教育理论与 LLM 的创新结合**：将 Bloom 分类学的"知识维度×认知过程维度"矩阵和知识空间理论的学习路径概念引入 LLM 任务分析，提供了超越单纯 prompt 工程的理论框架
- **Chain-of-Learning 概念新颖**：相对于 CoT 的"工作流视角"，CoL 提供了"学习视角"，揭示了任务能力的层次依赖结构
- **实用性强**：能力项构建流程简单（先分解任务→识别知识→生成演示），可直接应用于任何领域任务
- **效率良好**：Re-TASK (Lite) 的 token 开销与 One-shot CoT 相当，但效果远优，性价比极高
- **诊断价值**：框架不仅能增强性能，还能诊断 CoT 失败的根因——是知识缺失还是技能适配不足

## 局限性

- **能力项生成未优化**：依赖大模型直接生成能力项，没有引入检索过程或离线知识库匹配，实际部署中需要 RAG 配合
- **跨领域差异未深入分析**：不同领域提升幅度差异大（法律 27% vs 数学 7%），但未系统分析原因
- **仅测试开源小模型**：实验限于 7B-32B 级别的开源模型，未在 GPT-4 等闭源大模型上验证
- **能力项依赖人工预定义类型**：需要为每个任务预先定义能力项类别（知识检索/理解/应用），自动化程度有限
- **法律/金融数据集规模较小**：法律 200 条、金融 178 条测试样本，统计显著性有待更大规模验证

## 相关工作与启发

- **Bloom 分类学 → LLM 能力建模**：KoLA benchmark 已采用 Bloom 理论的简化版（记忆/理解/应用/创造四层）做 LLM 评估，Re-TASK 进一步将其用于能力增强
- **Skill-it（Chen et al. 2024）**：从训练数据角度形式化技能概念和技能顺序学习，Re-TASK 则从推理时（inference-time）的 prompt 设计角度实现技能增强
- **RAG → 知识注入的一种特例**：Re-TASK 将 RAG 视为"知识检索"这一特殊能力项，并指出仅有知识注入不够，还需技能适配
- **对 CoT 研究的启发**：现有 CoT 改进（自一致性、Plan-and-Solve、Step-Back）大多停留在工作流层面，Re-TASK 提示了从能力依赖角度重新设计 prompt 的新范式

## 评分

- 新颖性: ⭐⭐⭐⭐ — 教育理论与 LLM prompt 工程的跨学科融合独特，Chain-of-Learning 概念有新意
- 实验充分度: ⭐⭐⭐ — 覆盖 5 个数据集 3 个领域，但数据集规模偏小，未在闭源大模型上验证
- 写作质量: ⭐⭐⭐⭐ — 理论框架层次清晰，定义严谨，但部分概念（如能力项与 few-shot 示例的本质区别）可更明确
- 价值: ⭐⭐⭐⭐ — 提供了分析和增强 LLM 领域任务能力的系统性方法论，实用且易推广

<!-- RELATED:START -->

## 相关论文

- [Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)
- [Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)
- [Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)

<!-- RELATED:END -->
