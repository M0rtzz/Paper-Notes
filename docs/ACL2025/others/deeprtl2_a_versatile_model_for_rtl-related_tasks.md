---
title: >-
  [论文解读] DeepRTL2: A Versatile Model for RTL-Related Tasks
description: >-
  [ACL 2025][RTL代码] DeepRTL2是首个统一处理RTL（寄存器传输级）相关生成任务与嵌入任务的LLM，通过精心构建的数据集和GRIT训练策略，在代码生成、代码理解、自然语言代码搜索、功能等价检查和性能预测五大任务上达到SOTA。 在EDA（电子设计自动化）领域，LLM已在RTL代码生成和理解等生成型任务上展…
tags:
  - "ACL 2025"
  - "RTL代码"
  - "LLM"
  - "嵌入任务"
  - "代码生成"
  - "GRIT训练"
---

# DeepRTL2: A Versatile Model for RTL-Related Tasks

**会议**: ACL 2025  
**arXiv**: [2506.15697](https://arxiv.org/abs/2506.15697)  
**代码**: 无（数据集和benchmark可能公开）  
**领域**: 电子设计自动化 (EDA) / 代码生成与理解  
**关键词**: RTL代码, LLM, 嵌入任务, 代码生成, GRIT训练

## 一句话总结

DeepRTL2是首个统一处理RTL（寄存器传输级）相关生成任务与嵌入任务的LLM，通过精心构建的数据集和GRIT训练策略，在代码生成、代码理解、自然语言代码搜索、功能等价检查和性能预测五大任务上达到SOTA。

## 研究背景与动机

在EDA（电子设计自动化）领域，LLM已在RTL代码生成和理解等生成型任务上展现出突破性能力。然而，同样关键的**嵌入型任务**却被长期忽视。这些任务包括：

**自然语言代码搜索**：设计师用自然语言查询大型RTL代码库，快速识别可复用模块

**功能等价检查**：快速评估两个设计是否功能等价，大幅减少验证时间

**性能预测**：在RTL阶段早期评估功率、性能、面积（PPA），指导优化

以往方法或专注于生成任务（如CodeV、DeepRTL），或使用设计特定的机器学习方法进行验证和预测，缺乏通用的RTL表示能力。DeepRTL2的核心动机是：**一个模型同时解决生成和嵌入两类任务**，提供EDA领域的综合解决方案。

## 方法详解

### 整体框架

DeepRTL2采用decoder-only架构（基于Llama-3.1 8B和DeepSeek-Coder 6.7B），通过两阶段训练策略同时学习生成和嵌入能力。生成侧处理代码生成和理解，嵌入侧处理搜索、等价检查和性能预测。

### 关键设计

1. **全面的数据集构建**:

    - **代码生成/理解**：从GitHub收集Verilog文件→按模块分割→MinHash去重→语法检查→GPT-4o链式思维标注（行级注释→模块级规范→高级功能描述）。还融合了RTLCoder、MG-Verilog、DeepCircuitX等开源数据
    - **自然语言代码搜索**：复用理解数据集，用GPT-4o将功能描述改写为用户查询格式（去除标识符、保留核心逻辑、清晰简洁）
    - **功能等价检查**：创新的反馈驱动CoT策略——用GPT-4o修改Verilog模块内部逻辑→Yosys进行逻辑等价检查→根据反馈迭代2-3轮→生成等价/不等价设计对。基于RTLLM v2.0的50个设计扩展至400对
    - **性能预测**：使用Yosys+SkyWater 130nm工艺库综合，ABC工具提取delay和area指标

2. **两阶段训练策略**:

    - **第一阶段**：课程学习（行级数据→模块级规范→高级描述→多样化prompt），仅训练生成/理解任务
    - **第二阶段**：结合GRIT框架，生成和嵌入任务联合训练。生成使用第一阶段第四子阶段的高质量数据，嵌入使用对比学习。先无hard negative，后加入hard negative

3. **GRIT训练框架适配**: 将Originally针对通用NLP的GRIT方法适配到RTL领域。核心思想是让decoder-only模型同时具备生成能力（自回归）和编码能力（固定长度向量），通过多任务训练目标函数实现。

4. **代码理解的标注质量保证**: 500个扩展benchmark经专业硬件设计师标注。GPT-4o的CoT标注策略比直接标注更准确，使得在CoT数据上训练的DeepRTL2在理解任务上超越了GPT-4o本身。

### 损失函数 / 训练策略

- 生成任务：标准自回归语言建模损失
- 嵌入任务：对比学习损失，包含hard negative mining
- 嵌入提取：使用cosine similarity计算向量间相似度
- 性能预测：嵌入→XGBoost回归模型

## 实验关键数据

### RTL代码生成（表1，pass@k）

| 模型 | syntax pass@1 | syntax pass@5 | function pass@1 | function pass@5 |
|------|-------------|-------------|----------------|----------------|
| GPT-4o | 72.00% | 77.31% | 49.70% | 56.80% |
| o1-preview | 76.20% | 83.71% | 50.00% | 60.86% |
| DeepRTL2 (Llama) | 68.30% | **81.31%** | 33.70% | 49.57% |
| DeepRTL2 (DeepSeek) | **71.60%** | 80.58% | 38.50% | **52.62%** |
| Llama-3.1 base | 32.40% | 57.01% | 14.60% | 26.04% |

### 嵌入任务性能（表2、4、5）

**自然语言代码搜索 (F1):**

| 模型 | F1 |
|------|-----|
| text-embedding-3-large | 0.290 |
| GritLM-7B | 0.269 |
| DeepRTL2 (Llama) no-hard | **0.476** |
| DeepRTL2 (DeepSeek) | 0.453 |

**功能等价检查 (Average Precision):**

| 模型 | AP |
|------|------|
| text-embedding-3-small | 0.565 |
| GritLM-7B | 0.541 |
| DeepRTL2 (Llama) | **0.667** |

**性能预测 (Area, r2_score):**

| 模型 | r2_score | MAPE |
|------|---------|------|
| text-embedding-3-large | 0.699 | 4.446 |
| DeepRTL2 (DeepSeek) | **0.773** | **1.598** |

### 关键发现

1. **生成与嵌入的统一不冲突**：DeepRTL2在生成任务上达到开源SOTA（超越CodeV和原DeepRTL），同时在嵌入任务上大幅超越通用嵌入模型
2. **领域特定嵌入远超通用模型**：在代码搜索任务上，DeepRTL2的F1（0.476）比OpenAI text-embedding-3-large（0.290）高出64%
3. **CoT标注的学生超越老师**：在代码理解任务上，DeepRTL2超越了其标注数据来源GPT-4o，因为训练数据使用了更精细的CoT标注
4. **课程学习和数据多样性的贡献**：从base模型到DeepRTL2的巨大提升证明了数据构建和训练策略的有效性

## 亮点与洞察

- **首创性**：首个统一RTL生成和嵌入任务的模型，填补了EDA领域嵌入任务的空白
- **数据工程的重视**：反馈驱动的CoT策略生成等价/不等价代码对是一个精巧的设计——利用Yosys作为验证器在循环中确保数据质量
- **实际价值**：自然语言搜索可加速代码复用、等价检查可减少验证时间、性能预测可指导早期优化——这三个任务直接对应硬件设计的核心痛点
- **超越标注者**：通过CoT增强的训练数据使模型在理解任务上超越GPT-4o，这一"学生超越老师"的现象值得关注

## 局限与展望

- 仅限Verilog语言，未涵盖VHDL或SystemVerilog等其他HDL
- 生成benchmark可能与OpenAI模型训练数据存在重叠，公平性需考量
- 功能等价检查的AP仅0.667，实际应用中的精度要求更高
- 性能预测仅预测area和delay，未覆盖power（需要workload信息）
- 嵌入维度和模型大小可能限制部署效率

## 相关工作与启发

- **DeepRTL (Liu et al., 2025)**：本文的前身，仅处理生成任务，DeepRTL2扩展到嵌入领域
- **GRIT (Muennighoff et al., 2025)**：统一生成与表示学习的训练策略，本文将其从通用NLP迁移到RTL领域
- **CodeV (Zhao et al., 2024)**：RTL代码生成的竞争模型，DeepRTL2在多数指标上超越
- RTL领域与通用代码生成的关键差异：Verilog的语义结构化更强，功能验证可通过EDA工具自动化

## 评分

- **新颖性**: 8/10 — 首个统一RTL生成+嵌入的模型，嵌入任务的benchmark构建也是新贡献
- **实验充分度**: 8/10 — 覆盖五大任务、多个baseline、多种评估指标，但缺少对训练策略的消融
- **写作质量**: 7/10 — 结构清晰但内容密集，部分细节需参考附录
- **价值**: 8/10 — 对EDA领域有直接推动作用，为RTL相关的AI应用提供了统一框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)
- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)
- [\[ACL 2025\] Model Extrapolation Expedites Alignment](expo_model_extrapolation.md)

</div>

<!-- RELATED:END -->
