---
title: >-
  [论文解读] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration
description: >-
  [NeurIPS 2025][LLM Agent][轨迹建模] 提出 TrajAgent——一个基于 LLM Agent 的轨迹建模框架，通过统一环境 UniEnv、自动化工作流和大小模型协作学习机制，实现跨任务、跨数据集的自动化轨迹建模，在多项任务上超越基线方法 2.38%–69.91%。
tags:
  - NeurIPS 2025
  - LLM Agent
  - 轨迹建模
  - LLM代理
  - 大小模型协作
  - 自动化机器学习
  - 数据增强
---

# TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration

**会议**: NeurIPS 2025  
**arXiv**: [2410.20445](https://arxiv.org/abs/2410.20445)  
**代码**: [GitHub](https://github.com/tsinghua-fib-lab/TrajAgent)  
**领域**: LLM Agent  
**关键词**: 轨迹建模, LLM代理, 大小模型协作, 自动化机器学习, 数据增强

## 一句话总结

提出 TrajAgent——一个基于 LLM Agent 的轨迹建模框架，通过统一环境 UniEnv、自动化工作流和大小模型协作学习机制，实现跨任务、跨数据集的自动化轨迹建模，在多项任务上超越基线方法 2.38%–69.91%。

## 研究背景与动机

**轨迹建模的广泛应用**：轨迹建模涵盖轨迹数据的模式挖掘和未来预测，在生活服务、城市交通、公共管理等领域有广泛应用。典型任务包括轨迹预测、轨迹恢复、轨迹分类、轨迹生成和轨迹表示学习。

**现有方法的根本困境**：

1. **任务碎片化**：每种方法只针对特定任务和数据格式设计。例如 TrajFormer 只能做轨迹分类，Flash-back 只适用于稀疏 check-in 轨迹预测，无法跨任务迁移
2. **数据异构性**：轨迹数据在分辨率、格式和地理区域上差异巨大，模型难以跨数据集复用
3. **统一框架不成熟**：已有的统一框架（如 TrajFM、UniTraj）在单任务性能上落后于专用模型，且训练过程复杂

**LLM Agent 的机遇**：LLM 凭借强大的推理和常识能力，已在自动化软件开发（MetaGPT、ChatDev）和自动化 ML（HuggingGPT、MLAgentBench）中取得成功。能否将 LLM Agent 引入轨迹建模，实现自动化和统一化？

## 方法详解

### 整体框架

TrajAgent 包含三大核心组件：

1. **UniEnv**：统一运行环境，提供标准化的数据和模型接口
2. **Agentic Workflow**：四阶段自动化工作流（理解→规划→执行→总结）
3. **Collaborative Learning Schema**：大模型（Agent）与小模型（专用模型）的协作学习机制

### 关键设计

**1. UniEnv 统一环境**

- **任务接口**：覆盖 5 大轨迹建模任务（预测、恢复、分类、生成、表示），共 9 个子任务、18 种方法
- **数据接口**：支持 Check-in 轨迹和 GPS 轨迹两种格式，通过 LLM 生成的预处理脚本实现数据清洗和标准化
- **模型接口**：集成 18 个模型（GETNext、DeepMove、TrajBERT 等），每个模型都附有语义描述便于 Agent 选择
- **外部工具**：集成论文上下文提取（txyz.ai）、超参优化（Optuna）、轨迹可视化（movingpandas）等

**2. 四阶段 Agentic Workflow**

- **任务理解模块**：接收自然语言指令，识别任务类型和关键信息
- **任务规划模块**：根据任务描述和 UniEnv 中的数据/模型信息，生成执行计划（选择数据集和模型）
- **任务执行模块**：调用 UniEnv 执行实验，并与协作学习模块交互
- **任务总结模块**：分析执行记录，生成优化总结报告

每个模块均配备 Reflexion 式的记忆（Memory）和反思（Reflection）机制。

**3. Collaborative Learning Schema 协作学习**

分为两个层面的学习：

- **Agent Learning via Reasoning**（高层）：Agent 基于实验记录进行推理学习
    - 设计 "think then action" 两阶段流程
    - 维护长期记忆（全部实验数据）和短期记忆（历史动作）
    - 支持对比反思（Contrastive Reflection）和动态记忆修剪（Dynamic Memory Pruning）

- **Model Learning via Training**（底层）：专用模型在目标数据上进行针对性训练
    - **数据增强**：定义 10 种轨迹增强算子（insert、replace、split 等），Agent 选择最优算子组合
    - **参数优化**：Agent 读取模型参数配置文件并生成代码更新参数
    - **联合优化**：先数据增强后参数优化的顺序执行

### 损失函数 / 训练策略

TrajAgent 本身不定义损失函数，而是编排已有模型的训练。协作学习的停止条件为：性能达到预设要求或达到最大探索轮数。

关键改进措施：
- **对比反思**：Agent 显式比较成功和失败的实验，调整算子参数以避免重复无效组合
- **动态记忆修剪**：定期丢弃低分记忆条目，仅保留高性能轨迹作为指导

## 实验关键数据

### 主实验 (含表格)

在 4 个真实数据集、5 大任务上的表现：

| 任务 | 子任务 | 模型 | 数据集 | 原始性能 | TrajAgent (联合优化) | 提升 |
|-----|-------|-----|-------|---------|-------------------|-----|
| 轨迹预测 | 下一位置预测 | GETNext | FSQ | 0.3720 (Acc@5) | 0.4002 | 7.58% |
| 轨迹预测 | 下一位置预测 | LLM-ZS | FSQ | 0.3110 (Acc@5) | 0.3350 | 7.72% |
| 轨迹预测 | 旅行时间估计 | MulT-TTE | Porto | 163.12 (MAE) | 128.57 | 21.18% |
| 轨迹恢复 | 恢复 | TrajBERT | Porto | 42.71 (MAE) | 27.78 | 34.96% |
| 轨迹恢复 | 地图匹配 | GraphMM | Tencent | 0.2014 (Acc) | 0.3422 | **69.91%** |
| 轨迹分类 | 用户关联 | S2TUL | FSQ | 0.5755 (Acc@5) | 0.7802 | 35.57% |
| 轨迹分类 | 意图预测 | LIMP | Beijing | 0.745 (Acc) | 0.7627 | 2.38% |
| 轨迹生成 | 生成 | DSTPP | Earthquake | 0.4611 (MAE) | 0.3584 | 22.27% |

### 消融实验 (含表格)

**LLM 后端对比**（以 DeepMove + 下一位置预测为例）：

| LLM | 整体成功率 | 联合优化 Acc@5 |
|-----|:--------:|:------------:|
| Qwen2-7B | 较低 | 0.2668 |
| Mistral-7B-V3 | 中等 | 0.2980 |
| GPT-3.5-Turbo | 高 | 0.3295 |
| Qwen2-72B | **最高** | **0.4333** |
| GPT-4o-mini | 高 | 0.3724 |

**工作流组件消融**：

| 配置 | 模型选择 Acc | 联合优化 Acc@5 |
|-----|:----------:|:------------:|
| 完整 TrajAgent | 98% | 0.3724 |
| 去除 Reflection | 95%↓ | 0.3212↓ |
| 去除 Memory | 80%↓ | 0.1804↓ |

### 关键发现

1. **跨任务、跨模型的一致提升**：TrajAgent 在所有测试配置下均带来正向提升，证明框架的通用性
2. **LLM 能力是关键**：72B 级别模型（Qwen2-72B）显著优于 7-9B 模型，尤其是在数据增强和参数优化阶段
3. **记忆机制至关重要**：去除记忆模块后成功率和性能均大幅下降
4. **优化陷阱现象**：过多的推理步骤（>20）或记忆条目（>10）反而导致性能下降
5. **对比 Optuna**：TrajAgent 用更少的试错迭代达到更好的性能，且通过联合优化额外提升 11.1%

## 亮点与洞察

1. **统一框架范式**：首个将 LLM Agent 应用于统一轨迹建模的框架，覆盖 5 大任务类别
2. **大小模型协作**：LLM 负责高层推理和策略决策，专用小模型负责底层训练执行，各取所长
3. **闭环优化系统**：Agent 推理→模型训练→性能反馈→Agent 学习的闭环，实现自动化优化
4. **失败模式分析深入**：详细分析了优化陷阱和记忆污染问题，并给出实用的缓解策略（对比反思 + 记忆修剪）

## 局限与展望

1. **Agent 效率问题**：多轮推理和训练迭代带来较大的计算开销和 API 调用成本
2. **弱 LLM 性能差**：框架高度依赖高能力 LLM，7B 级模型在关键阶段成功率低
3. **数据增强策略有限**：check-in 轨迹的 10 种增强算子是预定义的，缺乏自适应生成新算子的能力
4. **地理泛化性**：实验主要在有限的城市/数据集上进行，跨区域泛化能力未充分验证
5. **可解释性不足**：Agent 的决策过程（如为何选择某个模型或增强算子）缺乏自然语言解释输出

## 相关工作与启发

- **HuggingGPT / VisionLLM**：利用 LLM 管理和调度多种 AI 模型的先驱工作
- **Reflexion**：Agent 通过语言反思进行强化学习的框架，TrajAgent 的记忆和反思机制直接借鉴于此
- **AutoML (Optuna)**：传统参数优化方法，TrajAgent 通过 LLM 推理超越了纯搜索的方法
- **CityGPT / UrbanLLM**：城市计算领域的 LLM 应用，但主要依赖微调而非 Agent 框架
- 对 Agent 框架设计的启示：记忆管理（大小和内容）是影响 Agent 系统性能的关键因素

## 评分

⭐⭐⭐⭐ (4/5)

理由：论文提出了一个完整且系统的 LLM Agent 框架用于轨迹建模，实验覆盖面广（5 大任务、18 个模型、4 个数据集），性能提升显著（最高 69.91%）。大小模型协作学习的设计理念新颖，失败模式分析深入且实用。主要不足在于框架对高能力 LLM 的强依赖性、计算开销较大，以及实验中参与的数据集和城市有限。整体而言是一项扎实且有实用价值的系统工作。

<!-- RELATED:START -->

## 相关论文

- [Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

<!-- RELATED:END -->
