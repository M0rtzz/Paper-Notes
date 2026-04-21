---
title: >-
  [论文解读] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent system] 提出Traj-CoA多agent框架，通过chain-of-agents架构配合EHRMem长期记忆模块对长且噪声的纵向EHR进行时序推理，在零样本肺癌风险预测任务中（5年EHR数据，最高160k tokens）超越ML/DL/BERT/LLM等多类基线。
tags:
  - NeurIPS 2025
  - LLM Agent
  - multi-agent system
  - EHR
  - patient trajectory
  - temporal reasoning
  - lung cancer prediction
---

# Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2510.10454](https://arxiv.org/abs/2510.10454)  
**代码**: 无  
**领域**: llm_agent  
**关键词**: multi-agent system, EHR, patient trajectory, temporal reasoning, lung cancer prediction

## 一句话总结
提出Traj-CoA多agent框架，通过chain-of-agents架构配合EHRMem长期记忆模块对长且噪声的纵向EHR进行时序推理，在零样本肺癌风险预测任务中（5年EHR数据，最高160k tokens）超越ML/DL/BERT/LLM等多类基线。

## 研究背景与动机
**领域现状**: 纵向电子健康记录（EHR）包含丰富的时序数据，可用于患者轨迹建模和临床结局预测。LLM在零样本临床预测方面展现出潜力，有望替代需要复杂特征工程和任务专用训练的传统方法。

**现有痛点**: EHR数据面临两大核心挑战——(1) **极长上下文**: 患者历史跨越数年，token数常超过128k，超出LLM有效处理范围，且存在"lost-in-the-middle"问题；(2) **固有噪声**: EHR本身为临床护理而非研究设计，包含格式不一致、录入错误、缺失数据、不规则采样，大量无关信息掩盖关键预测信号。

**核心矛盾**: 现有LLM方法局限于短EHR（<16k tokens）或ICU数据，对>32k甚至128k tokens的长纵向EHR的时序推理仍是未解难题。直接扩大上下文窗口反而会降低性能。

**本文目标**: 如何在不进行额外训练的情况下（零样本），对极长且噪声的纵向EHR数据进行有效时序推理？

**切入角度**: 借鉴chain-of-agents多agent协作架构，结合专为EHR设计的外部长期记忆模块，将长上下文推理分解为多agent协作的短上下文推理链。

**核心 idea**: Worker agent链逐chunk处理时间感知分段的EHR + EHRMem记忆关键临床事件 + Manager agent综合摘要和记忆做预测。

## 方法详解

### 整体框架
Traj-CoA包含三个核心组件：(1) 数据预处理流水线（XML统一格式 + 时间感知分块）；(2) Chain-of-Agents工作流（Worker agents序列处理 + Manager agent最终决策）；(3) EHRMem长期记忆模块。

### 关键设计
1. **数据统一化与时间感知分块（Data Preprocessing）**:

    - **XML统一格式**: 将患者全部多模态历史（诊断码、实验室结果、生命体征、临床笔记、影像报告）转换为单一XML格式。按时间顺序组织，根节点包含人口统计信息，后续为时间戳嵌套的事件记录。利用LLM对结构化标签数据的良好理解能力。
    - **时间感知分块**: 不同于固定大小的硬分块，按时间戳为单位动态分段，在最大k tokens限制下保持时间完整性。当单个时间戳记录超过k tokens时，进一步拆分但保留原始时间戳。最终生成C个时间上连贯的chunk {c₁, c₂, ..., c_C}，C因患者而异。

2. **Chain-of-Agents工作流**:

    - **Stage 1 - Worker Agents**: 一系列Worker agent W_i 序列处理每个chunk c_i。每个agent接收当前chunk、任务指令I_W和前一agent的摘要S_{i-1}，提取任务相关显著信息、分析时序模式、生成更新摘要S_i。公式：S_i = W_i(I_W, S_{i-1}, c_i)。实现跨整个纵向EHR的渐进式信息聚合。
    - **Stage 2 - Manager Agent**: 接收最终Worker的摘要S_C和任务指令I_M，综合信息产生最终输出O。公式：O = M(I_M, S_C)。

3. **EHRMem长期记忆模块**:

    - **设计动机**: 直接应用vanilla CoA会导致关键临床事件在长序列传递中逐步抽象和丢失（"遗忘"问题）。
    - **机制**: 结构化存储任务相关事件及其时间戳。Worker agent在处理每个chunk时提取可能相关的临床事件/风险因素，存入记忆ℳ。采用去重机制：每个agent的提示包含ℳ中最近k个事件，仅存储新的未记录信息，防止EHR"copy-forwarding"导致的冗余。
    - **增强推理**: Manager agent决策同时基于最终摘要S_C和完整记忆ℳ。公式：S_i, E_i = W_i(I_W, S_{i-1}, c_i, ℳ[-k:]); ℳ ← ℳ ⊕ E_i; O = M(I_M, S_C, ℳ)。
    - **包容性提取策略**: Worker agent有意识地提取略宽泛的可能相关事件集，而非严格筛选。因为局部agent缺乏全局上下文来判断事件的长期重要性，将最终重要性判断委托给拥有完整时间上下文的Manager agent。

### 损失函数 / 训练策略
- **零样本设置**: 无需训练，完全通过任务特定指令（prompts）驱动
- 基础模型：MedGemma-27B
- 默认chunk大小：8k tokens，最多15个chunks（支持最高120k tokens上下文）
- 对于需要训练的基线（ML/DL/BERT），使用12,266个训练样本和1,363个验证样本

## 实验关键数据

### 主实验（300个测试样本，28例阳性/272例对照）

| 模型家族 | 模型 | 预测方式 | 上下文窗口 | AUROC | Precision | Recall | F1 |
|---------|------|---------|-----------|-------|-----------|--------|-----|
| ML | LR | SFT | — | 0.741 | 0.306 | 0.393 | 0.344 |
| ML | XGBoost | SFT | — | 0.763 | 0.367 | 0.393 | 0.379 |
| DL | RETAIN | SFT | — | 0.757 | 0.346 | 0.321 | 0.333 |
| DL | PatientTM | SFT | — | 0.730 | 0.361 | 0.464 | 0.406 |
| BERT | C-MBERT | SFT | 8k | 0.749 | 0.367 | 0.393 | 0.379 |
| LLM | Vanilla | Zero-shot | 32k | 0.743 | 0.345 | 0.357 | 0.351 |
| LLM | RAG | Zero-shot | 1k×32 | 0.753 | 0.221 | 0.607 | 0.324 |
| LLM | Traj-CoA (w/o EHRMem) | Zero-shot | 8k×15 | 0.748 | 0.183 | 0.821 | 0.299 |
| LLM | **Traj-CoA** | **Zero-shot** | **8k×15** | **0.766±0.019** | **0.358±0.057** | **0.436±0.105** | **0.380±0.018** |

### 消融实验与敏感性分析

| 分析维度 | 设置 | AUROC | 关键发现 |
|---------|------|-------|---------|
| EHRMem消融 | 移除EHRMem | 0.748 | AUROC下降1.8%，F1下降8.1%，证明长期记忆不可或缺 |
| Chunk大小 | 2k（固定80k总量） | 较低 | 过长的agent链导致灾难性遗忘 |
| Chunk大小 | 8k（固定80k总量） | 峰值 | 平衡局部细节保持和全局聚合的最优点 |
| Chunk大小 | 16k（固定80k总量） | 较低 | 短链但每个agent面临"lost-in-the-middle" |
| 上下文窗口 | 40k→160k（8k chunk） | 持续提升 | 区别于vanilla LLM在64k时性能下降的趋势 |
| Vanilla LLM | 32k→64k | 0.743→0.714 | 扩大窗口反而降低性能 |

### 关键发现
- **Traj-CoA在零样本设置下超越大多数监督训练基线**，AUROC 0.766与最佳SFT基线XGBoost（0.763）可比
- **EHRMem是性能关键组件**: 移除后F1从0.380暴跌至0.299，证明纯摘要传递会丢失关键信号
- **独特的长上下文扩展能力**: Traj-CoA性能随上下文窗口扩大至160k持续提升，而vanilla LLM在64k时已退化
- **时序推理具有临床相关性**: 模型识别的显著事件涵盖诊断、症状、实验室检查等7大类别，Top主题（高龄、贫血、COPD、咳嗽、炎症标志物、肺结节、肺炎、肺功能、吸烟、体重减轻）与临床筛查指南高度一致
- **全时间轴利用**: 事件分布显示模型在关注最近一年事件的同时，仍能从更早期历史中提取有价值信息

## 亮点与洞察
- **EHRMem设计精巧**: 包容性提取 + 去重机制 + 局部提取全局判断的分工，优雅解决了长序列中的信息遗忘问题
- **时间感知分块优于固定分块**: 按时间戳动态分段保持时间完整性，避免了硬分块打断临床事件的问题
- **零样本即可比肩监督学习**: 无需任何训练数据即可达到与XGBoost等传统方法可比的性能，体现了框架的通用性
- **chunk大小的trade-off洞察深刻**: 揭示了小chunk导致灾难性遗忘 vs 大chunk导致lost-in-the-middle的基本张力

## 局限与展望
- **单机构小规模评估**: 仅在一个医疗机构的300个测试样本上验证，泛化性待证明
- **单一预测任务**: 仅验证肺癌风险预测，框架的任务无关性需在更多临床场景中检验
- **依赖精心设计的提示**: 任务特定的prompt工程是必要的，限制了完全自动化应用
- **计算复杂度高于RAG**: 编码复杂度为O(L·L_C)，高于RAG的O(L_R²)，需在延迟和完整性间权衡
- **数据集类别不平衡**: 1:10的case-control比例可能影响评估的稳健性
- 可改进：引入外部知识增强、更强基础模型、多agent训练优化、prompt自动优化

## 相关工作与启发
- **vs Vanilla LLM**: 直接使用LLM处理长EHR面临上下文退化问题（64k时AUROC降至0.714），Traj-CoA通过分治策略打破了这一瓶颈
- **vs RAG**: RAG通过选择性检索降低延迟但存在信息丢失风险，Traj-CoA处理完整上下文，牺牲速度换取全面性
- **vs Chain-of-Agents原版**: 原版CoA缺乏长期记忆，直接应用于EHR导致早期关键事件遗忘，EHRMem是关键创新
- **vs EHR基础模型（BEHRT等）**: 受限于有限编码集和短上下文（<16k），无法充分利用非结构化文本和超长历史

## 评分
- 新颖性: ⭐⭐⭐⭐ 将多agent架构与EHR特有的长期记忆需求结合，EHRMem设计有创意
- 实验充分度: ⭐⭐⭐⭐ 4类基线对比+消融+敏感性+时序推理分析+临床相关性验证，分析全面
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，方法描述详细，分析深入有洞察
- 价值: ⭐⭐⭐⭐ 为纵向EHR的零样本时序推理提供了可行框架，对临床AI有实际应用潜力

<!-- RELATED:START -->

## 相关论文

- [TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)
- [Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](../../ACL2025/llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardnessaware_trajectory_synthesis_gui_agent.md)
- [Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)

<!-- RELATED:END -->
