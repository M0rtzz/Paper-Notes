---
title: >-
  [论文解读] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models
description: >-
  [ICLR 2026][时间序列][时间序列推理] TimeOmni-1 提出了首个统一的时间序列推理模型，通过 TSR-Suite（首个推理导向的时序数据集套件）和两阶段训练（SFT注入时序先验 + RL精炼推理），在多项时间序列推理任务上显著超越 GPT-4.1。
tags:
  - "ICLR 2026"
  - "时间序列"
  - "时间序列推理"
  - "LLM"
  - "强化学习"
  - "多任务联合训练"
  - "因果发现"
---

# TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2509.24803](https://arxiv.org/abs/2509.24803)  
**代码**: [GitHub](https://github.com/AntonGuan/TimeOmni-1)  
**领域**: 人体理解/时间序列  
**关键词**: 时间序列推理, LLM, 强化学习, 多任务联合训练, 因果发现

## 一句话总结

TimeOmni-1 提出了首个统一的时间序列推理模型，通过 TSR-Suite（首个推理导向的时序数据集套件）和两阶段训练（SFT注入时序先验 + RL精炼推理），在多项时间序列推理任务上显著超越 GPT-4.1。

## 研究背景与动机

时间序列理解正从基础模式分析向高级推理转变，但存在两大瓶颈：

**高质量数据匮乏**: 现有的时间序列QA数据集（如 Time-MQA）停留在表面问答层面，存在严重问题——(a) 问题过于简单，推理模型与非推理模型差距极小；(b) 上下文不充分，缺少关键信息导致模型被迫猜测而非推理

**缺乏可行的推理路径**: 尚不清楚哪些任务真正需要时间序列推理能力，现有方法局限于窄任务（如 TimeMaster 对6个数据集训练6个模型），无法跨任务迁移

作者提出两个核心**设计原则**：
- **原则1 — QA必须奖励推理**: 推理模型应显著优于非推理模型 $\bar{S}(M_{RM}) \gg \bar{S}(M_{NRM})$
- **原则2 — 上下文必须充分**: 提供足够的时间序列输入 $X$ 和辅助上下文 $C$，避免歧义

## 方法详解

### 整体框架

TimeOmni-1 采用**两阶段课程学习**：

- **Stage 1 (SFT)**: 注入时间序列推理先验——通过人类引导的推理链(CoT)进行监督微调，在 LLM 中建立感知、外推、决策三大时序推理能力
- **Stage 2 (RL)**: 精炼推理——使用 GRPO 和任务定制奖励函数，将模仿先验转化为稳健的推理行为

### 关键设计

1. **TSR-Suite（时间序列推理套件）**: 首个综合性时序推理数据集，涵盖 4 个原子任务和 3 大推理能力：

    - **感知(Perception)**: Task 1-场景理解（单序列归因）、Task 2-因果发现（多序列因果关系）
    - **外推(Extrapolation)**: Task 3-事件感知预测（在事件扰动下推断未来趋势）
    - **决策(Decision-making)**: Task 4-决策制定（整合感知与外推进行行动选择）

   包含23K+样本，其中2.3K通过人类引导的层级标注流程精心策划，跨越10个领域。

2. **层级CoT标注流程**: 三步标注——(a) LLM Analyzer 在人类引导模板下生成推理链（Step-1 CoT）；(b) 人类专家验证上下文充分性，对LLM错误案例撰写专家推理链（Step-2 CoT）；(c) LLM Rewriter 将专家推理链规范化。

3. **任务定制的RL奖励设计**: 

    - 格式奖励 $\mathcal{R}_{format}$：强制 `<think></think><answer></answer>` 格式
    - 离散任务（Task 1,2,4）：精确匹配准确率 $\mathcal{R}_{discrete} \in \{0,1\}$
    - 序列任务（Task 3）：计数奖励 $\mathcal{R}_{count}=0.1$（预测序列长度正确）+ 指数衰减映射的 MAE 归一化奖励

4. **多任务联合训练**: 将所有任务统一在单一模型中训练。两个互补实验验证了跨任务增益：

    - **渐进能力迁移**: 未直接训练决策任务，仅通过感知+外推训练，决策准确率从25.5%提升至31.3%
    - **渐进能力补充**: 逐步加入前置任务的联合训练使决策准确率从40.9%升至47.9%

### 损失函数 / 训练策略

- Stage 1: 标准的 SFT 损失（交叉熵），使用人类引导的 CoT 数据
- Stage 2: GRPO (Group Relative Policy Optimization)，奖励函数为：
    - Task 1/2/4: $R = \mathcal{R}_{format} + \mathcal{R}_{discrete}$
    - Task 3: $R = \mathcal{R}_{format} + \mathcal{R}_{count} + \text{exp-decay}(\text{MAE})$

## 实验关键数据

### 主实验

**四个任务的 ID/OOD 测试（ACC %，Task 3 为 MAE↓）:**

| 方法 | 场景理解(ID) | 因果发现(ID) | 事件预测(ID/MAE) | 决策(ID) |
|------|------------|------------|----------------|---------|
| GPT-4.1 | 85.5 | 28.7 | 13.79 | 25.5 |
| Qwen2.5-7B | 48.5 | 21.6 | 23.28 | 25.5 |
| Time-R1 | 30.9 | 30.2 | 17.61 | 27.8 |
| **TimeOmni-1** | **90.7** | **69.3** | **14.30** | **47.9** |

TimeOmni-1 在因果发现上超越 GPT-4.1 达 40.6%（ID），决策任务超越 22.4%。

### 消融实验

| 配置 | 因果发现(ID) | 决策(ID) | 说明 |
|------|------------|---------|------|
| Base model | 21.6 | 25.5 | LLM 缺乏时序先验 |
| ANS-SFT (答案监督) | 30.5 | 51.0 | 仅拟合答案分布，无推理 |
| CoT-SFT (Stage 1) | 67.7 | 40.9 | 推理链注入显著提升因果发现 |
| CoT-SFT+RL (Stage 2) | 69.3 | 47.9 | RL 精炼进一步提升 |
| 单任务训练(CoT-SFT+RL) | 67.5 | 40.9 | 联合训练优于单任务 |

### 关键发现

- **LLM 天然缺乏时序推理先验**: 基础模型因果发现仅21.6%（接近随机33.3%），单独RL无法建立此能力
- **人类引导模板至关重要**: GPT-4.1 零样本因果发现28.7%，使用人类引导模板后升至71.1%
- **联合训练产生互益**: 跨任务联合训练在所有任务上优于单任务训练，支持"训练一次、跨任务使用"范式
- **通用推理能力未退化**: TimeOmni-1 在 DROP、GPQA、ReClor 等通用推理基准上平均准确率比基础模型提升16.5%
- **SR(有效响应率)**: TimeOmni-1 在所有任务上 SR≥93.8%，远优于现有时序专用模型（如 ChatTS 在事件预测上 SR=0%）

## 亮点与洞察

1. **对时序推理任务的系统化定义**: 首次明确提出"推理必要性"和"上下文充分性"两个设计原则，构建了真正需要推理的任务体系
2. **感知→外推→决策的渐进能力路径**: 体现了"先理解再预测再行动"的认知逻辑，任务设计有深度
3. **两阶段训练的互补关系**: SFT 负责"知道该怎么想"，RL 负责"想得更准"——缺一不可
4. **跨任务正向迁移的实证**: 通过精心设计的渐进实验，证明了时序推理的三大能力存在内在联系

## 局限与展望

1. **数据规模有限**: TSR-Suite 仅23K样本（其中人工标注2.3K），与通用NLP数据集相比较小
2. **任务类型集中**: 4个任务以分类和预测为主，缺少如异常检测、趋势解释等更多样的推理任务
3. **OOD泛化仍有差距**: 事件预测任务 OOD MAE 为145.53（ID为14.30），跨域泛化仍需提升
4. **基础模型限制**: 仅基于 Qwen2.5-7B 验证，未探索更大模型的 scaling 行为
5. **推理链质量依赖人类模板**: 数据构建高度依赖人类引导模板，扩展性受限

## 相关工作与启发

- **Time-R1** 是最接近的时序推理模型，但局限于经典预测，TimeOmni-1 扩展至多任务推理
- **DeepSeek-R1** 证明了 RL 可以提升推理能力，TimeOmni-1 将这一范式引入时序领域
- **Time-MQA** 数据集虽大但任务过简单且上下文不足，TSR-Suite 针对性改进
- 对**时间序列智能**领域有重要启发：通用时序模型需要注入推理先验而非仅仅模式拟合

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次系统构建时序推理任务体系 + 首个统一时序推理模型
- **实验充分度**: ⭐⭐⭐⭐⭐ 四任务ID/OOD评测 + 渐进实验 + 消融 + 通用能力评估，非常全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，设计原则驱动，但部分图表信息密度过高
- **价值**: ⭐⭐⭐⭐⭐ 开辟了时间序列推理的新方向，数据+模型+代码全部开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](../../NeurIPS2025/time_series/planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[ICML 2026\] TimeOmni-VL: Unified Models for Time Series Understanding and Generation](../../ICML2026/time_series/timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] Unlocking the Value of Text: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting](unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)

</div>

<!-- RELATED:END -->
