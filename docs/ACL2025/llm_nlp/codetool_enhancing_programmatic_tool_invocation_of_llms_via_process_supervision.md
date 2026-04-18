---
title: >-
  [论文解读] CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision
description: >-
  [ACL 2025][LLM/NLP][工具调用] 提出 CodeTool，一种逐步代码生成框架，通过即时奖励（On-the-spot Reward）和潜在奖励（Latent Reward）两种过程奖励机制引导 LLM 选择最优的工具调用路径，在 StableToolBench 和 RestBench-TMDB 上显著超越现有方法。
tags:
  - ACL 2025
  - LLM/NLP
  - 工具调用
  - 代码生成
  - 过程监督
  - 过程奖励模型
  - 逐步推理
---

# CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision

**会议**: ACL 2025  
**arXiv**: [2503.20840](https://arxiv.org/abs/2503.20840)  
**代码**: [LimOkii/CodeTool](https://github.com/LimOkii/CodeTool)  
**领域**: LLM/NLP  
**关键词**: 工具调用, 代码生成, 过程监督, 过程奖励模型, 逐步推理

## 一句话总结

提出 CodeTool，一种逐步代码生成框架，通过即时奖励（On-the-spot Reward）和潜在奖励（Latent Reward）两种过程奖励机制引导 LLM 选择最优的工具调用路径，在 StableToolBench 和 RestBench-TMDB 上显著超越现有方法。

## 研究背景与动机

工具调用（Tool Invocation）让 LLM 能访问外部 API 和服务，极大扩展了其能力边界。现有方法存在三个核心问题：

**JSON/文本格式的局限**：当前工具调用主要使用 JSON 或文本格式，在处理请求密集型任务时 token 消耗大、推理路径长，且容易被截断导致关键信息丢失

**缺乏中间步骤监督**：现有代码方式（如一次性生成完整代码）无法检测和纠正复杂场景中的中间步骤错误

**过程奖励可靠性差**：已有的过程奖励方法（如 StepTool）依赖强 LLM 生成奖励信号，客观性和可靠性存疑

**代码方式的优势**：代码可以利用循环（for/while）和数组等编程结构高效处理批量请求，减少交互轮次。同时代码天然可执行验证，为过程监督提供了客观可靠的信号源。

## 方法详解

### 整体框架

CodeTool 是一个逐步代码生成框架，核心流程：
1. 给定用户查询 $q$ 和工具集 $\mathcal{T}$ 及其文档协议 $\mathcal{D}$
2. 在每一步，LLM 生成一段 Python 代码 $\mathcal{C}_t$ 来调用适当的工具
3. 通过代码解释器执行 $\mathcal{C}_t$ 获取中间结果 $r_t$
4. 在每步采样多个候选动作，基于过程奖励选择最优动作进行下一步
5. 迭代直到获得最终答案

形式化表示：$\mathcal{C}_t = \mathcal{M}(q, \mathcal{T}, \mathcal{D}, I_c, r_{t-1})$，其中 $r_0 = \emptyset$

### 关键设计

**双重过程奖励机制**：

**1. 即时奖励（On-the-spot Reward）**：
- 评估当前步骤生成的代码是否可以正确执行
- 通过 Python 解释器自动验证，无需外部监督

$$R_{spot,t} = \begin{cases} 1, & \text{if Execute}(\mathcal{C}_t) \text{ is successful} \\ 0, & \text{otherwise} \end{cases}$$

- 优势：基于代码可执行性，信号客观可靠，不依赖 LLM 标注

**2. 潜在奖励（Latent Reward）**：
- 评估当前步骤对最终任务完成的潜在贡献
- 使用蒙特卡洛树搜索（MCTS）估算：从当前步骤扩展多条路径，统计成功率
- 引入惩罚机制抑制冗余调用和过长路径：

$$R_{latent,t}(q, s_{1:t}) = \alpha^{1-LR(q, s_{1:t})} \cdot \beta^{\frac{\tau}{L}}$$

其中 $\alpha, \beta \in (0,1]$，$\tau$ 为平均步骤数，$L$ 为常数超参数

**累积奖励**：$R_{total,t} = R_{spot,t} + R_{latent,t}$

在每步选择累积奖励最高的候选动作。

**过程潜在奖励模型（PRM）训练**：
- 推理时用 MCTS 估算 Latent Reward 成本太高
- 训练一个 PRM（基于 Qwen2.5-7B-Instruct）来直接预测 Latent Reward
- 训练数据的构建：对 ToolBench 训练集中仍可调用的查询，用深度优先搜索构建二叉动作树
- 采用生成式 PRM 训练方法，用两个特殊 token 表示"更有潜力"和"更无潜力"
- 全自动数据构建过程，无需人工标注

### 损失函数 / 训练策略

- PRM 训练使用标准 SFT 损失，学习率 1e-6，训练 2 个 epoch
- 代码生成模型（如 Qwen2.5-Coder-7B-Instruct）不需要额外微调
- 训练和推理分离：仅需训练 PRM，代码生成能力来自预训练的代码模型

## 实验关键数据

### 主实验

**StableToolBench 结果（SoPR%）**：

| 模型 | 策略 | 格式 | 平均 SoPR |
|------|------|------|-----------|
| ToolLLaMA-v2 | CoT | JSON | 33.39 |
| ToolLLaMA-v2 | DFSDT | JSON | 53.24 |
| StepTool | DFSDT | JSON | 44.02 |
| Qwen2.5-7B-Instruct | DFSDT | JSON | 60.52 |
| **Qwen2.5-7B + CodeTool** | CodeTool | Code | **64.19** |
| **Qwen2.5-Coder-7B + CodeTool** | CodeTool | Code | **69.75** |
| GPT-4-Turbo | DFSDT | JSON | 62.03 |
| **GPT-4-Turbo + CodeTool** | CodeTool | Code | **71.05** |

CodeTool 在开源和闭源 LLM 上均显著超越现有方法。

**RestBench-TMDB 泛化性**（无需重训练 PRM）：

| 方法 | Success Rate | Path Rate |
|------|-------------|-----------|
| ATC | 89% | 84.71% |
| **CodeTool** | **92%** | **91.15%** |

### 消融实验

**双重奖励的必要性（Qwen2.5-Coder-7B）**：

| 配置 | 平均 SoPR | 平均 SCEP |
|------|-----------|-----------|
| 完整 CodeTool | **69.75** | **86.86%** |
| - 去掉 On-the-spot Reward | 65.99 (-3.76) | 69.46% (-17.4%) |
| - 去掉 Latent Reward | 65.41 (-4.34) | 85.34% (-1.52%) |

- 去掉 On-the-spot Reward 对代码执行成功率（SCEP）影响最大（-17.4%），说明即时反馈对代码正确性至关重要
- 去掉 Latent Reward 对 SoPR 的影响更显著，说明长期方向指引对任务完成度重要

### 关键发现

1. 模型代码能力越强，CodeTool 效果越好（Qwen2.5-Coder > Qwen2.5-Instruct > CodeLlama）
2. PRM 在未见过的 RestBench-TMDB 上仍然有效，显示出良好的泛化性
3. 代码格式相比 JSON 格式的优势在请求密集型场景中更加突出（I3 子集中 Qwen2.5-Coder+CodeTool 达到 79.41%）
4. On-the-spot Reward 的客观性是 CodeTool 的核心优势——基于代码执行而非 LLM 标注

## 亮点与洞察

1. **代码作为工具调用的"语言"**：用代码替代 JSON 不仅减少 token 消耗，更提供了天然的可执行验证能力，这是整个框架的基石
2. **双重奖励的互补设计**：On-the-spot 确保每步正确，Latent 确保方向正确，两者缺一不可
3. **全自动过程数据构建**：不依赖GPT标注，用深度优先搜索和代码执行结果自动构建训练数据
4. **代码生成模型免训练**：只需训练一个轻量级 PRM，代码生成模型直接使用预训练模型

## 局限与展望

- 代码生成能力强烈依赖底层代码模型的质量（CodeLlama-7B效果较差说明了这点）
- StableToolBench 存在 API 缓存缺失问题，导致需要过滤测试集，可能影响公平性
- PRM 的训练数据仅来源于 ToolBench，是否在完全不同类型的工具场景中仍然有效需验证
- 逐步代码生成 + 多候选采样增加了推理时间和计算成本
- 评估主要依赖 GPT-4 作为判定者，引入了评估偏差

## 相关工作与启发

- **ToolLLaMA**（Qin et al., 2023）和 **StepTool**（Yu et al., 2024）是工具调用的主要基线
- **ATC**（Shi et al., 2024）采用一次性完整代码生成方式
- **过程奖励模型**（Lightman et al., 2023）在数学推理中的成功启发了本工作
- 启发思考：代码执行提供的 On-the-spot Reward 本质上是一种"grounded verification"，类似方法可推广到其他可验证领域（如数学证明、单元测试）

## 评分

- **创新性**：⭐⭐⭐⭐ — 双重过程奖励设计巧妙，On-the-spot Reward 利用代码可执行性是亮点
- **实验完整性**：⭐⭐⭐⭐ — 多基线对比充分，消融实验清晰，但数据集有限
- **实用价值**：⭐⭐⭐⭐ — 代码生成模型免训练降低了使用门槛
- **写作质量**：⭐⭐⭐⭐ — 框架图清晰，但公式符号较多
