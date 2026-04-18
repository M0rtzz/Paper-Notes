---
title: >-
  [论文解读] Trinity: An Evolved LLM Coordinator
description: >-
  [ICLR 2026][LLM协调] Trinity设计了一个轻量级coordinator（0.6B SLM + ~10K可训练参数的head），通过sep-CMA-ES优化，在多轮对话中将查询分配给不同LLM并指定Thinker/Worker/Verifier三种角色，在LiveCodeBench上达到86.2% pass@1的SOTA，在4个分布内和4个分布外任务上一致超越所有单模型和多agent基线。
tags:
  - ICLR 2026
  - LLM协调
  - 模型组合
  - 进化策略
  - CMA-ES
  - 多角色协作
  - test-time composition
---

# Trinity: An Evolved LLM Coordinator

**会议**: ICLR 2026  
**arXiv**: [2512.04695](https://arxiv.org/abs/2512.04695)  
**代码**: 无（Sakana AI）  
**领域**: 强化学习 / LLM协作  
**关键词**: LLM协调, 模型组合, 进化策略, CMA-ES, 多角色协作, test-time composition

## 一句话总结
Trinity设计了一个轻量级coordinator（0.6B SLM + ~10K可训练参数的head），通过sep-CMA-ES优化，在多轮对话中将查询分配给不同LLM并指定Thinker/Worker/Verifier三种角色，在LiveCodeBench上达到86.2% pass@1的SOTA，在4个分布内和4个分布外任务上一致超越所有单模型和多agent基线。

## 研究背景与动机

**领域现状**：LLM scaling law虽有效但代价高昂、收益递减。模型合并（model merging）受限于架构不兼容和闭源API。宏观层面的test-time模型组合（coordination）是一个有前景的替代方向。

**现有痛点**：(1) 现有routing/coordination方法（MasRouter、RouterDC、Smoothie等）无法有效利用多样化模型的互补优势，某些方法甚至降低性能到不如随机选择；(2) 缺乏对输入查询的丰富上下文理解来做出有效的delegation决策。

**核心矛盾**：Coordinator需要足够的语义理解力来正确分配任务，但又不需要（也不应该）像底层agent那样强大。如何用最少的参数学到最有效的coordination策略？

**本文目标**：(1) 如何从小模型的内部表示中提取足够的语义信号用于coordination？(2) 如何在极端参数预算（~10K）下优化coordination策略？(3) 如何设计有效的多轮协作模式？

**切入角度**：利用SLM隐藏状态（而非生成文本）作为上下文表示，用极轻量级head做routing决策，通过进化策略而非RL进行优化。

**核心 idea**：小模型的hidden states包含足够的语义信号，一个<20K参数的head就能协调多个顶级LLM超越任何单一模型。

## 方法详解

### 整体框架
Coordinator由Qwen3-0.6B SLM + linear head（~10K参数）组成。每轮将完整对话transcript输入coordinator，head从hidden state输出两组logits：一组选择LLM，一组分配角色（T/W/V）。消息处理模块注入角色特定prompt后发送给选中的LLM。

### 关键设计

1. **高效参数化**:

    - Head：单层线性映射，从hidden state $h \in \mathbb{R}^d$ 到 $\mathbb{R}^{L+3}$ 的logits（$L$个LLM + 3个角色）
    - SVD微调：对SLM选定权重矩阵做SVD分解，只学习奇异值缩放（固定正交矩阵）
    - 总参数量 < 20K，比典型微调小数个数量级
    - 关键洞察：coordinator的**生成文本被丢弃**，只使用hidden state的logit输出——可以使用早期token的hidden state做快速决策

2. **三角色协调（Tri-role Coordination）**:

    - **Thinker**: 策略规划——分析状态、返回高层指导（计划、分解、批判）
    - **Worker**: 具体执行——产出代码、推导、数值结果等可操作内容
    - **Verifier**: 质量评估——判断ACCEPT/REVISE + 可选诊断信息
    - 终止条件：Verifier被选中且输出ACCEPT，或达到固定轮次上限K
    - 设计动机：将复杂能力获取offload给底层LLM，coordinator只需做轻量级的分配决策

3. **sep-CMA-ES优化**:

    - 问题特征：高维（~10K参数）、弱参数耦合、高per-step代价（每步需运行coordinated agents推理）、二值终端奖励
    - 为何不用RL：REINFORCE的per-parameter gradients在此设置下SNR极低——弱inter-block耦合导致梯度病态、credit assignment差
    - 为何用sep-CMA-ES：维护对角协方差矩阵，特别适合block-diagonal景观；在高维+严格预算限制下理论上优于RL和random search
    - 理论保证：Proposition 1证明在小T regime下sep-CMA-ES的改进随迭代线性增长，而RS仅对数增长

### 目标函数
$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$，其中 $R(\tau) \in \{0,1\}$ 是终端奖励（答案正确/错误）。

## 实验关键数据

### 分布内评测（4个benchmark）

| 方法 | MATH500 | MMLU | RLPR | LiveCodeBench v6 |
|------|---------|------|------|------------------|
| GPT-5 (4K) | 0.91 | 0.92 | 0.34 | 0.56 |
| Gemini-2.5-pro (4K) | 0.92 | 0.91 | 0.41 | 0.47 |
| Claude-4-Sonnet (4K) | 0.90 | 0.89 | 0.37 | 0.51 |
| MoA | 0.83 | - | 0.38 | 0.39 |
| **Trinity** | **0.95** | **0.94** | **0.44** | **0.61** |

Trinity在所有4个任务上一致领先。MATH500相对error reduction 11.76%（vs Gemini-2.5-pro 5x CTX）。
LiveCodeBench SOTA: **86.2% pass@1**（V1 train → V6 test）。

### 零样本迁移（4个未见任务）

| 模型 | AIME | BigCodeBench | MT-Bench | GPQA-D | Average |
|------|------|-------------|----------|--------|---------|
| Gemini Pro 2.5 | 46.67 | 35.10 | 9.37 | 75.25 | 52.34 |
| GPT-5 | 46.67 | 33.80 | 9.35 | 72.73 | 51.07 |
| **Trinity** | **50.00** | **35.80** | **9.60** | **76.82** | **54.21** |

在所有4个未见任务上超越每个单模型，证明泛化能力。

### 关键发现
- 平均相对error reduction 21.9%（vs second-best方法）
- 某些baseline方法降低性能低于随机（如RouterDC在RLPR上0.28 < random 0.32），凸显effective coordination的难度
- Trinity在3/4任务上接近"Per-Question-Best"上限
- 涌现的task-aware策略：不同任务类型展现不同的T/W/V选择模式

## 消融实验
- Head架构：block-diagonal-10（极少参数）仍保留大部分性能 → 证实block-$\varepsilon$-separability
- SVD微调 vs 不微调：微调提供额外的表征改善
- sep-CMA-ES vs REINFORCE vs random search vs imitation learning：CMA-ES在此regime下大幅领先

## 亮点与洞察
- **极端参数效率**：<20K可训练参数协调7个顶级LLM（含GPT-5、Claude-4-Sonnet），这一参数量级令人惊叹
- **Hidden states的语义密度**：证明即使0.6B SLM的internal representation也足以为coordination提供丰富的上下文信号
- **进化策略 vs RL的niche**：在高维、弱耦合、稀疏奖励、高per-step成本的特定regime下，CMA-ES理论和实证上优于policy gradient——打破了"RL万能"的思维定式
- **三角色设计的优雅性**：T/W/V分工将coordinator从complex skill acquisition中解放出来，只需做assignment

## 局限与展望
- 依赖闭源API的LLM pool，成本和延迟是实际部署瓶颈
- Coordinator的SLM仍需推理每轮的完整transcript，对很长对话可能有效率问题
- 三角色的prompt设计是hand-crafted的，role自动化发现值得探索
- 训练集规模较小（400 LiveCodeBench samples），更大规模训练效果有待验证

## 相关工作与启发
- **vs MoA/LLM-Blender**: 简单的mixture/融合方法不够——有效coordination需要query-level的上下文理解
- **vs RouterDC/MasRouter**: 现有routing方法缺乏multi-turn推理和role assignment的能力
- **vs Model merging**: Trinity完全不修改底层模型权重，兼容闭源和异构模型
- **vs Self-reflection**: 单模型self-reflection（5x SR）仍不如Trinity，因为它无法进行inter-model互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SLM hidden state + 超轻量head + CMA-ES的组合极为创新
- 实验充分度: ⭐⭐⭐⭐⭐ 8个benchmark（4 in-dist + 4 zero-shot），全面的消融和理论分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精确，理论分析扎实，实验展示清晰
- 价值: ⭐⭐⭐⭐⭐ LiveCodeBench SOTA，开创了超轻量coordination的新范式
