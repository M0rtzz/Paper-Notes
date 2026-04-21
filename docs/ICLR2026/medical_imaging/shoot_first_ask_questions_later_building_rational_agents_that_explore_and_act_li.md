---
title: >-
  [论文解读] Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
description: >-
  [ICLR 2026][医学图像][信息搜索智能体] 提出 Collaborative Battleship 任务评估语言模型的信息搜索能力，设计三种贝叶斯推断策略（Bayes-Q/M/D）增强 LM 的提问、行动和决策能力，使弱模型（Llama-4-Scout）以 GPT-5 约 1% 的成本达到超人表现（82% 胜率）。
tags:
  - ICLR 2026
  - 医学图像
  - 信息搜索智能体
  - 贝叶斯实验设计
  - 语言模型推理
  - 探索-利用权衡
  - 蒙特卡洛推断
---

# Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People

**会议**: ICLR 2026  
**arXiv**: [2510.20886](https://arxiv.org/abs/2510.20886)  
**代码**: [项目页面](https://gabegrand.github.io/battleship)  
**领域**: 医学图像（分类为此，实际为 AI Agent / 认知科学）  
**关键词**: 信息搜索智能体, 贝叶斯实验设计, 语言模型推理, 探索-利用权衡, 蒙特卡洛推断

## 一句话总结
提出 Collaborative Battleship 任务评估语言模型的信息搜索能力，设计三种贝叶斯推断策略（Bayes-Q/M/D）增强 LM 的提问、行动和决策能力，使弱模型（Llama-4-Scout）以 GPT-5 约 1% 的成本达到超人表现（82% 胜率）。

## 研究背景与动机
- 许多 AI 应用（科学发现、医疗诊断）需要智能体战略性地获取信息：形成假设、提出针对性问题、在不确定性下做决策
- 当前语言模型主要被优化来回答用户问题，但能否为自己提出好问题？
- 需要评估和提升前沿模型在动态环境中提出目标导向问题和采取行动的能力
- **核心动机**：借鉴人类认知的资源理性（resource rationality）理论，用贝叶斯实验设计增强 LM 的信息搜索能力

## 方法详解

### 整体框架
1. 设计 Collaborative Battleship 二人协作任务：Captain（部分信息）平衡探索（提问）和利用（射击），Spotter（完全信息）提供 yes/no 回答
2. 收集 126 局人类行为数据（N=42），建立 BattleshipQA 基准
3. 基于贝叶斯实验设计框架，提出三种增强策略
4. 在 Guess Who? 任务上验证泛化性

### 关键设计

1. **贝叶斯信念更新框架**:

    - 隐藏棋盘 $S \in \mathcal{S}$，信念 $\pi_t(s) = \Pr(S=s | x, \mathcal{H}_{1:t})$
    - 噪声观测模型：Spotter 建模为二元对称信道 $\text{BSC}(\varepsilon)$，$\varepsilon=0.1$
    - 贝叶斯更新：$\pi_{t+1}(s) \propto \pi_t(s)[(1-\varepsilon)\mathbf{1}\{\tilde{a}_t = f_{q_t}(s)\} + \varepsilon\mathbf{1}\{\tilde{a}_t \neq f_{q_t}(s)\}]$
    - 使用序贯蒙特卡洛（SMC）近似实现
    - **动机**：精确求和不可行，粒子近似既高效又能处理 Spotter 噪声

2. **三种贝叶斯策略**:

    - **Bayes-Q（提问）**：从 LM 采样候选问题集 $\mathcal{Q}$，选 EIG 最高的：$q_t^* = \arg\max_{q \in \mathcal{Q}} \text{EIG}_\varepsilon(q | x, \mathcal{H}_{1:t})$
    - EIG 闭式解：$\text{EIG}_\varepsilon = H_b(\varepsilon + (1-2\varepsilon)p_t) - H_b(\varepsilon)$，最大化当 $p_t \approx 1/2$ 时
    - **Bayes-M（行动）**：选击中概率最大的格子：$u_t^* = \arg\max_u p_t^{\text{hit}}(u | x, \mathcal{H}_{1:t})$
    - **Bayes-D（决策）**：一步前瞻决策——若 $\gamma \cdot \widehat{p_{t+1}^{\text{hit}}}(q_t^*) > p_t^{\text{hit}}(u_t^*)$ 则提问，否则射击
    - **动机**：将 LM 的自然语言能力与贝叶斯最优推断结合，各取所长

3. **代码生成增强回答 (SpotterQA)**:

    - 将自然语言问题翻译为 Python 程序，在假设空间上执行
    - 比直接回答和 CoT 基线提高 14.7% 准确率
    - **动机**：代码生成提供精确的形式化接地（grounding）

### 损失函数 / 训练策略
- 无训练过程——所有策略都是推断时方法
- SMC 粒子近似用于贝叶斯信念维护
- $\gamma = 0.95$ 轻微偏向当前行动
- 采样最多 10 个候选问题做 EIG 排序

## 实验关键数据

### 主实验（CaptainQA - 完整游戏）

| Captain 策略 | Llama-4-Scout F1 | GPT-4o F1 | GPT-5 F1 | 说明 |
|-------------|-----------------|-----------|----------|------|
| LM only | 0.367 | 0.450 | 0.716 | 纯 LM 基线 |
| +Bayes-Q | 0.388 | 0.476 | 0.717 | 仅提问增强 |
| +Bayes-M | 0.621 | 0.663 | 0.731 | 仅行动增强 |
| +Bayes-QM | 0.733 | 0.753 | 0.734 | 提问+行动 |
| +Bayes-QMD | 0.764 | 0.782 | — | 全增强（超人） |
| 人类平均 | — | — | — | F1 ≈ 0.6-0.7 |

### SpotterQA 回答准确率

| 模型 | Base | CoT+Code | 提升 |
|------|------|----------|------|
| GPT-4.1 | 75.2% | 90.9% | +15.7% |
| Claude 4 Opus | 86.8% | 94.4% | +7.6% |
| Llama-4-Scout | 62.2% | — | — |
| 人类 | 92.5% | — | — |

### 关键发现
- **弱模型通过贝叶斯增强达到超人水平**：Llama-4-Scout 的胜率从 8% 跳到 82%（vs 人类），从 0% 到 67%（vs GPT-5），成本仅为 GPT-5 的约 1%
- **高 EIG 提问不够**：Bayes-Q 单独提高 EIG 但游戏表现仅略有提升，Bayes-M 的行动增强才是关键
- **冗余问题消除**：Bayes-Q 将 Llama-4-Scout 的零信息增益问题从 18.5% 降至 0.2%
- **GPT-5 已有高效内部策略**：贝叶斯增强对 GPT-5 几乎无效，暗示其内部已实现类似推理
- **技巧型玩家先问后打但不全问**：人类和 GPT-5 平均只问 8 个问题（上限 15），但每个问题信息量更大

## 亮点与洞察
- 巧妙的实验设计：用经典游戏（海战棋）作为贝叶斯实验设计的可控测试平台
- "资源理性"视角独特——不追求全局最优，而是在有限资源下最大化效用
- 推断时缩放（inference-time scaling）的精彩示例：无需训练，仅靠采样+排序就大幅提升性能
- 代码生成作为接地手段，将模糊的自然语言问题转化为可执行程序

## 局限与展望
- 环境相对简单（8×8 棋盘），泛化到复杂真实世界场景需进一步验证
- Spotter 噪声 $\varepsilon$ 固定为 0.1，实际中应自适应估计
- 贝叶斯策略依赖可高效采样的"世界模型"，对于无法手工实现的领域需要学习生成模型
- 没有建模语用学（pragmatics）——人类对话中的上下文依赖问题仍是挑战
- 在 Guess Who? 上的泛化验证虽好，但任务复杂度仍有限

## 相关工作与启发
- Battleship 任务源自认知科学（Gureckis 2009, Rothe 2017-2019），本文首次扩展为多轮对话+完整游戏
- 贝叶斯实验设计（BED）理论（Lindley 1956, Chaloner 1995）的现代 LM 实例化
- 资源理性理论（Anderson 1990, Lieder 2020）指导了策略设计——追求"足够好"而非最优

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 认知科学+BED+LM 的独特交叉研究
- 实验充分度: ⭐⭐⭐⭐⭐ 人类实验+15个LM+消融+Guess Who泛化
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，理论与实验完美结合
- 价值: ⭐⭐⭐⭐⭐ 对构建理性信息搜索智能体有重要启发

<!-- RELATED:START -->

## 相关论文

- [Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](../../CVPR2026/medical_imaging/act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)
- [From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)
- [X-WIN: Building Chest Radiograph World Model via Predictive Sensing](../../CVPR2026/medical_imaging/x-win_building_chest_radiograph_world_model_via_predictive_sensing.md)
- [Out-of-Distribution Detection Methods Answer the Wrong Questions](../../ICML2025/medical_imaging/out-of-distribution_detection_methods_answer_the_wrong_questions.md)

<!-- RELATED:END -->
