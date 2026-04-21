---
title: >-
  [论文解读] CoMind: Towards Community-Driven Agents for Machine Learning Engineering
description: >-
  [ICLR 2026][LLM Agent][LLM Agent] 提出MLE-Live——首个模拟Kaggle研究社区的实时评估框架，以及CoMind——一个能够系统性利用社区集体知识的多智能体ML工程系统，在75个历史Kaggle竞赛中获得36%奖牌率，并在4个进行中的竞赛中平均超越79.2%的人类参赛者（更新版本中达到92.6%）。
tags:
  - ICLR 2026
  - LLM Agent
  - 机器学习工程
  - Kaggle竞赛
  - 社区知识
  - 多智能体协作
---

# CoMind: Towards Community-Driven Agents for Machine Learning Engineering

**会议**: ICLR 2026  
**arXiv**: [2506.20640](https://arxiv.org/abs/2506.20640)  
**代码**: [https://github.com/comind-ml/CoMind](https://github.com/comind-ml/CoMind)  
**领域**: LLM Agent  
**关键词**: LLM Agent, 机器学习工程, Kaggle竞赛, 社区知识, 多智能体协作

## 一句话总结
提出MLE-Live——首个模拟Kaggle研究社区的实时评估框架，以及CoMind——一个能够系统性利用社区集体知识的多智能体ML工程系统，在75个历史Kaggle竞赛中获得36%奖牌率，并在4个进行中的竞赛中平均超越79.2%的人类参赛者（更新版本中达到92.6%）。

## 研究背景与动机
基于LLM的ML Agent已展现出自动化ML工程的巨大潜力。MLAB采用ReAct风格的结构化决策，AIDE利用树搜索探索，AutoKaggle引入多Agent专业化分工。这些系统已在Kaggle风格竞赛上取得进展。

**核心矛盾**：现有Agent在**孤立环境**中运行——仅依赖内部记忆和试错探索，完全忽略了现实ML工作流中至关重要的成分：**社区知识共享**。在真实的数据科学竞赛和研究中，参与者频繁从公开讨论、共享笔记本和社区洞察中学习。当前Agent因无法利用这种动态外部上下文，往往收敛到重复策略并在性能上遇到瓶颈。

**两个关键问题**：
1. 如何评估Agent利用集体知识的能力？（→ MLE-Live基准）
2. 如何设计能有效利用社区知识的Agent？（→ CoMind系统）

## 方法详解

### MLE-Live评估框架
基于MLE-Bench扩展，为每个竞赛增加模拟社区环境：

- **社区资源**：收集2,687个讨论帖和4,270个公开内核（来自22个Kaggle竞赛的low-complexity split）
- **元数据质量信号**：投票数（社区偏好）、公开分数（性能指标）、作者等级（Novice到Grandmaster）
- **时间约束**：所有内容均在竞赛截止日期前发布，防止事后泄漏
- **过滤规则**：移除非文本内容（图片、截图）和Jupyter系统输出（进度条、冗余日志）
- **评估指标**：Valid Submission（格式正确率）、Above Median（超过中位数的比例）、Win Rate（击败人类参赛者的百分比）、Medals（金/银/铜牌）

### CoMind Agent工作流
CoMind维护两个核心仓库：
- **Idea Pool**（想法池）：从社区内容和历史迭代中提炼的抽象洞察
- **Report Pool**（报告池）：包含代码、评估和分析的完整解决方案报告

每轮迭代包含四个阶段：

1. **Stage I - 想法选择（Idea Selection）**：
    - 访问Idea Pool中从公开内核、论坛讨论和历史解决方案中提炼的概念和策略
    - 利用Report Pool作为性能和相关性评估的指导，对条目排序和筛选
    - 模拟人类参赛者在形成新假设前浏览集体智慧的过程

2. **Stage II - 想法生成（Idea Generation）**：
    - 基于选定的想法和Report Pool中的上下文，生成高层解决方案草案
    - 通过重组或扩展已有想法来综合新策略
    - **关键约束**：避免简单复制，确保概念多样性和探索广度
    - 模拟人类从过去工作中抽象和创新的能力

3. **Stage III - 实现与改进（Implementation and Improvement）**：
    - 基于生成的草案启动ReAct风格循环
    - 迭代地编写代码、执行、观察反馈（验证指标、错误日志）、更新实现
    - **刻意限制上下文**：仅可访问问题描述和特定草案，排除Idea Pool和Report Pool
    - 保证实验模块性，防止上下文窗口爆炸（最多20步）

4. **Stage IV - 报告生成（Report Generation）**：
    - 编译解决方案报告：方法描述、组件分析、定量结果、局限性评估
    - 报告发布回Report Pool，对后续迭代可见
    - 模拟真实用户记录和分享最终解决方案的过程

### 并行Agent与共享洞察
- 多个Agent在同一任务上并行运行，共享社区知识库
- 某Agent生成新报告后，其他Agent在后续迭代中可读取
- Agent之间通过共享报告互相启发，形成集体探索和改进

### 关键设计思想
- **探索广度 vs 实现深度的平衡**：维护多个不同的解决方案草案并行发展，每次动态聚焦一个草案进行深度实现
- **知识累积**：跨迭代的Idea Pool和Report Pool不断增长，形成日益丰富的知识基础
- **避免上下文爆炸**：Stage III刻意限制可访问信息，仅关注当前草案

## 实验关键数据

### 主实验（20个历史Kaggle竞赛，使用o4-mini）

| 方法 | Valid Sub. | Win Rate | Any Medal | Above Median | Medal详情 |
|------|-----------|----------|-----------|-------------|----------|
| CoMind | **1.00** | **66.8%** | **45%** | **65%** | 5金4银 |
| AIDE | 0.90 | 46.9% | 20% | 50% | - |
| AIDE+Code | 0.90 | 51.0% | 25% | 50% | - |
| AIDE+RAG | 0.95 | 51.2% | 25% | 55% | - |

CoMind获得9枚奖牌（5金），相对此前SOTA AIDE提升125%。

### 在线竞赛结果（4个进行中的Kaggle竞赛）

| 竞赛 | CoMind WR | AIDE WR | CoMind排名 |
|------|-----------|---------|-----------|
| playground-series-s5e5 | **94.9%** | 66.2% | #120/2338 |
| forams-classification-2025 | **91.7%** | 69.4% | #4/48 |
| el-hackathon-2025 | **61.6%** | 8.5% | #128/333 |
| fathomnet-2025 (CVPR FGVC12) | **69.4%** | 28.6% | #15/47 |

### 按任务类别Win Rate

| 类别 | CoMind | AIDE | AIDE+Code | AIDE+RAG |
|------|--------|------|-----------|----------|
| Image Classification (8) | **59.7%** | 45.9% | 43.4% | 52.5% |
| Text Classification (3) | **74.0%** | 15.7% | 33.8% | 61.0% |
| Audio Classification (1) | **90.1%** | 27.2% | 25.9% | 27.1% |
| Tabular (4) | 66.4% | 67.3% | **68.8%** | 48.3% |
| Image Regression (1) | 99.2% | 34.2% | **99.2%** | 99.2% |

### 消融实验

| 配置 | Valid Sub. | Win Rate | Any Medal |
|------|-----------|----------|-----------|
| CoMind w/ 公共资源 | **1.00** | **66.8%** | **45%** |
| CoMind w/o 公共资源 | 0.90 | 54.5% | 35% |

### 关键发现
- **社区知识至关重要**：不使用公共资源时Win Rate下降12.3%，Valid Submission下降10%，说明社区知识不仅提升质量，还为基线可靠性提供保障
- **持续改进能力**：AIDE在前2小时快速上升后趋于平台期，而CoMind持续改进最终超越
- **代码复杂度更高**：CoMind生成的代码平均比AIDE长55.4%，暗示更深度的推理和更丰富的优化技术
- **新颖性评估**：排除外部想法后，CoMind的平均新颖性排名1.20（vs AIDE的3.05），证明其不是简单复制社区方案
- CoMind在Seq2Seq任务上表现较弱，因为其倾向于探索大模型微调策略，常无法在1小时运行时限内完成

## 亮点与洞察
- **"社区意识"的创新概念**：首次将数据科学竞赛中的社区协作动态引入LLM Agent评估，填补了"孤立Agent"与"真实研究实践"之间的巨大gap
- **四阶段迭代循环设计**：想法选择→想法生成→实现→报告的流程非常贴近真实研究者的工作模式
- **刻意限制Stage III的上下文**：既防止信息过载导致的LLM性能下降，又确保每个解决方案草案的独立性——这一设计insight值得学习
- **活跃竞赛的真实验证**：在进行中的Kaggle竞赛上提交真实成绩，大幅增强了说服力
- **MLE-Live基准的价值**：为社区驱动的Agent研究提供了标准化评估平台

## 局限与展望
- 当前仅支持报告级别的交互，缺乏评论、提问、数据/模型共享等更精细的社区互动方式
- 在需要大模型微调的任务（如Seq2Seq）上受限于运行时间约束
- 仅在Kaggle风格ML竞赛上验证，未扩展到更广泛的科学发现、开放编程或机器人等领域
- Agent的"创新"可能仍受限于LLM backbone的知识范围
- 多Agent并行的通信和协调机制相对简单（仅通过Report Pool），未探索更丰富的消息传递协议
- 代码执行环境受限（单个A6000 GPU，5小时总限），可能低估了计算密集型方案的潜力

## 相关工作与启发
- **AIDE**（Jiang et al., 2025）：基于树搜索的ML Agent，MLE-Bench上此前最强方法
- **MLAB**（Huang et al., 2024）：ReAct风格的ML Agent基准
- **MLE-Bench**（Chan et al., 2025）：基于75个Kaggle竞赛的ML Agent评估基准
- **AutoKaggle**（Li et al., 2024）：多Agent系统的MLE方案
- **MetaGPT**（Hong et al., 2023）：通用多Agent协作框架
- 启发：Agent不应只依赖内部推理和试错——利用外部"集体智慧"是提升Agent能力的关键维度。这一思路可能推广到科学发现、软件工程等需要社区协作的其他领域

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （社区驱动Agent + MLE-Live基准 = 全新研究方向）
- 实验充分度: ⭐⭐⭐⭐⭐ （20个历史竞赛 + 4个活跃竞赛 + 新颖性评估 + 消融 + 代码复杂度分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但部分实验数据的呈现可以更紧凑）
- 价值: ⭐⭐⭐⭐⭐ （开辟社区感知Agent新方向，对Data Science自动化有重大意义）

<!-- RELATED:START -->

## 相关论文

- [MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](../../NeurIPS2025/llm_agent/mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)

<!-- RELATED:END -->
