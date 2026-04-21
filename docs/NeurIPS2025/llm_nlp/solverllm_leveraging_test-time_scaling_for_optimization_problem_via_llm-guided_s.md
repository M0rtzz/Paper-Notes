---
title: >-
  [论文解读] SolverLLM: 通过LLM引导的搜索利用测试时缩放求解优化问题
description: >-
  [NeurIPS 2025][LLM/NLP][测试时缩放] 提出SolverLLM，一个无需训练的框架，将优化问题的数学建模视为搜索问题，通过改进的MCTS在六元素表述空间中探索最优formulation，引入动态扩展、提示反向传播和不确定性反向传播，在6个基准上以无训练方式超越prompt方法和微调方法。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 测试时缩放
  - 蒙特卡洛树搜索
  - 优化问题
  - 动态扩展
  - 不确定性传播
---

# SolverLLM: 通过LLM引导的搜索利用测试时缩放求解优化问题

**会议**: NeurIPS 2025  
**arXiv**: [2510.16916](https://arxiv.org/abs/2510.16916)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 测试时缩放, 蒙特卡洛树搜索, 优化问题, 动态扩展, 不确定性传播

## 一句话总结

提出SolverLLM，一个无需训练的框架，将优化问题的数学建模视为搜索问题，通过改进的MCTS在六元素表述空间中探索最优formulation，引入动态扩展、提示反向传播和不确定性反向传播，在6个基准上以无训练方式超越prompt方法和微调方法。

## 研究背景与动机

**领域现状**：优化问题求解包含三个阶段——问题表述（自然语言→数学建模）、代码生成（数学模型→可执行代码）、程序执行（调用Gurobi/Pyomo等求解器）。其中问题表述既需要领域知识又需要数学编程经验，是自动化的核心瓶颈。LLM为自动化这一过程提供了可能。

**现有痛点**：当前方法分两条路线。Prompt方法（Chain-of-Experts、OptiMUS）通过多智能体协作分工，但对prompt设计极度敏感，面对陌生问题类型泛化能力差。学习方法（ORLM、LLMOPT）通过监督微调获得领域能力，但高度依赖大规模标注数据，跨域泛化困难且训练成本高。

**核心矛盾**：通用性、高性能和低训练成本三者之间存在根本张力——prompt方法免训练但性能受限，微调方法性能好但缺乏通用性。如何在不训练的前提下同时获得高性能和跨域泛化能力？

**本文目标** 用测试时计算代替训练时计算，构建无需额外训练、在推理阶段通过结构化搜索来提升优化问题建模质量的通用框架。

**切入角度**：Test-time scaling在数学推理等任务上已证明通过增加推理计算可提升性能。优化问题的建模过程天然适合结构化搜索——可以将建模拆解为多个语义元素（类型、集合、参数、变量、目标、约束），逐层搜索最优表述。

**核心 idea**：把优化问题的数学建模视为搜索问题，用改进的MCTS在六元素表述空间中探索，通过推理信号反馈和不确定性估计来指导搜索方向。

## 方法详解

### 整体框架

SolverLLM将问题建模过程组织为MCTS树，每个节点对应一个部分formulation（六个元素之一），从根到叶的路径定义完整的数学模型。搜索过程遵循MCTS的四个阶段——选择、扩展、模拟、反向传播——但每个阶段都针对LLM引导的符号推理进行了改造。输入为自然语言优化问题描述，输出为数学formulation和solver-ready代码。

### 关键设计

1. **六元素建模分解（含Type元素）**:

    - 功能：为搜索提供结构化指引，将非结构化的建模任务转化为层次化的决策序列
    - 核心思路：在先前五元素（Sets, Parameters, Variables, Objective, Constraints）基础上新增Type元素，用于识别问题的高层类别（LP、IP、MIP等）。Type元素提供全局指导 $G_g$，帮助LLM在开始详细建模前先建立正确的问题心智模型。每个节点编码一个部分formulation，整条路径定义完整模型
    - 设计动机：类比"学生考前先复习关键概念"——早期明确问题类型帮助避免后续建模中的基础性错误，如将整数变量误建模为浮点型。消融实验显示Type元素对复杂问题（如图相关优化）的提升尤为显著

2. **动态扩展 + 提示反向传播（Prompt Backpropagation）**:

    - 功能：允许搜索树双向修正formulation，突破标准MCTS只在叶节点扩展的限制
    - 核心思路：分两个创新。(a) 非叶节点扩展：修改选择策略使得搜索可在非叶节点（triggered by $t_s$）停下并扩展，允许基于后续层反馈修正前面层的决策——如约束层的反馈可触发变量层的重新建模。(b) 提示反向传播：评估阶段产生每层的推理信号三元组 $\mathcal{S}_l = (t_{s_l}, E_{s_l}, G_l)$，其中 $t_s$ 是激活触发器，$E_{s_l}$ 是评估理由，$G_l$ 是修正指导。这些信号经不确定性过滤（局部不确定性 $U^{local}_{s_l}$ 需超过阈值 $\eta$）后传播回各层的知识库 $\mathcal{G}_l$，指导后续扩展。局部不确定性用预测熵估计
    - 设计动机：优化问题的建模不是严格从上到下的线性过程——约束可能依赖于变量定义的修改。动态扩展使SolverLLM能在搜索过程中"从错误中学习"

3. **不确定性反向传播（Uncertainty Backpropagation）**:

    - 功能：提升奖励传播的稳健性，减少LLM评估的噪声影响
    - 核心思路：使用LLM作为语义evaluator打分时，通过多次采样 objective_score 估计语义不确定性 $U^{global}_s$。反向传播时按不确定性下权：$Q_{s'} \leftarrow Q_{s'} + \rho_s \cdot (\bar{R} - Q_{s'}) / N_{s'}$，其中权重因子 $\rho_s = \exp(-U^{global}_s)$。高confidence评估强传播，noisy评估弱传播
    - 设计动机：LLM作为evaluator不可避免引入主观性和方差，直接传播不可靠的奖励信号会误导搜索方向。不确定性传播使搜索对评估噪声更鲁棒，在复杂问题上显著减少搜索时间

### 损失函数 / 训练策略

SolverLLM是training-free框架。奖励函数为加权组合：$R(f_s, x^*) = \alpha \cdot \mathbb{I}_{feasible} + \beta \cdot \text{objective\_score}(f_s, x^*) - \gamma \cdot \mathbb{I}_{error}$，选择策略使用UCT：$s_{child} = \arg\max_{s'} [Q_{s'} + c \cdot \sqrt{2\log N_s / N_{s'}}]$。

## 实验关键数据

### 主实验

与prompt方法对比（Solving Accuracy, SA）：

| 数据集 | GPT-4o直接 | OptiMUS | Chain-of-Experts | SolverLLM | 提升 |
|--------|-----------|---------|-------------------|-----------|------|
| NL4Opt | 81.0% | 78.8% | 64.2% | **97.0%** | +18.2% |
| NLP4LP | 32.4% | 72.0% | 53.1% | **87.0%** | +15.0% |
| ComplexOR | 27.3% | 66.7% | 38.1% | **77.8%** | +11.1% |

与学习方法对比：

| 数据集 | LLMOPT (SFT) | ORLM-LLaMa3 | SolverLLM | 提升 |
|--------|-------------|-------------|-----------|------|
| MamoEasy | **97.0%** | 82.3% | 96.0% | -1.0% |
| NL4Opt | 93.0% | 85.7% | **97.0%** | +4.0% |
| MamoComplex | 68.0% | 37.4% | **76.0%** | +8.0% |
| IndustryOR | 46.0% | 38.0% | **56.0%** | +10.0% |

### 消融实验

| 变体 | NL4Opt SA | MamoComplex SA | IndustryOR SA | 平均AGT变化 |
|------|----------|---------------|---------------|------------|
| SolverLLM (完整) | **97.0%** | **76.0%** | **56.0%** | 基准 |
| w/o Prompt Backprop | 93.0% (-4%) | 69.0% (-7%) | 46.0% (-10%) | 略快 |
| w/o Uncertainty Backprop | 97.0% | 75.0% (-1%) | 56.0% | 复杂数据集显著变慢 |
| w/o Type Element | 96.0% (-1%) | 59.0% (-17%) | 48.0% (-8%) | 略变 |

### 关键发现

- Prompt Backpropagation对复杂问题贡献最大（IndustryOR上-10%），因为复杂问题更需要双向修正formulation
- Uncertainty Backpropagation主要提升搜索效率而非最终精度——在MamoComplex上AGT从4.34分钟降至3.85分钟
- Type Element对图相关优化问题特别有效，帮助避免变量类型的基础性错误
- Token效率分析：在20次搜索迭代下，SolverLLM用40920 tokens达到76% SA，而AutoFormulation用43150 tokens仅达37% SA
- 执行率(ER)几乎全部达100%，得益于代码生成阶段的error backpropagation

## 亮点与洞察

- **将测试时缩放应用于结构化建模问题的思路很好**：不同于在自由文本推理上做test-time scaling，优化问题的建模有清晰的元素分解结构，天然适合树搜索。六元素schema提供了有意义的搜索粒度
- **Prompt Backpropagation实现了搜索树中的"经验传承"**：传统MCTS只传播标量reward，SolverLLM还传播语义级别的修正指导，使后续搜索能"从前几轮的错误中学习"。这个机制可以迁移到其他需要LLM进行结构化生成的任务
- **不确定性传播是使用LLM-as-judge的最佳实践**：当用LLM打分时引入不确定性估计来down-weight不靠谱的评估，这是一个通用且值得推广的trick

## 局限与展望

- 推理成本较高——每个问题需要多次LLM调用进行MCTS搜索，AGT约2-4分钟，在对延迟敏感的场景中不实用
- 依赖GPT-4级别的强LLM作为backbone，在开源模型上的表现未评估
- 奖励函数中的objective_score依赖LLM主观评估，可能对某些问题类型（如非线性优化）不够准确
- 六元素分解是手动设计的，对于超出标准优化问题范畴的场景（如随机优化、多目标优化）可能需要调整
- 与LLMOPT在简单数据集上持平（MamoEasy差1%），说明在训练数据分布内微调方法仍有竞争力，SolverLLM的优势主要体现在分布外泛化

## 相关工作与启发

- **vs Chain-of-Experts**: Chain-of-Experts使用固定的多智能体工作流（interpreter→modeler→coder→reviewer），依赖精心设计的prompt模板。SolverLLM通过MCTS动态搜索而非固定流程，灵活性更高
- **vs LLMOPT**: LLMOPT通过SFT+alignment在领域内表现出色，但受训练分布限制。在复杂OOD问题（MamoComplex, IndustryOR）上SolverLLM的优势明显，说明搜索式推理在泛化上优于记忆式学习
- **vs AutoFormulation（另一个MCTS方法）**: AutoFormulation使用标准MCTS做四元素搜索，SolverLLM通过动态扩展、prompt反传和不确定性传播增强了搜索效率和质量，在所有数据集上SA更高且token消耗更少

## 评分

- 新颖性: ⭐⭐⭐⭐ MCTS+LLM的组合不新，但动态扩展和推理信号反传的设计有创新性
- 实验充分度: ⭐⭐⭐⭐ 6个数据集、多种基线、详细消融和case study，token效率分析也很到位
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观，case study有助于理解各组件作用
- 价值: ⭐⭐⭐⭐ 展示了test-time scaling在结构化领域的潜力，对优化问题求解的实际应用有指导意义

<!-- RELATED:START -->

## 相关论文

- [EvoRefuse: 用进化提示优化评估和缓解LLM过度拒绝](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [Reparameterized LLM Training via Orthogonal Equivalence Transformation](reparameterized_llm_training_via_orthogonal_equivalence_transformation.md)
- [Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](../../ACL2025/llm_nlp/automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)
- [Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)

<!-- RELATED:END -->
