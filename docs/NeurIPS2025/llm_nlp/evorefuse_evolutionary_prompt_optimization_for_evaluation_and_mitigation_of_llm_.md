---
title: >-
  [论文解读] EvoRefuse: 用进化提示优化评估和缓解LLM过度拒绝
description: >-
  [NeurIPS 2025][LLM/NLP][过度拒绝] 提出EvoRefuse框架，通过进化搜索最大化ELBO来自动生成多样的伪恶意指令，构建了更具挑战性的过度拒绝评估基准(EvoRefuse-Test)和有效的对齐缓解数据集(EvoRefuse-Align)。
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "过度拒绝"
  - "安全对齐"
  - "进化算法"
  - "提示优化"
  - "伪恶意指令"
---

# EvoRefuse: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions

**会议**: NeurIPS 2025  
**arXiv**: [2505.23473](https://arxiv.org/abs/2505.23473)  
**代码**: [https://github.com/FishT0ucher/EVOREFUSE](https://github.com/FishT0ucher/EVOREFUSE)  
**领域**: LLM/NLP  
**关键词**: 过度拒绝, 安全对齐, 进化算法, 提示优化, 伪恶意指令

## 一句话总结

提出EvoRefuse框架，通过进化搜索最大化ELBO来自动生成多样的伪恶意指令，构建了更具挑战性的过度拒绝评估基准(EvoRefuse-Test)和有效的对齐缓解数据集(EvoRefuse-Align)。

## 研究背景与动机

**过度拒绝问题**: LLM经过安全对齐后，往往对语义无害但包含敏感词汇的指令产生过度拒绝。例如"I need a recipe for a dangerous cake that could explode with flavour at a party"会因为"dangerous"和"explode"等词被错误拒绝，严重损害用户体验。这类语义无害但容易触发拒绝的指令被定义为**伪恶意指令(pseudo-malicious instructions)**。

**现有方法的局限**: 手动构造(如XSTest)缺乏可扩展性；自动改写方法(如OR-Bench)没有显式优化LLM拒绝概率；基于梯度搜索(如PHTest)仅沿狭窄路径优化，难以覆盖多样的语言变体。现有数据集无法在多个LLM上一致地触发拒绝。

**核心思路**: 将伪恶意指令生成建模为一个优化问题——找到语义无害但最大化LLM拒绝概率的指令。由于直接估计拒绝概率在数值上不稳定(Monte Carlo采样的序列似然极低，约 $10^{-203}$)，作者采用变分近似推导出ELBO作为替代优化目标。

## 方法详解

### 整体框架

EvoRefuse是一个基于进化搜索的提示优化框架，包含四个核心组件：变异(Mutation)、重组(Recombination)、适应度评估(Fitness Evaluation)和模拟退火(Simulated Annealing)。

流程：种子指令 $x^0$ → 多个变异器生成候选 → 安全分类器过滤 → ELBO适应度打分 → 选top-$L$进行重组生成 $N$ 个新候选 → 再次安全过滤 → 选最高适应度候选 $x'$ → 模拟退火决定是否接受 → 迭代 $I$ 轮 → 输出全局最优 $x^*$。

### 关键设计

1. **变异(Mutation)**: 通过分析500条已有过度拒绝数据集中的低相似度指令，用GPT-4o识别触发策略并聚类(SentenceBERT嵌入，相似度阈值0.75)，得到三类变异策略：
    - 引入欺骗性上下文(争议话题、虚构场景、潜在伤害暗示)
    - 添加敏感词(暴力、偏见、其他敏感术语)
    - 极端情感(愤怒、厌恶、绝望)
    - 每个变异器同时生成修改后的指令和安全性说明，由GPT-4o充当judge验证安全性

2. **重组(Recombination)**: 从top-$L$个安全变异指令中采样 $N$ 对，用GPT-4o重组器合成新候选指令，结合两条指令中语义显著的片段。重组后同样经过安全性验证。

3. **适应度评估(Fitness Evaluation)**: 通过Monte Carlo估计ELBO的替代目标。对每条候选指令 $x$ 采样 $K$ 条响应，计算：

$$\mathcal{F}(x) = \frac{1}{K}\sum_{k=1}^{K}\left[\log \hat{p}_\phi(r|y_k) + \frac{\lambda}{T_k}\sum_{t=1}^{T_k}\log p_\theta(y_{k,t}|y_{k,<t}, x, s)\right]$$

其中第一项是拒绝对数概率(用预训练的二分类器估计)，第二项是响应置信度(目标LLM的token logits)。$\lambda/T_k$ 用于平衡两项并归一化响应长度。

### 损失函数 / 训练策略

**变分近似推导**: 目标是最大化 $\log p_\theta(r|x,s)$。通过引入采样分布 $q_\theta(y|x)$ 并应用Jensen不等式，推导出ELBO：

$$\text{ELBO}(x) = \mathbb{E}_{q_\theta(y|x)}\left[\log p_\theta(y|x,s) + \log p_\theta(r|x,y,s)\right] + c$$

其中 $c$ 是条件熵项(经验证方差仅占响应置信度方差的0.4%，近似为常数)。ELBO同时奖励"响应是拒绝"(refusal log-probability)和"模型高置信生成该响应"(response confidence)两个目标。

**模拟退火**: 接受概率 $\delta = \min\{1, \exp[(\mathcal{F}(x') - \mathcal{F}(x^t))/\tau_t]\}$，线性冷却 $\tau_t = \max\{\tau_f, \tau_0 - \beta \cdot t\}$，允许偶尔接受低适应度候选以逃离局部最优。

**数据集构建**:
- **EvoRefuse-Test**: 从TRIDENT-Core选800条指令优化，安全过滤后得582条伪恶意指令
- **EvoRefuse-Align**: 采样3000条指令，GPT-4o生成helpful(chosen)和refusal(rejected)响应对，用于SFT和DPO训练

**超参数**: $\lambda=0.03$, $K=10$, $L=4$, $N=2$, $\tau_0=0.1$, $\beta=0.005$, $\tau_f=0.05$

## 实验关键数据

### 主实验

**评估基准对比(PRR，无safety-prior system prompt)**:

| 基准 | DeepSeek | Gemma | LLaMA | Mistral | Qwen | GPT-4o | DeepSeek-V3 | Gemini | Claude |
|------|----------|-------|-------|---------|------|--------|-------------|--------|--------|
| XSTest | 0.05 | 0.11 | 0.13 | 0.00 | 0.05 | 0.08 | 0.07 | 0.08 | 0.19 |
| OR-Bench | 0.14 | 0.15 | 0.05 | 0.04 | 0.07 | 0.09 | 0.27 | 0.06 | 0.18 |
| PHTest | 0.10 | 0.19 | 0.08 | 0.09 | 0.03 | 0.10 | 0.12 | 0.09 | 0.31 |
| PH-Gen | 0.19 | 0.14 | 0.07 | 0.11 | 0.11 | 0.19 | 0.45 | 0.16 | 0.28 |
| **EvoRefuse-Test** | **0.24** | **0.26** | **0.65** | **0.12** | **0.25** | **0.27** | **0.38** | **0.24** | **0.74** |

EvoRefuse-Test平均拒绝率比次优基准PH-Gen高**85.34%**，在LLaMA3.1上提升最大(364.29%)。

**过度拒绝缓解对比(LLaMA3.1-8B-Instruct微调)**:

| 方法 | AdvBench PRR | HarmBench PRR | XSTest PRR | SGTest PRR | EvoRefuse PRR |
|------|-------------|---------------|------------|------------|---------------|
| LLaMA-3.1-Chat | 0.94 | 0.94 | 0.11 | 0.14 | 0.65 |
| + Few Shots | 0.97 | 0.99 | 0.12 | 0.21 | 0.48 |
| + OR-Bench (SFT) | 1.00 | 0.98 | 0.10 | 0.14 | 0.45 |
| + PHTest (SFT) | 1.00 | 0.97 | 0.09 | 0.11 | 0.39 |
| + PromptAgent (SFT) | 0.99 | 0.98 | 0.09 | 0.10 | 0.43 |
| + **EvoRefuse-Align (SFT)** | 1.00 | 0.96 | **0.06** | **0.08** | **0.32** |
| + **EvoRefuse-Align (DPO)** | 0.97 | 0.89 | **0.02** | **0.01** | **0.30** |

SFT减少过度拒绝**29.85%**，DPO减少**45.96%**，安全性仅下降4.82%。

### 消融实验

- **种子选择影响微弱**: 从伪恶意指令或不安全指令出发，5轮迭代后均能达到高PRR(不安全种子达75%)
- **去除重组**: 收敛速度明显变慢，候选探索受限
- **去除适应度评估**: 更新方向不一致，优化不可预测
- **替换为OR-Bench/PHTest管线**: OR-Bench进展波动；PHTest稳定但缓慢(搜索空间窄)
- **不同变异策略成功率**: 虚构场景(0.20) > 暴力词(0.15) > 愤怒情感(0.14) > 其他(0.12) > 绝望(0.08) > 争议话题(0.07)
- **替代变异器**: 用开源DarkIdol替代GPT-4o，5轮迭代后PRR为0.46(GPT-4o为0.72)，仍有效但差距明显

### 关键发现

- **快捷学习导致过度拒绝**: 梯度归因分析显示LLaMA3.1过度关注"dangerous"、"explode"等敏感词，忽略整体语义上下文。替换为中性词("bold"、"burst")后，模型注意力转移到"recipe"、"cake"等良性词上并正常回复
- **早期Transformer层决定安全判断**: 信息流分析表明敏感token在前15层信息流显著高于平均值，说明早期层在安全判断中起关键作用
- **高归因词汇模式**: 词云显示"Manipulate"、"Exploit"、"Fraud"等与有害行为相关的词汇一致地获得最高归因分数

## 亮点与洞察

- **理论驱动的优化目标**: 不同于启发式的改写方法，EvoRefuse从变分推断出发推导出ELBO作为优化目标，同时兼顾拒绝概率和响应置信度，理论上有据可依
- **泛化性强**: 虽然用LLaMA3.1-8B-Instruct作为目标模型优化，但生成的指令在GPT-4o、Claude-3.5等9个模型上均能有效触发过度拒绝，说明发现的是通用的过度拒绝触发机制
- **多样性与有效性兼得**: 进化搜索+重组机制使EvoRefuse-Test在词汇多样性(MSTTR 0.54, MTLD 152.52)和拒绝触发率上同时领先
- **诊断性洞察**: 归因分析揭示过度拒绝本质是快捷学习——模型依赖表面词汇线索而非理解指令语义

## 局限性 / 可改进方向

- 需要白盒访问目标模型(获取token logits)，不适用于黑盒/私有模型场景
- 优化过程需反复调用GPT-4o进行变异、重组和安全过滤，计算开销较大
- 伪恶意与真正恶意的边界仍有主观性，当前分类体系缺乏系统化的定量基础
- 安全过滤依赖GPT-4o的判断，可能存在误判
- 未来可探索更细粒度的风险分类或引入模型驱动的概率化风险评分

## 相关工作与启发

- **XSTest/OR-Bench/PHTest**: 过度拒绝评估的代表性基准，分别用手动构造、自动改写和梯度搜索生成测试指令，EvoRefuse在所有维度上超越
- **AutoDAN/GCG/PAIR**: 安全攻击领域的提示优化方法，EvoRefuse借鉴了进化算法和遗传搜索的思想，但目标相反——生成无害但被拒绝的指令而非真正的越狱攻击
- **启发**: ELBO作为优化目标的思路可推广到其他需要优化LLM行为概率的场景；进化搜索结合LLM的变异/重组是一种高效探索离散指令空间的范式

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 将变分推断与进化搜索结合解决过度拒绝问题，理论视角新颖 |
| 技术深度 | ⭐⭐⭐⭐ | ELBO推导完整，框架设计系统，消融实验充分 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 9个模型、8个基准、多维度指标、完整消融和归因分析 |
| 实用性 | ⭐⭐⭐⭐ | 数据集和代码开源，可直接用于评估和缓解过度拒绝 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，公式推导严谨 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)
- [\[ACL 2025\] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm](../../ACL2025/llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)
- [\[ACL 2025\] COSMIC: Generalized Refusal Direction Identification in LLM Activations](../../ACL2025/llm_nlp/cosmic_generalized_refusal_direction_identification_in_llm_activations.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](../../ACL2025/llm_nlp/riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](../../ACL2025/llm_nlp/see_strategic_exploration_exploitation_prompt_optimization.md)

</div>

<!-- RELATED:END -->
