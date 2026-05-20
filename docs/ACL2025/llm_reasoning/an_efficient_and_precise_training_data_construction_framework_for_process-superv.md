---
title: >-
  [论文解读] An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning
description: >-
  [ACL 2025][LLM推理][过程监督奖励模型] 本文提出EpicPRM框架，通过基于困惑度（perplexity）的Monte Carlo估计量化每个推理步骤的贡献度，并使用自适应二分搜索高效定位首个错误步骤，构建了高质量的过程监督数据集Epic50k（仅50k标注步骤）…
tags:
  - "ACL 2025"
  - "LLM推理"
  - "过程监督奖励模型"
  - "数据构建"
  - "Monte Carlo估计"
  - "二分搜索"
  - "EpicPRM"
---

# An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning

**会议**: ACL 2025  
**arXiv**: [2503.02382](https://arxiv.org/abs/2503.02382)  
**代码**: [https://github.com/xiaolizh1/EpicPRM](https://github.com/xiaolizh1/EpicPRM)  
**领域**: LLM推理 / 数学推理  
**关键词**: 过程监督奖励模型、数据构建、Monte Carlo估计、二分搜索、EpicPRM

## 一句话总结
本文提出EpicPRM框架，通过基于困惑度（perplexity）的Monte Carlo估计量化每个推理步骤的贡献度，并使用自适应二分搜索高效定位首个错误步骤，构建了高质量的过程监督数据集Epic50k（仅50k标注步骤），训练出的PRM性能可媲美甚至超越在PRM800k上训练的模型。

## 研究背景与动机

**领域现状**：提升LLM的数学推理能力是AI领域的核心挑战。过程监督（process supervision）通过对推理链中每个步骤提供正确性标注来训练PRM（Process-Supervised Reward Model），已被证明比结果监督更有效。

**现有痛点**：现有的过程监督数据构建方法面临成本和质量的两难：（1）人工标注法（如PRM800k）质量高但标注800k个步骤耗费巨大人力和财力，难以扩展到新领域；（2）自动标注法（如Math-Shepherd）使用Monte Carlo（MC）估计每个步骤的正确性，但需要对每个步骤采样大量rollout，计算成本高且标注精度受限于采样数量。

**核心矛盾**：MC估计用正确rollout的计数比例 $M/N$ 来近似某个步骤正确的概率，但当采样数 $N$ 不够大时，这个估计不够精确（如掷硬币两次都正面不代表概率100%）。此外，对一条错误推理链的所有步骤逐一进行MC估计过于浪费——实际只需找到第一个错误步骤。

**本文目标**：在降低计算成本的同时提高过程监督标注的精度。

**切入角度**：（1）用困惑度替代计数来更精确地估计MC概率；（2）用自适应二分搜索替代逐步搜索来更高效地定位首个错误步骤。

**核心 idea**：将步骤正确性定义为其对最终答案的"贡献度"，通过基于困惑度的MC估计（MC_PPL）量化贡献度，并使用根据题目难度自适应调整的二分搜索算法高效找到首个错误步骤。

## 方法详解

### 整体框架
EpicPRM框架分为三步：（1）使用多个LLM生成多样化的CoT推理链；（2）对每条错误链使用自适应二分搜索+MC_PPL定位首个错误步骤；（3）标注首个错误步骤之前的所有步骤为正确、之后为错误，构成训练数据。

### 关键设计

1. **基于困惑度的MC估计（MC_PPL）**:

    - 功能：更精确地估计从某个推理状态出发获得正确答案的概率
    - 核心思路：传统MC估计用 $M/N$ （$N$ 次rollout中 $M$ 次正确）来近似概率，但采样量小时误差大。本文用困惑度权重替代计数：$MC_{PPL}(s_t, \theta_{1:K}) = \frac{1}{K} \sum_{k=1}^{K} \frac{\sum_{m=1}^{M} \log PPL(j; s_t, \theta_k)}{\sum_{n=1}^{N} \log PPL(j; s_t, \theta_k)}$。其中 $PPL$ 直接计算模型生成每个rollout的概率。对比简单的计数，PPL权重考虑了每个rollout的生成概率，避免了"偶然正确但概率极低"的rollout误导估计
    - 设计动机：困惑度可以直接从模型获取，不需要额外采样。当一个正确的rollout生成概率极低时，计数法会高估正确概率，但PPL法不会

2. **步骤贡献度量化**:

    - 功能：判断某个步骤是"有贡献的正确步骤"还是"碰巧没影响的错误步骤"
    - 核心思路：定义步骤 $s_t$ 的贡献度为 $MC_{PPL}(s_t) - MC_{PPL}(s_{t-1})$。如果一个步骤对最终正确答案的概率没有正向贡献（贡献度≤0），即使后续rollout碰巧得到正确答案，也将其标记为潜在错误。这解决了"错误步骤+纠错"导致误标为正确的问题（如论文图1展示的案例）
    - 设计动机：传统方法假设"如果从某步出发能得到正确答案则该步正确"，忽略了完成器（completer）的自我纠错能力。实际上一个明显错误的步骤后面可能因为LLM的纠错能力而最终得到正确答案

3. **自适应二分搜索**:

    - 功能：高效定位推理链中的首个错误步骤
    - 核心思路：将"找到首个错误步骤"建模为有序序列中的搜索问题。利用二分搜索将MC估计次数从 $O(n)$（逐步搜索）降低到 $O(\log n)$。关键改进是自适应性：（1）根据题目难度调整搜索起点——难题的错误通常出现在早期步骤，简单题的错误出现在后期步骤，因此根据题目难度调整二分搜索的初始位置；（2）根据MC_PPL值的确信度动态调整每个搜索点的采样数量——确信度高的位置少采样，边界附近多采样
    - 设计动机：实验发现难题的首个错误步骤位置与总步数的比值较低（约0.3），简单题较高（约0.7）。利用这个先验知识可以减少搜索步骤

### 损失函数 / 训练策略
PRM的训练使用标准的过程监督损失——对每个标注的步骤预测其正确性标签（正确/错误），使用交叉熵损失。

## 实验关键数据

### 主实验

| PRM训练数据 | 数据量 | MATH best-of-N | GSM8K best-of-N |
|------------|--------|----------------|-----------------|
| PRM800k (人工标注) | 800k步骤 | 68.4 | 82.1 |
| Math-Shepherd | ~440k步骤 | 66.2 | 80.8 |
| **Epic50k** | **50k步骤** | **69.1** | **82.5** |

### 消融实验

| 配置 | MATH BoN | 说明 |
|------|---------|------|
| Full EpicPRM | 69.1 | 完整框架 |
| 用M/N替代MC_PPL | 65.8 | PPL贡献 +3.3 |
| 用逐步搜索替代二分搜索 | 68.7 | 精度相当但成本高64% |
| 无步骤贡献度 | 66.5 | 贡献度过滤很重要 |
| 无多模型rollout | 67.2 | 多模型增加多样性有帮助 |

### 关键发现
- Epic50k仅用PRM800k不到10%的数据量，训练出的PRM性能却更优，有力证明了"数据质量 > 数据量"
- MC_PPL相比传统计数法带来3.3分的提升，说明困惑度确实提供了更精确的MC估计
- 自适应二分搜索相比逐步搜索降低了64.39%的标注计算成本，而精度损失极小（<0.5分）
- 题目难度与首个错误步骤位置的强相关性是一个有趣的实验发现，为搜索优化提供了先验

## 亮点与洞察
- 用困惑度替代计数进行MC估计是一个简单但有效的改进，利用了LLM可以直接输出token概率这一特性。这个思路可以推广到所有使用MC估计的场景
- "步骤贡献度"概念巧妙地处理了LLM自我纠错能力导致的标注噪声问题。这在PRM文献中是一个被忽视但实际影响很大的问题
- 自适应二分搜索将标注成本降低64%，使得高质量过程监督数据的构建变得实际可行

## 局限与展望
- 当前仅在数学推理领域验证，对代码生成、科学推理等其他需要过程监督的领域还需要验证
- 自适应搜索依赖"难题错误靠前"的先验假设，对于某些题型可能不成立
- MC_PPL需要完成器模型能输出token级别概率，对于闭源API模型不适用
- Epic50k的数据质量可能受限于完成器模型的能力，更强的完成器可能产生更好的数据

## 相关工作与启发
- **vs PRM800k (Lightman et al.)**: PRM800k依赖大量人工标注，EpicPRM全自动且数据量减少20倍，质量更优
- **vs Math-Shepherd**: Math-Shepherd使用传统MC估计，精度受限。EpicPRM的PPL估计和贡献度过滤提供更高质量标注
- **vs OmegaPRM**: OmegaPRM提出了二分搜索思想，但EpicPRM加入了自适应性和PPL估计的改进

## 评分
- 新颖性: ⭐⭐⭐⭐ PPL替代计数和步骤贡献度是有价值的技术创新，但整体是现有方法的改进
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分，但评测基准较少（仅MATH和GSM8K）
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐⭐ 显著降低高质量PRM数据成本，对数学推理研究有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](../../NeurIPS2025/llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[ACL 2025\] Dynamic and Generalizable Process Reward Modeling (DG-PRM)](dgprm_dynamic_process_reward.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](../../ICML2026/llm_reasoning/grpo_is_secretly_a_process_reward_model.md)

</div>

<!-- RELATED:END -->
