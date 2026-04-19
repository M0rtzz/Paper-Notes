---
title: >-
  [论文解读] Enough Coin Flips Can Make LLMs Act Bayesian
description: >-
  [ACL 2025][LLM/NLP][贝叶斯推理] 通过抛硬币这一受控随机过程，系统研究LLM是否在in-context learning中执行贝叶斯推理，发现LLM通常具有偏置先验，但随着上下文证据增加会以近似贝叶斯更新的方式修正后验估计，偏差主要源于校准不良的先验而非错误的更新机制。
tags:
  - ACL 2025
  - LLM/NLP
  - 贝叶斯推理
  - In-Context Learning
  - 概率估计
  - 先验偏差
  - 注意力机制
---

# Enough Coin Flips Can Make LLMs Act Bayesian

**会议**: ACL 2025  
**arXiv**: [2503.04722](https://arxiv.org/abs/2503.04722)  
**代码**: 有（项目页面）  
**领域**: LLM可解释性  
**关键词**: 贝叶斯推理, In-Context Learning, 概率估计, 先验偏差, 注意力机制  

## 一句话总结

通过抛硬币这一受控随机过程，系统研究LLM是否在in-context learning中执行贝叶斯推理，发现LLM通常具有偏置先验，但随着上下文证据增加会以近似贝叶斯更新的方式修正后验估计，偏差主要源于校准不良的先验而非错误的更新机制。

## 研究背景与动机

**领域现状**: 大型语言模型通过in-context learning (ICL)展现出强大的少样本学习能力，已有理论工作（Xie et al. 2021等）认为ICL可能在隐式执行贝叶斯推理，但由于大多数任务中真实后验不可知，这一假设难以直接验证。

**现有痛点**: (1) 已有研究使用的任务（问答、语言建模等）后验分布未知，无法精确评估模型推理是否符合贝叶斯规范；(2) 受控理论研究依赖于对模型架构或数据域的强假设，难以推广到预训练LLM；(3) 尚不清楚LLM的测试时行为是模式匹配还是结构化概率推理。

**核心矛盾**: ICL的成功与贝叶斯推理的联系在直觉上合理，但在实证上缺乏直接证据——我们需要一个真实后验可精确计算的受控环境来验证。

**本文目标**: 预训练LLM在序列证据下是否真正执行贝叶斯后验更新？其偏差来源是先验校准问题还是更新机制缺陷？

**切入角度**: 使用有偏硬币抛掷作为受控随机过程——二项分布+Beta先验的共轭系统使所有贝叶斯量都可精确计算，从而直接对比LLM的预测与理论后验。

**核心 idea**: 通过可精确计算后验的硬币抛掷实验，直接验证LLM在ICL中以近贝叶斯方式更新先验。

## 方法详解

### 整体框架

设计一系列受控实验：(1) 提取LLM对硬币抛掷结果的先验分布（零样本设置）；(2) 通过显式偏置指令测试先验更新；(3) 通过ICL提供有偏硬币序列，对比模型预测与贝叶斯后验的total variation distance (TVD)；(4) 构造动态偏置切换场景（前50次θ₁=0.75，后50次θ₂=0.25），分析"在线"贝叶斯更新行为；(5) 分析注意力权重与更新质量的关系。

### 关键设计

1. **先验提取与偏置分析**
    - **功能**: 量化LLM对抛硬币结果的内在先验
    - **核心思路**: 用50种不同的prompt变体（如"I flipped a coin and it landed on"）查询模型，提取归一化logit值作为对"heads"概率的先验估计；再通过显式偏置陈述（如"coins land on heads X% of the time"）测试先验能否被更新
    - **设计动机**: 理解LLM的起始偏差是评估其贝叶斯行为的前提——如果先验严重偏差而更新机制正确，与先验正确而更新错误是完全不同的诊断

2. **ICL序列下的后验估计**
    - **功能**: 评估LLM是否随ICL证据累积而收敛到正确后验
    - **核心思路**: 提供不同长度（3到100个样本）的有偏硬币序列作为ICL示例，对比模型预测分布与理论Beta后验分布之间的TVD
    - **设计动机**: 如果LLM执行贝叶斯更新，随证据增加TVD应趋向零；如果只是模式匹配，收敛模式会有质的不同

3. **带衰减因子的贝叶斯滤波拟合**
    - **功能**: 精确刻画LLM的更新行为并量化其"记忆时间窗"
    - **核心思路**: 在动态偏置切换场景中，用指数衰减因子γ修正贝叶斯更新 $\alpha \leftarrow \gamma\alpha + I(H)$，通过L-BFGS-B优化每个模型的最佳γ值；γ<1表示模型更倾向"局部贝叶斯更新"，对近期证据赋予更大权重
    - **设计动机**: 经典贝叶斯滤波（γ=1）无法完全解释LLM行为，引入γ参数发现模型执行的是"近视贝叶斯"更新，且γ值因模型架构而异

### 损失函数 / 训练策略

本文为分析性研究，不涉及模型训练。核心评估指标为total variation distance (TVD)：

$$\delta(p^*, \hat{p}_{\mathcal{M}}) = \frac{1}{2} \sum_{o \in \Omega} |p^*(o) - \hat{p}_{\mathcal{M}}(o)|$$

通过提取模型logits并归一化到定义的输出空间 $\Omega = \{\text{heads}, \text{tails}\}$ 上来获取模型预测分布。

## 实验关键数据

### 主实验

不同模型在有偏硬币ICL任务中的最佳拟合γ值：

| 模型 | Best-Fit γ | 含义 |
|------|-----------|------|
| Llama3.1-8B | 0.8807 | 接近经典贝叶斯，长时间窗 |
| Llama3.1-8B-Instruct | 0.4655 | 更局部更新，短时间窗 |
| Phi-2 | 0.8781 | 接近经典贝叶斯 |
| Mistral-7B | 0.6903 | 中等时间窗 |
| Mistral-7B-Instruct | 0.9107 | 例外——指令微调后更接近经典贝叶斯 |
| Gemma-2-2B | 0.4910 | 较短时间窗 |
| Gemma-2-2B-Instruct | 0.3087 | 最局部的更新 |
| OLMoE-1B-7B | 0.3268 | 较短时间窗 |

### 消融实验

关键贝叶斯行为特性验证：

| 实验设置 | 关键发现 |
|---------|---------|
| 显式偏置指令（非Instruct模型） | 忽略指令，始终输出~60-80%偏向heads |
| 显式偏置指令（Instruct模型） | 略有改善，极端偏置（0%/100%）表现最好 |
| 模型大小缩放（Pythia 70M→12B） | 模型大小对先验质量和ICL效果无显著影响 |
| ICL样本数量 | 3个样本即显著改善TVD，但100个仍不能完全消除先验偏差 |
| 注意力权重 vs 后验质量 | 两者几乎无相关性（R=0.02, p=0.48） |

### 关键发现

1. **所有模型都有偏向heads的先验**: 可能与tokenization结构有关——"tails"在某些模型中需要两个token编码
2. **显式指令不如ICL示例有效**: 非Instruct模型几乎完全忽略偏置指令，Instruct模型也仅在极端值处改善
3. **ICL整体符合贝叶斯后验更新**: 偏差主要来自先验校准不良，而非更新机制错误
4. **模型执行"近视贝叶斯"**: 拟合的γ<1表明模型对近期证据更敏感，等效于有限时间窗的贝叶斯更新
5. **Instruct微调降低γ值**: 指令微调使模型更"健忘"，更愿意在新证据面前切换行为
6. **注意力幅度与更新质量无关**: 推翻了先前认为attention直接调控贝叶斯更新的假设
7. **模型缩放对贝叶斯行为影响有限**: 更大模型不一定更好地执行概率推理

## 亮点与洞察

- 实验设计非常巧妙——利用二项/Beta共轭系统使所有贝叶斯量可精确计算
- "近视贝叶斯"（γ<1）的发现为ICL在长上下文中的loss偏高提供了自然解释
- 指令微调降低γ值这一发现有深远意义——暗示Instruct模型更倾向"局部适应"而非"全局积累"
- 注意力幅度与更新质量不相关的发现挑战了先前的理论解释

## 局限与展望

- 仅评估离散二元输出（heads/tails），未扩展到连续分布或更高维随机过程
- 无法应用于闭源模型（需要logits访问）
- 硬币抛掷过于简化，在更复杂的NLP任务中贝叶斯行为是否仍成立有待验证
- 先验偏差的来源未深入追踪——是训练数据中"heads"出现频率更高还是其他原因？
- 未探讨思维链(CoT)提示对贝叶斯行为的影响

## 相关工作与启发

- Xie et al. (2021): ICL作为隐式贝叶斯推理的理论基础
- Wang et al. (2024): ICL中prompt顺序敏感性与隐变量模型的联系
- Falck et al. (2024): 从鞅论视角分析ICL的贝叶斯性质
- 本文的γ参数概念可解释为Bayesian filtering中的"forgetting factor"

## 评分

- **新颖性**: 4/5 — 受控实验设计巧妙，"近视贝叶斯"发现新颖
- **技术深度**: 4/5 — 贝叶斯理论框架严谨，多维度实证分析
- **实验充分度**: 4/5 — 多模型、多缩放、注意力分析全面
- **实用性**: 3/5 — 以理论洞察为主，对实际应用启发间接
- **综合评分**: 4/5

<!-- RELATED:START -->

## 相关论文

- [LLMs Can Be Easily Confused by Instructional Distractions](llms_can_be_easily_confused_by_instructional_distractions.md)
- [Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [LLMs Can Simulate Standardized Patients via Agent Coevolution](evopatient_standardized_patient.md)
- [LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)
- [Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)

<!-- RELATED:END -->
