---
title: >-
  [论文解读] How to Make the Most of Your Masked Language Model for Protein Engineering
description: >-
  [ICLR 2026][医学图像][蛋白质语言模型] 提出基于温度退火随机束搜索（SBS）的MLM采样方法，利用伪似然的野生型边际近似实现高效全序列评估，在真实抗体治疗优化的体外实验中证明采样算法选择至少与模型选择同等重要，SBS+引导达到100%成功率。
tags:
  - ICLR 2026
  - 医学图像
  - 蛋白质语言模型
  - 掩码语言模型
  - 随机束搜索
  - 抗体优化
  - 多目标优化
---

# How to Make the Most of Your Masked Language Model for Protein Engineering

**会议**: ICLR 2026  
**arXiv**: [2603.10302](https://arxiv.org/abs/2603.10302)  
**代码**: 无  
**领域**: 蛋白质工程 / 抗体设计  
**关键词**: 蛋白质语言模型, 掩码语言模型, 随机束搜索, 抗体优化, 多目标优化

## 一句话总结
提出基于温度退火随机束搜索（SBS）的MLM采样方法，利用伪似然的野生型边际近似实现高效全序列评估，在真实抗体治疗优化的体外实验中证明采样算法选择至少与模型选择同等重要，SBS+引导达到100%成功率。

## 研究背景与动机

**领域现状**：大量蛋白质语言模型（ESM-2、AbLang2等）已发布，被广泛用于抗体治疗药物的迭代优化——给定种子序列，用模型生成突变候选，在实验室测试后进入下一轮。但如何从MLM中高效采样高质量序列的研究极度匮乏。

**现有痛点**：(1) 主流MLM采样方法是"突变中心"的（如去噪采样、Gibbs采样），逐位掩码-填充，计算开销 $O(EL^3)$，且经验上倾向生成低质量序列；(2) 现有方法难以整合不可微的评分函数（如OASis免疫原性评分、等电点）；(3) MLM采样算法的系统性评估——特别是体外验证——几乎不存在。

**核心矛盾**：突变中心的采样范式将决策局限于单个位置，无法利用全序列信息做全局评判；同时 $O(EL^3)$ 的计算复杂度限制了候选多样性。

**本文目标** 如何从MLM高效采样出高似然、多样且贴近种子序列的突变序列？如何灵活集成多种评分函数（可微/不可微）进行多目标优化？什么模型+什么采样策略在实际抗体优化中最有效？

**切入角度**：将MLM采样从"突变中心"转变为"序列中心"——不是让模型逐位生成突变，而是让模型评估完整序列的伪似然（PLL），将问题转化为搜索问题。关键发现：计算一个序列的PLL后，其所有单位点邻居的PLL几乎免费。

**核心 idea**：利用野生型边际近似，从单次PLL计算免费获得整个1-编辑邻域的近似PLL，将MLM采样转化为高效的随机束搜索。

## 方法详解

### 整体框架
种子序列 → 计算PLL及全部单突变邻居的近似PLL → 随机束搜索（SBS）扩展 $E$ 步 → 用温度参数平衡似然与多样性 → 可选：与评分函数（结合亲和力模型、OASis、等电点等）进行多目标优化（NDS或STS加权） → 输出候选序列集合 → 体外测试。

### 关键设计

1. **基于PLL的高效全序列评估**:

    - 做什么：以 $O(BEL^3)$ 的代价获得 $20BEL$ 个候选序列的近似PLL
    - 核心思路：对模板序列 $\mathbf{x}$ 计算完整PLL需要 $L$ 次前向传播得到精确条件概率 $\hat{p}(x_i|x_{j \neq i})$。对于距模板一个替换的序列 $\mathbf{x}'$（位置 $k$ 不同），其PLL近似为：$PLL(\mathbf{x}') \approx \sum_i \log(\text{softmax}_\tau(\tilde{Y}^{(i)}_{i,\text{index}(x'_i)}))$，其中 $k$ 位置使用精确条件概率，其他位置复用模板的条件概率（野生型边际近似）。这样一次模板PLL计算就覆盖了全部 $20L$ 个单突变邻居。
    - 设计动机：突变中心方法每生成一个序列需 $O(EL^3)$，而本方法将束搜索每步扩展的代价降到 $O(BL^3)$（$B$ 为束宽），$E$ 步总代价 $O(BEL^3)$ 获得 $20BEL$ 个候选，实现了指数级效率提升。

2. **温度退火随机束搜索（SBS）**:

    - 做什么：在搜索过程中平衡序列似然与候选多样性
    - 核心思路：采用Gumbel-top-k技巧：对每个候选序列的PLL加上Gumbel噪声后排序选择束内序列。softmax温度 $\tau$ 缩放似然项但不缩放Gumbel项，因此 $\tau$ 控制了确定性（高似然）和随机性（多样性）的权衡。从种子出发搜索 $E$ 步自然保证了近端约束。
    - 设计动机：纯贪心选择（argmax）会丧失多样性，而纯随机采样序列质量低。SBS通过Gumbel噪声实现了无替换的序列采样，同时保持批次内多样性。

3. **灵活的多目标优化（MOO）引导**:

    - 做什么：无缝集成任意评分函数（可微或不可微）进行多目标联合优化
    - 核心思路：由于采用序列中心评估，MLM和所有评分函数都被视为黑箱——只需输入完整序列输出分数。支持两种多目标标量化：Pareto非支配排序（NDS）在目标间做trade-off，平滑切比雪夫标量化（STS）同时最大化满足所有目标。STS允许对不同目标加权。
    - 设计动机：突变中心方法必须在部分掩码序列上评估评分函数，许多实际评分函数（OASis、等电点）无法处理掩码输入。序列中心方法彻底消除了这一限制。

### 损失函数 / 训练策略
MLM本身为预训练模型，不涉及训练。评估了9种MLM（ESM-2 35M/150M/650M、Sapiens、AbLang2、AMPLIFY 120M/350M、DiffAbOpt、内部SAbDabMLM）和3种CLM（pIgGen、pIgGen-dev、CloneLM）。体外实验使用单一FAb抗体种子，每种方法生成≥21个样本。

## 实验关键数据

### 主实验（体外）

| 方法 | 成功率↑ | 说明 |
|--------|------|------|
| AbLang2 + Beam Search | ~65% | 无监督最佳之一 |
| ESM2-650M + Beam Search | ~60% | 非抗体专属模型表现优秀 |
| AbLang2 + Gibbs | ~40% | 同模型，束搜索显著优于Gibbs |
| Sapiens + Gibbs | ~25% | 弱模型+弱采样 |
| AbLang2 + 监督排序 | ~75% | 利用729样本训练的分类器排序 |
| AbLang2 + STS引导 | **100%** | 多目标引导生成+排序 |
| AbLang2 + NDS引导 | ~90% | Pareto排序也显著提升 |

### 消融实验

| 配置 | 关键观察 | 说明 |
|------|---------|------|
| Beam vs Gibbs (同模型) | Beam在所有3个模型上均胜出 | 采样算法比模型选择更重要 |
| ESM2-650M（通用蛋白） | 与抗体专属AbLang2相当 | 通用模型意外地适用于抗体 |
| Gibbs-argmax vs Gibbs | argmax更高成功率但多样性低 | Gibbs倾向生成低质量序列 |
| 有/无监督引导 | 引导后弱结合被消除 | 但引导降低了人源性得分 |

### 关键发现
- 采样算法的选择至少与模型选择同样重要——同一模型换不同采样器可导致成功率翻倍差距
- ESM2-650M虽然训练于通用蛋白质数据（非抗体特异），在抗体优化中仍表现优秀，说明蛋白质通用分布的捕获比抗体特异性更重要
- Gibbs采样倾向于未能忠实反映模型偏好的序列——模型"想要"的序列和Gibbs"给出"的序列之间存在系统性偏差
- 监督引导方法消除了弱结合抗体的生成，但带来了人源性降低的副作用，需要将人源性纳入多目标优化

## 亮点与洞察
- 将MLM采样从"突变中心"思维转变为"序列中心"思维是关键洞察：利用野生型边际近似实现了指数级的效率提升，使束搜索成为可能。这一视角揭示了之前MLM采样方法的根本低效——它们丢弃了大量已计算的信息。
- 罕见的体外验证：不仅做了in silico评估，还在真实抗体治疗项目中进行了289个样本的head-to-head比较，结果可直接指导工业实践。

## 局限与展望
- 体外实验基于单一FAb抗体种子，泛化性有待更多治疗项目验证
- 野生型边际近似对多位点突变的精度随编辑距离增加而下降，束搜索每步展开时刷新模板缓解了这一问题，但累积误差仍需研究
- STS引导在获得100%成功率的同时降低了人源性，多目标优化中的trade-off管理需进一步研究

## 相关工作与启发
- **vs ESM-3 (Hayes et al., 2025)**: ESM-3用去噪采样+无导数引导做蛋白质优化，但需要评分函数接受部分掩码输入。本文方法使用完整序列评估，兼容任何评分函数
- **vs DiffAbOpt/DiffAb+ (Raghu et al., 2025)**: 基于结构的扩散方法仅在CDR区域操作。本文方法可在序列任意区域采样，且体外实验中DiffAb+表现一般

## 评分
- 新颖性: ⭐⭐⭐⭐ 序列中心+PLL近似的采样范式转变简洁有力，但技术原理相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 罕见的大规模体外验证，9种MLM+3种CLM的系统比较，监督/无监督/引导的完整对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验发现的呈现和讨论到位
- 价值: ⭐⭐⭐⭐⭐ 对抗体工程实践有直接指导价值，"采样算法比模型更重要"的结论颠覆性强

<!-- RELATED:START -->

## 相关论文

- [Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)
- [mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)

<!-- RELATED:END -->
