---
title: >-
  [论文解读] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation
description: >-
  [ICML 2026][计算生物][GFlowNets] 本文提出一种无需训练的 GFlowNets 组合框架，通过用每个预训练模型的"到达概率"作为权重去混合各自的前向策略，使得在推理阶段就能针对任意线性标量化或逻辑算子的多目标组合直接采样，并在线性情形下被证明可精确恢复目标分布。
tags:
  - "ICML 2026"
  - "计算生物"
  - "GFlowNets"
  - "多目标生成"
  - "模型组合"
  - "推理时混合"
  - "分子设计"
---

# Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation

**会议**: ICML 2026  
**arXiv**: [2602.21565](https://arxiv.org/abs/2602.21565)  
**代码**: https://github.com/ml-postech/gflownet-composition (有)  
**领域**: 科学计算 / 生成式流网络 / 分子生成  
**关键词**: GFlowNets, 多目标生成, 模型组合, 推理时混合, 分子设计  

## 一句话总结
本文提出一种无需训练的 GFlowNets 组合框架，通过用每个预训练模型的"到达概率"作为权重去混合各自的前向策略，使得在推理阶段就能针对任意线性标量化或逻辑算子的多目标组合直接采样，并在线性情形下被证明可精确恢复目标分布。

## 研究背景与动机

**领域现状**：GFlowNets 是一类把"按奖励比例采样多样化候选"作为训练目标的生成模型，在分子、图、生物序列等离散结构的科学发现任务上特别受欢迎，因为它不像传统优化那样收敛到单点最优，而是覆盖整个高奖励区域。当问题涉及多个奖励（如结合亲和力 SEH、合成可达性 SA、药物相似性 QED）时，主流做法分两条路线：一是把多个奖励加权求和后整体训练一个偏好条件模型（MOGFN、HN-GFN）；二是把已经分别训练好的单目标模型通过分类器引导（compositional sculpting）组合成逻辑算子（调和均值代表"合取"、对比算子代表"差集"）。

**现有痛点**：两条路线都有共同的硬伤——任何对目标集合的修改都意味着要重新训练。偏好条件方法的奖励函数集合在训练阶段就被冻结，新增一个目标必须从头训；分类器引导虽然不动 GFlowNet，但每出现一组新组合就要重新训一个辅助分类器，而训练这个分类器的代价往往超过单独训练一个 GFlowNet。更进一步，线性标量化和逻辑算子被处理为两个互不兼容的子问题，没有一个统一的语言能同时覆盖。

**核心矛盾**：组合的"通用性"和"训练代价"之间存在 trade-off。要支持任意组合就必须知道组合算子的形式后再训练；要避免训练则必须放弃灵活性。但两者本质都建立在"分布层面操作"上，缺少一个能够直接在 GFlowNet 的内部结构——前向策略——上做组合的工具。

**本文目标**：(i) 把预训练的若干个单目标 GFlowNet 当作零样本组合的"积木"，在推理时拼出任意组合分布；(ii) 用同一个机制同时支持线性标量化和逻辑算子；(iii) 给出理论保证。

**切入角度**：作者注意到 GFlowNet 在终止态 $x$ 上的边缘分布可以分解为"到达概率 $u(s)$"与"终止动作概率 $p_F(s_f\mid x)$"之积，即 $p(x)=u(x)\cdot p_F(s_f\mid x)$。到达概率天然刻画了"某个模型把多少概率质量路由到状态 $s$"——也就是这个状态对这个模型来说有多"相关"。

**核心 idea**：用每个预训练 GFlowNet 在当前状态的到达概率作为局部权重，把各自的前向转移概率按目标组合函数 $\mathcal{G}$ 混合起来；理论上证明这一规则在线性标量化下精确恢复目标分布，对非线性算子则在高密度区可控失真。

## 方法详解

### 整体框架
给定 $k$ 个已经分别训练好的单目标 GFlowNet（第 $i$ 个针对奖励 $R_i(x)$，有前向策略 $p_{i,F}$、到达概率 $u_i$、终止分布 $p_i\propto R_i$），本文要解决的是：怎样在推理阶段就拼出任意目标组合 $p_M^*(x)\propto\mathcal{G}(p_1,\dots,p_k)$ 的采样器，而完全不碰任何训练。做法是把"组合"从终止分布层面下沉到 DAG 的每一步——在每个非汇状态上现场算出一个混合前向策略 $p_{M,F}$，按它逐步采样到终止态，得到的边缘分布就（精确或近似）等于目标 $p_M^*$。

### 关键设计

**1. 基于到达概率的混合策略：让每个模型在自己熟悉的状态上说话更响**

如果只是把 $k$ 个前向策略均匀平均，就退化成了对策略本身做集成，完全忽略了"在当前状态上哪个模型更靠谱"。本文的核心算子是用每个模型的到达概率当局部权重去混合前向转移：对状态 $s$ 的每个孩子 $s'$，

$$p_{M,F}(s'\mid s)=\frac{\mathcal{G}\bigl(u_1(s)\,p_{1,F}(s'\mid s),\dots,u_k(s)\,p_{k,F}(s'\mid s)\bigr)}{N_M(s)},$$

其中 $N_M(s)$ 是把该状态所有孩子上述量加起来的归一化常数。这里 $u_i(s)$ 衡量"模型 $i$ 把多少概率质量路由到了 $s$"，送得越多说明它越熟悉这块状态空间，就该在 $s$ 上获得越高的发言权。把每个模型的贡献按它在当前轨迹上的"在场感"加权，正是后面理论能恰好闭合的关键——朴素均匀混合（$u_i$ 全设为 $1$）那条路在实验里会退化成集成基线，$L_1$ 误差暴涨一两个数量级。

**2. 可计算的到达概率读取：覆盖两大类训练范式**

混合策略要能在推理时即查即用，就得高效拿到 $u_i(s)$。直接按递推 $u(s)=\sum_{s_*}u(s_*)p_F(s\mid s_*)$ 求解需要枚举所有父状态，不可行。作者改用恒等式 $u_i(s)=F_i(s)/Z_i$（$F_i$ 是状态流，$Z_i=F_i(s_0)$ 是总流），从而把问题分成两条互补的路线：若预训练目标本身显式参数化了流（flow matching / detailed balance / sub-trajectory balance），就直接读取，称为 "Model $F$" 路线；若用的是 trajectory balance 这类不显式建模 $F$ 的目标，就借 detailed balance 条件 $F_i(s')p_{i,B}(s\mid s')=F_i(s)p_{i,F}(s'\mid s)$，沿当前轨迹累乘 $\prod_j p_{i,F}/p_{i,B}$ 在线恢复 $F_i$，称为 "DB $F$" 路线。两条路线一起覆盖了 GFlowNet 训练目标的两大类，使框架不挑积木的训练范式，最大化了现成模型的复用面。

**3. 失真因子分析：同一个量说清何时精确、何时近似**

要解释混合策略到底恢复出什么分布，把它走完一条完整轨迹的边缘分布展开成 $p_M(x)=\delta(x)\cdot\mathcal{G}(p_1(x),\dots,p_k(x))$，其中失真因子 $\delta(x)=u_M(x)/N_M(x)$ 直接刻画与目标分布的偏差。对线性标量化 $\mathcal{G}=\sum_i\omega_i Z_i p_i$，文章证明 $\delta(x)$ 在所有 $x$ 上恒为常数 $1/Z_M$，于是 $p_M$ 精确等于 $p_M^*\propto\sum_i\omega_i R_i$（命题 4.1）；对调和均值、对比、带温度 $\beta\neq 1$ 的标量化这些非线性算子，$\delta(x)$ 不再是常数，但实验显示它在 $\mathcal{G}$ 取值较大的高密度区仍接近 $1/Z_M$，$L_1$ 误差主要由低密度区贡献，整体可控。把"何时精确、何时近似"全部收进同一个数学量 $\delta(x)$，使框架既保留了线性情形的可证明保证，又给出了非线性情形可解释的退化模式，而不是把非线性当黑箱。

### 损失函数 / 训练策略
完全无训练。所有计算都发生在推理时：从 $s_0$ 出发逐步采样，每一步现场读取或在线累乘得到 $u_i$、按上面的加权公式构造混合分布、归一化后采样下一个状态，直到终止。单步推理开销仅相当于对 $k$ 个模型各做一次前向传播再做一次组合，远低于训练任何一个新模型的代价。

## 实验关键数据

### 主实验

在 $32\times 32$ 的 2D 网格上（真实分布可解析计算，便于直接评 $L_1$）和分子生成上（fragment-based + atom-based QM9）评估。

| 任务 / 设置 | 指标 | 本文 | 之前 SOTA | 提升 |
|--------|------|------|----------|------|
| 2D 网格 5 目标线性标量化 | $L_1$ ↓ | 0.003 | HN-GFN 0.035 / MOGFN 0.048 | 约 10–16 倍 |
| 2D 网格 多目标可扩展性 | $L_1$（2→5 目标）| 0.003→0.003 几乎不变 | MOGFN 0.021→0.048 显著退化 | 线性扩展性 |
| 2D 网格 调和均值 $p_{\text{Circle1}}\otimes p_{\text{Circle2}}$ | $L_1$ ↓ | 0.229 | Classifier 0.397 | -0.168 |
| Fragment SEH-QED 标量化 | Top-10 平均奖励 ↑ | 0.777 (DB $F$) | MOGFN 0.764 | +0.013 且无需重训 |
| Fragment 三目标 ALL | Top-10 奖励 ↑ | 0.742 (DB $F$) | MOGFN 0.723 | +0.019 |
| Fragment SEH⊗SA⊗QED 调和均值 | 三高目标命中率 (%) | 65–66 | Classifier 40 | +25 pt |
| QM9 GAP-SA 标量化 | Top-10 奖励 ↑ | 0.873 | MOGFN 0.799 | +0.074 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整方法 (reaching-weighted mixing) | 2D 网格 5 目标 $L_1$ = 0.003 | 含到达概率加权 |
| Ensemble (去掉 $u_i$ 改为均匀混合) | $L_1$ ≈ 0.10–0.12 | 验证到达概率是关键，不是简单的策略平均 |
| Model $F$ 路线 vs DB $F$ 路线 | Fragment ALL 奖励 0.741 vs 0.742 | 两种读取方式效果几乎相同，框架不挑训练目标 |
| 失真因子 $\delta(x)$ 分布检查 | 线性算子下恒等于 $1/Z_M$；逻辑算子下高密度区接近常数 | 实证支持理论分析 |

### 关键发现
- 到达概率加权是必需的：移除后退化为集成基线，$L_1$ 误差暴涨一到两个数量级，说明"谁负责哪块状态空间"的局部信息比任何全局权重都重要。
- 在目标数从 2 增到 5 时，本文方法 $L_1$ 几乎不变，而偏好条件基线随目标数线性退化——这把"目标集合可扩展性"从"重新训练"问题转化为"零额外成本"问题。
- 在分子生成的三目标合取上（SEH⊗SA⊗QED 同时取高），本文从 40% 命中率拉到 65%，且分类器引导每换一组目标都要重训分类器，本文则完全免训。
- 推理速度上，逻辑算子组合显著快于分类器引导（后者还要前向分类器），匹配作者宣称的"最小推理开销"。

## 亮点与洞察
- 把"到达概率 $u_i(s)$"提取出来作为混合权重，把 GFlowNet 内部的轨迹结构当作可组合资源，这是一个非常巧妙的视角切换——以往组合都在终止分布层面做，本文把组合下沉到 DAG 上每一步的前向策略。
- 用同一个公式 $p_{M,F}\propto \mathcal{G}(u_i p_{i,F})$ 同时覆盖线性标量化、温度缩放、调和均值、对比算子，提供了一种 GFlowNet 组合的"代数式"语言，相当于扩散模型里 score 组合的离散对应物。
- 失真因子 $\delta(x)$ 的引入把"何时精确"问题量化为"$\delta$ 是否随 $x$ 变化"，并通过实证在高密度区近似常数来解释逼近精度，这是一种把理论与实证有机串起来的写法，可以迁移到任何组合分布的近似分析。

## 局限与展望
- 失真因子只在高密度区接近常数，低密度尾部仍有不可忽略的偏差；如果下游任务对尾部稀有结构敏感（如安全性筛选、罕见副反应预测），本文的近似可能引入系统性盲区。
- 整个方法依赖"每个奖励都有一个高质量预训练 GFlowNet"，但实践中新奖励往往本身就是要快速试验的对象，预训练成本并未消失，只是被前置到了"积木库构建"阶段。
- DB $F$ 路线在线累乘 $\prod p_F/p_B$ 在长轨迹上可能出现数值不稳定（极小概率连乘），论文没有充分讨论数值精度退化的临界点。
- 当前框架是"非自适应"的——给定 $\mathcal{G}$ 就有固定的混合规则，没有让混合权重根据采样目标自适应学习的机制；若允许少量微调，可能进一步压缩失真因子方差。

## 相关工作与启发
- **vs MOGFN / HN-GFN [Jain 2023; Zhu 2024]**: 他们用偏好条件 GFlowNet 在训练时就把所有偏好打包；本文完全免训练，更适合"先训单目标库再随用随组合"的工作流，但前提是单目标库已经覆盖了想要的奖励。
- **vs Compositional Sculpting [Garipov 2024]**: 分类器引导处理逻辑算子时每换组合都要重训分类器；本文用同一套混合策略覆盖逻辑算子，质量相当、速度更快、训练成本归零。
- **vs Products of Experts / 扩散模型组合 [Hinton 2002; Liu 2022; Du 2023]**: 连续场景下的 score 组合早有成熟工具，本文相当于在离散 DAG 上提供了等价物，并指出到达概率是离散域里"局部权重"的天然候选——这一思路也可能反哺扩散模型中关于状态依赖混合系数的研究。

## 评分
- 新颖性: ⭐⭐⭐⭐ "用到达概率混合前向策略"是一个干净的新角度，统一了之前两条路线。
- 实验充分度: ⭐⭐⭐⭐ 同时覆盖合成网格（可解析 $L_1$）和真实分子任务，包含 SEH/SA/QED/GAP 多组合 + 调和均值/对比/标量化全谱算子。
- 写作质量: ⭐⭐⭐⭐ 理论与实证对应清晰，失真因子的引入让分析非常可读。
- 价值: ⭐⭐⭐⭐ 在 GFlowNet 多目标这块属于"立刻能用"的工程友好型工作，对药物/分子领域工作流的提速有直接意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](../../ICML2025/computational_biology/graph_generative_pre-trained_transformer.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](../../ICML2025/computational_biology/peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[ICML 2025\] eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis](../../ICML2025/computational_biology/eccdnamamba_a_pre-trained_model_for_ultra-long_eccdna_sequence_analysis.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)

</div>

<!-- RELATED:END -->
