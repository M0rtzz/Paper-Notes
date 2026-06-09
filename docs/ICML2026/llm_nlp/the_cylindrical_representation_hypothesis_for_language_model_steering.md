---
title: >-
  [论文解读] The Cylindrical Representation Hypothesis for Language Model Steering
description: >-
  [ICML 2026][LLM/NLP][Activation Steering] 本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Activation Steering"
  - "Linear Representation Hypothesis"
  - "表征几何"
  - "可控性"
  - "概念向量"
---

# The Cylindrical Representation Hypothesis for Language Model Steering

**会议**: ICML 2026  
**arXiv**: [2605.01844](https://arxiv.org/abs/2605.01844)  
**代码**: https://github.com/mbzuai-nlp/CRH  
**领域**: LLM/NLP / 表示几何 / 可解释性  
**关键词**: Activation Steering、Linear Representation Hypothesis、表征几何、可控性、概念向量

## 一句话总结
本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何，从而首次几何化地解释了 activation steering 为什么在样本层面不可预测但在群体层面可观测。

## 研究背景与动机

**领域现状**：LLM 的"激活转向"（activation steering）已经成为可解释性与对齐研究的主流工具：在某层残差流加一个概念方向向量 $\mathbf{v}$，就能在推理时按概念促进或抑制输出。现有理论几乎都基于 Linear Representation Hypothesis（LRH，Park et al. 2024）—— 概念对应线性方向，且通过"因果内积"可正交化、独立操控。

**现有痛点**：实际 steering 极不稳定，同一个方向在不同样本上效果差异巨大。基于 LRH 的"可控性预测"（如表征分离度）和实测 steering 成功率相关性很弱，工程上几乎不可信。

**核心矛盾**：LRH 假设的"无损正交化"在有限维 + 概念数量 > 维度时根本不可能成立 —— 任何二维空间最多放 2 个正交方向，但 LLM 内部要表达成千上万的概念。所以概念之间一定会有重叠，正交化的理论假设从根上就站不住。

**本文目标**：(i) 把 LRH 放松成"线性叠加但允许非正交"，看看几何结构会变成什么；(ii) 用这个新结构解释 steering 为何在样本层面随机、却在群体层面可估；(iii) 给出可实证检验的预测。

**切入角度**：作者保留"概念向量是线性的"这一软核心，但允许它们之间任意夹角；然后从最朴素的"差向量 = 多概念线性组合"出发，推导出局部几何不可避免地呈现"中心轴 + 法平面 + 相位"三件套。

**核心 idea**：把样本-概念的局部几何建模成圆柱：差向量定义中心轴，所有概念在法平面上的投影互相抵消，steering 向量在法平面里的相位（phase）才是决定 steering 成败的关键 —— 而相位无法从轴和 $\mathbf{v}$ 推出，因此 steering 内禀地具有不可预测性。

## 方法详解

### 整体框架
CRH 不训练任何模型，而是从一个最朴素的假设出发做几何推导：把"概念促进 / 抑制"对应的差向量写成多个概念方向的线性叠加 $\mathbf{v}_d=\mathbf{r}_a-\mathbf{r}_b=\sum_i\alpha^{(i)}\mathbf{a}^{(i)}$。论文证明，只要保留这个线性叠加、放掉 LRH 的正交假设，局部几何就会被迫呈现"中心轴 + 法平面 + 敏感扇区"的圆柱结构：差向量定义轴，所有概念在法平面上的投影互相抵消，而 steering 向量落在法平面里的相位决定它到底是激活还是压制目标概念。沿着这条主线，方法依次给出轴-法分解、敏感扇区判据、可预测性二分定理，最后落到 3 条能在 Gemma-2B / LLaMA2-7B 上实测的推论。

### 关键设计

**1. 轴-法分解 + 法向平衡定理：把"概念叠加"几何化成一根圆柱**

LRH 假设概念沿一组全局正交方向独立排布，但在有限维里塞下成千上万个概念时这根本不成立。CRH 换一个角度：对任意差向量 $\mathbf{v}_d$，先定义中心轴 $\mathbf{a}_d=\mathbf{v}_d/\|\mathbf{v}_d\|$，再把每个概念方向 $\mathbf{a}^{(i)}$ 按这根轴做标准投影 $\mathbf{v}^{(i)}=d^{(i)}\mathbf{a}_d+\mathbf{v}_{\perp}^{(i)}$。把它代回 $\mathbf{v}_d=\sum_i\mathbf{v}^{(i)}$，立刻得到两条约束：轴向分量之和正好补满整根轴 $\sum_i d^{(i)}=\|\mathbf{v}_d\|$，而所有法向分量必须互相抵消 $\sum_i\mathbf{v}_{\perp}^{(i)}=\mathbf{0}$。也就是说，差向量同时定义了一个一维主轴和一个"非轴贡献彼此平衡"的法向态。再用 PCA 从这些法向分量里取出一张二维法平面 $\mathcal{P}_d=\text{span}(\mathbf{a}_{\perp}^{(c)},\text{PC}_1(\{\mathbf{a}_{\perp}^{(i)}\}_{i\neq c}))$，平衡关系在这张平面上依旧成立。这一步是整套理论的地基——它把"LRH 的全局单方向"换成"样本特定的轴 + 局部法平面"，于是 steering 不再是沿一条线滑行，而是沿圆柱面"穿过"或"绕过"概念。

**2. 敏感扇区与 steering 分解：为什么角度相似的向量效果天差地别**

有了圆柱，下一步要回答 steering 向量 $\mathbf{v}$ 加进去会发生什么。把它沿圆柱拆开：$\mathbf{v}=\mathbf{v}_{\text{axis}}+\mathbf{v}_{\perp,\mathcal{P}_d}+\boldsymbol{\epsilon}$，其中落在法平面里的那一部分再按概念展开成 $\mathbf{v}_{\perp,\mathcal{P}_d}=\beta_c\mathbf{v}_{\perp,\mathcal{P}_d}^{(c)}+\sum_{i\neq c}\beta_i\mathbf{v}_{\perp,\mathcal{P}_d}^{(i)}$。论文用一个最朴素的"谁贡献大"作判据：当目标概念的系数 $\beta_c>\sum_{i\neq c}\beta_i$ 时，向量落进**高敏感扇区**，强化轴向驱动、快速激活目标概念；反之落进**低敏感扇区**，被竞争概念主导，激活被延后甚至压制。这个充分条件不引入任何额外参数，却恰好解释了工程上最反直觉的现象：两个夹角几乎一样的 steering 向量，投影到形状相同但在圆柱上定位不同的法平面后，会落进不同扇区，于是在不同样本上产生完全相反的效果。

**3. 可预测性二分定理 + 三条可观测推论：把几何性质变成可证伪的实验**

CRH 最有分量的论断是一组对偶定理：法平面的"幅值"可预测，"扇区"不可预测。定理 4.1 说 $\|\mathbf{v}_{\perp,\mathcal{P}_d}\|$ 是 steering 强度的可靠 proxy，相当于圆柱外壳的形状由差向量唯一决定；引理 4.2 加定理 4.3 则指出，一旦在 $d$ 维空间里塞进多于 $d$ 个概念方向，差向量到概念强度的映射就不再是单射，因此从 $\mathbf{v}_d$ 根本反推不出敏感扇区的位置。这条"可预测 vs 不可预测"的对偶把 CRH 和单纯的几何描述区分开，让它成为可证伪的理论，并直接派生出三条实验入口：其一，抑制法向分量 $\rho\mathbf{v}_{\perp}\to0$ 会同时延后概念激活和输出崩坏，呈现一个 trade-off；其二，把 $\text{St}_c(\mathbf{r};\mathbf{v})/\|\mathbf{v}_d\|^k$ 拟合到 $\sin^m\theta\cos^{k-m}\theta$ 时应出现单峰，证明法平面可由轴决定；其三，若扇区也由轴决定，那么 $\mathbf{v}_d$ 相似的样本 steering 效果就该相似——一旦实验里它们不相似，就反证扇区不可由轴预测。

### 损失函数 / 训练策略
本文不训练模型，所有 steering vector 都从 contrastive 对（正 / 负样本）上用 DiffMean、PCA、Mean-Centering、probe-based 等标准方法构造。探测实验沿用 one-shot 优化（Dunefsky & Cohan 2025）：冻结模型，优化一个可训练向量去最大化目标句概率、抑制原句概率，跑 30 步、lr=0.1，用它把"输出空间反映射回表示空间"当作局部几何探针。

## 实验关键数据

### 主实验

| 模型 / 层 | 验证 | 关键结果 | 解读 |
|-----------|------|----------|------|
| Gemma-2B-IT layer 9 | 推论 1（trade-off） | $\rho$↓ 概念激活提前 + 输出崩坏提前 | 法向分量幅值确实双向调控 steering |
| Gemma-2B-IT layer 9 | 推论 2（轴决定法平面） | $\rho_k$ 曲线单峰 + 最低 p-value | 法平面可由 $\mathbf{v}_d$ 决定 |
| Gemma-2B-IT layer 9 | 推论 3（扇区不可决） | $\mathbf{v}_d$ cos 相似度 vs steering 差异 Pearson = -0.034 (p > 0.05) | $\mathbf{v}_d$ 相似不蕴含 steering 行为相似 |

实验同步在 LLaMA2-7B-Chat 层 16/24 验证，结论一致。

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| Full CRH | 推论 1/2/3 全部满足 | 圆柱结构成立 |
| 不同 steering 构造法（DiffMean / PCA / MC / probe） | 同样符合 CRH 预测 | 圆柱结构与 steering 方法选择无关 |
| 法向分量完全置零 $\rho=1$ | 输出最稳定但概念激活慢 | 验证轴向单独无法快速激活，需要法向辅助 |

### 关键发现
- 概念激活与输出崩坏是同一法向分量的两面：增加 $\|\mathbf{v}_{\perp}\|$ 能更快激活目标概念，但也更早把表示推离合理语义流形，工程上能解释"为什么 steering 总在阈值附近一刀切"。
- 推论 2 的单峰证实"轴 → 法平面"是可决的，等价于说圆柱的"外壳形状"由差向量唯一决定，但敏感扇区在外壳上的位置完全随样本而异。
- 推论 3 中"$\mathbf{v}_d$ 相似不蕴含 steering 行为相似"的零相关结果是 CRH 最强的反例武器 —— 它直接证明仅靠差向量预测 steering 成败注定失败，工程上能解释"为什么按相似度选概念向量的策略一直失败"。

## 亮点与洞察
- 放弃"正交性"而不放弃"线性"这一步走得非常聪明：保留了 LRH 的工程友好性（仍然可以做向量加减），又一次性消解了"为什么相似方向效果差很多"的工程谜团。
- 圆柱几何把 steering 失败从"工程噪声"重新定义为"内禀几何不确定性"，这意味着任何试图用"更好的概念向量构造法"消除 steering 波动的努力都注定有上限，社区策略应该转向"在敏感扇区里搜索"而非"找更纯净的方向"。
- 用 one-shot optimization 来探测圆柱结构是个聪明的实验设计：把"输出空间反映射回表示空间"的工具借过来当作"局部几何探针"，绕开了真概念方向不可观测的难题。

## 局限与展望
- CRH 把概念建模成有限多固定方向，但 LLM 的"概念"在不同层、不同上下文里可能本身就在漂移；本文未讨论概念方向 context-dependent 的情形。
- 敏感扇区的判据 $\beta_c$ vs $\sum_{i\neq c}\beta_i$ 是充分条件而非充要，实际边界更复杂；论文也未给出如何主动估计扇区的可行方法。
- 验证只覆盖了两个中等规模 LLM（2B / 7B），70B+ 上是否仍然成立、扇区结构是否更复杂是开放问题。
- 探测实验只在两层做（层 9/13 与 16/24），CRH 是否在所有层都成立、是否随深度变化值得后续研究。

## 相关工作与启发
- **vs Linear Representation Hypothesis (Park et al. 2024)**：CRH 是 LRH 的严格扩展，去掉了"正交可分"假设；LRH 是 CRH 在 $d\geq n$（维度足够大）时的退化情形。
- **vs Toy Models of Superposition (Elhage et al. 2022)**：superposition 强调"特征数 > 维度"时不可避免的干扰，CRH 给这种干扰提供了一个具体可几何刻画的局部模型（圆柱）。
- **vs AxBench / steering 基准评测**：CRH 解释了为什么 AxBench 等评测里 steering 成功率波动巨大，并给出"应在群体层面而非样本层面评估"的方法论建议。
- **vs 多维概念几何（Engels et al. 2025）**：他们指出"并非所有特征都是一维线性的"，CRH 也认同这一点但走的是另一条路 —— 不质疑线性，而是允许非正交，几何上更易实证。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "用圆柱几何取代正交假设"的视角原创性极强，且把样本特异性的随机性几何化地归因到敏感扇区，是社区急需的解释框架。
- 实验充分度: ⭐⭐⭐⭐ 探测实验设计精巧，三条推论分别有定量验证，多模型多构造法横向对照都做了。
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但符号密度大、几何图示也偏抽象，初次阅读上手不易。
- 价值: ⭐⭐⭐⭐⭐ 直接为整个 activation steering 子领域提供新理论坐标，对未来"可解释 + 可控"的工程改进有方向性意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](../../ICLR2026/llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](../../ACL2025/llm_nlp/repbend_representation_bending_safety.md)
- [\[ACL 2025\] SR-LLM: Rethinking the Structured Representation in Large Language Model](../../ACL2025/llm_nlp/sr-llm_rethinking_the_structured_representation_in_large_language_model.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)

</div>

<!-- RELATED:END -->
