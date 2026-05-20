---
title: >-
  [论文解读] The Cylindrical Representation Hypothesis for Language Model Steering
description: >-
  [ICML 2026][可解释性][Activation Steering] 本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何…
tags:
  - "ICML 2026"
  - "可解释性"
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
CRH 的整体逻辑：(1) 从核心假设"差向量 $\mathbf{v}_d=\mathbf{r}_a-\mathbf{r}_b=\sum_i\alpha^{(i)}\mathbf{a}^{(i)}$"出发；(2) 把每个概念分解为轴向 + 法向两部分；(3) 由线性叠加约束推出"所有概念的法向分量之和为零"这一关键平衡；(4) 在法平面上定义敏感扇区，把 steering 效果归结为"$\mathbf{v}$ 在法平面上的相位是否落入敏感扇区"；(5) 给出 3 个可观测的实验推论并在 Gemma-2B / LLaMA2-7B 上系统验证。

### 关键设计

1. **轴-法分解 + 法向平衡定理**:

    - 功能：把任意差向量 $\mathbf{v}_d$ 自然诱导出一个中心轴 $\mathbf{a}_d=\mathbf{v}_d/\|\mathbf{v}_d\|$ 和一组互相抵消的法向分量，把"概念叠加"几何化成圆柱结构。
    - 核心思路：对每个概念方向 $\mathbf{a}^{(i)}$ 做标准投影 $\mathbf{v}^{(i)}=d^{(i)}\mathbf{a}_d+\mathbf{v}_{\perp}^{(i)}$，代入 $\mathbf{v}_d=\sum_i\mathbf{v}^{(i)}$，可推出 $(\sum_i d^{(i)})=\|\mathbf{v}_d\|$ 且 $\sum_i\mathbf{v}_{\perp}^{(i)}=\mathbf{0}$。也就是说，差向量同时定义了一个一维的"主轴"和一个所有非轴贡献相互抵消的"法向平衡态"。再用 PCA 取出二维法平面 $\mathcal{P}_d=\text{span}(\mathbf{a}_{\perp}^{(c)},\text{PC}_1(\{\mathbf{a}_{\perp}^{(i)}\}_{i\neq c}))$，平衡关系在法平面上仍然成立。
    - 设计动机：这个分解是把"LRH 假设的全局单方向"替换为"样本特定的轴 + 局部法平面"的核心步骤；它告诉你 steering 不是沿单一方向滑行，而是沿圆柱面"穿过"或"绕过"概念。

2. **敏感扇区与 steering 分解**:

    - 功能：把法平面分成"高敏感扇区"（steering 会被加速到目标概念）和"低敏感扇区"（被压制或拖延），用一个简单的β系数对比给出充分条件。
    - 核心思路：把 steering 向量分解为 $\mathbf{v}=\mathbf{v}_{\text{axis}}+\mathbf{v}_{\perp,\mathcal{P}_d}+\boldsymbol{\epsilon}$，其中法平面分量进一步写成 $\mathbf{v}_{\perp,\mathcal{P}_d}=\beta_c\mathbf{v}_{\perp,\mathcal{P}_d}^{(c)}+\sum_{i\neq c}\beta_i\mathbf{v}_{\perp,\mathcal{P}_d}^{(i)}$。当目标概念贡献 $\beta_c>\sum_{i\neq c}\beta_i$ 时落入高敏感扇区（强化轴向驱动，快速激活目标概念）；反之落入低敏感扇区（被竞争概念主导，激活被延后甚至抑制）。
    - 设计动机：用一个最朴素的"谁的贡献大"作为判据，避免引入额外参数；同时这个判据天然解释了为什么"角度相似的 steering 向量"会在不同样本上产生完全相反的效果 —— 它们投影到同样形状但定位不同的法平面后会落入不同扇区。

3. **可预测性二分定理 + 三条可观测推论**:

    - 功能：形式化指出"法平面的幅值可由 $\mathbf{v}_d$ 可靠预测，但敏感扇区不可"，并把这条几何性质翻译成 3 条可实验测的趋势。
    - 核心思路：定理 4.1（幅值可预测）说 $\|\mathbf{v}_{\perp,\mathcal{P}_d}\|$ 是 steering 强度的可靠 proxy；引理 4.2 + 定理 4.3（扇区不可预测）说在 $d$ 维空间放 > $d$ 个概念方向时，差向量到概念强度的映射是非单射的，因此从 $\mathbf{v}_d$ 反推不出敏感扇区。三条推论分别给实验入口：(i) 抑制法向分量 $\rho\mathbf{v}_{\perp}\to0$ 会同时延后概念激活与延后输出崩坏（trade-off）；(ii) 把 $\text{St}_c(\mathbf{r};\mathbf{v})/\|\mathbf{v}_d\|^k$ 拟合到 $\sin^m\theta\cos^{k-m}\theta$ 时应该出现单峰，说明法平面可由轴决定；(iii) 若扇区可由轴决定，那么 $\mathbf{v}_d$ 相似的样本 steering 效果应类似 —— 实验若反向说明扇区不可决。
    - 设计动机：作者刻意做"可预测 vs 不可预测"的对偶论断，把 CRH 跟单纯的"概念几何描述"区分开，转化为可证伪的几何理论；这一节是论文最有理论分量的部分。

### 损失函数 / 训练策略
本文不训练模型，所有"steering vector"都是从 contrastive 对（正/负样本）上用 DiffMean、PCA、Mean-Centering、probe-based 等标准方法构造的；探测实验用 one-shot 优化（Dunefsky & Cohan 2025）：冻结模型，优化一个可训练向量去最大化目标句概率、抑制原句概率，跑 30 步、lr=0.1。

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

- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[ICML 2026\] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](../../NeurIPS2025/interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)

</div>

<!-- RELATED:END -->
