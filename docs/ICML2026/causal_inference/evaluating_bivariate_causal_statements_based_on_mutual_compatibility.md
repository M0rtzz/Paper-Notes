---
title: >-
  [论文解读] Evaluating Bivariate Causal Statements Based on Mutual Compatibility
description: >-
  [ICML 2026][因果推理][二元因果陈述] 本文针对"只有成对(bivariate)因果陈述、没有 ground truth"的场景，提出两个无需 faithfulness 的相容性评分（线性情形的 `comp` + 图结构情形的 `incomp`）…
tags:
  - "ICML 2026"
  - "因果推理"
  - "二元因果陈述"
  - "相容性评分"
  - "混淆后门"
  - "LLM 因果评估"
---

# Evaluating Bivariate Causal Statements Based on Mutual Compatibility

**会议**: ICML 2026  
**arXiv**: [2606.00278](https://arxiv.org/abs/2606.00278)  
**代码**: https://github.com/ejahn17/compatibility-scores  
**领域**: 因果推断 / 因果发现评估 / LLM 因果推理  
**关键词**: 二元因果陈述、相容性评分、混淆后门、LLM 因果评估  

## 一句话总结
本文针对"只有成对(bivariate)因果陈述、没有 ground truth"的场景，提出两个无需 faithfulness 的相容性评分（线性情形的 `comp` + 图结构情形的 `incomp`），通过判断这些两两陈述拼起来的多元模型是否需要"反常的额外混淆"来解释观测协方差，从而识别错误的因果论断，并用它给 LLM 的因果输出打分。

## 研究背景与动机
**领域现状**：医学、经济等领域的因果推断长期依赖专家给出的成对 cause-effect 论断；最近大量工作转而让 LLM 直接产出此类成对因果关系。两条路径都缺乏"出错就能被发现"的机制——传统因果发现算法在强假设下有一致性保证，而 LLM 既没有训练目标也没有理论约束。

**现有痛点**：在缺乏 ground truth 的真实场景下，几乎无法验证一份"$X_i \to X_j, \alpha_{ij}$"清单到底对不对。已有的"无 GT 评估"工作（Textor 2016、Eulig 2025、Sheth 2026）大多衡量"估计图与数据的一致性"，但只针对完整 DAG 估计；Faller 2024 用"多个子集图能否拼成同一大图"做相容性，但要求输入是子图，不直接适用于"逐对论断"这种最常见的人类/LLM 输出形态。

**核心矛盾**：对于线性、无环的成对论断 $\{\alpha_{ij}\}$，作者首先证明（Lemma 2.3）：任意一份这样的清单都**唯一**地诱导一个能完美拟合观测协方差的多元线性 SEM，所以在 Faller 那套"硬相容"意义下根本不会冲突——没有可利用的约束。要做评估，必须从"软"角度（plausibility）切入。

**本文目标**：(1) 为线性成对论断设计一个不依赖 faithfulness 的连续相容性分数；(2) 为只指明"有无因果/有无混淆"的图结构论断设计一个不相容计数；(3) 在合成数据和 LLM 输出上验证两者能区分对错。

**切入角度**：作者提出一个**混淆后置假设（Confounding Postulate）**——一个"通用"的多元因果模型不应该比它的逐对边缘模型有**更多**未观测混淆。直观上，把多元模型边缘化到两两后，本来被观测的后门路径（如 $X_i \leftarrow X_k \to X_j$）会变成不可观测的混淆，因此边缘模型的混淆量只会增大；除非诱导出的多元模型刻意把额外的观测因果路径"调到正好抵消"自身混淆，这种 fine-tuning 在随机模型下极少发生（Theorem 2.9 给出期望意义下的保证）。

**核心 idea**：用"诱导多元 SEM 的混淆量"与"逐对边缘 SEM 的混淆量"之差作为相容性分数；若多元模型反而"需要更少"的混淆来解释协方差，说明这份论断要求一种反直觉的精细抵消，可判为不可信。

## 方法详解

### 整体框架
输入：$n$ 个变量的观测协方差矩阵 $\Sigma$（或样本估计 $\hat{\Sigma}$）+ 一份 $\binom{n}{2}$ 条成对因果论断。

- **线性情形**：每条论断给出方向 $i\to j$ 和系数 $\alpha_{ij}$。把它们装进单位下三角矩阵 $A$（$A_{ji}=\alpha_{ij}$, $i<j$），令 $\Gamma = I - A^{-1}$，得到唯一的多元 SEM $X=\Gamma X + N$（Lemma 2.3）。然后比较"多元 vs 逐对"两种粒度下的"未被观测因果路径解释的协方差平方"，求和得 `comp` 分数。
- **图结构情形**：每条论断给出一张两节点的 ADMG（有/无 $\to$ 边、有/无 $\leftrightarrow$ 双向边）。把所有两节点 ADMG 合并成 statement graph $G$，定义 `incomp(G)` 为把 $G$ 改成"可拼接成大 ADMG"所需的最少边增删次数（Hamming 距离）。

输出：`comp` 分数（线性）或 `incomp` 分数（图结构）。`comp` 为负 → 违反 confounding postulate → falsify 该论断清单；`incomp` 为正 → 必有论断错误，且数值反映错误数量级。

### 关键设计

1. **线性 comp 分数 = 边缘混淆 − 多元混淆**:

    - 功能：用一个标量度量"这份成对论断诱导出的多元模型有多反常"。
    - 核心思路：对每对 $i<j$ 定义两种混淆量。逐对 SEM $X_j = \alpha_{ij}X_i + \tilde N_{ij}$ 中，未被因果效应解释的协方差平方为 $C^{biv}_{ij} = (\Sigma_{ij} - \alpha_{ij}\Sigma_{ii})^2$。多元 SEM 中，按 Wright path tracing 减去所有"已观测有向路径 + 完整后门路径"的贡献，得到 $C^{mult}_{ij}(\Sigma,\Gamma) = (\Sigma_{ij} - \sum_{k\le i}\Sigma_{kk}\sum_{P_1: k\rightsquigarrow i, P_2: k\rightsquigarrow j} \Gamma_{P_1}\Gamma_{P_2})^2$。最终 $\text{comp}(\Sigma,A) = \sum_{i<j} C^{biv}_{ij}(\Sigma,A_{ji}) - C^{mult}_{ij}(\Sigma, I-A^{-1})$。直觉：把多元模型边缘化到两两后，原本"观测的后门"会被吞进噪声里成为混淆，所以 $C^{biv} \ge C^{mult}$ 是"plausible"模型应有的性质；`comp<0` 等价于违反 Assumption 2.4（confounding postulate）。
    - 设计动机：作者证明 Lemma 2.3 之后发现"硬相容"无法用，于是寻找一个不依赖 faithfulness、却能在随机模型下"绝大多数情况成立"的软约束。`comp` 的设计同时满足：(a) 在两变量退化情形等于经典 bivariate 混淆度量（Janzing-Schölkopf）；(b) Theorem 2.9 证明在"系数 0 均值 + 机制独立 + 非退化"的随机 SEM 下，真值论断的期望 `comp > 0`；(c) Theorem 2.10 给出 $N = O(n^4(1+a+b)^4 V^4 / \varepsilon^2 \cdot \log(n/\delta))$ 样本即可把 $|\text{comp}(\hat\Sigma,A) - \text{comp}(\Sigma,A)|$ 控制到 $\varepsilon$ 内，实验中 100 样本就够 90% 符号正确。

2. **图结构 incomp 分数 = 到"可拼接"图的最小编辑距离**:

    - 功能：当论断只是定性的（有无因果、有无混淆）时，统计违反全局一致性的次数。
    - 核心思路：所有 $\binom{n}{2}$ 个两节点 ADMG 取并集得到 statement graph $G$。Lemma 3.5 给出"$G$ 能被某个大 ADMG marginal 出来"的充要条件：(i) 有向部分无环；(ii) 有向部分**传递闭包**——若 $X_i \rightsquigarrow X_j$ 在 $G$ 中有路径，则必须有直接边 $X_i \to X_j$（因为边缘化保留可达性）；(iii) 任意两点间若存在"双 arrowhead 端点 + 中间无两 arrowhead"的混淆路径，则必须有 $X_i \leftrightarrow X_j$ 双向边（混淆路径在边缘化后必然变为双向边）。定义 $\text{incomp}(G) = \min_{G^*} d(G, G^*)$，其中 $d$ 是 mixed graph Hamming 距离，$G^*$ 遍历所有满足 (i)-(iii) 的图。
    - 设计动机：扩展 Faller 2024 的硬相容到"违反度量"——硬相容只回答"yes/no"，无法对"几乎对，只错一条"和"全错"做区分；用 Hamming 距离量化后，分数大小直接反映错误论断的下界数量，且天然继承 faithfulness + 无环假设带来的全局约束。

3. **finite-sample 估计与 LLM 评测协议**:

    - 功能：把 `comp` 落到真实数据上，并定义一套"评 LLM 因果推理能力"的工作流。
    - 核心思路：实际拿到的是样本协方差 $\hat\Sigma = \frac{1}{N}\sum_r X^{(r)} X^{(r)\top}$；评分前先把变量标准化到单位方差，等价于把 $\Sigma$ 替换成相关系数矩阵；论断系数 $A$ 也按同一缩放变换。LLM 评测时（实验用 gapminder 7 个国家级指标），把变量描述 + 经验相关阵作 prompt 喂给不同 LLM，要求其输出因果排序 + 两两线性系数，对 15 次独立运行取平均，再与"系数从 0 均值高斯采样、方差匹配 LLM 输出"的随机 baseline 对比。
    - 设计动机：让方法不仅停留在合成实验，而是给出一个**可复现的 LLM 因果能力排行**——这正是论文实用价值的落脚点：在没有 ground truth 的现实题（gapminder）上，能力强的 LLM 倾向于更高 `comp`，许多 LLM 仍得到负分（即被本方法 falsify），证明评分确实能挑出错误输出。

### 损失函数 / 训练策略
本方法**纯评估、无训练**。`comp` 是协方差矩阵和系数矩阵的闭式函数；`incomp` 是组合优化问题（在 $n$ 较小时可枚举/启发式求解）。

## 实验关键数据

### 主实验

**合成数据上 `comp` 区分对错的能力**（Figure 2）：从随机线性 Gaussian SEM 采样真值，对真值系数加方差递增的高斯噪声 $\sigma$ 模拟"质量递减的论断清单"，每点 50 模型 × 20 噪声平均。

| 设置 | $\sigma=0$（真值）正分比例 | $\sigma$ 增大趋势 | 关键观察 |
|------|--------|--------|------|
| $n=10, p=0.5$, 变 $m$（隐变量数 0/1/3/5） | ≈ 95%+ | 单调下降 | 对隐变量数鲁棒，曲线几乎重合 |
| $n=10, m=3$, 变 $p$（稀疏度 0.3/0.5/0.7/0.9） | ≈ 95%+ | 单调下降 | $p$ 越大（图越稠）区分度越强 |
| $p=0.5, m=3$, 变 $n$（变量数 5/10/15/20） | ≈ 95%+ | 单调下降 | $n$ 越大区分度越强 |

**LLM 因果论断评测**（Figure 4，gapminder 7 变量）：

| 模型组 | `comp` 分数（15 次平均，相对随机 baseline） | 结论 |
|------|------|------|
| 高能力 LLM | 显著高于 0、且高于随机 baseline | 通过 falsification |
| 中等 LLM | 接近随机 baseline | 不能判定 |
| 低能力 LLM | **负分** | 被 `comp` falsify，证据表明因果论断不可信 |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| `comp` 用真 $\Sigma$ vs $\hat\Sigma$（Figure 3, $N$ 变化） | $N \ge 100$ 时符号一致率 > 90%；$N$ 增大相对误差快速降到 < 5% | 经验收敛远快于 Theorem 2.10 的最坏界 |
| 隐变量数 $m \in \{0,1,3,5\}$ | 正分比例曲线几乎重合 | 方法对未观测混淆鲁棒（这正是它存在的意义） |
| 稀疏度 $p$ 从 0.3 升到 0.9 | 真值的正分比例几乎不变；错值的正分比例随 $p$ 大幅下降 | 稠密图 → 后门路径更多 → 边缘混淆 vs 多元混淆差异更大 → 更容易抓住错论断 |
| 随机 baseline LLM | 略高于 0 | 即使随机猜也能蒙对"非常错的模型应该是负的" |

### 关键发现
- **核心结论**：`comp` 在合成数据上几乎不漏报真值（≥ 95% 正分）、能稳定区分加噪后的错值；样本复杂度在实际中远好于理论界。
- **稠密图更有利**：$n$ 大或 $p$ 大时，多元 SEM 中的观测后门路径变多，边缘化时丢掉的信息也变多，`comp` 区分力随之上升——这与方法的工作机理（postulate 来自"边缘化吃掉后门"）完全自洽。
- **LLM 应用层洞察**：高能力 LLM 在 gapminder 上拿到更高 `comp`，多个低能力 LLM 拿到**负分**且差于随机 baseline，说明本方法可以直接当作"无 GT 场景下的 LLM 因果能力体检"。

## 亮点与洞察
- **绕开 faithfulness 是真正的卖点**：传统因果发现一致性保证都建在 faithfulness 上，而本文 Theorem 2.9 在"系数 0 均值 + 机制独立 + 非退化"这一更弱的随机性假设下，就能保证真值论断期望 `comp > 0`——对那些怀疑 faithfulness 是否成立的实际场景（医学、社会经济）非常有吸引力。
- **"边缘化必然增加混淆"作为先验**：用这一物理直觉来代替 faithfulness，把"诱导多元模型必须 fine-tune 抵消"作为反例，是把因果可识别性问题转译成"参数空间稀有事件"问题的优雅做法，可以迁移到任何"局部估计拼全局模型"的评估任务（如分布式联邦因果发现、跨模态对齐验证）。
- **评估 LLM 因果输出的可操作模板**：很多 LLM-causal 工作只能在有 GT 的人造数据集上比较，而本文给出了一条"喂相关阵 + 取系数 + 算 comp"的零标注流程——对 LLM 评测社区是一个具体可用的 metric，特别是评测科学发现/医疗诊断类任务时。

## 局限与展望
- **作者承认**：(1) 正分仍可能错（即论断错却恰好诱导出"plausible"模型），所以 `comp` 只能**falsify**、不能**verify**；(2) 线性 + 无环 + Gaussian 假设较强；(3) `comp` 数值依赖变量缩放，必须先标准化。
- **自己看到的局限**：(a) Theorem 2.9 假设系数 0 均值，这在很多领域不成立（如医学中"治疗→好转"系数通常正），评估时若真实分布有偏，期望保证就弱；(b) 实验只到 $n=20$，对真实数据中的 $n=100+$ 大图，path tracing 求和的计算量是否可控未论证；(c) `incomp` 的最小化是组合问题，论文没具体给出 $n$ 大时的近似算法；(d) LLM 实验只用 gapminder 7 变量，对更复杂的医学/生物因果系统效果未知。
- **可改进方向**：把 `comp` 推广到非线性 SEM（用条件均值 / kernel 协方差替代线性余项）；引入"分数衰减"使其能区分"轻微错误"和"严重错误"，而不仅看符号；与已有 LLM 因果方法（agent 自洽、多次采样投票）结合，把 `comp` 作为 reward signal 反向训练或筛选 LLM 输出。

## 相关工作与启发
- **vs Faller et al. (2024)**：他们要求输入是"多张子集 ADMG"并做硬相容检查；本文输入只需"$\binom{n}{2}$ 条两两论断"且给出**软**分数，覆盖了人类专家/LLM 的最自然输出形态，并把硬相容（Lemma 3.5）拓展成"违反次数"的连续度量。
- **vs Janzing & Schölkopf (2018) 的 bivariate 混淆度量**：他们只在两变量情形定义未解释协方差平方；本文用 Wright path tracing 把同样的思想推广到多变量，并把"两变量 vs 多变量"的差作为评估信号——本质上把原来 standalone 的指标变成"自我对照"的指标。
- **vs Textor 2016 / Eulig 2025 等"无 GT 因果评估"**：他们都基于"估计的完整图与数据的一致性"，要求输入是完整 DAG；本文针对"只有 pairwise"这一更稀疏、更现实的输入形态，是该子方向上的首批工作之一。
- **vs LLM-causal 评估工作（Kiciman 2024, Sheth 2025）**：那些工作多数在带 GT 的合成基准上比较 LLM 准确率；本文给出**无 GT** 真实数据上排名 LLM 的方法，更接近实际部署条件。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用"边缘化必然增加混淆"作为软相容性证据、绕开 faithfulness 来评估 pairwise 因果，思路新且有理论 backing。
- 实验充分度: ⭐⭐⭐⭐ 合成实验在多个参数维度做了消融，LLM 实验有 baseline，但只在 7 变量 gapminder 上，多领域验证欠缺。
- 写作质量: ⭐⭐⭐⭐ 定义—引理—定理—实验链条清晰，Figure 1 的反例直观；附录占大头但主文自洽。
- 价值: ⭐⭐⭐⭐ 在"评估 LLM 输出因果声明"这一上升期问题上提供了第一批可落地、零标注的工具，对因果推断 + LLM 双方向社区都有用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evaluating Counterfactual Strategic Reasoning in Large Language Models](../../ACL2026/causal_inference/evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)
- [\[ICML 2026\] Formalizing and Falsifying Causal Pathways of Rare Events](formalizing_and_falsifying_causal_pathways_of_rare_events.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)

</div>

<!-- RELATED:END -->
