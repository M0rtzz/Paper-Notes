---
title: >-
  [论文解读] Generative Social Choice: The Next Generation
description: >-
  [ICML2025][人体理解][生成式社会选择] 将生成式社会选择框架扩展至带成本/预算约束和近似查询的场景，提出 DemocraticProcess 算法并给出近乎最优的近似比例代表性理论保证，实现了实用系统 PROSE（基于 GPT-4o）在药物评论和城市治理数据集上验证有效性。
tags:
  - ICML2025
  - 人体理解
  - 生成式社会选择
  - 比例代表性
  - 参与式预算
  - LLM查询
  - 近似保证
---

# Generative Social Choice: The Next Generation

**会议**: ICML2025  
**arXiv**: [2505.22939](https://arxiv.org/abs/2505.22939)  
**代码**: [github.com/sara-fish/gen-soc-choice-next-gen](https://github.com/sara-fish/gen-soc-choice-next-gen)  
**领域**: 社会选择 / AI与民主  
**关键词**: 生成式社会选择, 比例代表性, 参与式预算, LLM查询, 近似保证

## 一句话总结
将生成式社会选择框架扩展至带成本/预算约束和近似查询的场景，提出 DemocraticProcess 算法并给出近乎最优的近似比例代表性理论保证，实现了实用系统 PROSE（基于 GPT-4o）在药物评论和城市治理数据集上验证有效性。

## 研究背景与动机
- **Polis 等集体响应系统**: 在线参与者提交观点并互相投票，系统选出能代表不同立场的陈述子集，已被台湾、澳大利亚等地用于国家级政策制定
- **现有局限**: Fish et al. [2024] 提出的生成式社会选择框架是里程碑工作，但存在两个关键假设过强：
  1. **固定 k 条等长陈述**: 无法控制陈述长度，小 k 可能产生过长摘要
  2. **精确查询**: 理论保证依赖判别查询（discriminative）和生成查询（generative）的精确响应，而 LLM 不可能完美回答
- **核心动机**: 在实际部署中，LLM 查询不可避免存在误差，且陈述有不同长度/成本，需要在总预算约束下做比例分配——这恰好对应从委员会选举到参与式预算的自然推广

## 方法详解

### 问题建模
- **陈述宇宙** $\mathcal{U}$（可无穷），成本函数 $c: \mathcal{U} \to \mathbb{N}_0$（如单词数）
- **$n$ 个 agent**，每个 agent $i$ 有效用函数 $u_i: \mathcal{U} \to [r]$（$r$ 级效用，如"强烈赞同"→"强烈反对"）
- **目标**: 在总预算 $B$ 约束下选出 slate $W \subseteq \mathcal{U}$，使其比例地代表所有 agent

### 查询模型
通过两类查询访问未知的 $\mathcal{U}$ 和效用函数：

| 查询类型 | 精确版本 | 近似参数 |
|---------|---------|---------|
| **判别查询** Disc$(i, \alpha)$ | 返回 $u_i(\alpha)$ | $\beta$-准确：误差 $\leq \beta$ |
| **生成查询** Gen$(S, \ell, x)$ | 返回成本 $\leq x$ 且获 $S$ 中最多 agent 在级别 $\ell$ 支持的陈述 | $(\gamma, \delta, \mu)$-准确 |

生成查询的三类误差：
- $\gamma$：支持者数量的乘性误差
- $\delta$：效用判断的加性误差
- $\mu$：成本判断的乘性误差（GPT-4o 常低估目标长度）

近似生成查询的形式化：返回 $\alpha^*$ 满足 $c(\alpha^*) \leq x$ 且

$$\frac{\mathrm{sup}(\alpha^*, S, \ell - \delta)}{\max_{\alpha \in \mathcal{U}: c(\alpha) \leq \lceil \mu x \rceil} \mathrm{sup}(\alpha, S, \ell)} \geq \gamma$$

### 比例代表性公理: $(b, d)$-costBJR
定义了带成本的平衡正当代表制（Balanced Justified Representation）：
- 存在平衡映射 $\omega: N \to W$，不存在联盟 $S$、陈述 $\alpha$、阈值 $\theta$ 同时满足：
  1. $|S| \geq d \cdot \lceil c(\alpha) \cdot n / B \rceil$（联盟足够大）
  2. $u_i(\alpha) \geq \theta, \forall i \in S$（联盟一致偏好 $\alpha$）
  3. $u_i(\omega(i)) < \theta - b, \forall i \in S$（当前分配远不如 $\alpha$）
- 精确版 cBJR 对应 $b=0, d=1$

### DemocraticProcess 算法
核心思想：迭代式贪心——从高效用级别到低级别扫描，每轮尝试生成并添加获足够支持的陈述：

1. 外层循环：效用级别 $\ell$ 从 $r$ 降至 $1$
2. 内层循环：遍历成本列表 $C$，对每个成本 $C[j]$ 调用生成查询
3. 用判别查询筛选在级别 $\ell$ 上支持返回陈述的 agent 集合 $S_\alpha$
4. 若最佳陈述 $\alpha^*$ 的支持者数 $\geq \lceil c(\alpha^*) \cdot n / B \rceil$，加入 slate 并移除对应 agent
5. 否则增大成本继续搜索

两个关键变体：
- **Fast-DemocraticProcess**: $C = \{\lfloor j \cdot B/n \rfloor \mid j \in [n]\}$, $f(\ell) = \{\ell\}$ → 精确查询下保证 cBJR
- **Complex-DemocraticProcess**: $C = [B]$, $f(\ell) = [\ell, r]$ → 近似查询下的最优保证

### 理论保证

**定理 3.1**（精确查询）: Fast-DemocraticProcess 在精确判别查询和 $(\gamma, 0, 1)$-准确生成查询下满足 $(0, 1/\gamma)$-cBJR。

**定理 3.2**（近似查询，核心结果）: Complex-DemocraticProcess 在 $\beta$-准确判别查询和 $(\gamma, \delta, \mu)$-准确生成查询下满足 $(2\beta + \delta, \frac{1}{\gamma\mu})$-cBJR。

**定理 3.3–3.4**（下界）: 近乎匹配的不可能性结果，证明近似保证对误差参数的依赖接近最优。

### PROSE: Proportional Slate Engine
实用系统实现：
- 判别查询：用 GPT-4o 作为人类偏好模型，预测用户对陈述的效用
- 生成查询：两步策略——① 用 text-embedding-3-large 嵌入 + 聚类/近邻找到偏好一致的子群，② 用 GPT-4o 为子群生成共识陈述
- 关键优势：只需无结构文本数据 + 目标 slate 长度作为输入，无需数据集特定调参

## 实验关键数据

### 数据集

| 数据集 | 来源 | agent 数 | 预算 $B$ |
|-------|------|---------|---------|
| Birth Control (Balanced) | UCI Drug Review | 80 | 160 词 |
| Birth Control (Imbalanced) | UCI Drug Review | 80 | 160 词 |
| Obesity | UCI Drug Review | 80 | 160 词 |
| Bowling Green | Polis 城市治理讨论 | 41 | 164 词 |

### 基线方法

| 方法 | 描述 |
|------|------|
| Contextless Zero-Shot | 仅给主题和字数限制，无用户数据 |
| Zero-Shot | 提供所有用户描述，一次性生成 |
| Clustering | 嵌入 + 亲和传播聚类，每簇生成一条 |
| PROSE-UnitCost | 等成本版，对应 Fish et al. 原框架 |

### 核心结果
- PROSE 在**用户满意度**和**比例代表性**两个维度上均超越所有四个基线
- 合成环境验证：所有算法变体实际表现远优于最坏情况理论保证（图 1 灰色区域）
- Fast 和 Complex 变体性能接近且均显著优于 Uniform 变体
- 随误差增大，Fast 和 Complex 的 BJR 违反量逐渐增加，但增长可控

## 亮点与洞察
1. **从委员会选举到参与式预算的自然推广**: 引入成本/预算约束使框架更贴合实际——控制 slate 总长度比固定条数更合理
2. **容错设计**: 近似查询模型优雅地量化了 LLM 不完美性，理论保证随误差参数平滑退化而非完全崩溃
3. **近乎匹配的上下界**: 定理 3.2 与 3.3–3.4 的组合证明算法的近似比接近信息论极限
4. **实用性强**: PROSE 只需无结构文本输入，无需数据集特定调参，适用场景广泛
5. **算法不需知道误差大小**: DemocraticProcess 在执行时无需知道 $\beta, \gamma, \delta, \mu$ 的具体值

## 局限与展望
1. **GPT-4o 的不透明性**: 查询实现依赖黑盒 LLM，无法保证单次响应质量，可能存在偏见和幻觉
2. **Complex-DemocraticProcess 计算开销大**: $C=[B]$ 导致内层循环遍历所有成本值，实际部署时需权衡
3. **生成查询实现困难**: 作者观察到 GPT-4o 在识别偏好一致子群方面表现不佳，需依赖额外的嵌入聚类步骤
4. **数据集规模有限**: 实验仅涉及 41–80 个 agent，未验证大规模（数千用户）场景
5. **$\gamma$ 误差的上下界差距**: 定理 3.2 给出 $1/\gamma$ 而下界为 $|W|/(|W|\gamma+1)$, 当 $|W|$ 较小时差距不可忽略

## 相关工作与启发
- Fish et al. [2024]: 生成式社会选择原始框架，本文直接扩展
- Polis [Small et al., 2021]: 最广泛使用的集体响应系统
- Tessler et al. [2024], Bakker et al. [2022]: 用 LLM 生成单一共识陈述（vs. 本文的多陈述比例代表）
- Peters et al. [2021]: 参与式预算中的比例代表理论

## 评分
- 新颖性: ⭐⭐⭐⭐ — 成本/预算 + 近似查询的双重扩展有实质理论贡献
- 实验充分度: ⭐⭐⭐ — 合成+真实数据验证，但规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-实践框架清晰，动机到结论一气呵成
- 价值: ⭐⭐⭐⭐ — AI+民主方向的重要进展，但离实际部署仍有距离

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](../../ICCV2025/human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[CVPR 2025\] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters](../../CVPR2025/human_understanding/solami_social_vision-language-action_modeling_for_immersive_interaction_with_3d_.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/human_understanding/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[CVPR 2025\] Omni-ID: Holistic Identity Representation Designed for Generative Tasks](../../CVPR2025/human_understanding/omni-id_holistic_identity_representation_designed_for_generative_tasks.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/human_understanding/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

<!-- RELATED:END -->
