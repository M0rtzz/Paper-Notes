---
title: >-
  [论文解读] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][目标检测][分布鲁棒优化] 首次为分布鲁棒平均奖励强化学习（DR-AMDP）建立了有限样本收敛保证，提出两种算法（折扣归约法和锚定法），在KL和$f_k$-散度不确定集下均达到$\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$的近最优样本复杂度。
tags:
  - NeurIPS 2025
  - 目标检测
  - 分布鲁棒优化
  - 平均奖励强化学习
  - 样本复杂度
  - Markov决策过程
  - KL散度
---

# Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.10007](https://arxiv.org/abs/2505.10007)  
**作者**: Zijun Chen (HKUST), Shengbo Wang (USC), Nian Si (HKUST)  
**代码**: 未公开  
**领域**: object_detection  
**关键词**: 分布鲁棒优化, 平均奖励强化学习, 样本复杂度, Markov决策过程, KL散度  

## 一句话总结

首次为分布鲁棒平均奖励强化学习（DR-AMDP）建立了有限样本收敛保证，提出两种算法（折扣归约法和锚定法），在KL和$f_k$-散度不确定集下均达到$\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$的近最优样本复杂度。

## 研究背景与动机

### 问题背景
强化学习在机器人控制、运筹学、医疗等领域取得巨大成功，但标准RL假设训练与部署环境一致，实践中这一假设往往不成立。分布鲁棒强化学习（DR-RL）通过引入对转移概率的对抗扰动来应对环境不匹配问题。

### 已有工作的不足
- 现有DR-RL理论主要集中在**折扣奖励**或**有限时间域**设定，样本复杂度已被充分研究
- **平均奖励**设定在许多实际应用中更为关键（如机器人稳态控制、库存管理、排队系统），但理论上几乎空白
- 已有平均奖励鲁棒MDP的工作（Wang等2023）仅证明了算法收敛性，缺乏**有限样本复杂度保证**
- 非鲁棒设定下，折扣奖励的极小极大样本复杂度早在2013年已被解决，但平均奖励的类似结果直到近年才被建立

### 核心动机
填补DR-RL在平均奖励设定下的理论空白——建立首个有限样本收敛保证，并设计无需先验知识（如混合时间$t_{\mathrm{mix}}$）的实用算法。

## 方法详解

### 问题建模
- **名义MDP**：给定表格型MDP $(S, A, r, P)$，转移核$P$未知但可通过生成模型采样
- **不确定集**：以KL散度或$f_k$-散度为度量，定义SA-矩形不确定集 $\mathcal{P}_{s,a}(D,\delta) = \{p : D(p \| p_{s,a}) \leq \delta\}$
- **目标**：在不确定集下，学习最优鲁棒平均奖励策略 $g_{\mathcal{P}}^* = \max_\pi \inf_{\mathbf{P} \in \mathcal{P}} g_{\mathbf{P}}^\pi$
- **关键假设**：名义MDP均匀遍历（Assumption 1），对抗扰动半径$\delta$足够小（Assumption 2）

### 关键理论贡献 1：不确定集稳定性
传统不确定集可能包含非单链MDP，导致标准Bellman方程失效。本文在Assumption 2约束下证明：对所有$Q \in \mathcal{P}$和$\pi \in \Pi$，其minorization time满足 $t_{\mathrm{minorize}}(Q_\pi) \leq 2 t_{\mathrm{minorize}}$，即对抗扰动不会破坏系统的均匀遍历性（Proposition 4.2）。

### 算法1：折扣归约法（Reduction to DR-DMDP）
1. 从生成模型采集$n$个样本，构建经验转移概率$\hat{p}_{s,a}$
2. 设定校准折扣因子$\gamma = 1 - 1/\sqrt{n}$，平衡统计误差和归约偏差
3. 求解经验DR折扣Bellman方程，得到$V_{\hat{\mathcal{P}}}^*$和最优策略$\hat{\pi}^*$
4. 输出：策略$\hat{\pi}^*$和平均奖励估计$V_{\hat{\mathcal{P}}}^*/\sqrt{n}$

### 算法2：锚定DR-AMDP（Anchored DR-AMDP）
1. 同样构建经验转移概率
2. 选取锚定状态$s_0$和强度参数$\xi = 1/\sqrt{n}$
3. 构建锚定不确定集：$\underline{\hat{\mathcal{P}}}_{s,a} = \{(1-\xi)p + \xi \mathbf{1} e_{s_0}^\top : D(p\|\hat{p}_{s,a}) \leq \delta\}$
4. 直接求解经验DR平均奖励Bellman方程，避免折扣子问题

### 样本复杂度结果
两种算法均达到相同的样本复杂度上界：

$$\widetilde{O}\left(|S||A| \cdot t_{\mathrm{minorize}}^2 \cdot \mathfrak{p}_\wedge^{-1} \cdot \varepsilon^{-2}\right)$$

其中$\mathfrak{p}_\wedge$为最小转移概率支撑。在$\varepsilon$和$|S||A|$的依赖关系上，该结果是最优的。

### 证明核心思路
1. 通过Proposition 4.2确保不确定集内所有MDP的均匀遍历性
2. 将策略误差归约为DR Bellman算子的估计误差
3. 利用DR函数的强对偶性和Bernstein型不等式建立集中不等式
4. 通过$\mathrm{Span}(V_{\mathcal{P}}^\pi) \leq O(t_{\mathrm{minorize}})$完成最终bound

## 实验关键数据

### 实验1：Hard MDP收敛速率验证

使用Wang等(2023)的Hard MDP族进行验证，该MDP族有2个状态、2个动作，由参数$\mathfrak{p}$控制mixing time。

| 设定 | 不确定集 | 算法 | 经验收敛速率 | 理论预测 |
|------|---------|------|------------|---------|
| Hard MDP | KL散度 | Algorithm 2 (折扣归约) | $n^{-1/2}$ 斜率 | $O(n^{-1/2})$ |
| Hard MDP | $\chi^2$散度 | Algorithm 2 (折扣归约) | $n^{-1/2}$ 斜率 | $O(n^{-1/2})$ |
| Hard MDP | KL散度 | Algorithm 3 (锚定法) | $n^{-1/2}$ 斜率 | $O(n^{-1/2})$ |
| Hard MDP | $\chi^2$散度 | Algorithm 3 (锚定法) | $n^{-1/2}$ 斜率 | $O(n^{-1/2})$ |

实验在对数-对数尺度下观察到斜率约$-1/2$，验证了$n^{-1/2}$的收敛速率。每个数据点为单次独立运行结果，回归线周围方差极低。

### 实验2：SOTA样本复杂度对比总结

| 问题类型 | 设定 | 样本复杂度 | 来源 |
|---------|------|-----------|------|
| 标准RL | 折扣 | $\widetilde{\Theta}(|S||A|(1-\gamma)^{-3}\varepsilon^{-2})$ | Azar et al. 2013 |
| 标准RL | 折扣+Mixing | $\widetilde{\Theta}(|S||A|t_{\mathrm{mix}}(1-\gamma)^{-2}\varepsilon^{-2})$ | Wang et al. 2023 |
| 标准RL | 平均+Mixing | $\widetilde{\Theta}(|S||A|t_{\mathrm{mix}}\varepsilon^{-2})$ | Wang et al. 2024 |
| DR-RL | 折扣 | $\widetilde{O}(|S||A|(1-\gamma)^{-4}\varepsilon^{-2})$ | Shi & Chi; Wang et al. |
| **DR-RL** | **折扣+Mixing** | $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2(1-\gamma)^{-2}\varepsilon^{-2})$ | **本文 Thm 4.3** |
| **DR-RL** | **平均+Mixing** | $\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$ | **本文 Thm 4.4 & 4.5** |

本文将DR-DMDP的有效时间域依赖从$(1-\gamma)^{-4}$改进至$(1-\gamma)^{-2}$，并首次给出DR-AMDP的样本复杂度结果。

## 亮点

- **开创性工作**：首次为分布鲁棒平均奖励RL建立有限样本保证，填补了理论空白
- **双算法设计**：折扣归约法和锚定法各有优势——前者建立了与折扣设定的优雅联系，后者直接在平均奖励框架下操作，避免子问题
- **不确定集稳定性分析**：严格建立了对抗扰动不破坏均匀遍历性的充分条件，解决了DR-AMDP的建模困难
- **无需先验知识**：两种算法均不需要预先知道mixing time $t_{\mathrm{mix}}$，通过自适应参数选取实现
- **最优依赖关系**：在$|S||A|$和$\varepsilon$的依赖关系上达到信息论下界，$(1-\gamma)^{-2}$的有效时间域依赖在非鲁棒设定下已知是最优的

## 局限与展望

- **均匀遍历性假设较强**：要求所有策略下MDP均有有限mixing time，排除了weakly communicating和multichain MDP
- **不确定集半径约束**：Assumption 2要求$\delta$足够小（$\delta \leq O(\mathfrak{p}_\wedge / m_\vee^2)$），限制了对抗者的力量
- **$t_{\mathrm{mix}}^2$ vs $t_{\mathrm{mix}}$**：与非鲁棒设定的$\widetilde{\Theta}(t_{\mathrm{mix}})$相比，本文结果多了一个$t_{\mathrm{mix}}$因子，是否能消除仍为开放问题
- **表格型设定**：仅适用于有限状态-动作空间，未涉及函数逼近
- **仅考虑KL和$f_k$散度**：未覆盖$\ell_p$-球和Wasserstein度量等其他重要不确定集
- **生成模型假设**：需要能从任意$(s,a)$采样，不适用于实际交互的online设定

## 与相关工作的对比

- **Shi & Chi (2023), Wang et al. (2024c)**：DR-DMDP下KL不确定集的$\widetilde{O}((1-\gamma)^{-4}\varepsilon^{-2})$，本文利用mixing结构改进为$(1-\gamma)^{-2}$
- **Clavier et al. (2024)**：$\ell_p$-范数约束下达到$\widetilde{\Theta}((1-\gamma)^{-3}\varepsilon^{-2})$极小极大率，本文不覆盖此不确定集类型
- **Wang et al. (2023d/e, 2024d)**：鲁棒平均奖励RL的渐近收敛结果，但无有限样本保证
- **Grand-Clément et al. (2025)**：(s,a)-矩形不确定集下最优策略的结构，不涉及样本复杂度
- **Wang & Si (2024)**：s-矩形不确定集下的Bellman方程存在性，本文在SA-矩形设定下给出定量结果
- **Jin & Sidford (2020), Wang et al. (2024a)**：非鲁棒平均奖励RL的最优$\widetilde{\Theta}(t_{\mathrm{mix}}\varepsilon^{-2})$，本文鲁棒版本多一个$t_{\mathrm{mix}}$因子

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次解决DR-AMDP的样本复杂度开放问题
- 实验充分度: ⭐⭐⭐ — 仅在简单Hard MDP上验证，缺乏更大规模和实际应用实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论体系完整，上下文引用详尽
- 价值: ⭐⭐⭐⭐ — 填补重要理论空白，但强假设限制了实际适用范围
