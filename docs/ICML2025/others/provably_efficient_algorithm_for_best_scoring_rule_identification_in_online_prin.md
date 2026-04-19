---
title: >-
  [论文解读] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition
description: >-
  [ICML 2025][在线信息获取] 本文在委托-代理（principal-agent）在线信息获取框架下研究最佳评分规则识别（Best Scoring Rule Identification, BSRI）问题，提出 OIAFC（固定置信度）和 OIAFB（固定预算）两种算法，首次建立了实例依赖的样本复杂度上界 $\widetilde{O}(MH_\Delta)$，并将实例无关的样本复杂度从已有工作的 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ 大幅改进至 $\widetilde{O}(MK\epsilon^{-2})$。
tags:
  - ICML 2025
  - 在线信息获取
  - 最优评分规则
  - 委托-代理博弈
  - 多臂赌博机
  - 最佳臂识别
  - 样本复杂度
---

# Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition

**会议**: ICML 2025  
**arXiv**: [2505.17379](https://arxiv.org/abs/2505.17379)  
**代码**: 无  
**领域**: 其他（在线学习 / 机制设计 / 博弈论）  
**关键词**: 在线信息获取, 最优评分规则, 委托-代理博弈, 多臂赌博机, 最佳臂识别, 样本复杂度

## 一句话总结

本文在委托-代理（principal-agent）在线信息获取框架下研究最佳评分规则识别（Best Scoring Rule Identification, BSRI）问题，提出 OIAFC（固定置信度）和 OIAFB（固定预算）两种算法，首次建立了实例依赖的样本复杂度上界 $\widetilde{O}(MH_\Delta)$，并将实例无关的样本复杂度从已有工作的 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ 大幅改进至 $\widetilde{O}(MK\epsilon^{-2})$。

## 研究背景与动机

### 问题场景

信息获取问题研究的是一个分层博弈场景：**委托人**（principal，如经理）雇佣**代理人**（agent，如领域专家）来收集关于未知环境状态 $\omega$ 的信息。核心困难在于委托人无法直接观察代理人的努力程度，必须设计合理的**评分规则**（scoring rule）来激励代理人既选择最优行动又如实汇报其对环境状态的信念。

典型应用包括：
- **在线问卷调查**：委托人设计问卷，代理人选择调查方式（不同成本和信息质量）
- **广告效果评估**：经理评估广告策略的市场适配度，专家选择研究方法获取市场信息
- **众包验证**：平台（委托人）通过评分规则激励工人（代理人）提供高质量数据

### 现有工作的不足

之前只有 Chen et al. (2023) 和 Cacciamani et al. (2023) 研究了在线信息获取：

**Chen et al. (2023)** 侧重于遗憾最小化，虽然给出了固定置信度 BSRI 的结果，但样本复杂度为 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$，远大于标准多臂赌博机最佳臂识别的 $\widetilde{O}(K\epsilon^{-2})$

**Cacciamani et al. (2023)**：ETC（先探索后提交）架构天然难以获得实例依赖的样本复杂度
3. 完全**不存在实例依赖**的样本复杂度结果

**固定预算**设定下的 BSRI 完全未被探索

### 与标准最佳臂识别的区别

BSRI 比标准多臂赌博机的最佳臂识别更具挑战性：
- 委托人不能直接"拉臂"，只能通过设计评分规则来**间接激励**代理人选择特定行动
- 评分规则需要同时实现两个目标：(1) 激励代理人选择最优行动，(2) 激励代理人如实汇报信念
- 需要额外学习信念分布 $q_k$ 和成对成本差异 $C(k, k')$

## 方法详解

### 整体框架

本文的核心思路是将信息获取问题**化简为类赌博机问题**，然后设计 UCB 风格的算法进行求解。

**交互协议**（每轮 $t$）：
1. 委托人宣布评分规则 $S_t : \Omega \times \Delta(\Omega) \to \mathbb{R}_+$
2. 代理人选择行动 $k_t$（可被委托人观察），产生成本 $c_{k_t}$
3. 环境生成隐藏状态 $\omega_t$，发送观测 $o_t$ 给代理人
4. 代理人提交信念报告 $\hat{\sigma}_t$
5. 环境揭示 $\omega_t$，委托人按评分规则支付并获得效用

**问题化简**：定义 $V_k = \{S \in \mathcal{S} \mid g(k, S) \geq g(k', S), \forall k'\}$ 为代理人选择行动 $k$ 的评分规则区域，原始双层优化问题化简为：

$$\max_{k \in \mathcal{A}} h(S_k^*), \quad h(S_k^*) = \sup_{S \in V_k} \mathbb{E}_{\sigma \sim q_k}[u(\sigma) - S(\sigma)]$$

当已知 $V_k$、$q_k$ 和成对成本差异 $C(k,k')$ 时，内层问题可转化为线性规划 $\text{LP}_k$。

### 关键设计一：UCB 线性规划 (UCB-LP)

由于委托人不知道信念分布 $q_k$ 和成对成本差异 $C(k,k')$，算法需要**在线学习**这些参数。定义经验估计器：

$$\hat{q}_k^t(\sigma) = \frac{1}{N_k^t} \sum_{s=1}^{t-1} \mathbf{1}\{\sigma_s = \sigma, k_s = k\}$$

以及置信半径（固定置信度设定）：

$$I_q^t(k) = \sqrt{\frac{2\log(4K^2 M t^2 / \delta)}{N_k^t}}$$

将估计值和置信半径代入 $\text{LP}_k$，构建上置信界线性规划 UCB-LP$_{k,t}$：

$$\hat{h}_k^t = \max_{S \in \mathcal{S}} \hat{u}_k^t + B_u I_q^t(k) - v$$
$$\text{s.t.} \quad |v - \hat{v}_S^t(k)| \leq B_S I_q^t(k)$$
$$v - \hat{v}_S^t(k') \geq \hat{C}^t(k,k') - (I_c^t(k,k') + B_S I_q^t(k')), \quad \forall k' \neq k$$

这给出了 $h(S_k^*)$ 的高概率上界估计 $\hat{h}_k^t$，用于选择当前最优臂 $k_t^* = \arg\max_k \hat{h}_k^t$。

### 关键设计二：自适应权衡参数与终止条件

**保守评分规则**：为确保代理人按预期选择行动 $k_t^*$，算法采用混合策略：

$$S_t = \alpha_k^t \tilde{S}_{k_t^*} + (1 - \alpha_k^t) \hat{S}_{k_t^*, t}$$

其中 $\tilde{S}_k$ 来自"行动知情预言机"（保证代理人选择行动 $k$），$\hat{S}_{k,t}$ 是学到的近似最优评分规则。

**实例依赖参数设计**（本文核心贡献）：

$$\alpha_k^t = \min\left(\sqrt{\frac{M}{L_k^t}}, 1\right), \quad \beta_t = \frac{\epsilon^{-2} \alpha_{k_t^*}^t (B_S + B_u)}{1 - \alpha_{k_t^*}^t}$$

Chen et al. (2023) 使用固定的 $\alpha_k^t = \min(K/t^{1/3}, 1)$，导致样本复杂度仅为实例无关的次优界。本文的自适应参数使得强制探索轮数满足 $L_k^\tau = \widetilde{O}(M^2(B_S + B_u)^2 \Delta_k^{-2})$，从而获得实例依赖界。

**终止规则**：当 $2(B_S + B_u) I_q^t(k_t^*) \leq \beta_t$ 时停止，输出 $\hat{S}^* = S_t$。

### 关键设计三：二分搜索学习成本差异

当代理人实际选择的行动 $k_t \neq k_t^*$ 时，利用凸组合的二分搜索来估计成对成本差异 $C(k, k')$：

$$C(k, k') = v_S(k) - v_S(k'), \quad \forall S \in V_k \cap V_{k'}$$

在 $\tilde{S}_k$（触发行动 $k$）和 $\tilde{S}_{k'}$（触发行动 $k'$）之间进行二分搜索，找到边界处的评分规则 $S \in V_k \cap V_{k'}$，从而利用等式（7）更新成本差异估计。

## 理论结果

由于本文是纯理论工作，不含数值实验，以下用表格总结关键理论结果：

### 样本复杂度对比

| 算法/结果 | 设定 | 样本复杂度 | 类型 |
|:--|:--|:--|:--|
| Chen et al. (2023) | 固定置信度 | $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ | 实例无关 |
| **OIAFC (本文)** | 固定置信度 | $\widetilde{O}(\epsilon^{-2} B_S^2 M H_\Delta)$ | **实例依赖** |
| **OIAFC (本文)** | 固定置信度 | $\widetilde{O}(\epsilon^{-2} B_S^2 M K \epsilon^{-2})$ | 实例无关 |
| **OIAFB (本文)** | 固定预算 | $T = \widetilde{O}(\epsilon^{-2} B_S^2 M K \epsilon^{-2})$ | 实例无关 |
| 标准 MAB BAI | 固定置信度 | $\widetilde{O}(K\epsilon^{-2})$ | 实例无关 |

其中 $H_\Delta = 4(B_S+B_u)^2\epsilon^{-2} + \sum_{k \neq k^*} \Delta_k^{-2}$，$M \leq K \times C_O$。

### 核心定理对比

| 定理 | 内容 | 条件 |
|:--|:--|:--|
| 定理 1 (OIAFC) | 输出满足 $(ε, δ)$-条件的评分规则，样本复杂度 $\widetilde{O}(\epsilon^{-2}B_S^2 M H_\Delta)$ | $\alpha_k^t = \min(\sqrt{M/L_k^t}, 1)$ |
| 推论 1 (OIAFC) | 实例无关界 $\widetilde{O}(MK\epsilon^{-2})$ | $\alpha_k^t = \epsilon/(4(B_S+B_u))$（固定） |
| 定理 2 (OIAFB) | 给定预算 $T$，输出 $(ε, \tilde{\delta})$-评分规则 | $\alpha_k^t = \epsilon/(4(B_S+B_u))$ |
| 推论 2 (OIAFB) | 预算 $T = \widetilde{O}(MK\epsilon^{-2})$ 时 $\tilde{\delta} \leq \delta$ | 与 OIAFC 实例无关界匹配 |

## 关键发现

1. **首个实例依赖结果**：OIAFC 建立了 BSRI 的首个实例依赖样本复杂度上界，在简单设定下回退到标准 MAB BAI 的已知最优界 $\widetilde{O}(H_\Delta)$
2. **大幅改进实例无关界**：从 Chen et al. 的 $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ 改进至 $\widetilde{O}(MK\epsilon^{-2})$，消除了对 $\epsilon$ 的三次方依赖
3. **固定预算与固定置信度统一**：OIAFB 在实例无关设定下匹配 OIAFC 的样本复杂度，说明两种设定具有相同的复杂度刻画
4. **改进的终止策略**：Chen et al. 使用来自 Jin et al. (2018) 的固定迭代终止，需要 $\widetilde{O}(\epsilon^{-6} K^6 C_O^3)$ 轮；本文设计了自适应终止规则，显著提升效率

## 亮点与洞察

- **从间接激励到直接识别的桥梁**：核心洞察是将评分规则设计问题化简为以 $h(S_k^*)$ 为奖励的多臂赌博机问题，但关键区别在于委托人不能直接"拉臂"，需要通过评分规则间接控制代理人行为
- **自适应权衡参数 $\alpha_k^t$**：这是本文最关键的技术贡献。Chen et al. 使用对所有臂统一的固定衰减 $\alpha_k^t = \min(K/t^{1/3}, 1)$，而本文为每个臂设计了自适应参数 $\alpha_k^t = \min(\sqrt{M/L_k^t}, 1)$，依赖于各臂的实际探索次数，从而获得实例依赖的复杂度
- **UCB-LP 的巧妙结合**：将 UCB 的乐观估计思想与线性规划框架结合，既保证了高概率上界性质，又利用了问题的线性结构以高效求解
- **理论框架完整性**：同时覆盖固定置信度 + 固定预算、实例依赖 + 实例无关四种组合，构建了完整的理论图景

## 局限性

1. **假设行动可观察**：要求委托人能观察到代理人选择的行动（研究方法），虽然论文给出了在线/离线调查的实际例子，但许多实际场景中行动难以观察
2. **需要行动知情预言机**：算法依赖提供 $K$ 个评分规则 $\{\tilde{S}_k\}$ 的"行动知情预言机"（Assumption 1），这在实际中可能不易获得
3. **无数值实验验证**：纯理论工作，缺乏实验验证理论界的紧度和算法的实际收敛速度
4. **$M$ 因子的代价**：样本复杂度中的 $M \leq K \times C_O$ 意味着信念空间较大时复杂度可能较高
5. **未探索下界**：没有提供 BSRI 的信息论下界，无法判断所得上界是否最优

## 相关工作与启发

- **最佳臂识别**（Even-Dar et al. 2002/2006; Gabillon et al. 2012; Jamieson et al. 2013）：本文将 BAI 的技术（UCB + 消除）推广到具有间接控制机制的策略博弈场景
- **在线策略环境学习**：包括 Stackelberg 博弈（Sessa et al. 2020）、在线拍卖（Feng et al. 2017）、合同设计（Ho et al. 2014; Zhu et al. 2022）和贝叶斯说服（Castiglioni et al. 2020）等
- **离线评分规则设计**（Neyman et al. 2021; Li et al. 2022; Hartline et al. 2023）：本文将离线问题推广到在线学习设定
- **启发**：该框架可推广到更复杂的委托-代理场景，如多代理人信息获取、部分可观察行动、或连续评分规则空间

## 评分

⭐⭐⭐

理论贡献扎实，首次建立了在线信息获取问题的实例依赖样本复杂度，改进幅度显著。但问题设定的假设较强（行动可观察 + 行动知情预言机），且缺少实验验证，实际应用场景偏窄。整体是一篇规范的理论工作，对在线学习与机制设计的交叉领域有推进意义。

<!-- RELATED:START -->

## 相关论文

- [Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)
- [Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification](fixing_the_loose_brake_exponential-tailed_stopping_time_in_best_arm_identificati.md)
- [OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](ood-chameleon_is_algorithm_selection_for_ood_generalization_learnable.md)
- [TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](../../ACL2025/others/taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)
- [Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](fishers_for_free_approximating_the_fisher_information_matrix_by_recycling_the_sq.md)

<!-- RELATED:END -->
