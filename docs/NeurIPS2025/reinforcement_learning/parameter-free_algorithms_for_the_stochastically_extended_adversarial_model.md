---
title: >-
  [论文解读] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model
description: >-
  [NeurIPS 2025][强化学习][在线凸优化] 针对桥接对抗性和随机在线凸优化的 SEA 模型，首次开发无参数算法：在未知域直径 $D$ 和/或 Lipschitz 常数 $G$ 条件下…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "在线凸优化"
  - "无参数算法"
  - "SEA模型"
  - "比较器自适应"
  - "Lipschitz自适应"
---

# Parameter-Free Algorithms for the Stochastically Extended Adversarial Model

**会议**: NeurIPS 2025  
**arXiv**: [2510.04685](https://arxiv.org/abs/2510.04685)  
**代码**: 无  
**领域**: Reinforcement Learning / Online Learning  
**关键词**: 在线凸优化, 无参数算法, SEA模型, 比较器自适应, Lipschitz自适应

## 一句话总结

针对桥接对抗性和随机在线凸优化的 SEA 模型，首次开发无参数算法：在未知域直径 $D$ 和/或 Lipschitz 常数 $G$ 条件下，基于 Optimistic Online Newton Step (OONS) 实现与已知参数情况相当的 regret 界。

## 研究背景与动机

在线凸优化（OCO）的 SEA 模型弥合了纯对抗和纯随机设置之间的差距，其 regret 界依赖于累积随机方差 $\sigma_{1:T}^2$ 和累积对抗变化 $\Sigma_{1:T}^2$ 而非时间 $T$。然而现有 SEA 模型算法（OFTRL、OMD）的最优步长依赖于预先知道域直径 $D$ 和 Lipschitz 常数 $G$，限制了实际应用。开发能自适应这些参数的"无参数"算法是一个重要的开放问题。直接套用已有的无参数 OCO 算法无法获得理想的 $\sigma_{1:T}^2$-dependent 界（只能得到更松的 $\tilde{\sigma}_{1:T}^2$），因为实现 $\sigma_{1:T}^2$-scaling 需要利用 RVU 性质，而这在无界域下更加困难。

## 方法详解

### 整体框架

分三步递进：(1) 建立 OONS 作为基础算法，使用自适应步长在已知参数下匹配 SOTA 界；(2) 通过多尺度基学习者+MsMwC 元算法框架构建 CA-OONS（比较器自适应）；(3) 结合梯度截断和域直径倍增构建 CLA-OONS（比较器+Lipschitz 自适应）。

### 关键设计

1. **Optimistic Online Newton Step (OONS)**: 基于 ONS 算法引入自适应步长 $\eta_t = \min\{\frac{1}{64Dz_T}, \frac{1}{D\sqrt{\sum_{s=1}^{t-1}\|g_s-m_s\|_2^2}}\}$。维护两个序列 $\{x_t\}$ 和 $\{x_t'\}$（乐观预测），使用二阶信息矩阵 $A_t$ 自适应调整更新方向。关键性质：regret 分解中 $D$ 仅出现在 $D(z_T - z_1)$ 项（仅依赖端点），而非贯穿所有轮次，这使得后续消除 $D$ 依赖成为可能。OONS 满足 RVU 性质，包含负稳定性项 $-z_1^2 \sum_t \|x_t - x_{t-1}\|_2^2$。

2. **CA-OONS（比较器自适应, Algorithm 2）**: 构建三层结构处理未知 $D$。底层：$N = \lceil \log T \rceil$ 个 OONS 基学习者，每个在半径 $D_j = 2^j$ 的受限域上运行。中层：MsMwC 算法为每个尺度 $k$ 学习基学习者的最优组合权重 $w_t^k$。顶层：MsMwC-Master 算法学习尺度间的最优混合 $p_t \in \Delta_\mathcal{S}$，最终决策 $x_t = \sum_k p_{t,k} w_t^k$。Regret 分解为 MetaRegret + BaseRegret，各层分别控制。

3. **CLA-OONS（比较器+Lipschitz 自适应, Algorithm 4）**: 在 $G$ 也未知时，引入梯度截断 $\tilde{g}_t = m_t + \frac{B_{t-1}}{B_t}(g_t - m_t)$，其中 $B_t = \max_{s \leq t} \|g_s - m_s\|_2$ 。域直径 $D_t$ 通过倍增策略动态更新：当 $D_t < \sqrt{\sum_s \|g_s\|_2 / \max_{k \leq s} \|g_k\|_2}$ 时倍增。最多触发 $M = O(\log T)$ 次重置，每次重置 $A_{t+1}$ 矩阵和 $x'_{t+1}$。

### 损失函数 / 训练策略

在线学习设置，无显式损失函数训练。算法依赖乐观预测 $m_t = \nabla f_{t-1}(x_{t-1})$。CA-OONS 中元算法初始化 $p_1' \propto \beta_k^2$，各层步长精心设计以控制多尺度 regret。OONS 的自适应步长结合了最小值形式 $\eta_t = \min\{\frac{1}{64Dz_T}, \frac{1}{D\sqrt{V_{t-1}}}\}$，确保步长随梯度变化自适应。CLA-OONS 的域直径倍增最多触发 $M = O(\log T)$ 次，每次重置二阶信息矩阵 $A_t$。专家集 $\mathcal{S}$ 的大小为 $O(N \log T)$，总计算开销可接受。

## 实验关键数据

### 主实验

| 算法 | 无需 $D$ | 无需 $G$ | 期望 Regret 界 |
|------|----------|----------|---------------|
| OFTRL/OMD (Sachs et al.) | ✗ | ✗ | $O(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2})$ |
| OONS (Theorem 3.2) | ✗ | ✗ | $\tilde{O}(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2})$ |
| CA-OONS (Theorem 4.1) | ✓ | ✗ | $\tilde{O}(\|u\|_2^2 + \|u\|_2(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}))$ |
| CLA-OONS (Theorem 4.5) | ✓ | ✓ | $\tilde{O}(\|u\|_2^2(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}) + \|u\|_2^4 + \sqrt{\sigma_{1:T} + \mathfrak{G}_{1:T}})$ |

### 消融实验

| 配置 | 说明 |
|------|------|
| OONS vs OMD 作为基算法 | OMD 的 regret 依赖 $D\sqrt{\sum\|g_t\|^2}$，无界域下退化为 $O(T)$ |
| CA-OONS 在有界域 | 恢复 $\tilde{O}(D^2 + D(\sqrt{\sigma_{1:T}^2} + \sqrt{\Sigma_{1:T}^2}))$，匹配 SOTA |
| $\|u\|_2^2$ vs $\|u\|_2$ 依赖 | 当前技术下 gradient-variation regret 中 $\|u\|_2^2$ 可能不可避免 |

### 关键发现

- OONS 的二阶自适应性质是消除 $D$ 依赖的关键——OMD/OFTRL 做不到这一点
- CA-OONS 的 $\|u\|_2^2$ 项在 gradient-variation 界下可能是固有的（类比离线加速优化中 $d_0^2$ 的出现）
- CLA-OONS 引入的 $\sqrt{\mathfrak{G}_{1:T}}$ 附加项与 $\|u\|$ 无关，不随比较器范数增长
- $\sigma_{1:T}^2$ 比 $\tilde{\sigma}_{1:T}^2$ 更优（后者可任意大），但实现 $\sigma_{1:T}^2$-scaling 需要 RVU 性质

## 亮点与洞察

- 首次在 SEA 模型中实现无参数算法，同时保持对 $\sigma_{1:T}^2$ 和 $\Sigma_{1:T}^2$ 的依赖
- OONS 中 $D$ 仅出现在端点项 $D(z_T - z_1)$ 的观察非常精巧，是技术突破的关键
- 多尺度三层结构（基学习者-中层MsMwC-顶层Master）的系统设计清晰
- 关于 gradient-variation 界和 worst-case 界之间自适应的讨论（Remark 4.3）揭示了深层理论联系

## 局限与展望

- CLA-OONS 的 leading term 是 $\|u\|_2^2$ 而非理想的 $\|u\|_2$，改进这一依赖是重要开放问题
- CA-OONS 需要 $O(\log T)$ 个梯度查询，降至 $O(1)$ 待解决
- 缺乏实验验证，纯理论工作
- 高概率 regret 界（而非期望）尚未建立
- 能否设计同时适应 gradient-variation 和 worst-case 两种界的单一算法仍是开放问题

## 相关工作与启发

- Sachs et al. 和 Chen et al. 建立了 SEA 模型的基线 regret，但步长依赖 $D$ 和 $G$
- chen2021impossible 的 MsMwC 框架被借用用于多尺度学习率管理
- cutkosky2019artificial 的梯度截断技术被改造用于 CLA-OONS 的 Lipschitz 自适应
- RVU 性质连接了在线学习与博弈论和加速优化，是重要的理论桥梁
- Jacobsen et al. 的无参数镜像下降可获得 gradient-variation 界但依赖 $\tilde{\sigma}_{1:T}^2$
- orabona2016coin 和 cutkosky2018black 的比较器自适应方法仅适用于纯对抗设置
- SEA 模型在专家预测和 bandit 设置中已有较好理解，但 OCO 设置更具挑战
- 从在线到离线的转换（online-to-batch conversion）将 gradient-variation regret 映射为加速收敛率，揭示了深层联系
- 倍增技巧可去除对 $T$ 的依赖，实现 anytime 算法
- tuning-free SGD 和 DoG 步长等工作提供了相关但不可直接套用的技术路线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次解决 SEA 模型无参数问题，OONS 端点性质的发现精巧
- 实验充分度: ⭐⭐ 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐ 技术内容深入，架构层次清晰，但符号密度极高
- 价值: ⭐⭐⭐⭐ 填补了 SEA 模型无参数算法的理论空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] When Can Model-Free Reinforcement Learning be Enough for Thinking?](when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)
- [\[NeurIPS 2025\] Improving Planning and MBRL with Temporally-Extended Actions](improving_planning_and_mbrl_with_temporally-extended_actions.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)

</div>

<!-- RELATED:END -->
