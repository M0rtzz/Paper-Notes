---
title: >-
  [论文解读] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 提出多人联邦学习（MpFL）框架，将FL中的客户端建模为博弈论中的理性玩家，并设计PEARL-SGD算法通过局部更新减少通信开销，同时收敛到Nash均衡。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "博弈论"
  - "Nash均衡"
  - "局部SGD"
  - "通信效率"
---

# Multiplayer Federated Learning: Reaching Equilibrium with Less Communication

**会议**: NeurIPS 2025  
**arXiv**: [2501.08263](https://arxiv.org/abs/2501.08263)  
**代码**: 无  
**领域**: 优化  
**关键词**: 联邦学习, 博弈论, Nash均衡, 局部SGD, 通信效率

## 一句话总结

提出多人联邦学习（MpFL）框架，将FL中的客户端建模为博弈论中的理性玩家，并设计PEARL-SGD算法通过局部更新减少通信开销，同时收敛到Nash均衡。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：传统联邦学习假设所有客户端共享协作目标，共同优化一个全局模型。但现实中，很多场景下参与者是理性个体，有各自的目标函数，甚至目标可能相互冲突（如Cournot竞争、电力市场、移动机器人控制等）。现有FL框架无法处理这种非合作环境。

作者观察到：(1) 经典FL的Local SGD无法直接用于多玩家博弈，因为每个玩家只优化自己的目标函数；(2) 联邦极小极大优化虽然涉及博弈，但所有客户端同时调整双方变量，不是"每个客户端只管自己"的设定；(3) 个性化FL是MpFL的一个子类，但MpFL的适用范围更广。

### 解决思路

**本文目标**：### 整体框架

MpFL将n个客户端视为n-player博弈中的玩家。


## 方法详解

### 整体框架

MpFL将n个客户端视为n-player博弈中的玩家。每个玩家 $i$ 有自己的策略 $x^i \in \mathbb{R}^{d_i}$ 和目标函数 $f_i(x^1, \ldots, x^n)$，目标是找到Nash均衡 $\mathbf{x}_\star$，使得没有玩家能通过单方面改变策略获益。
均衡条件为 $f_i(x_\star^i; x_\star^{-i}) \leq f_i(x^i; x_\star^{-i}), \forall x^i, \forall i$。通信通过中央服务器协调。

### 关键设计

1. **PEARL-SGD算法**: 每个玩家独立执行 $\tau$ 步局部SGD，固定其他玩家的策略不变，仅更新自己的策略。每 $\tau$ 步进行一次同步——服务器收集所有玩家的策略，将拼接后的联合策略向量广播回所有玩家。这与经典FL中的Local SGD不同，因为每个玩家只对自己的变量求梯度。

2. **Player Drift控制**: 类比经典FL中的client drift，但本质不同。在MpFL中，如果步长不随 $\tau$ 缩放，玩家的局部迭代会收敛到局部极小值 $x_\star^i(x_{\tau p}^{-i})$，由于这些极小值依赖于其他玩家的策略，可能导致发散。因此步长需满足 $\gamma = \mathcal{O}(1/\tau)$ 的约束。

3. **理论假设体系**: 在函数层面假设凸性（CVX）和光滑性（SM），在联合梯度算子层面假设拟强单调性（QSM）和星余强制性（SCO）。QSM+SCO蕴含 $\mu \|\mathbf{x}-\mathbf{x}_\star\| \leq \|\mathcal{F}(\mathbf{x})\| \leq \ell \|\mathbf{x}-\mathbf{x}_\star\|$，条件数 $\kappa = \ell/\mu$。

### 损失函数 / 训练策略

每个玩家最小化自己的期望损失 $f_i(x^i; x^{-i}) = \mathbb{E}_{\xi^i \sim \mathcal{D}_i}[f_{i,\xi^i}(x^i; x^{-i})]$，使用随机梯度 $g_k^i = \nabla f_{i,\xi_k^i}(x_k^i; x_{\tau p}^{-i})$ 更新。支持常数步长和递减步长两种策略：常数步长收敛到均衡邻域，递减步长精确收敛到均衡点。
常数步长下邻域大小为 $\mathcal{O}(\gamma \sigma^2 / \mu)$，递减步长 $\gamma_k \propto 1/k$ 可实现精确收敛但速率为次线性。

## 实验关键数据

### 主实验

| 实验设置 | 指标 | 结果 | 说明 |
|---------|------|------|------|
| 二次函数博弈 (n=5, d=10) | $\|\mathbf{x}_T - \mathbf{x}_\star\|^2$ | 验证理论收敛速率 | τ越大通信越少，收敛行为与理论一致 |
| 移动机器人控制 (n=4) | 到均衡距离 | PEARL-SGD匹配分布式GDA精度 | τ=10时通信减少10x而精度相当 |
| 不同噪声水平 σ | 收敛邻域大小 | $\propto \sigma^2/\mu$ | 与Theorem 3.4完全吻合 |
| 不同玩家数 n=2,5,10 | 收敛曲线 | 收敛行为稳定 | n增大不影响每玩家的收敛速率 |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| τ=1 vs τ=10 vs τ=50 | 收敛曲线 | τ=10达到最佳通信-精度平衡 |
| 常数步长 vs 递减步长 | 最终距离 | 递减步长可精确收敛，常数步长收敛到邻域 |
| 不同条件数κ | 通信复杂度 | κ越小，τ可取越大，通信节省越多 |

### 关键发现

- 在随机设定下，PEARL-SGD通信复杂度可从 $T$ 降到 $\Theta(\sqrt{T})$，条件是 $\tau = \mathcal{O}(\sqrt{\mu T / L_{\max}})$
- 确定性设定下由于步长约束 $\gamma = \mathcal{O}(1/\tau)$，PEARL-SGD无法节省通信
- 当 $L_{\max} \ll \ell$ 时，通信增益更显著

## 亮点与洞察

- 首次将Local SGD思想引入多玩家博弈框架，提出MpFL这一新范式
- Player drift是MpFL特有的现象，与经典FL的client drift有本质区别（可导致发散而非仅停在次优点）
- 理论分析紧密：τ=1时退化为标准SGDA分析，与已有文献一致
- 框架统一了个性化FL的正则化共识模型作为特例
- 通信复杂度从 $T$ 降到 $\Theta(\sqrt{T})$ 的幅度在实际分布式系统中意义重大
- 条件数 $\kappa = \ell/\mu$ 的角色清晰：$\kappa$ 越小（问题越well-conditioned），通信节省越多
- 异构数据设定（无任何数据分布约束）增强了实际适用性

## 局限与展望

- 同步步骤通信代价为 $\mathcal{O}(D) = \mathcal{O}(d_1 + \cdots + d_n)$，随玩家数线性增长，更适合cross-silo场景
- 未结合梯度压缩技术减少每轮通信量
- player drift的缓解策略（类似SCAFFOLD的方差修正）留待未来研究
- 仅考虑了凸函数设定，非凸情况未涉及
- 实验仅在合成数据和简单控制问题上验证，缺乏ML任务的实际评估
- 未考虑部分参与（partial participation），即每轮只有部分玩家通信的场景
- 异步通信版本的PEARL-SGD尚未探索

## 相关工作与启发

- 与联邦极小极大优化（Local SGDA/SEG）的区别：后者所有客户端共同优化一个极小极大问题，MpFL中每个客户端仅优化自己的变量
- QSM和SCO假设类似变分不等式(VIP)文献中的条件，比强单调性更弱
- 可启发在分布式multi-agent系统、竞争市场建模中应用FL技术
- 个性化FL中的共识正则化公式 $\phi(x^1,\ldots,x^n) = \frac{\lambda}{2n}\sum_{i=1}^n \|x^i - \bar{x}\|^2$ 是MpFL的子问题
- 并发工作Decoupled SGD与PEARL-SGD算法一致，但在弱耦合博弈假设下分析确定性通信收益
- 分布式Nash均衡搜索文献通常假设廉价通信，MpFL首次将通信效率引入这一领域
- 未来可探索结合梯度压缩（如QSGD）和Local SGD思想实现双重通信节省
- 非凸非单调设定下的MpFL是重要的开放问题，需要新的分析工具

## 评分

- 新颖性: ⭐⭐⭐⭐ 提出了FL与博弈论交叉的新框架，问题定义清晰且有实际意义
- 实验充分度: ⭐⭐⭐ 实验偏理论验证，缺乏大规模实际应用场景的评估
- 写作质量: ⭐⭐⭐⭐ 符号定义清晰，理论推导严谨，表格总结到位
- 价值: ⭐⭐⭐⭐ 开辟了MpFL新方向，为通信高效的分布式多玩家博弈求解奠定理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [\[NeurIPS 2025\] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines](learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)
- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](../../ICML2025/optimization/the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[AAAI 2026\] Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)

</div>

<!-- RELATED:END -->
