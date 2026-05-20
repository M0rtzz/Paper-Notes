---
title: >-
  [论文解读] On the Expressive Power of GNNs to Solve Linear SDPs
description: >-
  [ICML 2026][图学习][半定规划] 本文从 Weisfeiler–Leman 层级的角度首次刻画了学习线性 SDP 解所需的最小 GNN 表达力，证明标准的变量-约束二部图消息传递（VC-WL）和高阶 VC-2-WL 都不够…
tags:
  - "ICML 2026"
  - "图学习"
  - "半定规划"
  - "图神经网络"
  - "Weisfeiler-Leman"
  - "PDHG"
  - "warm-start"
---

# On the Expressive Power of GNNs to Solve Linear SDPs

**会议**: ICML 2026  
**arXiv**: [2604.27786](https://arxiv.org/abs/2604.27786)  
**代码**: 未公开  
**领域**: 优化 / 图神经网络 / 学习优化  
**关键词**: 半定规划, GNN 表达力, Weisfeiler-Leman, PDHG, warm-start

## 一句话总结
本文从 Weisfeiler–Leman 层级的角度首次刻画了学习线性 SDP 解所需的最小 GNN 表达力，证明标准的变量-约束二部图消息传递（VC-WL）和高阶 VC-2-WL 都不够，而 2-FWL 等价的 VC-2-FWL 架构足以仿真 PDHG 求解器的更新步骤，并在合成与 SDPLIB 上把高质量预测用作 warm-start，最多带来约 80% 的加速。

## 研究背景与动机

**领域现状**：把线性规划（LP）和混合整数规划（MILP）用 GNN 学习求解（learning-to-optimize, L2O）已经相当成熟，主流做法是把约束-变量构造为二部图，再用 1-WL 等价的消息传递（如 Gasse 等人的 IPM/PDHG 对齐 GNN），在中小规模问题上能逼近最优解或显著加速 IPM。

**现有痛点**：把同一思路直接搬到 **半定规划（SDP）** 上失败了。SDP 的核心变量不是向量分量，而是一整个对称半正定矩阵 $X \in S^n_+$，其条目 $X_{ij}$ 在"行列同时置换"下等变；传统 V-C 二部图根本没法表达这种 2-阶对称性，导致 GNN 无论训练多久都拟不上最优解。

**核心矛盾**：现有 L2O 文献只研究了 LP/QP/SOCP 的表达力门槛（多数 1-WL 就够），但 SDP 的"约束-变量-条目"是三阶张量结构，**需要哪一层 WL 才足以恢复最优解**至今没有答案，这直接卡住了"神经 SDP 求解器"的方向。

**本文目标**：1) 形式化 SDP 实例到最优矩阵解 $X^*$ 的映射所需的最小表达力；2) 证明标准 GNN 类架构的不可能性；3) 给出充分性架构并实验验证；4) 把预测用作经典求解器 warm-start。

**切入角度**：将 SDP 写成约束矩阵 $A_k$、目标矩阵 $C$ 与变量矩阵 $X$ 共同构成的"超图"，把 1-WL/2-WL/2-FWL 的 hash 更新原样套到这个张量图上，分别构造 VC-WL、VC-2-WL、VC-2-FWL，然后通过模拟一阶求解器 PDHG 来证明充分性——只要 GNN 能仿真求解器一步迭代，就一定能逼近收敛解。

**核心 idea**：用 2-FWL 等价的"成对节点联合聚合"取代标准消息传递，使 GNN 的稳定着色细于 PDHG 迭代下的轨迹划分，从而获得表达力上的充分条件。

## 方法详解

### 整体框架
输入是 SDP 实例 $(C, \{A_k\}_{k=1}^m, \{b_k\}_{k=1}^m)$。作者把它编码成一张**双类型节点的图**：变量节点对应矩阵 $X$ 的条目 $(i,j) \in [n]\times[n]$，约束节点对应每个 $A_k$；变量节点之间的"二阶交互"通过 $A_{k,ij}$ 加权。模型经 $T$ 轮置换等变的消息传递后，把变量嵌入解码出预测矩阵 $\hat X$，与最小 Frobenius 范数最优解 $X^*$（命题 1.1 证明唯一）做监督。最后把 $\hat X$ 作为 PDHG 求解器的 warm-start 初始点，加速收敛。

### 关键设计

1. **不可能性结果（VC-WL / VC-2-WL 都不够）**:

    - 功能：界定"哪些 GNN 一定学不会 SDP"。
    - 核心思路：作者构造一族结构同构但最优解不同的 SDP 实例，证明 1-WL 等价的 VC-WL 在稳定着色后给出相同节点表示，因此任何基于它的 GNN 必给出相同输出，从而无法区分不同最优解。把分析推广到 VC-2-WL（标准 2-WL 的变种）后结论依然成立。
    - 设计动机：把 GNN 表达力研究中常用的"WL 同色 $\Rightarrow$ GNN 同输出"经典论证迁移到 SDP 设定，给出一个**清晰的下界**，警告后续工作不要再在 1-WL 类架构上浪费算力。

2. **充分性架构 VC-2-FWL（仿真 PDHG）**:

    - 功能：给出一个能学会线性 SDP 最优解的可实现架构。
    - 核心思路：颜色赋给节点对 $(u,v)\in V^2$，更新规则用 folklore 2-FWL：$c^t_{uv} := \text{hash}(c^{t-1}_{uv}, \{\!\{(c^{t-1}_{wv}, c^{t-1}_{uw}) \mid w\in V\}\!\})$。关键证明是把 PDHG 的一步迭代（包含矩阵-矩阵乘积、PSD 投影中的特征分解结构信息）显式映射为 VC-2-FWL 的若干 hash 步骤——只要嵌入维度足够，对偶变量和原变量的更新都能精确仿真。这意味着 VC-2-FWL 的稳定着色细于 PDHG 收敛轨迹的状态划分，从而具备拟合最优解的能力。
    - 设计动机：直接证"能逼近最优解"很难；改证"能仿真一个已知收敛到最优解的求解器"是 L2O 的标准技巧（Qian et al. IPM 工作的 SDP 版）。这让 VC-2-FWL 成为目前已知**最弱的能解 SDP 的 GNN 架构**。

3. **Warm-start 集成**:

    - 功能：把学到的预测 $\hat X$ 真正变成实用加速。
    - 核心思路：训练后用 $\hat X$ 作为 PDHG 的初始点 $X^{(0)}$，让经典求解器从离最优更近的位置出发，剩余的优化负担由 PDHG 来收敛精度。这避免了"GNN 输出未必严格可行"的部署难点：可行性和最优性由 PDHG 保证，GNN 只需提供好的起点。
    - 设计动机：纯神经求解器在精度要求高的科学计算场景没法用；warm-start 是把 ML 与传统优化结合的最稳妥范式，能把表达力上的 $\hat X \approx X^*$ 折合成"实际墙钟时间减少"的工程收益。

### 损失函数 / 训练策略
监督目标是预测矩阵与最小 Frobenius 范数最优解的 Frobenius 误差 $\|\hat X - X^*\|_F$，外加目标差距 $|\langle C, \hat X\rangle - \langle C, X^*\rangle|$ 作为辅助指标。训练数据是合成 SDP 实例（覆盖随机生成的对称矩阵族）和 SDPLIB 中各类 SDP（如 max-cut 松弛、$\theta$-函数等）。

## 实验关键数据

### 主实验
在合成 SDP 与 SDPLIB 多类基准上，VC-2-FWL 架构在预测误差、目标差距两项上**一致优于** VC-WL、VC-2-WL 等理论上更弱的基线。

| 设置 | 指标 | VC-WL | VC-2-WL | VC-2-FWL |
|------|------|------|------|------|
| 合成 SDP | $\|\hat X - X^*\|_F$ | 最高 | 中等 | 最低 |
| SDPLIB | 目标差距 | 最大 | 中等 | 最小 |
| Warm-start PDHG | 收敛时间减少 | 几乎无 | 一般 | **最多 80%** |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| VC-WL（1-WL 等价） | 拟合误差最高，部分实例完全无法区分 | 印证不可能性定理 |
| VC-2-WL（2-WL 等价） | 优于 VC-WL，但仍偏离最优 | 标准 2-WL 不够 |
| VC-2-FWL（2-FWL 等价） | 误差最低，目标差距最小 | 满足充分性 |
| 仅冷启动 PDHG | 基线收敛时间 | 对照 |
| VC-2-FWL warm-start | 收敛时间减少最高约 80% | 表达力 → 实用加速 |

### 关键发现
- 表达力层级与"实际拟合质量"严格对齐：VC-WL ⊏ VC-2-WL ⊏ VC-2-FWL 的理论关系在合成与真实基准上都翻译为预测误差的严格递减。
- 把"GNN 误差降一点"转成"求解器迭代少一截"的杠杆效应明显——只要预测落到最优解的吸引盆内，PDHG 的剩余迭代数会大幅减少。
- 错误模式分析显示 VC-WL 主要在"高度对称的约束矩阵族"上失败，正是 1-WL 区分能力的经典盲区，与理论预测一致。

## 亮点与洞察
- **首个 SDP-GNN 表达力刻画**：把 LP/QP 上成熟的"WL 层级即 GNN 表达力"框架推广到 SDP，填补了 L2O 理论的一大空白；这套分析模板可以直接迁移到 SOCP 之外的更广凸锥规划。
- **"仿真求解器即获得表达力"的范式漂亮**：不是直接证 $\hat X$ 收敛，而是证 GNN 能模拟 PDHG 一步——这把"逼近最优解"问题转换为"模拟离散动力系统"问题，更易处理。
- **理论指导架构的典型案例**：当未来有人想训神经 SDP 求解器，可以直接跳过 1-WL/2-WL 失败的"踩坑期"，从 2-FWL 等价架构起步。
- **有限门槛上界**：证明 VC-2-FWL “足以”不代表“必要”，但不可能性结果隐含 1-WL/2-WL 不够，这里面的 gap （3-WL？3-FWL？）留下后续可探究的机会。

## 局限与展望
- 实验主要集中在合成数据和中等规模 SDPLIB，**实际超大规模 SDP（如组合优化松弛中 $n>10^4$）下 VC-2-FWL 的内存成本（节点对数量 $O(n^2)$）很快爆炸**，尚未给出可扩展实现。
- 充分性证明仅针对"线性 SDP + 唯一最小范数最优解"，对解集非唯一、退化解、半无限 SDP 等情形并未覆盖。
- Warm-start 收益依赖 PDHG 作为下游求解器；与 IPM、ADMM 等更精确求解器的兼容性还需验证。
- 未来方向：稀疏化或子图采样使 VC-2-FWL 可扩展、把分析推广到二阶锥/混合整数 SDP、与现代低秩 SDP 算法结合。

## 相关工作与启发
- **vs Qian et al. (PDHG-GNN for LP)**：他们证明 1-WL 类 GNN 足以模拟 LP 的 PDHG，本文则证 SDP 必须升到 2-FWL，揭示了"凸锥维度"与"WL 层级"之间的对应。
- **vs Yau et al. (GNN for low-rank SDP relaxation of Max-CSP)**：对方把 GNN 作为 CSP 的近似算法分析，本文把一般线性 SDP 本身作为目标对象，研究 GNN 能否恢复最优矩阵解。
- **vs Chen et al. (GNN for QP / SOCP 表达力)**：本文延续他们"WL 层级 ↔ 凸优化解恢复"的研究线，把 SDP 的位置精确放进表达力地图。
- **vs Yau et al. (low-rank SDP 松弛上的 GNN)**：他们把 GNN 看作 CSP 近似算法，本文直接研究 GNN 能否恢复 SDP 原始最优矩阵解 $X^*$，调查对象不同。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把 SDP 解恢复挂到 WL 层级上的工作。
- 实验充分度: ⭐⭐⭐⭐ 合成 + SDPLIB 都覆盖，但缺超大规模与多求解器对比。
- 写作质量: ⭐⭐⭐⭐ 理论部分清晰，warm-start 工程实验细节可再充实。
- 价值: ⭐⭐⭐⭐ 对"神经凸优化"理论社区有标杆意义，工业落地仍需扩展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](../../NeurIPS2025/graph_learning/graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)

</div>

<!-- RELATED:END -->
