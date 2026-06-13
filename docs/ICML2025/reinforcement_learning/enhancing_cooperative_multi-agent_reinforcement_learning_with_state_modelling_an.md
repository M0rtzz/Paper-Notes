---
title: >-
  [论文解读] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration
description: >-
  [ICML 2025][强化学习][多智能体强化学习] 提出 SMPE² 算法，通过变分推断学习有意义的状态信念表示并结合对抗式内在探索，在部分可观测的合作多智能体环境中显著提升协调能力，在 MPE、LBF、RWARE 三个基准上超越 SOTA。 在合作多智能体深度强化学习（MARL）中，智能体需要在分布式部分可观测环境下学…
tags:
  - "ICML 2025"
  - "强化学习"
  - "多智能体强化学习"
  - "部分可观测性"
  - "状态建模"
  - "对抗探索"
  - "内在奖励"
---

# Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration

**会议**: ICML 2025  
**arXiv**: [2505.05262](https://arxiv.org/abs/2505.05262)  
**代码**: [github.com/ddaedalus/smpe](https://github.com/ddaedalus/smpe)  
**领域**: 强化学习  
**关键词**: 多智能体强化学习, 部分可观测性, 状态建模, 对抗探索, 内在奖励

## 一句话总结

提出 SMPE² 算法，通过变分推断学习有意义的状态信念表示并结合对抗式内在探索，在部分可观测的合作多智能体环境中显著提升协调能力，在 MPE、LBF、RWARE 三个基准上超越 SOTA。

## 研究背景与动机

在合作多智能体深度强化学习（MARL）中，智能体需要在分布式部分可观测环境下学习协作策略。现实应用广泛，如多机器人协作、无线网络优化、自动驾驶和空中交通管理等。

**核心挑战**：在 CTDE（集中训练分散执行）框架下，训练时智能体可共享信息，但执行时只能依赖自身局部观测，且无显式通信通道。

**现有方法的不足**：

**智能体建模（AM）的表示与策略脱节**：标准 AM 学到的信念表示并非针对最大化价值函数优化，导致次优（如 LIAM、SoMM 等）

**未过滤冗余状态信息**：直接使用全局状态的压缩嵌入反而会损害性能，因为联合状态中存在对单个智能体不具信息量的冗余特征

**AM 未用于改善探索**：现有 AM 方法无法提升初始随机探索策略，在稀疏奖励环境中尤为低效

**实用性限制**：许多方法假设单一控制器、需要先验知识或执行时访问其他智能体的动作等

**本文聚焦两个关键问题**：
- Q1: 智能体能否仅从自身观测推断有意义的状态表示来增强协调？
- Q2: 能否利用这些表示高效探索状态空间以发现更优策略？

## 方法详解

### 整体框架

SMPE²（State Modelling for Policy Enhancement through Exploration）基于 MAA2C 构建，包含两大核心模块：

**1. 自监督状态建模**：每个智能体 $i$ 通过变分编码器-解码器（ED）从自身观测 $o^i$ 推断信念嵌入 $z^i$，用于重建其他智能体的（经过滤的）观测。引入 AM 过滤器 $w^i$ 自动过滤非信息性特征。

**2. 对抗式计数内在探索**：利用 $z^i$ 的 SimHash 计算内在奖励，鼓励智能体发现新颖状态。这形成对抗框架——智能体 $i$ 发现的新观测同时构成其他智能体 ED 的"对抗目标"，促进彼此的状态推断能力。

**架构组成**（参见 Figure 1）：
- **Actor（蓝色）**：部分可观测，策略为 $\pi_\psi(a_t^i \mid h_t^i, z_t^i)$，将信念嵌入显式融入策略网络
- **Critic（红色）**：两个 critic 均有全局可观测性
    - $V_\xi(s)$：标准 MAA2C critic
    - $V_k(\hat{s})$：在 AM 过滤器净化后的状态 $\hat{s} = o^i \oplus (w^i \cdot o^{-i})$ 上训练的 critic

### 关键设计

#### 1. 变分编码器-解码器（ED）与 AM 过滤器

每个智能体 $i$ 的 ED 包含：
- **编码器** $q_{\omega^i}(z^i \mid o^i)$：从自身观测推断高斯潜变量 $z^i$
- **解码器** $p_{\phi^i}(w^i \cdot o^{-i} \mid z^i)$：重建经过滤的其他智能体观测

**AM 过滤器**的设计至关重要。$w^i_j = \sigma(\phi^i_w(o^j))$ 为可学习的 sigmoid MLP，每个维度值在 $[0,1]$ 之间，控制其他智能体 $j$ 的每个观测特征对智能体 $i$ 的信息量贡献。

**非信息性特征的定义**（需同时满足）：
- 与智能体 $i$ 最大化未来奖励无关
- 由于部分可观测性和非平稳性，无法通过 $z^i$ 推断

**为什么 ED 只以 $o^i$ 为条件（而非历史 $h^i$）？** 因为 $z^i$ 被用于内在探索，应当奖励"新颖观测"而非"新颖轨迹"。如果以 $h^i$ 为条件，一个已充分探索轨迹中的新颖高价值观测会被过度压缩，无法获得足够的内在激励。策略网络本身已经以 $h_t^i$ 为条件，隐式利用了所有时间步的信念 $z_t^i$。

#### 2. 对抗式计数探索

采用基于 SimHash 的计数方法，但哈希域为 $z^i \in \mathcal{Z}$ 而非原始观测：

$$\hat{r}^i = \frac{1}{n(SH(z^i))}$$

其中 $n(SH(z^i))$ 为 $z^i$ 哈希值的访问计数。选择 SimHash 是因为它能将邻近的 $z^i$ 映射为邻近的哈希值，计算成本低。

**对抗性体现**：智能体 $i$ 被激励发现新颖 $o^i$（导致新颖 $z^i$），这些新观测同时成为其他智能体 ED 的未见过的重建目标——相当于为其他智能体提供"对抗样本"，帮助它们学习更强的状态推断能力。

**稳定性保证**：对 ED 参数 $(\omega^i, \phi^i)$ 进行周期性硬更新（每 $N_{ED}=2000$ 步），避免 $z^i$ 在连续训练回合间剧烈变化，确保内在奖励平稳递减至最小平台。

#### 3. 双 Critic 设计

- $V_\xi(s)$：标准 critic，为 actor 提供优势估计
- $V_k(\hat{s})$：在过滤状态 $\hat{s}$ 上训练的 critic，让 $w^i$ 能够针对策略优化学习过滤策略

消融实验表明单一 critic 会导致高方差和发散。

### 损失函数 / 训练策略

**状态表示学习的总损失**：

$$L_\text{encodings} = L_\text{critic}^w + \lambda_\text{rec} \cdot L_\text{rec} + \lambda_\text{norm} \cdot L_\text{norm} + \lambda_\text{KL} \cdot L_\text{KL}$$

各分项含义：

| 损失项 | 公式 | 作用 |
|-------|------|------|
| $L_\text{rec}$ | $\|\tilde{w}^i \cdot o^{-i} - w^i \cdot \hat{o}^{-i}\|^2$ | 自监督重建损失（带目标网络稳定化） |
| $L_\text{norm}$ | $-\|w^i\|_2^2$ | 防止 AM 过滤器退化为零 |
| $L_\text{KL}$ | $\text{KL}(q_{\omega^i}(z^i\|o^i) \| \mathcal{N}(0,I))$ | 使信念表示变分化、促进智能体间信念一致性 |
| $L_\text{critic}^w$ | $[r_t^i + V_{k'}^\pi(\hat{s}_{t+1}) - V_k^\pi(\hat{s}_t)]^2$ | 针对策略优化训练 AM 过滤器 |

**Actor 损失**：

$$L_\text{actor}(\psi^i) = -\beta_H \cdot H(\pi_{\psi^i}(a_t^i \mid h_t^i, z_t^i)) - \log \pi_{\psi^i}(a_t^i \mid h_t^i, z_t^i) \cdot (r_t^i + V_{\xi'}^\pi(s_{t+1}) - V_\xi^\pi(s_t))$$

**总损失**：$L_\text{SMPE} = L_\text{actor} + L_\text{critic} + L_\text{encodings}$

**关键超参数设置**（以 LBF 为例）：$\lambda_\text{rec}=1$, $\lambda_\text{norm}=0.1$, $\lambda_\text{KL}=1$，内在奖励系数 $=0.1$，ED 用三层 MLP（非 RNN），潜变量维度 $=32$。

## 实验关键数据

### 主实验

在三大基准上评估，对比 MAA2C、COMA、MAPPO、ATM、EOI、EMC、MASER 等方法。

| 基准/任务 | 指标 | SMPE² | 最强基线 | 表现 |
|-----------|------|-------|---------|------|
| MPE Spread-3/4/5/8 | 平均回合奖励 | 最优 | MAA2C/ATM | 随智能体数增加优势更明显 |
| LBF 2s-9x9-3p-2f | 平均回合奖励 | **完全解决** | 其他方法均为 0 | 解决了开放挑战 |
| LBF 4s-11x11-3p-2f | 平均回合奖励 | **完全解决** | 其他方法均为 0 | 解决了开放挑战 |
| LBF 2s-12x12-2p-2f | 平均回合奖励 | 最优 | EOI 次优 | 更快收敛 |
| LBF 7s-20x20-5p-3f | 平均回合奖励 | 持续领先 | MAA2C | 大网格持续优势 |
| RWARE tiny-2ag-hard | 平均回合奖励 | 最优 | EOI | 显著领先 |
| RWARE tiny-4ag-hard | 平均回合奖励 | 最优 | EOI | 显著领先 |
| RWARE small-4ag-hard | 平均回合奖励 | 最优 | MAA2C | EOI 在此完全失败 |

运行效率对比（LBF 2s-12x12-2p-2f）：

| 方法 | 运行时间 | 相对 SMPE² |
|------|---------|-----------|
| MAA2C | 37min | 0.5x |
| SMPE² | **1h13min** | 1x |
| ATM | 2h32min | 2x |
| EOI | 17h11min | 17x |
| MASER | 25h8min | 25x |
| EMC | 29h34min | 30x |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| SMPE²（完整） | 最优 | 全部组件协同 |
| no_intr（无内在奖励） | 完全失败 | 仅有状态建模不足以在稀疏奖励下发现高价值状态 |
| no_filters（无 AM 过滤器） | 显著下降 | 冗余信息干扰信念学习 |
| no_kl（无 KL 正则） | 明显下降 | 信念不一致、探索效率低 |
| no_L2_norm（无范数正则） | 下降 | $w^i$ 可能退化为零 |
| obs_rew（标准 SimHash 哈希观测） | 劣于 SMPE² | 哈希 $z^i$ 优于哈希原始观测 |
| no_critic_w（无第二个 critic） | 下降 | $w^i$ 无法针对策略优化学习 |
| SMPE_PPO（MAPPO 骨干） | 优于 MAPPO | 框架灵活，可迁移至其他骨干 |

### 关键发现

1. **探索与协调相辅相成**：仅有状态建模（no_intr）或仅有探索（标准方法）都不够，SMPE² 的成功在于两者的紧密结合
2. **t-SNE 可视化**：启用 $L_\text{KL}$ 时三个智能体的信念在大面积区域重叠（一致性 57.5%），禁用时完全分离（99.3%），证明 KL 正则促进信念一致性
3. **AM 过滤器的可解释性**：在 LBF 任务中，过滤器自动学会屏蔽不可见的和无关的特征（如不在视野内的食物位置），保留与协调相关的信息
4. **EMC 和 MASER 失败原因**：高度依赖初始随机策略，基于低价值数据生成误导性内在奖励或子目标

## 亮点与洞察

1. **状态建模与策略优化的统一**：证明了 $V_{SM}^* = V^*$（Proposition 2.1），状态建模目标不会约束最优策略空间，从理论上保证了框架的最优性
2. **对抗探索的隐式涌现**：无需显式设计对抗机制，通过鼓励发现新颖 $z^i$，自然形成智能体之间的互利"对抗"——你的新发现是我的训练素材
3. **自监督过滤的优雅设计**：AM 过滤器同时出现在损失函数的目标和预测中，形成自洽的自监督学习回路
4. **极高的计算效率**：相比 MASER 快 25 倍、EMC 快 30 倍，额外开销仅为 MAA2C 的约 2 倍
5. **解决了多个公认的开放挑战**：LBF 中 2s-9x9-3p-2f 和 4s-11x11-3p-2f 被此前工作标记为未解决的难题

## 局限与展望

1. **架构扩展**：目前 ED 使用 MLP，作者提出将 Transformer 融入架构可能进一步提升状态建模能力
2. **可扩展性**：重建目标随智能体数线性增长，虽然 AM 过滤器缓解了部分问题，但大规模场景下仍需验证
3. **随机环境适应性**：当前实验均为确定性动力学，论文未验证在带噪声观测或复杂动力学下的表现
4. **基准范围**：仅在 MPE、LBF、RWARE 上验证，缺少 SMAC（StarCraft Multi-Agent Challenge）等更复杂基准
5. **通信设定**：完全无通信假设，未与结合轻量通信的混合方法对比

## 相关工作与启发

- **LIAM** (Papoudakis et al., 2021)：ED 重建其他智能体信息但表示与策略脱节，本文通过联合优化解决
- **SIDE** (Xu et al., 2022)：推断状态但执行时不使用 $z^i$，SMPE² 显式利用信念嵌入增强策略
- **EMC** (Zheng et al., 2021)：基于好奇心的探索依赖 QMIX，在 LBF 上表现极差
- **MASER** (Jeon et al., 2022)：子目标生成依赖初始策略质量，稀疏奖励下产生误导性子目标
- **EOI** (Jiang & Lu, 2021)：通过多样性鼓励探索，在同质智能体场景（LBF）中不适用
- **MAVEN** (Mahajan et al., 2019)：探索联合行为空间但不鼓励发现新颖状态

**启发**：将信念表示同时服务于策略增强和探索驱动的思路可推广至其他需要"推断未观测信息"的场景，如 POMDP 中的单智能体任务、去中心化通信网络等。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 状态建模框架和对抗探索的结合很有创意，但各组件（VAE、SimHash、AM）本身并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ — 三大基准、大量消融、t-SNE 可视化、AM 过滤器可解释性分析、运行时间对比，极其充分
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰、理论与直觉兼顾，但符号较多导致部分推导可读性略差
- 价值: ⭐⭐⭐⭐⭐ — 解决了公认的开放挑战，方法高效可扩展，代码开源，对 MARL 社区价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets](adversarial_cooperative_rationalization_the_risk_of_spurious_correlations_in_eve.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](learning_progress_driven_multi-agent_curriculum.md)
- [\[ICML 2025\] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)

</div>

<!-- RELATED:END -->
