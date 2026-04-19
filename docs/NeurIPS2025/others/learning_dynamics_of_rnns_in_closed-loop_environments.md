---
title: >-
  [论文解读] Learning Dynamics of RNNs in Closed-Loop Environments
description: >-
  [NeurIPS 2025][RNN学习动力学] 从数学理论上揭示了 RNN 在闭环（agent-环境交互）与开环（监督学习）训练下呈现根本不同的学习动力学，闭环学习遵循三阶段过程，由短期策略改进与长期稳定性之间的竞争驱动。
tags:
  - NeurIPS 2025
  - RNN学习动力学
  - 闭环学习
  - 开环vs闭环
  - 控制理论
  - 内部表征
---

# Learning Dynamics of RNNs in Closed-Loop Environments

**会议**: NeurIPS 2025  
**arXiv**: [2505.13567](https://arxiv.org/abs/2505.13567)  
**代码**: [GitHub](https://github.com/yoavger/closed_loop_rnn_learning_dynamics) (有)  
**领域**: 理论 / 循环神经网络 / 控制论  
**关键词**: RNN学习动力学, 闭环学习, 开环vs闭环, 控制理论, 内部表征

## 一句话总结

从数学理论上揭示了 RNN 在闭环（agent-环境交互）与开环（监督学习）训练下呈现根本不同的学习动力学，闭环学习遵循三阶段过程，由短期策略改进与长期稳定性之间的竞争驱动。

## 研究背景与动机

**领域现状**：RNN 广泛用于神经科学建模和序列任务。现有理论工作主要分析 RNN 在开环（监督学习）设置下的学习动态和最终解的性质。

**现有痛点**：真实的生物学习发生在闭环环境中（动物的动作影响后续输入），但闭环 RNN 的学习动态理论几乎空白。

**核心矛盾**：开环分析假设 i.i.d. 输入，忽略了反馈环路；而闭环环境中输出影响下一步输入，使得学习动态本质不同。

**本文目标**：建立闭环 RNN 学习动态的数学理论，揭示闭环学习为何以及如何不同于开环学习。

**切入角度**：选择经典的二阶积分器控制任务（double integrator），通过线性化 RNN 和秩-1 连接权重假设获得可解析的理论框架。

**核心 idea**：闭环 RNN 的学习由联合 agent-环境系统的特征值演化主导，学习过程分三个阶段展开，本质上受短期策略改进与长期系统稳定性的竞争驱动。

## 方法详解

### 整体框架

研究框架包含：
- **环境**：离散时间二阶积分器（位置-速度控制任务），仅观测位置（部分可观测）
- **Agent**：$N=100$ 神经元的 RNN，用策略梯度训练
- 构建联合系统矩阵 $\bm{P}$ 将 RNN 和环境统一建模

### 关键设计

1. **联合闭环系统**：将环境状态 $\bm{x}_t$ 和 RNN 隐状态 $\bm{h}_t$ 合并为联合状态 $\bm{s}_t = (\bm{x}_t, \bm{h}_t)^\top$，得到线性动力系统：
   $$\bm{s}_{t+1} = \bm{P} \bm{s}_t, \quad \bm{P} = \begin{bmatrix} \bm{A} & \bm{B}\bm{z}^\top \\ \bm{m}\bm{C}\bm{A} & \bm{W} \end{bmatrix}$$
   系统稳定性由 $\bm{P}$ 的特征值决定。

2. **有效低维系统**：在秩-1 连接权重假设下（$\bm{W} = \bm{u}\bm{v}^\top$），隐状态被限制在 $\bm{m}$ 和 $\bm{u}$ 张成的子空间中，系统降维为 4 维，由 4 个标量序参量（overlap）控制：$\sigma_{\bm{z}\bm{m}}, \sigma_{\bm{z}\bm{u}}, \sigma_{\bm{v}\bm{m}}, \sigma_{\bm{v}\bm{u}}$。

3. **有效反馈增益**：将高维非线性 RNN 策略嵌入到 2D 可解释空间 $(k_1, k_2)$ 中：
   $$u_t \approx -k_1 x_t^{(1)} - k_2 x_t^{(2)}$$
   通过闭环矩阵 $\bm{M}_{\text{cl}} = \bm{A} - \bm{B}\bm{K}$ 分析稳定性区域。

### 三阶段学习动态

**阶段 1 - 负位置策略**：
- 损失快速下降，RNN 学会 $u_t \propto -\text{position}$ 的比例控制策略
- 特征多项式简化为 $\chi_{\bm{P}}(\lambda) = \lambda^2 - 2\lambda + (1 - \sigma_{\bm{z}\bm{m}})$
- 非对称损失面驱使 $\sigma_{\bm{z}\bm{m}}$ 收敛到小的负值
- 此时系统不稳定（$\rho(\bm{P}) > 1$），表现为振荡发散

**阶段 2 - 构建世界模型**：
- 损失进入平台期，RNN 需要学习推断隐变量（速度）
- 引入代理损失 $\mathcal{L}_{\text{surrogate}} = \alpha \cdot \mathcal{L}_\infty + (1-\alpha) \cdot \mathcal{L}_2$
- 短期控制目标（$\mathcal{L}_2$）与长期稳定性目标（$\mathcal{L}_\infty$）的梯度方向近乎相反，产生锯齿形轨迹
- 阶段结束标志：主特征值进入单位圆（系统稳定化）

**阶段 3 - 策略精化**：
- 损失再次下降，轨迹变为快速非振荡
- 第三个实特征值 $\lambda_3$ 增长，出现第二慢模态
- 低维有效模型能精确复现此阶段动态

## 实验关键数据

### 主实验：闭环 vs 开环学习

作者比较了架构完全相同、初始化相同的两个 RNN，分别在闭环和开环模式下训练：

| 训练模式 | 初始行为 | 中期表现 | 最终性能 |
|----------|---------|---------|---------|
| 闭环 | 类似 → 进入平台期 | 三阶段渐进 | 稳定收敛 |
| 开环 | 类似 → loss尖峰 | 闭环测试loss急剧恶化 | 最终恢复但路径不同 |

关键发现：两者在有效反馈增益 $(k_1, k_2)$ 空间中走过完全不同的轨迹。

### 多频追踪任务验证

| 特征 | 观察结果 |
|------|---------|
| 学习阶段 | 阶梯式loss下降，每级对应一个频率成分的习得 |
| 频率习得顺序 | 低频→高频，与人类运动控制实验一致 |
| 竞争现象 | 习得新频率时，已掌握频率的性能暂时下降 |

### 消融实验

- 线性 vs 非线性 RNN：非线性 RNN 定性上表现出相同的三阶段动态
- 不同 episode 长度 $T$：短 $T$ 使特征值沿虚轴上升，长 $T$ 使特征值下降——验证了短期/长期竞争理论
- 使用 Adam 优化器：锯齿形轨迹被缓解（因为自适应优化缓解了梯度方向冲突）

### 关键发现

- 闭环与开环训练即使架构、初始化完全相同，也会产生本质不同的学习轨迹
- 闭环学习的平台期不是"卡住"，而是系统在构建内部世界模型的必经阶段
- 跟踪联合 agent-环境系统的特征值（而非仅 RNN 的特征值）是理解闭环学习的充要条件

## 亮点与洞察

- **理论贡献突出**：首次建立了闭环 RNN 学习动态的解析理论，填补了重要空白
- **简洁深刻的物理图像**：短期策略改进 vs 长期稳定性的竞争统一解释了平台期、锯齿轨迹等现象
- **与神经科学的联系**：RNN 习得频率的顺序与人类运动学习实验惊人吻合，暗示共享的归纳偏置
- **低维有效模型**：将 100 维 RNN 动态压缩到 4 个标量序参量，保留了关键学习动态

## 局限与展望

- 理论分析依赖线性化和秩-1 权重简化，向完全非线性设置的推广尚需更多工作
- 仅使用直接梯度计算（策略梯度），未涉及稀疏奖励或 actor-critic 等更复杂的 RL 算法
- 有效系统仅在 episode 内保持精确的谱等价性，跨 episode 的学习动态尚未完全刻画
- 控制任务较简单（二阶积分器），更高维度、非线性环境的适用性有待探索

## 相关工作与启发

- 与 Saxe et al. 2013 的前馈网络学习动态分析类似，本文是闭环 RNN 的对应理论
- 与 Bordelon et al. 2025 对开环 RNN 学习动态的分析互补
- 对理解 RL 训练中的"阶段性突破"现象提供了理论视角（类似 grokking？）
- 启发：设计 RL 算法时应考虑短期/长期目标的平衡，可能通过课程学习或episode长度调度来缓解竞争

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次建立闭环RNN学习动态的数学理论，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ 理论验证详实，多频任务扩展有说服力，但任务复杂度受限
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导优雅，图表精美，物理直觉清晰
- 价值: ⭐⭐⭐⭐ 对理解闭环学习和生物学习有深远意义，但直接应用价值有限

<!-- RELATED:START -->

## 相关论文

- [Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback](../../ACL2025/others/dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)
- [Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [RNNs Perform Task Computations by Dynamically Warping Neural Representations](rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)
- [Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)
- [FlashMD: Long-Stride, Universal Prediction of Molecular Dynamics](flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)

<!-- RELATED:END -->
