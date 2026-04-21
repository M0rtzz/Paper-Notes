---
title: >-
  [论文解读] In-Context Learning for Pure Exploration
description: >-
  [ICLR 2026][模型压缩][上下文学习] 提出 ICPE（In-Context Pure Exploration），一种结合监督学习和强化学习的上下文学习框架，使用 Transformer 从经验中直接学习探索策略，在主动序列假设检验/纯探索问题中实现接近最优的实例自适应算法性能，无需显式建模信息结构。
tags:
  - ICLR 2026
  - 模型压缩
  - 上下文学习
  - 纯探索
  - 假设检验
  - Best Arm Identification
  - Transformer
---

# In-Context Learning for Pure Exploration

**会议**: ICLR 2026  
**arXiv**: [2506.01876](https://arxiv.org/abs/2506.01876)  
**代码**: 有（附论文）  
**领域**: 强化学习 / 在线学习  
**关键词**: 上下文学习, 纯探索, 假设检验, Best Arm Identification, Transformer

## 一句话总结

提出 ICPE（In-Context Pure Exploration），一种结合监督学习和强化学习的上下文学习框架，使用 Transformer 从经验中直接学习探索策略，在主动序列假设检验/纯探索问题中实现接近最优的实例自适应算法性能，无需显式建模信息结构。

## 研究背景与动机

主动序列假设检验（也称纯探索）中，agent 需要主动控制数据收集过程以高效识别正确假设。该问题广泛存在于医疗诊断、图像识别、推荐系统等领域。当前方法面临三大挑战：

**归纳偏置难以编码**：设计自适应探索策略需要对问题结构有深刻理解，但在隐含信息结构未知时尤为困难

**RL 方法的局限性**：当相关信息结构未被充分表示时，传统 RL 方法往往表现不佳

**BAI 方法的限制**：Best Arm Identification 等经典方法虽然理论优雅，但通常依赖显式的建模假设，且在复杂环境（如 MDP）中优化问题变为非凸

核心问题：能否让 agent **自主**从经验中发现和利用隐含结构来增强探索效率？

## 方法详解

### 整体框架

ICPE 采用双网络架构：
- **推断网络 $I$**（Inference Network）：通过监督学习训练，从当前数据中推断真实假设
- **探索网络 $\pi$**（Exploration Network）：通过 RL 训练，选择动作以最大化推断网络的准确性

两个网络都使用 Transformer 架构，将数据轨迹 $\mathcal{D}_t = (x_1, a_1, \ldots, x_t)$ 作为序列输入。

### 关键设计

1. **问题建模为 MDP**：

    - 状态 $s_t = (\mathcal{D}_t, \emptyset_{t:N})$，包含历史数据轨迹和填充 token
    - 动作空间 $\mathcal{A}$（包含 stop 动作用于固定置信度设置）
    - 设计动机：将纯探索问题转化为 RL 可求解的形式

2. **固定置信度设置（Fixed Confidence）**：

    - 目标：在满足 $\mathbb{P}(\hat{H}_\tau = H^*) \geq 1 - \delta$ 的约束下最小化停止时间 $\tau$
    - 通过对偶问题求解：$\min_{\lambda \geq 0} \max_{I, \pi} V_\lambda(\pi, I)$
    - 奖励设计：$r_\lambda(z) = -1 + d \cdot \lambda \log I_{\bar{\phi}}(H^* | s')$，其中 $d$ 为终止指示符
    - 包含专门的 stop 动作，其 Q 值可在任意状态回溯更新

3. **固定预算设置（Fixed Horizon）**：

    - 目标：在给定预算 $N$ 步内最大化正确识别概率
    - 奖励仅在最后一步给出：$r_N = h(\hat{H}_N; M)$
    - 不包含 stop 动作

4. **多时间尺度优化**：

    - 最慢时间尺度：更新对偶变量 $\lambda$
    - 中间时间尺度：监督学习优化推断网络 $I_\phi$（交叉熵损失）
    - 最快时间尺度：DQN + Replay Buffer 优化策略网络 $Q_\theta$
    - 使用目标网络 $Q_{\bar{\theta}}$ 和 $I_{\bar{\phi}}$ 保持稳定性

### 损失函数 / 训练策略

- 推断网络：交叉熵损失 $-\log I_\phi(H^* | s_\tau)$
- 策略网络：TD 损失 + stopping action 损失
- Transformer 架构：3 层、2 注意力头、隐藏维度 256、GELU 激活、GPT-2 配置
- 训练使用 Adam 优化器，学习率 $10^{-4}$ 到 $10^{-6}$

## 实验关键数据

### 主实验

**1. 确定性 Bandit（固定预算）**

| K（动作数） | ICPE 正确率 | DQN | Uniform | I-DPT |
|-----------|-----------|-----|---------|-------|
| 4-20 | ≈1.0 | 逐渐下降 | 快速下降 | 中等 |

- ICPE 自发学会了"每个动作恰好选一次"的最优策略

**2. 随机 Bandit（固定置信度，$\delta=0.1$）**

| K | ICPE 平均停止时间 | TaS | TTPS | Uniform |
|---|----------------|-----|------|---------|
| 4-14 | 最低 | 中等 | 中等 | 最高 |

- ICPE 在样本复杂度上接近理论下界

**3. Magic Action Bandit（隐含信息结构）**

| $\sigma_m$ | ICPE | I-IDS | 理论下界 |
|-----------|------|-------|---------|
| 0.0-1.0 | 接近下界 | 明显更高 | - |

- ICPE 在所有噪声水平下均优于 I-IDS

**4. MNIST 像素采样**

| 方法 | 准确率 | 平均采样区域数 |
|------|--------|-------------|
| ICPE | 最高 | 更少 |
| Deep CMAB | 中等 | 较多 |
| Uniform | 最低 | 相同 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定置信度 vs 固定预算 | 固定置信度更优 | stop 动作引入了课程学习效果 |
| ICPE 策略 vs 近似 TaS | 总变差有差异 | ICPE 利用了先验信息 |
| 类别特定采样 | ICPE 显示最多变化 | 卡方检验证实数字间策略显著不同 |

### 关键发现

1. **ICPE 自发发现最优策略**：在确定性 bandit 中学会每个动作恰好选一次，在二分搜索任务中学会 $O(\log_2 K)$ 的搜索策略
2. **在有隐含结构的环境中优势最大**：Magic action 环境中，ICPE 能发现并利用信息链，而 IDS 等基于贪心信息增益的方法无法做到
3. **固定置信度中的 stop 动作起关键作用**：相当于一种课程学习，使 agent 学会适应问题难度
4. **策略具有实质性的上下文适应能力**：在 MNIST 任务中，不同数字类别的采样策略显著不同

## 亮点与洞察

- **双网络设计的优雅性**：推断网络 $I$ 提供奖励信号给探索网络 $\pi$，形成良性循环——$I$ 变好 → 奖励信号更准确 → $\pi$ 学到更好的探索策略 → 数据更有信息量 → $I$ 进一步改善
- **算法发现能力**：ICPE 在二分搜索任务中自动发现了概率版本的二分搜索算法（停止时间精确匹配 $\log_2 K$）
- **IDS 非最优的理论证明**（Theorem B.1）：在有 magic action 的结构化环境中，贪心信息增益策略（IDS）是次优的，因为它无法做长程规划
- **连接认知科学**：ICPE 的双网络架构类似于认知地图（探索网络）+ 目标导向评估（推断网络）

## 局限与展望

1. **有限的假设空间 $\mathcal{H}$**：当前假设 $\mathcal{H}$ 是有限集，需要扩展到连续情况（主动回归）
2. **依赖先验分布 $\mathcal{P}(\mathcal{M})$**：需要假设任务分布已知且静态
3. **Oracle 假设**：训练时需要完美验证器，实际中可能不可用
4. **Transformer 的水平线限制**：受限于固定的最大视野 $N$
5. **扩展性**：当前在小规模问题上验证，扩展到更大问题需要架构和训练方面的改进
6. 可探索与 LLM 的结合，利用语言先验来辅助探索

## 相关工作与启发

- **与 RL² 的关系**：类似地将策略表示为 RNN/Transformer 的隐藏状态，但目标不同（识别 vs 累积奖励）
- **与 ICEE 的区别**：ICEE 处理探索-利用权衡（返回条件学习），ICPE 专注于纯识别目标
- **与 Track-and-Stop 的联系**：ICPE 学到的策略在某些情况下与 TaS 相似，但能利用先验信息做得更好
- 启发：上下文学习能力 + 序列建模 = 自动化的算法设计平台

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （将 ICL 引入纯探索问题，双网络设计优雅）
- 实验充分度: ⭐⭐⭐⭐⭐ （从简单 bandit 到 MNIST 到 MDP，层层递进）
- 写作质量: ⭐⭐⭐⭐ （理论和实验结合好，但论文较长）
- 价值: ⭐⭐⭐⭐⭐ （为主动假设检验提供了通用的深度学习框架）

<!-- RELATED:START -->

## 相关论文

- [Modality-free Graph In-context Alignment](modality-free_graph_in-context_alignment.md)
- [Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/model_compression/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](../../CVPR2026/model_compression/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)
- [Disentangling Latent Shifts of In-Context Learning with Weak Supervision](../../NeurIPS2025/model_compression/disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)
- [MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/model_compression/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)

<!-- RELATED:END -->
