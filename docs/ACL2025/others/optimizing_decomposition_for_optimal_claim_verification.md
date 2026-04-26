---
title: >-
  [论文解读] Optimizing Decomposition for Optimal Claim Verification
description: >-
  [ACL 2025][Decompose-Then-Verify] 提出动态分解（Dynamic Decomposition）框架，通过强化学习从验证器反馈中学习分解策略，将声明（claim）分解为验证器偏好的原子性粒度，弥合分解器与验证器之间的性能差距。
tags:
  - ACL 2025
  - Decompose-Then-Verify
  - atomicity
  - 强化学习
  - PPO
  - fact-checking
---

# Optimizing Decomposition for Optimal Claim Verification

**会议**: ACL 2025  
**arXiv**: [2503.15354](https://arxiv.org/abs/2503.15354)  
**代码**: [github.com/yining610/dynamic-decomposition](https://github.com/yining610/dynamic-decomposition)  
**领域**: 事实核查 / 文本分解 / 强化学习  
**关键词**: Decompose-Then-Verify, atomicity, reinforcement learning, PPO, fact-checking  

## 一句话总结

提出动态分解（Dynamic Decomposition）框架，通过强化学习从验证器反馈中学习分解策略，将声明（claim）分解为验证器偏好的原子性粒度，弥合分解器与验证器之间的性能差距。

## 研究背景与动机

### 问题背景
Decompose-Then-Verify 范式是当前事实核查系统的主流方法：先用分解器（decomposer）将复杂声明拆分为子声明（subclaims），再用验证器（verifier）逐一验证。然而，现有工作通常将分解和验证视为独立模块，忽略了二者之间的交互和潜在错位。

### 核心发现与动机
- 作者引入了"原子性"（atomicity）指标来量化子声明的信息密度，定义为 $\text{atomicity} = \log_2(\text{\# atomic information})$
- 实验发现**不同的验证器在不同的原子性水平上达到最优验证置信度**，即每个验证器有自己偏好的输入粒度
- 现有的基于 prompt 的分解策略（如 FActScore 使用 8 个标注 demo）无法生成最优原子性的子声明，导致验证结果次优
- 例如 FActScore 的分解策略生成 atomicity=0 的子声明，但 Inst-Llama-7B 配合检索策略时在 atomicity=1 表现最佳

### 问题本质
这是一个**双层优化问题**（bilevel optimization）：上层优化验证准确率，下层优化分解策略。该问题是强 NP-hard 的。

## 方法详解

### 整体框架
将动态分解建模为有限 MDP（马尔可夫决策过程），使用 PPO（Proximal Policy Optimization）风格的 A2C 强化学习来近似求解双层优化问题。

**与现有方法的关键区别**：现有方法只调用一次分解 prompt，而动态分解**迭代地**产生分解调用，在每一步由策略决定是否继续分解当前子声明。

### MDP 定义 $M = (\mathcal{S}, \mathcal{A}, \kappa, r)$

**状态（Atomicity State）**：
- 每个状态是一个 $d$ 维向量 $s_t \in \mathbb{R}^d$，反映当前子声明列表的整体原子性
- 使用 GRU 建模状态转移：$s_{t+1} = \text{GRU}[s_t, (1+\sigma(\Delta\text{Info}))\text{Enc}(\{c_j\})]$
- $\Delta\text{Info}$ 通过 CPMI（条件逐对互信息）差异量化分解导致的局部原子性变化

**动作（Action）**：
- 二元动作空间：1（分解）或 0（不分解）
- 从策略分布 $a_t \sim \pi_d(a_t|s_t)$ 中采样

**奖励（Reward）**：
- 定义**验证置信度** $\text{Conf}(c, \mathcal{V}, \pi_v) = |P_{\mathcal{V}}(\text{True}|c, \pi_v) - P_{\mathcal{V}}(\text{False}|c, \pi_v)|$
- 奖励 = 分解后子声明的平均置信度 - 分解前原声明的置信度
- 实验证实置信度与准确率高度相关（Pearson's r = 0.88）

### 关键设计

**广度优先分解顺序**：
- 采用广度优先策略优先分解高原子性的子声明
- 避免深度优先导致的原子性方差过大问题
- 新生成的子声明按 FIFO 排队

**二元分解**：每步对子声明做二元分解，与 atomicity 的 log₂ 定义一致，确保最大化探索子声明空间

### 训练（PPO 目标函数）

$$L^{\text{PPO}} = \mathbb{E}_t\left[L^{\text{clip}} - c_1 \hat{A}_t^2 + c_2 S[\pi_d](s_t)\right]$$

- 使用 GAE（Generalized Advantage Estimator）计算优势函数
- 策略网络和价值网络均为两层 MLP
- 总可训练参数仅 **4.73M**
- 熵奖励项促进探索

## 实验

### 实验设置
- **数据集**：基于 FActScore 构建，包含 ChatGPT 和 PerplexityAI 两个来源的声明，每个声明有 6 种原子性水平（-1 到 4）
- **分解 LLM**：Llama3-Inst-70B、DeepSeek-V3
- **验证 LLM**：FT-T5-3B、Inst-Llama-7B、Llama3-Inst-8B
- **验证策略**：Retrieval / In-Context Example / No-Context
- **基线**：FActScore、FActScore-Atom、WICE、R-ND

### 主实验结果

| 指标 | DyDecomp 提升 |
|------|-------------|
| 验证置信度（平均） | **+0.07**（atomicity 1-2） |
| 验证准确率（PerplexityAI，平均） | **+0.12**（atomicity 1-2） |

- DyDecomp 在 atomicity=1 和 atomicity=2 上**一致性地**达到最高验证置信度
- 在 atomicity=0（已接近该验证器最优）时，DyDecomp 不一定优于基线，符合预期

### 消融实验（atomicity=4 上评估）

| 变体 | 验证置信度 |
|------|-----------|
| DyDecomp（完整） | 0.446 |
| - 单层 NN | 0.398（-0.048） |
| - 三元分解替代二元 | 0.424（-0.022） |
| - 移除熵奖励 | 0.356（-0.090） |
| - 移除 atomicity 1 训练数据 | 0.353（-0.093） |
| - 移除 atomicity 1,2,3 训练数据 | 0.401（-0.045） |

### 关键发现
1. 熵奖励项对性能影响最大（-0.090），说明探索多样分解轨迹至关重要
2. 跨原子性训练数据对泛化能力非常重要
3. 置信度提升不一定转化为准确率提升——取决于验证器能力（"木桶效应"）
4. PerplexityAI 数据集上置信度和准确率同步提升，ChatGPT 数据集上仅置信度提升

## 亮点与洞察

1. **原子性概念的形式化**：首次系统性地用 atomicity 量化子声明粒度对验证的影响，揭示不同验证器有不同的最优粒度偏好
2. **极轻量策略**：仅需 4.73M 参数即可显著改善验证效果，不需修改分解 LLM 或验证 LLM
3. **双层优化 + RL 的创新建模**：将分解-验证问题建模为双层优化，并用 PPO 近似求解
4. **通用兼容性**：框架兼容任何现有的事实核查系统，分解和验证 LLM 均保持冻结
5. **置信度与准确率的强相关性**（r=0.88）为无标签场景下的策略优化提供了可行的代理信号

## 局限性

1. 仅关注信息密度（atomicity）这一分解特征，未考虑子声明的可验证性、自包含性等其他特征
2. 奖励设计依赖验证置信度而非准确率，可能存在过度优化置信度但不提升准确率的风险
3. 验证器能力成为系统瓶颈——弱验证器限制了整体性能上限

## 相关工作

- **分解策略**：FActScore (Min et al., 2023) 精确性导向、WICE (Kamoi et al., 2023) 覆盖性导向、R-ND (Wanner et al., 2024) 等静态 prompt 方法
- **验证策略**：检索证据、构造 in-context 示例、零样本提示等
- **NLP 中的 RL**：用 RL 优化 ICL 示例选择、用 PPO 联合训练奖励和策略模型等

## 评分

⭐⭐⭐⭐ — 问题定义清晰，方法优雅（轻量 RL 策略 + 冻结 LLM），实验充分跨多个验证器/数据集/原子性水平。局限在于仅关注原子性一个维度，且置信度-准确率不完全对齐的问题值得进一步探索。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)
- [\[ACL 2025\] RePanda: Pandas-powered Tabular Verification and Reasoning](repanda_pandas-powered_tabular_verification_and_reasoning.md)
- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)

<!-- RELATED:END -->
