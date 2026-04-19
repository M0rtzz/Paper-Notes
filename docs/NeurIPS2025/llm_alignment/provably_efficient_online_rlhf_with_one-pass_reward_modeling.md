---
title: >-
  [论文解读] Provably Efficient Online RLHF with One-Pass Reward Modeling
description: >-
  [NeurIPS 2025][LLM对齐][online RLHF] 提出一种基于 online mirror descent（OMD）的 one-pass reward modeling 方法，消除了 online RLHF 中需要存储历史数据并重新从头优化的计算瓶颈，实现每次迭代 $\mathcal{O}(1)$ 的时间和存储复杂度，同时在统计效率上也优于 MLE 方法。
tags:
  - NeurIPS 2025
  - LLM对齐
  - online RLHF
  - reward modeling
  - online mirror descent
  - contextual dueling bandit
  - computational efficiency
---

# Provably Efficient Online RLHF with One-Pass Reward Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2502.07193](https://arxiv.org/abs/2502.07193)  
**代码**: [github.com/ZinYY/Online_RLHF](https://github.com/ZinYY/Online_RLHF)  
**领域**: llm_alignment  
**关键词**: online RLHF, reward modeling, online mirror descent, contextual dueling bandit, computational efficiency

## 一句话总结

提出一种基于 online mirror descent（OMD）的 one-pass reward modeling 方法，消除了 online RLHF 中需要存储历史数据并重新从头优化的计算瓶颈，实现每次迭代 $\mathcal{O}(1)$ 的时间和存储复杂度，同时在统计效率上也优于 MLE 方法。

## 研究背景与动机

RLHF（基于人类反馈的强化学习）是对齐 LLM 的核心技术。传统 RLHF 依赖固定偏好数据集，但覆盖有限导致 reward model 对分布外样本泛化困难。**Online RLHF** 通过迭代收集数据和持续改进模型来解决此问题，Claude 和 LLaMA-2 都已论证其有效性。

然而 online RLHF 面临严重的计算瓶颈：
- 每轮迭代需将新数据加入历史数据集，并在**整个数据集**上重新优化 reward model
- 使用 MLE 估计时，第 $t$ 轮的计算复杂度为 $\mathcal{O}(t \log t)$，存储为 $\mathcal{O}(t)$
- 这在长期迭代中（尤其是边缘设备上）变得不可承受

核心问题：**能否设计既有统计效率又有计算效率的 online RLHF 算法？**

## 方法详解

### 整体框架

将 RLHF 形式化为 **contextual preference bandit** 问题。采用 Bradley-Terry 偏好模型：

$$\mathbb{P}[y=1 \mid x, a, a'] = \sigma(\phi(x,a)^\top \theta^* - \phi(x,a')^\top \theta^*)$$

其中 $\phi(x,a)$ 为已知特征映射，$\theta^*$ 为未知参数，$\sigma$ 为 sigmoid 函数。定义非线性系数 $\kappa = \max 1/\dot{\sigma}(\cdot)$，刻画学习难度（$\kappa$ 可能指数级大）。

### 关键设计：One-Pass Reward Modeling

**传统 MLE 的问题**：第 $t$ 轮需求解

$$\hat{\theta}_{t+1} = \arg\min_{\theta} \sum_{i=1}^t \ell_i(\theta)$$

需要遍历所有历史数据，每轮 $\mathcal{O}(t \log t)$ 计算量。

**本文方案**：用 OMD + 定制 local norm 的二阶 Taylor 展开替代 MLE：

$$\widetilde{\theta}_{t+1} = \arg\min_{\theta \in \Theta} \left\{ \langle g_t(\widetilde{\theta}_t), \theta \rangle + \frac{1}{2\eta} \|\theta - \widetilde{\theta}_t\|_{\widetilde{\mathcal{H}}_t}^2 \right\}$$

其中 local norm $\widetilde{\mathcal{H}}_t = \mathcal{H}_t + \eta H_t(\widetilde{\theta}_t)$，$\mathcal{H}_t = \sum_{i=1}^{t-1} H_i(\widetilde{\theta}_{i+1}) + \lambda I$。

关键等价形式（closed-form 解）：

$$\widetilde{\theta}_{t+1}' = \widetilde{\theta}_t - \eta \widetilde{\mathcal{H}}_t^{-1} g_t(\widetilde{\theta}_t), \quad \widetilde{\theta}_{t+1} = \text{Proj}_\Theta(\widetilde{\theta}_{t+1}')$$

这一更新只依赖当前样本，无需存储历史数据 → **每轮 $\mathcal{O}(1)$**。

**设计精髓**：local norm $\mathcal{H}_t$ 捕捉二阶信息（近似 Hessian），是确保统计效率的核心。标准 OMD 用一阶近似会牺牲收敛率，本文的二阶 Taylor 展开兼顾了 closed-form 和统计效率。

### 三种 Online RLHF 场景

**场景 1：Passive Data Collection** — 算法不控制数据收集，采用「pessimism in the face of uncertainty」原则：

$$\widetilde{J}_{T+1}(\pi) = \Phi^\top \widetilde{\theta}_{T+1} - \widetilde{\beta}_{T+1} \|\Phi\|_{\mathcal{H}_{T+1}^{-1}}$$

**场景 2：Active Data Collection** — 算法选择不确定性最大的样本查询：

$$(x_{t+1}, a_{t+1}, a_{t+1}') = \arg\max \|\phi(x,a) - \phi(x,a')\|_{\mathcal{H}_{t+1}^{-1}}$$

**场景 3：Deployment-Time Adaptation** — 在线部署中平衡利用与探索，第一个动作最大化预估奖励，第二个动作同时最大化奖励和与第一个动作的距离。

### 实用化技术

- **Hessian-Vector Product (HVP) + 共轭梯度**：将 $\mathcal{O}(d^3)$ 的 Hessian 逆运算降至 $\mathcal{O}(d)$
- **Rejection Sampling** 近似模型不确定性：采样 $n$ 个回答，用 reward ranking 替代精确的 confidence bound
- $\lambda$ 采用时间递增调度 $\lambda_t = \lambda_0 \cdot \min(1, f(t/T))$ 近似累积 Hessian 效应

### 损失函数 / 训练策略

- 基座模型：Llama-3-8B-Instruct 和 Qwen2.5-7B-Instruct
- 特征维度 $d = 4096$（最后一层 embedding）
- 数据集：Ultrafeedback-binarized（64K prompts）和 Mixture2
- Passive 设置：随机采样 $T = 30,000$ 数据点
- Active 设置：从全量数据中仅选择 6,400 样本

## 实验关键数据

### 主实验

**Passive Data Collection（Llama-3-8B, Ultrafeedback）**：

| 指标 | MLE | Ours (OMD) |
|------|-----|------------|
| 收敛速度 | 慢 | 快（尤其 $T < 10,000$ 时优势显著） |
| 评估准确率 | 低 | 高 |
| 评估 loss | 高 | 低 |

在小样本区域（$T < 10,000$），OMD 方法以更少样本达到更高评估准确率，验证了统计效率的提升。

**Active Data Collection（仅 6,400 样本）**：

| 方法 | ACC (%) | 训练时间 (s) |
|------|---------|-------------|
| Rand-MLE | 69.51 ± 0.5 | 4876 ± 47 |
| Active-MLE | 69.82 ± 0.4 | 4982 ± 52 |
| Rand-OMD | 68.97 ± 0.6 | 1456 ± 31 |
| **Ours** | **70.43 ± 0.3** | **1489 ± 36** |

OMD 方法训练时间约为 MLE 方法的 **1/3**，同时准确率更高。主动数据选择进一步提升了性能。

### 消融实验

**Deployment-Time Adaptation**：将数据分为 20 个 chunk 顺序处理，对比多种动作选择策略：
- 本文策略（best + top-1/q percentile）在 MLE 和 OMD 两种基准上都优于随机选择、best+second-best、best+worst 策略
- Win rate 分析表明 OMD 方法与 MLE 方法竞争力相当，但计算成本大幅降低

### 关键发现

1. OMD 方法在统计效率上至少改进 $\sqrt{\kappa}$ 倍（理论保证），$\kappa$ 可能指数级大
2. 训练时间减少约 3 倍，同时准确率持平或更高
3. 存储需求从 $\mathcal{O}(T)$ 降至 $\mathcal{O}(1)$，对边缘设备部署意义重大
4. 主动数据选择 + OMD 是最优组合：准确率最高、速度最快

## 亮点与洞察

1. **理论与实践的优秀统一**：从 contextual dueling bandit 的理论分析出发，得到实际可部署的 LLM 算法
2. **三个场景的统一处理**：passive/active/deployment 三种设置都从同一 OMD 框架自然推导
3. **$\mathcal{O}(1)$ 复杂度的实际意义**：online RLHF 的核心瓶颈是 reward model 的持续更新，本文将其复杂度从随迭代次数线性增长变为常数
4. **Local norm 设计精妙**：用 lookahead 点的 Hessian 构建 local norm，既保二阶信息又有 closed-form，设计巧妙
5. **HVP + 共轭梯度的实用化**：从 $\mathcal{O}(d^3)$ 到 $\mathcal{O}(d)$，使 4096 维特征空间下的实时更新成为可能

## 局限与展望

1. 假设固定特征映射 $\phi$ — 实际中 LLM fine-tuning 过程中特征空间会变化
2. 基于 Bradley-Terry 模型 — 未扩展到 Plackett-Luce 等更一般的偏好模型
3. 线性奖励函数假设 — 现实中 reward function 通常非线性，需要神经网络近似
4. 实验仅在最后一层 embedding 上操作 — 未探讨多层特征或 full fine-tuning 的效果
5. Rejection sampling 近似不确定性的理论保证较弱 — 连接理论和实践的 gap 仍存在

## 相关工作与启发

- **Online RLHF (Dong et al., Guo et al.)**: 证明了在线迭代训练优于离线，但计算成本高
- **Faury et al. (Implicit OMD for logistic bandits)**: 提供了 OMD 方法的理论基础，本文扩展到 RLHF 场景
- **Zhang & Sugiyama**: 二阶 Taylor 展开近似思想的来源
- **Das et al. (Active RLHF)**: Active data collection 设置的 baseline，本文在保持相同统计保证的前提下大幅提升计算效率
- **Foster et al.**: 关注枚举指数级大的回答空间的问题，与本文关注的迭代计算成本互补

**启发**：OMD + local norm 的组合可推广到其他在线学习问题（如在线 DPO、在线 preference optimization），对边缘设备上的个性化对齐特别有价值。

## 评分

- 新颖性: ⭐⭐⭐⭐ OMD + local norm 在 RLHF 中的应用新颖，但基础技术来自 bandit 文献
- 实验充分度: ⭐⭐⭐⭐ 覆盖三种场景，有理论和实验双重验证，但规模有限（Llama-3 8B）
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，算法描述清晰，场景分类井然有序
- 价值: ⭐⭐⭐⭐ 解决了 online RLHF 的核心计算瓶颈，对边缘部署和持续学习有实际意义

<!-- RELATED:START -->

## 相关论文

- [Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)
- [ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](../../ICML2025/llm_alignment/can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)
- [Reward Generalization in RLHF: A Topological Perspective](../../ACL2025/llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)
- [Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)

<!-- RELATED:END -->
