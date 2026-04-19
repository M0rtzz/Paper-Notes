---
title: >-
  [论文解读] Expert Divergence Learning for MoE-based Language Models
description: >-
  [ICLR 2026][LLM效率][混合专家] 解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。
tags:
  - ICLR 2026
  - LLM效率
  - 混合专家
  - 专家同质化
  - 路由多样性
  - Jensen-Shannon散度
  - 领域特化
---

# Expert Divergence Learning for MoE-based Language Models

**会议**: ICLR 2026  
**arXiv**: [2603.00054](https://arxiv.org/abs/2603.00054)  
**代码**: 未公开  
**领域**: LLM效率 / MoE  
**关键词**: 混合专家, 专家同质化, 路由多样性, Jensen-Shannon散度, 领域特化

## 一句话总结
解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

## 研究背景与动机
**领域现状**：混合专家模型（MoE）通过稀疏激活实现高参数量低计算量，但训练中经常出现"专家同质化"——不同专家学到高度相似的功能，浪费了参数容量。

**现有痛点**：现有方法（如负载均衡损失）只确保专家被均匀使用，但不保证不同专家学到不同技能。专家可能均匀使用但功能相同。

**核心矛盾**：负载均衡和功能特化是不同的概念——均匀使用不等于各有专长。

**核心 idea**：不同数据域应该激活不同的专家组合——通过最大化域间路由分布的 JS 散度来鼓励专家特化。

## 方法详解

### 整体框架
在标准 MoE 训练目标（语言建模损失 $\mathcal{L}_{LM}$ + 负载均衡损失 $\mathcal{L}_{LB}$）上增加专家散度损失 $\mathcal{L}_{ED}$：$\mathcal{L}_{final} = \mathcal{L}_{LM} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{ED}$。

### 关键设计
1. **三步聚合**：Token→Sequence→Domain 层次化聚合路由概率

    - Token 级：每个 token 经 router 得到 N 个专家的概率分布 $p(x_t)$
    - Sequence 级：$\bar{p}_s = \frac{1}{T}\sum_{t=1}^T p(x_t)$，平均每个序列的所有 token 分布
    - Domain 级：$\bar{p}_j = \frac{1}{|\mathcal{B}_j|}\sum_{s \in \mathcal{B}_j} \bar{p}_s$，按域标签分组平均
2. **JS 散度最大化**：$\mathcal{L}_{ED} = \frac{1}{\binom{M_B}{2}}\sum_{j<k} -\log(D_{JS}(\bar{p}_j || \bar{p}_k) + \epsilon)$

    - 最大化所有域对之间路由分布的 Jensen-Shannon 散度
    - 使用负对数放大小散度值的梯度，防止梯度消失
3. **域标签方案**：两种粒度

    - 3-Class：英语/中文/数学三大域（直接用数据来源）
    - 49-Class：用分类器将英文→24 主题、中文→24 主题、数学→1 个，共 49 个细粒度域

### 理论动机——多样性分配
- **分解定理（Proposition 1）**：总路由多样性 $D_{total} = D_{inter} + D_{intra}$
    - $D_{inter}$：域间散度——不同域使用不同专家的程度
    - $D_{intra}$：域内散度——同一域内 token 使用不同专家的程度
- **Proposition 2**：$\mathcal{L}_{ED}$ 直接增加 $D_{inter}$，将全局多样性重新分配到域间差异上
- 标准 $\mathcal{L}_{LB}$ 只关注 $D_{total}$ 而不管如何分配，$\mathcal{L}_{ED}$ 提供更精细的方向引导
- 两个损失协同：$\mathcal{L}_{LB}$ 确保总多样性，$\mathcal{L}_{ED}$ 引导多样性流向域间差异→专家特化

## 实验关键数据

### 主实验（三个模型规模，100B tokens 从头预训练）
| 模型 | 方法 | CEval | MMLU | CMMLU | ARC-e | ARC-c | RACE-m | RACE-h | 平均 |
|------|------|-------|------|-------|-------|-------|--------|--------|------|
| 15B-A1.5B | 标准 MoE | 28.0 | 25.8 | 25.6 | 47.4 | 28.2 | 50.5 | 43.6 | 35.59 |
| 15B-A1.5B | **+ED(49类)** | **28.9** | **27.1** | **26.3** | **48.6** | **28.5** | **51.7** | **45.5** | **36.65** |
| 8B-A0.8B | 标准 MoE | 25.8 | 24.5 | 25.0 | 43.2 | 23.6 | 42.7 | 36.5 | 31.61 |
| 8B-A0.8B | **+ED(49类)** | **26.1** | **25.2** | **25.2** | **44.1** | **24.9** | **44.3** | **38.2** | **32.57** |
| 3B-A0.3B | 标准 MoE | 23.8 | 23.1 | 24.2 | 35.0 | 22.6 | 37.8 | 32.1 | 28.37 |
| 3B-A0.3B | +ED(49类) | 24.5 | 23.4 | 24.5 | 36.2 | 22.8 | 37.5 | 32.8 | 28.81 |

### 训练动态与专家分析
| 分析维度 | 发现 |
|---------|------|
| LM 损失 | 所有 ED 配置收敛到更低的 $\mathcal{L}_{LM}$，不同 $\beta$ 均优于基线 |
| 域粒度 | 49 类 > 3 类 > 基线，细粒度域标签帮助更大 |
| 专家特化 | Layer 4 的特化程度远超其他层（中间层专家最分化） |
| 计算开销 | 额外训练开销可忽略（仅需每个 batch 计算域间散度） |
| 规模效应 | 性能增益随模型规模增大而增大（15B > 8B > 3B） |

### 关键发现
- 负载均衡 ≠ 功能特化：均匀使用不保证各有专长
- ED 损失引导专家开发不同域的路由策略，形成有组织的专家团队
- 49 类细粒度域分类比 3 类更有效，说明域标签的信息量直接影响特化质量

## 亮点与洞察
- **从均衡到特化的范式转变**：标准 MoE 训练关注负载均衡（$D_{total}$），本文关注功能特化（$D_{inter}$），是更本质的目标
- **域标签的利用**：利用预训练数据已有的域标签作为免费的监督信号来引导专家特化，零额外标注成本
- **JS 散度的选择**：对称且有界的 JS 散度比 KL 散度更适合衡量路由分布差异
- **理论清晰**：多样性分解定理优雅地揭示了 $\mathcal{L}_{LB}$ 和 $\mathcal{L}_{ED}$ 的互补关系

## 局限与展望
- 需要数据的域标签，纯无标签场景不直接适用（但可用分类器自动打标，如本文所做）
- 3B/8B/15B 三个模型尺度上验证，但训练规模有限（100B tokens）
- 域分类的粒度（49 vs 3）需要手工设定，最优粒度的自适应确定是开放问题
- 未探索与 shared expert 架构（如 DeepSeek-MoE）的交互效应
- 是否可以在预训练结束后通过域标签引导的微调来追加特化？

## 相关工作与启发
- **vs DeepSeek-MoE**：DeepSeek 用 shared expert 捕获共性来减轻路由专家冗余，EMET 用域间散度最大化直接引导路由专家分化——两者正交可组合
- **vs ERNIE 4.5**：ERNIE 用 router 权重矩阵的正交性（无监督），EMET 用域标签（有监督）引导特化——有监督方法更有效
- **vs Qiu et al. (global LB)**：global batch 负载均衡增强了整体多样性，EMET 进一步引导多样性的分配方向
- **启发**：MoE 的"divide and conquer"设计意图需要训练目标的显式支持，否则退化为"redundant generalists"

## 补充分析
- 核心洞察：load balancing 只鼓励全局路由多样性，不指导多样性如何分布——ℒ_ED 通过域标签将多样性定向分配为域间差异
- Divergence Decomposition ($D_{total} = D_{inter} + D_{intra}$) 非常优雅——ℒ_LB 促进 $D_{total}$，ℒ_ED 导向 $D_{inter}$
- 49-class 表现优于 3-class，暗示更细粒度域标签带来更精细专家分工
- 性能增益随模型规模正向增长（3B < 8B < 15B），更大模型有更多潜力被有效分工利用
- 计算开销几乎为零——ℒ_ED 仅在已有路由 logit 上计算 JSD

## 评分
- 新颖性: ⭐⭐⭐⭐ 专家特化 via 域间散度最大化是新颖的角度，理论分解优雅
- 实验充分度: ⭐⭐⭐⭐ 三个模型尺度+两种域分类粒度+专家行为分析
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，理论动机完整
- 价值: ⭐⭐⭐⭐ 对 MoE 训练有实际指导，域标签利用成本低

<!-- RELATED:START -->

## 相关论文

- [Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [Steering Language Models with Weight Arithmetic](steering_language_models_with_weight_arithmetic.md)
- [DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](../../AAAI2026/llm_efficiency/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)

<!-- RELATED:END -->
