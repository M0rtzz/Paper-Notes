---
title: >-
  [论文解读] FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models
description: >-
  [NeurIPS 2025][联邦强化学习] 提出 FedRAIN-Lite 联邦强化学习框架，将 RL 智能体分配到不同纬度带学习局部气候参数化策略并定期全局聚合，在层次化理想能量平衡模型上验证 DDPG 在热带和中纬度区域可将面积加权 RMSE 降低 50% 以上，为 RL 扩展到全尺度 GCM 提供了可行路径。
tags:
  - NeurIPS 2025
  - 联邦强化学习
  - 气候模型
  - 参数化
  - 能量平衡模型
  - DDPG
---

# FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models

**会议**: NeurIPS 2025  
**arXiv**: [2508.14315](https://arxiv.org/abs/2508.14315)  
**代码**: [GitHub](https://github.com/p3jitnath/climate-rl-fedrl)  
**领域**: reinforcement_learning  
**关键词**: 联邦强化学习, 气候模型, 参数化, 能量平衡模型, DDPG

## 一句话总结
提出 FedRAIN-Lite 联邦强化学习框架，将 RL 智能体分配到不同纬度带学习局部气候参数化策略并定期全局聚合，在层次化理想能量平衡模型上验证 DDPG 在热带和中纬度区域可将面积加权 RMSE 降低 50% 以上，为 RL 扩展到全尺度 GCM 提供了可行路径。

## 研究背景与动机

**领域现状**：气候模型中的亚网格参数化（如辐射、对流、湍流）传统上采用静态参数，通过离线调优拟合观测数据。这种方法开销大、缺乏适应性，难以应对气候状态的动态变化。

**现有痛点**：Nath et al. (2024) 的 RAIN 框架证明了 RL 在理想气候模型参数化中的可行性，但其将整个系统视为单一智能体，缺乏空间分解和区域适应性，无法扩展到真实 GCM 的空间网格结构。

**核心矛盾**：真实 GCM 天然具有空间分解结构（物理模块按纬度带运行），而现有 RL 方法将全球作为整体建模，无法利用区域特异性也无法并行化。

**本文目标**：如何让 RL 参数化策略镜像 GCM 的空间分解结构？不同纬度带的智能体如何协调学习？哪种 RL 算法最适合这种场景？

**切入角度**：采用联邦学习范式——每个纬度带对应一个本地智能体，周期性通过 FedAvg 聚合策略网络参数，既保留区域特化能力又维持全局一致性。

**核心 idea**：用联邦强化学习的空间分解策略映射 GCM 架构，实现区域自适应的气候参数学习。

## 方法详解

### 整体框架
基于 Budyko-Sellers 能量平衡模型（EBM）构建三层递进的 climateRL 环境：ebm-v1（单智能体全局输入）→ ebm-v2（多智能体全局输入+局部奖励）→ ebm-v3（多智能体局部输入+局部奖励，模拟 GCM 结构）。在 ebm-v2/v3 上使用 Flower 框架实现联邦 RL，每 $K$ 个 episode 同步聚合一次。

### 关键设计

1. **Budyko-Sellers EBM 及其 RL 化**:

    - 功能：将纬度分辨的能量平衡方程转化为 RL 环境
    - 核心思路：EBM 方程 $C(\phi)\frac{\partial T_s}{\partial t} = (1-\alpha(\phi))Q(\phi) - (A + BT_s) + \frac{D}{\cos\phi}\frac{\partial}{\partial\phi}(\cos\phi \frac{\partial T_s}{\partial\phi})$ 中，OLR 系数 $A$ 和 $B$ 作为 RL 智能体的策略输出，状态为纬度分辨温度剖面 $T_s(\phi)$，奖励为与目标气候学的 MSE
    - 设计动机：$A$、$B$ 是物理上最关键且最难静态标定的参数，将其动态化是提升模型精度的自然切入点

2. **三层递进环境设计**:

    - **ebm-v1**：单智能体观测完整 96 纬度带温度、输出全局 $\{A_\phi, B_\phi\}$——集中式基线
    - **ebm-v2**：多智能体（2 或 6 区域），每个智能体接收完整温度剖面但只优化区域奖励，通过 FedRL 定期聚合
    - **ebm-v3**：每个智能体只接收本区域的温度切片输入——完全模拟 GCM 的去中心化、部分可观测结构
    - 设计动机：通过逐步增加空间复杂度验证 FedRL 的鲁棒性和可扩展性

3. **联邦聚合策略**:

    - 功能：控制局部适应与全局同步的频率
    - 核心思路：使用 Flower 实现同步 FedAvg，测试 fed05（每 5 个 episode 聚合）、fed10（每 10 个 episode）和 nofed（无聚合）三种方案
    - 设计动机：频繁聚合（fed05）有助于全局稳定性，但可能抑制区域特化；稀疏聚合（fed10）给区域更多自由度但可能导致分歧

4. **RL 算法选择**:

    - 对比 DDPG、TD3、TQC 三种连续控制算法
    - DDPG 始终表现最优：对超参数最鲁棒、计算成本最低、跨场景泛化最好
    - TD3 和 TQC 偶尔可比但方差更大，尤其在赤道和极地区域不稳定

### 训练策略
每种配置用 10 个随机种子训练，报告 95% 置信区间。策略网络架构和超参数在 ebm-v1/v2/v3 之间保持一致，确保可比性。

## 实验关键数据

### 主实验

| 环境 | 算法 | 聚合方式 | 收敛速度 | 热带 RMSE 改善 | 整体表现 |
|------|------|---------|---------|---------------|---------|
| ebm-v1 | DDPG | — | >10K 步 | 基线 | 单智能体基线 |
| ebm-v2 | DDPG | fed05 | 2.5K-5K 步 | >50% | **最佳** |
| ebm-v2 | DDPG | fed10 | 2.5K-5K 步 | 30-40% | 方差较大 |
| ebm-v2 | DDPG | nofed | 2.5K-5K 步 | 20-30% | 不稳定 |
| ebm-v3 | DDPG | fed05 | 2.5K-5K 步 | >50% | 接近 ebm-v2 |
| ebm-v3 | TD3 | fed05 | 不稳定 | 变化大 | 方差大 |
| ebm-v3 | TQC | fed05 | 不稳定 | 变化大 | 赤道区域不稳定 |

### 消融实验

| 设计维度 | 发现 | 影响程度 |
|---------|------|---------|
| 空间分解粒度（2agent vs 6agent） | 两者均有效，6agent 在热带改善更显著 | 中等 |
| 聚合频率（fed05 vs fed10 vs nofed） | fed05 >> fed10 > nofed | 核心因素 |
| 输入范围（全局 vs 局部切片） | ebm-v3 局部输入的区域特化更强 | 积极 |
| 静态基线 vs RL | RL（尤其 DDPG+fed05）全面超越静态 climlab | 显著 |

### 关键发现
- **DDPG 是最稳健的选择**：超参数鲁棒性高、计算效率好、在所有空间配置下均能稳定收敛
- **频繁聚合（fed05）至关重要**：在热带 30°S-0° 和 0°-30°N 区间 areaWRMSE 降低超过 50%
- **ebm-v3 的区域输入有助于特化**：局部观测反而让智能体学到更好的区域策略
- **极地区域**所有方案表现相当——可能需要不同的建模策略
- 收敛加速明显：FedRL 在 2.5K-5K 步稳定，而单智能体需要 >10K 步

## 亮点与洞察
- **GCM 物理结构 ↔ 联邦学习架构的优美映射**：纬度带→本地智能体、全局气候一致性→FedAvg 聚合。这种结构匹配让 ML 和领域知识自然融合
- **层次化实验设计**：从 v1→v2→v3 递进验证，既建立了基线又最终打通了到 GCM 的路径
- **DDPG 的"返璞归真"**：在气候参数化这种平滑、低维连续控制任务中，最简单的 DDPG 比更复杂的 TD3/TQC 更可靠，说明算法选择需匹配问题特性

## 局限与展望
- **仅在理想 EBM 上验证**，未接入真实 GCM——从 EBM 到 GCM 的跨越仍是巨大工程挑战
- **物理复杂度有限**：EBM 只有辐射+扩散输运，缺少对流、云物理等关键过程
- **RL 动作空间仅限 A 和 B 两个参数**，真实模型需要调节数十个参数
- 聚合策略仅测试了简单的 FedAvg，异构方法（如 FedProx）可能更适合极地-热带差异大的场景
- 未考虑非平稳气候（如全球变暖过程中参数应如何随时间演化）

## 相关工作与启发
- **vs RAIN (Nath et al., 2024)**：RAIN 是本文的前置工作，用单智能体 RL 做全局参数化；本文将其扩展为多智能体联邦版本，引入空间分解维度
- **vs EnKI (Dunbar et al., 2021)**：集合卡尔曼反演做贝叶斯参数标定，但是批量离线方法，缺乏在线适应能力；本文的 RL 方法支持实时交互学习
- 该工作展示了 RL+FL 在科学计算中的潜力，类似思路可迁移到海洋模型、大气化学模型等具有空间分解结构的领域

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 FedRL 应用于气候参数化，物理-ML 架构的映射设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三层递进环境+三种算法+三种聚合策略+10 seeds，系统且严谨
- 写作质量: ⭐⭐⭐⭐ 物理背景交代充分、实验可视化清晰，附录详尽
- 价值: ⭐⭐⭐⭐ 为 RL 在气候科学中的实际落地铺路，方向意义大于当前技术贡献

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Improving Planning and MBRL with Temporally-Extended Actions](improving_planning_and_mbrl_with_temporally-extended_actions.md)

<!-- RELATED:END -->
