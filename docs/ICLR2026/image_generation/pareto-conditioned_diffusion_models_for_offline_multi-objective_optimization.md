---
title: >-
  [论文解读] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization
description: >-
  [ICLR 2026][图像生成][离线多目标优化] 提出 Pareto-Conditioned Diffusion (PCD)，将离线多目标优化重构为条件采样问题，直接以目标权衡为条件生成高质量解，无需显式代理模型，在多种基准上实现最佳一致性。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "离线多目标优化"
  - "条件扩散模型"
  - "Pareto 前沿"
  - "无代理模型"
  - "参考方向"
---

# Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization

**会议**: ICLR 2026  
**arXiv**: [2602.00737](https://arxiv.org/abs/2602.00737)  
**代码**: [GitHub](https://github.com/jatan12/PCD)  
**领域**: 图像生成  
**关键词**: 离线多目标优化, 条件扩散模型, Pareto 前沿, 无代理模型, 参考方向

## 一句话总结

提出 Pareto-Conditioned Diffusion (PCD)，将离线多目标优化重构为条件采样问题，直接以目标权衡为条件生成高质量解，无需显式代理模型，在多种基准上实现最佳一致性。

## 研究背景与动机

- **离线 MOO 挑战**：仅有静态数据集，无法查询真实目标函数
- **现有方法依赖代理模型**：DNN 或 GP 近似目标函数 → MOEA 搜索 → 代理精度瓶颈
- **生成模型方法（如 ParetoFlow）仍依赖代理预测器引导**，继承了代理模型的不准确性风险
- **核心想法**：直接将 MOO 建模为条件生成任务 $p(\boldsymbol{x} | \boldsymbol{y}; \sigma)$

## 方法详解

### 整体框架

PCD 统一了方案生成和 Pareto 前沿建模：训练条件扩散模型 → 以目标向量为条件采样新方案。

### 1. 多目标重加权策略

基于 dominance number 的分箱加权：

$$w_i = \frac{|B_i|}{|B_i| + K} \exp\left(\frac{-\frac{1}{|B_i|}\sum_{j=1}^{|B_i|} o(\boldsymbol{x}_{b_j})}{\tau}\right)$$

其中 $o(\boldsymbol{x}) = \sum_{\boldsymbol{x}' \in \mathcal{D}} \mathbb{I}[\boldsymbol{f}(\boldsymbol{x}) \prec \boldsymbol{f}(\boldsymbol{x}')]$ 为 dominance number。

两个期望性质：
1. 包含更多数据点的箱获得更高权重（更可靠）
2. 平均表现更好的箱获得更高权重（更重要）

### 2. 参考方向条件点生成

受 NSGA-III 启发的三步流程：
1. **方向向量生成**：用 Riesz s-Energy 方法生成 $L$ 个方向向量 $\boldsymbol{w}_i$
2. **点-方向配对**：按非支配排序迭代分配数据点到最近方向向量
3. **外推 + 高斯扰动**：将代表点沿方向外推，加零均值高斯噪声增加多样性

### 3. Classifier-Free Guidance 采样

修改 ODE：

$$d\boldsymbol{x}/d\sigma = -(\gamma D_\theta(\boldsymbol{x}; \hat{\boldsymbol{y}}, \sigma) + (1-\gamma) D_\theta(\boldsymbol{x}; \sigma) - \boldsymbol{x})/\sigma$$

$\gamma > 1$ 增强条件目标的影响，引导样本到与 $\hat{\boldsymbol{y}}$ 一致的区域。

### 训练目标

重加权条件去噪 $L_2$ 损失：

$$\theta = \arg\min_\theta \mathbb{E} [w(\boldsymbol{y}) \lambda(\sigma) \|D_\theta(\boldsymbol{x} + \boldsymbol{n}; \boldsymbol{y}, \sigma) - \boldsymbol{x}\|_2^2]$$

## 实验关键数据

### 跨任务平均排名（100th percentile HV, ↓ 越低越好）

| 方法 | 合成 | MORL | RE | Scientific | MONAS | 总平均 |
|------|------|------|-----|-----------|-------|-------|
| $\mathcal{D}$(best) | 5.45 | **1.70** | 2.60 | 9.35 | 11.53 | 7.43 |
| ParetoFlow | **2.44** | 8.50 | 1.74 | 9.05 | 11.19 | 6.74 |
| MM + IOM | 5.16 | 12.70 | 5.76 | 4.40 | 5.77 | 5.80 |
| E2E | 6.16 | 9.70 | 6.06 | 4.20 | 5.13 | 5.71 |
| **PCD** | 3.38 | 5.50 | **1.51** | **4.05** | 7.54 | **4.80** |

### 消融实验：组件贡献

| 变体 | ZDT2 | MO-Swimmer | RE34 | Regex | C10/MOP2 |
|------|------|------------|------|-------|----------|
| Ideal + N/A | 7.59 | 1.76 | 9.19 | 5.60 | 10.46 |
| Ref.Dir. + N/A | 7.89 | 3.53 | 10.11 | 5.55 | 10.47 |
| Ref.Dir. + Pruning | 5.64 | 3.63 | 10.16 | 4.20 | 10.55 |
| **PCD (完整)** | 6.25 | **3.69** | **10.17** | 4.80 | **10.59** |

### 关键发现

1. PCD 使用单一固定超参数组在所有任务类别上实现最佳总体排名
2. 参考方向机制在 MO-Swimmer 上将 HV 提升近一倍（1.76→3.53）
3. 重加权策略一致优于简单剪枝（Xue et al., 2024 的方法）
4. 引导尺度 $\gamma$ 的增益有限（2.5 已接近饱和），因为重加权已偏置了数据分布

## 亮点与洞察

1. **端到端框架**：将多阶段管线（代理+搜索）简化为单一条件生成模型
2. **跨任务一致性**：这是 PCD 最显著的优势——在连续、离散、分类任务上均表现稳健
3. **NSGA-III 启发的条件点生成**：巧妙结合了进化算法的方向向量思想和扩散模型的条件生成

## 局限性

- MORL 任务（~10,000 维参数空间）因 MLP 去噪器直接操作参数空间而受限
- MONAS 纯类别搜索空间对连续扩散模型构成挑战
- 未处理组合优化任务（如 TSP）
- 重加权在数据质量本身较好的数据集上可能反而有害

## 相关工作

- **代理模型方法**：COMs, ICT, IOM, Tri-Mentoring
- **生成模型方法**：ParetoFlow, LaMBO, MOGFNs
- **条件扩散**：DDOM, MINs, Reward-Directed Diffusion

## 评分

- 新颖性：⭐⭐⭐⭐ — 将离线 MOO 重构为条件采样是自然但有效的贡献
- 技术深度：⭐⭐⭐⭐ — 重加权策略和参考方向机制设计合理
- 实验完整性：⭐⭐⭐⭐⭐ — 覆盖 5 大类基准，对比 13 种基线方法
- 实用价值：⭐⭐⭐⭐ — 超参数鲁棒性使实际部署更可行

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](../../CVPR2026/image_generation/mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](../../ICML2026/image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[ICLR 2026\] Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions](temporal_concept_dynamics_in_diffusion_models_via_prompt-conditioned_interventio.md)
- [\[ICLR 2026\] Intention-Conditioned Flow Occupancy Models](intention-conditioned_flow_occupancy_models.md)

</div>

<!-- RELATED:END -->
