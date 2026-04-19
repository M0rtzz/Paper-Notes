---
title: >-
  [论文解读] Diversity by Design: Leveraging Distribution Matching for Offline Model-Based Optimization
description: >-
  [ICML 2025][离线模型优化] DynAMO 在离线模型优化 (MBO) 中将设计多样性建模为分布匹配问题，通过对抗训练使生成设计的分布捕捉离线数据集中的固有多样性，在多个科学领域显著提升候选方案的多样性而不牺牲质量。
tags:
  - ICML 2025
  - 离线模型优化
  - 设计多样性
  - 分布匹配
  - 对抗训练
  - MBO
---

# Diversity by Design: Leveraging Distribution Matching for Offline Model-Based Optimization

**会议**: ICML 2025  
**arXiv**: [2501.18768](https://arxiv.org/abs/2501.18768)  
**代码**: 无  
**领域**: Optimization / Generative Models  
**关键词**: 离线模型优化, 设计多样性, 分布匹配, 对抗训练, MBO

## 一句话总结
DynAMO 在离线模型优化 (MBO) 中将设计多样性建模为分布匹配问题，通过对抗训练使生成设计的分布捕捉离线数据集中的固有多样性，在多个科学领域显著提升候选方案的多样性而不牺牲质量。

## 研究背景与动机
**领域现状**: 离线 MBO 旨在根据离线数据集提出新设计以最大化奖励函数，广泛应用于药物设计、材料发现、蛋白质工程等科学领域。

**现有痛点**: 现有 MBO 方法主要关注发现单个最高奖励的设计，但在科学应用中，一组多样的高质量候选方案远比单一最优解更有价值——因为实际验证昂贵，多样性提供了更多选择。

**核心矛盾**: 质量 vs 多样性——优化奖励会使所有候选收敛到同一区域，丧失多样性。

**本文切入**: 将多样性显式建模为优化目标，而非仅靠隐式正则化。

**核心 idea**: 将设计多样性形式化为分布匹配——生成设计的分布应捕捉离线数据中高分区域的固有多样性结构。

## 方法详解

### 整体框架
输入：离线数据集 $\{(x_i, y_i)\}$ → 训练代理模型 (proxy) → DynAMO: 优化生成器以同时最大化奖励 + 匹配高分数据的分布 → 输出：多样且高质量的设计候选集。

### 关键设计

1. **分布匹配的多样性目标**:

    - 不仅生成高奖励设计，还要求生成分布匹配数据集中高分设计的分布
    - 使用对抗训练框架：判别器区分生成设计和高分真实设计
    - 设计动机：数据集中高分设计的分布自然编码了"有多少种好的方式"——即固有多样性

2. **DynAMO 优化框架**:

    - 生成器 $G$：$\min_G -\mathbb{E}[\text{proxy}(G(z))] + \lambda \cdot D_{\text{adversarial}}(p_G, p_{\text{data}}^{\text{high}})$
    - 判别器 $D$：标准 GAN 判别器训练
    - $\lambda$ 控制质量-多样性权衡
    - 设计动机：对抗损失比 MLE 更适合高维分布匹配

3. **与现有 MBO 方法的兼容性**:

    - DynAMO 是一个"插件"——可以与 CbAS, BONET, Grad 等现有方法结合
    - 在原有优化目标上添加分布匹配正则项
    - 设计动机：不替换而是增强现有方法

### 损失函数 / 训练策略
- 生成器：奖励最大化 + 对抗分布匹配损失
- 判别器：标准二元分类损失
- 交替训练生成器和判别器

## 实验关键数据

### 主实验
| 领域 | 指标 | DynAMO + 基线 | 基线 | 多样性提升 |
|------|------|-------------|------|----------|
| 蛋白质设计 | 多样性 + 奖励 | ✓ 更多样 | 收敛单一 | 显著 |
| 分子设计 | 多样性 + 奖励 | ✓ 更多样 | 收敛单一 | 显著 |
| 材料设计 | 多样性 + 奖励 | ✓ 更多样 | 收敛单一 | 显著 |
| 质量损失 | 最高奖励 | 微小下降 | — | 可接受 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 DynAMO | 最优多样性 | 分布匹配有效 |
| 无对抗损失 | 多样性下降 | 退化为标准 MBO |
| 不同 $\lambda$ | 权衡曲线 | 可调节质量-多样性平衡 |
| 不同基线 MBO | 一致提升 | DynAMO 作为插件通用有效 |

### 关键发现
- DynAMO 可与多种 MBO 方法组合，一致地提升多样性
- 多样性提升的代价（奖励下降）很小——因为数据中本就存在多种高分配置
- 对抗分布匹配比简单的正则化（如添加噪声）更有效
- 在不同科学领域的跨任务泛化性良好

## 亮点与洞察
- **多样性的形式化**: 将"多样性"从模糊概念转化为可优化的分布匹配目标
- **离线数据利用**: 利用数据集本身编码的多样性结构，而非凭空创造多样性
- **插件式设计**: 不替换现有方法，而是增强它们——提升了实用性和采用门槛

## 局限与展望
- 对抗训练的不稳定性可能影响结果质量
- "高分数据"的阈值选择影响匹配目标
- 当离线数据的高分区域本身不多样时，方法可能退化
- 52 页论文可能降低可读性（大量附录）

## 相关工作与启发
- Design-Bench (Trabucco et al. 2022) 提供 MBO 标准基准
- CbAS, BONET 等是主流 MBO 方法
- 生成多样性文献：DPP、多样性正则化
- 启发：科学发现不仅需要找到最好的方案，更需要探索多种可能性

## 评分
- 新颖性: ⭐⭐⭐⭐ 在 MBO 中显式优化多样性的首次系统化研究
- 实验充分度: ⭐⭐⭐⭐ 多领域验证 + 多基线组合 + 消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，52页详尽
- 价值: ⭐⭐⭐⭐ 对科学发现场景中的 MBO 有实际意义

<!-- RELATED:START -->

## 相关论文

- [Score Matching with Missing Data](score_matching_with_missing_data.md)
- [Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)

<!-- RELATED:END -->
