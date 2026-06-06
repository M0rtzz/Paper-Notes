---
title: >-
  [论文解读] Optimized Learned Count-Min Sketch
description: >-
  [NeurIPS 2025 (Workshop: ML for Systems)][Count-Min Sketch] 提出 OptLCMS，通过将分数空间分区并用 KKT 条件解析求解 CMS 参数、动态规划优化阈值，大幅加速构建过程，同时提供不可容忍误差概率的理论保证。
tags:
  - "NeurIPS 2025 (Workshop: ML for Systems)"
  - "Count-Min Sketch"
  - "频率估计"
  - "学习型数据结构"
  - "动态规划"
  - "KKT条件"
---

# Optimized Learned Count-Min Sketch

**会议**: NeurIPS 2025 (Workshop: ML for Systems)  
**arXiv**: [2512.12252](https://arxiv.org/abs/2512.12252)  
**代码**: 无  
**领域**: 数据结构 / 学习增强系统  
**关键词**: Count-Min Sketch, 频率估计, 学习型数据结构, 动态规划, KKT条件

## 一句话总结

提出 OptLCMS，通过将分数空间分区并用 KKT 条件解析求解 CMS 参数、动态规划优化阈值，大幅加速构建过程，同时提供不可容忍误差概率的理论保证。

## 研究背景与动机

Count-Min Sketch (CMS) 是一种概率数据结构，用于在内存高效的条件下估计多重集中元素的频率。它在网络流量监控、搜索查询统计等场景中广泛应用。CMS 存在**误差与内存**之间的固有权衡：减少内存会增加误差，降低误差则需要更多内存。

Learned CMS (LCMS) 通过引入机器学习模型来生成近似频率分数，利用阈值决定元素存储方式（字典 vs CMS），在固定内存下实现更低的估计误差。但 LCMS 有两个关键问题：

**构建速度慢**：需要在验证数据上进行经验参数调优（穷举搜索阈值和 CMS 参数）

**缺乏理论保证**：无法提供关于"不可容忍误差"概率（估计误差超过用户指定阈值的概率）的理论上界

## 方法详解

### 整体框架

OptLCMS 基于分区学习型 CMS（PL-CMS）进行扩展。整体结构包含三个组件：

- **预训练的 ML 模型**：为每个元素生成频率分数
- **G 个 CMS 表**：对应 G 个分数区间，每个组有独立的参数 $(\epsilon_g, \delta_g)$
- **唯一桶（UB）**：用于精确计数高频元素

元素根据模型分数被路由到不同组件：高分数元素进入 UB 精确计数，其他元素进入对应分数组的 CMS。

### 关键设计

**优化问题建模**：在固定内存预算 $M$ 下，最小化不可容忍误差概率的上界：

$$\min_{\boldsymbol{t}, \boldsymbol{\epsilon}, \boldsymbol{\delta}} \sum_{g=1}^{G} \delta_g q_g$$

约束条件包括：
- 内存约束：所有 CMS 表加 UB 的内存等于预算 $M$
- CMS 约束：$\delta_g \leq 1$
- 全局误差一致性：$\epsilon_g = \frac{\epsilon N}{N_g}$，确保每个 CMS 的允许误差与全局一致

**δ 的解析解（KKT 条件）**：当阈值 $\boldsymbol{t}$ 固定时，问题对 $\boldsymbol{\delta}$ 是凸的，可通过 KKT 条件得到闭式解：

$$\delta_g = \min\left\{1, \frac{1}{q_g \epsilon_g} \exp\left[-\frac{\frac{M-cn}{be} - I}{\sum_{g \in \mathcal{D}} \frac{1}{\epsilon_g}}\right]\right\}$$

其中 $I = \sum_{g \in \mathcal{D}} \frac{1}{\epsilon_g} \ln(q_g \epsilon_g)$，$\mathcal{D}$ 是 $\delta_g < 1$ 的组集合。

**阈值优化（动态规划）**：将最优 $(\boldsymbol{\epsilon}, \boldsymbol{\delta})$ 代入目标函数后，最优阈值的确定可转化为最大化数据分布和查询分布之间 KL 散度的问题，通过动态规划求解。

### 损失函数 / 训练策略

OptLCMS 不涉及传统神经网络训练。其核心是：
- ML 模型沿用 LCMS 的预训练模型
- CMS 参数 $(\epsilon, \delta)$ 通过解析解/DP 直接计算
- 设定 $G=10$ 个分区
- $\epsilon$ 设置为 CMS 在给定内存下可行的最小值 $\epsilon = e/M$

## 实验关键数据

### 主实验：内存 vs 不可容忍误差概率

在 AOL 查询日志数据集（2100万查询，380万唯一词项，Zipf 分布）上评估：

| 数据结构 | 内存效率 | 不可容忍误差概率 | 平均误差 |
|---------|---------|----------------|---------|
| CMS | 基线 | 最高 | 最高 |
| LCMS | 较好 | 中等 | 较低 |
| OptLCMS (G=10) | 较好 | **最低（约 LCMS 的 1/20）** | **与 LCMS 相当** |

核心发现：OptLCMS 在不可容忍误差概率上比 LCMS 低约 **20倍**，同时平均误差保持一致。

### 构建时间比较

| 数据结构 | 构建时间(s) - 频率加权 | 构建时间(s) - 均匀 |
|---------|---------------------|-------------------|
| LCMS | 10.712 | 10.250 |
| OptLCMS | **0.003** (-10.709) | **4.873** (-5.377) |

频率加权查询下构建速度提升约 **3500倍**，均匀查询下提升约 **2倍**。

### 关键发现

1. **解析推导消除了验证环节**：OptLCMS 从验证数据分布直接计算最优参数，不需要在实际数据上反复测试
2. **最小化不可容忍误差上界不牺牲平均误差**：两个目标并不矛盾
3. **分区策略有效利用了分数空间的异质性**：不同分数组的数据分布和查询分布差异被 KL 散度捕捉

## 亮点与洞察

- **优美的理论推导**：将学习型数据结构的参数调优问题转化为凸优化问题，利用 KKT 条件得到解析解，再用 DP 优化阈值
- **实用性强**：构建速度大幅提升，特别适合需要频繁重建 CMS 的在线场景
- **理论保证**：LCMS 缺少的不可容忍误差概率保证被 OptLCMS 填补
- **整数松弛的巧妙处理**：CMS 维度的上取整约束被松弛为连续变量，实践中引入的非凸性可忽略

## 局限与展望

1. **仅在 AOL 数据集上验证**：未在 CAIDA 网络流量数据集上测试
2. **未与 confidence-aware LCMS 对比**：缺少与最新改进版本的比较
3. **分区数 G 固定为 10**：未探索 G 的选择对性能的影响
4. **依赖预训练 ML 模型的质量**：模型分数的准确性直接影响分区效果
5. **Workshop 论文，篇幅有限**：理论分析充分但实验规模较小

## 相关工作与启发

- **Learned Bloom Filter → Learned CMS**：将学习型 Bloom Filter 的分区和优化思想扩展到 CMS 领域
- **PLBF 的优化框架**：Partitioned Learned Bloom Filter 的分区策略提供了直接灵感
- **数据结构 × 凸优化**：展示了传统凸优化工具（KKT、DP）在学习增强数据结构中的有效应用

## 评分

- 新颖性：⭐⭐⭐⭐ — 将凸优化框架引入学习型 CMS 参数优化
- 理论贡献：⭐⭐⭐⭐⭐ — 完整的解析解推导和理论保证
- 实验充分度：⭐⭐⭐ — 仅一个数据集，但指标全面
- 实用价值：⭐⭐⭐⭐ — 构建速度和误差保证的提升对实际应用有直接意义
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](adjusted_count_quantification_learning_on_graphs.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[ICCV 2025\] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation](../../ICCV2025/others/stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
