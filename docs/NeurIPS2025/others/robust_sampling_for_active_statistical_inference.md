---
title: >-
  [论文解读] Robust Sampling for Active Statistical Inference
description: >-
  [NeurIPS 2025][主动推断] 提出基于预算保持路径的鲁棒采样策略，通过在均匀采样和主动采样之间最优插值，确保估计器的方差永远不比两者中任何一个更差，解决了主动统计推断中不确定性估计不准确导致性能恶化的问题。 现有痛点 现有痛点：领域现状：主动统计推断利用AI模型的不确定性评分优先采集高不确定性样本的标签来提高估计…
tags:
  - "NeurIPS 2025"
  - "主动推断"
  - "鲁棒采样"
  - "预测增强推断"
  - "不确定性估计"
  - "逆概率加权"
---

# Robust Sampling for Active Statistical Inference

**会议**: NeurIPS 2025  
**arXiv**: [2511.08991](https://arxiv.org/abs/2511.08991)  
**代码**: [有](https://github.com/lphLeo/Robust-Active-Statistical-Inference)  
**领域**: 统计推断 / 主动采样  
**关键词**: 主动推断, 鲁棒采样, 预测增强推断, 不确定性估计, 逆概率加权

## 一句话总结
提出基于预算保持路径的鲁棒采样策略，通过在均匀采样和主动采样之间最优插值，确保估计器的方差永远不比两者中任何一个更差，解决了主动统计推断中不确定性估计不准确导致性能恶化的问题。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：主动统计推断利用AI模型的不确定性评分优先采集高不确定性样本的标签来提高估计精度。然而，当不确定性估计质量差（如LLM过度自信、分布偏移下校准不良）时，主动采样可能产生高方差估计，甚至比简单均匀采样更差。本文的核心问题是：能否设计同时保证不比均匀采样和初始主动采样更差的鲁棒策略？

### 解决思路

**本文目标**：### 整体框架
给定初始采样规则$\pi$和均匀采样$\pi^{\text{unif}}$，通过保持预算约束的路径$\pi^{(\rho)}$连接两者，最优选择参数$\rho$使方差最小。


## 方法详解

### 整体框架
给定初始采样规则$\pi$和均匀采样$\pi^{\text{unif}}$，通过保持预算约束的路径$\pi^{(\rho)}$连接两者，最优选择参数$\rho$使方差最小。进一步引入鲁棒优化以抵御误差函数的误指定。

### 关键设计

**预算保持路径**：连续路径$\pi^{(\rho)}$，$\rho \in [0,1]$，满足$\pi^{(0)}=\pi$，$\pi^{(1)}=\pi^{\text{unif}}$，且$\mathbb{E}[\pi^{(\rho)}(X)] = \mathbb{E}[\pi(X)]$。三种实例：
- **线性路径**：$\pi^{(\rho)} = (1-\rho)\pi + \rho\pi^{\text{unif}}$
- **几何路径**：$\pi^{(\rho)} \propto \pi^{1-\rho}(\pi^{\text{unif}})^\rho$（推荐默认）
- **Hellinger路径**：$\pi^{(\rho)} \propto ((1-\rho)\sqrt{\pi} + \rho\sqrt{\pi^{\text{unif}}})^2$

**最优$\rho$估计**：拟合误差函数$\hat{e}^2(\cdot) \approx \mathbb{E}[(Y-f(X))^2|X]$，通过网格搜索最小化方差的经验近似。

**鲁棒优化**：引入误差误指定集$\mathcal{C}$的极小化极大问题：$\rho_{\text{robust}} = \arg\min_\rho \max_{\epsilon \in \mathcal{C}} \frac{1}{n}\sum_{i=1}^n \frac{\hat{e}^2(X_i)+\epsilon_i}{\pi^{(\rho)}(X_i)}$。默认使用$\ell_2$约束$\|\epsilon\|_2 \leq c$，$c$通过交叉验证选择。

### 损失函数 / 训练策略
- Burn-in阶段：收集初始标注数据用于拟合误差函数$\hat{e}$
- 主阶段：用鲁棒采样规则$\pi^{(\rho_{\text{robust}})}$执行采样
- 定理保证：当$\hat{\rho}$一致估计$\rho^*$时，$\sqrt{n}(\hat{\theta}^{\pi^{(\hat{\rho})}} - \theta^*) \xrightarrow{d} \mathcal{N}(0, \sigma_{\rho^*}^2)$，且$\sigma_{\rho^*}^2 \leq \min\{\sigma_0^2, \sigma_1^2\}$

## 实验关键数据

### 主实验（Pew选后调查 - 总统支持率估计）

| 采样方法 | 有效样本量趋势 | 覆盖率 |
|---------|-------------|-------|
| 均匀采样 | 基线 | ~90% |
| 主动采样（差误差估计） | **低于均匀** | 覆盖偏差大 |
| **鲁棒主动** | **始终不低于两者** | ~90% |

### Burn-in大小影响

| Burn-in比例 | 鲁棒$\rho$ | 策略 |
|------------|-----------|------|
| 极小 | ~1（接近均匀） | 自动退化为均匀采样 |
| 中等 | 中间值 | 平衡两者 |
| 充足 | ~0（接近主动） | 充分利用不确定性 |

### LLM计算社会科学实验

| 任务 | 均匀 | 主动 | 鲁棒 |
|------|------|------|------|
| 政治偏见检测 | 基线 | 大方差 | ≥基线 |
| 礼貌性分析 | 基线 | 不稳定 | ≥基线 |
| 虚假信息检测 | 基线 | 大方差 | **显著优于两者** |

### 关键发现
- 几何路径在所有测试中表现最优
- 鲁棒约束集$\mathcal{C}$通过交叉验证自动调节松紧度
- 误差估计质量差时$\rho_{\text{robust}}$接近1（均匀），质量好时接近0（主动）
- LLM的逐字化置信度常过度自信，导致采样概率极小、逆概率加权爆炸

## 亮点与洞察
1. **理论优雅**：通过几何路径和鲁棒优化，保证了渐近最优性和方差下界
2. **实际问题导向**：直接解决LLM不确定性不可靠这一普遍问题
3. **自适应**：当误差估计质量改善时，策略自动从均匀过渡到主动
4. **通用性**：从均值估计推广到一般凸M-估计（线性回归、逻辑回归等）

## 局限与展望
- 约束集$\mathcal{C}$的选择对性能敏感，如何自动选择$\mathcal{C}$仍是开放问题
- 最优路径的数据驱动选择未充分探索
- 渐近理论，有限样本保证为经验性验证
- 未涉及在线/顺序设定

## 相关工作与启发
- 扩展了prediction-powered inference和active statistical inference两个框架
- 与半参数推断、缺失数据、因果推断中的AIPW估计密切相关
- 鲁棒优化思路可应用于其他涉及不确定性估计的主动方法

## 评分
- 新颖性：⭐⭐⭐⭐（鲁棒路径插值是创新点）
- 技术深度：⭐⭐⭐⭐⭐（理论推导严密完整）
- 实验完整性：⭐⭐⭐⭐（多领域多场景验证）
- 实用价值：⭐⭐⭐⭐⭐（直接解决LLM不确定性在标注中的实际问题）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)

</div>

<!-- RELATED:END -->
