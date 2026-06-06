---
title: >-
  [论文解读] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection
description: >-
  [ICML 2025][时间序列][多变量异常检测] 提出 CAROTS——将因果关系融入对比学习的多变量时间序列异常检测框架，用因果保持增强作为正样本（正常变化），因果破坏增强作为负样本（模拟异常），训练编码器基于因果结构区分正常与异常。
tags:
  - "ICML 2025"
  - "时间序列"
  - "多变量异常检测"
  - "因果关系"
  - "对比学习"
  - "数据增强"
  - "因果发现"
---

# Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection

**会议**: ICML 2025  
**arXiv**: [2506.03964](https://arxiv.org/abs/2506.03964)  
**代码**: [https://github.com/kimanki/CAROTS](https://github.com/kimanki/CAROTS)  
**领域**: 时间序列  
**关键词**: 多变量异常检测, 因果关系, 对比学习, 数据增强, 因果发现

## 一句话总结
提出 CAROTS——将因果关系融入对比学习的多变量时间序列异常检测框架，用因果保持增强作为正样本（正常变化），因果破坏增强作为负样本（模拟异常），训练编码器基于因果结构区分正常与异常。

## 研究背景与动机

**领域现状**：多变量时间序列异常检测（MTSAD）是网络安全、医疗监控等关键领域的核心需求。

**现有痛点**：现有无监督方法（重建/对比学习）关注数据值或分布的表面差异，忽略变量间的因果关系——导致正常的因果变化被误报为异常（如空调因高温运行更久是正常的因果变化，不是异常）。

**核心矛盾**：区分"因果关系保持的正常变化"和"因果关系被破坏的真正异常"。

**本文目标**：将因果结构融入异常检测。

**切入角度**：用因果发现模型提取变量间因果图，然后设计因果保持和因果破坏的数据增强策略。

**核心 idea**：因果保持增强 = 正样本，因果破坏增强 = 负样本 → 对比学习训练因果感知编码器。

## 方法详解

### 整体框架
1. 用预测型因果发现模型提取变量间因果关系
2. 因果保持增强器：修改数据但保持因果结构不变
3. 因果破坏增强器：打破因果关系模拟异常
4. 对比学习：正样本（因果保持）vs 负样本（因果破坏）
5. 异常检测：距离正常聚类中心的距离 + 因果预测误差

### 关键设计

1. **因果保持增强**:

    - 功能：生成遵循原始因果结构的多样化正常样本
    - 核心思路：对因果图中的"因"变量施加变化，"果"变量按因果关系相应调整
    - 设计动机：扩展正常模式的多样性，防止将正常变化误判为异常

2. **因果破坏增强**:

    - 功能：生成打破因果关系的合成异常
    - 核心思路：随机打断因果链——如改变"果"变量但不改变对应的"因"变量
    - 设计动机：模拟真实异常（如传感器故障导致的因果断裂）

3. **相似度过滤的单类对比损失**:

    - 功能：渐进式引入更多样化的正样本
    - 核心思路：初始只将高相似度样本作为正样本，随训练逐步放宽阈值
    - 设计动机：防止训练初期噪声过大导致聚类不稳定

### 损失函数 / 训练策略
- 对比损失 + 因果预测损失的集成
- 两个异常分数的加权组合

## 实验关键数据

### 主实验
5 个真实世界数据集的 F1 分数：

| 方法 | SWaT | WADI | SMD | MSL | PSM |
|------|------|------|-----|-----|-----|
| USAD | 79.2 | 56.3 | 82.4 | 73.4 | 96.2 |
| AnomalyTransformer | 83.5 | 58.1 | 86.2 | 78.8 | 97.1 |
| **CAROTS** | **86.7** | **63.4** | **88.9** | **82.1** | **97.8** |

### 消融实验

| 配置 | F1 (SWaT) | 说明 |
|------|-----------|------|
| 无因果增强 | 81.2 | 退化为普通对比 |
| 仅因果保持 | 83.5 | 缺少异常模拟 |
| 仅因果破坏 | 84.1 | 缺少正常多样性 |
| **完整 CAROTS** | **86.7** | 最优 |

### 关键发现
- 在有明显因果关系的数据集上提升最大（SWaT: +3.2%）
- 在合成数据集上优势更明显（因果关系显式可控）
- 相似度过滤使训练更稳定，提升 +1.5%

## 亮点与洞察
- **因果关系 × 对比学习**的组合自然且有效——异常=因果断裂的概念非常清晰
- 数据增强策略兼顾了正样本多样性和负样本真实性
- 可推广到任何有变量间因果关系的多变量监控场景

## 局限与展望
- 因果发现模型的准确性直接影响下游性能
- 假设因果关系在正常状态下稳定，时变因果场景需要扩展
- 仅测试线性和简单非线性因果

## 评分
- 新颖性: ⭐⭐⭐⭐ 因果+对比学习的组合有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 5真实+2合成数据集，充分消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法自然
- 价值: ⭐⭐⭐⭐ 推进因果感知异常检测

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](../../NeurIPS2025/time_series/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[ICML 2025\] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/time_series/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[ICML 2025\] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)

</div>

<!-- RELATED:END -->
