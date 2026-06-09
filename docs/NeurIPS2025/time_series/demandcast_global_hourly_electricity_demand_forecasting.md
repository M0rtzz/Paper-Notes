---
title: >-
  [论文解读] DemandCast: Global hourly electricity demand forecasting
description: >-
  [NeurIPS 2025][时间序列][电力需求预测] 构建 DemandCast 开源机器学习框架，基于 XGBoost 融合历史电力需求、ERA5 温度和社会经济特征进行全球 56 个国家/地区的小时级电力需求预测，通过归一化目标变量（年度分数）实现跨国家可比，在时间外推测试集上达到 MAPE 9.2%…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "电力需求预测"
  - "XGBoost"
  - "全球尺度"
  - "ERA5 气象"
  - "归一化目标"
---

# DemandCast: Global hourly electricity demand forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2510.08000](https://arxiv.org/abs/2510.08000)  
**代码**: [GitHub](https://github.com/open-energy-transition/demandcast)  
**领域**: 时间序列预测 / 能源  
**关键词**: 电力需求预测, XGBoost, 全球尺度, ERA5 气象, 归一化目标

## 一句话总结

构建 DemandCast 开源机器学习框架，基于 XGBoost 融合历史电力需求、ERA5 温度和社会经济特征进行全球 56 个国家/地区的小时级电力需求预测，通过归一化目标变量（年度分数）实现跨国家可比，在时间外推测试集上达到 MAPE 9.2%。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：全球能源转型（脱碳）要求精确预测电力需求的时空分布，以支持可再生能源的并网和电网管理。然而，未来电力消费模式受人口增长、经济发展、城市化、技术变革等多因素影响，存在巨大不确定性。这种不确定性在全球南方国家尤为突出——这些国家面临在扩大电力供给的同时减少碳排放的双重挑战。

现有方法的不足：(1) 大多限于单个国家或少数地区；(2) 数据覆盖时段短（如仅用 2015 年数据）；(3) 缺乏端到端的开源可复现管线。

DemandCast 的切入角度：构建覆盖全球 56 个国家/地区、跨越 2000-2025 年的大规模预测框架，通过归一化设计使模型能够在数据异质的国家间泛化。

## 方法详解

### 整体框架

DemandCast 是模块化的端到端管线：(1) 数据收集与清洗（电力需求、气象、社会经济）；(2) 特征工程（温度特征、时间特征、社会经济特征）；(3) XGBoost 模型训练与预测；(4) 后处理还原绝对需求值。

### 关键设计

1. **归一化目标变量**:
    - 功能：将小时电力需求归一化为年度总需求的分数
    - 核心思路：$D_n(t) = \frac{D(t)}{D_Y} \cdot \frac{\sum_Y H_{\text{avail}}}{\sum_Y H}$，模型预测的是需求的时间分布轮廓而非绝对量
    - 设计动机：不同国家的绝对需求量差异巨大（数个数量级），直接建模绝对值无法跨国家泛化；归一化后模型专注于捕捉日、周、季节模式

2. **多源特征融合**:
    - 功能：融合 ERA5 再分析温度数据、GDP、人口密度等社会经济指标
    - 核心思路：从人口最密集的 1-3 个网格点提取温度特征，构建月均温和月排名以捕捉季节变化；使用人均 GDP 和人均电力消费作为国家级特征
    - 设计动机：电力需求与气温高度相关（制冷/采暖），社会经济因素决定了基础需求水平

3. **时间分割评估策略**:
    - 功能：对每个地区，最后一年用作测试集，倒数第二年用作验证集，其余为训练集
    - 核心思路：按时间先后顺序严格分割（train 81.25% / val 9.84% / test 8.91%），确保评估的是对未来时段的外推能力
    - 设计动机：避免时间序列中常见的数据泄露问题

### 训练策略

- 模型：XGBoost（梯度提升树）
- 训练数据：6,041,222 个小时级观测
- 验证数据：731,538 条
- 测试数据：662,369 条

## 实验关键数据

### 主实验

| 指标 | 数值 |
|------|------|
| 平均 MAPE（测试集） | **9.2%** |
| 覆盖国家/地区数 | 56 |
| 时间跨度 | 2000-2025 |
| 训练样本数 | 6,041,222 |
| 先前研究 MAPE | ~8%（更小规模） |

### 部分地区表现

| 地区 | MAPE (test) | 地区 | MAPE (test) |
|------|-----------|------|-----------|
| 西班牙 (ES) | 2.34% | 德国 (DE) | 6.74% |
| 巴西南部 (BR_S) | 8.00% | 日本关东 (JP) | 8.53% |
| 韩国 (KR) | 8.00% | 意大利 (IT) | 10.19% |
| 阿尔巴尼亚 (AL) | 15.39% | 墨西哥北部 (MX_NOR) | 18.90% |

### 关键发现

- 发达国家（数据丰富）的预测误差普遍低于发展中国家
- 温度特征是最重要的预测因子，尤其在供暖/制冷需求强的地区
- 部分地区 MAPE 较高（如阿尔巴尼亚 15%、加拿大阿尔伯塔 19%），主要因训练数据量不足或需求模式突变

## 亮点与洞察

- **规模新颖**：首个覆盖 56 个国家/地区、跨 25 年的全球小时级电力需求预测开源框架
- **归一化设计简洁有效**：将问题从"预测绝对需求"转化为"预测需求时间分布"，大幅提升跨国家泛化能力
- **完全开源可复现**：端到端管线从数据收集到预测均开源，对能源规划社区有直接实用价值

## 局限与展望

- 缺乏预测不确定性量化——能源规划决策需要知道预测的置信区间
- XGBoost 未与深度学习方法（Transformer、N-BEATS 等）进行对比
- 数据可用性在不同地区高度不均，部分地区训练数据仅几年
- 未考虑需求侧管理、极端天气事件等因素的影响
- 超参数优化尚未系统进行

## 相关工作与启发

- **GlobalEnergyGIS (Mattsson et al.)**: 先前工作覆盖 44 个国家但仅用 2015 年数据
- **ERA5 再分析数据集**: 提供全球一致的高分辨率气象数据
- **对能源转型的意义**: DemandCast 可帮助规划可再生能源并网和电网扩容

## 评分

- 新颖性: ⭐⭐⭐ 方法简单（XGBoost）但规模和开源框架新颖
- 实验充分度: ⭐⭐⭐ 56 国家全覆盖但缺乏模型对比
- 写作质量: ⭐⭐⭐⭐ 清晰简洁，数据展示充分
- 价值: ⭐⭐⭐⭐ 能源规划领域的实用开源工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[CVPR 2025\] L2GTX: From Local to Global Time Series Explanations](../../CVPR2025/time_series/l2gtx_from_local_to_global_time_series_explanations.md)
- [\[CVPR 2026\] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](../../CVPR2026/time_series/stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)
- [\[ICML 2025\] TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness](../../ICML2025/time_series/tcp-diffusion_a_multi-modal_diffusion_model_for_global_tropical_cyclone_precipit.md)

</div>

<!-- RELATED:END -->
