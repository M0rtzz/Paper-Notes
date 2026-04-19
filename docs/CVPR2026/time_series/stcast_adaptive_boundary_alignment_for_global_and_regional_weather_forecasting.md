---
title: >-
  [论文解读] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting
description: >-
  [CVPR 2026][时间序列][weather forecasting] 提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。
tags:
  - CVPR 2026
  - 时间序列
  - weather forecasting
  - 注意力机制
  - temporal MoE
  - global-regional coupling
  - adaptive boundary
---

# STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting

**会议**: CVPR 2026  
**arXiv**: [2509.25210](https://arxiv.org/abs/2509.25210)  
**代码**: 无  
**领域**: 时空预测 / 气象预报  
**关键词**: weather forecasting, spatial-aligned attention, temporal MoE, global-regional coupling, adaptive boundary  

## 一句话总结

提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。

## 研究背景与动机

**精确的区域天气预报**需要全球大气动力学的支持——西伯利亚冷涌能触发东亚寒潮，青藏高原地表加热能同时改变东亚季风和北美急流。因此区域预报的真正"边界"不是邻近区域，而是整个地球。

**现有方法的痛点**：传统NWP方法通过求解PDE在精细网格上预报，计算成本极高。数据驱动方法（如Pangu-Weather、GraphCast）大幅降低了成本，但面临两个核心问题：（1）直接训练高分辨率（~1km, 0.01°）全球模型在计算上不可行（需要19980×39960的网格）；（2）现有的全球-区域耦合方法仅使用**静态的邻近区域**作为边界——如OneForecast直接拼接邻域裁剪，违背了大气-海洋-陆地-生物圈耦合理论。

**STCast的核心创新**：（1）用可学习的全球-区域分布替代静态邻域边界，基于大圆距离初始化、在训练中自适应优化；（2）用月份特异的高斯先验引导MoE专家路由，显式建模不同月份的大气变化特征。

## 方法详解

### 整体框架

STCast统一处理四个任务：低分辨率全球预报 → 高分辨率区域预报（通过SAA融合全球信息）→ 台风路径预测（利用区域预测的海平面气压）→ 集合预测（注入Perlin噪声做多次模拟取均值）。骨干网络采用Encoder-Processor-Decoder架构，Processor交替使用窗口注意力和自注意力。

### 关键设计

1. **Spatial-Aligned Attention（SAA）**:

    - 功能：自适应聚合全球大气信息到区域预报，替代静态边界拼接
    - 核心思路：用全球特征做Query和Key、区域特征做Value的线性交叉注意力。关键创新在先验初始化——计算每个全球点到目标区域的Great Circle距离 $d(\phi,\lambda)$，用指数衰减函数 $f(\phi,\lambda) = \exp(-\alpha \cdot d^2)$ 初始化全球-区域分布，通过Hadamard积调制注意力权重，并在训练中持续优化。使用 $O(n)$ 线性注意力降低计算复杂度
    - 设计动机：大气影响随距离衰减但不消失，指数衰减先验编码了这种物理直觉，同时允许模型学习到超出地理邻近的长程关联（如遥相关）

2. **Temporal Mixture-of-Experts（TMoE）**:

    - 功能：将不同月份的预报任务动态路由到专门的专家模型
    - 核心思路：为每个月学习一个离散高斯分布，峰值旋转对齐到输入变量的月份。月份embedding $M \in \mathbb{R}^{12\times 1}$ 与输入特征的路由权重拼接后经softmax选Top-K专家：$I = \text{Softmax}(\text{Conv}(X^t) + M)$。高斯分布确保激活概率随时间距离单调递减
    - 设计动机：大气变量在不同月份有显著差异（夏季对流 vs 冬季辐射），隐式MoE分配难以学到这种时序特异性且容易同质化，TMoE通过显式月份embedding提供清晰的专家分工信号

3. **统一四任务框架**:

    - 功能：用同一框架处理全球预报、区域预报、台风路径和集合预测
    - 核心思路：全球预报 $X_g^{t+1} = \Phi_g(X_g^t)$；区域预报通过SAA耦合 $X_r^{t+1} = \Phi_r(X_r^t, X_g^t)$；台风路径从区域预测的海平面气压推断；集合预测向全球初始态注入Perlin噪声做n次模拟取均值
    - 设计动机：四个任务共享大气物理的底层表征，统一框架能更好利用全球-区域-时序的耦合关系

### 损失函数 / 训练策略

AdamW优化器，学习率0.0002，100 epochs，batch size 16。在ERA5数据集上训练（1979-2019, 0.25°分辨率, 721×1440, 70个变量）。全球和区域模型分别训练。16× NVIDIA A100 GPUs。

## 实验关键数据

### 主实验

**全球预报（ERA5, 归一化RMSE↓ / ACC↑）**：

| 方法 | 1天RMSE↓ | 4天RMSE↓ | 7天RMSE↓ | 10天RMSE↓ |
|------|---------|---------|---------|----------|
| Pangu-Weather | 0.1571 | 0.3380 | 0.5092 | 0.6215 |
| GraphCast | 0.1304 | 0.2861 | 0.4597 | 0.6009 |
| OneForecast | 0.1231 | 0.2732 | 0.4468 | 0.5918 |
| **STCast** | **0.1197** | **0.2578** | **0.4348** | **0.5763** |

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| 无SAA（直接拼接邻域） | 区域预报显著下降 | 静态边界不足 |
| 无TMoE（标准MoE） | 时序泛化减弱 | 月份特异性丢失 |
| 距离衰减先验固定不更新 | 性能中等 | 可学习先验更好 |
| 欧氏距离替代Great Circle | 略差 | 球面距离更准确 |

### 关键发现

- SAA的自适应边界在所有区域预报变量上均优于OneForecast的邻域拼接和直接训练策略
- TMoE的月份embedding有效防止了MoE同质化，不同月份的专家学到了不同的大气模式
- STCast在10天长期预报上优势更大（RMSE 0.5763 vs 0.5918），全球-区域耦合对长程预报帮助更大
- 台风路径预测和集合预测也展示了统一框架的泛化能力

## 亮点与洞察

- Great Circle距离+指数衰减初始化的先验设计非常物理直觉——它编码了"大气影响随距离衰减"的基本规律，同时留给模型学习不符合纯距离衰减的遥相关模式的空间。这种"物理先验初始化+数据驱动优化"的范式在其他地球科学应用中也很有复用价值。
- TMoE用离散高斯分布做月份routing——相比让模型隐式学习时间模式，显式提供月份信息是一个低成本高回报的设计。

## 局限与展望

- 区域预报目前仅在东亚验证，其他地理区域（如赤道、极地）的效果待验证
- 不确定性量化仅通过Perlin噪声集合预测实现，缺乏概率校准分析
- 计算成本：16× A100训练100 epochs，对资源受限的研究组仍然较高

## 相关工作与启发

- **vs OneForecast**: 最直接的对比者，OneForecast用邻域拼接做全球-区域耦合，STCast用SAA做自适应分布学习，在所有四个任务上全面优于OneForecast
- **vs GraphCast/FuXi**: 这些方法专注全球预报，STCast通过SAA扩展到高分辨率区域预报，填补了AI气象预报中全球→区域的关键gap

## 评分

- 新颖性: ⭐⭐⭐⭐ SAA的距离衰减先验和TMoE的月份embedding都有很好的物理动机
- 实验充分度: ⭐⭐⭐⭐ 四个任务×多个baseline，消融充分
- 写作质量: ⭐⭐⭐ 框架描述清晰但部分细节隐藏在附录
- 价值: ⭐⭐⭐⭐ 统一四任务且全面SOTA，对AI气象预报社区很有价值

<!-- RELATED:START -->

## 相关论文

- [VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](../../ICCV2025/time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)
- [Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [DemandCast: Global hourly electricity demand forecasting](../../NeurIPS2025/time_series/demandcast_global_hourly_electricity_demand_forecasting.md)
- [Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](../../NeurIPS2025/time_series/time-o1_time-series_forecasting_needs_transformed_label_alignment.md)

<!-- RELATED:END -->
