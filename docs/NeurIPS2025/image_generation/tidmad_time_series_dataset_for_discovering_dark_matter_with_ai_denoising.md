---
title: >-
  [论文解读] TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising
description: >-
  [NeurIPS 2025][图像生成][暗物质探测] 发布 TIDMAD——首个面向暗物质搜索的超长时间序列去噪基准数据集，包含 ABRACADABRA 实验的训练/验证/科学数据、去噪评分指标和完整分析框架，使 AI 算法能直接产出物理学界标准的暗物质搜索结果。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "暗物质探测"
  - "时间序列去噪"
  - "ABRACADABRA"
  - "基准数据集"
  - "信号恢复"
---

# TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2406.04378](https://arxiv.org/abs/2406.04378)  
**代码**: [GitHub](https://github.com/jessicafry/TIDMAD)  
**领域**: 数据集 / 时间序列去噪  
**关键词**: 暗物质探测, 时间序列去噪, ABRACADABRA, 基准数据集, 信号恢复

## 一句话总结

发布 TIDMAD——首个面向暗物质搜索的超长时间序列去噪基准数据集，包含 ABRACADABRA 实验的训练/验证/科学数据、去噪评分指标和完整分析框架，使 AI 算法能直接产出物理学界标准的暗物质搜索结果。

## 研究背景与动机

暗物质占宇宙总质量约 85%，但从未在地球实验室中被直接观测到。轴子（axion）是当前最主要的暗物质候选粒子之一，ABRACADABRA（ABRA-10cm）实验通过超导环形磁铁和 SQUID 传感器搜索轴子信号。

核心挑战在于：

**信号极其微弱**：暗物质信号表现为时间序列中的正弦振荡模式，但被各种噪声源淹没

**频率范围极广**：信号频率跨越七个数量级（1.1 kHz 至 4.9 MHz）

**数据量巨大**：采样率高达 10 MS/s（每秒千万样本）

**噪声非高斯**：检测器噪声由多种独立噪声源叠加形成，无法用简单模型描述

传统去噪方法（如傅里叶平均）在高频区域存在信号丢失问题。机器学习去噪技术有潜力显著提升暗物质实验的灵敏度——噪声降低一倍等效于数据采集时间增加 16 倍。然而，此前缺乏一个标准化的物理实验数据集供 ML 社区使用。TIDMAD 的发布填补了这一空白，架起了 AI 与粒子物理之间的桥梁。

## 方法详解

### 整体框架

TIDMAD 并非提出新算法，而是构建了一个完整的数据集+基准+分析框架生态：

- **训练集**：含注入信号的超长时间序列，提供 SQUID 噪声序列（CH1）和注入信号地面真值（CH2）的一一对应
- **验证集**：独立采集的带注入信号数据，用于计算去噪评分（Benchmark 1）
- **科学数据集**：24 小时不含注入信号的纯实验数据，用于产出暗物质极限（Benchmark 2）

### 关键设计

1. **信号注入机制**

   通过校准回路向检测器硬件注入模拟暗物质信号。注入信号遵循轴子理论形式：

    $\boldsymbol{J}_{eff} = g_{a\gamma\gamma}\sqrt{2\rho_{DM}}\boldsymbol{B}_0 \cos(m_a t)$

   共注入 309 个不同频率（1.1 kHz 至 4.9 MHz），模拟 309 种轴子质量，振幅统一为 50 mV。训练任务即从噪声 SQUID 序列中恢复注入的正弦信号。

2. **去噪评分（Benchmark 1: Denoising Score）**

   基于改进的信号噪声比设计，计算流程：
    - 将时间序列分为 1 秒片段，做 FFT 转换为功率谱密度（PSD）
    - 在注入信号的 PSD 中定位信号频率 $\nu_0$
    - 计算信号区域与噪声区域的 PSD 比值得到 SNR
    - 归一化并加权求和得到 $\Lambda$
    - 对数变换：$\text{Denoising Score} = \log_{5.27}\Lambda$

   去噪评分 = 1 对应未去噪的原始数据，分数越高表示去噪效果越好。经验证，该评分与噪声幅度呈线性关系。

3. **暗物质极限（Benchmark 2: Dark Matter Limit）**

   将去噪后的科学数据输入自动化分析流程，使用频率论对数似然比检验统计量，在 1110 万个独立质量点上重复极限设定过程，最终生成物理学界标准的排除图。ML 开发者只需运行 `brazilband.py` 脚本即可自动产出暗物质搜索结果。

### 损失函数 / 训练策略

针对 8 种基线去噪算法的训练策略：
- 传统方法（移动平均、Savitzky-Golay 滤波、傅里叶平均）直接应用，无需训练
- 深度学习模型（FC Net、PU Net、Transformer、WaveNet、RNN Seq2Seq）在训练集上训练
    - 由于超长序列的内存限制，将序列分段（段长 2×10⁴ ~ 4×10⁴）
    - PU Net / Transformer / RNN Seq2Seq 将去噪重新定义为 256 类分类任务，使用 Focal Loss 训练
    - 除 WaveNet 外，其余模型使用频率分割（多个专业化版本处理不同频率范围）

## 实验关键数据

### 主实验

| 算法 | 类型 | 段长 | 频率分割 | Fine Score | Coarse Score |
|------|------|------|---------|-----------|-------------|
| 无去噪 | - | - | - | 1.00 | 1.10 |
| 傅里叶平均 | 传统 | 1×10⁸ | - | 0.24 | 0.26 |
| 移动平均 | 传统 | 1×10⁶ | - | 0.86 | 0.95 |
| SG 滤波 | 传统 | 1×10⁶ | - | 0.95 | 1.04 |
| FC Net | DL | 4×10⁴ | 是 | **6.43** | **6.55** |
| WaveNet | DL | 4×10⁴ | 否 | 4.99 | 5.16 |
| Transformer | DL | 2×10⁴ | 是 | 3.95 | 4.18 |
| PU Net | DL | 4×10⁴ | 是 | 3.69 | 3.84 |
| RNN Seq2Seq | DL | 4×10⁴ | 是 | 3.38 | 3.79 |

### 暗物质极限对比

| 配置 | 灵敏度 | 说明 |
|------|--------|------|
| ABRA-TIDMAD Raw | 弱 | 未去噪，24h 数据，覆盖区域小于 Run 3 |
| ABRA-TIDMAD Denoised (FC Net) | 强 | AI 去噪后极限提升 1-2 个数量级 |
| ABRA-10cm Run 3 | 最强 | 3 个月数据，当前世界领先 |
| 去噪后差距 | 近平 | 仅 1% 数据量即接近 Run 3 水平，小质量端甚至超越 |

### 关键发现

1. **所有传统方法都降低了去噪评分**，因为时域平均擦除了高频信号；所有深度学习方法都显著提升评分
2. **FC Net 出乎意料地表现最优**（评分 6.43），简单的自编码器结构在该任务上最有效
3. 仅用 1% 数据量（24h vs 3 个月），AI 去噪后的暗物质极限已接近 Run 3 水平
4. Coarse Score（10 倍下采样）与 Fine Score 高度一致，可用于快速评估
5. 去噪分数设计保证了与噪声水平的线性关系，是一个可靠的评估度量

## 亮点与洞察

- **跨学科桥梁**：首次让 ML 社区无需理解底层物理就能直接推动暗物质搜索进展
- **双基准设计**：Denoising Score 提供快速模型迭代，Dark Matter Limit 直接连接物理意义
- **数据规模与真实性**：超 800 Gigasamples 的真实探测器数据，非模拟合成
- 简单模型（FC Net）打败复杂模型的发现值得反思——可能因为任务本质是窄带信号提取

## 局限与展望

- 由于硬件变更（拾取环替换拾取筒）和采集时长限制，基线未能超越 Run 3 结果
- 当前假设零结果（无暗物质候选），尚未实现发现分析代码
- 数据集针对特定频率范围和检测器，泛化到其他暗物质实验仍需验证
- 深度学习模型的频率分割策略增加了工程复杂度

## 相关工作与启发

- 时间序列去噪技术可推广到引力波检测、脉冲星计时、地震信号提取等领域
- 物理约束下的信号恢复任务可能需要不同于通用去噪的架构设计
- 超长序列处理仍是技术瓶颈，基础模型时代的长上下文建模可能带来突破

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个面向暗物质搜索的 ML 基准，跨学科意义重大
- **实验充分度**: ⭐⭐⭐⭐ 8 种基线方法、双基准评估、物理验证闭环
- **写作质量**: ⭐⭐⭐⭐ 物理背景与 ML 方法的讲解对 ML 读者友好
- **价值**: ⭐⭐⭐⭐⭐ 开创了 AI 直接推动基础科学发现的标准化路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](../../ICCV2025/image_generation/learning_to_see_in_the_extremely_dark.md)
- [\[ICML 2025\] LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation](../../ICML2025/image_generation/lscd_lomb-scargle_conditioned_diffusion_for_time_series_imputation.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)

</div>

<!-- RELATED:END -->
