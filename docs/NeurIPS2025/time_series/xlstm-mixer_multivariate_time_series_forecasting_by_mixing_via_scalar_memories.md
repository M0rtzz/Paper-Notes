---
title: >-
  [论文解读] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories
description: >-
  [NeurIPS2025][时间序列][时间序列预测] 提出 xLSTM-Mixer，首次将扩展长短期记忆网络（sLSTM）与混合架构（Mixer）结合，通过时间混合、联合时间-变量混合和多视角混合三阶段架构实现多变量长期时间序列预测的 SOTA 性能，同时保持极低的内存占用。
tags:
  - NeurIPS2025
  - 时间序列
  - 时间序列预测
  - xLSTM
  - 混合架构
  - 循环神经网络
  - 多变量预测
---

# xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories

**会议**: NeurIPS2025  
**arXiv**: [2410.16928](https://arxiv.org/abs/2410.16928)  
**作者**: Maurice Kraus, Felix Divo, Devendra Singh Dhami, Kristian Kersting (TU Darmstadt, TU Eindhoven)
**代码**: [mauricekraus/xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer)  
**领域**: time_series  
**关键词**: 时间序列预测, xLSTM, 混合架构, 循环神经网络, 多变量预测

## 一句话总结
提出 xLSTM-Mixer，首次将扩展长短期记忆网络（sLSTM）与混合架构（Mixer）结合，通过时间混合、联合时间-变量混合和多视角混合三阶段架构实现多变量长期时间序列预测的 SOTA 性能，同时保持极低的内存占用。

## 研究背景与动机
时间序列预测在医疗、制造、交通、金融和天气建模等关键领域无处不在，但现有方法存在显著局限：

- **Transformer 系列**（PatchTST, iTransformer）：注意力机制对序列长度呈二次复杂度，在长序列和资源受限场景下效率低下
- **纯线性模型**（DLinear, NLinear）：虽然高效但表达能力不足，无法捕捉复杂的非线性时序动态
- **SSM/Mamba 系列**（S-Mamba, Chimera）：独立处理序列元素，难以直接学习变量间关系
- **现有混合架构**（TimeMixer, TSMixer）：缺少循环模型的记忆能力和长程依赖建模优势
- **xLSTMTime**：初步尝试将 xLSTM 用于时序，但未能超越 TimeMixer 等强基线，且难以复现

核心观察：(1) 通道独立性假设（如 PatchTST）提供正则化但牺牲跨变量信息；(2) 联合混合更具表达力但易过拟合；(3) sLSTM 的标量记忆和指数门控机制天然适合序列混合。因此需要一种架构将循环模型的记忆能力与混合架构的高效性结合。

## 方法详解

### 整体架构
xLSTM-Mixer 由三个阶段组成：时间混合 → 联合混合 → 视角混合。

输入 $\bm{X} \in \mathbb{R}^{V \times T}$（$V$ 个变量，$T$ 个时间步），预测 $\bm{Y} \in \mathbb{R}^{V \times H}$（$H$ 步未来值）。

### 阶段1：归一化与初始线性预测（时间混合）

1. **RevIN 归一化**：对每个时间序列实例做可逆实例归一化，学习缩放 $\bm{\gamma}$ 和偏移 $\bm{\beta}$：
$$\bm{x}_t^{\text{norm}} = \bm{\gamma} \odot \frac{\bm{x}_t - \mathbb{E}[\bm{x}]}{\sqrt{\text{Var}[\bm{x}]} + \epsilon} + \bm{\beta}$$

2. **NLinear 初始预测**：对每个变量独立应用共享的线性层，从 $T$ 步映射到 $H$ 步：
$$\bm{x}^{\text{initial}} = \text{FC}(\bm{x}_{1:T}^{\text{norm}} - x_T^{\text{norm}}) + x_T^{\text{norm}}$$
   权重在所有变量间共享→参数少+正则化效果。线性预测本身已是不错的基线。

### 阶段2：sLSTM 精化（联合时间-变量混合）

1. **上投影**：将初始预测 $\bm{x}^{\text{initial}} \in \mathbb{R}^{V \times H}$ 投影到更高隐维度 $D$：$\bm{x}^{\text{up}} = \text{FC}^{\text{up}}(\bm{x}^{\text{initial}}) \in \mathbb{R}^{V \times D}$，同样跨变量共享权重

2. **sLSTM 块堆叠**：$M$ 层 sLSTM 块沿**变量维度**（而非时间维度）递归处理。每个 token 代表单个变量的所有时间步嵌入。关键特性：
    - **指数门控**：输入门和遗忘门使用指数函数（$\bm{i}_t = \exp(\tilde{\bm{i}}_t - \bm{m}_t)$），增强记忆控制
    - **多头记忆混合**：循环权重矩阵 $\bm{R}$ 为块对角结构，允许头部专门化
    - **数值稳定化**：$\bm{m}_t = \max(\tilde{\bm{f}}_t + \bm{m}_{t-1}, \tilde{\bm{i}}_t)$ 防止指数爆炸

3. **可学习初始嵌入 $\bm{\eta}$**：借鉴 LLM 的软提示（soft prompt），学习一个初始 token 预置于序列前端，使 sLSTM 的初始隐状态适应特定数据集特征

**为何选择 sLSTM 而非 mLSTM**：mLSTM 独立处理序列元素，无法直接学习变量间关系；sLSTM 通过循环权重矩阵实现元素间交互。

**为何沿变量递归**：沿变量方向参数量不随变量数增加而增长（线性时间缩放），实证效果优于沿时间方向。

### 阶段3：多视角混合

1. 分别对原始嵌入 $\bm{x}^{\text{up}}$ 和**翻转后**的嵌入 $\hat{\bm{x}}^{\text{up}}$ 通过共享权重的 sLSTM 堆栈产生两组预测 $\bm{y}', \bm{y}''$
2. 线性投影融合两个视角：$\bm{y}^{\text{norm}} = \text{FC}^{\text{view}}(\bm{y}', \bm{y}'')$
3. 反 RevIN 得到最终预测 $\bm{y} = \text{RevIN}^{-1}(\bm{y}^{\text{norm}})$

多视角可视为对不同变量排序的集成（weight-sharing ensembling），提供额外正则化。

## 实验关键数据

### Table 1: 长期预测主实验（7 数据集，4 预测长度 {96,192,336,720} 的平均值）

| 模型 | Weather MSE | Electricity MSE | Traffic MSE | ETTh1 MSE | ETTh2 MSE | ETTm1 MSE | ETTm2 MSE | MSE Wins |
|---|---|---|---|---|---|---|---|---|
| **xLSTM-Mixer** | **0.219** | **0.153** | 0.392 | 0.397 | 0.340 | **0.339** | **0.248** | **11** |
| Chimera | 0.219 | 0.154 | 0.403 | 0.405 | **0.318** | 0.345 | 0.250 | 8 |
| TimeMixer | 0.222 | 0.156 | **0.387** | 0.411 | 0.316 | 0.348 | 0.256 | 2 |
| CycleNet | 0.223 | 0.156 | 0.403 | 0.435 | 0.367 | 0.360 | 0.263 | 1 |
| PatchTST | 0.241 | 0.159 | 0.391 | 0.413 | 0.324 | 0.353 | 0.256 | 1 |
| TimeMixer++ | 0.226 | 0.165 | 0.416 | 0.419 | 0.339 | 0.369 | 0.269 | 0 |
| DLinear | 0.246 | 0.166 | 0.434 | 0.423 | 0.431 | 0.357 | 0.267 | 0 |

**关键发现**：
- xLSTM-Mixer 在 28 个设置中赢得 **11 次 MSE 最优 + 16 次 MAE 最优**
- 在 7 个数据集中的 **6 个**上定义新 SOTA
- 统计显著性检验（Friedman + Conover post-hoc, p=0.05）：xLSTM-Mixer 显著优于除 xLSTMTime 外的所有方法，但 xLSTM-Mixer 平均排名 1.5 远优于 xLSTMTime 的 4.0

### Table 2: GIFT-Eval 概率预测基准（Top 10）

| 模型 | MASE ↓ | CRPS ↓ | 排名 |
|---|---|---|---|
| TiRex | 0.724 | 0.498 | 1 |
| **xLSTM-Mixer** | **0.780** | **0.510** | **2** |
| TEMPO_ensemble | 0.862 | 0.514 | 3 |
| Toto_Open_Base | 0.750 | 0.517 | 4 |
| TabPFN-TS | 0.771 | 0.544 | 5 |
| timesfm_2_0_500m | 0.758 | 0.550 | 7 |

**关键发现**：
- 按 CRPS 排名第 2（仅次于 TiRex），**纯监督模型中排名第 1**
- 在涵盖单变量/多变量、短/长预测的异构基准上表现稳健
- 通过添加分位数预测头即可扩展为概率预测

### 消融实验（Table 3 摘要，Weather + ETTm1）

| 消融项 | MSE 增幅 | MAE 增幅 |
|---|---|---|
| sLSTM → LSTM (#3) | +7.0% | +6.2% |
| 沿变量递归 → 沿时间递归 (#5) | +4.7% | +4.3% |
| 移除时间混合 (#11) | +3.1% | +2.7% |
| 移除初始嵌入 η (#7) | +0.7% | +0.4% |
| 移除视角混合 (#8) | +0.7% | +0.6% |

sLSTM 块和时间混合是最关键组件。所有组件均正向贡献，完整配置性能最优。

### 效率分析
- xLSTM-Mixer 内存占用比 TimeMixer 低 **1-2 个数量级**
- 随回看窗口 $T$ 增长，时间和内存几乎不增加（对比 Transformer 的二次增长）
- 可高效利用更长回看窗口（$T$ 从 96 增至 1440），性能持续提升

## 亮点
- **首次融合循环+混合架构**：将 sLSTM 的表达力和记忆混合能力与 Mixer 的高效结构结合，开辟时序预测新范式
- **极低内存占用**：比 TimeMixer 低 1-2 个数量级，比 Transformer 方法低更多，适合边缘设备部署
- **多视角混合创新**：原始+翻转嵌入的双视角集成为变量排序敏感问题提供了优雅方案，且共享权重不增加参数
- **全面的统计严谨性**：Friedman + Conover post-hoc 检验确认显著性，而非仅靠 win count
- **通用性强**：从长期点预测到 GIFT-Eval 概率预测再到分类任务，均表现优异

## 局限与展望
- **均匀采样假设**：要求所有变量在统一时间网格上采样，不规则或缺失时间戳需预处理
- **高维变量瓶颈**：沿变量递归使运行时间与变量数线性相关，极高维（如数千变量）时可能成为瓶颈
- **变量排序敏感性**：虽然实验显示标准排序已足够好，但最优排序搜索仍是开放问题
- **可解释性受限**：多视角混合融合了时间和跨变量信息，难以做细粒度归因分析
- **仅限点/分位数预测**：概率预测通过分位数头实现，未探索更丰富的分布建模

## 相关工作
- **循环模型**：LSTM/GRU → xLSTM（指数门控+记忆混合）→ xLSTMTime（首次用于时序但效果有限）→ **xLSTM-Mixer 通过混合架构显著提升**
- **混合架构**：MLP-Mixer → TSMixer/TimeMixer/TimeMixer++（交替混合时间和变量维度）→ xLSTM-Mixer 用 sLSTM 替代 MLP 做联合混合
- **Transformer 系列**：Autoformer → PatchTST → iTransformer，高精度但资源消耗大
- **SSM 系列**：S-Mamba, Chimera，允许并行推理但混合能力受限
- **预训练模型**：Chronos, Moirai, Timer-XL，需要大规模预训练数据
- **xLSTM-Mixer 的定位**：填补循环模型与混合架构的交叉空白，以极低内存实现超越 Transformer 的精度

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 sLSTM 与 Mixer 架构结合，多视角混合和沿变量递归的设计选择有方法论贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 7 基准 + GIFT-Eval + 分类 + 13 组消融 + 统计检验 + 效率分析，极为全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，消融分析严谨；符号系统较重
- 价值: ⭐⭐⭐⭐ — 为资源受限场景下的高精度时序预测提供实用方案，推动循环模型在时序领域的复兴

<!-- RELATED:START -->

## 相关论文

- [Free Energy Mixer](../../ICLR2026/time_series/free_energy_mixer.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](../../ICML2025/time_series/timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [Winner-takes-all for Multivariate Probabilistic Time Series Forecasting](../../ICML2025/time_series/winner-takes-all_for_multivariate_probabilistic_time_series_forecasting.md)
- [Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)

<!-- RELATED:END -->
