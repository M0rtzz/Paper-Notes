---
title: >-
  [论文解读] Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative
description: >-
  [ICLR2026][时间序列][多模态] 发现时间序列配对文本具有与时间序列相似的周期性（Chronological Textual Resonance），提出 TaTS 框架将文本表征转化为辅助变量，以即插即用方式增强任意现有时间序列模型的预测和插补性能。
tags:
  - ICLR2026
  - 时间序列
  - 多模态
  - text-augmented forecasting
  - Chronological Textual Resonance
  - plug-and-play framework
---

# Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative

**会议**: ICLR2026  
**arXiv**: [2502.08942](https://arxiv.org/abs/2502.08942)  
**代码**: [iDEA-iSAIL-Lab-UIUC/TaTS](https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS)  
**领域**: time_series  
**关键词**: multimodal time series, text-augmented forecasting, Chronological Textual Resonance, plug-and-play framework  

## 一句话总结
发现时间序列配对文本具有与时间序列相似的周期性（Chronological Textual Resonance），提出 TaTS 框架将文本表征转化为辅助变量，以即插即用方式增强任意现有时间序列模型的预测和插补性能。

## 背景与动机
- 现实中时间序列数据常伴随文本信息（如疫情期间感染率 + 政府公告、经济指标 + 新闻报道），但现有模型大多只用数值数据，忽略了文本所含的互补信息
- 当前最佳多模态方法（MM-TSFLib）虽使用了文本，但忽略了时间序列配对文本特有的位置特性和周期性
- 受 Platonic Representation Hypothesis（PRH）启发：不同模态描述同一事物时，其表征会收敛到共享空间。若时间序列和配对文本描述同一变化事件，两者应展现相似的周期性

## 核心问题
时间序列配对文本具有哪些独特属性？如何系统性地将这些文本信息整合以提升时间序列建模？

## 方法详解

### 1. Chronological Textual Resonance (CTR) 现象
- 对三类真实数据集（经济、社会公益、交通）进行频域分析，发现配对文本嵌入的 lag-similarity 具有与时间序列高度一致的周期性
- 具体做法：用文本编码器对每个时间步的文本 $s_t$ 编码得到 $e_t$，计算滞后相似度 $d_l = \sum_t \cos(e_t, e_{t+L})$，再对 $d_l$ 做 FFT 分析主频率
- 三个原因解释 CTR：(1) 共享外部驱动（季节、经济周期等）；(2) 文本反映时间序列趋势；(3) 文本中包含具有对齐周期性的额外变量

### 2. TT-Wasserstein 度量
- 提出 TT-Wasserstein 指标量化 CTR 程度：计算时间序列与文本频谱归一化分布之间的 Wasserstein 距离
- 值越低表示文本与时间序列对齐度越高
- 验证：对 Time-MMD 数据集做时间戳打乱后，TT-Wasserstein 显著增大，证明该指标可以衡量模态对齐质量

### 3. Texts as Time Series (TaTS) 框架
整体流程分三步：

**步骤一：文本编码**  
用预训练语言模型（默认 GPT-2）对每个时间步文本编码：$e_t = \mathcal{H}_{\text{text}}(s_t) \in \mathbb{R}^{d_{\text{text}}}$

**步骤二：维度映射**  
通过三层 MLP 将高维文本嵌入映射到低维空间：$z_t = \text{MLP}(e_t) \in \mathbb{R}^{d_{\text{mapped}}}$

**步骤三：拼接与建模**  
将映射后的文本表征作为辅助变量与原始时间序列拼接：$U = [X; Z^\top] \in \mathbb{R}^{T \times (N + d_{\text{mapped}})}$，然后送入任意现有时间序列模型，联合训练 MLP 和时间序列模型参数

- 关键设计：文本被视为时间序列的辅助变量（auxiliary variables），不修改下游模型架构
- 预测时只取前 $N$ 个变量作为输出，丢弃辅助变量维度

## 实验关键数据

### 预测任务（9 个 Time-MMD 数据集 × 9 个模型）
- TaTS 在所有数据集上均优于 uni-modal 和 MM-TSFLib
- 6/9 数据集平均提升超过 5%，最大数据集 Environment 提升超过 30%
- Economy 数据集提升最显著：iTransformer MSE 从 0.014 降至 0.008（↓ 42.9%），Transformer MSE 从 0.584 降至 0.079（↓ 86.5%）

### 插补任务（Climate/Economy/Traffic）
- 最高提升 67.2%（Economy 数据集 PatchTST MAE）

### 与其他基线对比（Table 4）
- 显著优于 N-BEATS、N-HiTS、TCN 等协变量/卷积方法
- 优于 ChatTime（零样本多模态基础模型）和 GPT4MTS

### TT-Wasserstein 与性能提升的关系
- TT-Wasserstein 原始/打乱比越低，TaTS 提升越大（如 Economy 比值 22.3%，提升 64.8%）

### 效率
- MLP 仅增加约 1% 参数量，训练时间增加约 8%，但性能平均提升约 14%

## 亮点
- **洞察新颖**：首次发现并形式化 CTR 现象，为多模态时间序列提供了理论视角
- **简洁有效**：不修改任何下游模型架构，仅添加轻量 MLP 即可即插即用
- **通用性强**：兼容 9 种主流时间序列模型（Transformer/Linear/Frequency-based），支持预测和插补两类任务
- **度量实用**：TT-Wasserstein 可预判文本对建模的潜在增益，指导实际应用决策

## 局限与展望
- 文本编码器固定为预训练 LM（GPT-2），未探索端到端微调文本编码器的效果
- MLP 映射维度 $d_{\text{mapped}}$ 需手动设定，虽然实验表明不太敏感，但缺乏自动选择机制
- 当文本质量极低（如随机打乱）时，TaTS 可能略差于纯数值模型，虽然论文提供了丢弃低质量文本的缓解策略，但未实现自动检测
- 仅考虑时间戳级配对文本，未处理不规则或异步文本-时间序列配对场景
- 融合方式仅测试了 MLP/门控残差/交叉注意力，更复杂的融合策略可能进一步提升

## 与相关工作的对比
| 方法 | 特点 | 不足 |
|------|------|------|
| MM-TSFLib | 首个多模态时间序列库 | 忽略文本位置特性 |
| ChatTime | 零样本多模态推理 | 性能不及有监督 TaTS |
| N-BEATS/N-HiTS | 协变量建模 | 非针对文本设计，效果差 |
| StockNet/Dandelion | 金融领域文本融合 | 非时间戳对齐，通用性差 |
| **TaTS（本文）** | 即插即用，将文本视为辅助变量 | 需要时间戳对齐的配对文本 |

## 启发与关联
- CTR 现象本质上是 PRH 在时序多模态场景的具体实例，为探索其他模态（如图像、音频）与时间序列的对齐提供了思路
- TT-Wasserstein 作为数据质量度量，可推广到其他多模态场景中评估模态间对齐程度
- "文本即辅助变量" 的思路可能适用于其他异构数据融合场景（如将知识图谱嵌入作为时间序列的辅助变量）
- 随着 LM 规模增大（BERT → GPT-2 → LLaMA2）性能略有提升，暗示更强的文本编码器可进一步释放多模态潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ — CTR 现象和 TT-Wasserstein 度量具有原创性
- 实验充分度: ⭐⭐⭐⭐⭐ — 18 个数据集、9 个模型、完善的消融实验
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，图表丰富
- 价值: ⭐⭐⭐⭐ — 即插即用设计实用性强，但适用场景受限于配对文本可用性

<!-- RELATED:START -->

## 相关论文

- [Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)

<!-- RELATED:END -->
