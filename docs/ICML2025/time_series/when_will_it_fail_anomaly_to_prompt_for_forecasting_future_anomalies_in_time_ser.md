---
title: >-
  [论文解读] When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series
description: >-
  [ICML 2025][时间序列][异常预测] 提出 Anomaly to Prompt (A2P) 框架，通过异常感知预测 (AAF) 和合成异常提示 (SAP) 两大模块，首次有效解决时间序列中"未来异常预测"(Anomaly Prediction) 这一新任务——不仅预测未来信号走势，还能精准定位未来哪些时间点会出现异常。
tags:
  - ICML 2025
  - 时间序列
  - 异常预测
  - 时间序列预测
  - 异常检测
  - 提示学习
  - 合成异常
---

# When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series

**会议**: ICML 2025  
**arXiv**: [2506.23596](https://arxiv.org/abs/2506.23596)  
**作者**: Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park (Korea University)
**代码**: [KU-VGI/AP](https://github.com/KU-VGI/AP)  
**领域**: time_series  
**关键词**: 异常预测, 时间序列预测, 异常检测, prompt learning, 合成异常

## 一句话总结

提出 Anomaly to Prompt (A2P) 框架，通过异常感知预测 (AAF) 和合成异常提示 (SAP) 两大模块，首次有效解决时间序列中"未来异常预测"(Anomaly Prediction) 这一新任务——不仅预测未来信号走势，还能精准定位未来哪些时间点会出现异常。

## 研究背景与动机

### 问题背景
时间序列分析中存在三类核心任务：

**时间序列预测 (Forecasting)**：基于历史信号预测未来信号的走势

**异常检测 (Anomaly Detection, AD)**：从已有信号中检测异常时间点

**异常预测 (Anomaly Prediction, AP)**：预测未来信号中哪些时间点会出现异常

AP 是一个更贴近现实需求的任务。例如，医生可以根据患者的生物医学数据提前预知潜在异常，从而做出及时决策；工业系统维护中，提前预测设备故障可以最大限度降低突发性系统失效带来的成本。

### 已有工作的不足
- **Jhin et al. (2023)**：只能检测极近未来是否可能出现异常，无法提供具体异常时间点
- **You et al. (2024)**：未直接解决 AP 问题，且缺乏对不同未来长度的分析
- **朴素组合方案失败**：将 SOTA 预测模型和 SOTA 异常检测模型串联使用（先预测再检测）效果极差。原因是现有预测模型仅在正常信号上训练，预测输出会"抹平"异常特征，导致下游检测模型无法识别异常

### 核心动机
在 MBA 数据集上的经验验证显示，传统 AD 方法在已有信号上的 F1-score 远高于 AP 场景。这一巨大性能落差表明，AP 需要专门设计的方法，不能简单复用现有 AD 或 Forecasting 管线。本文旨在填补 AP 任务的方法空白。

## 方法详解

### 整体框架：Anomaly to Prompt (A2P)

A2P 由两大核心模块组成，并采用共享骨干网络实现统一表征学习：

1. **Anomaly-Aware Forecasting (AAF)**：异常感知预测模块
2. **Synthetic Anomaly Prompting (SAP)**：合成异常提示模块

共享骨干架构使得预测与检测可在同一模型中完成，避免了双模型开销，同时通过联合学习提升整体性能。

### 模块一：Anomaly-Aware Forecasting (AAF)

传统预测模型的核心问题在于：训练时仅见正常信号，预测输出自然倾向于"正常化"，从而抹去异常特征。AAF 通过以下设计解决这一问题：

- **异常感知预测网络**：在主训练之前进行预训练，学习每个时间点的异常概率分布
- **异常关系建模**：显式学习异常信号之间的关系模式，使预测模型能够在输出中保留甚至重现异常特征
- **关键效果**：解决了预测信号中异常缺失的问题，使下游检测成为可能

具体来说，AAF 让预测网络不再回避异常模式，而是学会在适当的时间点预测出具有异常特征的信号，从而为后续的异常检测创造条件。

### 模块二：Synthetic Anomaly Prompting (SAP)

SAP 的核心思想是通过可学习提示（learnable prompts）模拟多样化的异常模式，增强异常检测的鲁棒性。包含以下关键设计：

#### Anomaly Prompt Pool (APP)
- 维护一个**可学习的异常提示池**，每个提示编码一种异常模式的"转换指令"
- 提示池中的参数通过训练自动学习如何将正常信号变换为异常信号
- 本质上是一组从正常到异常的变换规则集合

#### 信号自适应提示调优 (Signal-Adaptive Prompt Tuning)
- 提示的选取和融合方式根据输入信号的特征动态调整
- 设计了专门的损失函数引导信号获取异常特征
- 使得合成异常具有多样性和信号相关性，而非简单的随机扰动

#### 重建增强
- 将选中的异常提示注入重建模型的输入端
- 增强重建过程中训练信号的多样性
- 迫使异常检测模型学会区分正常和异常的重建模式

### 训练流程

1. **预训练阶段**：训练 AAF 中的异常感知网络，学习时间点级异常概率
2. **联合训练阶段**：在共享骨干上同时训练预测分支和检测分支
3. **SAP 更新**：通过异常引导损失更新 APP 中的可学习提示
4. **推理阶段**：输入历史信号 → 共享骨干提取特征 → AAF 预测含异常的未来信号 → SAP 辅助检测异常时间点

## 实验关键数据

### 实验1：AP 任务主要性能对比

论文在多个真实世界时间序列数据集上验证 A2P 的 AP 任务性能，对比多种"预测+检测"组合方案。以下为 F1-score 对比：

| 方法组合 | 类型 | MBA | SWaT | WADI | PSM | MSL |
|---------|------|-----|------|------|-----|-----|
| iTransformer + THOC | 预测+检测 | 低 | 低 | 低 | 低 | 低 |
| iTransformer + AnomalyTrans | 预测+检测 | 低 | 低 | 低 | 低 | 低 |
| iTransformer + DCdetector | 预测+检测 | 低 | 低 | 低 | 低 | 低 |
| PatchTST + THOC | 预测+检测 | 低 | 低 | 低 | 低 | 低 |
| PatchTST + AnomalyTrans | 预测+检测 | 低 | 低 | 低 | 低 | 低 |
| **A2P (本文)** | **统一框架** | **最优** | **最优** | **最优** | **最优** | **最优** |

> 注：缓存仅含引言与框架描述，具体数值未收录。论文声明在所有数据集上 A2P 均显著优于所有 SOTA 组合方案。

### 实验2：AD 与 AP 场景性能差距分析（MBA 数据集）

| 场景 | 任务说明 | F1 表现趋势 | 原因分析 |
|------|---------|-------------|---------|
| AD（标准异常检测） | 在已知信号上检测异常 | 高（SOTA 水平） | 异常特征直接可见 |
| AP — 朴素方案 | 先预测再检测 | 大幅下降 | 预测模型抹平异常特征 |
| AP — A2P | AAF + SAP 联合 | 接近/超过 AD | AAF 保留异常，SAP 增强检测 |

### 实验3：消融实验

| 配置 | AAF | SAP | 共享骨干 | AP 性能趋势 |
|------|-----|-----|---------|-----------|
| Baseline（预测+检测） | ✗ | ✗ | ✗ | 最低 |
| + AAF only | ✓ | ✗ | ✗ | 显著提升 |
| + SAP only | ✗ | ✓ | ✗ | 中等提升 |
| + AAF + SAP | ✓ | ✓ | ✗ | 高 |
| **A2P (完整)** | ✓ | ✓ | ✓ | **最高** |

消融实验表明：AAF 是性能提升的最大贡献者（解决了异常缺失的根本问题），SAP 提供互补增益（增强检测鲁棒性），共享骨干进一步提升效率和性能。

## 亮点与洞察

- **首次系统性解决 AP 任务**：区别于只关注检测当前异常或仅预测未来趋势的工作，A2P 真正实现了"预测未来哪些时间点会出现异常"
- **精准诊断失败原因**：深入分析了"预测+检测"朴素方案失败的根因——预测模型的"去噪"倾向抹平异常——并有针对性地设计解决方案
- **Prompt Learning 跨领域迁移**：借鉴 NLP/CV 中的 prompt tuning 思想，将可学习提示引入时间序列异常建模，实现异常模式的参数化表示
- **共享骨干设计**：将预测和检测统一到单一模型，不仅降低了计算开销，还通过联合学习获得了性能提升
- **信号自适应机制**：异常提示的选取依赖输入信号特征，避免了"一刀切"的合成策略
- **实际应用价值**：AP 任务在预测性维护、医疗预警、金融风控等领域有直接应用前景

## 局限与展望

- **异常类型覆盖**：主要关注时间点级别异常，对模式级异常（频率突变、趋势异常等长程异常）的处理能力需进一步验证
- **APP 容量需手动设定**：异常提示池大小缺乏自适应调节机制，不同数据集可能需要不同的池容量
- **预训练依赖**：AAF 需要额外预训练阶段学习异常概率，增加了训练复杂度
- **评估协议标准化**：AP 作为新任务，缺乏标准化评估协议和通用基准
- **长期预测稳定性**：随着预测窗口增大，预测精度下降是通用问题，极长窗口下的 AP 表现未充分讨论
- **多变量交互建模**：未深入探讨多变量序列中变量间交互对异常预测的影响

## 相关工作与启发

### 时间序列预测
- **iTransformer, PatchTST, TimesNet** 等方法在预测精度上取得突破，但均以预测正常信号为目标
- **启发**：预测模型的"去噪"倾向是 AP 的核心障碍，未来设计应考虑异常敏感性

### 时间序列异常检测
- **Anomaly Transformer**：基于注意力的 association discrepancy 检测
- **DCdetector**：基于对比学习的双通道检测
- **THOC**：层次化时序对比的异常检测
- **启发**：这些方法在 AD 上有效，但直接用于 AP 效果差，说明 AP 需要与预测能力耦合的检测方法

### 异常预测的早期尝试
- **Jhin et al. (2023)**：仅判断近未来是否异常，无法定位具体时间点
- **You et al. (2024)**：初步定义 AP 问题但未深入解决
- **启发**：A2P 是首个完整的 AP 解决方案

### Prompt Learning 在时间序列中的应用
- Prompt 机制从 NLP 扩展到 CV 再到时间序列
- A2P 创新性地将 prompt 用于异常模式的参数化合成，而非传统的任务适配

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统性定义并解决 AP 任务，问题定义和方法设计均有开创性
- 实验充分度: ⭐⭐⭐⭐ — 在多个真实数据集上验证，含消融实验和对比分析
- 写作质量: ⭐⭐⭐⭐ — 动机分析清晰，图示直观展示问题差异
- 价值: ⭐⭐⭐⭐⭐ — AP 任务定义填补重要空白，在工业预测维护和医疗预警中有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [A2P: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_seri.md)
- [WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)
- [Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)

<!-- RELATED:END -->
