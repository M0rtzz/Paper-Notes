---
title: >-
  [论文解读] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series
description: >-
  [ICML 2026][时间序列][时间序列预测] 通过 **MMD 检测分布漂移**并动态扩展异构专家池，结合**时间记忆路由器**保证选择一致性，Dynamic-TMoE 在九个时间序列基准上达到新的 SOTA——相比所有基线平均降低 MSE 10.4%、MAE 7.8%。
tags:
  - "ICML 2026"
  - "时间序列"
  - "时间序列预测"
  - "动态专家"
  - "分布漂移检测"
  - "混合专家网络"
  - "非平稳数据"
---

# Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series

**会议**: ICML 2026  
**arXiv**: [2605.20678](https://arxiv.org/abs/2605.20678)  
**代码**: 待确认  
**领域**: 时间序列 / 混合专家  
**关键词**: 时间序列预测, 动态专家, 分布漂移检测, 混合专家网络, 非平稳数据

## 一句话总结
通过 **MMD 检测分布漂移**并动态扩展异构专家池，结合**时间记忆路由器**保证选择一致性，Dynamic-TMoE 在九个时间序列基准上达到新的 SOTA——相比所有基线平均降低 MSE 10.4%、MAE 7.8%。

## 研究背景与动机

**领域现状**：时间序列预测是从能源管理到医疗监测等关键决策系统的基石。然而真实世界时间序列本质上非平稳，表现为持续的分布漂移和演化的时间依赖性。

**现有痛点**：现有方法主要分两类——一类是输入级标准化（RevIN、SAN、IN-Flow）通过移除-预测-恢复范式将非平稳输入映射到稳定分布，但往往丢弃对预测至关重要的非平稳信号；另一类是模型内适配（Non-stationary Transformer、Koopa、TimeStacker）通过重新设计注意力或利用物理 / 频谱动态捕捉演化特征，但单体架构缺乏模块性。最近的 MoE 方法（TFPS、Time-MoE）采用静态专家池和无记忆路由，无法适应突变的分布漂移。

**核心矛盾**：MoE 框架存在两个本质问题——（1）**时间刚性**：固定专家池无法容纳来自严重分布漂移的新模式，无记忆的网关忽视时间连续性导致选择不稳定；（2）**专门化不足**：同构专家缺乏功能多样性，无法解耦不同漂移分量（趋势 vs 季节性）。

**本文目标**：设计一个既能适应演化分布又能保持选择一致性的 MoE 框架。

**切入角度**：（1）分布漂移可通过 MMD 量化并触发专家池演化；（2）通过带历史记忆的 GRU 路由器维持时间连续性；（3）采用异构专家设计来专门化不同时间模式。

**核心 idea**：学习阶段动态扩展和修剪异构专家池（基于漂移检测），同时用支持异常状态库的时间记忆路由器保证上下文感知的稳定选择。

## 方法详解

### 整体框架
"感知-决策-适配"闭环——补丁嵌入 → 分布漂移检测器（感知）→ 时间记忆路由器（决策）→ 可进化专家管理器（适配）→ 异构专家池（基础专家：恒等、趋势、季节性、波动 + 按需实例化的漂移专家）。

### 关键设计

1. **MMD 漂移检测 + 自适应阈值**:

    - 功能：连续监测分布漂移，触发专家扩展。
    - 核心思路：采用核方法（RBF 核）定义参考窗口与当前窗口的 MMD 距离 $\mathcal{D}_{\text{mmd}}^2 = \frac{1}{N_r^2} \sum_{i, j} k(x_i^{\text{ref}}, x_j^{\text{ref}}) - \frac{2}{N_r N_c} \sum_{i, j} k(x_i^{\text{ref}}, x_j) + \frac{1}{N_c^2} \sum_{i, j} k(x_i, x_j)$。为应对时变噪声水平，采用动态阈值 $\epsilon = \mu_\mathcal{H} + \lambda \sigma_\mathcal{H}$（k-sigma 规则），当 $\mathcal{D}_{\text{mmd}}^2 > \epsilon$ 时触发专家管理器。
    - 设计动机：固定阈值无法适应波动的噪声水平，动态阈值提供鲁棒的漂移检测；通过泛化界证明 MMD 增加直接提升目标预测风险的上界。

2. **时间记忆路由器 + 异常状态库**:

    - 功能：在演化专家池上实现上下文感知的稳定专家选择，加速对重复漂移的适配。
    - 核心思路：用 GRU 维持隐状态 $\mathbf{h}_t = \text{GRU}(\phi(\mathbf{x}_{p, t}), \mathbf{h}_{t-1})$，将隐状态投影得每个专家的 logits，通过 Top-k 稀疏网关激活前 $k$ 个专家。异常状态库 $\mathcal{A}$ 存储漂移检测时的隐状态；路由时通过余弦相似度检索相关历史状态，用可学习网关 $\alpha$ 融合当前和历史状态 $\tilde{\mathbf{h}}_t = \alpha \mathbf{h}_t + (1 - \alpha) \mathbf{h}_{\text{ref}}$。
    - 设计动机：无记忆路由导致相邻补丁之间的专家选择不稳定；GRU 维持序列一致性；异常库加速对周期性漂移的适配避免冷启动。

3. **异构专家设计 + 漂移模式分析**:

    - 功能：用多样化的专家架构专门化不同时间成分（趋势、季节性、波动），并通过漂移模式分析器智能选择新增专家类型。
    - 核心思路：恒等专家 $E_{\text{id}} = \text{Linear}(\mathbf{X}_p)$；趋势专家 $E_{\text{trend}} = \text{MLP}(\text{AvgPool}(\mathbf{X}_p))$；季节性专家通过 FFT-MLP-iFFT 在频域处理 $\mathbf{Z} = \text{iFFT}(\text{MLP}(\text{FFT}(\mathbf{X}_p)))$；波动专家采用因果卷积 + 门控线性单元。漂移检测后，分析残差计算趋势 / 季节性 / 波动得分，自动实例化最匹配的专家类型。新专家冻结其他参数微调，避免破坏已有学习。
    - 设计动机：同构专家无法解耦复杂模式；异构设计赋予每个专家不同的感应偏差；漂移分析避免盲目扩展；冷启动对齐确保稳定集成。

## 实验关键数据

### 主实验

| 数据集 | 指标 | **Dynamic-TMoE** | TFPS | ST-MTM | RAFT | 提升 vs TFPS |
|--------|------|-------------|------|--------|------|------------|
| Weather | MSE | **0.240** | 0.241 | 0.262 | 0.271 | ↓0.4% |
| Exchange | MSE | **0.351** | 0.395 | 0.408 | 0.432 | ↓11.1% |
| ETTh1 | MSE | **0.429** | 0.448 | 0.432 | 0.432 | ↓4.2% |
| Electricity | MSE | **0.170** | 0.183 | 0.208 | 0.184 | ↓7.1% |
| ILI | MSE | **1.981** | 2.642 | 2.820 | 5.916 | ↓25.0% |

18 个评估指标中排名前 2 达 16 次（第 1 名 11 次，第 2 名 5 次）。

### 消融实验

| 配置 | 漂移感知 | 时间记忆 | 异常库 | ETTh1 MSE | Weather MSE |
|------|--------|--------|-------|-----------|-------------|
| ① 完整 | ✓ | GRU | ✓ | **0.429** | **0.240** |
| ② 无漂移 | ✗ | GRU | ✓ | 0.436 | 0.246 |
| ③ 线性路由 | ✓ | Linear | ✓ | 0.436 | 0.247 |
| ④ MLP 路由 | ✓ | MLP | ✓ | 0.438 | 0.245 |
| ⑤ 无异常库 | ✓ | GRU | ✗ | 0.438 | 0.246 |
| 异构 → 同构 | ✓ | GRU | ✓ | 0.440 | 0.246 |

### 关键发现
- 漂移感知和时间连续性都是处理非平稳性的必要条件；去掉任一都导致显著退化。
- GRU 路由相比静态路由提升 1.6%-2.1%——序列记忆对非平稳环境下的专家分配至关重要。
- 异构专家相比同构设计提升 2.5%——专家多样性有效解耦复杂分布。
- 关系层（循环相关建模）提升 2.8%。
- 异常状态库贡献相对较小（< 1%），但在周期性漂移数据上稳定性提升显著。

## 亮点与洞察
- **动态-异构-记忆的三位一体**：框架统一了三个维度——架构演化（动态专家）、功能多样性（异构设计）、时间一致性（记忆路由）；相比只关注其中某一维的现有工作，这种结合处理了非平稳预测的本质矛盾。
- **MMD 作为漂移信号的妙用**：相比人工设定固定阈值或简单的残差监控，核方法更能捕捉复杂分布变化；动态阈值（k-sigma）避免过度敏感。
- **冷启动对齐的工程智慧**：新专家添加后冻结现有参数、仅在漂移数据上微调，巧妙平衡新旧知识整合，避免灾难性遗忘。
- **频域处理季节性的经验设计**：FFT-MLP-iFFT 再加周期激活，是一个可复用的时间序列分解模式。

## 局限与展望
- 计算开销未充分讨论：MMD 计算、GRU 推理、漂移时新增专家的成本在大规模数据或长序列下的影响未给出具体数据。
- 超参数敏感性不完整：漂移阈值 $\lambda$、融合系数 $\alpha$ 等的敏感性分析仅在补充材料中。
- 泛化边界条件：什么样的漂移模式、多大的分布变化该框架仍然有效的理论刻画还不够清晰。
- 改进：在线学习范式；自适应超参；针对超长序列分析内存使用与索引效率。

## 相关工作与启发
- **vs RevIN / IN-Flow**：处理分布不变性但丢弃非平稳信号；Dynamic-TMoE 通过异构专家在模型内部保留并利用这些信号。
- **vs Non-stationary Transformer**（单体设计）：单体架构难以专门化多个共存模式；本文通过 MoE 分解复杂性。
- **vs TFPS / Time-MoE**（现有 MoE）：采用静态同构专家和无状态路由；Dynamic-TMoE 加入动态异构池和时间记忆。
- **vs Koopa / DERITS**（频谱方法）：全局频率处理；本文季节性专家在频域处理后回到时域细化，混合策略更灵活。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  把 MoE 框架从静态扩展到动态、从同构扩展到异构、从无状态路由扩展到时间记忆路由。
- 实验充分度: ⭐⭐⭐⭐⭐  9 个数据集 + 多基线对比 + 完整消融 + 多预测长度，证据充分。
- 写作质量: ⭐⭐⭐⭐  逻辑清晰，方法描述详细；图表丰富但部分细节放在附录。
- 价值: ⭐⭐⭐⭐⭐  解决了非平稳时间序列预测的核心问题——分布漂移与时间连续性，对工业级实际应用（金融、能源、医疗）有重要价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[AAAI 2026\] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](../../AAAI2026/time_series/loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)

</div>

<!-- RELATED:END -->
