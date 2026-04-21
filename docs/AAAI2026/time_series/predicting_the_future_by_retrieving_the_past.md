---
title: >-
  [论文解读] Predicting the Future by Retrieving the Past
description: >-
  [AAAI 2026][时间序列][检索增强预测] 提出 PFRP（Predicting the Future by Retrieving the Past），构建全局记忆库(GMB)存储历史模式，通过预测性对比学习训练编码器实现高效检索，将检索到的全局预测与任意局部预测模型动态融合，在 7 个数据集上平均提升 8.4% 的预测性能。
tags:
  - AAAI 2026
  - 时间序列
  - 检索增强预测
  - 全局记忆库
  - 对比学习
  - 单变量时间序列
  - 即插即用
---

# Predicting the Future by Retrieving the Past

**会议**: AAAI 2026  
**arXiv**: [2511.05859](https://arxiv.org/abs/2511.05859)  
**代码**: [github.com/ddz16/PFRP](https://github.com/ddz16/PFRP)  
**领域**: 时间序列  
**关键词**: 检索增强预测, 全局记忆库, 对比学习, 单变量时间序列, 即插即用

## 一句话总结

提出 PFRP（Predicting the Future by Retrieving the Past），构建全局记忆库(GMB)存储历史模式，通过预测性对比学习训练编码器实现高效检索，将检索到的全局预测与任意局部预测模型动态融合，在 7 个数据集上平均提升 8.4% 的预测性能。

## 研究背景与动机

当前深度学习时间序列预测模型（MLP、Transformer、TCN等）基于滑动窗口训练后，会将历史信息隐式压缩到模型参数中。训练完成后原始训练数据被丢弃，推理时模型只能利用当前 lookback window 内的局部上下文，无法显式访问全局历史知识。

**关键观察**：时间序列常包含跨周期高度相似的子序列。例如，家庭用电数据中 2019 年某周的消费模式与 2018 年的某周极为相似——都呈现日周期波动，且前3天峰值递减、后4天高位稳定。这意味着如果当前 lookback window 与历史某段相似，其未来走势也很可能类似。

**核心动机**：
1. 现有模型属于"局部预测模型"——只看当前窗口，浪费了丰富的全局历史模式
2. RAG（检索增强生成）在 NLP 中已证明有效，但现有时间序列 RAG 方法（RATD、TimeRAF、TimeRAG）要么依赖扩散模型/LLM导致效率低下，要么需要遍历整个训练集导致检索缓慢
3. 需要一种高效、模型无关的检索增强预测框架

## 方法详解

### 整体框架

PFRP 分两个阶段：
- **阶段1 — 构建全局记忆库(GMB)**：训练对比学习编码器 → 编码所有历史样本的 lookback window → K-medoids 聚类降冗余 → 存储 $K$ 个代表性(特征, 预测序列)对
- **阶段2 — 检索增强预测**：编码当前 lookback window → 从 GMB 检索 top-k 相似历史 → 置信门+输出门调制 → 与局部预测模型动态融合

### 关键设计

1. **预测性对比学习(PCL)**：训练 lookback window 编码器时，正样本的选择方式是关键创新。不同于传统对比学习按 lookback 序列的 MSE 选正样本，PCL **按 prediction horizon 序列的 MSE 选正样本**：

   $$i^+ = \arg\min_{1 \leq j \leq B, j \neq i} \|y_i - y_j\|_2^2$$

   然后用 InfoNCE 损失拉近正样本对在特征空间的距离：

   $$\mathcal{L}_{pcl} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\epsilon^{(i)} \cdot \epsilon^{(i^+)}/\tau)}{\sum_{j=1}^{B} \exp(\epsilon^{(i)} \cdot \epsilon^{(j)}/\tau)}$$

   **设计动机**：我们关心的不是"哪些过去看起来相似"，而是"哪些过去有相似的未来"。按未来走势选正样本使编码器学到的特征更直接服务于检索目的——找到未来最可能相似的历史段。此外，为避免时间重叠样本成为伪正样本，排除了与锚样本时间重叠>48步的样本。

2. **K-medoids 聚类构建 GMB**：编码所有训练样本后，在特征空间做 K-medoids 聚类，仅保留 $K$ 个聚类中心点作为记忆单元 $\{(\epsilon^{(i)}, y^{(i)})\}_{i=1}^{K}$。

   **设计动机**：
   - 去冗余：避免存储重复相似的历史样本，提升检索效率
   - K-medoids 而非 K-means：K-medoids 使用真实历史样本而非合成平均作为聚类中心，保证记忆库中的模式是真实、连贯的历史序列
   - GMB 按最长预测长度 720 构建，短预测只取前几步即可，无需为不同预测长度重建

3. **置信门 + 输出门 + 动态融合**：检索到 top-k 最相似历史后，经过三重调制生成最终预测：

   **置信门(Confidence Gate)**：判断检索结果与当前窗口拼接后是否构成合理序列：
   $$p_i = \text{Sigmoid}(\text{MLP}([x; y^{(a_i)}]))$$
   用此概率调制原始相似度权重，确保"看起来相似但未来不匹配"的历史被降权。

   **输出门(Output Gate)**：对加权融合后的全局预测 $\bar{y}_1$ 做仿射变换以适应当前的 scale 和 shift：
   $$y_1 = \alpha \cdot \bar{y}_1 + \beta$$
   其中 $\alpha, \beta \in \mathbb{R}^H$ 由 MLP 从当前 lookback window 生成，$\alpha$ 初始化为全1，$\beta$ 初始化为全0。

   **动态融合**：根据检索的相似度权重动态决定全局预测 $y_1$ 和局部预测 $y_2$ 的权重：
   $$y = w_1 \cdot y_1 + w_2 \cdot y_2, \quad w_1, w_2 = \text{Softmax}(\text{MLP}(\bar{w}^{(a_1)}, \ldots, \bar{w}^{(a_k)}))$$

   **设计动机**：当历史中无高度相似序列时，调制权重偏小，融合自动倾向局部模型；周期性强的数据中调制权重偏大，全局预测占主导。

### 损失函数 / 训练策略

- **阶段1**（PCL训练）：batch size 256，温度 $\tau=0.05$，学习率 0.001，排除时间重叠>48的样本
- **阶段2**（PFRP训练）：Adam 优化器，L2 损失，初始学习率 0.0001，遵循各基线模型的官方超参数
- **公平比较**：无论是否使用 PFRP，基线模型的超参数和训练配置保持不变
- **lookback window** $L=96$，预测长度 $H \in \{96, 192, 336, 720\}$

## 实验关键数据

### 主实验

7 个数据集的单变量预测（取最后一个变量），所有预测长度平均结果：

| 基线模型 | 原始 MSE | +PFRP MSE | 提升 |
|----------|----------|-----------|------|
| SparseTSF | 0.2404 (Traffic) | 0.1919 | **20.2%** |
| DLinear | 0.2778 (Traffic) | 0.1793 | **35.5%** |
| PatchTST | 0.1797 (Traffic) | 0.1712 | **4.7%** |
| TimesNet | 0.2165 (Traffic) | 0.1799 | **16.9%** |
| SparseTSF | 0.0841 (ETTh1) | 0.0766 | **8.9%** |
| DLinear | 0.3951 (Electricity) | 0.3666 | **7.2%** |

跨7个数据集的平均提升：SparseTSF +8.4%，DLinear +7.1%，PatchTST 和 TimesNet 提升略小但一致。周期性强的数据集（Traffic +17.4%, Electricity +10.1%）提升最大。

### 消融实验

| 配置 | Traffic MSE | Electricity MSE | 说明 |
|------|-------------|-----------------|------|
| SparseTSF (基线) | 0.2404 | 0.4968 | 无PFRP |
| +PFRP 完整 | **0.1919** | **0.3561** | 所有模块 |
| w/o confidence gate | 0.2385 | 0.3960 | 置信门的贡献 |
| w/o output gate | 0.2130 | 0.5140 | 输出门的贡献 |
| w/o both gates | 0.2128 | 0.5763 | 两个门都去掉 |
| w/o prediction model | **0.1686** | 0.3952 | 仅全局预测 |

GMB 相关消融：

| 维度 | 最优选择 | 对比 | 说明 |
|------|----------|------|------|
| 检索准则 | Feature cosine (本文) | MSE/DTW/PCC | 特征级相似度优于原始序列级 |
| 编码器类型 | MLP (默认) | PatchTST/TimesNet | 各有优劣，MLP 效率最高 |
| 训练策略 | PCL (本文) | CL/PL | 按未来相似度选正样本效果最好 |

### 关键发现

1. **周期性越强收益越大**：Traffic/Electricity 数据集周期性分别为 0.32/0.19，全局预测权重 $w_1$ 更高，收益最显著
2. **仅全局预测也能超越基线**：在 Traffic 上 w/o prediction model 的 MSE(0.1686)甚至优于完整 PFRP(0.1919)，说明强周期数据中历史检索本身就足够有效
3. **简单模型收益更大**：DLinear/SparseTSF 等 MLP 模型的提升大于更复杂的 PatchTST/TimesNet
4. **可增强大模型**：冻结 TimeCMA/Moirai/Sundial 预训练参数，仅微调 PFRP 参数也能提升性能
5. **效率开销极小**：GMB 构建一次性耗时 186 秒（PCL 134s + 聚类 52s），模型大小仅增加 1.57MB

## 亮点与洞察

- **模型无关的即插即用架构**：PFRP 可无缝集成到任何单变量预测模型中，无需修改基线模型的超参数
- **PCL 正样本选择策略新颖**：按未来走势而非历史形态选正样本，让编码器学到的特征空间更匹配检索目标
- **动态融合自适应周期性**：全局预测权重与数据周期性成正比，自动在"检索驱动"和"模型驱动"之间切换
- **可解释性**：检索到的历史序列直观展示了"为什么模型做出这个预测"，提供了局部可解释能力

## 局限与展望

1. **仅限单变量**：PFRP 目前只处理单变量时间序列，扩展到多变量需要重新设计检索和融合机制
2. **超参数 $K$ 和 $k$ 需调优**：GMB 大小 $K$ 和检索数 $k$ 在不同数据集上最优值差异较大（$K$ 从 1000 到 4000，$k$ 从 10 到 200）
3. **弱周期数据收益有限**：ETT 数据集（周期性<0.13）的平均提升仅 2~3%
4. **GMB 静态不更新**：训练集构建完成后 GMB 固定不变，无法适应数据分布漂移

## 相关工作与启发

- **RAG 在时间序列中的应用**：RATD 使用扩散模型、TimeRAF 使用基础模型、TimeRAG 使用 LLM——都需要重量级推理。PFRP 用简单 MLP 编码器+GMB 检索实现了更高效的检索增强
- **记忆增强网络**：GMB 可视为外部记忆的一种形式，类似 Neural Turing Machine 但更简单高效
- **K-medoids 的巧妙应用**：保留真实历史样本（而非合成平均）作为记忆单元，保证了检索结果的可解释性

## 评分

- 新颖性: ⭐⭐⭐⭐ — PCL + GMB + 动态融合的组合在时间序列RAG中有独特价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 7 数据集×4 基线，全面的 GMB/PFRP 消融，超参数敏感性，大模型增强实验
- 写作质量: ⭐⭐⭐⭐ — 图示清晰，方法描述完整，对比分析到位
- 价值: ⭐⭐⭐⭐ — 即插即用、开销小、效果稳定；但单变量限制了实用范围

<!-- RELATED:START -->

## 相关论文

- [Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [A2P: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](../../ICML2025/time_series/when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_seri.md)
- [When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](../../ICML2025/time_series/when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)
- [Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)

<!-- RELATED:END -->
