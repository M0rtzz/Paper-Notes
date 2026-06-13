---
title: >-
  [论文解读] TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation
description: >-
  [ICML 2025][时间序列][无监督域适应] 提出 TransPL，通过将时间序列 patch 离散化为 VQ 码并构建类别-通道级转移矩阵，利用贝叶斯定理在目标域生成可解释伪标签，实现时间序列无监督域适应中平均 6.1% 准确率和 4.9% F1 的提升。 时间序列数据在不同域（如不同用户、不同设备）之间常表现出强烈…
tags:
  - "ICML 2025"
  - "时间序列"
  - "无监督域适应"
  - "伪标签"
  - "向量量化"
  - "转移矩阵"
  - "最优传输"
---

# TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation

**会议**: ICML 2025  
**arXiv**: [2505.09955](https://arxiv.org/abs/2505.09955)  
**代码**: [有](https://github.com/eai-lab/TransPL)  
**领域**: 时间序列  
**关键词**: 无监督域适应, 伪标签, 向量量化, 转移矩阵, 最优传输

## 一句话总结

提出 TransPL，通过将时间序列 patch 离散化为 VQ 码并构建类别-通道级转移矩阵，利用贝叶斯定理在目标域生成可解释伪标签，实现时间序列无监督域适应中平均 6.1% 准确率和 4.9% F1 的提升。

## 研究背景与动机

时间序列数据在不同域（如不同用户、不同设备）之间常表现出强烈的分布差异，主要体现在两个层面：**时间动态转变**和**通道级（传感器级）偏移**。例如在人体活动识别中，加速度计的重力分量可能跨用户稳定，但陀螺仪读数可能因佩戴松紧不同而发生剧烈偏移。

现有的伪标签策略（如 Softmax、NCP、SHOT、T2PL 等）大多源自图像领域，存在两个核心问题：

**忽略时序动态**：将时间序列当作静态数据处理，未建模时间维度上的状态转变模式

**忽略通道级偏移**：多通道时间序列中各传感器受域偏移的影响程度不同，但现有方法未做区分

**核心矛盾**在于：时间序列域适应需要同时建模时序转变和选择性通道偏移，但现有方法将其作为黑盒操作，既不显式建模也不提供可解释性。

**本文方案**：TransPL 将时间序列分割为 patch，通过向量量化（VQ）将 patch 映射为离散码，构建**类别级转移矩阵**（建模源域各类的时序模式）和**通道级转移矩阵**（量化各通道的域偏移），利用贝叶斯定理生成加权伪标签。这不仅实现了显式的时序-通道联合建模，还使伪标签生成过程具备可解释性。

## 方法详解

### 整体框架

TransPL 包含三个阶段：

**阶段 A — 源域训练**：使用有标签的源域数据训练编码器（Transformer）、解码器、粗-细双码本和分类器。时间序列 $\mathbf{X} \in \mathbb{R}^{D \times T}$ 被分割为 $N = \lfloor T/m \rfloor$ 个 patch，编码后通过 VQ 映射为离散码。

**阶段 B — 转移矩阵构建**：冻结模型参数，从源域和目标域推断粗码序列，构建三组转移矩阵：
- 源域类别级 TM：$\mathbf{P}_{\text{cl}}^{\mathcal{S}} \in \mathbb{R}^{K \times D \times n_c \times n_c}$
- 源域通道级 TM：$\mathbf{P}_{\text{ch}}^{\mathcal{S}} \in \mathbb{R}^{D \times n_c \times n_c}$
- 目标域通道级 TM：$\mathbf{P}_{\text{ch}}^{\mathcal{T}} \in \mathbb{R}^{D \times n_c \times n_c}$

**阶段 C — 伪标签生成**：基于类别级 TM 计算通道级类条件似然，结合通道对齐权重和先验分布，通过贝叶斯定理生成最终伪标签。

### 关键设计

1. **粗-细双码本（Coarse-Fine Codebook）**: 受经典时间序列加性分解（趋势+残差）启发，设计了两级码本结构。粗码本 $\mathcal{C}_c$（$n_c = 8$ 个码）捕获 patch 的短期趋势，细码本 $\mathcal{C}_f$（$n_f = 64$ 个码）捕获残差细节。量化过程为：
    $\tilde{c} = \arg\min_c \|\ell_2(\mathbf{z}) - \ell_2(\mathbf{e}_c)\|_2^2, \quad \mathbf{e}_c \in \mathcal{C}_c$
    $\tilde{f} = \arg\min_f \|\ell_2(\mathbf{z}) - \ell_2(\mathbf{e}_{\tilde{c}}) - \ell_2(\mathbf{e}_f)\|_2^2, \quad \mathbf{e}_f \in \mathcal{C}_f$
   置换熵（Permutation Entropy）分析验证了粗码确实捕获了更简单的全局趋势（低 PE），细码编码了更复杂的残差模式（高 PE）。关键优势是 $n_c \ll n_f$ 使转移矩阵计算可行且无死码（dead codes）。

2. **VQ 码转移矩阵（Transition Matrices）**: 将粗码序列视为离散马尔可夫链，转移概率为：
    $p(s_{t+1} = \mathbf{e}_j | s_t = \mathbf{e}_i) = \frac{\text{count}(\mathbf{e}_i, \mathbf{e}_j)}{\text{count}(\mathbf{e}_i)}$
   **类别级 TM** 从源域有标签数据按类别和通道分别统计，用于给定类别 $k$ 时目标序列的类条件似然计算（类似 HMM 的极大似然估计）。**通道级 TM** 分别从源域和目标域不区分类别地构建，用于后续通道对齐打分。

3. **通道对齐与贝叶斯伪标签**: 伪标签核心公式为加权通道级类后验的聚合：
    $\hat{y}_k = \frac{1}{D} \sum_{d=1}^{D} w_d \frac{p(\mathbf{X}^d | y=k) \, p(k)}{\sum_{c=1}^{K} p(\mathbf{X}^d | y=c) \, p(c)}$
   其中通道对齐分数 $w_d$ 通过最优传输计算：先求源域和目标域通道 TM 行之间的 Earth Mover's Distance（代价矩阵为码间余弦距离），再经 RBF 核转为对齐分数：
    $w_d = \exp\left(-\left(\frac{1}{n_c}\sum_{i=1}^{n_c}\langle\gamma_i^*, \mathbf{M}\rangle\right)^2 / \sigma^2\right)$
   直觉是：偏移较小的通道权重更高。类条件对数似然：
    $\log p(\mathbf{X}^d | y=k) = \frac{1}{N}\sum_{t=1}^{N-1}\log p(s_{t+1} | s_t, y=k)$

### 损失函数 / 训练策略

**源域训练损失**：
$$\mathcal{L}_{\text{src}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{VQ}}, \quad \mathcal{L}_{\text{VQ}} = \mathcal{L}_{\text{code}} + \mathcal{L}_{\text{rec}}$$

- $\mathcal{L}_{\text{code}}$：标准 VQ 损失含 stop-gradient 和 commitment loss（$\beta=0.25$），同时优化粗码和细码
- $\mathcal{L}_{\text{rec}}$：解码器从 $\mathbf{e}_c + \mathbf{e}_f$ 重建原始时间序列的 MSE
- $\mathcal{L}_{\text{ce}}$：[CLS] token 的分类交叉熵

**目标域微调损失**：
$$\mathcal{L}_{\text{trg}} = \lambda_1 \mathcal{L}_{\text{ce}} + \lambda_2 \mathcal{L}_{\text{VQ}}$$

微调时每个 mini-batch 选取置信度前 $r_{\text{top}}$ 比例的伪标签样本。$\lambda_1, \lambda_2$ 可用多任务学习的可学习权重自动调节。

**弱监督扩展**：当已知目标域标签分布时，可直接将其作为先验 $p(k)$ 加入贝叶斯公式，使用对数先验 $\log p(k) / \tau$ 并以温度 $\tau$ 调节强度。

## 实验关键数据

### 主实验

| 数据集 | 指标 | TransPL | 之前SOTA (SHOT) | 提升 |
|--------|------|---------|-----------------|------|
| UCIHAR | Acc | **69.0** | 67.8 | +1.2 |
| UCIHAR | MF1 | **64.9** | 64.3 | +0.6 |
| WISDM | Acc | **64.0** | 62.2 | +1.8 |
| WISDM | MF1 | **56.2** | 54.6 | +1.6 |
| HHAR | Acc | **68.4** | 64.8 | +3.6 |
| HHAR | MF1 | **65.3** | 63.2 | +2.1 |
| PTB | Acc | **67.2** | 61.6 | +5.6 |
| PTB | MF1 | **74.0** | 66.9 | +7.1 |

伪标签准确率方面，TransPL 相比所有基线平均提升 **6.1% 准确率** 和 **4.9% F1**。加入弱监督后提升进一步扩大至 10.7% 和 5.2%。

### 消融实验

| 配置 | UCIHAR Acc | HHAR Acc | PTB Acc | 说明 |
|------|-----------|---------|---------|------|
| 无 CA 无 WS | 68.0 | 63.2 | 68.3 | 基线 |
| +CA | 69.0 | 68.4 | 67.2 | 通道对齐有效 |
| +WS | 68.6 | 62.3 | 72.2 | 弱监督有效 |
| +CA +WS | **71.2** | **70.4** | **72.4** | 联合使用最优 |

码本配置消融（HHAR）：粗 8 + 细 64 最优（PL Acc 68.4%，0% 死码），单一大码本（128）死码率高达 66.8%。

### 关键发现

- TransPL 在 4 个数据集上全面超越所有 DA 方法和伪标签方法，无适应基线平均提升 11.4% Acc 和 12.2% MF1
- 与判别式模型（1D-CNN、LSTM、GRU）直接建模粗码序列相比，TransPL 的生成式转移矩阵方法显著更优，表明 TM 能更好捕获跨域不变的时序动态
- CoDATS 的弱监督在多个数据集上反而降低性能（如 PTB 下降 5.3%），而 TransPL 通过贝叶斯先验实现了所有数据集的一致提升
- 通道对齐分析表明基于原型距离的方法无法准确测量通道偏移，而基于最优传输的方法能给出良好校准的距离

## 亮点与洞察

1. **离散化建模时序联合分布的新范式**：将连续时间序列→VQ 码→马尔可夫转移矩阵的路径，把高维连续密度估计问题转化为可计算的离散转移概率计算，思路巧妙
2. **可解释性**：类条件似然可视化直接展示了不同类别的时序模式差异，使伪标签生成不再是黑盒，这在实际部署中具有重要价值
3. **弱监督的优雅整合**：通过贝叶斯先验自然地纳入标签分布信息，比 CoDATS 的 ad-hoc KL 散度最小化更有数学根基
4. **最优传输度量通道偏移**：利用码间语义距离作为 OT 的代价矩阵，比简单的欧氏距离更能捕捉语义相似转移间的关系

## 局限与展望

1. **通道重要性未区分**：当前方法对偏移大的通道降权，但若该通道恰好含有关键分类信息则适得其反。未来可结合通道重要性度量
2. **粗-细分工缺乏数学约束**：虽然实验验证了粗码捕获趋势、细码捕获残差，但缺少显式的正则化来强制这种层级关系
3. **马尔可夫假设的局限**：仅用一阶马尔可夫性可能无法捕获更长程的时序依赖
4. **码本大小固定**：所有数据集使用同样的 $n_c=8, n_f=64$，未探讨自适应码本大小的可能
5. **标签空间假设**：要求源域和目标域标签空间完全相同，未考虑部分标签重叠的场景

## 相关工作与启发

- **VQVAE 在时间序列中的新应用**：此前 VQ 主要用于生成和自监督学习，TransPL 首次将其用于 UDA，启发了离散表示在更多下游任务中的潜力
- **与 HMM 的联系**：类条件似然的计算本质上是 HMM 的前向概率计算的简化版（一阶马尔可夫），可考虑引入更完整的 HMM 机制
- **SSSS-TSA** 同期工作也关注通道级偏移，但采用自注意力机制选择通道，与 TransPL 的 OT 方法形成互补
- **可扩展到其他模态**：转移矩阵的思路可推广到其他序列数据（如 NLP、音频）的域适应问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 VQ 码转移矩阵用于时序 UDA 伪标签是全新视角，粗-细双码本和 OT 通道对齐设计有创意，但整体在已有组件上的组合色彩较重
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、多种基线对比、消融充分、可解释性分析直观，但数据集规模和多样性偏小（主要是传感器数据）
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，方法推导完整，图表直观，符号定义规范，可解释性可视化很有说服力
- 价值: ⭐⭐⭐⭐ 为时序域适应提供了可解释的伪标签新范式，弱监督扩展具有实用价值，但需在更大规模数据集上验证泛化性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICML 2025\] Customizing the Inductive Biases of Softmax Attention using Structured Matrices](customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)
- [\[ICML 2025\] Causal Discovery from Conditionally Stationary Time Series](causal_discovery_from_conditionally_stationary_time_series.md)
- [\[ICML 2025\] WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
