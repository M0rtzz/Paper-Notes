---
title: >-
  [论文解读] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products
description: >-
  [NeurIPS 2025][视频理解][Linear RNN] 提出 DeltaProduct，通过将 DeltaNet 的单步梯度下降扩展至每个 token 的多步梯度下降，使状态转移矩阵成为 $n_h$ 个广义 Householder 变换的乘积，实现了表达力与效率之间的可调平衡，显著提升了状态跟踪能力和长度外推性能。
tags:
  - NeurIPS 2025
  - 视频理解
  - Linear RNN
  - Householder Product
  - State-Tracking
  - DeltaNet
  - Length Extrapolation
---

# DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products

**会议**: NeurIPS 2025  
**arXiv**: [2502.10297](https://arxiv.org/abs/2502.10297)  
**代码**: [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) (有)  
**领域**: 序列建模 / 线性RNN  
**关键词**: Linear RNN, Householder Product, State-Tracking, DeltaNet, Length Extrapolation

## 一句话总结

提出 DeltaProduct，通过将 DeltaNet 的单步梯度下降扩展至每个 token 的多步梯度下降，使状态转移矩阵成为 $n_h$ 个广义 Householder 变换的乘积，实现了表达力与效率之间的可调平衡，显著提升了状态跟踪能力和长度外推性能。

## 研究背景与动机

**领域现状**: 线性 RNN（如 Mamba、GLA、mLSTM）已成为 Transformer 的竞争替代方案，具有高效训练和线性时间推理的优势。其核心由状态转移矩阵决定，当前主流模型使用对角矩阵。

**现有痛点**: 对角状态转移矩阵虽然运行高效，但**表达力严重受限**——例如无法在有限精度下完成模 3 加法等简单任务。DeltaNet 采用对角加秩 1 结构部分缓解了此问题，但仍需多层才能处理复杂状态跟踪任务（如 $S_5$ 对称群问题）。

**核心矛盾**: 表达力与效率之间存在根本性的权衡：对角矩阵高效但表达力弱，全矩阵表达力强但训练代价极高且不稳定。

**本文目标**: 如何在保持训练效率和递推稳定性的前提下，系统性地提升线性 RNN 的状态转移矩阵的表达力？

**切入角度**: 从 DeltaNet 的在线梯度下降视角出发，将单步梯度下降扩展为 $n_h$ 步，自然地产生了 $n_h$ 个 Householder 变换乘积的结构。

**核心 idea**: 通过增加每个 token 的梯度下降步数 $n_h$，将状态转移矩阵从秩 1 更新（DeltaNet）扩展为秩 $n_h$ 更新，在对角矩阵与稠密矩阵之间实现连续插值。

## 方法详解

### 整体框架

DeltaProduct 建立在 DeltaNet 的线性递推基础上：

$$H_i = A(x_i) H_{i-1} + B(x_i)$$

DeltaNet 的每步递推可视为在关联记忆损失 $\mathcal{L}_i(H) = \frac{1}{2}\|H^\top k_i - v_i\|_2^2$ 上做一步在线梯度下降。DeltaProduct 则对每个 token 做 $n_h$ 步梯度下降，使用不同的 key/value 对。

### 关键设计

#### 1. 多步梯度下降生成 Householder 乘积

对每个输入 $x_i$，生成 $n_h$ 组 key $k_{i,j}$、value $v_{i,j}$ 和步长 $\beta_{i,j}$（$j=1\ldots n_h$），依次做梯度更新：

$$H_{i,j} = (I - \beta_{i,j} k_{i,j} k_{i,j}^\top) H_{i,j-1} + \beta_{i,j} k_{i,j} v_{i,j}^\top$$

展开后，状态转移矩阵为 $n_h$ 个广义 Householder 变换的乘积：

$$A(x_i) = \prod_{j=1}^{n_h} (I - \beta_{i,j} k_{i,j} k_{i,j}^\top)$$

#### 2. 谱范数保证稳定性

每个 Householder 因子的谱范数 $\leq 1$（当 $\beta \in [0,2]$ 时），乘积的谱范数同样 $\leq 1$，确保递推稳定。这是相比 RWKV-7 的关键优势——RWKV-7 的状态转移矩阵谱范数可能 $> 1$，存在不稳定风险。

#### 3. Gated DeltaProduct

类似 Gated DeltaNet，引入标量门 $g_i \in [0,1]$：

$$A(x_i) = g_i \prod_{j=1}^{n_h} (I - \beta_{i,j} k_{i,j} k_{i,j}^\top)$$

#### 4. 扩展特征值范围至 $[-1, 1]$

将 $\beta$ 的范围从 $[0,1]$ 扩展到 $[0,2]$（通过 $2 \times \text{sigmoid}$），使特征值可取负值，这对状态跟踪至关重要。实验显示限制在 $[0,1]$ 时模型完全无法学习。

#### 5. Cartan-Dieudonné 定理的几何直觉

两个反射组合可产生旋转（如图 3 所示）。当 $n_h$ 足够大时，Householder 乘积可表示任意正交矩阵。$n_h=2$ 时已能表示 $\text{SO}(3)$ 中的旋转，足以解决 $S_4$（立方体旋转群）和 $A_5$（正十二面体旋转群）问题。

### 实现细节

将 $n_h$ 组 key/value/beta 排列为序列长度的 $n_h$ 倍输入给现有 DeltaNet 的 Triton 并行实现，门控值在非首步位置设为 1。递推输出仅保留每 $n_h$ 步的结果。

### 损失函数/训练策略

- 使用 FineWeb 数据集训练语言模型
- 训练上下文长度 4096 tokens
- 通过缩放 head 数或 head 维度匹配参数量
- 1.3B 规模在 H100 上测得吞吐量

## 实验关键数据

### 主实验：状态跟踪

| 任务 | 单层所需最小 $n_h$ | 3层可解 | 理论下界 |
|------|---------------------|---------|----------|
| $S_3$ | 2 | ✓ | $n_h = 2$ |
| $S_4$ | 2（利用 SO(3) 同构） | ✓ | $n_h = 3$ |
| $A_5$ | 2（利用 SO(3) 同构） | ✓ | $n_h = 4$ |
| $S_5$ | 4 | ✓ | $n_h = 4$ |

相比之下，DeltaNet（$n_h=1$）即使用 10 层也无法解决 $S_5$。

### 主实验：语言建模

- DeltaProduct$_2$[-1,1]（8 heads, 392M 参数）在长度外推上显著优于 DeltaNet（12 heads, 同参数量）
- $n_h=3$ 时，cross-entropy loss 在超出训练长度后几乎不退化
- DeltaProduct$_3$[-1,1] 在无门控情况下达到了与 Gated DeltaNet[-1,1] 相当的性能

### 消融实验

| 配置 | CodeParrot 外推 | TriviaQA 外推 | OpenThoughts 外推 |
|------|----------------|--------------|-------------------|
| $n_h=1$ (DeltaNet) | 快速退化 | 快速退化 | 快速退化 |
| $n_h=2$ | 大幅改善 | 大幅改善 | 大幅改善 |
| $n_h=3$ | 几乎不退化 | 几乎不退化 | 几乎不退化 |

### 关键发现

1. **有效秩分析**：DeltaNet 的隐状态有效秩在超出训练长度后持续增长（导致分布偏移），而 DeltaProduct$_3$ 的部分 head 学会在 BOS token 处重置有效秩
2. **PCA 验证**：在 $S_4$ 任务上，$n_h=2$ 模型的 key 向量确实集中在 3 维子空间（方差解释率 $> 95\%$），且 $\beta$ 值聚集在 2 附近（反射），验证了利用 SO(3) 同构的理论
3. **缩放分析**：增加 $n_h$ 比增加 head 维度获得更好的训练 perplexity 缩放

## 亮点与洞察

1. **在线学习视角的优雅扩展**：从 DeltaNet 的"单步梯度下降"到 DeltaProduct 的"多步梯度下降"，既直观又数学上自然
2. **理论与实验的完美对应**：模型在 $S_4$ 上自动发现了利用 SO(3) 同构的策略，PCA 和 $\beta$ 值分析提供了令人信服的证据
3. **稳定性保证**：Householder 乘积的谱范数 $\leq 1$，这是该参数化相比 RWKV-7 的结构性优势
4. **遗忘机制加速**：DeltaProduct 可以以 $n_h$ 倍的速度重置隐状态，这解释了长度外推的改善

## 局限与展望

1. **训练代价线性增长**：递推计算量随 $n_h$ 线性增长（主要局限）
2. **参数开销**：额外的 key/value 投影增加参数量，需通过调整 head 数/维度来匹配
3. **未来方向**：自适应确定每个 token 的 $n_h$（类似 Graves 的自适应计算时间）；使用 LoRA MLP 减少参数开销；与 Fixed-Point RNN 结合

## 相关工作与启发

- **DeltaNet/Gated DeltaNet**：直接基础，$n_h=1$ 的特例
- **RWKV-7**：类似的非对角结构但不保证稳定性；DeltaProduct 在 3 层以内的表达力甚至优于 RWKV-7
- **Grazzi et al. (ICLR 2025)**：负特征值的重要性，本文扩展了其理论
- **Fixed-Point RNN**：正交方向的工作，通过不动点迭代增加对角 RNN 的表达力，可与 DeltaProduct 结合

## 评分

⭐⭐⭐⭐⭐

理论完备（有限精度下的 state-tracking 能力刻画）、实验充分（从 toy task 到语言建模）、工程可用（集成到 flash-linear-attention 库），是线性 RNN 表达力研究的重要进展。
