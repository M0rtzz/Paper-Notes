---
title: >-
  [论文解读] Customizing the Inductive Biases of Softmax Attention using Structured Matrices
description: >-
  [ICML 2025][时间序列][结构化矩阵] 提出用高效结构化矩阵（BTT 和 MLR）替换 softmax attention 中的低秩打分函数，既解决了标准 attention 的低秩瓶颈问题，又通过 MLR 引入了距离依赖的计算偏置，在上下文回归、语言建模和长程时间序列预测上均取得改进。
tags:
  - ICML 2025
  - 时间序列
  - 结构化矩阵
  - 注意力机制
  - 归纳偏置
  - 多层低秩矩阵
  - 时间序列预测
---

# Customizing the Inductive Biases of Softmax Attention using Structured Matrices

**会议**: ICML 2025  
**arXiv**: [2509.07963](https://arxiv.org/abs/2509.07963)  
**代码**: [YilunKuang/structured-attention](https://github.com/YilunKuang/structured-attention)  
**领域**: 时序分析  
**关键词**: 结构化矩阵, 注意力机制, 归纳偏置, 多层低秩矩阵, 时间序列预测

## 一句话总结

提出用高效结构化矩阵（BTT 和 MLR）替换 softmax attention 中的低秩打分函数，既解决了标准 attention 的低秩瓶颈问题，又通过 MLR 引入了距离依赖的计算偏置，在上下文回归、语言建模和长程时间序列预测上均取得改进。

## 研究背景与动机

标准 Transformer 的核心是注意力机制，其打分函数将输入投影到低维的 query 和 key 空间后做点积。这种设计存在两个根本性缺陷：

**低秩瓶颈（Low-Rank Bottleneck）**：head 维度 $r$ 远小于 embedding 维度 $D$（$D = H \cdot r$），导致 $\mathbf{W}_Q \mathbf{W}_K^\top$ 是秩为 $r$ 的低秩矩阵。对于本征高维的输入（如高维回归任务），信息在投影过程中被大量丢失。Amsel et al. (2024) 证明了除非 $r \gtrsim d_{\text{input}}$，attention 甚至无法近似解决高维球面上的最近邻问题。

**缺乏距离依赖的计算偏置**：标准 attention 对序列中所有 token 对使用相同的打分函数，不区分局部和全局交互。然而自然语言等现实数据具有显著的局部性——同一段落中的词联系紧密，而相隔很远的词很少直接相关。滑动窗口注意力（SWA）等稀疏方案虽能节省计算，但通常是脆弱的且会降低精度。

本文的核心洞察：attention 的打分函数本质上是一个双线性变换 $s(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{M} \mathbf{x}'$，矩阵 $\mathbf{M}$ 的结构决定了 attention 的归纳偏置。通过选择合适的**结构化矩阵**替换低秩的 $\mathbf{W}_Q \mathbf{W}_K^\top$，可以精确定制归纳偏置以适配不同任务。

## 方法详解

### 整体框架

本文从打分函数的双线性形式出发，提出了一个通过结构化矩阵定制 attention 归纳偏置的统一框架。核心思想是：将 attention 打分矩阵 $\mathbf{S}_{j,j'} = s(\mathbf{x}_j, \mathbf{x}_{j'})$ 中的底层矩阵从低秩替换为其他结构化矩阵族，实现两个目标：

- **目标一**：解决低秩瓶颈——使用高秩但参数/计算高效的结构化矩阵（BTT、MLR）作为双线性形式的核心矩阵
- **目标二**：引入距离依赖的计算分配——使用 MLR 结构对打分矩阵 $\mathbf{S}$ 本身施加层级结构

两种应用方式分别针对上述两个问题，但都基于同一个数学工具——结构化矩阵族。

### 关键设计

#### 1. 结构化矩阵族

本文考察了四种 $D \times D$ 结构化矩阵：

| 结构 | 定义形式 | 参数量 | 秩 |
|------|---------|--------|-----|
| Dense（密集） | $\mathbf{W}$ | $D^2$ | $D$ |
| Low Rank（低秩） | $\mathbf{L}\mathbf{R}^\top$ | $2Dr$ | $r$ |
| MLR（多层低秩） | $\sum_{l=1}^L \bigoplus_{k=1}^{p_l} \mathbf{L}_{l,k}\mathbf{R}_{l,k}^\top$ | $2D\sum_l r_l$ | $\sum_l r_l p_l$ |
| BTT（块张量训练） | $\mathbf{P}_L(\bigoplus \mathbf{L}_{k'}) \mathbf{P}_R(\bigoplus \mathbf{R}_k^\top)$ | $2D^{3/2}s$ | $D$ |

关键要点：BTT 和 MLR 都能在参数和计算量远小于 $D^2$ 的条件下实现高秩甚至满秩。

#### 2. 结构化双线性打分函数（解决低秩瓶颈）

将标准 attention 的打分函数 $s(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \mathbf{W}_Q \mathbf{W}_K^\top \mathbf{x}_{j'}$ 替换为：

**MLR 打分函数**：
$$s_{\text{MLR}}(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \left(\sum_{l=1}^L \bigoplus_{k=1}^{2^{l-1}} \mathbf{L}_{l,k}\mathbf{R}_{l,k}^\top \right) \mathbf{x}_{j'}$$

**BTT 打分函数**：
$$s_{\text{BTT}}(\mathbf{x}_j, \mathbf{x}_{j'}) = \mathbf{x}_j^\top \left(\mathbf{P}_L \bigoplus_{k'=1}^{b} \mathbf{L}_{k'} \mathbf{P}_R \bigoplus_{k=1}^{c} \mathbf{R}_k^\top \right) \mathbf{x}_{j'}$$

BTT 在 $a=b=c=d=\sqrt{D}$ 时仅需 $O(D^{3/2})$ 参数和 FLOPs 即可达到满秩 $D$。MLR 通过设置 $p_l = 2^{l-1}$ 且 $\sum_l r_l = r$，在匹配标准 attention 效率的同时实现高秩。

#### 3. MLR Attention：距离依赖的计算偏置

这是本文最具创新性的设计。不再对打分矩阵的底层参数做结构化，而是对**打分矩阵 $\mathbf{S}$ 本身**施加 MLR 结构：

$$\mathbf{S} = \sum_{l=1}^L \bigoplus_{k=1}^{p_l} \mathbf{Q}_{l,k} \mathbf{K}_{l,k}^\top$$

其中 $\mathbf{Q}_{l,k}$ 和 $\mathbf{K}_{l,k}$ 分别是 query/key 矩阵在第 $l$ 层第 $k$ 块的切片。直观含义：

- **Level 1** ($p_1 = 1$)：一个全局低秩分量，所有 token 对共享，秩为 $r_1$
- **Level 2** ($p_2 = 2$)：序列被分成 2 块，同块 token 对额外获得秩 $r_2$ 的打分分量
- **Level $l$** ($p_l = 2^{l-1}$)：序列被分成 $2^{l-1}$ 块，同块 token 对再获得秩 $r_l$ 的分量
- 距离越近的 token 对，累积的打分函数秩越高，信息交互越丰富

这种层级结构带来两大实际优势：

**计算节省**：标准 attention 形成 $\mathbf{S}$ 需要 $T^2 r$ FLOPs，而 MLR attention 仅需 $T^2 \sum_{l=1}^L \frac{r_l}{2^{l-1}}$。例如 8 层 MLR 且 $r_l = r/8$ 时可节省约 4 倍计算。

**KV Cache 压缩**：自回归生成时，level $l$ 只需保留最后一个块的 key $\mathbf{K}_{l,p_l}$，总缓存大小为 $T \sum_{l=1}^L \frac{r_l}{2^{l-1}}$，同样可实现约 4 倍压缩。

#### 4. MLBTC 统一框架

本文提出 Multi-Level Block Tensor Contraction（MLBTC）作为统一框架：

$$\text{MLBTC}(\mathbf{L}, \mathbf{R}) = \sum_{l=1}^L \alpha_l \mathbf{P}_L \bigoplus_{k'=1}^{p'_l} \mathbf{L}_{l,k'} \mathbf{P}_R \bigoplus_{k=1}^{p_l} \mathbf{R}_{l,k}^\top$$

通过设置不同参数，MLBTC 可退化为：
- MLR（$\mathbf{P}_L = \mathbf{P}_R = \mathbf{I}$）
- BTT（只保留一个 level）
- Monarch、Butterfly、Kronecker、Low Rank 等

这个理论统一为未来探索更多结构化注意力变体提供了基础。

### 损失函数 / 训练策略

- **上下文回归任务**：使用均方误差损失，训练 6 层 Transformer，8 个 attention head
- **语言建模**：使用标准交叉熵损失，在 FineWeb 数据集上训练
- **时间序列预测**：使用 MSE 损失，在标准长程预测 benchmark 上评估
- **稳定性技巧**：文中提到了用于稳定结构化矩阵训练的特征学习技术（Section 3.5）
- 所有结构化注意力变体与 Grouped-Query Attention (GQA) 兼容

## 实验关键数据

### 主实验

**实验一：上下文回归（In-Context Regression）**

| 方法 | $d=128$ 表现 | $d=64$ 表现 | 参数效率 |
|------|---------|---------|---------|
| 标准 Attention (8-head, $r < d$) | 回归误差高 | 回归误差高 | 需 $r \geq d$ 才能学会 |
| Bilinear BTT | 小宽度即可学会 | 小宽度即可学会 | 满秩，参数 $O(D^{3/2})$ |
| Bilinear MLR | 小宽度即可学会 | 小宽度即可学会 | 高秩，参数与标准等同 |
| 标准 Attention (1-head, 满秩) | 可学会但参数多 | 可学会但参数多 | 需大模型宽度 |

关键发现：BTT 和 MLR 在**任何固定计算预算**下都优于标准 attention，因为它们打破了低秩瓶颈。

**实验二：语言建模 Scaling Laws**

| 方法 | Scaling Law 表现 | 与标准 Attention 对比 | 与 SWA 对比 |
|------|---------|---------|---------|
| 标准 Attention | 基线 | — | 优于 SWA |
| 滑动窗口注意力 (SWA) | 差于标准 | 劣于标准 | — |
| MLR Attention | **优于标准** | 更优 scaling | 显著优于 SWA |

MLR attention 在语言建模上展现出比标准 attention 和 SWA 都更优的 scaling law 趋势。

**实验三：长程时间序列预测**

MLR attention 的层级结构天然适合时间序列中的多尺度时间依赖模式，论文报告了 promising results。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $L=1$（退化为标准 attention） | 基线 | 无距离偏置 |
| $L=2$（2 层 MLR） | 改进 | 引入局部/全局区分 |
| $L \leq 8$（多层 MLR） | 最佳 | 层级化距离依赖计算 |
| 秩分配 $r_1 \vert r_2 \vert \cdots \vert r_L$ | 影响精度/效率权衡 | 各 level 秩的分配是关键超参数 |
| Head 维度 $r$ vs 输入维度 $d$ | $r < d$ 时标准 attention 失败 | 验证低秩瓶颈 |
| BTT $s=1$ vs $s=2$ | 更大 $s$ 更强但更贵 | 参数量-表达力权衡 |

### 关键发现

1. **标准 attention 的低秩瓶颈是真实的**：在 $d_{\text{input}} = 128$ 的上下文回归任务上，8-head attention 在 $r < d$ 时完全无法学会，而 BTT/MLR 可以在更小的模型中成功学习
2. **MLR attention 的 scaling law 优于标准 attention**：在语言建模上首次展示了结构化注意力在 scaling 上的优势
3. **距离依赖计算偏置是"非脆弱"的**：不同于 SWA 的硬截断，MLR 通过层级化的秩分配实现了灵活的局部-全局平衡，不会牺牲准确性
4. **KV cache 压缩是免费的**：MLR attention 在推理时天然支持约 4 倍 KV cache 压缩

## 亮点与洞察

1. **视角新颖**：从双线性变换而非 query/key 的角度重新审视 attention，将结构化矩阵理论与注意力机制的设计连接起来
2. **MLBTC 统一框架**：首次将 MLR、BTT、Monarch、Butterfly、Kronecker 等结构化矩阵统一到一个框架下，为后续研究提供了清晰的理论地图
3. **一石二鸟**：同一个数学工具（结构化矩阵）同时解决了两个看似不相关的问题（低秩瓶颈 + 缺乏局部性偏置）
4. **与 GQA 兼容**：MLR/BTT attention 可以直接嵌入使用 Grouped-Query Attention 的现代 LLM 架构
5. **实用价值**：KV cache 压缩和 FLOPs 节省在大模型推理中具有直接的工程价值

## 局限与展望

1. **时间序列实验不够充分**：论文只提到 MLR attention 在长程预测上有 "promising results"，缺乏与 PatchTST、iTransformer 等 SOTA 时序模型的详细对比
2. **缺乏大规模 LLM 验证**：scaling law 实验的模型规模有限，尚需在 7B+ 参数的模型上验证
3. **MLBTC 框架尚未实验验证**：论文提出了统一框架但留待 future work，目前只验证了 MLR 和 BTT 两个特例
4. **层级结构是固定的**：$p_l = 2^{l-1}$ 的等比划分不一定最优，论文提到了动态划分（按段落/文档）的可能性但未实现
5. **FlashAttention 兼容性**：结构化打分矩阵能否与 FlashAttention 等高效实现融合尚不明确
6. **序列长度限制**：当前要求序列长度 $T > \max_l p_l$，对超长序列场景需要更灵活的设计

## 相关工作与启发

- **Monarch/Butterfly Matrices (Dao et al., 2022/2020)**：BTT 是对 Monarch 矩阵的推广，本文将其引入 attention 打分
- **Linear Attention (Katharopoulos et al., 2020)**：另一条高效 attention 路线，但往往以牺牲精度为代价
- **Longformer (Beltagy et al., 2020)**：结合滑动窗口和全局 attention 的早期尝试，MLR attention 是其更优雅的替代
- **SSM/Mamba (Gu & Dao, 2024)**：SSM 自带距离衰减偏置，MLR attention 在 Transformer 框架内达到类似效果
- **In-Context Learning (Garg et al., 2022)**：本文实验的重要基准任务来源
- **对时序领域的启发**：MLR attention 的多尺度层级结构与时间序列中的多频率/多周期模式天然契合，有潜力替换时序 Transformer 中的标准 attention

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | 结构化矩阵 × attention 的交叉视角很新颖 |
| 理论深度 | 4.5 | MLBTC 统一框架有扎实的理论贡献 |
| 实验充分度 | 3.5 | 上下文回归和语言建模实验扎实，时序实验偏少 |
| 实用价值 | 4.0 | KV cache 压缩和 FLOPs 节省有直接工程意义 |
| 写作质量 | 4.5 | 结构清晰，数学推导严谨 |
| **综合** | **4.2** | 理论漂亮、视角新颖的工作，实验可进一步加强 |

<!-- RELATED:START -->

## 相关论文

- [TransPL: VQ-Code Transition Matrices for Pseudo-Labeling of Time Series Unsupervised Domain Adaptation](transpl_vq-code_transition_matrices_for_pseudo-labeling_of_time_series_unsupervi.md)
- [FLAVC: Learned Video Compression with Feature Level Attention](../../CVPR2025/time_series/flavc_learned_video_compression_with_feature_level_attention.md)
- [Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](../../NeurIPS2025/time_series/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)

<!-- RELATED:END -->
