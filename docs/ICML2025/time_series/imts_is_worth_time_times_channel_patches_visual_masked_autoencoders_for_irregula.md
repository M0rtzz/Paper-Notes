---
title: >-
  [论文解读] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction
description: >-
  [ICML 2025][时间序列][不规则多变量时间序列] 提出 VIMTS 框架，将不规则多变量时间序列（IMTS）转化为 time × channel 的类图像 patch 结构，借助在大规模 RGB 图像上预训练的视觉 MAE 的稀疏多通道建模能力，结合 GCN 跨通道补全与粗到细预测策略，在 IMTS 预测任务上实现 SOTA 性能和强 few-shot 能力。
tags:
  - ICML 2025
  - 时间序列
  - 不规则多变量时间序列
  - 视觉掩码自编码器
  - 自监督学习
  - 图卷积网络
  - 粗到细预测
---

# IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction

**会议**: ICML 2025  
**arXiv**: [2505.22815](https://arxiv.org/abs/2505.22815)  
**代码**: [https://github.com/WHU-HZY/VIMTS](https://github.com/WHU-HZY/VIMTS)  
**领域**: 时序分析  
**关键词**: 不规则多变量时间序列, 视觉掩码自编码器, 自监督学习, 图卷积网络, 粗到细预测

## 一句话总结

提出 VIMTS 框架，将不规则多变量时间序列（IMTS）转化为 time × channel 的类图像 patch 结构，借助在大规模 RGB 图像上预训练的视觉 MAE 的稀疏多通道建模能力，结合 GCN 跨通道补全与粗到细预测策略，在 IMTS 预测任务上实现 SOTA 性能和强 few-shot 能力。

## 研究背景与动机

**问题定义**：不规则多变量时间序列（IMTS）预测面临两大核心挑战——多通道信号的时间戳不对齐，以及大量缺失值的存在。这类数据广泛存在于金融、医疗、交通、气象等场景。

**现有方法的局限**：

- **统计插补方法**（如线性插值）：需要对系统动力学的深入理解，且会丢弃缺失点所包含的信息
- **GCN-based 方法**：交替建模时间和通道信息，由于数据稀疏性导致严重的累积误差
- **Neural-ODE-based 方法**：难以从缺失严重的单通道构建准确模型，且计算开销大
- **预训练基础模型**（如 VisionTS）：虽然在 RTS 上表现不错，但只适用于规则采样时间序列

**核心动机**：视觉 MAE 在图像预训练中具有建模语义稀疏多通道信息的强大能力，而 IMTS 数据本身具有类似图像的多通道稀疏特性。VisionTS 已经证明了视觉 MAE 与时间序列之间的模式相似性。因此，本文提出将 IMTS 重构为类图像的 time × channel 结构，充分利用视觉 MAE 的预训练能力。

## 方法详解

### 整体框架

VIMTS 由三个核心模块组成：

1. **Time × Channel Patchify**：将非结构化 IMTS 数据转化为规整 patch，并通过跨通道信息补全缓解缺失值影响
2. **Time-wise Reconstruction**：利用预训练视觉 MAE 建模通道内 patch 之间的时间依赖关系
3. **Patch2Point Prediction**：通过粗到细策略，从 patch 级时间段表示生成精确的时间点预测

### 关键设计

#### 1. Time-wise Dividing and Embedding（时间分段与嵌入）

将 IMTS 数据沿时间轴划分为 P 个等间隔的时间窗口（窗口大小为 s），每个窗口产生所有通道的 patch。采用可学习时间嵌入捕获非周期和周期性时间模式：

$$\phi(t)[d] = \begin{cases} \omega_0 \cdot t + \alpha_0, & \text{if } d=0 \\ \sin(\omega_d \cdot t + \alpha_d), & \text{if } 0 < d < D_{te} \end{cases}$$

其中 $\omega_d$ 和 $\alpha_d$ 为可学习参数，线性项捕获趋势，正弦项捕获周期模式。

#### 2. Transformable Time-aware Convolutional Network (TTCN)

TTCN 通过自适应卷积滤波器处理每个时间窗口内变长序列，生成对齐形状和语义的 patch。具体步骤：

- 在每个通道的时间段内，拼接时间嵌入 $\phi(t_i^n)$ 与观测值 $x_i^n$
- 通过元滤波器（MLP）生成自适应 softmax 归一化的卷积核
- 执行时间卷积得到固定维度的特征 patch $h_p^{n'}$
- 拼接二值 mask 指示 patch 中是否有观测值（$m_p^n=1$ 有观测，$m_p^n=0$ 空 patch）
- 添加通道专属可学习嵌入 $e_n$ 以区分异质通道

这一设计巧妙地将变长不规则输入转化为规整特征 patch，同时保留了缺失信息。

#### 3. Cross-channel Information Interaction（跨通道信息交互）

由于 IMTS 中大量缺失值导致单通道 patch 信息不足，采用 GCN 建模双向通道依赖关系：

- **混合图嵌入**：维护两组可学习的静态嵌入字典 $\mathbf{E}_1^s, \mathbf{E}_2^s$（代表入/出节点），通过门控机制融合动态 patch 特征：
$$\mathbf{E}_{p,k} = \mathbf{E}_k^s + g_{p,k} \odot \mathbf{H}_p \mathbf{W}_k^d$$

- **自适应邻接矩阵**：动态捕获通道间方向性依赖：
$$\mathbf{A}_p = \text{Softmax}(\text{ReLU}(\mathbf{E}_{p,1} \mathbf{E}_{p,2}^\top))$$

- **图卷积 + 残差**：交换通道信息并保留原始表示：
$$\mathbf{H}_p^{gcn} = \text{ReLU}\left(\sum_{m=0}^{M} (\mathbf{A}_p)^m \mathbf{H}_p \mathbf{W}_m^{gcn}\right) + \mathbf{H}_p$$

- 最终拼接原始与 GCN 增强的表示 $\mathbf{H}_p^{in} = [\mathbf{H}_p \| \mathbf{H}_p^{gcn}] \in \mathbb{R}^{N \times 2D}$

#### 4. Time-wise Reconstruction（时间维度重建）

利用预训练视觉 MAE 建模 patch 序列的时间依赖：

- **输入投影**：线性层将 $2D$ 维特征压缩到 MAE 编码器维度 $D_e$
- **时间周期位置嵌入（TPE）**：采用 2D 正弦余弦编码初始化，将 patch 序列视为 $T \times 1$ 的 patch 序列，复用 MAE 预训练的位置理解能力
- **编码**：MAE encoder 编码可见 patch
- **重建**：附加可学习 mask token 和对应 TPE，由 MAE decoder 重建目标时间段的 patch 表示

#### 5. Patch2Point（粗到细预测）

采用两阶段策略从 patch 级到点级生成预测：

- **粗阶段**：MAE 重建目标时间段的 patch
- **细阶段**：给定查询时间戳 $t_q$，生成查询嵌入 $\phi(t_q)$，定位对应 patch，通过 2 层 MLP 生成点级预测：
$$\hat{x}_q = \mathcal{F}(\phi(t_q), \hat{z}_{i_q}^m)$$

这一策略支持在连续时间戳上做灵活精确预测，同时过滤无关的时间-通道上下文。

### 损失函数 / 训练策略

采用**两阶段训练策略**：

**阶段一：自监督学习（SSL）**

- 随机 mask 一定比例 $r$ 的 patch，只编码可见 patch，然后用 decoder 重建被 mask 的 patch
- 损失为被 mask patch 对应时间点的 MSE 重建误差：
$$\mathcal{L}_{ssl} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{\mathcal{H}_n} \sum_{h=1}^{\mathcal{H}_n} \|\mathcal{F}(\phi(t_h^n), \hat{z}_{i_h}^{m,n}) - x_h^n\|_2^2$$
- 此阶段优化所有参数，将视觉 MAE 的能力适配到 IMTS 数据

**阶段二：有监督微调（Fine-tuning）**

- 编码全部历史 patch，附加 mask token 重建未来时间段 patch，预测未来时间点
- 损失为预测值与真实值的 MSE：
$$\mathcal{L}_{ft} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{\mathcal{Q}_n} \sum_{q=1}^{\mathcal{Q}_n} \|\mathcal{F}(\phi(t_q^n), \hat{z}_{i_q}^{m,n}) - x_q^n\|_2^2$$
- **选择性冻结**：对 USHCN/PhysioNet/Human Activity 冻结 GCN 和 MAE（保留 LayerNorm 可训练）；对 MIMIC 仅冻结 MAE（保留 LayerNorm、位置嵌入、patch 投影层可训练）

## 实验关键数据

### 主实验

在 4 个真实数据集上与 19 个基线方法对比（包括 RTS 方法、GNN 方法、IMTS 专用方法和预训练方法）。

| 数据集 | 指标 | VIMTS (100%) | T-PatchGNN (之前SOTA) | 提升 |
|--------|------|-------------|----------------------|------|
| PhysioNet | MSE (×10⁻³) | **4.81±0.07** | 4.98±0.08 | 3.4% |
| PhysioNet | MAE (×10⁻²) | **3.54±0.04** | 3.72±0.03 | 4.8% |
| Human Activity | MSE (×10⁻³) | **2.65±0.01** | 2.66±0.03 | 0.4% |
| Human Activity | MAE (×10⁻²) | **3.08±0.01** | 3.15±0.02 | 2.2% |
| USHCN | MSE (×10⁻³) | **4.86±0.02** | 5.00±0.04 | 2.8% |
| MIMIC | MSE (×10⁻²) | **1.36±0.02** | 1.36±0.02 | 持平 |
| MIMIC | MAE (×10⁻²) | **6.40±0.17** | 6.56±0.11 | 2.4% |

Few-shot 能力：仅用 20% 训练数据的 VIMTS 已接近或超过 T-PatchGNN 使用 100% 数据的性能。

### 消融实验

| 配置 | PhysioNet MSE (×10⁻³) | Human Activity MSE (×10⁻³) | MIMIC MSE (×10⁻²) | 说明 |
|------|----------------------|---------------------------|-------------------|------|
| Complete | **4.81±0.07** | **2.65±0.01** | **1.36±0.02** | 完整模型 |
| w/o Pre | 5.13±0.04 | 2.73±0.02 | 1.39±0.02 | 去掉视觉预训练权重 |
| w/o SSL | 5.46±0.30 | 2.76±0.08 | 1.41±0.03 | 去掉自监督学习阶段 |
| w/o Pre & SSL | 5.70±0.42 | 2.84±0.06 | 1.45±0.05 | 同时去掉预训练和 SSL |
| w/o GCN | 4.94±0.03 | 2.66±0.01 | 2.25±0.02 | 去掉跨通道 GCN（MIMIC 影响最大） |
| rp Transformer | 5.57±0.34 | 2.84±0.07 | 1.40±0.04 | 用 Transformer 替代 MAE |

### 关键发现

1. **预训练和 SSL 缺一不可**：去掉预训练导致 PhysioNet MSE 从 4.81 升到 5.13（+6.7%），去掉 SSL 升到 5.46（+13.5%），两者都去掉升到 5.70（+18.5%）
2. **GCN 跨通道补全在高缺失率数据上至关重要**：MIMIC 缺失率 96.7%，去掉 GCN 后 MSE 从 1.36 飙升到 2.25（+65.4%）
3. **MAE 优于普通 Transformer**：替换为 Transformer 后所有数据集性能均下降，说明视觉预训练带来的稀疏数据建模能力是关键
4. **Patch2Point 策略有效**：两阶段都使用 Patch2Point 时效果最优，验证了粗到细预测策略的价值
5. **强 few-shot 能力**：在 Human Activity 上仅 10% 数据的 VIMTS (MSE=2.87) 大幅优于 T-PatchGNN (MSE=3.21)

## 亮点与洞察

1. **视觉-时序的桥梁**：深刻洞察到 IMTS 的稀疏多通道特性与被 mask 的图像 patch 结构高度类似，巧妙地将视觉预训练能力迁移到时序领域
2. **time × channel 二维重构**：将一维时间序列重构为二维类图像结构的思路非常优雅，使得视觉 MAE 的 2D positional embedding 可以自然复用
3. **两阶段训练设计合理**：SSL 阶段让模型学会在 IMTS 数据上做 patch 重建（领域适配），FT 阶段聚焦预测任务（任务适配），分工明确
4. **选择性冻结策略**：根据数据集特性差异化冻结不同模块，兼顾知识保留和任务适配
5. **可处理连续时间戳查询**：Patch2Point 机制支持任意时间点预测，不局限于固定步长

## 局限与展望

1. **仅在 4 个数据集上验证**：虽然涵盖医疗、运动、气候等领域，但缺少金融等高频场景的验证
2. **计算效率未充分讨论**：GCN + MAE 的组合计算量较大，与 Neural-ODE 方法的效率对比不够充分
3. **通道数扩展性**：MIMIC 有 96 通道且缺失率 96.7%，VIMTS 仅与 T-PatchGNN 持平，高通道高缺失场景仍有提升空间
4. **固定 patch 大小**：所有通道使用统一时间窗口大小 s，但不同通道的采样频率可能差异很大，自适应 patch 策略值得探索
5. **MAE backbone 固定为 MAE-base**：未探索更大规模视觉预训练模型（如 MAE-large/huge）的效果

## 相关工作与启发

- **VisionTS** (Chen et al., 2025)：证明视觉 MAE 可适配 RTS 预测，是本文的直接灵感来源
- **T-PatchGNN** (Zhang et al., 2024a)：之前的 IMTS SOTA，结合 GNN 和 patch 设计
- **MAE** (He et al., 2022)：视觉自监督学习的里程碑工作，提供了强大的稀疏信息重建能力
- **启发**：可以进一步探索其他视觉预训练模型（如 DINOv2、SAM）在时序任务上的迁移潜力；跨模态预训练 → 领域适配 → 任务微调的范式值得推广到更多非图像领域

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|:-----------:|------|
| 创新性 | 8 | 视觉 MAE → IMTS 的适配路线新颖，time×channel patchify 设计巧妙 |
| 技术深度 | 8 | TTCN + GCN + MAE + Patch2Point 多模块协同设计完整 |
| 实验充分度 | 8 | 19 个基线对比、多种消融、few-shot 实验，但数据集偏少 |
| 写作质量 | 7 | 结构清晰，但公式较密集，方法描述略冗长 |
| 实用价值 | 7 | 开源代码，但部署复杂度较高（需视觉 MAE 预训练权重 + GCN） |
| **总分** | **7.8** | 将视觉预训练迁移到 IMTS 的首创性工作，方法设计完整有效 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Rotary Masked Autoencoders are Versatile Learners](../../NeurIPS2025/time_series/rotary_masked_autoencoders_are_versatile_learners.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)

<!-- RELATED:END -->
