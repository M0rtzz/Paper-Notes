---
title: >-
  [论文解读] Controllable Financial Market Generation with Diffusion Guided Meta Agent
description: >-
  [AAAI 2026][金融市场生成] 提出Diffusion Guided Meta Agent（DigMA）模型，将可控金融市场生成形式化为条件生成任务，用条件扩散模型捕捉市场状态动态（中间价收益率与订单到达率的时变分布参数），结合具有金融经济学先验的Meta Agent生成订单流，在可控性和生成保真度上均超越现有方法。
tags:
  - AAAI 2026
  - 金融市场生成
  - 扩散模型
  - 限价订单簿
  - 可控生成
  - 多智能体模拟
---

# Controllable Financial Market Generation with Diffusion Guided Meta Agent

**会议**: AAAI 2026  
**arXiv**: [2408.12991](https://arxiv.org/abs/2408.12991)  
**代码**: [microsoft/TimeCraft](https://github.com/microsoft/TimeCraft)  
**领域**: 其他（金融AI/生成模型）  
**关键词**: 金融市场生成, 扩散模型, 限价订单簿, 可控生成, 多智能体模拟

## 一句话总结

提出Diffusion Guided Meta Agent（DigMA）模型，将可控金融市场生成形式化为条件生成任务，用条件扩散模型捕捉市场状态动态（中间价收益率与订单到达率的时变分布参数），结合具有金融经济学先验的Meta Agent生成订单流，在可控性和生成保真度上均超越现有方法。

## 研究背景与动机

### 问题背景
金融市场是数据密集且复杂的系统，订单（order）是市场中最基本的事件单元（类似于语言中的word、图像中的pixel）。通过订单流，研究者可以探究市场微观结构和交互机制。**订单流建模**因此构成金融领域最基础的生成任务。

### 现有方法的局限

**基于规则的多智能体方法**（如RFD、RMSC）：依赖过于简化的市场假设和手工规则，未在真实数据上训练，模拟保真度有限

**学习型智能体方法**（如LOBGAN）：基于历史订单流预测下一个订单，但一分钟内可能有数百个订单，一个交易日跨越数百分钟，难以捕捉长期依赖；倾向于关注局部分布而忽视全局动态

**可控性完全缺失**：所有现有方法都无法指定目标场景（如极端事件、高/低波动率）来生成对应的订单流，无法进行情景实验和反事实分析

### 核心挑战
在宏观控制目标（如日收益率、波动率）与微观订单之间建立有效连接非常困难，因为：
- 真实订单流序列极长且不规则，扩散模型难以直接应用于原始订单级数据
- 单个订单的信噪比极低，将宏观控制映射到每个微观订单不切实际

## 方法详解

### 整体框架

DigMA采用**两阶段设计**，避免将扩散模型直接应用于原始订单流：
- **Meta Controller**（元控制器）：条件扩散模型，学习给定场景$c$下市场状态$\mathbf{x}$的分布$q(\mathbf{x}|c)$
- **Order Generator**（订单生成器）：包含模拟交易所 + Meta Agent，后者结合金融经济学先验，在元控制器指导下通过随机过程生成订单

### 关键设计

#### 1. **问题形式化：可控金融市场生成**

定义为条件生成任务，两个目标：
- **可控性目标**：最小化目标指标$a$与生成订单流计算指标$\tilde{a}$之间的差异
  $$\min_{\mathcal{M}} \mathbb{E}_{a,\tilde{a}}[\|\tilde{a} - a\|^2]$$
- **保真度目标**：最小化真实与生成订单流"stylized facts"分布之间的散度
  $$\min_{\mathcal{M}} \mathcal{D}(p(\mathcal{F}'(\tilde{\boldsymbol{O}})) \| p(\mathcal{F}'(\boldsymbol{O}})))$$

控制指标包括：日收益率、日振幅、日内波动率。

#### 2. **Meta Controller：条件扩散模型学习市场状态动态**

**市场状态定义**：从真实订单流中提取**逐分钟中间价收益率$\mathbf{r}$**和**订单到达率$\boldsymbol{\lambda}$**，定义市场状态$\mathbf{x} = \{\mathbf{r}, \boldsymbol{\lambda}\}$。

**扩散训练**：采用DDPM的$\epsilon$参数化，训练损失：
$$L_M = \mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),n}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n)\|^2]$$

**条件控制**：设计两种控制编码器：
- **离散控制编码器**：将目标条件映射到预定义bin的类别标签，通过可学习embedding矩阵嵌入
- **连续控制编码器**：用全连接网络将实值条件映射到潜在向量

使用**Classifier-Free Guidance**实现控制：训练时随机丢弃条件信息（丢弃概率0.5），采样时计算引导分数：
$$\tilde{\boldsymbol{\epsilon}}_{\theta,\phi}(\mathbf{x}_n, n, \mathbf{c}) = (1-s)\boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n) + s\boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n, \boldsymbol{\phi}(\mathbf{c}))$$
其中$s$控制引导强度。

**模型骨干**：以1D卷积层为主的U-Net，参数在扩散时间步间共享；采用DDIM采样提高效率。

#### 3. **Order Generator：融合金融经济学先验的Meta Agent**

Meta Agent代表市场中所有交易者的聚合行为，核心流程：

**唤醒过程**：在每个交易分钟$t$内，Agent在指数分布间隔$\delta_i$后"唤醒"，参数$\lambda_t$由元控制器提供。

**Actor Agent初始化**：每次唤醒实例化一个异质Agent，包含三个组件（基本面/图表派/噪声），权重从期望值比为10:1.5:1的指数分布独立采样。

**收益估计**：加权平均三个组件的收益信号：
$$\hat{r} = \frac{g_f r_t + g_c \bar{r} + g_n r_\sigma}{g_f + g_c + g_n}$$
其中$r_t$来自元控制器，$\bar{r}$来自模拟交易所历史，$r_\sigma$是高斯噪声。

**CARA效用优化**：基于估计收益计算需求函数$u(p) = \frac{\ln(\hat{p}_t/p)}{aVp}$（$a$为风险厌恶系数，$V$为历史价格波动率），确定最低可接受价格$p_l$，均匀采样订单价格$p_i \sim \mathcal{U}(p_l, \hat{p}_t)$。

设计动机：这种Agent结构融合了经典金融经济学的异质交易者模型，使生成的订单自然具备市场微观结构特性。

### 训练策略

- 每个数据集训练10个epoch，200个扩散步骤
- AdamW优化器，学习率$1\times10^{-5}$，mini-batch=256
- 条件随机丢弃概率0.5
- 数据预处理：z-score标准化，从tick-by-tick数据提取逐分钟中间价收益率和订单到达率

## 实验关键数据

### 数据集
中国A股市场两个tick-by-tick订单簿数据集：A-Main（316K日×股票对）和ChiNext（122K日×股票对），2020年深交所全年数据。

### 主实验一：可控性评估

| 目标 | 方法 | 控制 | A-Main最小MSE | ChiNext最小MSE |
|------|------|------|--------------|---------------|
| Return | No Control | - | 0.529 | 0.684 |
| Return | Discrete | 离散 | 0.228 | 0.243 |
| Return | **Continuous** | 连续 | **0.161** | **0.342** |
| Amplitude | No Control | - | 0.268 | 0.427 |
| Amplitude | **Continuous** | 连续 | **0.054** | **0.110** |
| Volatility | No Control | - | 0.021 | 0.029 |
| Volatility | **Continuous** | 连续 | **0.011** | **0.028** |

### 主实验二：保真度评估（KL散度↓）

| 模型 | MinR | RetAC | VolC | OIR |
|------|------|-------|------|-----|
| RFD | 1.198 | 5.010 | 0.839 | 0.015 |
| RMSC | 2.640 | 10.170 | 1.237 | 0.563 |
| LOBGAN | 0.151 | **1.903** | 1.101 | 0.309 |
| **DigMA** | **0.084** | 2.781 | **0.273** | **0.009** |

### 消融实验：高频交易RL评估

| 训练环境 | Ret(%)↑ | Vol↓ | SR↑ | MDD(%)↓ |
|---------|---------|------|-----|---------|
| Replay | 0.009 | 0.413 | 0.014 | 1.133 |
| RFD | 0.000 | 0.159 | 0.011 | 0.803 |
| DigMA-c (无控制) | 0.015 | **0.147** | 0.006 | **0.715** |
| **DigMA** | **0.029** | 0.411 | **0.049** | 1.313 |

### 关键发现

1. **可控性**：连续控制编码器在大多数指标上实现最低MSE，相比无控制基线降低了3-10倍
2. **保真度**：DigMA在4个stylized facts中的3个上实现最低KL散度，唯一例外RetAC是因为LOBGAN的自回归特性
3. **下游价值**：在DigMA生成环境中训练的RL交易Agent获得最高日收益和Sharpe比率（0.049 vs 基线最高0.014）
4. **计算效率**：DigMA每订单生成速度0.017ms，比LOBGAN快100倍

## 亮点与洞察

1. **两阶段设计精妙解决了宏观-微观gap**：扩散模型工作在压缩的市场状态空间（逐分钟收益率+到达率），而非原始订单流（一天数万条），巧妙规避了信噪比低和序列过长的问题
2. **金融经济学先验的融合**：异质交易者（基本面/图表/噪声）+CARA效用+双向拍卖机制，使生成的订单自然满足stylized facts
3. **Classifier-Free Guidance在金融领域的首次应用**：使模型同时支持无条件和条件生成
4. **下游任务验证了生成环境的实用价值**：RL交易策略的性能提升证明了DigMA生成的订单流质量

## 局限与展望

1. **仅支持单资产生成**：未考虑多资产间的相关性，未来需扩展到多资产联合生成
2. **控制指标有限**（仅收益率、振幅、波动率），未支持更细粒度的场景控制如特定的价格走势形态
3. **Meta Agent结构固定**：异质交易者的组件权重分布和效用函数形式是预定义的
4. **数据仅限中国A股**，跨市场（如美股）的泛化性未验证

## 相关工作与启发

- DigMA是扩散模型在金融订单流生成领域的开创性应用
- 金融市场模拟从"规则驱动"→"数据驱动"→本文"扩散引导+经济学先验"的演进路径值得关注
- 启发：在复杂系统模拟中，将深度生成模型与领域先验知识结合（而非纯数据驱动）是提高保真度的有效策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首次将扩散模型与金融经济学先验结合实现可控订单流生成)
- 实验充分度: ⭐⭐⭐⭐⭐ (可控性/保真度/下游任务/效率四维评估全面)
- 写作质量: ⭐⭐⭐⭐ (问题定义清晰，但方法细节分散在正文和附录)
- 价值: ⭐⭐⭐⭐⭐ (对金融模拟研究有重要实际意义，代码开源)

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](../../ICLR2026/others/compositional_diffusion_long_horizon_planning.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[AAAI 2026\] DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](../../ACL2025/others/bone_soups_multi_objective_gen.md)
- [\[AAAI 2026\] Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)

<!-- RELATED:END -->
