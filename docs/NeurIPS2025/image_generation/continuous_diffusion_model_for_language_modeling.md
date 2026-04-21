---
title: >-
  [论文解读] Continuous Diffusion Model for Language Modeling
description: >-
  [NeurIPS2025][图像生成][扩散模型] 提出 RDLM（Riemannian Diffusion Language Model），在统计流形（超球面）上构建连续扩散过程来建模离散分布，建立了离散扩散与连续流的理论联系，通过径向对称性实现无模拟训练和维度分裂技术处理大词表，在 Text8 上以 1.32 BPC 超越所有离散和连续扩散模型。
tags:
  - NeurIPS2025
  - 图像生成
  - 扩散模型
  - statistical manifold
  - discrete data
  - sphere
  - language model
---

# Continuous Diffusion Model for Language Modeling

**会议**: NeurIPS2025  
**arXiv**: [2502.11564](https://arxiv.org/abs/2502.11564)  
**代码**: [GitHub](https://github.com/harryjo97/RDLM)  
**领域**: 生成模型 / 语言建模  
**关键词**: [Riemannian diffusion, statistical manifold, discrete data, sphere, language model]

## 一句话总结

提出 RDLM（Riemannian Diffusion Language Model），在统计流形（超球面）上构建连续扩散过程来建模离散分布，建立了离散扩散与连续流的理论联系，通过径向对称性实现无模拟训练和维度分裂技术处理大词表，在 Text8 上以 1.32 BPC 超越所有离散和连续扩散模型。

## 研究背景与动机

**领域现状**：离散扩散模型（D3PM、SEDD、MDLM）直接在离散状态空间上通过马尔可夫链建模，已在语言建模中展现竞争力。但离散状态之间的跳跃式转移导致信号丢失，无法充分利用迭代精修（iterative refinement）这一连续扩散的核心优势。

**现有痛点**：(1) 离散扩散在状态间直接跳转，错误转移不可逆，限制了生成质量和可控性；(2) 已有连续扩散方法（如在欧氏空间做松弛）忽略了分类分布的几何结构，性能显著落后于离散方法；(3) 在统计流形上的流匹配方法（Fisher-Flow、CatFlow）仅限于短序列和小词表。

**核心矛盾**：如何在保留分类分布几何结构的同时，让离散数据享受连续扩散的迭代精修优势，并扩展到大词表长序列场景。

**本文目标**：建立离散扩散与连续流的统一理论联系，设计实用的连续扩散框架替代离散跳跃。

**切入角度**：利用分类分布的统计流形（概率单纯形）与超球面正象限的微分同胚 $\pi: p_i \mapsto \sqrt{p_i}$。

**核心 idea**：离散扩散的转移分布可以建模为统计流形上的连续流，而在超球面上构建桥过程的混合可以泛化并改进离散扩散。

## 方法详解

### 整体框架

RDLM 的核心思路是将离散 token 通过 one-hot 编码映射到超球面 $\mathbb{S}^{d-1}$ 上的标准基 $\mathbf{e}_k$，然后在超球面上构建从初始点（掩码 token $\mathbf{e}_m$ 或均匀点 $\sum \mathbf{e}_i/\sqrt{d}$）到目标 token $\mathbf{e}_k$ 的桥过程，再用扩散混合表示法组合所有桥过程生成完整的生成过程。模型预测到达概率 $p_{T|t}(\mathbf{e}_k|\mathbf{X}_t)$，用交叉熵损失训练，通过黎曼正态分布近似转移分布实现无模拟训练。

### 关键设计

1. **离散扩散到连续流的统一 (Proposition 3.1)**:

    - 功能：证明离散扩散过程的转移分布可以由超球面上的连续流建模
    - 核心思路：分类分布的参数空间（概率单纯形 $\Delta^{d-1}$）通过 Fisher-Rao 度量形成统计流形 $\mathcal{P}(\mathcal{X})$，微分同胚于 $\mathbb{S}^{d-1}_+$。在此映射下，离散扩散的转移矩阵 $\bar{Q}_t$ 对应的分类分布 $\text{Cat}(x_t; \bar{Q}_t x)$ 可以由超球面上的测地线 ODE $\frac{d\mathbf{Y}_t}{dt} = -\frac{d\log\kappa_t}{dt}\exp^{-1}_{\mathbf{Y}_t}(\mathbf{y}_1)$ 的流精确重现。特别地，$\mathbf{y}_1 = \mathbf{e}_m$ 得到掩码扩散，$\mathbf{y}_1 = \sum \mathbf{e}_i/\sqrt{d}$ 得到均匀扩散
    - 设计动机：建立理论联系后，可以将离散跳跃"平滑化"为连续轨迹，中间状态提供持续修正机会

2. **基于径向对称性的无模拟训练**:

    - 功能：利用超球面的径向对称性推导出转移分布的可控近似，避免训练时模拟昂贵的 SDE
    - 核心思路：将 $d$ 维桥过程的转移分布近似为黎曼正态分布 $\mathcal{N}_{\mathbb{S}^{d-1}}(\boldsymbol{\mu}_t, \rho_t^2 \mathbf{I})$。参数 $\alpha_t, \rho_t$ 通过一维投影过程 $z_t^T = \langle \mathbf{X}_t, \mathbf{e}_k \rangle$ 和 $z_t^0 = \langle \mathbf{X}_t, \mathbf{X}_0 \rangle$ 推导，只需预计算一维 SDE 的矩即可。训练目标采用交叉熵损失 $\mathcal{L}^{CE}(\theta) = \mathbb{E}[-\log\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_k\rangle]$，与离散扩散的训练目标形式一致
    - 设计动机：直接模拟高维超球面上的 SDE 计算成本极高；径向对称性使得所有方向的统计量相同，可从一维投影恢复高维分布参数，实现约 50 倍加速

3. **维度分裂 (Dimension Splitting)**:

    - 功能：将大词表 token 用 $b$ 进制表示，从 $\mathbb{S}^{d-1}$ 映射到 $(S^b)^m$（$m = \lceil\log_b d\rceil$），降低每个超球面的维度
    - 核心思路：高维超球面上的桥过程在终端时间附近表现出"尖锐转变"，神经网络难以学习。将 $d$ 维球面分裂为 $m$ 个 $b$ 维球面后，每个球面上的过程更加平缓。配合掩码扩散和均匀扩散的混合路径（Eq. 9）$\lambda_t \mathbb{Q}_t^{mask} + (1-\lambda_t)\mathbb{Q}_t^{unif}$ 使用效果最佳
    - 设计动机：语言模型的词表通常数万级别，直接在 $\mathbb{S}^{30000}$ 上训练不可行；维度分裂加混合路径是使框架扩展到实际词表的关键技术

### 损失函数 / 训练策略

交叉熵损失 $\mathcal{L}^{CE}(\theta) = \mathbb{E}[\int_0^T -\log\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_k\rangle dt]$，配合重要性采样 $q(t)$ 集中在困难时间点。几何噪声调度用 $\sigma_t = \sigma_0^{T-t}\sigma_T^t$（$\sigma_0 < \sigma_T$）保证渐进收敛。采样时使用测地线随机游走：$\mathbf{X} \leftarrow \exp_{\mathbf{X}}(\eta_\theta \delta t + \sigma_t\sqrt{\delta t}\mathbf{w})$。

## 实验关键数据

### 主实验

Text8（字符级语言建模，BPC↓）：

| 方法 | 类型 | BPC |
|------|------|-----|
| Transformer AR | 自回归 | 1.23 |
| ARDM | 任意序自回归 | ≤1.43 |
| D3PM Absorb | 离散扩散 | ≤1.45 |
| SEDD Absorb | 离散扩散 | ≤1.39 |
| MDLM | 离散扩散 | ≤1.40 |
| MD4 | 离散扩散 | ≤1.37 |
| BFN | 连续扩散 | ≤1.41 |
| **RDLM (Ours)** | **连续扩散** | **≤1.32** |

LM1B（PPL↓）：

| 方法 | 参数量 | PPL |
|------|--------|-----|
| MDLM | 110M | ≤27.04 |
| Diffusion-LM | 80M | ≤118.62 |
| **RDLM (Ours)** | 110M | **≤28.44** |

CIFAR-10（像素级图像建模，BPD↓）：

| 方法 | BPD |
|------|-----|
| MD4 | ≤2.78 |
| Sparse Transformer | 2.80 |
| **RDLM (Ours)** | **≤2.73** |

### 消融实验

| 配置 | Text8 BPC | 说明 |
|------|-----------|------|
| MSE 损失 | 较高 | 收敛慢 |
| 交叉熵损失 | 更低 | 收敛快，性能好 |
| 无重要性采样 | 较高 | 困难时间点欠训练 |
| 有重要性采样 | 最低 | 集中训练困难区间 |
| 无维度分裂 (LM1B) | 失败 | 高维球面训练不可行 |
| 维度分裂 + 混合路径 | 最优 | 大词表必需 |

### 关键发现

- RDLM 在 Text8 上 BPC=1.32 是当前所有扩散模型（离散+连续）的最佳结果
- 在 CIFAR-10 像素建模上也超越连续域的自回归模型，展示了跨模态潜力
- 黎曼正态近似的 MMD 距离在高维时趋近零，证实近似质量随维度提升
- DNA 序列设计上也达到 SOTA（MSE=0.027），验证了框架的通用性

## 亮点与洞察

- **理论优美**：Proposition 3.1 建立了离散扩散与统计流形上连续流的精确对应，为两类方法的统一提供了数学基础
- **连续化的优势**：离散跳跃不可逆，连续轨迹允许渐进修正——这正是扩散模型在连续域成功的核心
- **无模拟训练**：径向对称性将 $d$ 维问题降至一维预计算，训练速度提升 50 倍
- **混合路径创新**：混合掩码和均匀扩散的时变路径，泛化了离散流匹配和状态相关调度

## 局限与展望

- LM1B 上 PPL=28.44 仍落后于 MDLM (27.04) 和自回归模型 (22.32)，大规模语言任务上差距待缩小
- 维度分裂引入了 base 选择的超参，且 base 编码可能破坏 token 的语义邻接关系
- 采样需要多步 SDE 模拟（测地线随机游走），速度慢于离散扩散的并行解码
- 未探索条件生成和可控生成等应用场景

## 相关工作与启发

- **D3PM / SEDD / MDLM**：离散扩散基线，RDLM 证明了它们是连续框架的特例
- **Fisher-Flow / CatFlow**：统计流形上的流匹配方法，但限于短序列小词表；RDLM 通过维度分裂突破了这一限制
- **Dirichlet 扩散 (DDSM)**：用 Dirichlet 分布做概率单纯形上的先验，但未利用 Fisher-Rao 几何
- **启发**：统计流形提供了离散-连续统一的自然框架，维度分裂思想可能启发其他高维结构化生成问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献突出——离散-连续统一视角、径向对称无模拟训练、维度分裂
- 实验充分度: ⭐⭐⭐⭐ 跨三个模态（文本/图像/DNA）验证，消融充分
- 写作质量: ⭐⭐⭐⭐ 理论部分推导清晰，但部分符号较重
- 价值: ⭐⭐⭐⭐ 为离散数据的扩散建模开辟了统一的几何视角

<!-- RELATED:START -->

## 相关论文

- [Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](../../ICML2025/image_generation/generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)
- [NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](../../ECCV2024/image_generation/nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)

<!-- RELATED:END -->
