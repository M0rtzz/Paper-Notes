---
title: >-
  [论文解读] Graph Generative Pre-trained Transformer (G2PT)
description: >-
  [ICML2025][图像生成][图生成] 提出 G2PT——将图编码为节点+边的 token 序列，用 GPT 风格的自回归 Transformer 做 next-token prediction 来生成图，并通过拒绝采样微调(RFT)和 PPO 强化学习实现目标导向分子生成，在通用图和分子数据集上均达到 SOTA。
tags:
  - ICML2025
  - 图像生成
  - 图生成
  - Transformer
  - 边序列表示
  - 分子生成
  - 预训练微调
---

# Graph Generative Pre-trained Transformer (G2PT)

**会议**: ICML2025  
**arXiv**: [2501.01073](https://arxiv.org/abs/2501.01073)  
**代码**: [tufts-ml/G2PT](https://github.com/tufts-ml/G2PT)  
**领域**: 图生成 / 分子设计  
**关键词**: 图生成, 自回归Transformer, 边序列表示, 分子生成, 预训练微调

## 一句话总结

提出 G2PT——将图编码为节点+边的 token 序列，用 GPT 风格的自回归 Transformer 做 next-token prediction 来生成图，并通过拒绝采样微调(RFT)和 PPO 强化学习实现目标导向分子生成，在通用图和分子数据集上均达到 SOTA。

## 研究背景与动机

- **核心问题**: 图生成模型大多基于邻接矩阵表示，计算复杂度为 $O(n^2)$，对稀疏图效率低；离散扩散模型在每步独立采样矩阵元素，存在累积解码误差(compounding decoding error)
- **已有方案不足**:
    - 扩散模型(DiGress, EDGE)需要多步去噪 + 排列不变的 GNN 架构，架构选择受限
    - 早期序列模型(GraphRNN, DeepGMG)基于 RNN/LSTM，仍以邻接矩阵行展开，序列长 $O(n^2)$
    - 边列表方法(DeepGMG, BiGG)依赖 GNN 学节点表示，表达能力有限
- **动机**: 借鉴 LLM 的巨大成功，设计基于边列表的高效 token 序列表示，让通用 Transformer 架构可直接用于图生成

## 方法详解

### 1. 图的 Token 序列表示

将图 $G = (V, E)$ 编码为先列节点后列边的一维序列：

$$[\underbrace{v_1^c, v_1^{id}, \ldots, v_n^c, v_n^{id}}_{n \times 2},\ a_\Delta,\ \underbrace{v_{src}^{id}, v_{dest}^{id}, e_1^c, \ldots, v_{src}^{id}, v_{dest}^{id}, e_m^c}_{m \times 3}]$$

- 节点 token: $(v^c, v^{id})$，类型 + 索引
- 边 token: $(v_{src}^{id}, v_{dest}^{id}, e^c)$，源节点、目标节点、边类型
- $a_\Delta$: 分隔节点段和边段的特殊 token
- 序列长度 $O(2n + 3m)$，对稀疏图远优于邻接矩阵的 $O(n^2)$

### 2. 边排序策略

采用**基于度数的边移除逆序**(Degree-Based Edge Removal)：
- 每步移除当前最低度节点的边，逆序后得到"先构建稠密核心、再扩展外围"的边序列
- 相比 BFS/DFS 排序，更适合稀疏图的生成学习

### 3. 自回归训练

使用标准 Transformer Decoder，词表包含节点ID $\{1,\ldots,n_{max}\}$、节点类型、边类型和特殊 token [SOG]/[EOG]/$a_\Delta$。

预训练损失为负对数似然：

$$\mathcal{L}_{pt}(\mathbf{s};\theta) = -\log p_\theta(\mathbf{s}) = \sum_{l=1}^{L} -\log p_\theta(s_l | \mathbf{s}_{<l})$$

**理论保证**: 最大化序列似然等价于最大化图似然的下界，通过数据增强（同一图多种序列）可收紧该下界。

### 4. 目标导向微调

**拒绝采样微调 (RFT)**:
- 用预训练模型采样，保留满足 $|z^* - \zeta(G)| < \omega$ 的样本构建微调集
- Self-Bootstrap (SBS): 逐步收紧阈值 $\omega_1 > \omega_2 > \ldots > \omega_k$，多轮迭代逼近目标分布

**强化学习 (PPO)**:
- KL 正则化目标: $\phi^* = \arg\max_\phi \mathbb{E}_{p_\phi}[r_{z^*}(\mathbf{s}) - \rho_1 \mathrm{KL}(p_\phi \| p_\theta)]$
- 仅在 [EOG] token 处给奖励，其余位置奖励为 0
- Critic 模型同架构初始化，综合损失: $\mathcal{L}_{ppo} = \mathcal{L}_{pg\text{-}clip} + \rho_2 \mathcal{L}_{critic} + \rho_3 \mathcal{L}_{pt}$

### 5. 图属性预测微调

取最后 token 的最后一层 Transformer 激活 $\mathbf{h}$ 作为图表示，接 MLP 做分类：

$$p(y|\mathbf{s}) = \text{Softmax}(\text{Dropout}(\text{Linear}(\mathbf{h})))$$

解冻 Transformer 后半部分参数进行微调，效果显著优于冻结全部参数。

## 实验关键数据

### 模型规格

| 规格 | 层数 | 注意力头 | $d_{model}$ | 参数量 |
|------|------|---------|------------|--------|
| Small | 6 | 6 | 384 | ~10M |
| Base | 12 | 12 | 768 | ~85M |
| Large | 24 | 16 | 1024 | ~300M |

### 通用图生成 (Table 2, V.U.N. ↑)

| 模型 | Planar | Tree | Lobster | SBM |
|------|--------|------|---------|-----|
| DiGress | 77.5 | 90 | - | 60 |
| HSpectre | 95 | 100 | - | 45 |
| DeFoG | 99.5 | 96.5 | - | 90 |
| **G2PT-base** | **100** | 99 | **100** | **100** |

### 分子生成 (Table 4)

| 模型 | MOSES Validity↑ | MOSES FCD↓ | GuacaMol Validity↑ | GuacaMol FCD↑ |
|------|-----------------|-----------|--------------------|--------------| 
| DiGress | 85.7 | 1.19 | 85.2 | 68.0 |
| DeFoG | 92.8 | 1.95 | 99.0 | 73.8 |
| **G2PT-large** | **97.2** | **1.02** | **95.3** | **92.7** |

### QM9 (FCD↓)

| 模型 | FCD↓ |
|------|------|
| DisCo | 0.25 |
| Cometh | 0.11 |
| **G2PT (all sizes)** | **0.06** |

### 分子属性预测 (MoleculeNet, ROC-AUC, Table 5)

| 模型 | 平均 |
|------|------|
| GraphMAE | 73.3 |
| **G2PT-base** | **73.3** |
| G2PT-base (无预训练) | 64.9 |

预训练带来 +8.4% 的平均提升，验证了 G2PT 学到的图表示的有效性。

## 亮点与洞察

1. **表示创新**: 用边列表 token 序列替代邻接矩阵，序列长度从 $O(n^2)$ 降至 $O(n+m)$，在 Planar 数据集上 token 数从 2018 降到 737，性能反而更优
2. **统一预训练-微调范式**: 像 NLP 一样走"预训练 + 任务微调"路线，RFT 和 PPO 均可有效引导目标分子生成
3. **RFT+SBS vs PPO**: RFT 配合多轮自举在 GSK3β 这种难任务上远超 PPO——PPO 受 KL 正则约束过强，难以跨越分布壁垒
4. **缩放规律**: 模型参数从 1M 扩到 1.5B 呈现清晰的缩放趋势，数据增强(同图多序列)亦有显著效果
5. **无需排列不变**: 放弃图的排列不变性假设，用数据增强弥补，实际效果优于需要排列不变 GNN 的扩散方法

## 局限与展望

1. **排序敏感**: 不同图域可能需要不同的边排序策略，目前缺乏通用的排序方案
2. **未利用 3D 信息**: 仅用 2D 拓扑，未融合分子的立体构型/手性信息
3. **Scaffold 多样性不足**: MOSES 上 scaffold similarity 较低(G2PT-large 仅 2.9)，说明模型倾向于记忆训练集骨架
4. **大图可扩展性**: 虽然比邻接矩阵高效，但 Transformer 的 $O(L^2)$ 注意力在超大图上仍是瓶颈
5. **PPO 效果有限**: 在高难度目标属性任务(GSK3β)上 PPO 无法突破分布壁垒，RFT+SBS 更可靠但需要多轮采样

## 相关工作与启发

- **DiGress / EDGE**: 离散扩散基线，全邻接矩阵去噪，排列不变但架构选择受限
- **BiGG**: 二分图生成，直接建模边列表但依赖 GNN 做节点表示
- **GraphMAE**: 自监督预训练图表示，G2PT 的属性预测与之持平但仅用 2D 信息
- **启发**: 将 GPT 范式引入结构化数据(图/分子)是有前景的方向；token 化表示设计是关键瓶颈

## 评分

- 新颖性: ⭐⭐⭐⭐ (边序列表示 + GPT 式图生成范式，思路清晰且有效)
- 实验充分度: ⭐⭐⭐⭐⭐ (7个生成数据集 + 8个预测数据集 + 缩放分析 + 目标导向生成)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论推导完整)
- 价值: ⭐⭐⭐⭐ (统一了图生成的预训练-微调范式，实用性强)

<!-- RELATED:START -->

## 相关论文

- [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation](simple_and_critical_iterative_denoising_a_recasting_of_discrete_diffusion_in_gra.md)
- [Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](../../NeurIPS2025/image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation](revisiting_diffusion_models_from_generative_pre-training_to_one-step_generation.md)
- [Normalizing Flows are Capable Generative Models](normalizing_flows_are_capable_generative_models.md)

<!-- RELATED:END -->
