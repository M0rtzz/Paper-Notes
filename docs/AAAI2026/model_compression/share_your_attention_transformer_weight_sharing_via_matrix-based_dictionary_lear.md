---
title: >-
  [论文解读] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning
description: >-
  [AAAI 2026][模型压缩][weight sharing] 受字典学习启发，提出 MASA 框架，将 Transformer 各层注意力投影矩阵（Q/K/V/O）分解为共享矩阵原子的线性组合，以 66.7% 的注意力参数压缩率实现与原始 Transformer 持平甚至更优的性能。
tags:
  - AAAI 2026
  - 模型压缩
  - weight sharing
  - dictionary learning
  - Transformer
  - 注意力机制
  - parameter efficiency
---

# Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning

**会议**: AAAI 2026  
**arXiv**: [2508.04581](https://arxiv.org/abs/2508.04581)  
**代码**: [https://github.com/mts-ai/MASA](https://github.com/mts-ai/MASA)  
**领域**: 模型压缩  
**关键词**: weight sharing, dictionary learning, transformer compression, attention compression, parameter efficiency

## 一句话总结

受字典学习启发，提出 MASA 框架，将 Transformer 各层注意力投影矩阵（Q/K/V/O）分解为共享矩阵原子的线性组合，以 66.7% 的注意力参数压缩率实现与原始 Transformer 持平甚至更优的性能。

## 研究背景与动机

大语言模型（LLM）在实际部署中面临巨大的计算和内存压力。现有压缩技术主要聚焦于**块内优化**（如低秩近似、注意力头剪枝），但 Transformer 的重复层状结构暗示了巨大的**层间冗余**——这一维度除了 KV 缓存之外几乎未被探索。

具体来说，$L$ 层、隐藏维度 $d$ 的 Transformer 需要 $\mathcal{O}(L \cdot d^2)$ 个参数，光注意力模块就占了 LLaMA、Mistral 等基础模型一半的参数量。GQA、Sequential-sharing、Repeat-all-over 等近期方法虽然探索了跨层共享，但要么性能下降明显（尤其推理任务），要么缺乏捕捉跨层统计规律的原则性框架。

MASA 的核心动机是：**不同 Transformer 层的注意力权重矩阵之间存在可被字典学习捕获的跨层统计规律**，可以用少量共享矩阵原子的线性组合来表示各层权重，从而实现大幅参数压缩。

## 方法详解

### 整体框架

MASA 为 Q、K、V、O 四种投影分别维护独立的字典池（dictionary pool），每个字典包含 $S$ 个共享矩阵原子 $\mathbf{D}_s \in \mathbb{R}^{d \times h}$。每层每种投影的权重矩阵通过各自的线性系数向量 $\mathbf{c}_l \in \mathbb{R}^S$ 对共享原子进行加权求和重建：

$$\hat{\mathbf{W}}_l = \sum_{s=1}^{S} c_{ls} \mathbf{D}_s$$

字典原子和线性系数均通过反向传播在训练损失上联合学习，无需额外的字典学习损失约束。

### 关键设计

1. **矩阵原子共享机制（Matrix Atom Sharing）**:

    - 功能：将每层注意力投影矩阵表示为共享矩阵原子的线性组合
    - 核心思路：借鉴经典信号处理中的字典学习，将每个权重视为"信号"，用共享"字典"来重建
    - 设计动机：不同于低秩方法对所有层施加统一秩约束，MASA 允许有效秩随层和投影类型变化，更灵活地捕获跨层冗余
    - 压缩率公式：$r \approx 1 - S/L$，当 $S = L/3$ 时实现 66.7% 压缩

2. **MLP 参数化的混合系数（Embedding-based Coefficient Parameterization）**:

    - 功能：为每个 Transformer 块分配独立可训练嵌入向量，通过 3 层 MLP 预测混合系数 $\mathbf{c}_l$
    - 核心思路：将混合系数的优化与直接梯度更新解耦，平滑训练过程
    - 设计动机：减少梯度波动，起到隐式正则化作用；训练后丢弃 MLP 和嵌入，仅保留最终系数矩阵 $\mathbf{C}$，推理无额外开销

3. **预训练模型的 Matrix PCA 适配**:

    - 功能：对已有预训练 LLM 进行无微调压缩
    - 核心思路：解析求解最优正交矩阵基（Matrix PCA），再通过分组策略和局部残差精修提升重建精度
    - 分组策略：利用 KL 散度衡量相邻层输出分布的变化，将功能相似的层聚为一组共享字典
    - 局部精修：对残差 $\Delta \mathbf{W}_l$ 进行 Cholesky 白化变换后低秩近似，并根据矩阵角色（Q/K/V/O）自适应分配秩预算

### 损失函数 / 训练策略

- 训练损失：标准语言模型交叉熵损失，无额外辅助损失
- 优化器：AdamW（$\beta_1=0.9$, $\beta_2=0.999$, weight decay=0.1）
- 遵循 Chinchilla 最优训练策略：训练 token 数为模型参数量的 20 倍
- 学习率前 10% steps 线性预热，余弦退火
- 梯度裁剪为全局范数 1.0
- 使用 FlashAttention 加速长序列训练

## 实验关键数据

### 主实验

| 模型 (规模) | 方法 | 注意力压缩率 | AVG Acc (%) ↑ | WikiText PPL ↓ | LAMBADA PPL ↓ |
|------------|------|-------------|---------------|----------------|---------------|
| Transformer-S (110M) | Vanilla | 0% | 33.48 | 76.11 | 167.39 |
| Transformer-S | MASA-QKV | 50% | **34.43** | **72.08** | **112.23** |
| Transformer-S | MASA-QKVO | 66.7% | 33.74 | 72.82 | 133.62 |
| Transformer-S | Low-Rank | 66.7% | 32.27 | 83.25 | 264.52 |
| Transformer-S | GQA | 41.7% | 33.34 | 78.41 | 187.71 |
| Transformer-L (729M) | Vanilla | 0% | 42.12 | 30.88 | 20.73 |
| Transformer-L | MASA-QKV | 50% | 41.74 | **30.83** | 22.08 |
| Transformer-L | MASA-QKVO | 66.7% | 41.30 | 31.34 | 21.21 |

### 预训练模型压缩实验

| 模型 | 方法 | 压缩率 | AVG Acc (%) ↑ | WikiText PPL ↓ |
|------|------|--------|---------------|----------------|
| Llama 3.2 1B | Vanilla | 0% | 57.61 | 11.57 |
| Llama 3.2 1B | SVD-LLM | 20% | 53.11 | 15.08 |
| Llama 3.2 1B | Matrix PCA (本文) | 20% | **55.34** | **12.61** |
| Llama 3.1 8B | Vanilla | 0% | 70.93 | 7.33 |
| Llama 3.1 8B | Matrix PCA (本文) | 20% | **70.09** | **7.84** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 独立字典 vs QV共享字典 | Separate: 34.43% vs QV-Common: 33.95% | Q/K/V/O 使用独立字典效果最佳 |
| O 投影是否压缩 | QKV: 34.43% vs QKVO: 33.74% | O 投影更敏感，QKV 更可压缩 |
| 字典大小 S=2→8 | Acc 33.82%→33.94%，PPL 74.79→70.66 | 更大字典持续改善 |
| 大规模数据训练 (65B tokens) | MASA-QKV vs Vanilla: Acc 差仅 0.23% | 大规模训练下 MASA 保持竞争力 |

### 关键发现

- **MASA-QKV（50%压缩）可超越未压缩 Transformer**：在小模型上准确率高出约 1%，困惑度显著降低
- **Q/K/V 投影比 O 投影更具可压缩性**：即使高度压缩 Q/K/V（S=2, 62.5%），性能与 vanilla 模型相当
- **计算开销极小**：MASA-QKVO 相比原始模型仅降速约 8.3%（1240 vs 1352 tokens/sec）
- **ViT 上同样有效**：在 CIFAR-10/100、TinyImageNet 上 66.7% 压缩注意力参数，性能持平甚至超越 vanilla
- **预训练适配**：为 Llama 3.1 8B 做 20% 注意力压缩，保留约 99% 的下游准确率

## 亮点与洞察

- **理论优雅**：将注意力压缩重新构造为字典学习问题，建立了经典信号处理与 Transformer 效率之间的原则性联系
- **即插即用**：不需要蒸馏、正则化或架构修改，使用标准优化器训练，保留原始训练流程
- **预训练适配设计精巧**：Matrix PCA + KL 散度分组 + 自适应秩分配的组合，在无微调条件下显著优于 SVD-LLM
- **实用设计原则**：优先压缩 Q/K/V 投影，保留 O 投影的参数独立性——这为后续 Transformer 压缩工作提供了重要的经验指导

## 局限与展望

- 目前仅压缩注意力模块，FFN 模块（占参数量另一半）未涉及，可探索联合压缩
- 更大模型（>8B）上的 Matrix PCA 适配需要更细致的分组策略
- 字典原子之间的冗余随 $S$ 增大而增加，可引入字典稀疏化或秩约束学习
- 未探索与量化等其他压缩技术的组合效果
- 主要在语言建模上验证，在 instruction tuning 和下游微调场景中的表现有待检验

## 相关工作与启发

- **GQA**（Ainslie et al., 2023）：在 KV 头层面做参数共享，但局限于单层内部
- **Basis Sharing**（Wang et al., ICLR 2025）：从 SVD 共享奇异向量，但缺乏层级自适应控制
- **Repeat-all-over / Sequential-sharing**：确定性地跨层重复权重，过于刚性导致推理性能下降
- MASA 统一了字典学习与 Transformer 设计，提供了一个「full sharing ↔ full independence」的连续谱

## 评分

- 新颖性: ⭐⭐⭐⭐ 将字典学习引入 Transformer 跨层权重共享是新颖的视角，理论基础扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 3 个规模的 LLM、Vision Transformer、预训练模型适配，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论推导与实验结合良好
- 价值: ⭐⭐⭐⭐ 提供了实用的 Transformer 压缩方案和设计原则（优先压缩 QKV），对后续工作有指导意义

<!-- RELATED:START -->

## 相关论文

- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [Spark Transformer: Reactivating Sparsity in FFN and Attention](../../NeurIPS2025/model_compression/spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection](explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)
- [CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)

<!-- RELATED:END -->
