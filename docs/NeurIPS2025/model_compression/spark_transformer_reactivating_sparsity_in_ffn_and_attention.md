---
title: >-
  [论文解读] Spark Transformer: Reactivating Sparsity in FFN and Attention
description: >-
  [NeurIPS 2025][模型压缩][激活稀疏性] 提出 Spark Transformer 架构，通过 Statistical Top-k 算子在 FFN 和注意力机制中同时实现高水平激活稀疏性（FFN 仅 8% 神经元激活、每个 token 最多关注 256 个 token），在保持与 Gemma-2 相当质量的同时实现 2.5× FLOPs 降低和高达 1.79× 的推理加速。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 激活稀疏性
  - Statistical Top-k
  - 推理加速
  - FFN稀疏
  - 注意力稀疏
---

# Spark Transformer: Reactivating Sparsity in FFN and Attention

**会议**: NeurIPS 2025  
**arXiv**: [2506.06644](https://arxiv.org/abs/2506.06644)  
**代码**: 无（Google 内部项目，使用 Gemma-2 和 gemma.cpp）  
**领域**: Model Compression / Efficient Transformers  
**关键词**: 激活稀疏性, Statistical Top-k, 推理加速, FFN稀疏, 注意力稀疏

## 一句话总结
提出 Spark Transformer 架构，通过 Statistical Top-k 算子在 FFN 和注意力机制中同时实现高水平激活稀疏性（FFN 仅 8% 神经元激活、每个 token 最多关注 256 个 token），在保持与 Gemma-2 相当质量的同时实现 2.5× FLOPs 降低和高达 1.79× 的推理加速。

## 研究背景与动机

激活稀疏性是降低大模型计算成本的关键技术。早期使用 ReLU 激活的 Transformer（如 T5、ViT）天然具有"lazy neuron"现象——FFN 中绝大多数神经元对每个 token 不活跃。已有工作成功在 CPU、GPU、TPU 上利用这种稀疏性获得实际加速。

然而，核心挑战在于：**现代最先进的 Transformer 已抛弃 ReLU，转向 GELU 等非 ReLU 门控激活函数**（如 Gemma、LLaMA、Mistral），这些函数产生的激活几乎没有自然稀疏性。如何在不损害模型质量的前提下重新引入高水平稀疏性？

现有尝试面临三大挑战：
- **Challenge #1（质量损失）**：直接切换回 ReLU 或简单 top-k 掩码会降低模型质量
- **Challenge #2（训练减速）**：标准 top-k 需要排序，在 TPU 等加速器上可能导致 10× 训练减速
- **Challenge #3（额外参数/复杂度）**：引入稀疏预测器往往增加参数量和训练管线复杂度

## 方法详解

### 整体框架

Spark Transformer 包含两个组件：Spark FFN 和 Spark Attention。两者基于统一的设计理念——将 FFN 和注意力都视为**键值查找表**，使用低秩预测器识别活跃条目，然后仅对活跃部分进行完整计算。核心技术工具是 Statistical Top-k 算子，替代传统的排序型 top-k。

### 关键设计

1. **Spark FFN（核心架构创新）**：

    - 将输入 q 的维度分为两部分 q[:r] 和 q[r:]
    - q[:r] 与 K₁ 相乘作为低秩预测器，通过 Top-k 选出 k 个最重要的神经元
    - q[r:] 与 K₂ 的稀疏列相乘，仅计算被选中的 k 列
    - V 也仅计算对应的 k 行
    - 公式：Spark-FFN(q) = V · (σ(Top_k(K₁ᵀ·q[:r])) ⊙ (K₂ᵀ·q[r:]))
    - **FLOPs 分析**：当 r≈d_model/2，总 FLOPs 约为 d_model·d_ff + 3·d_model·k，当 k 很小时约为标准 FFN 的 1/4
    - **参数不增加**：K₁、K₂ 是从原始 K 的维度分割得到，总参数量等同于 Gated FFN

2. **Spark Attention（统一稀疏框架）**：

    - 注意力与 FFN 形式相同（Eq.6 vs Eq.1），因此可应用相同策略
    - 将键向量 K 的维度分为 K₁（预测用）和 K₂（计算用）
    - σ₁ = softmax, σ₂ = softplus（经验选择）
    - 每个 token 最多关注 k_attn = 256 个 token
    - FLOPs 约为 d_model·n_ctx + 3·d_model·min{k_attn, n_ctx}，当 k_attn ≪ n_ctx 时接近 4× 降低

3. **Statistical Top-k（核心技术工具）**：

    - **动机**：标准 top-k 需要 O(d log d) 的排序，在加速器上极慢；且硬阈值不可微
    - **核心思想**：假设激活分数近似高斯分布，用样本均值和标准差估计一个阈值 θ，使得约 k 个条目超过该阈值
    - 阈值公式：θ(x,k) = mean(x) + std(x) · Q(1-k/d)，其中 Q 是标准正态分位函数
    - 然后使用**软阈值算子**（而非硬阈值）：Soft-Threshold(x,θ) = max{x-θ·1, 0}
    - **理论保证（Theorem 3.1）**：实际选中数量与 k 的偏差为 O(√(log d / d))
    - **计算复杂度**：仅需 2d FLOPs（类似 LayerNorm），远低于排序的 O(d log d)
    - **可微性（Theorem 3.2）**：使用 Huber 函数平滑后连续可微

4. **与 Gated FFN 的关系**：Spark FFN 与 Gated FFN（如 Gemma 使用的）结构类似——都有两个线性映射在第一层、一个在第二层。关键差异是：(1) 加入 Top-k 获得稠密稀疏；(2) 输入维度分割而非共享。

### 损失函数 / 训练策略

使用与 Gemma-2 2B 完全相同的训练数据和流程（2T tokens 英文文本）。Spark Transformer 设计的巧妙之处在于：
- 不引入额外参数（通过维度分割复用 FFN 参数和注意力键嵌入）
- 不需要多阶段训练（所有参数在一个阶段训练）
- Statistical Top-k 的线性复杂度和近似可微性避免了训练减速

## 实验关键数据

### 主实验

| 模型 | 训练损失（相对） | FLOPs/token（相对）| 下游任务 | 说明 |
|------|---------------|------------------|---------|------|
| Gemma-2 2B | 1.00 | 1.00 | 基准 | 无稀疏性 |
| ReLU 替换 | ~1.02 | ~0.65 | 质量损失大 | 自然稀疏但质量差 |
| ReLU² 替换 | ~1.005 | ~0.80 | 质量轻降 | FLOPs 降低不足 |
| Top-k + GELU | ~1.02 | ~0.55 | 质量损失大 | 无预测器的 top-k |
| **Spark FFN** | ~1.005 | **~0.55** | 质量几乎不变 | 预测器同时提升质量和效率 |
| **Spark Transformer** | **~1.00** | **~0.40** | **与 Gemma-2 持平** | FFN + Attention 稀疏 |

### 消融实验

| 平台 | Gemma-2 Decode | Spark Decode | 加速比 |
|------|---------------|-------------|--------|
| 4-Core CPU | 141 ms/token | 86 ms/token | 1.64× |
| 16-Core CPU (L=512) | baseline | - | 1.79× |
| NVIDIA T4 GPU (L=4096) | baseline | - | 1.40× |

| CPU Prefill | Gemma-2 | Spark | 加速比 |
|------------|---------|-------|--------|
| 4-Core (4096 tokens) | 28 ms/token | 15 ms/token | 1.86× |

### 关键发现
- **预测器的惊人效果**：Spark FFN 与简单 Top-k+GELU 的唯一区别是低秩预测器的引入，但这不仅减少 FLOPs，还意外地**提升了模型质量**
- Statistical Top-k 在整个训练过程中稳定维持目标稀疏度（8% FFN, ≤256 attention）
- 稀疏性不仅减少 FLOPs，还减少内存带宽需求，在 memory-bound 场景（如 decode）同样加速
- Spark Attention + Spark FFN 的组合进一步提升质量，暗示两种稀疏性有互补效应

## 亮点与洞察
- **Statistical Top-k 的精妙设计**：用高斯假设 + 分位数函数在 O(d) 时间内近似 top-k，类 LayerNorm 的计算让其在加速器上高效运行，避免了困扰该领域的排序瓶颈
- **维度分割复用参数**：将 FFN 的键矩阵按维度分为"预测器"和"计算器"，不增加参数就实现了稀疏预测
- **FFN 与 Attention 的统一视角**：将两者都视为键值查找表，用统一框架引入稀疏性
- 该技术对 MoE 的 routing 机制也可能适用（避免排序的 O(d log d) 开销）

## 局限与展望
- 目前仅在 Gemma-2 2B 上验证，更大规模模型（如 7B/70B）上的效果有待验证
- Statistical Top-k 的高斯假设在训练后可能不精确（虽然实验证明仍近似成立）
- CPU/GPU 上的加速实现可能需要针对不同硬件平台做定制优化
- 8% 的 FFN 稀疏度和 256 的注意力限制是固定的，自适应调整可能更优
- 仅展示了语言模型实验，视觉和多模态模型上的效果未知

## 相关工作与启发
- **vs ReLU 稀疏性 (Li et al. 2022)**: 原始 lazy neuron 现象仅存在于 ReLU Transformer；Spark 通过 top-k 在 GELU 上也实现稀疏，且更可控
- **vs DejaVu (Liu et al. 2023)**: 使用额外 MLP 预测器，增加参数和训练复杂度；Spark 通过维度分割避免了这些问题
- **vs Soft Top-k (Lei et al. 2023) / SparseK (Lou et al. 2024)**: 这些方法通过优化问题定义 top-k，需要迭代求解；Statistical Top-k 有闭式解且保证近似 k 个非零输出

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Statistical Top-k 和维度分割预测器都是优雅的创新，统一框架令人信服
- 实验充分度: ⭐⭐⭐⭐ 完整的 Gemma-2 预训练验证和多平台推理评估，但模型规模单一
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、理论完整、图示优秀
- 价值: ⭐⭐⭐⭐⭐ 解决了现代 Transformer 激活稀疏性的核心痛点，实用性极强

<!-- RELATED:START -->

## 相关论文

- [Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)

<!-- RELATED:END -->
