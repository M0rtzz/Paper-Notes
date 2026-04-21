---
title: >-
  [论文解读] MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training
description: >-
  [ICML 2025][人体理解][大批量训练] 提出 MERIT 优化器，通过最大范数归一化与逐元素信任比率扩展 LAMB，有效解决大批量训练中注意力 logit 爆炸导致的性能退化问题。
tags:
  - ICML 2025
  - 人体理解
  - 大批量训练
  - 优化器
  - 信任比率
  - 最大范数
  - 语言模型
---

# MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training

**会议**: ICML 2025  
**arXiv**: [2508.20577](https://arxiv.org/abs/2508.20577)  
**代码**: [GitHub](https://github.com/NUS-HPC-AI-Lab/MERIT) (有)  
**领域**: LLM效率/优化  
**关键词**: 大批量训练, 优化器, 信任比率, 最大范数, 语言模型

## 一句话总结

提出 MERIT 优化器，通过最大范数归一化与逐元素信任比率扩展 LAMB，有效解决大批量训练中注意力 logit 爆炸导致的性能退化问题。

## 研究背景与动机

**领域现状**：大规模语言模型训练通常依赖数据并行来加速，增大 batch size 可以线性提升吞吐量。AdamW 是当前最主流的优化器，LAMB 等层自适应方法在 BERT 时代展示了大批量训练的潜力，但在 decoder-only 架构上的表现尚未被充分验证。

**现有痛点**：当 batch size 显著增大时（如从 512 扩展到 4K-8K），AdamW 的验证损失出现明显退化。这一现象在 GPT-2 等自回归模型上尤为严重，直接限制了训练效率的线性扩展。

**核心矛盾**：大批量训练需要更强的梯度统计稳定性，但 L2 范数归一化在面对极端值时过于"温和"——注意力权重矩阵中最大 logit 的急剧增长形成信息瓶颈（softmax 退化为 one-hot），而现有优化器的层级信任比率无法捕捉参数矩阵内部行列之间的结构差异。

**本文目标** 设计一种能在大批量场景下维持训练稳定性、抑制注意力 logit 爆炸、同时保持几乎零额外计算开销的新优化器。

**切入角度**：从分析 AdamW 大批量退化的根因入手——发现 max attention logit 的急剧上升是关键瓶颈（上界与 $W_Q$、$W_K$ 的 max norm 直接相关），进而将 LAMB 的 L2 范数替换为 max 范数、并利用多头注意力和 outlier dimension 带来的行列内部相似性引入逐元素的细粒度信任比率。

**核心 idea**：用最大范数替代 L2 范数、用逐元素信任比率替代层级信任比率，让优化器能精细控制每个参数的更新幅度。

## 方法详解

### 整体框架

MERIT 在 LAMB 优化器基础上做了三处关键改进。首先计算标准的 Adam 一阶/二阶矩估计得到更新方向 $u_t$，然后通过最大范数（而非 L2 范数）计算权重与更新的比值作为全局信任比率 $b_t$。接着，对权重矩阵的每一行和每一列分别计算局部信任比率 $r_t^{(i)}$ 和 $c_t^{(j)}$，取其与全局比率的最大值组合为逐元素的缩放因子 $s_t^{(i,j)}$。最后，通过逐元素裁剪防止单个参数更新过大，完成参数更新。

### 关键设计

1. **最大范数信任比率（Maximum-normalized Trust Ratio）**:

    - 功能：提供全局的参数更新缩放基准，直接约束导致 attention logit 爆炸的 Q/K 权重极值
    - 核心思路：将 LAMB 中的 $b_t = \|w_t\|_2 / \|u_t + \lambda w_t\|_2$ 替换为 $b_t = \|w_t\|_m / \|u_t + \lambda w_t\|_m$，其中 $\|\cdot\|_m$ 为最大范数。由于 attention logit 上界与 $M_Q \cdot M_K$ 成正比，而 $\ell_2$ 范数和 max norm 之间相对差异高达 99%+，L2 范数缩放无法有效抑制极端权重值
    - 设计动机：LAMB 在浅层注意力的 max attention logit 上成功降低，但中间层仍然存在急剧增长——分析发现这些层正是 max norm 与 L2 norm 差距最大的层。最大范数直接关注参数矩阵中的最大值，能在第一时间感知并限制异常增长

2. **逐元素信任比率（Element-wise Trust Ratio）**:

    - 功能：为矩阵中每个元素提供独立的更新缩放，利用行列内部权重相似性
    - 核心思路：分别计算行级比率 $r_t^{(i)} = \|w_t^{(i,\cdot)}\|_m / \|g_t^{(i,\cdot)}\|_m$ 和列级比率 $c_t^{(j)}$，最终 $s_t^{(i,j)} = \max\{\max\{r_t^{(i)}, c_t^{(j)}\}, b_t\}$。取最大值确保每个元素受到行和列两方向中更保守的约束，全局比率作为下界防止更新过小
    - 设计动机：多头注意力导致行内权重相似，outlier dimension 导致列内权重相似。层级 ratio 把所有行列混在一起——某一行的极端值通过全局范数影响其他行的更新，导致训练不稳定。逐元素机制隔离了这种跨行列干扰

3. **逐元素裁剪（Element-wise Clipping）**:

    - 功能：防止单个参数更新幅度超出安全范围
    - 核心思路：$w_{t+1} = w_t - \eta_t \cdot \text{clip}(s_t \cdot (u_t + \lambda w_t), 1)$，将缩放后的更新量裁剪到 $[-1, 1]$。分析显示裁剪主要发生在中间层（第 6 层裁剪率峰值 12%），浅层和深层几乎不裁剪
    - 设计动机：逐元素信任比率可能在某些位置产生过大的缩放因子，裁剪操作提供了最后一道安全保障，确保训练过程不会因个别参数的剧烈变化而不稳定

### 损失函数 / 训练策略

MERIT 使用标准的自回归语言建模交叉熵损失，不引入额外的正则化项。训练策略上采用线性 warmup + 余弦退火学习率调度。权重衰减通过优化器内置的 $\lambda w_t$ 项实现。收敛分析（Theorem 1）证明了简化版本（$\beta_1=0, \lambda=0$）在光滑性和有界梯度假设下的 $O(1/\sqrt{T})$ 收敛速率。整个优化器的额外计算仅涉及 max 操作和逐元素除法，wall-clock 开销不到 1%。

## 实验关键数据

### 主实验

| 模型 | Batch Size | AdamW | LAMB | MERIT | 提升 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-2 Small (125M) | 1K | 3.470 | 3.355 | **3.280** | -5.5% vs AdamW |
| GPT-2 Medium (355M) | 4K | 3.172 | 3.068 | **2.982** | -6.0% vs AdamW |
| GPT-2 Large (770M) | 8K | 3.039 | 2.971 | **2.897** | -4.7% vs AdamW |
| Llama-130M | 1K | 3.277 | 3.265 | **3.199** | -2.4% vs AdamW |
| Llama-350M | 4K | — | 3.001 | **2.957** | -1.5% vs LAMB |

| 模型 | 方法 | Zero-shot 平均准确率 | Hessian 最大特征值 | Hessian Trace |
|:---:|:---:|:---:|:---:|:---:|
| GPT-2 Small | AdamW | 43.56 | 37.231 | 12994.91 |
| GPT-2 Small | MERIT | **43.87** | **12.326** | **3444.92** |

### 消融实验

| 配置 | GPT-2 Small Val Loss | 变化 |
|:---:|:---:|:---:|
| MERIT（完整） | **3.280** | — |
| 去掉逐元素裁剪 | ~3.320 | +0.04 |
| 去掉权重级比率下界 | ~3.360-3.380 | +0.08-0.10 |
| maxLAMB（仅替换范数，无逐元素） | 3.304 | +0.024 |
| L2 范数 + 逐元素比率 | 3.312 | +0.032 |
| maxLAMB vs LAMB | 3.304 vs 3.355 | 仅 norm 选择改进有限 |

### 关键发现

- MERIT 在所有模型规模和 batch size 配置下均一致优于 AdamW 和 LAMB，验证损失改善幅度为 2.4%-6.0%
- Scaling law 分析显示 MERIT 的"ideal scaling"区间更宽——GPT-2 Medium 在 6K batch 下与 AdamW 480 batch 性能持平
- Hessian 分析证明 MERIT 收敛到更平坦的损失景观（最大特征值降低 67%，trace 降低 73%），解释了更好的泛化性能
- 计算开销极低（<1% FLOPS），因为额外操作仅为简单的 max 和逐元素运算
- QK-Norm 在大 batch 下反而有害——它过度限制了注意力层的信息流通能力，而 MERIT 只约束极端值不约束整体分布

## 亮点与洞察

- 对大批量训练失败的根因分析非常精确：不是模糊的"泛化差距"，而是精确定位到 max attention logit 爆炸 → softmax 退化为 one-hot → 注意力熵坍塌这一因果链
- 从 L2 到 max 范数的切换虽简单但理论动机清晰——attention logit 上界直接与 max norm 相关（Appendix D 证明），L2 范数与 max norm 间 99%+ 的差距解释了 LAMB 在中间层失效的原因
- 逐元素信任比率的行列分解利用了 Transformer 的结构先验（多头→行相似，outlier→列相似），计算复杂度仅为 $O(m+n)$
- 1% 额外开销 + 即插即用的特性使其在工业场景中极具吸引力

## 局限与展望

- 实验规模限于 770M 参数，未验证在 7B+ 真正大模型上的效果——大批量训练的真正需求场景在更大规模
- 收敛分析基于简化版本（$\beta_1=0, \lambda=0$），完整 MERIT 的理论保证尚未给出
- 仅评估了 GPT-2 和 Llama 架构，对 MoE、GQA 等新架构的适用性未知
- Zero-shot 评估结果提升较小（43.56→43.87），需要更充分的下游评估来验证实际价值
- 未讨论与混合精度训练（FP16/BF16）的交互作用

## 相关工作与启发

- **vs LAMB**: LAMB 的 $\ell_2$-norm 层级 trust ratio 适合 CNN（BatchNorm 和均匀权重）但不适合 Transformer 的异质权重结构。MERIT 用 max-norm + element-wise ratio 弥补了这一结构性不足
- **vs σReparam**: σReparam 用谱归一化解决 attention entropy collapse——与 MERIT 目标一致但需要修改模型架构（重训练），而 MERIT 仅修改优化器（即插即用）
- **vs QK-Norm**: LayerNorm 归一化 Q/K 能稳定注意力，但本文证明在大 batch 下反而有害——过度限制信息流，MERIT 只约束极端值不约束整体分布

## 评分

- 新颖性: ⭐⭐⭐⭐ Max attention logit 诊断 + max-norm/element-wise ratio 设计原创且实用，因果分析链条清晰
- 实验充分度: ⭐⭐⭐⭐ 三种规模 GPT-2 + Llama、多 baseline、Hessian 分析和详尽消融，但缺少 >1B 模型验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、图表丰富、从现象到方法到实验的逻辑一气呵成
- 价值: ⭐⭐⭐⭐⭐ 1% 额外开销、即插即用、已开源——对大批量训练场景有直接且显著的实用价值

<!-- RELATED:START -->

## 相关论文

- [LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](../../CVPR2026/human_understanding/laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)
- [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)
- [Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)

<!-- RELATED:END -->
