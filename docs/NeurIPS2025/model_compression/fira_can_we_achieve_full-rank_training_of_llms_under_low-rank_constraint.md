---
title: >-
  [论文解读] FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?
description: >-
  [NeurIPS 2025][模型压缩][low-rank training] 提出 Fira，首个在低秩约束下实现全秩训练（全秩梯度+全秩权重）的 LLM 训练框架，通过观察到低秩与全秩训练中优化器的缩放因子高度相似，用低秩缩放因子近似校正子空间外梯度，配合 norm-growth limiter 防止 loss spike，在预训练和微调中均超越 LoRA 和 GaLore。
tags:
  - NeurIPS 2025
  - 模型压缩
  - low-rank training
  - memory-efficient
  - full-rank gradient
  - Adam optimizer
  - gradient projection
---

# FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?

**会议**: NeurIPS 2025  
**arXiv**: [2410.01623](https://arxiv.org/abs/2410.01623)  
**代码**: 有 (github.com/xichen-fy/Fira)  
**领域**: 模型压缩 / LLM 效率  
**关键词**: low-rank training, memory-efficient, full-rank gradient, Adam optimizer, gradient projection

## 一句话总结

提出 Fira，首个在低秩约束下实现全秩训练（全秩梯度+全秩权重）的 LLM 训练框架，通过观察到低秩与全秩训练中优化器的缩放因子高度相似，用低秩缩放因子近似校正子空间外梯度，配合 norm-growth limiter 防止 loss spike，在预训练和微调中均超越 LoRA 和 GaLore。

## 研究背景与动机

LLM 训练的内存瓶颈主要在优化器状态：LLaMA-7B 训练需 58GB 显存，其中 Adam 优化器状态占 28GB（比参数本身还大）。低秩训练是减少显存的有效方案，但现有方法都受限于低秩子空间：

**LoRA**：将权重分解为低秩矩阵 $W = W_0 + BA$，训练限制在权重的低秩子空间，表示能力受损

**GaLore**：通过 SVD 将梯度投影到低秩子空间 $R_t = P_t^\top G_t$，虽然训练全秩权重但**丢弃了子空间外的梯度信息**

**ReLoRA**：试图用多次低秩更新近似全秩，但仍需全秩 warmup，无法实现完全的低内存训练

核心矛盾：低秩约束节省内存 ↔ 全秩训练保证性能。能否两者兼得？

关键困难：子空间外的梯度 $(G_t - P_t R_t)$ 没有对应的优化器状态来做 Adam 校正。直接将其加回（GaLore-add）等价于对这部分做 SGD，效果极差且引入梯度不一致。

## 方法详解

### 整体框架

Fira 的核心思想是将全秩梯度分为两部分，分别处理：

$$W_{t+1} = W_t - \eta P_t \psi_t(R_t) - \eta \phi_t(R_t)(G_t - P_t R_t)$$

- **$P_t \psi_t(R_t)$**：子空间内梯度，由低秩 Adam 正常校正
- **$\phi_t(R_t)(G_t - P_t R_t)$**：子空间外梯度，用 norm-based scaling 近似校正
- **$\phi_t(R_t)$**：缩放因子，从低秩优化器状态计算

### 关键设计

#### 1. 核心观察：缩放因子的相似性

定义缩放因子为 Adam 对梯度范数的校正倍率：

$$\phi_t(R_t) = \frac{\|\psi_t(R_t)\|}{\|R_t\|}$$

**关键发现**：在 LLM 训练中，低秩训练与全秩训练的缩放因子在矩阵级别高度相似：

| 模型大小 | 矩阵级 Cosine Sim | 矩阵级 MSE | 列级 Cosine Sim | 列级 MSE |
|---------|-------------------|-----------|----------------|---------|
| 60M | 0.9922 | 3e-04 | 0.9273 | 3e-05 |
| 130M | 0.9901 | 2e-04 | 0.9046 | 2e-05 |
| 350M | 0.9893 | 1e-04 | 0.9174 | 1e-05 |
| 1B | 0.9795 | 2e-04 | 0.9229 | 1e-05 |

余弦相似度 >0.97，MSE 极小。这意味着低秩优化器的缩放效果可以近似全秩优化器的缩放效果。

#### 2. Norm-Based Scaling

**矩阵级缩放**（Fira-matrix）：用低秩梯度的缩放因子统一缩放子空间外梯度的整个矩阵。

**列级缩放**（Fira，更精细）：对权重矩阵的每一列独立计算缩放因子：

$$\phi_t(R_t)_i = \frac{\|\psi(R_{t,:,i})\|}{\|R_{t,:,i}\|}, \quad i=1,2,\dots,n$$

列级也显示强相似性（Cosine Sim >0.90），能提供更精确的近似校正。

#### 3. Norm-Growth Limiter

**问题**：低秩优化器的不稳定性 + 投影矩阵切换 → 训练初期梯度突增 → loss spike

**原因分析**：
- 每 T 步切换投影矩阵 $P_t$，旧优化器状态与新投影不匹配
- 子空间外梯度保留了原始方向但缺乏 Adam 的梯度稳定化效果

**解决方案**：限制梯度范数的相对增长率：

$$\text{if } \frac{\|S_t\|}{\|S_{t-1}\|} > \gamma \text{ then } S_t \leftarrow \frac{S_t}{\|S_t\|} \cdot \gamma \|S_{t-1}\|$$

其中 $\gamma = 1.01$（所有实验通用，对选择不敏感）。这将突增转化为渐进增长，比绝对梯度裁剪更灵活（后者不考虑不同矩阵间的梯度量级差异）。

### 实现细节

- 相比 GaLore，Fira 仅额外存储每个权重矩阵一个标量 $\|S_{t-1}\|$，内存开销可忽略
- 仅需 3 行额外代码，即插即用
- 超参仅多一个 $\gamma$，固定 1.01 即可

## 实验关键数据

### 主实验：LLaMA 预训练（C4 数据集，验证困惑度↓）

| 方法 | 60M | 130M | 350M | 1B |
|------|-----|------|------|-----|
| Full-Rank | 34.06 (0.48G) | 25.08 (1.01G) | 18.80 (2.74G) | 15.56 (10.40G) |
| **Fira** | **31.06 (0.36G)** | **22.73 (0.77G)** | **16.85 (1.90G)** | **14.31 (6.98G)** |
| GaLore | 34.88 (0.36G) | 25.36 (0.77G) | 18.95 (1.90G) | 15.64 (6.98G) |
| LoRA | 34.99 (0.44G) | 33.92 (0.99G) | 25.58 (2.12G) | 19.21 (7.36G) |
| ReLoRA | 37.04 (0.44G) | 29.37 (0.99G) | 29.08 (2.12G) | 18.33 (7.36G) |

Fira 在**所有规模上大幅超越** GaLore/LoRA/ReLoRA，甚至超越全秩训练（31.06 vs 34.06）！在相同内存约束下性能最优。

### LLaMA 7B 预训练

使用 **8× 更小的 rank**（即优化器状态内存仅为 GaLore 的 1/8），Fira 仍然显著优于 GaLore。这验证了 Fira 在大规模场景下的有效性。

### 微调实验（LLaMA-7B，常识推理 8 任务）

| 方法 | 内存 | BoolQ | HellaSwag | WinoGrande | 平均 |
|------|------|-------|-----------|------------|------|
| **Fira** | 14.44G | 69.4 | **76.8** | **81.2** | **76.9** |
| GaLore | 14.44G | 69.5 | 32.2 | 18.0 | 62.7 |
| LoRA | 14.53G | 68.9 | 78.1 | 78.8 | 74.7 |
| Full-rank | 56.00G | 64.2 | 42.3 | 66.5 | 58.6 |

GaLore 在 HellaSwag/WinoGrande 上严重失败，Fira 在 8 任务中 5 个最优，平均 76.9 最高。

### 消融实验（LLaMA 60M 预训练）

| 变体 | 困惑度↓ |
|------|---------|
| **Fira（完整）** | **31.06** |
| Fira-matrix（矩阵级缩放） | 31.52 |
| Fira-w.o.-limiter（无 limiter） | 32.22 |
| Fira-gradient-clipping（用梯度裁剪替代） | 31.22 |
| Fira-gradient-shrink | 33.98 |
| Fira-tensor-wise-scaling | 33.81 |
| Fira-w.o.-scaling（无缩放，等效 GaLore-add） | 37.06 |

关键结论：
- 无缩放（37.06）远差于有缩放（31.06），验证了 norm-based scaling 的必要性
- 列级优于矩阵级（31.06 vs 31.52）
- Norm-growth limiter 优于所有替代稳定方案

### 不同 Rank 下的性能

| Rank | Fira | GaLore | 差距 |
|------|------|--------|------|
| 4 | ~35 | ~48 | 巨大优势 |
| 16 | ~32 | ~37 | 显著优势 |
| 64 | ~31 | ~33 | 明显优势 |
| 128 | ~31 | ~32 | 仍有优势 |

Fira 在极低 rank 下仍接近全秩性能，而 GaLore 急剧退化。这说明 Fira 有效利用了子空间外信息。

### 关键发现

1. 低秩与全秩训练的 Adam 缩放因子确实相似——这是一个跨规模（60M→1B）稳定成立的现象
2. Fira 甚至超越全秩训练，可能因为 norm-based scaling 引入的随机性有助于逃离局部最优
3. 投影矩阵切换是低秩训练 loss spike 的主要来源，norm-growth limiter 有效解决
4. 越低的 rank，Fira 相对 GaLore 的优势越大

## 亮点与洞察

- **理论洞察新颖**："低秩与全秩的缩放因子相似"是一个有趣且实用的发现，为全秩训练提供了低秩近似的理论基础
- **超越全秩训练**：罕见地在低秩约束下性能超越无约束全秩训练
- **极简实现**：仅 3 行额外代码，即插即用，不修改模型架构
- **全面验证**：从 60M 到 7B，预训练到微调，全面验证了有效性

## 局限与展望

1. **SVD 计算开销**：继承自 GaLore，需每 T 步做 SVD（虽然 <10% 开销但非零）
2. **缩放因子相似性的理论解释不足**：论文观察到了现象但缺乏深入理论分析
3. **仅验证了 Adam 优化器**：对 AdaFactor、Lion 等其他优化器的适用性未知
4. **7B 预训练未报最终困惑度**：仅展示了 loss 曲线对比
5. 可扩展方向：与量化训练（QLoRA）结合、探索自适应 rank 调度、将方法应用到 diffusion model 训练

## 相关工作与启发

- **GaLore**：低秩梯度投影的先驱，Fira 在其基础上恢复子空间外梯度
- **LoRA/ReLoRA**：参数低秩分解路线，Fira 证明了梯度投影路线的上限更高
- **Flora**：随机投影方法，在微调中表现一般
- 启发：优化器的缩放效应具有跨子空间的稳定性，这暗示 Adam 的自适应机制主要作用于梯度的全局统计特性而非逐元素细节

## 评分

- 新颖性：★★★★★（首次实现低秩约束下的全秩训练，核心观察新颖）
- 技术深度：★★★★☆（缩放因子分析深入，但理论证明可更严格）
- 实验充分度：★★★★★（60M→7B、预训练+微调、详尽消融和 rank 分析）
- 实用价值：★★★★★（即插即用、3行代码、大幅提升低秩训练性能，工程落地容易）

<!-- RELATED:START -->

## 相关论文

- [BeamLoRA: Beam-Constraint Low-Rank Adaptation](../../ACL2025/model_compression/beamlora_beam_constraint_lora.md)
- [Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)

<!-- RELATED:END -->
