---
title: >-
  [论文解读] SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping
description: >-
  [AAAI 2026][模型压缩][低秩压缩] SkipCat 提出了一种秩最大化的低秩压缩框架，通过层内共享投影（Cat）和块跳跃（Skip）两项技术，在相同压缩率下保留更多有效秩，无需微调即可在零样本任务上比现有低秩方法提升7%准确率。
tags:
  - AAAI 2026
  - 模型压缩
  - 低秩压缩
  - SVD
  - 秩最大化
  - Block Skipping
  - 共享投影
---

# SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping

**会议**: AAAI 2026  
**arXiv**: [2512.13494](https://arxiv.org/abs/2512.13494)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 低秩压缩, SVD, 秩最大化, Block Skipping, 共享投影

## 一句话总结
SkipCat 提出了一种秩最大化的低秩压缩框架，通过层内共享投影（Cat）和块跳跃（Skip）两项技术，在相同压缩率下保留更多有效秩，无需微调即可在零样本任务上比现有低秩方法提升7%准确率。

## 研究背景与动机

### 领域现状
大语言模型在各种任务上表现出色，但巨大的参数规模使其在边缘设备上的部署面临计算和内存挑战。低秩压缩通过将权重矩阵分解为两个低秩矩阵来减少参数和计算量，是一种有前景的压缩方向。

### 现有痛点
朴素低秩压缩（如SVD分解）有一个本质限制——**必须将保留秩降低到原始秩的一半以下，才能真正获得计算和内存收益**。

具体来说，对于权重矩阵 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，低秩分解为 $B \in \mathbb{R}^{d_{out} \times r}$ 和 $A \in \mathbb{R}^{r \times d_{in}}$。只有当：
$$r < \frac{d_{in} \cdot d_{out}}{d_{in} + d_{out}}$$
时才有压缩收益。对方阵（$d_{in} = d_{out}$），这意味着 $r < R/2$。

### 核心矛盾
秩越低压缩越多，但性能下降越大；秩越高保持性能，但没有实际压缩收益。如何在相同压缩率下保留更多有效秩？

### 本文切入角度
利用模型架构的结构特性——注意力模块中Q/K/V共享输入、MLP中Gate/Up共享输入——通过共享投影矩阵和块跳跃技术，从根本上改变秩与压缩率的关系。

## 方法详解

### 整体框架
SkipCat由两个核心技术组成：
1. **Cat（层内共享低秩投影）**：共享输入的矩阵使用同一个投影矩阵
2. **Skip（块跳跃）**：跳过低秩投影中某些子块的计算

两者联合使用，在相同压缩预算下显著增加有效秩数量。

### 关键设计

#### 1. **Cat：层内共享低秩投影（Matrix Concatenation）**
- **核心思路**：注意力模块中 $W_Q, W_K, W_V$ 共享相同输入 $x$，MLP中 $W_{Gate}, W_{Up}$ 也共享输入。将共享输入的矩阵沿输出维度拼接后进行联合SVD分解
- **具体操作**：
  $$W_{QKV} = [W_Q^T, W_K^T, W_V^T]^T \in \mathbb{R}^{3d_{out} \times d_{in}}$$
  分解为 $B_{QKV} \in \mathbb{R}^{3d_{out} \times r}$ 和共享投影 $A_{QKV} = W_{S1} \in \mathbb{R}^{r \times d_{in}}$
- **收益分析**：分摊后每个矩阵的参数量为 $r(d_{in} + Cd_{out})/C$，$C$ 为拼接矩阵数。对于注意力模块 $C=3$，投影矩阵开销被分摊到三份
- **设计动机**：投影矩阵 $A$ 的作用是将输入映射到低维空间，共享输入的矩阵可以共用这个映射，减少冗余参数

#### 2. **Skip：块跳跃（Block Skipping via Schur Complement）**
- **核心思路**：将投影矩阵 $A$ 分块为 $[A_1, A_2]$，将 $A_1$ 吸收进重建矩阵 $B$，跳过 $A_1$ 的显式计算
- **关键推导**：
  $$Wx \approx BAx = B(A_1 x_1 + A_2 x_2) = BA_1(x_1 + A_1^{-1}A_2 x_2) = B'(x_1 + A'x_2)$$
  其中 $B' = BA_1$，$A' = A_1^{-1}A_2$
- **参数量**：从 $r(d_{in} + d_{out})$ 减少到 $r(d_{in} + d_{out} - r)$
- **数值稳定性问题**：$A_1$ 病态时 $A_1^{-1}$ 产生大值，导致FP16溢出
- **解决方案——列置换**：使用Strong Rank-Revealing QR分解找到使 $\tilde{A}_1$ 条件良好的列子集，应用列置换 $P$ 后重新分解。置换后激活值幅度减小近两个数量级，分布更均匀

#### 3. **SkipCat联合使用**
- 所有共享投影均配备块跳跃，二者互补
- Cat在低压缩率时作用有限但Skip效果显著；高压缩率时Cat弥补Skip的不足
- 联合后在整个压缩范围内始终处于有效压缩区域

### 训练策略
- **无训练**：无需微调即可获得强性能
- **白化预处理**：使用WikiText-2和C4混合校准集（512样本）进行权重白化
- **可选微调**：与LoRA微调兼容，20%压缩率下微调后仅损失0.39%准确率
- **量化兼容**：用Hadamard变换和通道缩放稳定8-bit量化

## 实验关键数据

### 主实验

| 模型 | 压缩率 | 方法 | WikiText2 PPL | C4 PPL | 零样本平均准确率 | 准确率下降 |
|------|--------|------|---------------|--------|----------------|-----------|
| LLaMA2-7B | 0% | Dense | 5.47 | 7.26 | 54.79% | — |
| LLaMA2-7B | 20% | ASVD | 9.06 | 11.66 | 48.81% | 5.98% |
| LLaMA2-7B | 20% | SVD-LLM | 8.82 | 13.42 | 44.84% | 9.95% |
| LLaMA2-7B | 20% | **SkipCat** | **6.29** | **8.95** | **52.59%** | **2.20%** |
| LLaMA2-7B | 30% | SVD-LLM | 11.75 | 19.37 | 41.25% | 13.54% |
| LLaMA2-7B | 30% | **SkipCat** | **7.65** | **11.57** | **48.46%** | **6.34%** |
| Qwen3-8B | 20% | SVD-LLM | 14.33 | 23.21 | 51.66% | 8.62% |
| Qwen3-8B | 20% | **SkipCat** | **11.68** | **19.09** | **56.42%** | **3.86%** |

### 消融实验

| Cat | Skip | Quant | WikiText2 PPL | C4 PPL | 说明 |
|-----|------|-------|---------------|--------|------|
| ✗ | ✗ | ✗ | 8.82 | 13.42 | 朴素SVD基线 |
| ✓ | ✗ | ✗ | 7.84 | 11.99 | Cat单独贡献 |
| ✗ | ✓ | ✗ | 6.71 | 9.32 | Skip贡献更大 |
| ✓ | ✓ | ✗ | **6.29** | **8.95** | 联合效果最佳 |
| ✓ | ✓ | ✓ | 6.29 | 8.96 | 8-bit量化无损失 |

### 微调后结果（LLaMA2-7B + LoRA）

| 压缩率 | SVD-LLM | SkipCat | 差距 |
|--------|---------|---------|------|
| 20% | 51.02% (-3.78) | **54.41%** (-0.39) | SkipCat仅损失0.39% |
| 40% | 46.91% (-7.88) | **48.65%** (-6.15) | |
| 60% | 39.50% (-15.29) | **41.16%** (-13.64) | |

### 关键发现
1. 30%压缩率下，SkipCat的PPL（7.65）甚至低于其他方法20%压缩率的结果
2. 零样本准确率提升约**7%**（30%压缩率下vs SVD-LLM）
3. Cat和Skip互补——Cat减少投影冗余，Skip减少子块计算，联合效果最佳
4. 列置换是Skip在FP16下工作的关键，无置换时BF16 PPL可达31628
5. 在更大模型（13B/14B）和不同架构（Qwen3）上一致有效

## 亮点与洞察
1. **问题定义精准**：识别出"秩必须低于一半才有压缩收益"这一本质限制，并系统解决
2. **数学优雅**：Skip技术基于Schur补的理论基础，列置换保证数值稳定性
3. **无训练强性能**：20%压缩率下仅2.2%准确率下降，这在无训练设置中极为出色
4. **与量化正交兼容**：参数级压缩+精度级量化可叠加使用

## 局限与展望
1. Skip技术需要 $A_1$ 可逆，虽然列置换缓解了这个问题，但极端情况下仍可能受限
2. Cat要求矩阵共享相同输入，对于不同架构的适用性需要case-by-case分析
3. 高压缩率（>60%）下性能下降仍然显著
4. 仅评估了语言模型，视觉/多模态模型上效果未验证
5. 执行时需要列置换预处理，部署复杂度略增

## 相关工作与启发
- ASVD通过激活分布缩放缓解离群值影响，但SVD本身的秩限制仍在
- Basis Sharing跨层共享基矩阵减少存储，但对推理计算和内存传输无帮助
- 本文的层内共享与跨层共享互补，可以组合使用
- 秩最大化思路可推广到其他低秩方法（如LoRA的rank选择）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 秩最大化视角新颖，Cat+Skip设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型、多压缩率、消融+微调+量化全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，Figure 1完美展示核心idea
- 价值: ⭐⭐⭐⭐⭐ — 实用性强，7%准确率提升在低秩压缩中是显著进步

<!-- RELATED:START -->

## 相关论文

- [DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](../../ACL2025/model_compression/denselora_dense_low-rank_adaptation_of_large_language_models.md)
- [TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](../../ACL2025/model_compression/table_lora_structure_understanding.md)
- [Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [From Low Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications](../../ICML2025/model_compression/from_low_rank_gradient_subspace_stabilization_to_low-rank_weights_observations_t.md)
- [C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](../../NeurIPS2025/model_compression/c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)

<!-- RELATED:END -->
