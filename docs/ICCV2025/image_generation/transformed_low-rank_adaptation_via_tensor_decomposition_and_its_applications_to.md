---
title: >-
  [论文解读] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models
description: >-
  [ICCV 2025][图像生成][参数高效微调] 提出 TLoRA 方法，将预训练权重的微调分解为 **变换（Transform）** 和 **残差（Residual）** 两个适应部分，分别采用张量环矩阵（TRM）和张量环（TR）分解进行参数化，在 SDXL 上实现了仅 0.4M 参数的超参数高效微调…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "参数高效微调"
  - "LoRA"
  - "张量分解"
  - "文本到图像"
  - "主题驱动生成"
---

# Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models

**会议**: ICCV 2025  
**arXiv**: [2501.08727](https://arxiv.org/abs/2501.08727)  
**代码**: [https://github.com/taozerui/tlora_diffusion](https://github.com/taozerui/tlora_diffusion)  
**领域**: 图像生成  
**关键词**: 参数高效微调, LoRA, 张量分解, 文本到图像, 主题驱动生成

## 一句话总结

提出 TLoRA 方法，将预训练权重的微调分解为 **变换（Transform）** 和 **残差（Residual）** 两个适应部分，分别采用张量环矩阵（TRM）和张量环（TR）分解进行参数化，在 SDXL 上实现了仅 0.4M 参数的超参数高效微调，同时性能优于 LoRA 等基线方法。

## 研究背景与动机

参数高效微调（PEFT）是大规模文本到图像模型个性化的核心技术。LoRA 及其变体因简洁高效而被广泛采用，其核心假设是微调权重更新 $\Delta = W_* - W_0$ 具有低秩结构。然而，现实中大规模基础模型的微调权重通常不满足低秩假设，这导致 LoRA 在困难任务上存在显著的近似误差。

**现有方法的痛点**：
- **LoRA**：低秩假设在实际微调中可能不成立，近似误差大
- **OFT（正交微调）**：使用块对角矩阵参数化，极度稀疏，限制了神经元间的信息传递
- **DoRA**：虽然引入了幅度-方向分解（隐式对角变换），但对角矩阵的表达能力有限
- **FouRA 等固定变换方法**：使用预定义的固定变换（如 DFT），无法自适应不同模型和任务

**核心洞察**：作者观察到，如果先对预训练权重 $W_0$ 施加一个可学习的线性变换 $T$，使其尽可能对齐目标权重 $W_*$，则残差 $\Delta'_* = W_* - W_0 T$ 的秩会显著降低。这意味着可以用更紧凑的结构来近似残差部分，从而同时实现更高的参数效率和更好的性能。

**切入角度**：提出"变换+残差"的统一适应框架 $y' = (W_0 T + \Delta) x$，并利用张量分解技术（TRM 和 TR）来高效参数化这两个部分。

## 方法详解

### 整体框架

TLoRA 的核心公式为：

$$y' = (W_0 T + \Delta) x$$

其中 $T$ 是变换适应部分（用 TRM 参数化），$\Delta$ 是残差适应部分（用 TR 参数化）。两者协同工作：$T$ 负责将预训练权重旋转/投影到接近目标权重的空间，$\Delta$ 负责补偿剩余的低秩差异。

### 关键设计

1. **张量环矩阵（TRM）变换适应**：

    - 功能：参数化变换矩阵 $T$，使其成为全秩、稠密但参数高效的结构
    - 核心思路：将 $I \times I$ 的矩阵 $T$ 张量化为 $D$ 个 4 阶核心张量 $\mathcal{A}^d \in \mathbb{R}^{I_d \times I_d \times R \times R}$，矩阵元素通过 $T[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = \text{tr}(\mathcal{A}^1[i_1,j_1,:,:] \cdots \mathcal{A}^D[i_D,j_D,:,:])$ 计算
    - 空间复杂度：$\mathcal{O}(D I^{2/D} R^2)$，远小于原始的 $\mathcal{O}(I^2)$
    - 设计动机：TRM 天然能表示全秩稠密矩阵（当核心张量稠密且满秩时），解决 OFT 块对角矩阵稀疏性的问题。实验验证（Fig. 3）表明 TRM 在相同参数量下的近似误差显著优于 BOFT

2. **张量环（TR）残差适应**：

    - 功能：用极致紧凑的结构参数化残差 $\Delta$
    - 核心思路：将 $\Delta$ 分解为 $2D$ 个 3 阶核心张量 $\mathcal{B}^d \in \mathbb{R}^{I_d \times R \times R}$ 和 $\mathcal{C}^d \in \mathbb{R}^{J_d \times R \times R}$
    - 空间复杂度：$\mathcal{O}(D I^{1/D} R^2)$，比 TRM 更紧凑
    - 设计动机：在变换的帮助下，残差部分的秩被有效降低，因此可以使用更紧凑的 TR 结构。模拟实验（Fig. 2c）显示 TRM+TR 组合在极低参数量下（< LoRA R=1 的 10%）就能达到可比较的近似误差

3. **零初始化策略**：

    - 功能：确保训练开始时微调模型与原始模型一致
    - 核心思路：TRM 部分初始化为单位矩阵（Proposition 1：令每个核心张量切片 $\mathcal{A}^d[:,:,r,r'] = I_{I_d}/R$）；TR 部分的 $\mathcal{B}^1$ 初始化为零张量，其余核心张量用 $\mu P$ 框架的高斯初始化（$\sigma = \Theta(\sqrt{n\_out}/n\_in)$）
    - 设计动机：与 LoRA 的零初始化一致，避免破坏预训练模型的信息。先前 TR-PEFT 工作使用随机高斯初始化所有因子，导致训练不稳定

### 损失函数 / 训练策略

- 变换部分的约束正则化：可选择单位矩阵正则化 $\mathcal{R}_I$ 或正交正则化 $\mathcal{R}_O$，且均可在小尺寸核心张量上高效计算（利用 Proposition 2），无需显式构造全尺寸矩阵
- 实践中发现单位矩阵正则化效果优于正交正则化，与 OFT/BOFT 文献的结论一致
- 训练使用 AdamW 优化器，学习率从 {5e-4, 1e-4, 5e-5} 中调优

## 实验关键数据

### 主实验（可控生成，SD v1.5 + ControlNet）

| 方法 | 参数量(M) | L2I Error↓ | S2I mIoU↑ | S2I aACC↑ | C2I IoU↑ | C2I F1↑ |
|------|----------|-----------|----------|----------|---------|--------|
| LoRA r=4 | 0.80 | 5.32 | 27.72 | 64.99 | 0.180 | 0.305 |
| DoRA r=4 | 0.90 | 6.43 | 27.11 | 65.70 | 0.143 | 0.248 |
| OFT r=32 | 1.50 | 7.35 | 28.52 | 66.04 | 0.165 | 0.281 |
| BOFT(2,2) | 2.41 | 7.64 | 28.45 | 66.19 | 0.161 | 0.276 |
| BOFT(4,8) | 20.76 | 5.67 | 28.83 | **67.74** | — | — |
| **TLoRA*(2,4)** | **0.94** | **5.32** | **29.23** | 69.21 | **0.203** | **0.337** |
| TLoRA(2,6) | 0.40 | 5.84 | 27.03 | 65.15 | 0.184 | 0.310 |

### 消融实验（模拟实验 - SDXL Inpaint 权重近似误差）

| 配置 | 近似误差趋势 | 说明 |
|------|------------|------|
| LoRA | 低秩假设下好，全微调权重下误差大 | 只在假设满足时有效 |
| OFT + LoRA | 与 LoRA 相当，无改善 | 正交变换不改变列空间 |
| TRM + LoRA | 显著低于 LoRA | TRM 变换有效对齐权重 |
| TRM + TR | 极低参数量下最优 | TR 比 LoRA 在极小预算下更灵活 |
| TR (无变换) | 极低参数量优于 LoRA | 但参数增加后改善不显著 |

### 关键发现
- TLoRA 在仅 0.4M 参数（LoRA R=1 的 27%）下仍能取得有竞争力的结果
- 变换部分（TRM）几乎在所有情况下都能提升性能，尤其当目标权重偏离低秩假设时
- 使用零初始化的 TR 显著优于随机初始化的 TR（LoRETTA 在可控生成中 Error 高达 118.14）
- 张量分解方法（LoRETTA、TLoRA）在主题驱动生成中具有更好的采样多样性

## 亮点与洞察
- 提出"变换+残差"的统一框架，将 DoRA、OFT、FouRA 等方法都纳入该框架下理解（DoRA ≈ 对角变换 + LoRA，FouRA ≈ 固定 DFT 变换 + LoRA）
- 利用已有预训练和微调模型的权重差异进行模拟研究（Fig. 2），提供了直观的经验验证
- TRM 作为全秩稠密矩阵的紧凑参数化方式，在表达能力和参数效率间取得了极好的平衡
- 初始化策略对高阶张量结构至关重要，零初始化 + μP 框架使训练更稳定

## 局限与展望
- 目前仅在 SD v1.5 和 SDXL 上验证，未在 DiT 架构（如 Flux）上测试
- 需要手动选择张量化方式（子索引维度划分），超参数选择对结果有影响
- 变换和残差的秩选择缺乏理论指导
- 推理时需要额外计算 TRM 矩阵乘法（虽然可以预计算并合并到权重中）

## 相关工作与启发
- LoRA → DoRA → TLoRA 的演进路径：从低秩假设 → 隐式稀疏变换 → 显式稠密变换
- 张量分解在模型压缩领域有深厚基础（TT、TR），将其引入 PEFT 是有价值的方向
- "找到更好的投影空间，使残差更低秩" 的思路可以推广到其他模型适应场景

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICML 2026\] Bayesian Tensor Decomposition with Diffusion Model Prior](../../ICML2026/image_generation/bayesian_tensor_decomposition_with_diffusion_model_prior.md)

</div>

<!-- RELATED:END -->
