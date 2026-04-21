---
title: >-
  [论文解读] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning
description: >-
  [NEURIPS2025][图像生成][LoRA] 提出 GraLoRA——将 LoRA 的权重更新矩阵分割为 $k^2$ 个独立子块、每块配独立低秩适配器，在不增加参数量和计算量的前提下将有效秩从 $r$ 提升至 $kr$，解决 LoRA 在高秩下因梯度纠缠导致的性能退化问题，在代码生成上 Pass@1 最高提升 +8.5%。
tags:
  - NEURIPS2025
  - 图像生成
  - LoRA
  - 低秩适配
  - 参数高效微调
  - 梯度纠缠
  - 分块分解
  - 高秩表达
---

# GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning

**会议**: NEURIPS2025  
**arXiv**: [2505.20355](https://arxiv.org/abs/2505.20355)  
**作者**: Yeonjoon Jung, Daehyun Ahn, Hyungjun Kim, Taesu Kim, Eunhyeok Park (SqueezeBits, POSTECH)
**代码**: 未开源  
**领域**: 参数高效微调 (PEFT) / LoRA 改进  
**关键词**: LoRA, 低秩适配, 参数高效微调, 梯度纠缠, 分块分解, 高秩表达

## 一句话总结

提出 GraLoRA——将 LoRA 的权重更新矩阵分割为 $k^2$ 个独立子块、每块配独立低秩适配器，在不增加参数量和计算量的前提下将有效秩从 $r$ 提升至 $kr$，解决 LoRA 在高秩下因梯度纠缠导致的性能退化问题，在代码生成上 Pass@1 最高提升 +8.5%。

## 研究背景与动机

- **核心痛点**: LoRA 在 rank 32–64 时效果最佳，进一步增大 rank 反而性能下降甚至不如低 rank，与"更多参数应更好"的直觉矛盾
- **已有改进的局限**: OLoRA、PiSSA 改进初始化，MoRA/RaSA 改变结构，但都未从根本上解决高 rank 退化的根源
- **关键观察**: 作者发现 LLaMA3.1-8B 的 Layer 1 down-projection 存在严重的通道激活值不均衡（outlier channel），少数异常通道的梯度会主导整个低秩适配器的更新方向

## 核心问题：LoRA 的梯度纠缠

### FFT vs LoRA 的梯度传播差异

在全参微调 (FFT) 中，outlier 通道的影响是局部的——只影响权重矩阵 $W$ 中与该通道直接交互的那一列。但 LoRA 的低秩约束导致：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y} X^\top A$$

矩阵 $A$ 将所有输入通道的信息混合在一起，因此 outlier 通道的巨大激活值会污染整个 $\partial L / \partial B$，使得与 outlier 无关的通道也被不当地拉偏。

### 高秩加剧问题

当 rank 增大时，$A \in \mathbb{R}^{N \times r}$ 混合更多通道信息，梯度偏移更严重。作者通过实验验证：LoRA 在 rank 128 时的梯度与 FFT 的偏差远大于 rank 32。

## 方法详解：GraLoRA

### 1. 分块低秩分解

将权重更新矩阵 $R \in \mathbb{R}^{M \times N}$ 分割为 $k \times k$ 的网格，每个子块 $(i,j)$ 配独立的低秩适配器对：

$$R_{\text{GraLoRA}} = \begin{bmatrix} B_{1,1}A_{1,1}^\top & \cdots & B_{1,k}A_{1,k}^\top \\ \vdots & \ddots & \vdots \\ B_{k,1}A_{k,1}^\top & \cdots & B_{k,k}A_{k,k}^\top \end{bmatrix}$$

其中 $A_{i,j} \in \mathbb{R}^{N/k \times r/k}$，$B_{i,j} \in \mathbb{R}^{M/k \times r/k}$。

### 2. 表达力分析

将 GraLoRA 重写为稀疏矩阵乘积 $R = B_{\text{GraLoRA}} A_{\text{GraLoRA}}^\top$，利用 Sylvester 秩不等式可证明：

- LoRA 有效秩 = $r$
- GraLoRA 有效秩 = $kr$（$k$ 倍提升）

这意味着用相同的参数量，GraLoRA 能表达更高秩的权重更新。

### 3. 梯度局部化

GraLoRA 天然将 outlier 的影响限制在与其交互的 $k$ 个适配器块中，其余 $k^2 - k$ 个块不受干扰。这种梯度传播模式与 FFT 的行为高度一致，有效避免了全局梯度扭曲。

### 4. 开销分析

| 维度 | 对比 LoRA |
|------|----------|
| **参数量** | 完全相同：$N \times r + M \times r$ |
| **FLOPs** | 精确计算显示 GraLoRA 实际 FLOPs 更少：$\text{GraLoRA} = \text{LoRA} - (k-1)rT$ |
| **训练内存** | 中间表征 $A^\top X$ 扩大 $k$ 倍，但 $r \ll M, N$，影响微小；使用 gradient checkpointing 后几乎无差异 |
| **推理** | 可合并回原始权重，零额外开销 |

### 5. $k$ 的选择策略

经验规则：保持每个子块的最小表达力 $r/k^2 \approx 8$

- rank 16/32 → $k = 2$
- rank 64/128 → $k = 4$

### 6. Hybrid GraLoRA

低秩场景下（rank ≤ 16），纯 GraLoRA 的每个子块 rank 太小，表达力不足。解决思路：将部分 rank 分配给标准 LoRA，与 GraLoRA 拼接使用。经验上，将不超过 1/2 的 rank 分配给 LoRA 组件效果最佳。

## 实验关键数据

### 代码生成 (HumanEval+, LLaMA3.1-8B)

| Rank | 方法 | Pass@1 | Pass@5 | Pass@10 |
|------|------|--------|--------|---------|
| 64 | LoRA | 58.1% | 66.4% | 68.5% |
| 64 | **GraLoRA** | **60.5%** | **71.2%** | **72.6%** |
| 128 | LoRA | 55.8% | 64.8% | 68.6% |
| 128 | **GraLoRA** | **64.3%** | **71.7%** | **73.7%** |

Rank 128 时 LoRA 性能严重退化（反不如 rank 32），而 GraLoRA 持续提升，差距达 **+8.5% Pass@1**。

### 常识推理 (8 任务平均)

| 模型 | LoRA | GraLoRA | 提升 |
|------|------|---------|------|
| Qwen2.5-1.5B | 78.7% | **79.8%** | +1.1% |
| Qwen2.5-7B | 85.6% | **86.4%** | +0.8% |
| LLaMA3.2-3B | 81.3% | **84.6%** | +3.3% |
| LLaMA3.1-70B | 91.3% | **92.4%** | +1.1% |

GraLoRA 在 32 个子任务中赢下 26 个。

### 数学推理 (MATH, Qwen2.5-1.5B)

| Rank | LoRA | GraLoRA |
|------|------|---------|
| 64 | 23.6% | **25.7%** |
| 128 | 24.7% | **28.9%** (+4.2%) |

### GLUE (RoBERTa-base)

Best GraLoRA 平均 86.0%，超越 LoRA (84.2%)、VeRA (85.2%)、FourierFT (85.0%)。

### 图像生成 (SDXL 微调)

| 指标 | LoRA | GraLoRA |
|------|------|---------|
| CLIP Similarity | 91.4% | **91.9%** |
| DINOv2 Similarity | 79.2% | **81.3%** |

## 亮点

1. **问题定位精准**: 从梯度动力学角度清晰解释了 LoRA 高秩退化的根因——outlier channel 导致的梯度纠缠，这是之前被忽视的分析视角
2. **方法极简优雅**: 只是把矩阵分块，不改变参数量/计算量/推理流程，却获得 $k$ 倍有效秩提升，属于"一行代码就能实现"的改进
3. **理论与实验一致**: 用 Sylvester 秩不等式严格证明秩提升，梯度分布实验直观展示了 GraLoRA 的梯度修复效果
4. **覆盖面广**: 5 个任务（代码/常识/数学/NLU/图像）× 多种模型架构（LLaMA/Qwen/RoBERTa/SDXL）× 多种 rank，实验非常充分
5. **训练时间增加极少**: 相对 LoRA 仅增加 3%–10% 训练时间，远小于 MoRA 的 40%+

## 局限与展望

1. **均匀分块假设**: 当前设计假设所有子块等大小，未考虑不同通道重要性差异，自适应分块可能更优
2. **$k$ 需要手动选择**: 虽然给了经验规则 $r/k^2 \approx 8$，但不同任务/模型的最优 $k$ 仍需 sweep
3. **分类归属存疑**: 论文被放在 image_generation 分类下，但核心贡献是通用 PEFT 方法，图像生成实验只是验证之一
4. **缺少与 DoRA、LoRA+ 等新方法的全面对比**: 常识推理任务有对比，但代码生成和数学推理中缺少
5. **未探索自适应 rank 分配**: 不同层可能需要不同的 $k$ 值，论文中对所有层使用相同 $k$

## 与相关工作的对比

| 方法 | 核心思路 | vs GraLoRA |
|------|----------|------------|
| **LoRA** | 固定低秩分解 | 高 rank 退化，梯度纠缠 |
| **MoRA** | 用方阵替代低秩矩阵 | 训练时间增加 40%+，且性能不稳定 |
| **RaSA** | 跨层共享部分低秩分量 | 在代码生成上不如 GraLoRA，高 rank 同样受限 |
| **DoRA** | 解耦方向与幅度更新 | 常识推理上 GraLoRA 领先 2.9% |
| **VeRA** | 用共享随机矩阵极致压缩参数 | GLUE 上 GraLoRA 领先 0.8% |
| **PiSSA** | SVD 初始化 LoRA | 未解决高 rank 退化根因 |

## 启发与关联

1. **与量化的联系**: outlier channel 问题在量化领域早有研究（SmoothQuant、OWQ），但本文首次系统分析其对 LoRA 训练的影响，提示 PEFT 和量化可以共享 outlier 处理策略
2. **分块思想的普适性**: "大矩阵→小块独立处理"的思想可推广到其他低秩方法（如低秩注意力、低秩 MoE），是一种通用的表达力增强范式
3. **对实际部署的启示**: 很多实践者发现 LoRA rank 调大反而掉点却不知原因，GraLoRA 给出了理论解释和简单解法
4. **可与 LoRA 合并推理**: 零额外推理开销这点对实际部署极其友好

## 评分

- 新颖性: ⭐⭐⭐⭐ (问题分析新颖，方法本身是已有思想的精准应用)
- 实验充分度: ⭐⭐⭐⭐⭐ (5 任务、多模型、多 rank、消融实验齐全)
- 写作质量: ⭐⭐⭐⭐ (分析清晰，图表丰富)
- 价值: ⭐⭐⭐⭐⭐ (现成可用，改动最小，收益明确)

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](../../ICML2025/image_generation/zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)

<!-- RELATED:END -->
