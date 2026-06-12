---
title: >-
  [论文解读] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation
description: >-
  [CVPR 2026][模型压缩][RMSNorm] GPU矩阵乘法吞吐量提升(80x)远超reduction/elementwise操作(5-9x)，RMSNorm正成为低精度训练的新瓶颈。MXNorm直接复用MXFP8量化时已计算的block scales来估计RMS，实现32倍reduction大小缩减…
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "RMSNorm"
  - "MXFP8"
  - "block scales"
  - "normalization"
  - "low-precision training"
  - "LLM"
  - "Llama"
  - "量化"
  - "kernel optimization"
---

# MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation

**会议**: CVPR 2026  
**arXiv**: [2603.13180](https://arxiv.org/abs/2603.13180)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: RMSNorm, MXFP8, block scales, normalization, low-precision training, LLM, Llama, tensor quantization, kernel optimization

## 一句话总结

GPU矩阵乘法吞吐量提升(80x)远超reduction/elementwise操作(5-9x)，RMSNorm正成为低精度训练的新瓶颈。MXNorm直接复用MXFP8量化时已计算的block scales来估计RMS，实现32倍reduction大小缩减。理论上证明block absmax的广义p-mean可收敛到RMS的常数倍。Llama 3 125M/1B/8B预训练验证MXNorm(p=2)与RMSNorm训练精度差异minimal，torch.compile实测isolated kernel最高2.4x加速、Llama 3 8B transformer layer在MXFP8下+1.3%、NVFP4下+2.6%加速。Drop-in replacement，无额外超参数。

## 研究背景与动机

过去8年GPU AI加速器的性能提升极不均衡：

| GPU | 低精度MatMul (TFLOPS) | CUDA Core (TFLOPS) | 内存带宽 (TB/s) |
|-----|----------------------|---------------------|-----------------|
| V100 | 125 (FP16) | 15.7 | 0.9 |
| A100 | 312 [2.5x] | 19.5 [1.2x] | 2.0 [2.2x] |
| H100 | 1979 [15.8x] | 67.0 [4.3x] | 3.4 [3.7x] |
| GB200 | 10000 [**80x**] | 80.0 [**5.1x**] | 8.0 [**8.9x**] |
| Rubin* | 35000 [280x] | 130.0 [8.3x] | 22.0 [24.4x] |

矩阵乘法吞吐量8年提升80倍，而reduction/elementwise操作（受限于CUDA core和内存带宽）仅提升5-9倍。**这个差距还在加速扩大**（Rubin代预计280x vs 8.3x/24.4x）。

在Llama等前沿LLM中，RMSNorm被广泛使用（Pre-Norm架构），每个attention和FFN层前都有一个RMSNorm。RMSNorm的核心操作是沿hidden dimension做reduction来计算root mean square——这正是reduction操作，无法被矩阵乘法加速器加速。当矩阵乘法越来越快，RMSNorm等非MatMul操作就成为新瓶颈。

**关键观察**：在Pre-Norm transformer中，RMSNorm紧接在MXFP量化之前。两者都需要沿hidden dimension收集统计信息来rescale元素——RMSNorm计算RMS，MXFP计算每个block的absmax。**能否复用MXFP已算好的block scales来近似RMS，避免重复reduction？**

## 核心问题

如何将RMSNorm的reduction操作与MXFP8量化的block scale计算融合，用已有的block-level统计量近似全局RMS，在几乎不损失训练精度的前提下减少normalisation开销？

## 方法详解

### 整体框架

MXNorm 的出发点是一个"免费午餐"观察：在 Pre-Norm transformer 里，RMSNorm 紧挨在 MXFP 量化前面，两者都要沿 hidden dimension 收集统计量来 rescale——RMSNorm 算 RMS，MXFP 算每个 block 的 absmax。既然 MXFP 已经把 block absmax 算好了，何不直接拿它来近似 RMS，省掉 RMSNorm 那次独立的 reduction？给定 tensor $X \in \mathbb{R}^{T \times D}$，RMSNorm 每行要算 inverse RMS $\rho_t = \left(\frac{1}{D}\sum_{d=1}^{D} X_{td}^2\right)^{-1/2}$ 再乘 gain：$Y_{td} = \rho_t \cdot X_{td} \cdot \gamma_d$，瓶颈就是对 $D$（通常 4096-8192）个元素做 reduction；而 MXFP8 量化（MXCast）把 $D$ 列切成 $K=D/B$ 个大小 $B=32$ 的 block，每 block 算 absmax $m_{tk} = \max_b |Y_{tkb}|$、取 power-of-2 scale $S_{tk} = \text{cast}(m_{tk}/256; E8M0)$ 再量化 $V_{tkb} = \text{cast}(Y_{tkb}/S_{tk}; E4M3)$。MXNorm 就是把这两次 reduction 合成一次：先复用 block absmax 的广义 p-mean 估出 inverse RMS，指数取 $p=2$ 约束 output 上界以保住训练稳定，再把 gain $\gamma$ 融进下一层 Linear 权重、让归一化输出直接以 MXFP 格式进入下游矩阵乘。下图给出从输入激活到下游 MatMul 的完整数据流，核心是 block absmax 这一次计算同时喂给 RMS 估计与量化两条支路。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入 X（T×D），Pre-Norm 层激活"] --> B["切 K=D/B 个 block，算 block absmax<br/>reduction 规模从 D 降到 K（B=32 时省 32 倍）"]
    B -->|复用同一组 absmax| C["广义 p-mean 估 inverse RMS<br/>ρ̃ = c̃(p,B)·(mean of absmaxᵖ)^(−1/p)，取 p=2"]
    B -->|复用同一组 absmax| D["MXFP 量化：算 block scales + 量化值"]
    C --> E["MXNormLinear：gain γ 融进下一层权重<br/>H = MXNorm(X)·MXCast(W·γ)ᵀ"]
    D --> E
    E --> F["输出 H（MXFP 格式）→ 下游 MatMul"]
```

### 关键设计

**1. 用 block absmax 的广义 p-mean 估 RMS：把两次 reduction 合成一次**

RMSNorm 沿整个 hidden dimension 求平方和，是被矩阵乘法加速器越甩越远的非 MatMul 瓶颈。MXNorm 改用 MXFP 已算好的 $K$ 个 block absmax 来估 inverse RMS。理论支撑是定理 1：设 $X_i$ 为 $D=KB$ 个 i.i.d. 样本、block absmax 为 $m_k$，广义 $p$-mean $G_K^{(p)} = \left(\frac{1}{K}\sum_{k=1}^{K} m_k^p\right)^{1/p}$，则 $K \to \infty$ 时 $\frac{G_K^{(p)}}{\text{RMS}(X)} \to c(p, B)$，即 block absmax 的 power mean 与 RMS 之比收敛到只依赖 $p$、$B$ 和分布形状的常数——直觉上整个 tensor 被 $\sigma$ 缩放时，RMS 和 block absmax 的 power mean 同步被 $\sigma$ 缩放，比值是常数。落地实现就是 $\tilde{m}_{tk} = \max_b |X_{tkb}|$、$\tilde{\rho}_t = \tilde{c}(p,B) \cdot \left(\frac{1}{K}\sum_{k=1}^{K} \tilde{m}_{tk}^p\right)^{-1/p} + \epsilon$，其中 $\tilde{c}(p,B)$ 用 Gaussian 分布 Monte Carlo 预算好。reduction 规模从 $D$ 降到 $K=D/B$：$B=32$ 时缩 32 倍，$D=4096$ 时从 reduce 4096 个元素变成 reduce 128 个 block absmax。

**2. p=2 而非 p=1：output 上界决定训练稳定性**

$p$ 怎么选不只是近似精度问题，更是稳定性问题。$p=1$（算术平均）对 outlier feature 不敏感，但 output 上界是 $O(K)$ 太大；$p=2$（二次平均）output 上界 $O(\sqrt{K})$，正好和 RMSNorm 的 $O(\sqrt{D})$ 同量级。更紧的 bound 限制了极端值对权重更新的冲击，所以 8B 模型上 MXNorm(p=1) 会因 outlier 触发 loss spike 而落后，MXNorm(p=2) 则稳定、final loss 甚至略优于 RMSNorm（2.126 vs 2.132）。这条"bounds matter more than approximation quality"的洞察才是选 $p=2$ 的真正理由。

**3. MXNormLinear：把 gain 吸进下一层权重**

MXNorm 的输出是 MXFP 格式（block scales + 量化值），没法直接做 elementwise 乘 gain $\gamma$。利用线性运算的结合律，把 $\gamma$ 提前融进后续 Linear 层的权重矩阵：$H = \text{MXNorm}(X) \cdot \text{MXCast}(W \cdot \gamma)^\top$，于是 gain 不再需要单独一遍 elementwise。反向传播用 RMSNorm 的梯度作 straight-through estimator，缓存 $X$ 和 $\tilde{\rho}$ 供 backward（内存开销与标准 RMSNorm + MXCast 相同），整体是真正零额外超参数的 drop-in replacement。

## 实验关键数据

### 预训练稳定性（Learning Rate Sensitivity）

**125M & 1B模型**：在最优学习率附近，三种方案的training loss差异极小：
- RMSNorm 125M: 3.090±0.004, 1B: 2.692±0.011
- MXNorm(p=1) 125M: 3.113±0.012, 1B: 2.684±0.009
- MXNorm(p=2) 125M: 3.116±0.010, 1B: 2.691±0.007

### 8B模型预训练（300B tokens on SlimPajama）

| 方案 | Final Loss |
|------|-----------|
| RMSNorm | 2.132 |
| MXNorm(p=1) | 2.175 (显著更差，有loss spike) |
| MXNorm(p=2) | **2.126** (与RMSNorm几乎一致) |

MXNorm(p=2)甚至略优于RMSNorm（2.126 vs 2.132）。MXNorm(p=1)因为output上界过大导致outlier feature引发loss spike后落后。

### 零样本下游评估（OLMES, 10个NLP任务）

| 方案 | 胜出任务数 |
|------|-----------|
| RMSNorm | 5/10 |
| MXNorm(p=2) | **5/10** |
| MXNorm(p=1) | 0/10 |

MXNorm(p=2)与RMSNorm在下游零样本性能上打平（各赢5个benchmark），表明训练质量完全comparable。

### Kernel加速（torch.compile, GB200）

**Isolated kernel benchmark**（MXNorm vs RMSNorm+MXCast融合）：
- MXFP8 (B=32, E4M3)：平均加速**41.7%**，最高**2.4x**
- NVFP4 (B=16, E2M1)：平均加速**31.2%**
- 加速随token数增加而增大，随hidden dimension增大趋于稳定

**Full transformer layer benchmark**（Llama 3 8B, 8 transformer layers on GB200）：
- MXFP8：geometric mean加速**+1.3%**
- NVFP4：geometric mean加速**+2.6%**

随着精度更低（NVFP4 < MXFP8），MatMul更快，normalization占比更高→MXNorm的加速优势更大。

### 近似质量

MXFP dequantized tensor的$r^2$ goodness-of-fit随block数增加渐近趋向1。在hidden dim ≥ 1024（32 blocks）时已有excellent近似质量。Scale和value tensor的分布与RMSNorm+MXCast几乎identical。

## 亮点与洞察

- **优雅的"免费午餐"设计**：MXNorm复用MXFP8量化已有的block absmax计算，不增加任何新的统计量收集操作——将两个本来独立的reduction融合为一个，经典的compute reuse思路
- **理论保证严谨**：Theorem 1给出了block absmax power mean收敛到RMS的严格证明（基于强大数定律+连续映射定理），不仅是经验有效而是有理论基石
- **p=2 > p=1的洞察深刻**：不仅是近似精度的问题，而是output上界（$O(\sqrt{K})$ vs $O(K)$）决定了训练稳定性。这个分析为normalization层设计提供了新的视角：bounds matter more than approximation quality
- **真正的drop-in replacement**：无新超参数、无需修改训练pipeline、backward用RMSNorm梯度做straight-through estimator、gain参数融入权重矩阵→工程集成成本极低
- **前瞻性强**：随着GPU代际更新，MatMul vs non-MatMul的性能gap持续扩大(Rubin: 280x vs 8.3x)，MXNorm的价值将持续增长

## 局限与展望

1. **增益有限**：Llama 3 8B full layer加速仅+1.3%(MXFP8)/+2.6%(NVFP4)——因为当前模型中normalization本身占比不大。但这个比例会随硬件代际更新而增大
2. **仅验证Pre-Norm架构**：RMSNorm在Linear之前是MXNorm的前提条件。Post-Norm架构或其他normalization placement不适用
3. **仅适用于block量化格式**：MXNorm依赖MXFP的block scales。若模型使用per-tensor或per-channel量化（FP8），则block absmax不可用
4. **Gaussian分布假设**：$\tilde{c}(p,B)$通过Gaussian Monte Carlo预计算。实际activation分布可能偏离Gaussian（尤其有outlier feature时），但实验表明robustness足够
5. **未覆盖MoE架构**：Mixture-of-Experts模型中normalization的位置和行为可能不同，需要额外验证
6. **其他非MatMul瓶颈未解决**：论文指出RoPE、gated linear unit等同样受制于non-MatMul性能瓶颈，但留给了future work

## 相关工作与启发

- **FlashNorm**：异步计算RMS后用raw input乘weight→有swamping accumulator风险；MXNorm通过block scales避免了这一问题
- **Partial RMS**：仅用前$k$个元素计算RMS→容易漏掉outlier values；MXNorm用block absmax天然捕捉每block的极值
- **Transformers without normalization**：约束权重在hypersphere上或用tanh替代norm→引入其他开销；MXNorm保持了normalization的功能但降低了计算成本
- **MXFP/MX formats**：本文是MX格式量化生态中的自然延伸——不仅用block scales做量化，还顺带做normalization
- **启发**：这种"一个操作的副产品可以作为另一个操作的输入"的计算复用思路，可以推广到其他需要统计量的操作（如attention score scaling、layer scale等）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.0 | 核心idea简洁有力：复用block scales估计RMS。理论分析(Theorem 1)和p选择的stability分析是亮点 |
| 实用性 | 4.5 | 真正的drop-in replacement，零额外超参数，实测加速可观。随硬件更新价值持续增长 |
| 实验充分度 | 4.5 | 125M→1B→8B三个规模验证，LR sensitivity/zero-shot/kernel benchmark/loss spike分析全面深入 |
| 写作质量 | 4.5 | GPU性能表引出动机极为compelling，理论和实验紧密衔接，appendix详尽（含PyTorch实现） |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[CVPR 2026\] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)
- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)

</div>

<!-- RELATED:END -->
