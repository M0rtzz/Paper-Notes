---
title: >-
  [论文解读] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks
description: >-
  [ICML 2026][模型压缩][BNN] SURGE 给每个二值化层并联一个"全精度辅助分支"，前向输出不变但反向能从全精度分支额外回传一份"非 STE 截断"的高阶梯度，并用 AGS 按梯度范数比动态平衡两路贡献，让 BNN 在 ResNet-18/ImageNet 上做到 62.0% top-1…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "BNN"
  - "STE"
  - "梯度失配"
  - "双路径补偿"
  - "自适应梯度缩放"
---

# SURGE: Surrogate Gradient Adaptation in Binary Neural Networks

**会议**: ICML 2026  
**arXiv**: [2605.10989](https://arxiv.org/abs/2605.10989)  
**代码**: 暂未公开  
**领域**: 模型压缩 / 二值神经网络 / 量化感知训练  
**关键词**: BNN、STE、梯度失配、双路径补偿、自适应梯度缩放

## 一句话总结
SURGE 给每个二值化层并联一个"全精度辅助分支"，前向输出不变但反向能从全精度分支额外回传一份"非 STE 截断"的高阶梯度，并用 AGS 按梯度范数比动态平衡两路贡献，让 BNN 在 ResNet-18/ImageNet 上做到 62.0% top-1，比 ReCU 高 1.0%、比 IR-Net 高 3.9%。

## 研究背景与动机

**领域现状**：二值神经网络（BNN）把权重和激活量化到 $\{-1,+1\}$，理论上能给出 $32\times$ 内存压缩和 $58\times$ 推理加速，是边缘部署最激进的量化方案。训练上几乎所有 BNN 都依赖 Straight-Through Estimator（STE）：前向走 $\text{sign}(\cdot)$，反向直接把 $\frac{\partial\mathbf{B}_W}{\partial W}\approx 1$、$\frac{\partial\mathbf{B}_x}{\partial x}\approx\mathbb{1}_{\{|x|\le 1\}}$ 当作代理梯度。

**现有痛点**：STE 有两个根本问题。其一，sign 的真梯度几乎处处为零，用恒等函数做替身会引入系统性偏差，是公认的"梯度失配（gradient mismatch）"。其二，激活梯度被硬剪到 $[-1,1]$ 外即清零，大量信息被丢弃。已有工作（DSQ 的 sigmoid 近似、IR-Net 的渐近 sign、ReCU 的特征分布对齐）大多依赖手工设计的近似函数，无法保证最优。

**核心矛盾**：BNN 训练里"前向必须严格二值（保证推理加速）" 和 "反向必须有足够丰富的梯度（保证能学）"是一对硬矛盾——只要前向是 sign，反向就只能拿一阶恒等代理凑合。

**本文目标**：1) 在不改前向输出的前提下，从外部补一份"非 STE、低偏差"的梯度回 main branch；2) 防止补偿梯度量级失衡破坏主分支收敛；3) 推理阶段彻底丢掉辅助分支，零额外开销。

**切入角度**：既然 STE 是 sign 的一阶近似，那就给每层并联一个"全精度副本"用它的真实梯度去补 STE 缺的高阶项；同时由于两路梯度量级未知，用 norm-ratio 自适应缩放来动态平衡。

**核心 idea**：用"前向自抵消、反向开门"的 detach 技巧让全精度辅助分支只参与反向，再用 AGS 按 $\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$ 自适应缩放，把 STE 的一阶 surrogate 修成更接近真实梯度的混合估计。

## 方法详解

### 整体框架
对每个二值化线性算子（conv、linear、attention projection），SURGE 并行挂一个尺寸完全相同的全精度副本（auxiliary branch）。前向时利用 $\text{detach}$ 技巧让辅助分支的两次出现互相抵消，保证输出严格等于纯 BNN；反向时辅助分支正常回传，main branch 走 STE 回传，二者在输入处汇合。AGS 动态算缩放因子 $\lambda_{\text{AGS}}$ 平衡两路贡献。训练完毕后辅助分支被丢弃，推理就是标准 BNN。

### 关键设计

1. **Dual-Path Gradient Compensator（DPGC）**:

    - 功能：在不动前向输出的情况下，让每一个二值化层都额外获得一份"非 STE 截断"的高阶梯度信息。
    - 核心思路：定义二值前向 $f_b(x;W_b)=Q_W(W_b)^\top Q_x(x)$、全精度前向 $f_a(x;W_a)=W_a^\top x$，缩放 $f_{ao}(x)=\lambda f_a(x)$。输出写作 $\text{output}=f_b(x;W_b)-f_{ao}(x;W_a)\downarrow+f_{ao}(x;W_a)$，其中 $\downarrow$ 是 stop-gradient。前向时第二、三项数值相等正负抵消，输出 = $f_b$；反向时 detached 那一项的梯度被截断，剩下 $f_b$ 走 STE、$f_{ao}$ 走全精度，于是 $\frac{\partial\mathcal{L}}{\partial x}=g_b+\lambda g_a$，其中 $g_b$ 是 STE 一阶近似、$g_a$ 是辅助分支的真实梯度提供的高阶补偿。
    - 设计动机：传统改进 STE 的工作（piecewise polynomial、SignSwish 等）都是"换个函数继续骗"，没解决根本问题。DPGC 的 detach 技巧让"输出严格二值"和"反向能拿到全精度信号"这对矛盾共存，是非常巧妙的工程构造，而且推理时辅助分支可丢，零额外开销。

2. **Adaptive Gradient Scaler（AGS）**:

    - 功能：动态平衡 $g_b$ 和 $g_a$ 的量级，防止辅助路梯度过大冲垮主分支训练。
    - 核心思路：直接把 $\lambda$ 写成 $\lambda_{\text{AGS}}=\eta\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$，其中 $\eta$ 是基础缩放系数、$\epsilon=10^{-8}$ 防除零。论文从二阶矩模型出发证明最优 $\lambda^*=\frac{\langle\delta_b,\mu_a\rangle}{\|\mu_a\|_2^2+\text{tr}(\text{Var}(g_a))}$（$\delta_b$ 是 STE 的偏差向量），并在 alignment $\cos\theta$、relative bias ratio $\beta=\|\delta_b\|_2/\|\mu_b\|_2$、noise ratio $\rho$ 近似稳定的假设下，推得 $\lambda^*\approx\eta\frac{\|\mu_b\|_2}{\|\mu_a\|_2}$，再用 mini-batch 估计就得到上面的实用公式。
    - 设计动机：固定 $\lambda$ 不是太大就是太小：太大辅助路炸主分支、太小补偿失效。AGS 让两路始终量级相当，保证 STE 仍主导优化方向、辅助路只作"高阶修正"，理论上等价于在均方误差意义下的最优凸组合。

3. **训练-推理对称的双路架构**:

    - 功能：让 SURGE 成为通用插件，可用于 CNN（conv block）和 Transformer（attention projection / FFN 的 linear）。
    - 核心思路：DPGC 是 architecture-agnostic 的，凡是有一个二值化线性算子的地方都可以挂；图 2 同时给出 conv block 和 transformer block 的接入示意。训练时整网三种状态共存（main forward、main backward、auxiliary backward），推理时辅助 $W_a$ 全部丢掉。
    - 设计动机：以前的 BNN 训练 trick 多是任务/架构特定（如 ReActNet 的 RPReLU），SURGE 把"补偿梯度"做成层级插件，迁移成本极低。

### 损失函数 / 训练策略
端到端 cross-entropy（分类）/ detection loss（VOC）/ NLU loss（GLUE），不引入额外训练损失。$\eta$ 是少数需调的超参；推理零额外开销。

## 实验关键数据

### 主实验
覆盖 4 个 benchmark：CIFAR-10、ImageNet-1K（ResNet-18/34、ReActNet）、PASCAL VOC（Faster-RCNN + ResNet-18 backbone）、GLUE（BERT-base）。

| 网络 / 任务 | 方法 | W/A | Top-1 / mAP / 平均 |
|------------|------|-----|--------------------|
| ResNet-18 / CIFAR-10 | ReCU | 1/1 | 92.8% |
| ResNet-18 / CIFAR-10 | **SURGE** | 1/1 | **93.1%** (+0.3) |
| ResNet-20 / CIFAR-10 | ReCU | 1/1 | 87.4% |
| ResNet-20 / CIFAR-10 | **SURGE** | 1/1 | **88.0%** (+0.6) |
| VGG-Small / CIFAR-10 | ReCU | 1/1 | 92.2% |
| VGG-Small / CIFAR-10 | **SURGE** | 1/1 | **92.5%** (+0.3) |
| ResNet-18 / ImageNet (one-stage) | IR-Net | 1/1 | 58.1% |
| ResNet-18 / ImageNet (one-stage) | BONN | 1/1 | 59.3% |
| ResNet-18 / ImageNet (one-stage) | ReCU | 1/1 | ~61% |
| ResNet-18 / ImageNet (one-stage) | **SURGE** | 1/1 | **62.0%** (+3.9 over IR-Net) |

在 VOC、GLUE 上同样全面超越前 SOTA，且 OPs 与之前 BNN 一致（推理开销零增）。

### 消融实验

| 配置 | ImageNet ResNet-18 Top-1 (one-stage 量化) | 说明 |
|------|----------------------------|------|
| STE baseline | 较 SURGE 低数个百分点 | 仅一阶 surrogate |
| + DPGC（固定 $\lambda$） | 显著提升，但偶尔不稳定 | 缺乏量级平衡 |
| + AGS（norm-ratio）= **SURGE** | **62.0%** 且训练稳定 | 全模型 |
| 改 AGS 为定值 $\lambda$ | 大 $\lambda$ 训不动、小 $\lambda$ 无补偿 | 验证自适应必要性 |
| 仅在最后几层用 DPGC | 提升幅度大幅减小 | 越深层失配累积越严重 |

### 关键发现
- 图 1 的梯度统计显示：加 SURGE 后激活梯度分布明显**右移**且尾部更重，证实辅助分支确实恢复了 STE 剪掉的那部分信息。
- DPGC + AGS 的组合在 ImageNet 上比单 DPGC 提升 0.5~1%，说明量级平衡不仅是"工程稳定性"，而是收敛的必要条件。
- ResNet-18 训完丢掉辅助分支后，推理 OPs 与标准 BNN 一致（$1.63\times 10^8$），完美匹配"训练补偿、推理零额外开销"的目标。
- 在 BERT-base/GLUE 上同样有效，证明 SURGE 不局限于卷积，对 attention projection 这类 linear 算子也适用。

## 亮点与洞察
- "detach 自抵消"的写法是整篇文章最巧的工程 trick：$f-f\downarrow+f$ 在前向是 $f$、在反向是 $f$ 的真实梯度，对所有"想让前向走 A、反向走 B"的场景都通用，可以迁移到知识蒸馏、对抗训练、可微剪枝等任务。
- "把 STE 当低阶近似、用全精度副本补高阶项"的视角把 BNN 训练问题从"找个更聪明的 sign 近似"重构为"补偿一阶 Taylor 残差"，物理直觉清楚得多。
- AGS 用 norm-ratio 平衡两路，本质和 GradNorm、PCGrad 这类多任务梯度平衡同构，但理论推导明确给出了"最优 $\lambda^*$ 在等向噪声下退化为 $\eta\|\mu_b\|_2/\|\mu_a\|_2$"，比纯启发式有说服力。

## 局限与展望
- 训练时显存与 FLOPs 都几乎翻倍（辅助分支与主分支同尺寸），训成本不低。
- $\eta$ 仍需手调，对不同 backbone 最优值不一致；理论上 $\eta=\kappa c_\theta/(1+\rho)$，但这些量没人监控，实操还是 grid search。
- 假设 $g_b$、$g_a$ noise 不相关，深层网络里这个假设未必精确成立。
- 与多 bit 量化（W2A2、W4A4）的对比缺失，纯 1-bit 之外的迁移效果未知。

## 相关工作与启发
- **vs IR-Net / ReCU / BONN**：他们都在改 sign 的近似函数或权重分布，本质是"改前向"；SURGE 不动前向，只在反向开旁路，思路正交而且能跟前者叠加。
- **vs DSQ / LSQ**：DSQ 用 parametric sigmoid 渐近逼近 sign，LSQ 引入可学习 scale；SURGE 把"可学习"放到一个完全独立的全精度副本里，表达力更强且推理零负担。
- **vs Frequency-domain BNN（FDA-BNN）**：FDA-BNN 把 sign 转到频域去缓解失配；SURGE 直接在空域用全精度梯度补，工程实现更简单。

## 评分
- 新颖性: ⭐⭐⭐⭐ "前向自抵消、反向开门"的 detach 技巧 + AGS 的 norm-ratio 推导组合，构造干净。
- 实验充分度: ⭐⭐⭐⭐ 横跨 4 大 benchmark、3 类任务、CNN + Transformer，BNN 论文里属上乘。
- 写作质量: ⭐⭐⭐⭐ 图 1/2 把核心机制说得很直观，定理 5.3 与推论 5.4 的推导也清楚。
- 价值: ⭐⭐⭐⭐ ResNet-18/ImageNet 推到 62.0% 是当年 one-stage BNN 的新天花板，且推理零额外开销，工业落地友好。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](../../AAAI2026/model_compression/bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](../../ICML2025/model_compression/an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)

</div>

<!-- RELATED:END -->
