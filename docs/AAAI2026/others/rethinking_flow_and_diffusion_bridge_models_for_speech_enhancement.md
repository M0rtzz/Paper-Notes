---
title: >-
  [论文解读] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement
description: >-
  [AAAI 2026][语音增强] 本文提出了一个统一的理论框架，将语音增强中的 flow matching、score-based diffusion 和 Schrödinger bridge 模型统一为在配对数据之间构造不同高斯概率路径的过程，并揭示了这类生成模型每一步采样本质上等价于预测式语音增强，进而利用预测范式中的高性能骨干网络、改进损失函数和微调策略来增强桥模型性能。
tags:
  - AAAI 2026
  - 语音增强
  - 扩散桥模型
  - 流匹配
  - Schrödinger桥
  - 统一框架
---

# Rethinking Flow and Diffusion Bridge Models for Speech Enhancement

**会议**: AAAI 2026  
**arXiv**: [2602.18355](https://arxiv.org/abs/2602.18355)  
**代码**: [GitHub](https://github.com/Dahan-Wang/Rethinking-Flow-and-Diffusion-Bridge-Models-for-Speech-Enhancement)  
**领域**: 语音增强 / 生成模型  
**关键词**: 语音增强, 扩散桥模型, 流匹配, Schrödinger桥, 统一框架

## 一句话总结

本文提出了一个统一的理论框架，将语音增强中的 flow matching、score-based diffusion 和 Schrödinger bridge 模型统一为在配对数据之间构造不同高斯概率路径的过程，并揭示了这类生成模型每一步采样本质上等价于预测式语音增强，进而利用预测范式中的高性能骨干网络、改进损失函数和微调策略来增强桥模型性能。

## 研究背景与动机

语音增强（Speech Enhancement, SE）旨在从含噪观测中恢复干净语音，深度学习方法可分为**预测式**（直接学习含噪到干净信号的映射）和**生成式**（对干净语音的条件分布建模）。近年来，基于 flow 和 diffusion 的生成式方法在 SE 中大量涌现，主要包括：

**Score-based diffusion 模型**：通过 OU 过程或布朗桥设计 SDE 的漂移项，在干净和含噪信号之间建立扩散过程（SGMSE+、BBED 等）。

**Schrödinger bridge（SB）**：在 Dirac 端点约束下优化路径测度，配合数据预测训练策略达到 SOTA（SBVE 等）。

**Flow matching（FM）**：在含噪语音条件下构造概率路径，通过条件向量场实现高效采样（FlowSE 等）。

然而，这些方法基于不同的理论基础（score matching、SB 优化、flow matching），**尚未在 SE 领域被统一到一个共同框架中**。此外，SB 模型中使用的数据预测目标暗示其与预测式方法存在内在联系，但这一联系在先前工作中**未被充分探索**。

本文的核心动机有两点：
- 能否将这些看似不同的生成式 SE 方法统一为一种框架？
- 生成式桥模型与预测式 SE 模型之间到底存在什么本质联系？能否利用这种联系来改进模型？

## 方法详解

### 整体框架

作者提出的统一框架核心思想：**所有 flow 和 diffusion bridge 模型都可以解释为在配对数据（含噪 $\mathbf{y}$ 和干净 $\mathbf{s}$）之间构造不同的高斯概率路径**。概率路径定义为：

$$p_t(\mathbf{x}_t|\mathbf{s},\mathbf{y}) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_t(\mathbf{s},\mathbf{y}), \sigma_t^2 \mathbf{I})$$

其中均值为干净和含噪信号的插值：$\boldsymbol{\mu}_t = a_t \mathbf{s} + b_t \mathbf{y}$。不同方法的区别仅在于 $a_t$, $b_t$, $\sigma_t$ 的设计不同。一旦概率路径确定，对应的采样 ODE 和前向/后向 SDE 可直接推导得到。

### 关键设计

1. **统一概率路径框架**：通过条件流匹配导出采样 ODE：

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \frac{\sigma_t'}{\sigma_t}\mathbf{x}_t + \left(a_t' - a_t\frac{\sigma_t'}{\sigma_t}\right)\mathbf{s} + \left(b_t' - b_t\frac{\sigma_t'}{\sigma_t}\right)\mathbf{y}$$

并通过 Fokker-Planck 方程扩展到前向/后向 SDE。不同模型的概率路径参数如下：

| 方法 | $a_t$ | $b_t$ | $\sigma_t$ |
|------|--------|--------|-------------|
| OUVE | $e^{-\gamma t}$ | $1-e^{-\gamma t}$ | 复杂表达式 |
| BBED | $1-t$ | $t$ | $c(1-t)E_t$ |
| SB | $\alpha_t\bar{\rho}_t^2/\rho_1^2$ | $\bar{\alpha}_t\rho_t^2/\rho_1^2$ | $\alpha_t^2\bar{\rho}_t^2\rho_t^2/\rho_1^2$ |
| OT-CFM | $t$ | $1-t$ | $(1-t)\sigma_{\max}+t\sigma_{\min}$ |
| SB-CFM | $1-t$ | $t$ | $\sigma^2 t(1-t)$ |

这一统一使得不再需要从 SDE 设计或 KL 散度优化出发来推导各模型，大大简化了理论分析。

2. **预测等价性洞察**：这是本文最核心的发现。作者证明在数据预测训练策略下，每一步采样的网络输出本质上是在做预测式语音增强。最终采样结果可以表示为网络在各步输出的加权和：

$$\mathbf{x}_{t_0} = \sum_{n=1}^{N} w_n \mathbf{s}_{t_n} + w_y \mathbf{y}$$

通过 SB-CFM 参数化的数值模拟发现：最后一步的权重 $w_N$ 远远占主导地位（接近 1），其他步和含噪输入 $\mathbf{y}$ 的贡献可以忽略不计。这意味着：

- **单步采样几乎等价于预测模型**，完全依赖数据预测，不利用中间状态信息
- 多步采样的性能上界受限于预测模型的能力
- 训练时只在 $t=1$（即纯含噪输入）时有意义的单步情况下，其他时间步的训练变得冗余

3. **改进的桥模型**：基于上述洞察，作者将 SOTA 预测式 SE 模型 TF-GridNet 引入生成框架，替代常用的 NCSN++ 的 U-Net 架构。为使 TF-GridNet 能接受扩散时间 $t$ 作为输入，设计了**时间嵌入机制**：先通过 Fourier embedding + 全连接层 + SiLU 激活得到时间嵌入向量，然后在每个 TF-GridNet block 的开头通过专用全连接层加到输入特征上。参数量从 65.6M 降至 2.2M，MACs 从 66G 降至 38G。

### 损失函数 / 训练策略

作者对数据预测损失进行了改进，引入了预测式 SE 中常用的损失组合：

- **负 SI-SNR 损失**：$\mathcal{L}_{\text{SI-SNR}}(\hat{x}, x) = -\log_{10}\frac{\|x_t\|^2}{\|\hat{x}-x_t\|^2}$，强调信号级信噪比
- **功率压缩频谱幅度损失**：$\mathcal{L}_{\text{mag}} = \text{MSE}(|\hat{X}|^{0.3}, |X|^{0.3})$，更好地关注频谱幅度
- **功率压缩实/虚部损失**：$\mathcal{L}_{\text{real/imag}} = \text{MSE}(\hat{X}_{r/i}/|\hat{X}|^{0.7}, X_{r/i}/|X|^{0.7})$

总损失为：$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{SI-SNR}} + \lambda_2 \mathcal{L}_{\text{mag}} + \lambda_3 (\mathcal{L}_{\text{real}} + \mathcal{L}_{\text{imag}})$

此外，采用 **CRP 微调策略**（Correcting the Reverse Process）：在采样过程中只更新最后一步的模型权重，弥补最后一步可能的欠优化问题。这与权重分析的发现一致——最后一步对最终结果影响最大。

## 实验关键数据

### 主实验

**DNS3 测试集**：

| 模型 | 参数量 (M) | MACs (G) | SI-SNR | ESTOI | PESQ | DNSMOS |
|------|-----------|----------|--------|-------|------|--------|
| Noisy | - | - | 5.613 | 0.669 | 1.406 | 2.147 |
| SGMSE+ (OUVE) | 65.6 | 66×60 | 11.873 | 0.796 | 2.336 | 3.647 |
| SBVE | 65.6 | 66×60 | 14.959 | 0.844 | 2.592 | 3.729 |
| TF-GridNet (预测式) | 2.1 | 38 | 16.448 | 0.872 | 3.187 | 3.743 |
| **本文 (NFEs=1)** | **2.2** | **38×1** | **16.245** | **0.870** | **3.185** | **3.740** |
| **本文 (NFEs=5)** | **2.2** | **38×5** | **16.424** | **0.874** | **3.213** | **3.752** |

**VoiceBank+DEMAND 测试集**：

| 模型 | SI-SNR | ESTOI | PESQ | DNSMOS |
|------|--------|-------|------|--------|
| SGMSE+ | 17.3 | 0.87 | 2.93 | 3.56 |
| SBVE | 19.4 | 0.88 | 2.91 | 3.59 |
| FlowSE | 19.0 | 0.88 | 3.12 | 3.58 |
| **本文** | **19.6** | **0.89** | **3.30** | **3.57** |

### 消融实验

| 配置 | SI-SNR | PESQ | 说明 |
|------|--------|------|------|
| NCSN++ + 原始损失 + SBVE | 14.158 | 2.706 | 原始基线 |
| TF-GridNet + 改进损失 + SBVE | 16.646 | 3.068 | 骨干+损失改进，巨大提升 |
| TF-GridNet + 改进损失 + SBVE + CRP | 16.424 | 3.213 | CRP 微调提升 PESQ |
| TF-GridNet + 改进损失 + OUVE | 11.302 | 2.129 | OUVE 路径效果差 |
| TF-GridNet + 改进损失 + OT-CFM | 14.866 | 2.834 | FM 路径不如 SB |
| TF-GridNet + 改进损失 + SB-CFM | 16.177 | 3.102 | SB-CFM 接近 SBVE |

### 关键发现

1. **单步采样 ≈ 预测模型**：本文模型 NFEs=1 时的性能（PESQ 3.185）已非常接近 SOTA 预测式 TF-GridNet（3.187），直接验证了理论分析。
2. **Dirac 端点 + 指数积分采样器 = 最优配置**：SBVE 和 SB-CFM 的概率路径在采样起点方差为零（Dirac 分布），配合指数积分采样器效果最好。
3. **预测性质限制了性能上界**：5 步采样仅比单步略有提升，与预测模型水平相当，印证了生成框架的预测本质限制了其超越预测模型的空间。

## 亮点与洞察

- **理论统一的优雅性**：将四类不同理论基础的方法归纳为概率路径的 $(a_t, b_t, \sigma_t)$ 三元组参数化，极其简洁。
- **生成=预测的深刻洞察**：揭示了扩散桥模型在数据预测损失下"形式为生成、本质是预测"的属性。这不仅解释了为什么单步采样就能工作，也指出了多步采样改进有限的根因。
- **极致的效率提升**：参数量从 65.6M 降至 2.2M（约 30 倍），计算量从 66G×60 步降至 38G×5 步（约 20 倍），性能反而大幅提升。
- **预测范式对生成模型的反哺**：损失函数、骨干网络、微调策略均借鉴预测式方法，形成了良性融合。

## 局限与展望

1. **性能上界受限**：作者明确指出，由于生成框架的预测本质，其性能不太可能显著超越对应的预测模型。
2. **通用性待验证**：实验仅在去噪和去混响任务上验证，其他语音任务（如语音分离、带宽扩展）是否同样适用尚不清楚。
3. **CRP 微调的局限**：CRP 仅对最后一步微调，前面步骤的误差积累问题可能仍未完全解决。
4. **概率路径设计的自动化**：目前路径参数仍需手动选择，能否通过学习自动发现最优路径是一个值得探索的方向。

## 相关工作与启发

- 本文的统一框架思路可扩展到其他配对数据桥模型任务（如图像翻译、文本-语音合成）。
- "生成本质是预测"的洞察对扩散模型在其他信号处理任务（如图像恢复）中的应用具有深远启发意义。
- TF-GridNet 作为骨干网络的成功表明，为扩散模型选择高性能的任务专用骨干比盲目扩大 U-Net 更为有效。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 统一框架和预测等价性分析具有很强的理论贡献 |
| 实用性 | ⭐⭐⭐⭐⭐ | 大幅降低参数和计算量，性能超越所有基线 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 采样结果分解和权重分析非常扎实 |
| 实验充分性 | ⭐⭐⭐⭐ | 两个数据集 + 详细消融，但缺少更多任务验证 |
| 写作质量 | ⭐⭐⭐⭐ | 逻辑清晰，理论推导完整 |
| 总评 | ⭐⭐⭐⭐½ | 理论洞察深刻，实践指导性强，是扩散模型理解的重要工作 |

<!-- RELATED:START -->

## 相关论文

- [Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)
- [Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)
- [Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)

<!-- RELATED:END -->
