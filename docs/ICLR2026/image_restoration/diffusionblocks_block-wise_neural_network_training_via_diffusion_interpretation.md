---
title: >-
  [论文解读] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation
description: >-
  [ICLR2026][图像恢复][block-wise training] 提出 DiffusionBlocks，将残差网络的逐层更新解释为连续时间扩散过程的离散化步骤，从而将网络切分为可完全独立训练的 block，在保持端到端训练性能的同时按 block 数 B 倍减少训练显存。
tags:
  - ICLR2026
  - 图像恢复
  - block-wise training
  - 扩散模型
  - score matching
  - memory efficiency
  - residual networks
---

# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation

**会议**: ICLR2026  
**arXiv**: [2506.14202](https://arxiv.org/abs/2506.14202)  
**代码**: [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)  
**领域**: image_restoration  
**关键词**: block-wise training, diffusion models, score matching, memory efficiency, residual networks

## 一句话总结

提出 DiffusionBlocks，将残差网络的逐层更新解释为连续时间扩散过程的离散化步骤，从而将网络切分为可完全独立训练的 block，在保持端到端训练性能的同时按 block 数 B 倍减少训练显存。

## 背景与动机

- 端到端反向传播需要存储所有层的中间激活值，显存随网络深度线性增长，严重制约模型规模和实际部署
- 已有的 block-wise 训练方法（如 Forward-Forward、greedy layer-wise training）依赖临时性的局部目标函数，缺乏理论保证，且基本仅在分类任务上验证，无法自然扩展到生成任务
- Score-based diffusion models 的去噪目标天然具有"各噪声级别可独立优化"的性质——这恰好为 block-wise 训练提供了缺失的理论基础
- 残差连接（ResNet、Transformer 等）的更新规则 $\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f_{\theta_\ell}(\mathbf{z}_\ell)$ 可对应扩散过程 probability flow ODE 的 Euler 离散化

## 核心问题

如何为基于 Transformer 的网络设计一种**有理论根据**的 block-wise 训练框架，使得：

1. 每个 block 可以**完全独立**训练（不需要其他 block 的梯度或激活值）
2. 与端到端训练保持竞争力
3. 能通用于分类和生成等多种任务/架构

## 方法详解

### 核心洞察：残差连接 = 扩散过程离散步

在 Variance Exploding (VE) 扩散框架下，给定噪声级别 $\sigma_0 > \sigma_1 > \cdots > \sigma_T$，对 probability flow ODE 做 Euler 离散化得到：

$$\mathbf{z}_{\sigma_\ell} = \mathbf{z}_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}}\left(\mathbf{z}_{\sigma_{\ell-1}} - D_\theta(\mathbf{z}_{\sigma_{\ell-1}}, \sigma_{\ell-1})\right)$$

这与残差网络的 skip connection 更新规则 $\mathbf{z}_\ell = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$ 天然对应。

### 三步转换流程

**Step 1: Block 划分** — 将 $L$ 层网络分为 $B$ 个 block $\mathcal{F}_1, \ldots, \mathcal{F}_B$，每个 block 包含连续的若干层。

**Step 2: 噪声范围分配** — 定义噪声分布 $p_{\text{noise}}$（推荐 log-normal），将 $[\sigma_{\min}, \sigma_{\max}]$ 划分为 $B$ 个区间 $\{[\sigma_b, \sigma_{b-1}]\}_{b=1}^B$，每个 block 负责对应范围的去噪。

**Step 3: 噪声条件化改造** — 扩展每个 block 的输入为 $\tilde{\mathbf{x}} = (\mathbf{x}, \mathbf{z}_\sigma)$，其中 $\mathbf{z}_\sigma = \mathbf{y} + \sigma\epsilon$；加入噪声级别条件（如 AdaLN）。每个 block 独立训练预测目标 $\mathbf{y}$。

### 独立训练目标

每个 block $b$ 的损失函数为：

$$\mathcal{L}_b(\theta_b) = \mathbb{E}_{(\mathbf{x},\mathbf{y}), \sigma\sim p_{\text{noise}}^{(b)}, \epsilon\sim\mathcal{N}(0,I)}\left[w(\sigma)\cdot\|f_{\theta_b|\sigma}(\mathbf{x}, \mathbf{y}+\sigma\epsilon) - \mathbf{y}\|_2^2\right]$$

关键在于：$B$ 个 block 各自独立优化、无需相互通信，却能共同覆盖完整的噪声分布。

### 等概率划分策略 (Equi-probability Partitioning)

不采用均匀划分噪声区间（会在高/低噪声端浪费容量），而是按 log-normal 分布的累积概率质量等分：

$$\int_{\sigma_{b-1}}^{\sigma_b} p_{\text{noise}}(\sigma)\,d\sigma = 1/B$$

这确保每个 block 处理等量的训练分布，在去噪难度最大的中间噪声级别分配更细的区间，效率更优。

### 推理过程

推理时按从高噪声到低噪声的顺序依次调用各 block 的去噪步骤；对于 diffusion model，每个去噪步只需加载一个 block，带来 $B$ 倍推理加速。

## 实验关键数据

| 任务 / 架构 | 数据集 | 端到端基线 | DiffusionBlocks | Block 数 / 显存缩减 |
|---|---|---|---|---|
| ViT 分类 | CIFAR-100 | 60.25% Acc | 59.30% Acc | B=3 / 3× |
| DiT 图像生成 | CIFAR-10 | 32.84 FID | 30.59 FID | B=3 / 3× |
| DiT 图像生成 | ImageNet 256 | 12.09 FID | 10.63 FID | B=3 / 3× |
| Masked Diffusion 文本 | text8 | 1.56 BPC | 1.45 BPC | B=3 / 3× |
| AR Transformer 文本 | LM1B | 0.50 MAUVE | 0.71 MAUVE | B=4 / 4× |
| AR Transformer 文本 | OpenWebText | 0.85 MAUVE | 0.82 MAUVE | B=4 / 4× |
| Huginn (recurrent-depth) | LM1B | 0.49 MAUVE | 0.70 MAUVE | 消除 32 次迭代 |

- Forward-Forward 在 CIFAR-100 上仅达 7.85% 准确率，远逊于 DiffusionBlocks
- ImageNet 上 B=2 时 FID=9.90，**优于**端到端训练 (12.09)，适度划分反而提升性能
- 等概率划分在所有层分配方案下均显著优于均匀划分（CIFAR-10 FID: 38.03 vs 43.53）

## 亮点

1. **理论基础扎实**：从 score matching 的噪声级别独立性出发，自然推导出 block 独立训练目标，非启发式拼凑
2. **通用性极强**：一套三步转换流程适用于 ViT、DiT、AR Transformer、Masked Diffusion、Recurrent-depth 共五类架构
3. **等概率划分**是简洁而关键的设计——让每个 block 承担等量去噪难度，无需手工调整层分配
4. **多重效率收益**：训练 $B$ 倍显存缩减；diffusion model 推理 $B$ 倍加速；recurrent-depth 模型省去 BPTT
5. **部分场景超越端到端**：ImageNet B=2/3 的 FID 优于不分 block 的端到端训练，说明适度专业化有正收益

## 局限与展望

- 实验中 ViT 分类仅在 CIFAR-100 上验证（60.25→59.30），大规模 ImageNet 分类未测试
- 推理时仍需按序调用各 block，无法并行化推理步骤
- 噪声条件化改造（AdaLN 等）增加了少量参数和工程复杂度
- B 过大时性能下降（B=6 时 FID 14.43），block 粒度有下限
- 主要面向 Transformer 类残差架构，对无残差连接的网络适用性未讨论

## 与相关工作的对比

| 方法 | 理论基础 | 任务通用性 | 连续时间 | Block 独立 |
|---|---|---|---|---|
| Forward-Forward | 对比目标 | 仅分类 | ✗ | ✓ |
| NoProp | 扩散相关 | 仅分类 | ✓(CT) 或 ✗(DT) | ✗(CT) 或 ✓(DT) |
| DiffusionBlocks | Score matching | 分类+生成 | ✓ | ✓ |

- NoProp 与自定义 CNN 架构捆绑，无法直接迁移到 Transformer；DiffusionBlocks 在 NoProp 的架构上也优于其所有变体（46.88 vs 46.06/21.31/37.57）
- 与 stage-specific diffusion models (eDiff-I 等) 的区别在于：后者是联合训练或从共享参数微调，DiffusionBlocks 各 block **完全隔离**

## 启发与关联

- "残差连接 ≈ 扩散离散化步骤"的视角可进一步推广：任何具有残差结构的深层模型都可能受益于这种分块独立训练
- 等概率划分思想可迁移到其他需要"分段处理不同难度子任务"的场景（如课程学习、多尺度训练）
- 对 recurrent-depth 模型消除 BPTT 的能力值得关注——随着 universal transformer / Huginn 等模型兴起，该方法可降低其训练成本
- 结合模型并行（每个 block 放不同 GPU），可实现更激进的深度扩展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 将扩散独立性引入 block-wise 训练是原创性极高的理论贡献
- 实验充分度: ⭐⭐⭐⭐ — 五类架构覆盖面广，但分类任务规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰，三步流程直观易懂
- 价值: ⭐⭐⭐⭐ — 为大模型训练显存瓶颈提供了有理论保证的新范式

<!-- RELATED:START -->

## 相关论文

- [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](../../ECCV2024/image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [DiffFNO: Diffusion Fourier Neural Operator](../../CVPR2025/image_restoration/difffno_diffusion_fourier_neural_operator.md)
- [Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)

<!-- RELATED:END -->
