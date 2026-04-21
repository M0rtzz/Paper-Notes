---
title: >-
  [论文解读] TinyFusion: Diffusion Transformers Learned Shallow
description: >-
  [CVPR 2025][图像生成][Transformer] 提出 TinyFusion，一种可学习的深度剪枝方法，通过 Gumbel-Softmax 可微采样层掩码与协同优化权重更新模拟微调，显式优化剪枝后模型的可恢复性（而非最小化剪枝后损失），在 DiT-XL 上以不到 7% 预训练成本制造浅层扩散 Transformer，实现 2× 加速且 FID 仅 2.86。
tags:
  - CVPR 2025
  - 图像生成
  - Transformer
  - 深度剪枝
  - 可学习压缩
  - 可恢复性
  - Gumbel-Softmax
---

# TinyFusion: Diffusion Transformers Learned Shallow

**会议**: CVPR 2025  
**arXiv**: [2412.01199](https://arxiv.org/abs/2412.01199)  
**代码**: [GitHub](https://github.com/VainF/TinyFusion)  
**领域**: 图像生成/模型压缩  
**关键词**: 扩散Transformer, 深度剪枝, 可学习压缩, 可恢复性, Gumbel-Softmax

## 一句话总结

提出 TinyFusion，一种可学习的深度剪枝方法，通过 Gumbel-Softmax 可微采样层掩码与协同优化权重更新模拟微调，显式优化剪枝后模型的可恢复性（而非最小化剪枝后损失），在 DiT-XL 上以不到 7% 预训练成本制造浅层扩散 Transformer，实现 2× 加速且 FID 仅 2.86。

## 研究背景与动机

- **扩散 Transformer 的推理负担**：DiT、MAR、SiT 等模型参数规模巨大，部署成本高。
- **深度剪枝的优势**：相比宽度剪枝（50% 宽度仅 1.6× 加速），深度剪枝可实现与压缩率成线性比例的加速（50% 深度 = 2× 加速），且在并行和非并行设备上均有效。
- **损失最小化原则的缺陷**：传统剪枝（包括特征相似度、灵敏度分析）追求剪枝后损失最低，但实验发现剪枝后低损失并不等于微调后高性能——对扩散 Transformer 该原则不成立。
- **可恢复性才是关键**：10 万随机采样模型的分析显示，剪枝后损失低的模型微调后性能反而未必好；应优化的是"微调后能恢复到多好"。
- **可微搜索的困难**：层选择是离散的非可微操作；可恢复性评估需要实际微调，搜索空间巨大（如 $\binom{28}{14} = 4000$ 万种组合）。

## 方法详解

### 整体框架

TinyFusion 将 $L$ 层 Transformer 分为 $K$ 个局部块，每块用 N:M 方案保留 $N$ 个层。将层掩码建模为 Gumbel-Softmax 可微采样的分类分布：$p(\mathfrak{m}) = p(\mathfrak{m}_1) \cdot p(\mathfrak{m}_2) \cdots p(\mathfrak{m}_K)$。同时学习权重更新 $\Delta\Phi$（用 LoRA 实现）模拟微调效果。目标：$\min_{\mathfrak{m}} \min_{\Delta\Phi} \mathbb{E}_x[\mathcal{L}(x, \Phi + \Delta\Phi, \mathfrak{m})]$。

### 关键设计

**设计一：可恢复性导向的优化目标**
- **功能**：找到微调后能恢复到最佳性能的剪枝方案
- **核心思路**：区别于传统 $\min_{\mathfrak{m}} \mathbb{E}[\mathcal{L}(x, \Phi, \mathfrak{m})]$（最小化剪枝后损失），TinyFusion 引入 $\Delta\Phi$ 模拟微调：$\min_{\mathfrak{m}} \min_{\Delta\Phi} \mathbb{E}[\mathcal{L}(x, \Phi + \Delta\Phi, \mathfrak{m})]$。用 LoRA 作为轻量的 $\Delta\Phi$ 代理，与掩码采样联合优化。
- **设计动机**：实证发现剪枝后损失与微调后性能无显著相关性——ShortGPT 等方法选择的低损失方案微调后 FID 反而很差（22.28 vs TinyFusion 的 5.73）。

**设计二：Gumbel-Softmax 可微层掩码采样**
- **功能**：使离散的层选择可通过梯度下降优化
- **核心思路**：将模型分为 $K$ 个块，每块枚举所有 $\binom{M}{N}$ 种 N:M 掩码，用 Gumbel-Softmax + STE 实现可微采样：$y = \text{one-hot}(\exp((g_i + \log p_i)/\tau) / \sum_j \exp((g_j + \log p_j)/\tau))$，$\mathfrak{m} = y^\top \hat{\mathfrak{m}}$。温度 $\tau$ 从高到低退火，从探索走向收敛。
- **设计动机**：直接枚举搜索空间太大；概率建模让搜索变为分布优化——正反馈的采样模式获得更高概率，逐步收敛到最优。

**设计三：MaskedKD — 掩码知识蒸馏**
- **功能**：增强剪枝模型微调时的性能恢复
- **核心思路**：以原始未剪枝模型为教师，蒸馏到浅层模型。关键改进：对隐藏层中的异常激活（massive/outlier activations）施加掩码，避免它们负面影响微调稳定性和蒸馏效果。
- **设计动机**：扩散 Transformer 中存在的 outlier 激活在蒸馏时会被放大，掩码处理后 FID 从 5.73 改善到 3.73。

### 损失函数

标准扩散损失 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$，搜索阶段同时优化掩码分布参数和 LoRA 权重。微调阶段可选标准重训或 MaskedKD。

## 实验关键数据

### DiT-XL/2 深度剪枝 (28→14 层, 50% 压缩)

| 方法 | FID ↓ | 采样速度 (it/s) ↑ | 微调成本 |
|------|------|-----------------|---------|
| DiT-XL/2 原始 (28层) | 2.27 | 6.91 | 7000K iters |
| ShortGPT (28→14) | 22.28 | 13.54 | 100K iters |
| Flux-Lite (28→14) | 25.92 | 13.54 | 100K iters |
| Sensitivity (28→14) | 21.15 | 13.54 | 100K iters |
| **TinyFusion (28→14)** | **5.73** | **13.54** | **100K iters** |
| TinyFusion + MaskedKD | **3.73** | 13.54 | 100K iters |
| TinyFusion + MaskedKD (500K) | **2.86** | 13.54 | 500K iters |

### 不同模型的泛化性

| 模型 | 压缩方式 | 结果 |
|------|---------|------|
| DiT-XL/2 | 28→14层 | FID 2.86 @ 2× 加速 |
| MAR | 深度剪枝 | 有效 |
| SiT | 深度剪枝 | 有效 |

### 关键发现

1. TinyFusion 100K iters 微调后 FID 5.73，远优于 ShortGPT 的 22.28，相同剪枝量相同微调预算
2. 500K iters（仅 7% 预训练成本）后 FID 降至 2.86，仅比原始 28 层模型高 0.59
3. MaskedKD 将 FID 从 5.73 降至 3.73——处理 outlier 激活对蒸馏至关重要
4. 50% 宽度剪枝仅 1.6× 加速 vs 50% 深度剪枝 2× 加速——深度剪枝在实际设备上加速更有效
5. 方法泛化到 DiT、MAR、SiT 三种不同架构

## 亮点与洞察

- **挑战了"最小化剪枝后损失"的传统范式**：通过大规模实证分析说服力强，为剪枝社区提供了新的优化目标
- **LoRA 作为微调代理**：用低秩更新模拟实际微调效果，使可恢复性可在搜索阶段高效评估
- **局部 N:M 方案**：将全局搜索空间分解为独立的局部搜索，保留了有价值的局部结构模式
- 仅 1 epoch 即可找到合适方案——搜索本身的成本极低

## 局限与展望

- N:M 方案假设每块保留相同比例层数，可能不是全局最优解——某些块可能更需要保留更多层
- 当前仅在 ImageNet 256×256 条件生成上验证，对更高分辨率和文本条件生成的适用性未知
- MaskedKD 的掩码阈值策略仍为启发式（基于激活值的标准差倍数），可能需要针对不同模型调优
- LoRA 作为微调代理虽然高效，但不能完全代表全参数微调的恢复行为

## 相关工作与启发

- **vs ShortGPT / Flux-Lite**: 基于特征相似度的启发式方法虽能找到低校准损失的方案，但微调后 FID 远差于 TinyFusion（22+ vs 5.73），证实了可恢复性 ≠ 低校准损失
- **vs Diff-Pruning (宽度剪枝)**: 50% 宽度剪枝仅 1.6× 加速且 FID 3.85，TinyFusion 50% 深度剪枝实现 2× 加速且 FID 2.86，深度剪枝在实际设备上更有效
- 可恢复性导向的剪枝思路可推广到 LLM 等其他大模型的深度压缩
- Gumbel-Softmax + 协同权重优化范式可用于任何离散结构搜索问题

## 评分

⭐⭐⭐⭐ — 深刻的洞察（可恢复性 vs 剪枝后损失）、优雅的方法设计（可微采样 + LoRA 代理）、强大的实验结果（2× 加速 FID 2.86）。对扩散模型压缩领域有重要贡献。

<!-- RELATED:START -->

## 相关论文

- [Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [Scaling Diffusion Transformers Efficiently via μP](../../NeurIPS2025/image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)
- [Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)

<!-- RELATED:END -->
