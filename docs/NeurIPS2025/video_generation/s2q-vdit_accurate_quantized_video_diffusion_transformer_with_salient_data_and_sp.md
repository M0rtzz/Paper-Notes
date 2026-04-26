---
title: >-
  [论文解读] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
description: >-
  [NeurIPS 2025][训练后量化] 针对视频扩散 Transformer 的超长 token 序列导致的量化校准高方差和学习困难问题，提出 S²Q-VDiT 框架，利用 Hessian 感知的显著数据选择和注意力引导的稀疏 token 蒸馏两项技术，首次在 W4A6 设置下实现无损量化，带来 3.9× 模型压缩和 1.3× 推理加速。
tags:
  - NeurIPS 2025
  - 训练后量化
  - 视频扩散模型
  - 校准数据选择
  - 稀疏注意力
  - 模型压缩
---

# S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2508.04016](https://arxiv.org/abs/2508.04016)  
**代码**: [GitHub](https://github.com/wlfeng0509/s2q-vdit)  
**领域**: 扩散模型 / 视频生成 / 模型压缩  
**关键词**: 训练后量化, 视频扩散模型, 校准数据选择, 稀疏注意力, 模型压缩

## 一句话总结

针对视频扩散 Transformer 的超长 token 序列导致的量化校准高方差和学习困难问题，提出 S²Q-VDiT 框架，利用 Hessian 感知的显著数据选择和注意力引导的稀疏 token 蒸馏两项技术，首次在 W4A6 设置下实现无损量化，带来 3.9× 模型压缩和 1.3× 推理加速。

## 研究背景与动机

视频扩散 Transformer（如 CogVideoX、HunyuanVideo）已成为视频生成的主流范式，但动辄数十亿参数带来了巨大的计算和存储开销。训练后量化（PTQ）是一种高效的模型压缩手段，但将其直接应用于视频扩散模型（V-DMs）时面临严重的性能退化。

作者发现了两个 V-DMs 特有的量化挑战：

**校准数据方差大**：V-DMs 的时空联合建模产生了极长的 token 序列（如6秒视频有 $s \times 49$ 个 token），在相同计算预算下，V-DMs 只能使用几十个校准样本（对比图像模型的数千个）。在如此有限的数据下，不同的校准样本选择会导致量化性能巨大差异。

**token 学习困难**：V-DMs 的全时空注意力展现出稀疏模式——仅有少量 token 显著影响最终输出。但现有 PTQ 方法对所有 token 一视同仁地对齐全精度和量化输出，在长序列场景下效率低下。

此前的工作（Q-DiT、ViDiT-Q）主要从量化器设计的角度改进，而本文从**校准数据质量**和**优化策略**这两个新角度切入。

## 方法详解

### 整体框架

S²Q-VDiT 包含两个核心模块：（1）Hessian 感知显著数据选择（SDS），为量化校准构建高质量数据集；（2）注意力引导的稀疏 token 蒸馏（STD），在逐块优化中根据 token 重要性进行加权学习。

### 关键设计

1. **Hessian 感知显著数据选择（SDS）**：从两个维度评估校准样本的重要性。**扩散显著度** $C_{\text{diff}} = \|x_t - x_{t-1}\|^2 / \|x_t\|^2$ 度量相邻时间步特征差异，值越大表示该时间步包含越多去噪信息。**量化敏感度** $C_{\text{quant}} = \|x_t^\top x_t\|_2$ 基于 Hessian 矩阵的 Levenberg-Marquardt 近似 $H^X = \mathbb{E}[2X^\top X]$，值越大表示该样本对量化扰动越敏感。两个指标分别做 min-max 归一化后相乘得到统一评分 $C_{\text{sample}} = \bar{C}_{\text{diff}} \cdot \bar{C}_{\text{quant}}$，利用 AM-GM 不等式确保仅当两个维度都高时总分才高，避免单维度偏高的偏颇样本。

2. **注意力引导的稀疏 token 蒸馏（STD）**：利用 V-DMs 中全时空注意力的固有稀疏性，对不同 token 的量化损失进行加权：$\mathcal{L}_{\text{quant}} = \frac{1}{n}\sum_{j=1}^n \lambda_j \|\theta^f(x_{j,:}) - \theta^q(x_{j,:})\|^2$。权重 $\lambda_j$ 由多头注意力图计算：先对所有头和 query 位置求和得到 $S_j = \sum_{h,i} A_{h,i,j}$，再做 min-max 归一化到 $[\lambda_{\min}, \lambda_{\max}]$ 范围。这使优化过程聚焦于对输出影响大的少量关键 token，放松对不重要 token 的约束，提升了有限校准数据下的收敛质量。

### 损失函数 / 训练策略

- 采用逐块PTQ优化策略，对每个 Transformer 块分别校准量化参数
- 权重使用 per-channel 对称量化，激活使用 per-token 动态量化
- 校准数据量统一选择40个样本，兼顾性能和校准时间
- STD 的 $\lambda_{\min} = 0.5, \lambda_{\max} = 1.0$ 在主实验中使用

## 实验关键数据

### 主实验

W4A6 量化在 VBench 评估基准上的结果：

| 模型 | 方法 | 图像质量(IQ) | 美学质量(AQ) | 动态度(DD) | 场景一致性(ScC) | 整体一致性(OC) |
|------|------|:--------:|:-------:|:------:|:---------:|:---------:|
| CogVideoX-2B | FP | 58.69 | 55.25 | 50.00 | 33.79 | 25.91 |
| CogVideoX-2B | ViDiT-Q | 51.94 | 48.06 | 33.33 | 22.17 | 23.69 |
| CogVideoX-2B | **S²Q-VDiT** | **55.49** | **53.74** | **40.28** | **32.70** | **25.19** |
| HunyuanVideo | FP | 62.30 | 62.49 | 56.94 | 33.36 | 26.85 |
| HunyuanVideo | ViDiT-Q | 52.21 | 58.38 | 41.67 | 23.69 | 26.15 |
| HunyuanVideo | **S²Q-VDiT** | **58.83** | **59.62** | **48.61** | **33.65** | **26.91** |

W4A4 极低比特量化结果（CogVideoX-2B）：

| 方法 | IQ | AQ | DD | ScC | OC |
|------|:--:|:--:|:--:|:---:|:--:|
| FP | 58.69 | 55.25 | 50.00 | 33.79 | 25.91 |
| ViDiT-Q | 45.56 | 42.03 | 12.50 | 11.91 | 19.61 |
| **S²Q-VDiT** | **53.71** | **52.31** | **36.11** | **34.23** | **24.90** |

### 消融实验

| 组件 / 配置 | 关键指标变化 | 说明 |
|------------|-----------|------|
| ATOP (单prompt全时步) | IQ≈42-45 | 随机方法，质量差 |
| RTFP (随机时步) | IQ≈48-50 | 比ATOP好但不稳定 |
| **SDS (本文)** | **IQ≈54-56** | 显著优于所有启发式方法 |
| w/o STD | 基线 | 所有token等权 |
| STD (λ_min=0.5) | +2-3 IQ | token加权提升一致性 |
| 校准数据20→40 | IQ +2, OC +0.5 | 40个样本性价比最优 |
| 校准数据40→80 | 几乎无提升 | 边际收益递减 |

### 关键发现

- 在 W4A6 下三个不同规模（2B/5B/13B）的 V-DMs 上均实现接近无损的量化性能
- CogVideoX-5B 的场景一致性量化后 (46.66) 甚至超过全精度 (45.28)
- W4A4 是首次对4比特激活量化的探索，S²Q-VDiT 仍能保持约95%的模型性能
- 校准时间成本增加很小：比 PTQ4DiT 仅多约0.2小时和2GB显存
- 3.94× 模型存储压缩、1.56× 推理显存节省、1.28× 推理加速

## 亮点与洞察

- 从数据和优化策略角度解决量化问题是一个被忽视但有效的方向
- SDS 的设计思路巧妙：同时考虑扩散过程信息量和量化敏感度两个正交维度，用 AM-GM 不等式保证联合最优
- STD 利用注意力图的现有信息（零额外计算）指导 token 重要性，实用且高效
- 首次系统性地探索了 V-DMs 的 W4A4 量化设置

## 局限与展望

- 注意力稀疏模式假设可能不适用于所有 V-DM 架构
- SDS 需要预先计算所有候选样本的显著度分数，增加了前期开销
- 当前仅验证了 CogVideoX 和 HunyuanVideo 两个系列模型
- 更激进的量化（如 W2A4）场景下性能如何有待探索
- STD 的 $\lambda_{\min}$ 超参数需要调节，虽然实验显示对此不敏感

## 相关工作与启发

- 与 Q-DiT 和 ViDiT-Q 互补：它们改进量化器设计，本文改进数据和优化策略
- SDS 中扩散显著度的设计灵感来自时间步蒸馏和缓存文献中"跳过连续时间步影响有限"的观察
- Hessian 近似借鉴了 GPTQ 等 LLM 量化工作的最优权重搜索理论
- 可启发将类似的数据选择策略应用于图像扩散模型量化或其他模型压缩场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 新角度切入 V-DM 量化，两个技术点设计合理  
- **实验充分度**: ⭐⭐⭐⭐⭐ 多模型、多比特、完整消融、效率分析全面  
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，观察→方法→实验逻辑连贯  
- **价值**: ⭐⭐⭐⭐ 实际部署价值高，3.9×压缩+无损性能对视频生成落地意义重大

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](../../CVPR2025/video_generation/articulated_kinematics_distillation_from_video_diffusion_models.md)

<!-- RELATED:END -->
