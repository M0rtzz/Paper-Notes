---
title: >-
  [论文解读] LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal
description: >-
  [ICCV 2025][自动驾驶][镜头光斑去除] 提出 LightsOut，一个基于扩散模型的图像外推框架，通过预测和重建画面外的光源来增强现有单图光斑去除(SIFR)方法的性能，作为即插即用的预处理模块无需额外训练即可提升任意 SIFR 模型的效果。
tags:
  - ICCV 2025
  - 自动驾驶
  - 镜头光斑去除
  - 扩散模型
  - 图像外推
  - LoRA微调
  - 即插即用
---

# LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal

**会议**: ICCV 2025  
**arXiv**: [2510.15868](https://arxiv.org/abs/2510.15868)  
**代码**: https://ray-1026.github.io/lightsout/  
**领域**: 自动驾驶 / 图像复原  
**关键词**: 镜头光斑去除, 扩散模型, 图像外推, LoRA微调, 即插即用

## 一句话总结

提出 LightsOut，一个基于扩散模型的图像外推框架，通过预测和重建画面外的光源来增强现有单图光斑去除(SIFR)方法的性能，作为即插即用的预处理模块无需额外训练即可提升任意 SIFR 模型的效果。

## 研究背景与动机

镜头光斑(lens flare)严重降低图像质量，影响目标检测和自动驾驶等计算机视觉任务。现有的深度学习 SIFR 方法在完整光源可见时效果良好，但在画面外光源不完整或缺失时性能严重退化。关键观察：**完整的光源信息对有效去除光斑至关重要**。当光源被裁切到画面之外，现有方法缺少足够的上下文信息来理解和去除光斑伪影。

作者通过实验验证了这一核心动机：在"无光源"和"不完整光源"两种场景下，PSNR、LPIPS 等指标显著下降，证明光源完整性对光斑去除的关键作用。

## 方法详解

### 整体框架

LightsOut 采用**三阶段流水线**：
1. **光源预测与条件生成**：预测画面外光源参数并生成光源掩码
2. **光源外推**：使用 LoRA 微调的扩散模型外推补全缺失光源区域
3. **SIFR 增强**：将外推图像输入现有 SIFR 模型进行去光斑处理

### 关键设计

1. **多任务回归模块 (Multitask Regression Module)**：

    - 预测画面外光源的参数化表示，将光源建模为圆形实体 $(x, y, r)$
    - 同时预测 $N$ 组光源物理参数 $\mathbf{P} \in \mathbb{R}^{N \times 3}$ 和置信度分数 $\mathbf{c} \in [0,1]^{N \times 1}$
    - 采用 CNN 特征提取器 + 双 MLP 头架构（一个预测位置参数，一个预测置信度）
    - 使用二分匹配策略处理预测与真值的排列不变性
    - 渲染函数通过 Sigmoid 激活和置信度阈值生成最终光源掩码 $M_L$：
    $M_L(x,y) = \sum_{i=1}^{N} \tilde{c}_i \cdot \sigma(r_i - \sqrt{(x-x_i)^2 + (y-y_i)^2})$

2. **LoRA 微调扩散外推模型**：

    - 基于 Stable Diffusion v2 Inpainting 模型，注入 LoRA 权重进行高效微调
    - 训练损失：$\mathcal{L} = \mathbb{E}_{x,t,\epsilon,m} \|\epsilon_\theta(x_t, t, p, M, I_M) - \epsilon\|_2^2$
    - 使用 BLIP-2 自动生成文本提示作为条件信号
    - 推理时在 RGB 空间进行 Alpha 合成（而非隐空间混合），避免保留区域失真

3. **噪声重注入 (Noise Reinjection)**：

    - 解决去噪过程中遮蔽/非遮蔽区域独立处理导致的不一致问题
    - 在中间步骤重新引入噪声，让模型重新去噪以更好地对齐分布
    - 重复 $R$ 次噪声重注入操作，确保生成区域与原始区域的视觉连贯性

4. **光源条件模块 (Light Source Condition Module)**：

    - 利用预测的光源掩码 $M_L$ 指导外推过程
    - 通过可学习机制条件化生成过程，确保光源在物理合理的位置生成
    - 使用 L2 损失约束：$\mathcal{L}_{\text{light}} = \|\tilde{M}_L - M_L\|_2^2$

### 损失函数 / 训练策略

多任务回归模块采用不确定性感知加权机制平衡多个损失：
$$\mathcal{L} = \frac{1}{2\sigma_1^2}\mathcal{L}_{\text{pos}} + \frac{1}{2\sigma_2^2}\mathcal{L}_{\text{conf}} + \log(1+\sigma_1^2) + \log(1+\sigma_2^2)$$

其中 $\mathcal{L}_{\text{pos}}$ 使用 Smooth L1 损失监督位置参数，$\mathcal{L}_{\text{conf}}$ 使用二元交叉熵监督置信度。三个模块独立训练：
- 多任务回归模块：lr=1e-4, batch=32, 100 epochs, N=4
- 光源条件模块：lr=1e-5, batch=8, 20K steps
- LoRA 微调扩散模型：lr=1e-4, batch=8, 25K steps

## 实验关键数据

### 主实验

在 Flare7K 数据集（100 张真实图 + 100 张合成图）上评估，两种场景：无光源 / 不完整光源。

| 场景 | SIFR模型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|------|-------|-------|--------|
| 无光源 | Flare7k++ | Direct input | 26.29 | 0.8337 | 0.0442 |
| 无光源 | Flare7k++ | SD-Inpainting | 27.98 | 0.8938 | 0.0421 |
| 无光源 | Flare7k++ | PowerPaint | 27.10 | 0.8814 | 0.0839 |
| 无光源 | Flare7k++ | **Ours** | **28.41** | **0.8956** | **0.0397** |
| 不完整光源 | Flare7k++ | Direct input | 26.07 | 0.8333 | 0.0463 |
| 不完整光源 | Flare7k++ | SD-Inpainting | 28.02 | 0.8944 | 0.0431 |
| 不完整光源 | Flare7k++ | **Ours** | **28.15** | **0.8957** | **0.0409** |

核心发现：在"无光源"场景下，LightsOut 将 Flare7k++ 的 PSNR 从 26.29 dB 提升至 28.41 dB（+2.12 dB）。

### 消融实验

| 消融组件 | PSNR (Real) | SSIM | LPIPS |
|----------|-------------|------|-------|
| 无噪声重注入 | 28.28 | 0.8949 | 0.0412 |
| 有噪声重注入 | **28.41** | **0.8956** | **0.0397** |
| 隐空间混合 | 26.91 | 0.8859 | 0.0434 |
| RGB空间混合 | **27.09** | **0.8856** | **0.0424** |

组件贡献消融（SD-Inpainting → Ours）：

| LoRA | 条件模块 | PSNR↑ |
|------|----------|-------|
| ✗ | ✗ | 26.82 |
| ✓ | ✗ | 27.12 (+0.30) |
| ✗ | ✓ | 27.06 (+0.24) |
| ✓ | ✓ | **27.43** (+0.61) |

### 关键发现

- RGB 空间混合远优于隐空间混合，尤其在合成数据上（PSNR 31.55 vs 24.13）
- 噪声重注入显著提升视觉连贯性，LPIPS 从 0.0412 降至 0.0397
- LoRA 微调和光源条件模块的贡献互补，组合使用效果最佳
- 多任务回归优于 UNet 基线和可微渲染方法（mIoU 0.6310 vs 0.6216 vs 0.5212）

## 亮点与洞察

- **问题定义精准**：准确识别了 SIFR 退化的根本原因——画面外光源信息缺失
- **即插即用设计**：无需修改已有 SIFR 模型架构，可直接作为预处理增强任意方法
- **物理先验融合**：将光源建模为参数化圆形实体，引入物理约束指导扩散生成
- 下游任务验证：使用 YOLOv11 检测器验证了方法对目标检测的间接提升作用

## 局限与展望

- 三阶段流水线引入额外计算开销，可探索端到端优化
- 当整体图像亮度较高或光斑占比过大时效果受限
- 仅在 Flare7K 数据集上训练和评估，真实场景泛化性待验证
- 光源建模假设为圆形，对非规则光源可能不够准确

## 相关工作与启发

- 与 Difflare、MFDNet 等 SIFR 方法互补而非竞争
- LoRA 微调策略在特定任务扩散模型适配中的有效性值得借鉴
- 噪声重注入技术可推广到其他图像外推/补全任务
- 物理参数化光源预测思路可启发其他需要物理先验的视觉任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从外推光源角度解决光斑去除问题，视角新颖；将扩散模型与物理先验结合的设计有创意
- **实验充分度**: ⭐⭐⭐⭐ 多基线对比、多场景评估、详细消融、下游任务验证，实验较为全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法描述完整，图表直观
- **价值**: ⭐⭐⭐⭐ 即插即用设计实用性强，对自动驾驶等场景有实际意义

<!-- RELATED:START -->

## 相关论文

- [Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [Neurosymbolic Diffusion Models](../../NeurIPS2025/autonomous_driving/neurosymbolic_diffusion_models.md)
- [Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction](../../AAAI2026/autonomous_driving/global-lens_transformers_adaptive_token_mixing_for_dynamic_link_prediction.md)
- [Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

<!-- RELATED:END -->
