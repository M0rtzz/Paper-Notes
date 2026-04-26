---
title: >-
  [论文解读] Optimizing for the Shortest Path in Denoising Diffusion Model
description: >-
  [CVPR 2025][图像生成][扩散模型加速] 将扩散模型的去噪过程建模为图论中的最短路径问题，通过优化初始残差来压缩反向扩散路径，实现用 2 步采样即可达到甚至超越 DDIM 10 步的生成质量。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型加速
  - 最短路径优化
  - 残差传播
  - 图论
  - DDIM
---

# Optimizing for the Shortest Path in Denoising Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.03265](https://arxiv.org/abs/2503.03265)  
**代码**: [GitHub](https://github.com/UnicomAI/ShortDF)  
**领域**: 图像生成/扩散模型加速  
**关键词**: 扩散模型加速, 最短路径优化, 残差传播, 图论, DDIM

## 一句话总结

将扩散模型的去噪过程建模为图论中的最短路径问题，通过优化初始残差来压缩反向扩散路径，实现用 2 步采样即可达到甚至超越 DDIM 10 步的生成质量。

## 研究背景与动机

扩散模型在图像生成中取得了卓越性能，但其迭代采样过程导致高计算成本，限制了实时应用。

现有加速方法的局限：
- **蒸馏方法**（PD, InstaFlow 等）：将多步过程压缩到单步学生网络，但高维复杂采样过程的精确近似困难
- **快速采样器**（DDIM, DPM-Solver 等）：通过更好的数值求解器减少步数，但进一步减少步数时质量下降明显
- 核心矛盾在于快速采样和高质量生成之间的权衡

作者观察到 DDIM 采样过程中存在残差传播现象：每步的估计误差会累积并传播到后续步骤。如果能优化初始残差（第一步的估计误差），就能减少整条路径的累计误差。这与图论中的最短路径问题高度类似。

## 方法详解

### 整体框架

ShortDF 基于 DDIM 框架，将反向扩散的每一步视为图的节点，步间转移的误差视为边权重。通过构建反向步图（reverse-step graph）并应用最短路径松弛优化，找到从 $x_T$ 到 $x_0$ 的最优（最低误差）路径。

### 关键设计一：残差传播分析（Residual Propagation）

- **功能**：将扩散过程的多步误差形式化为可优化的路径问题
- **核心思路**：定义初始残差 $R(k_1, 0) = \frac{\sqrt{1-\bar{\alpha}_{k_1}}}{\sqrt{\bar{\alpha}_{k_1}}}(\epsilon_\theta(x_{k_1}, k_1) - \epsilon)$，推导出路径残差 $R(k_n, 0) = R(k_1, 0) - \sum_{i=1}^{n-1} R(k_i, k_{i+1})$。优化全路径残差等价于优化初始残差
- **设计动机**：直接优化多步累计残差不可行，但更小的初始残差会使后续所有步骤受益，因为每步都基于前一步的估计构建。这为路径优化提供了理论支撑

### 关键设计二：图构建与最短路径松弛（Graph Construction & Relaxation）

- **功能**：通过松弛条件动态选择最优转移路径
- **核心思路**：定义边权重 $\text{edge}(k,t) = |x_0 - \hat{x}'_{0|k}| - |x_0 - \hat{x}_{0|k}|$，使用真实 $x_0$ 消除路径残差，使步 $k$ 的最优解不受步 $t$ 的残差干扰。当松弛条件 $dist(x_t, t) > dist(x_k, k) + edge(k,t)$ 成立时，更新路径选择
- **设计动机**：直接用残差定义边权重会引入步间干扰使最优传播不可追踪。通过引入真实 $x_0$ 作为参考消除路径残差，再用图论松弛方法逐步优化

### 关键设计三：多状态优化策略（Multi-State Optimization）

- **功能**：稳定训练并实现图论优化的实际落地
- **核心思路**：维护三个模型实例——Base Model $\epsilon_\theta$ 处理噪声预测和残差更新；EMA Model $\epsilon_{\theta,ema}$ 提供步 $k$ 的稳定最优估计；Graph Model $\epsilon_{\theta,graph}$ 计算边权重，延迟更新捕捉全局信息
- **设计动机**：随机噪声导致的训练不稳定使直接优化困难。三模型各司其职——基础模型学习，EMA 模型提供稳定目标，图模型提供一致的边权重

### 损失函数

总损失 $L = \lambda \cdot L_\epsilon + cond \cdot L_r$，其中 $L_\epsilon$ 为标准噪声预测损失，$L_r = \|dist(x_k, k) + edge(k, t) - dist(x_t, t)\|_2$ 为松弛损失。$cond$ 为松弛条件的指示函数，仅在条件满足时激活路径优化。

## 实验关键数据

### 主实验：CIFAR-10 FID 对比

| 方法 | 1步 | 2步 | 3步 | 4步 | 5步 | 10步 |
|------|-----|-----|-----|-----|-----|------|
| DDIM | >100 | >100 | 123.54 | 66.13 | 26.91 | 11.14 |
| DPM-solver | - | - | 90.38 | 33.29 | 23.31 | 5.09 |
| DPM-solver++ | - | - | 107.02 | 30.46 | 18.87 | 7.83 |
| SDDPM | - | - | - | 19.20 | - | - |
| **ShortDF** | - | **9.08** | - | - | - | - |

### 效率对比

| 指标 | DDIM (10步) | ShortDF (2步) |
|------|-----------|-------------|
| FID | 11.14 | **9.08** |
| 速度提升 | 1.0x | **5.0x** |
| 每步时间 | 1.2ms | 1.2ms |

### 关键发现

- 2 步 ShortDF (FID=9.08) 超越 10 步 DDIM (FID=11.14)，质量提升 18.5%，速度提升 5 倍
- 在 CelebA 和 LSUN Churches 上同样表现优异
- 路径优化在 2 步时优势最明显，因为此时初始残差对最终结果影响最大
- 去掉图建模后退化为 DDIM，验证了最短路径优化的有效性

## 亮点与洞察

1. **图论视角新颖**：首次将扩散模型的反向过程形式化为带权图的最短路径问题，理论框架优雅
2. **路径压缩的直觉清晰**：通过迭代松弛，将 $x_0 \to x_k \to x_t$ 的多跳路径压缩为 $x_0 \to x_t$ 的直达路径
3. **保持采样器通用性**：不修改网络结构，仅在训练时引入额外损失，推理时使用标准 DDIM 采样

## 局限与展望

- 需要重新训练扩散模型，不能直接应用于已训练好的预训练模型
- 多状态优化策略增加了训练开销（需维护三个模型副本）
- 在文本到图像等复杂条件生成任务上的验证有限
- 理论最优路径与实际近似之间的差距缺乏定量分析

## 相关工作与启发

- **DDIM**：ShortDF 的基础框架，通过灵活采样路径加速
- **DPM-Solver**：利用 ODE 半线性结构的专用求解器
- **RDDM**：将去噪分解为残差扩散和噪声扩散，与 ShortDF 的残差分析思路有共鸣
- 最短路径优化的思想或可推广到其他迭代生成模型

## 评分

⭐⭐⭐⭐ — 图论视角独特，理论推导严谨，2 步超越 10 步 DDIM 的结果令人印象深刻。但需要重训练且对预训练模型不友好是较大局限。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2025\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](../../ICCV2025/image_generation/fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)

<!-- RELATED:END -->
