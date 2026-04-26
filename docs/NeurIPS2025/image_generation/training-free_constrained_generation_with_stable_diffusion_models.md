---
title: >-
  [论文解读] Training-Free Constrained Generation with Stable Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][约束生成] 提出一种无需重新训练的约束生成方法，通过在 Stable Diffusion 的反向去噪过程中嵌入近端 Langevin 动力学（Proximal Langevin Dynamics），将图像空间中的约束通过解码器反向传播到潜空间，实现对生成输出的严格约束满足。
tags:
  - NeurIPS 2025
  - 图像生成
  - 约束生成
  - 扩散模型
  - 近端映射
  - 潜空间校正
  - 无训练
---

# Training-Free Constrained Generation with Stable Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.05625](https://arxiv.org/abs/2502.05625)  
**代码**: [GitHub](https://github.com/RAISELab-atUVA/Constrained-Stable-Diffusion)  
**领域**: image_generation  
**关键词**: 约束生成, Stable Diffusion, 近端映射, 潜空间校正, 无训练

## 一句话总结

提出一种无需重新训练的约束生成方法，通过在 Stable Diffusion 的反向去噪过程中嵌入近端 Langevin 动力学（Proximal Langevin Dynamics），将图像空间中的约束通过解码器反向传播到潜空间，实现对生成输出的严格约束满足。

## 研究背景与动机

扩散模型在科学工程领域需要生成满足严格约束（物理定律、安全标准、设计规范）的输出，但现有方法存在明显局限：

**训练时约束**：仅提供分布级别的约束遵守，无法保证每个样本都满足约束，且无法泛化到未见约束

**推理时约束**（如 Projected Diffusion）：直接在图像空间修改反向过程，**与 Stable Diffusion 等潜空间扩散模型不兼容**，因为约束无法在潜空间中直接表示

**潜空间变体**：依赖特殊测量算子或学习的软惩罚，通用性有限

**核心挑战**：约束定义在图像空间，但 Stable Diffusion 的去噪过程发生在潜空间，两者之间存在鸿沟。

## 方法详解

### 整体框架

在 Stable Diffusion 的每一步去噪迭代中：
1. **Langevin 动力学步**：执行标准去噪更新，得到预校正潜变量 z'_t
2. **近端映射步**：通过解码器 D 将 z'_t 映射到图像空间，评估约束违反程度，计算梯度并反向传播回潜空间，迭代调整潜变量直到约束满足
3. 无需修改 score 网络或解码器，不增加可学习参数

### 关键设计

1. **潜空间到图像空间的约束传递**：核心洞察是虽然约束无法在潜空间直接表示，但可以在去噪的任何阶段通过解码器评估。梯度通过链式法则 ∇_{z_t}g = (∂D/∂z_t)^T · ∇_{x_t}g 从图像空间反传到潜空间。解码器 D 作为冻结的可微映射桥梁。

2. **近端 Langevin 动力学**：将投影 Langevin 动力学推广为近端映射形式 prox_{λg}(z_t) = argmin_y {g(D(y)) + 1/(2λ)·‖D(y)-D(z_t)‖²}。近端算子平衡约束满足和与更新样本的相似性。当约束为集合指示函数时退化为标准投影；但近端映射可处理非光滑正则化、复合惩罚和隐式约束。

3. **复杂约束处理**：

    - **可微代理模型**：用预训练分类器替代不可直接表达的约束函数（如版权检测）
    - **黑盒模拟器微分**：借鉴可微扰动优化器（DPO），通过注入随机扰动和有限差分估计梯度，使不可微的物理模拟器也能参与优化。平滑函数 φ̄_ν(x) = E[φ(x+νε)]，梯度通过 Monte Carlo 估计

### 损失函数 / 训练策略

**内部最小化器**（proximal mapping 求解）：
对近端目标函数进行梯度下降：z_t^{i+1} = z_t^i - ∇_{z_t^i}[g(D(z_t^i)) + 1/(2λ)·‖D(z_t^i)-D(z_t^0)‖²]

**外部最小化器**（整个采样过程）：
在每个去噪步完成标准更新后，迭代执行内部最小化直到约束违反 g(D(z_t)) < δ

**收敛保证**（凸约束情形）：
- Theorem 4.1：到可行集的距离以 (1-2β'γ_{t+1}) 的速率递减
- Theorem 4.2：与训练分布的 KL 散度增长最多 O(Σ_t γ_t)，随 γ_t→0 趋于可忽略

## 实验关键数据

### 主实验

三个应用场景验证方法的通用性：

| 任务 | 指标 | 本文 (Latent) | PDM | Cond | 提升 |
|------|------|--------------|-----|------|------|
| 微结构生成 (P=30%) | FID ↓ | 13.5±3.1 | 30.7±6.8 | **10.8±0.9** | FID vs PDM -56% |
| 微结构生成 (P=30%) | 约束违反>10% ↓ | **0%** | **0%** | 68.4% | 完美约束满足 |
| 超材料逆设计 | MSE ↓ | **1.4±0.6** | N/A | 7.1±4.5 | MSE 降低 80% |
| 超材料逆设计 | 物理无效率 ↓ | **5%** | N/A | 55% | 大幅减少无效生成 |
| 版权安全生成 | 约束满足率 ↑ | **90%** | 71% | 67% | 最高约束遵守 |
| 版权安全生成 | FID ↓ | **65.1** | 75.3 | 61.2 | 质量与约束兼顾 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 微结构 P=50% vs P=30% | 均 0% 违反 | 不同约束值下一致有效 |
| DPO 步数 0→5 | MSE: 179.5→1.2 | 每步迭代显著改善约束满足 |
| Bastek & Kochmann (SOTA) | MSE: 6.4±4.6 | 本文方法实现 4.6x 改进 |
| 高分辨率 1024² | PDM 无法处理 | 潜空间方法天然支持高分辨率 |

### 关键发现

- **约束满足与生成质量可以兼顾**：FID 分数与无约束基线接近，理论上由 Theorem 4.2 保证
- **通用性强**：同一算法框架处理凸约束（孔隙率）、黑盒模拟器约束（应力-应变）和代理约束（版权检测）
- **DPO 迭代可将误差降至任意低**：与基线的固定误差形成鲜明对比
- **潜空间方法天然支持高分辨率**：PDM 在 1024² 分辨率下失败，本文方法正常工作

## 亮点与洞察

- **简洁的核心思想**：通过冻结解码器作为可微桥梁，将图像空间约束无缝传递到潜空间，无需修改任何网络
- **理论保证扎实**：凸约束情形下有收敛性和分布保真度的双重保证
- **黑盒模拟器处理**：DPO 方法使得任何可查询的模拟器都能集成到约束优化中，大大拓展了适用范围
- **首个将约束优化集成到 Stable Diffusion 采样过程的工作**

## 局限与展望

- 每步去噪需要额外的内部优化迭代，增加推理时间
- 依赖解码器 D 的可微性和 Lipschitz 连续性假设
- 非凸约束下理论保证弱化，仅有经验验证
- DPO 方法需要多次模拟器调用（M=10），复杂模拟器下成本高
- 版权检测场景中 10% 的违反率受限于分类器精度

## 相关工作与启发

- **PDM (Christopher et al.)**：图像空间的投影扩散模型，是本文的直接前身，但不兼容潜空间扩散
- **Classifier Guidance**：本文方法与分类器引导有区别——后者是软引导，本文是硬约束满足
- **DPO (Differentiable Perturbed Optimizer)**：从可微优化领域借鉴的梯度估计技术，使黑盒模拟器可用于优化
- 方法可推广到任何基于潜空间的生成模型（如 VAE、流模型）的约束生成场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 潜空间约束传递思路简洁有效，DPO 集成是创新点
- 实验充分度: ⭐⭐⭐⭐⭐ 三个差异化应用场景，凸/非凸/黑盒约束全覆盖
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验展示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了 Stable Diffusion 约束生成的关键问题，工程和科学应用价值高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Safe and Stable Control via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](../../CVPR2025/image_generation/enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)

<!-- RELATED:END -->
