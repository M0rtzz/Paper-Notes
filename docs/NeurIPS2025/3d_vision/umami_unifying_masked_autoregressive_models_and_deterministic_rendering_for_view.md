---
title: >-
  [论文解读] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis
description: >-
  [NeurIPS 2025][3D视觉][新视角合成] 提出 UMAMI，一个统一掩码自回归模型（MAR）和确定性渲染的混合框架用于稀疏视角新视角合成：双向 Transformer 编码多视角图像 Token 和 Plücker 射线嵌入，两个轻量级 MLP 头分别处理可见区域（确定性回归）和遮挡区域（MAR 扩散生成），渲染速度比全生成基线快一个数量级。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 新视角合成
  - 掩码自回归
  - 扩散模型
  - 确定性渲染
  - 混合框架
---

# UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis

**会议**: NeurIPS 2025  
**arXiv**: [2512.20107](https://arxiv.org/abs/2512.20107)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 新视角合成, 掩码自回归, 扩散模型, 确定性渲染, 混合框架

## 一句话总结
提出 UMAMI，一个统一掩码自回归模型（MAR）和确定性渲染的混合框架用于稀疏视角新视角合成：双向 Transformer 编码多视角图像 Token 和 Plücker 射线嵌入，两个轻量级 MLP 头分别处理可见区域（确定性回归）和遮挡区域（MAR 扩散生成），渲染速度比全生成基线快一个数量级。

## 研究背景与动机

**领域现状**：新视角合成（NVS）两大范式——确定性方法（PixelSplat、MVSplat、LVSM）速度快但遮挡区域模糊；生成方法（CAT3D、Stable Video 3D）能幻想遮挡内容但训练推理成本高

**现有痛点**：全扩散方法用大型 UNet/Transformer 对**整张图像**迭代去噪，即使大部分目标视角已被上下文视角覆盖——这是巨大的计算浪费

**核心矛盾**：确定性方法无法生成未见区域，生成方法对已见区域效率极低

**切入角度**：将目标图像分为"确定性可渲染"（几何约束好的区域）和"需要生成"（遮挡/未见区域），分别用不同头处理

**核心 idea**：双向 Transformer 编码共享表示 → 前馈回归头直接渲染可见像素 → MAR 扩散头生成遮挡像素。端到端训练，无手工 3D 归纳偏置

## 方法详解

### 整体框架
输入：稀疏上下文图像 + 相机位姿（Plücker 射线嵌入）。将上下文图像 Token 化和目标图像的掩码 Token 拼接，送入双向 Transformer。输出的潜在表示 z 同时输入两个轻量 MLP 头：(1) 确定性头输出 RGB + 置信度图；(2) MAR 扩散头模型化遮挡区域的 Token 分布。训练时随机掩码目标图像；推理时迭代解掩，高置信区域用确定性输出、低置信区域用扩散采样。

### 关键设计

1. **数据表示——Plücker 射线嵌入**：

    - 功能：将相机位姿信息编码为逐像素的 Plücker 射线，与图像 Token 沿通道维拼接
    - 核心思路：$(\mathbf{d}, \mathbf{d} \times \mathbf{o}) \in \mathbb{R}^6$，其中 $\mathbf{d}$ 是射线方向、$\mathbf{o}$ 是原点
    - 设计动机：无需显式 3D 表示（如点云/深度），射线本身编码了足够的几何信息

2. **双头架构**：

    - **确定性头 φ**：轻量 MLP，从潜在 z 直接回归 RGB 像素值和逐像素置信度 $\mathbf{s}_p$。用 MSE + 感知损失训练
    - **MAR 扩散头 ϕ**：轻量 MLP + 时间步嵌入，对每个 Token 的潜在 z 做条件去噪。用 DDPM 目标训练
    - 分工机制：置信度 $\mathbf{s}_p$ 将 Token 分为 $\mathbf{x}_D$（确定性渲染）和 $\mathbf{x}_S$（扩散生成）
    - 形式化：$p(\mathbf{x}|\mathbf{c}) = \delta(\mathbf{x}_D - F(\mathbf{c})) \cdot p(\mathbf{x}_S | \mathbf{x}_D, \mathbf{c})$

3. **混合采样器**：

    - 功能：推理时高效合并两种头的输出
    - 核心思路：迭代解掩——每轮用确定性头填充高置信 Token（置信度 > τ），其余用扩散头解码。已填充的 Token 在下一轮作为条件
    - 设计动机：避免了全图扩散的高成本——只对遮挡部分做生成，可见部分实时渲染

4. **置信度损失**：

    - $\mathcal{L}_{conf} = \mathbf{m} \odot (\mathbf{s}_p \odot \|\hat{\mathbf{I}} - \mathbf{I}\|_2^2 - \lambda_s \cdot \log \mathbf{s}_p)$
    - 鼓励高置信度区域预测准确，同时惩罚过度自信

### 损失函数
- 总损失 = $\mathcal{L}_{render}$ (MSE + 感知) + $\mathcal{L}_{conf}$ (置信度) + $\mathcal{L}_{diff}$ (DDPM去噪)
- 所有损失端到端联合优化
- 实验使用 RealEstate10K 和 DL3DV 训练

## 实验关键数据

### 主实验 — RealEstate10K (1→1 视角合成)

| 方法 | 类型 | PSNR↑ | SSIM↑ | LPIPS↓ | 渲染时间 |
|------|------|-------|-------|--------|---------|
| PixelSplat | 确定性 | 25.9 | 0.856 | 0.143 | ~0.1s |
| MVSplat | 确定性 | 26.1 | 0.862 | 0.138 | ~0.1s |
| LVSM | 确定性 | 26.3 | 0.865 | 0.133 | ~0.1s |
| CAT3D | 全生成 | 25.8 | 0.842 | 0.158 | ~30s |
| SV3D | 全生成 | 24.5 | 0.821 | 0.192 | ~60s |
| **UMAMI** | **混合** | **26.8** | **0.872** | **0.125** | **~3s** |

### 消融实验

| 配置 | PSNR | LPIPS | 说明 |
|------|------|-------|------|
| 仅确定性头 | 26.3 | 0.133 | 遮挡区域模糊 |
| 仅MAR扩散头 | 25.5 | 0.145 | 可见区域不够精确 |
| **UMAMI混合** | **26.8** | **0.125** | 最优 |
| 无置信度分割 | 26.1 | 0.135 | 不知道哪些区域该生成 |

### 多视角输入实验 (3→1, 6→1)

| 输入视角数 | 方法 | PSNR↑ |
|-----------|------|-------|
| 3 | MVSplat | 28.5 |
| 3 | **UMAMI** | **29.3** |
| 6 | MVSplat | 30.1 |
| 6 | **UMAMI** | **30.7** |

### 关键发现
- UMAMI 在 PSNR/SSIM/LPIPS 上均优于纯确定性和纯生成方法——混合策略的价值得到验证
- 渲染速度比全生成基线（CAT3D）快 **10 倍**——因为只对遮挡部分做扩散
- 外推（extrapolation）场景改进最大——因为外推有更多未见区域需要生成
- 置信度预测准确反映了"可见性"——高置信区域与实际可见区域高度吻合

## 亮点与洞察
- **"不是所有像素都需要生成"**是简单而深刻的洞察——将 NVS 分解为"已知渲染"和"未知生成"两个子问题，各用最适合的方法处理
- **无 3D 归纳偏置**的纯数据驱动设计：不假设任何 3D 表示（NeRF/3DGS/深度图），纯靠 Plücker 射线+大规模数据+大 Transformer。这种"scalable"设计理念来自 LVSM/CAT3D 系列
- MAR+扩散损失的巧妙组合：MAR 提供高效的预训练框架，扩散损失提供高质量的像素级条件生成
- 置信度预测实现了两种头的自动分工——无需手动指定哪些区域该生成

## 局限与展望
- 仍需每个目标视角独立生成——不支持一次性渲染多目标视角的一致3D场景
- 极端外推（输入和目标视角差异很大）时确定性头的覆盖率很低，退化为几乎全生成
- 当前的 Token 大小为 8×8 patch，限制了细节分辨率
- 未与 3DGS/NeRF 后处理管线结合——可能进一步提升 3D 一致性

## 相关工作与启发
- **vs LVSM (NeurIPS'24)**：LVSM 是纯确定性 Transformer NVS，UMAMI 在其基础上增加了生成头处理遮挡
- **vs CAT3D (ECCV'24)**：CAT3D 是全扩散 NVS，UMAMI 只对必要区域做扩散，速度快 10倍
- **vs MAR (ICLR'24)**：MAR 原本用于图像生成，UMAMI 将其适配到条件 NVS——利用 Transformer 特征做条件扩散
- 混合确定性+生成的思路可以推广到视频编辑、虚拟试穿等"部分已知+部分需生成"的任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 确定性+生成的混合 NVS 是首创，理念清晰且执行优雅
- 实验充分度: ⭐⭐⭐⭐ RealEstate10K+DL3DV，多输入配置，详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 数学形式化完整，图示清晰，与相关工作对比到位
- 价值: ⭐⭐⭐⭐⭐ 在速度和质量之间找到了最佳平衡，推进了实用 NVS 的前沿

<!-- RELATED:START -->

## 相关论文

- [DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)
- [Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)

<!-- RELATED:END -->
