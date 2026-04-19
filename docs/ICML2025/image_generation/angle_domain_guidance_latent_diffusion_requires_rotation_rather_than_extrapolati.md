---
title: >-
  [论文解读] Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation
description: >-
  [ICML 2025][图像生成][分类器自由引导] 发现 Classifier-Free Guidance (CFG) 导致颜色失真的根本原因是潜空间样本范数被放大，提出 Angle Domain Guidance (ADG) 算法——在角度域而非幅度域增强引导，约束范数变化的同时优化角度对齐，在高引导权重下消除颜色饱和度异常并保持甚至改善文本-图像对齐。
tags:
  - ICML 2025
  - 图像生成
  - 分类器自由引导
  - 扩散模型
  - 颜色失真
  - 角度域
  - 潜空间
---

# Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation

**会议**: ICML 2025  
**arXiv**: [2506.11039](https://arxiv.org/abs/2506.11039)  
**代码**: https://github.com/jinc7461/ADG  
**领域**: 图像生成  
**关键词**: 分类器自由引导, 扩散模型, 颜色失真, 角度域, 潜空间

## 一句话总结
发现 Classifier-Free Guidance (CFG) 导致颜色失真的根本原因是潜空间样本范数被放大，提出 Angle Domain Guidance (ADG) 算法——在角度域而非幅度域增强引导，约束范数变化的同时优化角度对齐，在高引导权重下消除颜色饱和度异常并保持甚至改善文本-图像对齐。

## 研究背景与动机

**领域现状**：Classifier-Free Guidance (CFG) 是文本到图像扩散模型（如 Stable Diffusion）中的核心技术，通过线性外推条件和无条件分数函数来增强文本-图像对齐。CFG 已成为实际部署的标准技术。

**现有痛点**：高引导权重下，CFG 虽然显著增强文本对齐度，但同时导致严重的颜色失真——图像出现过饱和、不自然的色彩。现有修复方法（如变权方案、额外 Langevin 采样）仍基于线性组合框架，治标不治本。

**核心矛盾**：CFG 的线性外推不可避免地放大样本范数。当引导权重 $w > 1$ 时，条件分数的线性外推使得去噪路径偏离真实分布，表现为潜空间中样本的范数异常增大。样本范数放大→解码后的像素值被推向极端→颜色过饱和。

**本文目标**：在保持高文本对齐度的同时消除颜色失真。

**切入角度**：作者观察到——VAE 的潜空间假设样本近似高维高斯分布。在高维高斯中，样本几乎都分布在一个固定半径的球壳上（集中现象）。因此，范数不应显著变化——真正有意义的是方向（角度）。

**核心 idea**：将引导从"幅度域外推"（线性外推改变范数）转为"角度域旋转"（仅改变方向不改变范数），具体步骤：将分数函数转化为角度域上的期望→在角度域增强条件分布→映射回分数函数用于扩散。

## 方法详解

### 整体框架
ADG 修改了扩散模型的每一步去噪中的引导方式：
1. 计算条件分数 $\epsilon_\theta(x_t, c)$ 和无条件分数 $\epsilon_\theta(x_t, \varnothing)$
2. 不像 CFG 那样做线性外推 $\tilde{\epsilon} = (1+w)\epsilon_c - w\epsilon_\varnothing$
3. 而是在角度域进行操作：
    - 将预测的 $x_0$ 分解为范数和方向
    - 在方向（角度）上增强条件信号
    - 保持范数不变
4. 从增强后的方向和保留的范数重建引导后的分数

### 关键设计

1. **范数放大的理论分析**:

    - 功能：证明 CFG 的线性外推必然导致范数增长
    - 核心论证：设条件预测 $\hat{x}_{0,c}$ 和无条件预测 $\hat{x}_{0,\varnothing}$，CFG 给出 $\hat{x}_0^{\text{CFG}} = (1+w)\hat{x}_{0,c} - w\hat{x}_{0,\varnothing}$。当 $w > 0$ 时，只要两个预测不完全平行，外推结果的范数大于条件预测的范数：$\|\hat{x}_0^{\text{CFG}}\| > \|\hat{x}_{0,c}\|$
    - 设计动机：理论上解释了高 $w$ 下颜色失真的根源——范数放大导致 VAE 解码器输出超出正常范围
    - 异常扩散现象：范数放大在去噪过程中逐步累积，导致轨迹偏离真实分布流形

2. **角度域引导 (ADG) 算法**:

    - 功能：在潜空间中仅旋转方向、不改变范数
    - 核心思路：基于 VAE 潜空间的高维高斯假设——样本集中在固定半径球壳上，有意义的信息在方向中编码
    - 具体步骤：
      1. 从当前 $x_t$ 和分数估计得到 $\hat{x}_0$ 的预测
      2. 分解为方向 $\hat{d} = \hat{x}_0 / \|\hat{x}_0\|$ 和范数 $r = \|\hat{x}_0\|$
      3. 在方向空间（球面上）进行条件增强
      4. 保持 $r$ 不变，从增强后的方向和原范数重建 $\hat{x}_0$
    - 设计动机：角度包含语义信息（内容、结构），范数影响全局特性（亮度、饱和度）——增强前者、保留后者

3. **理论框架——分数函数的球面分解**:

    - 功能：将分数函数转化为范数和方向的联合分布上的期望
    - 核心思路：$\nabla \log p_t(x_t|c)$ 可以分解为径向分量（控制范数）和切向分量（控制方向）
    - ADG 仅增强切向分量——保证范数不变的同时优化方向对齐
    - 设计动机：严格的数学框架保证了 ADG 的合理性

### 损失函数 / 训练策略
- ADG 是推理时方法，不需要训练
- 直接替换 CFG 的引导步骤，与任何扩散采样器兼容
- 计算开销与 CFG 相同（同样需要两次分数计算）
- 超参数：引导强度 $w$（与 CFG 相同的语义）

## 实验关键数据

### 主实验
COCO 数据集上 SDXL 模型的文本-图像生成：

| 引导方法 | FID ↓ | CLIP Score ↑ | 饱和度偏差 ↓ | HPSv2 ↑ |
|---------|-------|------------|-----------|---------|
| CFG (w=3) | 24.8 | 0.31 | 0.08 | 0.267 |
| CFG (w=7) | 28.5 | 0.33 | 0.25 | 0.251 |
| CFG (w=15) | 42.3 | 0.35 | 0.52 | 0.223 |
| Rescaled CFG (w=7) | 26.1 | 0.32 | 0.12 | 0.263 |
| **ADG (w=7)** | **22.3** | **0.34** | **0.05** | **0.278** |
| **ADG (w=15)** | **23.1** | **0.35** | **0.06** | **0.275** |

### 消融实验
| 配置 | FID | CLIP | 饱和度偏差 | 说明 |
|------|-----|------|----------|------|
| CFG w=7 (基线) | 28.5 | 0.33 | 0.25 | 标准 CFG |
| 仅范数裁剪 | 25.8 | 0.32 | 0.09 | 修复颜色但文本对齐下降 |
| 仅角度增强（无范数约束） | 27.2 | 0.34 | 0.18 | 部分改善 |
| **ADG（角度增强+范数保持）** | **22.3** | **0.34** | **0.05** | 两方面最优 |
| ADG + DDIM 采样 | 22.1 | 0.34 | 0.05 | 与确定性采样兼容 |
| ADG + DDPM 采样 | 22.5 | 0.34 | 0.05 | 与随机采样兼容 |

### 定性结果对比
- 低 w (w=3)：ADG 已正确对齐文本，CFG 仍然对齐不足
- 高 w (w=15)：ADG 保持稳定的颜色和质量，CFG 严重过饱和失真
- 相同 w 下 ADG 的有效引导强度更高——因为没有能量浪费在范数放大上

### 关键发现
- ADG 在 w=15 时仍保持 FID 23.1，而 CFG 在 w=7 时就已恶化到 28.5
- 饱和度偏差从 0.25 (CFG w=7) 降到 0.05 (ADG w=7)——几乎消除颜色失真
- HPSv2（人类偏好分数）在高 w 下 ADG 明显优于 CFG——证实了人类视觉偏好
- ADG 不仅修复了 CFG 的问题，还在 FID 上显著改善——说明去除范数放大不仅避免了失真，还改善了生成分布的整体质量
- 在 SD 1.5、SDXL、SD 3.0 等多个模型上效果一致

## 亮点与洞察
- **"旋转而非外推"**这个 idea 极具画面感和直觉性——一句话就能理解核心贡献
- 用高维高斯的集中现象来解释潜空间结构非常精妙——VAE 的设计目的就是让潜分布接近高斯，所以球壳假设合理
- 理论和直觉完美对应：范数=亮度/饱和度（全局属性），方向=内容/结构（语义信息）
- 与 CFG 计算量相同但效果大幅提升——真正的"免费午餐"
- 在高 w 下的稳定性意味着用户可以更激进地提高文本对齐而不担心副作用

## 局限与展望
- 高维球壳假设在 VAE 训练不够好时可能不成立
- 对非 VAE 潜空间的扩散模型（如像素空间扩散）的适用性未分析
- 角度域增强的具体形式还有探索空间（本文用了较简单的旋转策略）
- 未讨论对视频扩散模型中 CFG 颜色闪烁问题的修复效果
- 与其他 CFG 改进方法（如 Autoguidance）的组合效果待探索

## 相关工作与启发
- **vs Rescaled CFG**: 事后缩放范数，但也改变了语义信息；ADG 从源头避免范数增长
- **vs Dynamic CFG**: 变权方案仍是线性外推框架内，ADG 跳出线性外推
- **vs Perp-Neg**: 在噪声空间做正交分解，与 ADG 在 $x_0$ 预测空间做球面分解互补
- **启发**：角度域操作的思想可推广到其他使用 CFG 的条件生成任务（3D 生成、视频生成等）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 角度域引导的 idea 简洁、深刻、实用
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型（SD1.5/SDXL/SD3）、多指标、充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论+直觉+实验的完美结合
- 价值: ⭐⭐⭐⭐⭐ 对所有使用 CFG 的扩散模型有普遍价值

<!-- RELATED:START -->

## 相关论文

- [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](../../NeurIPS2025/image_generation/vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [Visual Generation Without Guidance](visual_generation_without_guidance.md)
- [Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)

<!-- RELATED:END -->
