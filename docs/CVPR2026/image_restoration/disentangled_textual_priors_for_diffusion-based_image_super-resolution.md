---
title: >-
  [论文解读] Disentangled Textual Priors for Diffusion-based Image Super-Resolution
description: >-
  [CVPR 2026][图像恢复][扩散模型超分] 提出 DTPSR，通过将文本先验沿空间层级（全局/局部）和频率语义（低频/高频）两个维度解耦，构建解耦的跨注意力注入管线和多分支 CFG 策略，实现感知质量优越的扩散超分辨率。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "扩散模型超分"
  - "文本引导"
  - "解耦先验"
  - "频率感知"
  - "语义控制"
---

# Disentangled Textual Priors for Diffusion-based Image Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2603.07430](https://arxiv.org/abs/2603.07430)  
**代码**: [GitHub](https://github.com/JL6666JL/DTPSR)  
**领域**: 图像超分辨率  
**关键词**: 扩散模型超分, 文本引导, 解耦先验, 频率感知, 语义控制

## 一句话总结

提出 DTPSR，通过将文本先验沿空间层级（全局/局部）和频率语义（低频/高频）两个维度解耦，构建解耦的跨注意力注入管线和多分支 CFG 策略，实现感知质量优越的扩散超分辨率。

## 研究背景与动机

扩散模型（如 Stable Diffusion）在图像超分中展现了强大的生成能力，但其性能严重依赖语义先验的构造与注入方式。现有方法存在两类局限：

**语义粒度不足**：局部标签方法（SeeSR）关注细节但缺乏全局一致性；全局描述方法（SUPIR、PASD）关注全局但忽略细粒度细节

**频率信息纠缠**：现有方法将结构信息（低频：形状、布局）与纹理信息（高频：边缘、材质）混在同一嵌入中，导致语义可控性和解释性不足

**退化严重时的幻觉问题**：缺乏解耦语义引导时，扩散模型容易产生幻觉，如将墙壁误解为海洋纹理

核心洞察：将文本先验沿 **空间层级** 和 **频率语义** 两个正交维度解耦，让模型可以同时捕捉场景级结构和对象级细节。

## 方法详解

### 整体框架

给定低分辨率图像 $x_{lr}$，DTPSR 的流程为：
1. VAE 编码器将 $x_{lr}$ 映射到隐空间 $z_0$
2. 前向扩散加噪得到 $z_t$
3. 反向去噪过程中，依次通过四个专用跨注意力模块进行语义注入：

$$z_t \xrightarrow{\text{GTCA}} z_t^g \xrightarrow{\text{LFCA}} z_t^{lf} \xrightarrow{\text{HFCA}} z_t^{hf} \xrightarrow{\text{LRCA}} z_{t-1}$$

### 关键设计

1. **全局文本跨注意力 (GTCA)**：全局描述 $c_g$ 经 CLIP 编码为 $e_g$，通过跨注意力注入隐变量，建立场景级结构基础。设计动机：先确立全局布局（如"室内场景有3个物体"），再逐步细化。

2. **低频跨注意力 (LFCA)**：编码一组低频局部描述 $\{c_{lf}^{(i)}\}$（形状、大小、空间排列），注入 $z_t^g$ 进行对象级结构增强：
$$E_{lf} = [\text{CLIP\_TextEnc}(c_{lf}^{(1)}), \dots, \text{CLIP\_TextEnc}(c_{lf}^{(n)})]$$
设计动机：低频信息决定结构保真度，与高频纹理分开处理可避免相互干扰。

3. **高频跨注意力 (HFCA)**：编码高频描述 $\{c_{hf}^{(j)}\}$（纹理、边缘、表面细节），在 LFCA 输出基础上进一步注入高频语义：
$$z_t^{hf} = \text{HFCA}(z_t^{lf}, E_{hf})$$
设计动机：高频信息控制视觉真实感，独立注入可精确控制纹理生成。

4. **低分辨率特征跨注意力 (LRCA)**：使用冻结的 DAPE 编码器提取 $x_{lr}$ 的视觉特征 $f_{lr}$，通过跨注意力锚定输入图像的身份一致性，防止语义偏移。

5. **多分支无分类器引导 (Multi-branch CFG)**：为全局、低频、高频三个分支分别设计独立的负提示 $c_g^{\text{neg}}, c_{lf}^{\text{neg}}, c_{hf}^{\text{neg}}$，实现解耦的语义抑制：
$$\tilde{\epsilon} = \hat{\epsilon} + \lambda_s(\hat{\epsilon} - \hat{\epsilon}_{\text{neg}})$$
设计动机：单一负提示无法同时处理多个语义源的幻觉问题。

### 损失函数 / 训练策略

- **训练损失**：标准噪声预测 MSE 损失
$$\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, z_{lr}, t, c_g, c_{lf}, c_{hf})\|_2^2]$$
- **数据集 DisText-SR**：约 95K 图像-文本对，基于 LSDIR + FFHQ 前 10K 张，通过 Mask2Former 分割 + LLaVA 生成解耦描述
- **基座模型**：SD-2-base；DAPE 编码器提取 LR 嵌入
- **训练配置**：AdamW 优化器，lr $5 \times 10^{-5}$，batch 32，110K iterations，4× A800
- **推理**：DDPM 50 步，引导尺度 $\lambda_s = 7.0$

## 实验关键数据

### 主实验

| 数据集 | 指标 | DTPSR | FaithDiff | SUPIR | 提升 |
|--------|------|-------|-----------|-------|------|
| DIV2K-Val | MUSIQ↑ | **71.24** | 69.18 | 62.59 | +2.06 |
| DIV2K-Val | MANIQA↑ | **0.5866** | 0.4309 | 0.5224 | +0.0642 |
| DIV2K-Val | CLIPIQA↑ | **0.7549** | 0.6463 | 0.7040 | +0.0509 |
| RealSR | MUSIQ↑ | **71.84** | 68.86 | 58.51 | +2.98 |
| RealSR | MANIQA↑ | **0.6021** | 0.4644 | 0.4429 | +0.0432 |
| DRealSR | CLIPIQA↑ | **0.7640** | 0.6335 | 0.6307 | +0.0729 |

注：DTPSR 在所有无参考感知指标上均为 SOTA，但 PSNR/SSIM 低于 GAN 方法（感知-失真权衡）。

### 消融实验

| 配置 | MANIQA↑ | CLIPIQA↑ | MUSIQ↑ | 说明 |
|------|---------|----------|--------|------|
| 无先验 | 0.5271 | 0.7064 | 67.48 | 基线 |
| 仅局部 | 0.5851 | 0.7471 | 68.86 | 局部先验贡献更大 |
| 仅全局 | 0.5394 | 0.7211 | 67.80 | 全局提升较温和 |
| 全局+局部 | **0.6011** | **0.7640** | **69.24** | 互补效果最佳 |
| 频率混合 | 0.5947 | 0.7527 | 69.05 | 解耦优于混合 |
| 频率解耦 | **0.6011** | **0.7640** | **69.24** | 分离注入更有效 |

### 关键发现

- 局部先验的贡献远大于全局先验（MANIQA 提升 0.0580 vs 0.0123）
- 频率解耦比频率混合在所有指标上一致更优
- 多分支 CFG 比单一/无 CFG 显著提升感知质量（MUSIQ 66.73→69.24）
- 即使文本描述被随机损坏（替换为"None"），DTPSR 仍优于其他方法，展现鲁棒性
- 参数量 10.5B，推理 14.94s/张，与 SUPIR (17.8B) 和 FaithDiff (15.6B) 相比效率更优

## 亮点与洞察

1. **解耦设计优雅**：沿空间层级 × 频率语义两个正交维度解耦文本先验，概念清晰且有效
2. **DisText-SR 数据集**：首个结合全局-局部 + 低频-高频文本标注的大规模 SR 数据集，为可控超分研究提供基础
3. **多分支 CFG 策略**：无需额外训练即可抑制幻觉，用频率感知的负提示实现细粒度控制
4. **鲁棒性实验**：即使上游模块（分割、描述）输出不完美，系统仍能正常工作

## 局限与展望

1. PSNR/SSIM 等全参考指标不如 GAN 方法，存在感知-失真权衡
2. 依赖上游分割（Mask2Former）和描述（LLaVA）模型的质量
3. 推理时需要运行分割和描述生成流程，增加了端到端延迟
4. 仅处理 top-3 最大分割区域，可能遗漏小但重要的细节区域
5. 未来方向：自适应提示校正、与上游模块更紧密集成、更高效的扩散骨干

## 相关工作与启发

- **SeeSR**：使用局部语义标签，但仅关注细节而缺乏全局一致性
- **SUPIR/PASD/FaithDiff**：使用全局描述，但忽略频率分离
- **StableSR/DiffBIR**：未利用文本语义，错过了扩散先验的全部潜力
- 启发：解耦文本先验的思路可推广到其他条件生成任务（如编辑、修复）

## 评分

- 新颖性: ⭐⭐⭐⭐ 空间-频率双重解耦的文本先验设计新颖，多分支 CFG 策略有创意
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多指标、丰富消融（全局/局部、频率、CFG、鲁棒性）
- 写作质量: ⭐⭐⭐⭐ 动机清晰、框架图清楚、实验组织合理
- 价值: ⭐⭐⭐⭐ 为文本引导扩散超分提供了新范式，DisText-SR 数据集有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)

</div>

<!-- RELATED:END -->
