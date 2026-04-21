---
title: >-
  [论文解读] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?
description: >-
  [ICLR 2026][图像生成][图像超分辨率] 提出GLYPH-SR，一个视觉-语言引导的扩散框架，通过双分支Text-SR融合ControlNet和ping-pong调度器同时优化图像质量和文本可读性，在SVT ×8上将OCR F1提升15.18个百分点。
tags:
  - ICLR 2026
  - 图像生成
  - 图像超分辨率
  - 场景文本恢复
  - ControlNet
  - 扩散模型
  - OCR
---

# GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?

**会议**: ICLR 2026  
**arXiv**: [2510.26339](https://arxiv.org/abs/2510.26339)  
**代码**: 有（论文提到release）  
**领域**: 扩散模型  
**关键词**: 图像超分辨率, 场景文本恢复, ControlNet, 扩散模型, OCR

## 一句话总结
提出GLYPH-SR，一个视觉-语言引导的扩散框架，通过双分支Text-SR融合ControlNet和ping-pong调度器同时优化图像质量和文本可读性，在SVT ×8上将OCR F1提升15.18个百分点。

## 研究背景与动机
图像超分辨率(SR)是许多视觉系统的基础技术，但现有SR方法存在两个系统性偏差：(1) 指标偏差——PSNR/SSIM等全局指标对小文本区域（通常不到图像1%）的贡献极小，字符损坏几乎不受惩罚；(2) 目标偏差——常用训练损失将文字视为普通高频纹理而非OCR所需的离散语义单元。这导致两种失败模式：幻觉（生成清晰但错误的字符）和保守恢复（保持模糊不改善）。核心问题是如何同时实现视觉真实感和文本可读性——两个目标之间存在明显tension。

## 方法详解

### 整体框架
GLYPH-SR基于预训练LDM（Juggernaut-XL），在其上添加Text-SR融合ControlNet(TS-ControlNet)，通过OCR提取文本-位置对来提供文字级语义引导，利用ping-pong调度器在去噪过程中交替文本中心和图像中心引导。

### 关键设计
1. **条件分解(Condition Decomposition)**:

    - 功能：将引导信号显式分离为图像导向和文本导向
    - 核心思路：场景级标题 $\mathcal{S}_{\text{IMG}}$ 概括全局属性（光照、构图等）；OCR模块检测 $K$ 个文本实例返回位置-文本对 $\{(\mathcal{S}_{\text{text}}^k, \mathcal{S}_{\text{pos}}^k)\}_{k=1}^K$，转为结构化自然语言提示（如"HSBC显示在图像中心"）
    - 设计动机：当引导仅以整体形式提供时，小文本区域仍被视为通用高频纹理

2. **Text-SR融合ControlNet(TS-ControlNet)**:

    - 功能：在保持生成先验的同时平衡图像质量和文本可读性
    - 核心思路：双分支架构——SR分支冻结保持整体图像质量，文本分支可训练专注字形恢复。残差混合注入：$c = \frac{1}{2} s_{\text{ctrl}} [\mathcal{C}_{\text{SR}}(z_t; \phi_{\text{img}}(\mathcal{S}_{\text{IMG}}+P)) + \mathcal{C}_{\text{TXT}}(z_t; \phi_{\text{txt}}(\mathcal{S}_{\text{TXT}}+P))]$
    - 设计动机：直接分离两种引导虽改善文字但损害非文字区域

3. **Ping-Pong调度器**:

    - 功能：沿去噪轨迹动态重新加权文本和图像引导
    - 核心思路：时间依赖系数 $\lambda_t$ 同时调制嵌入融合和残差注入。采用二值方波策略交替 $\lambda_t=0$（文本中心）和 $\lambda_t=1$（图像中心），切换周期 $\tau=1$：$\lambda_t = 0$ 若 $\lfloor \frac{t-t_0}{\tau} \rfloor \bmod 2 = 0$，否则 $\lambda_t = 1$
    - 设计动机：连续渐变不如方波有效；文本阶段注入精确字形线索，图像阶段稳定全局结构

### 损失函数 / 训练策略
- 使用标准 $\varepsilon$-预测目标训练：$\mathcal{L}_{\text{text}} = \mathbb{E}_{z_0, t, \varepsilon} \| \varepsilon - \mathcal{D}_\theta(z_t, t, c) \|_2^2$
- 构建4分区合成语料，独立扰动字形质量和全局图像质量，实现针对性文本恢复
- LDM骨干和SR分支冻结，仅微调文本分支

## 实验关键数据

### 主实验（SVT ×4 OCR F1）

| 方法 | OpenOCR | GOT-OCR | LLaVA-NeXT | MANIQA | CLIP-IQA |
|------|---------|---------|------------|--------|----------|
| DiffBIR | 38.73 | 42.33 | 45.19 | 47.82 | 58.66 |
| InvSR | 57.79 | 60.96 | 65.00 | 46.78 | 57.30 |
| PiSA-SR | 63.30 | 65.23 | 67.75 | 37.41 | 44.30 |
| **GLYPH-SR** | **67.54** | **71.72** | **73.22** | **47.75** | **59.40** |

### 消融实验（核心组件贡献）

| 配置 | OCR F1 | MANIQA | 说明 |
|------|--------|--------|------|
| 仅分离条件 | 提升文字 | 下降 | 非文字区域退化 |
| +TS-ControlNet | 进一步提升 | 保持 | 双分支平衡 |
| +Ping-Pong | 最优 | 竞争力 | 方波优于连续渐变 |

### 关键发现
- SVT ×8上OCR F1比扩散/GAN基线提升最高15.18个百分点
- 在三个数据集(SVT/SCUT-CTW1500/CUTE80)×两个尺度(4×/8×)全面验证
- 在保持竞争力的MANIQA/CLIP-IQA/MUSIQ同时大幅提升OCR指标

## 亮点与洞察
- 将场景文本SR显式建模为双目标优化问题，首次标准化双轴评估协议
- 4分区合成数据设计巧妙：通过正交扰动字形和图像质量解耦学习
- Ping-pong调度器简单有效，比复杂的连续噪声级调度更优

## 局限与展望
- 依赖OCR模块提取文本位置，OCR模块本身可能在低分辨率下失败
- 合成训练数据可能不完全代表实际退化
- 仅验证了4×和8×，更高倍率的效果未知

## 相关工作与启发
- **vs StableSR/DiffBIR**: 这些方法优化感知质量但对字符完整性不敏感
- **vs TATT等文本SR**: 文本SR方法在全场景中表现不佳，因为假设简化场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 双目标SR框架和ping-pong调度器设计新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集两个尺度全面比较
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，动机充分
- 价值: ⭐⭐⭐⭐ 对场景文本SR有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation](../../NeurIPS2025/image_generation/repldm_reprogramming_pretrained_latent_diffusion_models_for_high-quality_high-ef.md)

<!-- RELATED:END -->
