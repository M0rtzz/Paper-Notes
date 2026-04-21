---
title: >-
  [论文解读] Precise Object and Effect Removal with Adaptive Target-Aware Attention
description: >-
  [CVPR 2026][图像生成][物体移除] 提出 ObjectClear 框架，通过自适应目标感知注意力（ATA）将前景移除与背景重建解耦，配合注意力引导融合（AGF）和空间变化去噪强度（SVDS）策略，实现对目标物体及其阴影、反射等附带效果的精准移除，同时构建了首个大规模 Object-Effect Removal 数据集 OBER。
tags:
  - CVPR 2026
  - 图像生成
  - 物体移除
  - 阴影/反射消除
  - 扩散模型
  - 注意力引导融合
  - 数据集构建
---

# Precise Object and Effect Removal with Adaptive Target-Aware Attention

**会议**: CVPR 2026  
**arXiv**: [2505.22636](https://arxiv.org/abs/2505.22636)  
**代码**: https://zjx0101.github.io/projects/ObjectClear  
**领域**: 图像生成  
**关键词**: 物体移除, 阴影/反射消除, 扩散模型, 注意力引导融合, 数据集构建

## 一句话总结
提出 ObjectClear 框架，通过自适应目标感知注意力（ATA）将前景移除与背景重建解耦，配合注意力引导融合（AGF）和空间变化去噪强度（SVDS）策略，实现对目标物体及其阴影、反射等附带效果的精准移除，同时构建了首个大规模 Object-Effect Removal 数据集 OBER。

## 研究背景与动机

**领域现状**：基于扩散模型的图像修复/物体移除已成为主流范式，通过结合目标分割掩码与扩散生成器来擦除图像中的不需要物体。代表方法包括 SDXL-Inpainting、PowerPaint、BrushNet、RORem 等。

**现有痛点**：现有方法存在三个核心问题——（a）**效果残留**：只能移除物体本身，难以同时去除阴影和反射等视觉效果；（b）**幻觉生成**：在移除区域生成不需要的新物体或纹理；（c）**背景篡改**：非目标区域的颜色、纹理被意外修改。

**核心矛盾**：缺乏对目标物体与其附带视觉效果之间关联的显式建模，以及缺少有效约束来引导生成模型的注意力聚焦于移除区域。现有数据集要么只有模拟数据（缺少真实效果标注）、要么规模太小或未公开。

**本文切入角度**：将前景移除与背景重建解耦——通过学习目标感知的注意力图来自适应定位物体及其效果区域，同时保持背景的高保真度。此外构建包含物体+效果掩码标注的大规模混合数据集。

**核心idea**：用自适应目标感知注意力（ATA）学习 object-effect 区域的注意力图，然后利用该图在推理时进行注意力引导融合，实现精准移除与背景保持的双重目标。

## 方法详解

### 整体框架
ObjectClear 基于 SDXL-Inpainting 构建。输入为 $\langle z_t, I_{in}, M_o, c \rangle$（噪声潜在表示、原始图像、物体掩码、文本提示），注意这里输入的是**完整原始图像** $I_{in}$ 而非传统方法中的掩码图像 $I_m$，目的是让模型更好地捕捉物体效果的视觉特征以及透明物体背后的背景信息。

### 关键设计

1. **自适应目标感知注意力（ATA）**

    - 功能：引导模型同时关注目标物体区域和其效果区域（阴影、反射）
    - 核心思路：将文本提示"remove the instance of"的文本嵌入与物体视觉嵌入（$I_{in} \cdot M_o$ 经 CLIP 视觉编码器编码后通过 MLP 投射）拼接，作为交叉注意力的引导信号。提取视觉嵌入 token 对应的交叉注意力图 $\mathbf{A}$，用 object-effect 掩码 $M_{fg}$ 进行监督
    - 关键损失：$\mathcal{L}_{mask} = \text{mean}(\mathbf{A}[1-M_{fg}]) - \text{mean}(\mathbf{A}[M_{fg}])$，最小化背景区域注意力、最大化前景区域注意力
    - 设计动机：显式建模物体-效果关联，而非让模型隐式学习

2. **注意力引导融合（AGF）**

    - 功能：在推理阶段利用 ATA 预测的注意力图进行输入-输出自适应融合
    - 核心思路：提取推理时第一层交叉注意力图（对应物体嵌入），上采样到原始图像分辨率后经高斯模糊得到软边缘 object-effect 掩码，用于 alpha blending 将生成结果与原始输入融合
    - 设计动机：减少扩散去噪和 VAE 重建引入的背景变化，保持颜色和纹理的细粒度一致性。与 BrushNet 依赖用户提供的掩码不同，AGF 使用模型自动生成的 object-effect 掩码

3. **空间变化去噪强度（SVDS）**

    - 功能：在物体掩码区域和背景区域使用不同的去噪强度
    - 核心思路：物体区域使用 $DS=1.0$（完全从噪声生成，彻底移除），背景区域使用 $DS=0.99$（保留原始信息，防止颜色偏移），通过在推理过程中重新注入背景实现
    - 设计动机：统一的去噪强度面临矛盾——$DS=1.0$ 移除彻底但全局颜色偏移，$DS=0.99$ 颜色一致但移除不完全

### OBER 数据集构建

混合数据集包含两部分：
- **相机拍摄数据（2,878对）**：固定相机拍摄物体存在/不存在的图像对，使用 DINO+SAM 获取物体掩码，通过输入/GT 像素差分计算 object-effect 掩码
- **模拟数据（10,000张）**：从 Internet 收集背景图（通过 Mask2Former 筛选平坦区域 + Depth Anything V2 验证深度一致性），用 alpha blending 合成前景物体与效果层，支持多物体遮挡场景。Alpha 计算公式：$\alpha(p) = (I_{gt} - I_{in})/(I_{gt} + \varepsilon)$ 对于效果区域

### 训练策略
- 基于 SDXL-Inpainting，512×512 分辨率，batch size 32，8× A100 训练 100k 步，学习率 1e-5
- 总损失 = 标准扩散损失 + $\mathcal{L}_{mask}$
- 推理时 guidance scale = 1.0，20步去噪

## 实验关键数据

### 主实验

| 数据集 | 指标 | ObjectClear | OmniPaint（前SOTA） | 提升 |
|--------|------|-------------|---------------------|------|
| RORD-Val | PSNR↑ | **26.24** | 22.75 | +3.49 |
| RORD-Val | PSNR-BG↑ | **29.78** | 24.66 | +5.12 |
| RORD-Val | LPIPS↓ | **0.1157** | 0.1178 | -0.002 |
| OBER-Test | PSNR↑ | **33.04** | 29.06 | +3.98 |
| OBER-Test | PSNR-BG↑ | **35.62** | 30.04 | +5.58 |
| OBER-Test | LPIPS↓ | **0.0342** | 0.0521 | -0.018 |

关键：即便 ObjectClear 仅使用物体掩码，也超越了使用 object-effect 掩码的所有方法。PSNR-BG 指标大幅领先（+5dB），说明背景保持优势显著。

### 消融实验

| 配置 | PSNR↑ | PSNR-BG↑ | LPIPS↓ | 说明 |
|------|-------|----------|--------|------|
| 仅 CC Data | 27.29 | 27.96 | 0.0910 | 基线 |
| + ATA | 27.56 | 28.37 | 0.0845 | 注意力的作用 |
| + Sim. Data | 28.04 | 28.80 | 0.0805 | 模拟数据的贡献 |
| + AGF | 32.77 | 35.50 | 0.0348 | **AGF 贡献最大** |
| + SVDS | **33.04** | **35.62** | **0.0342** | 完整模型 |

### 关键发现
- **AGF 是最大贡献者**：加入 AGF 后 PSNR 从 28.04 跃升到 32.77（+4.73dB），因为它直接利用学到的注意力图保护背景
- ATA 和 Sim. Data 各贡献约 0.5dB，SVDS 额外贡献约 0.3dB
- 多物体模拟数据对遮挡和物体交互等复杂场景的鲁棒性至关重要

## 亮点与洞察
- **ATA+AGF 的协同设计**极为巧妙：训练时学习的注意力图不仅提升移除精度，还作为推理时融合的自然引导信号，一个模块服务双重目的
- **SVDS 的空间异质去噪**思路可泛化——在任何需要区域差异化处理的扩散编辑任务中，都可以在不同区域使用不同去噪强度
- **数据集构建流程**（像素差分提取效果掩码 + alpha blending 模拟合成）是一套可复用的工具链

## 局限与展望
- 训练分辨率限于 512×512，面对高分辨率实际应用需要额外适配
- 物体掩码依赖外部分割模型（DINO+SAM），掩码质量直接影响结果
- 对于非常复杂的多光源场景（如多物体交叉阴影），效果掩码的自动提取可能不够准确
- 仅处理静态图像，视频物体移除需要额外的时序一致性设计

## 相关工作与启发
- **vs OmniPaint**：同为只需物体掩码的方法，OmniPaint 隐式学习效果移除，ObjectClear 通过 ATA 显式建模效果区域，后者在 PSNR-BG 上领先 5dB+
- **vs RORem**：RORem 需要人工标注质量保证，ObjectClear 通过相机拍摄+像素差分自动获取高质量效果掩码
- **vs Attentive Eraser**：都关注注意力机制，但 Attentive Eraser 在 test-time 优化，ObjectClear 在训练阶段通过 mask loss 学习

## 评分
- 新颖性: ⭐⭐⭐⭐ ATA/AGF/SVDS 三个组件协同设计巧妙，但核心思路（显式效果建模+注意力引导）相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 三个测试集（含自建Wild集）、完整消融、公平对比（两种掩码设置）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建流程描述详尽
- 价值: ⭐⭐⭐⭐ 数据集 OBER 和精准效果移除对实际应用价值大，但需要高分辨率版本

<!-- RELATED:START -->

## 相关论文

- [Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal](learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)
- [ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation](adapt_attention_driven_adaptive_prompt_scheduling_and_interpolating_orthogonal_c.md)

<!-- RELATED:END -->
