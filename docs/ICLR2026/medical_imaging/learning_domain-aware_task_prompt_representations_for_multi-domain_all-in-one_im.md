---
title: >-
  [论文解读] Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration
description: >-
  [ICLR 2026][医学图像][全能图像复原] 提出首个多域全能图像复原方法DATPRL-IR，通过双提示池（任务提示池+域提示池）学习域感知的任务提示表征，利用MLLM蒸馏域先验并通过自适应门控融合指导复原，在自然/医学/遥感三域9任务上显著超越SOTA。
tags:
  - ICLR 2026
  - 医学图像
  - 全能图像复原
  - 多域复原
  - 提示学习
  - 双提示池
  - 跨模态对齐
---

# Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration

**会议**: ICLR 2026  
**arXiv**: [2603.01725](https://arxiv.org/abs/2603.01725)  
**代码**: [GitHub](https://github.com/GuangluDong0728/DATPRL-IR)  
**领域**: 图像复原 / 计算机视觉  
**关键词**: 全能图像复原, 多域复原, 提示学习, 双提示池, 跨模态对齐

## 一句话总结
提出首个多域全能图像复原方法DATPRL-IR，通过双提示池（任务提示池+域提示池）学习域感知的任务提示表征，利用MLLM蒸馏域先验并通过自适应门控融合指导复原，在自然/医学/遥感三域9任务上显著超越SOTA。

## 研究背景与动机

**领域现状**：现有全能图像复原（AiOIR）方法（如PromptIR、MoCE-IR）能用单一模型处理多种退化任务，但仅局限于单一图像域（如自然图像或医学图像），尚未有方法同时处理跨域的多任务复原。

**现有痛点**：(1) 不同域的图像（自然、医学、遥感）有各自独特的视觉特征，单域方法无法迁移；(2) 现有方法侧重区分不同任务的差异，忽略了任务间的共享知识；(3) 随着任务和域的增加，模型学习难度急剧上升。

**核心矛盾**：多域多任务设置下，需要同时建模任务特异性、域特异性以及它们之间的共享知识，现有单一提示或单一编码机制无法有效捕获这种层次化的知识结构。

**本文目标** 如何用一个模型同时处理跨3个域（自然、医学、遥感）的多种复原任务？如何有效利用任务间和域间的共享知识来降低学习难度？

**切入角度**：不同域的图像虽然有独特特征，但也存在重叠的视觉特性（如"灰度+人体器官"对应医学，"鸟瞰+建筑"对应遥感）；通过双提示池分别编码任务和域知识，并在实例级自适应组合和融合。

**核心 idea**：用双提示池分别学习任务和域的专有/共享知识，通过提示组合机制和跨注意力融合生成域感知的任务提示表征来指导多域全能复原。

## 方法详解

### 整体框架
输入退化图像 → 编解码器骨干网络提取特征 → 中间特征查询任务提示池（TP Pool）获取任务表征 $\mathbf{PR}_t$ → 浅层特征查询域提示池（DP Pool）获取域表征 $\mathbf{PR}_d$ → 跨注意力融合为域感知任务提示表征 $\mathbf{PR}_{dt}$ → 通过自适应门控融合（AGF）注入骨干网络各层 → 输出复原图像。

### 关键设计

1. **任务提示池与提示组合机制（PCM）**:

    - 做什么：隐式编码不同复原任务的专有和共享知识，并为每张输入图像自适应生成实例级任务表征
    - 核心思路：构建 $N_t=15$ 个键值对提示 $(\mathbf{K}_j^{\text{task}}, \mathbf{V}_j^{\text{task}})$，用可学习投影器将编码器中间特征映射为查询 $\mathbf{Q}^{\text{task}}$，按余弦相似度选top-$k$（$k=3$）个提示，通过温度softmax加权组合：$\mathbf{PR}_t = \sum_{j \in k} \alpha_j^{\text{task}} \mathbf{V}_j^{\text{task}}$。提示随复原目标联合优化。
    - 设计动机：不同任务可能共享某些提示（如超分和去模糊都需要锐化），PCM通过加权组合而非硬分配实现了知识共享和任务特异性的平衡。

2. **域提示池与MLLM知识蒸馏**:

    - 做什么：学习域相关的视觉先验，并从多模态大语言模型中蒸馏域知识
    - 核心思路：构建 $N_d=15$ 个域提示，用浅层特征查询并组合。训练时，LLaVA-1.5-7B对高质量图像生成多角度文本描述（内容、色彩、物体、亮度、视角），经CLIP文本编码器得到 $\mathbf{F}_{\text{text}}$，通过跨模态对齐损失 $\mathcal{L}_{\text{align}} = 1 - \cos(\mathbf{PR}_d, \mathbf{F}_{\text{text}})$ 将域先验蒸馏入域提示。推理时不需要LLaVA和CLIP，无额外开销。
    - 设计动机：域感知需要理解图像的语义级特征（内容类型、拍摄方式等），MLLM具有强大的图像理解能力，可以提供丰富的域描述；使用蒸馏确保训练阶段获得知识而推理阶段无负担。

3. **自适应门控融合（AGF）**:

    - 做什么：在骨干网络每一层动态控制提示表征和特征的融合比例
    - 核心思路：任务和域表征通过跨注意力融合为 $\mathbf{PR}_{dt}$，每层通过可学习门控 $\alpha_l \in [0,1]$ 控制比例：$\mathbf{F}_l^e = \text{CrossAttn}(\alpha_l \mathbf{F}_l, (1-\alpha_l) \mathbf{PR}_{dt})$，允许不同深度的层独立学习最优融合策略。
    - 设计动机：浅层可能更需要域信息（辨别输入类型），深层可能更需要任务信息（执行具体复原操作），固定融合比例过于刚性。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \lambda_{\text{pix}}\mathcal{L}_{\text{pix}} + \lambda_{\text{fft}}\mathcal{L}_{\text{fft}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{bal}}\mathcal{L}_{\text{bal}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}}$。$\mathcal{L}_{\text{pix}}$ 和 $\mathcal{L}_{\text{fft}}$ 分别为RGB和傅里叶域的 $\ell_1$ 损失；$\mathcal{L}_{\text{div}}$ 鼓励提示多样性（余弦相似度阈值 $\tau=0.1$）；$\mathcal{L}_{\text{bal}}$ 提示使用均衡正则化（最大化选择熵）；Adam优化器，学习率 $4 \times 10^{-4}$ + cosine退火，batch=12，训练1000K iter。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | DATPRL-IR (6T) | MoCE-IR (SOTA) | 提升 |
|--------|------|------|----------|------|
| 自然SR / DIV2K-Val | PSNR | 28.98 | 28.16 | +0.82 |
| 去雨 / Rain100L | PSNR | 39.56 | 38.64 | +0.92 |
| MRI SR / IXI MRI | PSNR | 27.88 | 27.75 | +0.13 |
| CT去噪 / AAPM-Mayo | PSNR | 33.80 | 33.74 | +0.06 |
| 遥感SR / UCMerced | PSNR | 28.29 | 28.06 | +0.23 |
| 云去除 / CUHK CR1 | PSNR | 26.12 | 26.06 | +0.06 |
| **6任务平均** | **PSNR** | **30.77** | **30.40** | **+0.37** |

### 消融实验

| 配置 | 去雨PSNR | CT去噪PSNR | 遥感SR PSNR |
|------|---------|------|------|
| 无TP+无DP（基线） | 38.34 | 33.70 | 28.02 |
| 仅TP Pool | 39.32 | 33.76 | 28.16 |
| 仅DP Pool | 38.88 | 33.74 | 28.12 |
| TP+DP（完整） | 39.56 | 33.80 | 28.29 |

### 关键发现
- 从6任务扩展到9任务时，原有任务性能不降反升（如自然SR: 28.98→29.05），验证了不同任务间存在可迁移的共享知识
- 更换不同规模的MLLM（LLaVA-7B/13B、Qwen3-VL-2B）对性能影响极小，说明方法仅依赖粗粒度域语义
- 用固定文本提示（如"这是MRI图像"）替代域提示池会降低性能，验证了自适应选择和共享建模的必要性
- 提示池大小15、top-k=3/5 是最优配置，过大过小都会影响性能

## 亮点与洞察
- 首次将全能图像复原扩展到多域场景，提出的双提示池架构优雅地解耦了任务知识和域知识的学习，通过PCM实现了共享与专有知识的自适应平衡。扩展任务不降性能的特性具有很好的扩展性。
- 利用MLLM蒸馏域先验的设计思路巧妙：训练时利用LLaVA的强理解能力，推理时完全不需要，实现了"免费"的域感知能力。

## 局限与展望
- 域的扩展目前仅覆盖自然/医学/遥感三域，更多域（如水下、夜视、卫星等）的可扩展性有待验证
- 提示池的大小和top-k需要手动调节，缺乏自适应机制
- 仅使用PSNR/SSIM评估，缺少感知质量指标（如LPIPS）和下游任务评估

## 相关工作与启发
- **vs PromptIR**: PromptIR使用单一可学习提示编码退化信息，本文通过提示池+PCM实现更灵活的实例级表征，且额外引入域感知维度
- **vs MoCE-IR**: MoCE-IR用混合专家架构分配任务资源，本文通过双提示池的"查询-检索-组合"范式实现类似功能但更轻量，6任务平均高出0.37dB

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多域全能复原方法，双提示池+MLLM蒸馏的设计有新意
- 实验充分度: ⭐⭐⭐⭐ 3域9任务的完整实验，详尽的消融和扩展性验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ 多域统一复原具有重要实践意义，双提示池可迁移到其他多域多任务场景

<!-- RELATED:START -->

## 相关论文

- [Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [Towards All-in-One Medical Image Re-Identification](../../CVPR2025/medical_imaging/towards_all-in-one_medical_image_re-identification.md)
- [Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2026/medical_imaging/human_knowledge_integrated_multimodal_learning_for.md)

<!-- RELATED:END -->
