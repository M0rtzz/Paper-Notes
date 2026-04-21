---
title: >-
  [论文解读] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback
description: >-
  [NeurIPS 2025][图像生成][重光照] 提出UniLumos，一个统一的图像和视频重光照框架，通过在flow matching骨干中引入RGB空间的深度和法线几何反馈来增强物理合理性，同时借助路径一致性学习实现20倍加速。
tags:
  - NeurIPS 2025
  - 图像生成
  - 重光照
  - flow matching
  - 物理反馈
  - 视频生成
  - 几何监督
---

# UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback

**会议**: NeurIPS 2025  
**arXiv**: [2511.01678](https://arxiv.org/abs/2511.01678)  
**代码**: [GitHub](https://github.com/alibaba-damo-academy/Lumos-Custom)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 重光照, flow matching, 物理反馈, 视频生成, 几何监督

## 一句话总结

提出UniLumos，一个统一的图像和视频重光照框架，通过在flow matching骨干中引入RGB空间的深度和法线几何反馈来增强物理合理性，同时借助路径一致性学习实现20倍加速。

## 研究背景与动机

重光照（Relighting）是计算机视觉和图形学中的重要任务，需要在改变光照的同时保持场景固有属性。现有方法面临以下核心挑战：

**传统方法的局限性**：基于逆渲染的方法需要复杂输入（如HDR图像、球谐系数），受限于特定场景类型，难以适用于仅有单张图片或简短文本描述的实际场景。

**扩散模型的根本弱点**：现有扩散模型在语义潜空间中工作，潜空间的相似性并不保证视觉空间的物理正确性。IC-Light和SynthLight等方法缺乏显式物理监督，常产生阴影错位、过曝高光、光照方向错误等伪影。

**视频重光照的额外困难**：Light-A-Video采用无训练框架但推理开销大，RelightVid虽有联合训练策略但仍缺乏物理监督，导致光场交互不准确。

**评估体系缺失**：通用指标（FID、LPIPS）无法捕捉光照特有错误，缺少结构化的光照描述和评估协议。

这些问题促使作者设计一个在生成灵活性与物理正确性之间建立桥梁的统一框架。

## 方法详解

### 整体框架

UniLumos基于Wan 2.1 flow matching视频生成模型构建。输入包括退化视频 $\mathbf{V}_{\text{deg}}$、背景视频 $\mathbf{V}_{\text{bg}}$ 和光照条件 $\mathbf{C}$。通过Wan-VAE编码器获取潜表示后，将噪声输入与条件信号沿通道维度拼接注入DiT模块。核心创新包括物理反馈机制和结构化光照标注协议。

### 关键设计

1. **物理可信反馈机制（Physics-Plausible Feedback）**：与纯潜空间操作不同，UniLumos从生成的RGB输出中提取深度图 $\hat{\mathbf{D}}$ 和法线图 $\hat{\mathbf{N}}$（使用Lotus等冻结几何估计器），与输入参考图的精准几何信号对比。几何反馈损失定义为：

$$\mathcal{L}_{\text{phy}} = \mathbb{E}\left[\mathbf{M} \odot \left(\frac{\|\hat{\mathbf{D}} - \mathbf{D}\|_2}{\|\mathbf{D}\|_2} + \frac{\|\hat{\mathbf{N}} - \mathbf{N}\|_2}{\|\mathbf{N}\|_2}\right)\right]$$

其中 $\mathbf{M}$ 是前景主体掩码。深度和法线作为光照不变量，能有效对齐光照效果与场景结构，显著改善阴影对齐和着色一致性。

2. **路径一致性学习（Path Consistency Learning）**：物理反馈需要在RGB域进行监督，而标准多步去噪计算开销巨大。为此引入路径一致性学习，通过强制不同步长的速度预测一致性来支持少步训练：

$$\mathcal{L}_{\text{fast}} = \mathbb{E}\left\|v_\theta(x_t, t, 2d) - \frac{1}{2}\left[v_\theta(x_t, t, d) + v_\theta(x_{t+d}, t+d, d)\right]\right\|_2^2$$

这使模型无需单独的师生蒸馏阶段，即可在任意步数预算下快速生成高质量结果。

3. **LumosData数据管线与六维光照标注**：设计了一个结构化的六维标注协议，涵盖方向、光源类型、强度、色温、时间动态和光学现象。从真实视频中提取重光照对，使用BiRefNet分割前景，用IC-Light生成不同光照条件的重光照版本。标注由Qwen2.5-VL等VLM自动生成，为训练提供细粒度条件控制。

### 损失函数 / 训练策略

联合优化目标整合三个互补损失：

$$\mathcal{L} = \lambda_0 \mathcal{L}_0 + \lambda_1 \mathcal{L}_{\text{fast}} + \lambda_2 \mathcal{L}_{\text{phy}}$$

其中 $\lambda_0 = 1.0$，$\lambda_1 = \lambda_2 = 0.1$。训练采用选择性优化策略：每个批次20%用于路径一致性损失（3次前向+1次反向），80%用于标准flow matching损失，其中50%进一步用RGB空间几何反馈监督。基于Wan2.1-T2V-1.3B-480P，使用AdamW优化器（lr=1e-5），batch size 8，在8块H20 GPU上训练5000次迭代。

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Avg. Score↑ | Dense L2↓ |
|------|-------|-------|--------|-------------|-----------|
| IC-Light | 24.316 | 0.884 | 0.108 | 0.703 | 0.447 |
| SynthLight | 25.572 | 0.905 | 0.102 | 0.791 | 0.214 |
| **UniLumos (图像)** | **26.719** | **0.913** | **0.089** | **0.912** | **0.103** |
| IC-Light逐帧 | 20.132 | 0.851 | 0.133 | 0.672 | 0.432 |
| Light-A-Video+Wan2.1 | 20.784 | 0.876 | 0.129 | 0.682 | 0.371 |
| **UniLumos (视频)** | **25.031** | **0.891** | **0.109** | **0.871** | **0.147** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | Dense L2↓ | 说明 |
|------|-------|-------|-----------|------|
| w/o All Feedback | 21.433 | 0.862 | 0.297 | 去掉全部反馈，性能大幅下降 |
| w/o Normal Feedback | 22.115 | 0.874 | 0.173 | 法线比深度更关键 |
| w/o Depth Feedback | 23.472 | 0.883 | 0.265 | 深度反馈也有显著贡献 |
| w/o Path Consistency | 25.317 | 0.902 | 0.153 | 对质量影响小但效率收益大 |
| Only Video | 22.487 | 0.863 | 0.173 | 缺图像训练质量下降 |
| Only Image | 24.471 | 0.872 | 0.182 | 缺视频训练时序差 |
| **UniLumos** | **25.031** | **0.891** | **0.147** | 统一训练最优平衡 |

### 关键发现

- 法线监督比深度监督更关键：去掉法线导致更大性能下降，说明表面朝向对光影交互的塑造作用超过距离信息
- 路径一致性学习几乎不损失质量但带来巨大效率优势，在少步推理场景下尤其重要
- 统一图像-视频训练范式优于单模态训练，在质量和时序一致性间达到最佳平衡
- 推理速度比现有方法快20倍以上

## 亮点与洞察

- **RGB空间几何反馈是关键创新**：绕过了潜空间无法保证物理正确性的根本问题，通过深度和法线这两个光照不变量建立生成结果与场景结构的显式对齐
- **路径一致性学习解决了反馈-效率矛盾**：RGB域监督需要高质量输出，但多步去噪计算昂贵，路径一致性学习优雅地将两者统一
- **LumosBench提供了光照评估新范式**：基于VLM的属性级评估比像素指标更能捕捉光照控制的细粒度表现

## 局限与展望

- 依赖预训练几何估计器（如Lotus）的质量，在极端场景下估计可能不准
- 六维光照标注依赖VLM自动生成，可能存在标注偏差
- 当前仅在1.3B模型上验证，扩展到更大模型的效果待探讨
- 对复杂材质（如高反射、半透明）的处理能力未做深入分析

## 相关工作与启发

- 与IC-Light、SynthLight等潜空间方法相比，UniLumos的物理反馈思路可推广到其他需要物理约束的生成任务
- 路径一致性学习的应用为加速扩散模型少步训练提供了新思路
- LumosBench的属性级评估思路可应用于其他生成任务的细粒度质量评估

## 评分

- **新颖性**: ⭐⭐⭐⭐ RGB空间几何反馈机制在重光照中是新颖的设计，但各组件分别在其他领域有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ 涵盖图像和视频、多种基线、完整消融和效率分析，还提出了新的评估基准
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，问题阐述到位，但部分公式细节可更精简
- **价值**: ⭐⭐⭐⭐ 20倍加速和显著质量提升对实际应用有重要意义，新benchmark有推广价值

<!-- RELATED:START -->

## 相关论文

- [Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](../../ICLR2026/image_generation/pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)

<!-- RELATED:END -->
