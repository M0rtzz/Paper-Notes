---
title: >-
  [论文解读] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration
description: >-
  [CVPR 2025][图像生成][图像退化合成] 本文提出GenDeg，一个基于Stable Diffusion的退化合成框架，能在任意干净图像上生成多种可控退化（雾/雨/雪/运动模糊/低光/雨滴），合成55万+图像构成GenDS数据集，训练在其上的All-In-One复原模型在域外测试集上获得显著性能提升。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像退化合成"
  - "扩散模型"
  - "All-In-One复原"
  - "域外泛化"
  - "合成数据"
---

# GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2411.17687](https://arxiv.org/abs/2411.17687)  
**代码**: [https://sudraj2002.github.io/gendegpage/](https://sudraj2002.github.io/gendegpage/) (项目页面)  
**领域**: 图像生成  
**关键词**: 图像退化合成, 扩散模型, All-In-One复原, 域外泛化, 合成数据

## 一句话总结
本文提出GenDeg，一个基于Stable Diffusion的退化合成框架，能在任意干净图像上生成多种可控退化（雾/雨/雪/运动模糊/低光/雨滴），合成55万+图像构成GenDS数据集，训练在其上的All-In-One复原模型在域外测试集上获得显著性能提升。

## 研究背景与动机
1. **领域现状**：All-In-One Image Restoration（AIOR）用单一模型处理多种退化，代表方法有PromptIR、DA-CLIP、Diff-Plugin等。
2. **现有痛点**：现有AIOR模型在训练分布外（OoD）的退化模式和场景上泛化很差。①现有数据集规模小（远小于SAM/Depth-Anything的1.5M+），场景多样性不足；②合成数据集使用单一物理模型（如RESIDE只用大气雾模型），退化模式单一；③真实退化数据（如雾、低光、雨滴）采集困难，样本稀缺。
3. **核心矛盾**：AIOR方法过拟合训练分布，根因是训练数据在退化模式和场景多样性上均不足，而真实数据采集不可行。
4. **本文目标** 用生成模型合成大规模、多样化的退化数据来提升AIOR的OoD泛化能力。
5. **切入角度**：Latent Diffusion Model具有强大的生成先验和条件控制能力，可以在保持场景语义的同时生成逼真的退化模式。
6. **核心 idea**：用退化强度感知的条件扩散模型合成大规模多样退化数据，从数据侧解决All-In-One复原的泛化问题。

## 方法详解

### 整体框架
GenDeg基于InstructPix2Pix架构，输入为干净图像 $c_{img}$ + 文本prompt $c_{text}$ + 退化强度条件，输出退化图像。训练时用多个现有数据集的退化-干净配对数据，推理时在新的干净图像上生成退化。生成后的图像经过Structure Correction Module (SCM)校正，最终与原始数据集合并成GenDS数据集（75万+样本），用于训练复原模型。

### 关键设计

1. **退化强度感知的条件扩散模型**:
    - 功能：在生成退化时精细控制退化强度和空间分布
    - 核心思路：定义退化图 $c_{map} = |x_{in} - c_{img}|$，计算其均值 $\mu$ 和标准差 $\sigma$ 量化退化强度。将 $\mu$ 和 $\sigma$ 各编码为129维的one-hot向量（128个bin + 1个null），拼接后投影到 $\mathbb{R}^{77 \times 2}$，与CLIP文本嵌入 $e_{text} \in \mathbb{R}^{77 \times 768}$ 拼接后再投影回768维，作为Stable Diffusion的条件输入。
    - 设计动机：仅用文本prompt（如"hazy"）会导致扩散模型生成极端退化（过浓的雾或过密的雨），$\mu$-$\sigma$ 条件让模型感知目标退化强度，生成更真实可控的退化。

2. **结构校正模块 (SCM)**:
    - 功能：修复VAE编解码过程中丢失的细节
    - 核心思路：SCM是一个轻量网络 $S$，输入为生成图和干净图的拼接，输出残差：$x_S = x_{gen} + S([x_{gen}, c_{img}])$。训练时使用一步反向扩散获取生成图，损失函数为时间步加权的L2损失 $L_S = \sqrt{\bar{\alpha}_{t-1}} \cdot \sqrt{1-\bar{\alpha}_t} \cdot \|x_{in} - x_S\|_2^2$，权重在初始和末尾时间步较低。
    - 设计动机：LDM的VAE编解码会丢失细节，SCM仅对雾/雨滴/运动模糊等平滑退化有效；对雨/雪/低光则用VAE重编码干净图像 $\hat{c}_{img}$ 代替。

3. **数据生成与质量控制**:
    - 功能：大规模生成高质量退化配对数据
    - 核心思路：从训练数据集的约12万张干净图像出发，每张生成5种不属于其原始数据集的退化。$\mu_{gen}$ 从目标数据集的直方图中采样，$\sigma_{gen}$ 从对应 $\mu$ bin内的直方图中采样（保证统计相关性）。每20张图有1张使用随机 $\sigma$ 增加多样性。生成后基于退化图均值过滤质量差的图像。
    - 设计动机：从数据集直方图采样保证了退化强度的真实分布；交叉退化（在A类退化图像上生成B类退化）增加了场景-退化的组合多样性。

### 损失函数 / 训练策略
- GenDeg训练使用标准LDM去噪目标（公式1）
- 提出Swin Transformer复原网络：使用ImageNet预训练Swin编码器 + 轻量卷积解码器，层次化特征聚合，3×3卷积避免patch边界伪影
- 同时训练5个复原模型：NAFNet、PromptIR、Swin、DA-CLIP、Diff-Plugin

## 实验关键数据

### 主实验（OoD性能，LPIPS/FID，越低越好）

| 方法 | REVIDE(雾) | O-Haze(雾) | GoPro(模糊) | LOLv1(低光) | RainDS(雨滴) |
|------|----------|----------|-----------|----------|-----------|
| PromptIR | 0.262/62.0 | 0.333/150.9 | 0.186/32.9 | 0.258/111.8 | 0.208/106.8 |
| **PromptIR+GenDS** | **0.212/56.0** | **0.160/89.0** | 0.191/31.9 | **0.178/87.9** | **0.182/79.8** |
| NAFNet | 0.211/71.3 | 0.183/99.2 | 0.155/28.2 | 0.167/78.8 | 0.178/73.4 |
| **NAFNet+GenDS** | **0.151/52.5** | **0.143/76.7** | 0.149/28.7 | **0.147/63.7** | **0.170/60.5** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅现有数据 | 基准性能，OoD泛化差 |
| +GenDS数据 | 所有5个模型在OoD上均显著提升 |
| NAFNet提升最大 | REVIDE雾上LPIPS从0.211→0.151 (-28.4%) |
| 域内性能基本保持 | 加入GenDS后域内性能无明显下降 |

### 关键发现
- 所有5个复原模型（非生成式+生成式）在加入GenDS后OoD性能均显著提升，证明了合成数据的普适价值
- 去雾任务提升最显著（O-Haze上PromptIR FID从150.9降到89.0），因为雾的现有真实数据最匮乏
- t-SNE可视化显示GenDS生成的退化特征分布能有效桥接现有训练数据与OoD测试数据之间的域差距
- GenDS是首个同一干净图像具有多种退化版本的数据集，天然适合AIOR训练
- $\mu$-$\sigma$ 条件控制对退化的逼真度至关重要，无条件生成会产生极端退化

## 亮点与洞察
- **"从数据侧解决泛化"的范式**非常有价值：不改模型架构，只改训练数据，就让5个不同模型全部获得提升，证明了数据质量>模型复杂度
- **退化强度的$\mu$-$\sigma$条件化**是关键创新：将退化强度编码为与CLIP嵌入融合的条件信号，兼顾了全局强度和空间分布的控制
- **跨退化数据集训练**让GenDeg不依赖单一物理模型，综合学习了多种退化模式的生成能力
- 该方法可直接迁移到其他需要大规模配对训练数据的低层视觉任务，如超分辨率、去噪等

## 局限与展望
- 目前仅覆盖6种退化类型，未包括JPEG压缩、噪声、过曝等常见退化
- SCM对雨/雪/低光退化不适用，需要分别处理，统一性不足
- 生成图像的质量仍依赖于VAE的重建能力，高频细节损失无法完全避免
- 数据量（75万）虽然比现有数据集大，但远未达到SAM级别（1100万），继续扩大规模可能带来更大提升
- 退化之间的复合效果（如雾+雨）未被考虑

## 相关工作与启发
- **vs InstructPix2Pix**: GenDeg在其基础上增加了退化强度条件（$\mu$/$\sigma$），从图像编辑转型为退化合成
- **vs DA-CLIP**: DA-CLIP用CLIP的退化特征引导复原；GenDeg从数据侧增强，两者互补——实验中DA-CLIP+GenDS也获得显著提升
- **vs PromptIR**: PromptIR用可学习prompt识别退化类型；GenDS数据可直接提升PromptIR的OoD性能
- 这种"用生成模型合成训练数据"的范式与classification领域的数据增强思路类似，但在细节保持上要求更高

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个用扩散模型合成退化数据提升复原模型泛化的系统性工作，$\mu$-$\sigma$条件机制有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 5个不同模型、6种退化、多个OoD/域内数据集、t-SNE可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，数据分析详实，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 提供了可直接使用的75万样本数据集和退化合成工具，对社区有直接推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](../../ICCV2025/image_generation/mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[CVPR 2025\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)

</div>

<!-- RELATED:END -->
