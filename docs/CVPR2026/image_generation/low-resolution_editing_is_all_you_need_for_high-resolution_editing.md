---
title: >-
  [论文解读] Low-Resolution Editing is All You Need for High-Resolution Editing
description: >-
  [CVPR 2026][图像生成][高分辨率图像编辑] ScaleEdit 首次提出高分辨率图像编辑任务，通过在预训练生成模型的中间特征空间学习 1×1 卷积迁移函数来注入源图像的精细纹理细节，配合基于 Blended-Tweedie 的分块同步策略保证全局一致性…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "高分辨率图像编辑"
  - "测试时优化"
  - "细节迁移函数"
  - "分块同步"
  - "扩散模型"
---

# Low-Resolution Editing is All You Need for High-Resolution Editing

**会议**: CVPR 2026  
**arXiv**: [2511.19945](https://arxiv.org/abs/2511.19945)  
**代码**: 无  
**领域**: 扩散模型 / 图像编辑  
**关键词**: 高分辨率图像编辑, 测试时优化, 细节迁移函数, 分块同步, 扩散模型

## 一句话总结
ScaleEdit 首次提出高分辨率图像编辑任务，通过在预训练生成模型的中间特征空间学习 1×1 卷积迁移函数来注入源图像的精细纹理细节，配合基于 Blended-Tweedie 的分块同步策略保证全局一致性，以测试时优化方式实现 2K 甚至 8K 分辨率的高质量编辑。

## 研究背景与动机

1. **领域现状**：文本驱动的图像编辑方法（如 Step1X-Edit、ICEdit、KV-Edit、Nano Banana）在低分辨率（≤1K）下取得了出色效果，但受限于预训练模型的输入分辨率，无法直接处理更大尺寸的图像。

2. **现有痛点**：朴素方案是先低分辨率编辑再超分辨率，但超分方法无法恢复源图像中的微观纹理细节，因为编辑过程中未以高分辨率源图为条件——细节信息在降分辨率时已丢失且无法从低分编辑结果中重建。

3. **核心矛盾**：高分辨率编辑需要同时保持语义正确性和精细纹理保真度，但预训练生成模型的分辨率固定（通常 512²），直接在高分辨率上操作不可行。

4. **本文目标** 如何利用低分辨率编辑方法的强先验，同时忠实保留高分辨率源图像中的精细细节？

5. **切入角度**：核心观察——低分辨率轨迹和高分辨率轨迹在扩散过程的中间特征空间中有可学习的映射关系。通过一个轻量的特征迁移函数学习这种映射，就可以将高分辨率细节注入到低分辨率编辑结果中。

6. **核心 idea**：用可学习的 1×1 卷积作为特征迁移函数，在逆扩散过程中将高分辨率源图像的精细细节注入到低分辨率编辑结果的生成轨迹中，配合非重叠分块同步消除边界伪影。

## 方法详解

### 整体框架
给定高分辨率源图 $I_{\text{src}}^{\text{high}}$、其降采样版本 $I_{\text{src}}^{\text{low}}$ 和低分辨率编辑参考 $I_{\text{ref}}^{\text{low}}$（由标准编辑方法如 Nano Banana 生成），目标是生成高分辨率编辑结果 $I_{\text{ref}}^{\text{high}}$。方法分三步：(1) 将所有图像分为 $N \times M$ 个与模型原生分辨率匹配的非重叠 patch，通过 DDIM 前向过程提取各 patch 的扩散轨迹；(2) 对每个 patch 学习特征迁移函数，将高分辨率源的细节注入编辑结果；(3) 通过 Blended-Tweedie + 重采样策略同步相邻 patch，消除边界伪影。

### 关键设计

1. **细节增强模块 (Detail Enhancement Module)**:

    - 功能：将高分辨率源图的精细纹理细节迁移到编辑后的目标图像中
    - 核心思路：在预训练生成模型的中间特征空间定义时步相关的迁移函数 $\Delta\mathbf{h}_t[i] = \phi_\theta(\mathbf{h}_t[i], t)$，实现为 1×1 卷积。优化目标是将低分辨率源的生成轨迹引导向高分辨率源的轨迹：$\mathcal{L} = \|\mathbf{x}_{t-1}^{high}[i] - f^{rev}(\tilde{\mathbf{x}}_t[i], t; \Delta\mathbf{h}_t[i])\|_2^2$。优化后的迁移函数应用到参考图的反向过程中，实现细节注入。引入控制参数 $\tau$ 限制迁移函数仅在前 $\tau$ 个时步生效，平衡细节迁移和内容保留
    - 设计动机：直接用常量向量做特征偏移无法处理语义变化大的编辑（如猫→狗），因为图像不同区域需要不同程度的调整。1×1 卷积允许通道级自适应混合，保持空间布局的同时实现精细细节迁移

2. **混合 Tweedie 同步 (Blended-Tweedie Synchronization)**:

    - 功能：确保相邻分块边界处的视觉一致性
    - 核心思路：构造辅助 latent $\tilde{\mathbf{A}}_t[i,i+1]$——拼接当前块的下半部和相邻块的上半部。计算辅助 latent 的 Tweedie 估计 $\hat{\mathbf{x}}_{t\to 0}^{aux}$，与原始块的 Tweedie 估计进行线性插值混合，权重 $\mathbf{M}(v,t) = \frac{2v}{H_p} \cdot (1 - t/\tau)$ 从边界到中心线性增长，且随时步推进增大混合强度
    - 设计动机：独立反向去噪的分块在边界处会产生不连续，辅助 latent 跨越边界区域，其 Tweedie 估计自然捕获了更平滑的过渡，混合后实现块间一致

3. **重采样策略 (Resampling Strategy)**:

    - 功能：解决辅助 latent 没有对应迁移函数 $\Delta\mathbf{h}_t$ 的问题
    - 核心思路：对已注入细节的 latent $\tilde{\mathbf{y}}_{t-1}[i]$ 做一步前向过程（不带迁移函数）$\tilde{\mathbf{y}}_t^{rsp}[i] = f^{fwd}(\tilde{\mathbf{y}}_{t-1}[i], t-1)$，得到保留了细节但不依赖 $\Delta\mathbf{h}_t$ 的重采样 latent。然后用重采样 latent 构造辅助 latent 做同步，最后用混合 Tweedie + 重采样 latent 的噪声预测完成实际反向步
    - 设计动机：为辅助 latent 单独优化迁移函数计算开销太大。重采样将同步和细节注入解耦，只需额外一次前向+反向操作即可

### 损失函数 / 训练策略

纯测试时优化，无需训练。迁移函数对每个分块、每个时步独立优化。使用 Stable Diffusion v2.1-base 或 FLUX.1-dev，总时步 $T=50$，$\tau=15$，空 prompt。配合 Null-text inversion 实现准确重建。低分辨率编辑用 Nano Banana 方法完成。

## 实验关键数据

### 主实验

| 方法 | HaarPSI↑ | M-MSE↓ | M-SSIM↑ | M-PSNR↑ | LPIPS↓ |
|------|---------|--------|---------|---------|--------|
| **1K-editing** | | | | | |
| DiT-SR | 0.335 | 0.058 | 0.695 | 21.53 | 0.477 |
| PiSA-SR | 0.328 | 0.058 | 0.668 | 21.27 | 0.465 |
| **ScaleEdit (Ours)** | **0.342** | **0.054** | **0.739** | **22.13** | **0.460** |
| **2K-editing** | | | | | |
| DiT-SR | 0.316 | 0.057 | 0.754 | 21.38 | 0.507 |
| PiSA-SR | 0.312 | 0.056 | 0.755 | 21.32 | 0.472 |
| **ScaleEdit (Ours)** | **0.331** | **0.053** | **0.806** | **21.96** | **0.496** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无同步 | 可见边界伪影 | 分块独立去噪产生明显接缝 |
| 有同步 | 边界平滑自然 | Blended-Tweedie + 重采样消除伪影 |
| 常量向量 vs 1×1 卷积 | 后者显著更好 | 空间自适应的迁移函数更鲁棒 |

### 关键发现

- ScaleEdit 在所有指标上一致超越 SR 基线方法，验证了"编辑后超分"这种 pipeline 无法恢复源图细节的论点
- Masked 指标（M-MSE、M-SSIM、M-PSNR）的优势尤其明显，说明方法更好地保留了源图中应保持不变的区域
- 方法可泛化到 FLUX 等 Transformer 架构，不限于 U-Net 架构
- 可扩展到 8K 分辨率编辑，无需额外训练

## 亮点与洞察

- 首次正式定义了高分辨率图像编辑任务，将其与简单的"编辑+超分"pipeline 区分开
- 迁移函数的设计巧妙——用 1×1 卷积在特征空间实现通道级自适应混合，既轻量又有效
- 无重叠的同步策略显著降低了计算开销——传统方法需要重叠推理，计算量随重叠比例增长
- 测试时优化框架不需要任何训练数据，对任意编辑方法和生成模型都适用

## 局限与展望

- 测试时优化每张图像都需要独立计算，推理速度较慢（需要多次前向+反向+优化式迭代）
- 超参数 $\tau$ 需要手动设置来控制细节迁移和内容保留的平衡
- 依赖低分辨率编辑结果的质量——如果低分编辑失败，高分辨率结果也无法挽救
- 分块大小固定为模型原生分辨率，无法灵活调整
- 仅展示了基于 Stable Diffusion 和 FLUX 的结果，未验证其他架构

## 相关工作与启发

- Null-text inversion 的思路被借鉴到迁移函数的设计中——通过优化可学习参数对齐前向和反向轨迹
- 分块同步的挑战在视频扩散和全景生成中也存在，本文的 Blended-Tweedie 策略可能有更广泛的应用
- 高分辨率内容创作是产业刚需，但学术界对此关注不足

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义高分辨率编辑任务，迁移函数和无重叠同步策略设计新颖
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+消融+8K 演示，但数据集规模较小（100张源图）
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法推导严谨，图示精美
- 价值: ⭐⭐⭐⭐ 高分辨率编辑是实际需求，框架通用性强可作为即插即用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation](../../ECCV2024/image_generation/you_only_need_one_step_fast_super-resolution_with_stable_diffusion_via_scale_dis.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)

</div>

<!-- RELATED:END -->
