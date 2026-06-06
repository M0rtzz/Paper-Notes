---
title: >-
  [论文解读] SliderSpace: Decomposing the Visual Capabilities of Diffusion Models
description: >-
  [ICCV 2025][图像生成][扩散模型可控生成] SliderSpace 通过对扩散模型在给定提示下生成图像的 CLIP 特征做 PCA 分解，自动发现多个语义正交的可控方向，每个方向训练为 LoRA 适配器（slider），实现了无需人工指定属性的概念分解、艺术风格探索和多样性增强。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型可控生成"
  - "语义分解"
  - "LoRA适配器"
  - "潜在空间探索"
  - "无监督发现"
---

# SliderSpace: Decomposing the Visual Capabilities of Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2502.01639](https://arxiv.org/abs/2502.01639)  
**代码**: [https://github.com/rohitgandikota/sliderspace](https://github.com/rohitgandikota/sliderspace)  
**领域**: 扩散模型  
**关键词**: 扩散模型可控生成、语义分解、LoRA适配器、潜在空间探索、无监督发现

## 一句话总结

SliderSpace 通过对扩散模型在给定提示下生成图像的 CLIP 特征做 PCA 分解，自动发现多个语义正交的可控方向，每个方向训练为 LoRA 适配器（slider），实现了无需人工指定属性的概念分解、艺术风格探索和多样性增强。

## 研究背景与动机

**领域现状**：文本到图像扩散模型能从同一提示通过不同随机种子生成丰富多样的图像变体。然而这种巨大的创意潜力对用户是不透明的——我们只知道模型"能生成不同的图"，却不了解这些变化背后的结构。

**现有痛点**：（1）现有控制方法（ControlNet、IP-Adapter、Concept Sliders）都要求用户预先指定想要控制的属性，这限制了探索性；（2）与 GAN 的低维潜空间不同，扩散模型的高维潜空间缺乏可解释的结构，无法像 StyleGAN 那样做连续、组合式的编辑；（3）蒸馏模型（如 DMD2）虽然快，但存在严重的模式坍缩问题。

**核心矛盾**：扩散模型拥有极其丰富的视觉知识，但用户只能通过文本这个"窄带"接口与之交互。文本无法精确描述的视觉变化（如某种特定的光照风格、某种微妙的构图变化）就无法被访问。

**本文目标**：给定任意文本提示，自动发现扩散模型对该概念知识的主要变化方向，将其暴露为用户可控的 slider，让用户像调 GAN 潜空间一样探索扩散模型。

**切入角度**：既然扩散模型用不同种子生成的图像集合本身就蕴含了模型对概念的不同理解维度，那么对这个分布做主成分分析（PCA）就能发现最重要的变化方向。

**核心 idea**：对扩散模型生成大量图像的 CLIP 特征做谱分解（PCA），用得到的主成分方向作为训练目标，为每个方向训练一个 LoRA 适配器。每个适配器作为 slider 控制一个独立的语义变化维度。

## 方法详解

### 整体框架

SliderSpace 分三步：（1）Distribution Sampling——给定提示 $c$，用不同种子生成约 5000 张图像，在每个时间步 $t$ 提取预估最终图像 $\tilde{x}_{0,t}$；（2）Semantic Decomposition——将所有 $\tilde{x}_{0,t}$ 映射到 CLIP 特征空间，做 PCA 得到 $n$ 个主成分 $\{v_i\}$；（3）Slider Training——为每个主成分方向 $v_i$ 训练一个 LoRA 适配器 $\mathcal{T}_i$，使其在 CLIP 空间中的效果对齐 $v_i$。

### 关键设计

1. **Final Image Extrapolation + CLIP PCA**:

    - 功能：从扩散过程的中间状态提取语义信息用于分解
    - 核心思路：在时间步 $t$，通过公式 $\tilde{x}_{0,t} = (x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_t) / \sqrt{\bar{\alpha}_t}$ 提取预估最终图像，映射到 CLIP 空间 $\phi(\tilde{x}_{0,t})$，然后对所有种子和时间步的 CLIP 特征 $\{\phi(\tilde{x}_{0,t})\}_{j,t}$ 做 PCA 得到主成分 $V = \text{PCA}(\{\phi(\tilde{x}_{0,t})\}_{j,t})$。利用多时间步而非仅最终图像，因为不同时间步隐含了不同层次的语义变化（早期步骤决定构图和布局，晚期步骤决定细节和纹理）
    - 设计动机：直接对像素做 PCA 只能发现颜色和位置的变化，在 CLIP 这种高层语义空间中做 PCA 才能发现真正语义化的变化方向

2. **语义正交 LoRA Slider 训练**:

    - 功能：将抽象的 PCA 方向转化为可控的模型编辑器
    - 核心思路：对每个主成分 $v_i$，训练 LoRA 适配器 $\mathcal{T}_i$。训练目标是最大化适配器引起的 CLIP 特征变化 $\Delta\phi_i = \phi(\tilde{x}_{0,t}^i) - \phi(\tilde{x}_{0,t})$ 与 $v_i$ 的余弦相似度：$\mathcal{L}_{\text{sliderspace}} = \sum_{i=1}^n (1 - \cos(\Delta\phi_i, v_i))$。由于 PCA 方向天然正交，训练好的 sliders 控制不同的语义维度
    - 设计动机：LoRA 的低秩结构确保每个 slider 的参数量很小（可组合、可叠加），且比直接在 embedding 空间操作更稳定（直接操作 CLIP embedding 无法保证图像质量）

3. **Slider 组合控制与迁移**:

    - 功能：支持多 slider 叠加使用和跨概念迁移
    - 核心思路：多个 LoRA 适配器可以同时叠加到模型权重上，用不同缩放因子调节强度。由于语义正交性，组合效果近似线性叠加。此外在"person"概念上训练的 slider 可以迁移到"police"、"athlete"甚至"dog"等相关概念上
    - 设计动机：LoRA 的加法组合特性天然支持多 slider 协同；迁移能力来自扩散模型内部表征的共享结构

### 损失函数 / 训练策略

训练损失为 $\mathcal{L} = \sum_{i=1}^n (1 - \cos(\Delta\phi_i, v_i))$，使用 rank-1 LoRA 适配器（发现 rank-1 在固定训练预算下效率最高），默认 40 个 PCA 方向。在单张 A100 GPU 上约 2 小时完成 64 个 slider 的发现。

## 实验关键数据

### 主实验：概念分解多样性评估

在 6 个概念上比较 SliderSpace 增强生成 vs 基础模型的图像多样性和文本对齐度：

| 概念 | DreamSim 多样性 ↑ | CLIP-Score 文本对齐 ↓ |
|------|------|------|
| Mountain (SDXL-DMD) | 低 | ~25 |
| Mountain (SliderSpace) | **高 ~0.5** | ~25 |
| Monster (SDXL-DMD) | 低 | ~32 |
| Monster (SliderSpace) | **高 ~0.55** | ~33 |
| Person (SDXL-DMD) | 低 | ~20 |
| Person (SliderSpace) | **高 ~0.45** | ~20 |

用户研究：SliderSpace 在多样性/实用性/创意性上的胜率分别为 72.4%/66.0%/68.1%（vs SDXL-DMD）。

### 艺术风格探索实验

| 方法 | FID ↓ (vs ParrotZone 参考分布) |
|------|------|
| Generic Art Prompts | 高 |
| LLM-generated Art Prompts | 中 |
| Concept Sliders (64 个手动) | 中等 |
| **SliderSpace (64 个自动)** | **最低** |

SliderSpace 仅用 10 个方向就匹配了 64 个手动 Concept Sliders 的 FID。

### 消融实验

| 配置 | FID 效果 | 说明 |
|------|---------|------|
| SliderSpace (完整) | 最优 | CLIP 空间 PCA + 正交训练 |
| 无正交约束 (naive 多 LoRA) | 差 | 大量冗余/无意义方向 |
| 像素空间 PCA | 中等 | 发现颜色/形状变化但非语义 |
| 无 diversity expansion | 略差 | 蒸馏模型训练数据不够多样 |

### 关键发现

- **Rank-1 LoRA 最高效**：在固定训练预算下，rank-1 适配器优于更高秩版本
- **PCA 方向数量 ~40 时饱和**：超过 40 个方向后，边际收益减小
- **CLIP vs DINO-v2**：两者整体效果相当，但 FaceNet 在人脸概念上发现更细粒度的方向
- **时间步选择**：在所有步骤应用 slider 会改变整体结构；跳过前几步可实现更精确的局部编辑
- **多样性增强效果显著**：在 COCO-30k 上，DMD-SliderSpace 的 FID 从 15.52 降至 12.12，接近未蒸馏 SDXL 的 11.72

## 亮点与洞察

- **GAN latent space 精神的回归**：扩散模型一直缺少 GAN 那种"拖拽潜空间探索"的交互方式，SliderSpace 用 PCA + LoRA 优雅地弥补了这一缺失。用户可以像调 StyleGAN 一样"拨动旋钮"探索扩散模型的视觉能力
- **自监督 = 无需标注的概念发现**：不需要任何语义标注，仅从模型自身生成的样本中就能发现有意义的控制方向。这表明扩散模型内部确实存在结构化的概念表征
- **解决蒸馏模型的模式坍缩**：SliderSpace 发现的 diversity sliders 可以将蒸馏模型的多样性恢复到接近原始模型水平，这对实际部署非常有价值

## 局限与展望

- 依赖 CLIP 等语义编码器，其训练数据的偏差会传递到发现的方向中
- 发现过程需要约 2 小时（A100），对快速迭代有一定限制
- 发现的方向不一定与人类认知的"自然属性"一一对应，有些方向可能混合了多个语义
- 艺术风格方向不一定与真实艺术家一一对应
- 未来可以探索：更高效的发现算法、与用户交互的方向细化、跨模型方向迁移

## 相关工作与启发

- **vs Concept Sliders（Gandikota et al. 2024）**：Concept Sliders 需要用户指定属性（如"年龄"、"微笑"），SliderSpace 自动发现属性。两者互补——SliderSpace 帮用户"发现不知道的"，Concept Sliders 帮用户"精确控制已知的"
- **vs NoiseCLR（Dalva & Yanardag 2024）**：NoiseCLR 用对比学习在 text embedding 空间发现方向，但结果可能不具语义解释性。SliderSpace 在 CLIP 视觉空间做 PCA 更直接，且有正交保证
- **vs weights2weights（Dravid et al. 2024）**：w2w 需要为每个个体训练独立的 LoRA 再做 PCA，成本很高。SliderSpace 直接在生成分布上做 PCA，更高效
- 该方法暗示扩散模型在训练过程中自发形成了结构化的内部表征，这对理解生成模型的内部机制有启发

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 PCA 分解 + LoRA 训练优雅结合，实现了扩散模型的"GAN-style"潜空间探索
- 实验充分度: ⭐⭐⭐⭐⭐ 三个应用场景、用户研究、完整消融、多模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 表述清晰，公式推导简洁，图示精美
- 价值: ⭐⭐⭐⭐⭐ 实用性强（解决模式坍缩、增强可控性），理论洞察深刻（揭示模型内部结构）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity](diffsim_taming_diffusion_models_for_evaluating_visual_similarity.md)
- [\[AAAI 2026\] LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](../../AAAI2026/image_generation/longllada_unlocking_long_context_capabilities_in_diffusion_llms.md)
- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](../../CVPR2025/image_generation/the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[ICCV 2025\] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)

</div>

<!-- RELATED:END -->
