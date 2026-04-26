---
title: >-
  [论文解读] End-to-End Multi-Modal Diffusion Mamba
description: >-
  [ICCV 2025][图像生成][多模态模型] 提出 Multi-Modal Diffusion Mamba（MDM），一种基于 Mamba 架构的端到端多模态模型，通过统一的 VAE 编解码器和多步选择性扩散模型，实现图像和文本的同时生成，计算复杂度为 $\mathcal{O}(MLN^2)$，在图像生成、图像描述、VQA 等多任务上超越现有端到端模型。
tags:
  - ICCV 2025
  - 图像生成
  - 多模态模型
  - Mamba
  - 扩散模型
  - 端到端生成
  - 状态空间模型
---

# End-to-End Multi-Modal Diffusion Mamba

**会议**: ICCV 2025  
**arXiv**: [2510.13253](https://arxiv.org/abs/2510.13253)  
**代码**: 无  
**领域**: 扩散模型/图像生成  
**关键词**: 多模态模型, Mamba, 扩散模型, 端到端生成, 状态空间模型

## 一句话总结

提出 Multi-Modal Diffusion Mamba（MDM），一种基于 Mamba 架构的端到端多模态模型，通过统一的 VAE 编解码器和多步选择性扩散模型，实现图像和文本的同时生成，计算复杂度为 $\mathcal{O}(MLN^2)$，在图像生成、图像描述、VQA 等多任务上超越现有端到端模型。

## 研究背景与动机

当前多模态大模型面临三大结构性挑战：

**传统多编/解码器架构**（如 LLaVA、Flamingo）：使用不同的编码器和解码器处理各模态，阻碍了联合表示学习，推理时间长

**端到端 Transformer 模型的瓶颈**：
   - **二次复杂度**：Transformer 的 $O(L^2)$ 复杂度使其在高分辨率图像和长序列文本生成中效率低下
   - **多目标冲突**：同时优化图像和文本的目标函数存在优化冲突，阻碍收敛和联合表示学习

**现有端到端方案的局限**：
   - 自回归模型（如 Chameleon）受限于顺序依赖
   - 混合生成模型（如 Seed-X）引入额外复杂度
   - 混合自回归-扩散模型（如 MonoFormer）仍难以统一多模态处理

Mamba/状态空间模型（SSM）因线性复杂度和长距离依赖建模能力成为有前景的替代方案，但现有 Mamba 多模态工作仍采用多目标方法，缺乏真正的端到端联合表示学习。

MDM 的核心贡献是将 Mamba 与扩散过程结合，用**统一的 VAE 编解码器**处理所有模态，用**多步选择性扩散模型**作为统一的生成框架，通过 Score Entropy Loss 替代 Markov 链方法提升效率。

## 方法详解

### 整体框架

MDM 架构分为三个部分：(1) VAE 噪声潜空间编码器将图像/文本统一映射到噪声潜空间；(2) 基于 Mamba 的多步选择性扩散解码器逐步去噪恢复数据；(3) VAE 噪声潜空间解码器将去噪后的潜变量重建为图像或文本。整个流程同时处理和生成多模态数据。

### 关键设计

1. **VAE 统一编解码器**:

    - 功能：将图像patch和文本embedding统一编码到同一噪声潜空间
    - 核心思路：图像通过 patchify 操作，文本通过 SentencePiece+BPE 分词和 embedding，然后统一经过 VAE 采样得到 $z_n = s_n + \epsilon_n$。编码器生成高斯分布参数 $(\mu, \sigma)$，加入可学习的填充 token（time、class、pad）。
    - 设计动机：统一的编解码器消除了模态间的表示鸿沟，使模型能够在同一潜空间中学习真正的联合多模态表示。

2. **多步选择性扩散模型（Multi-step Selection Diffusion）**:

    - 功能：结合扩散过程和 Mamba 的选择机制来逐步生成多模态信息
    - 核心思路：前向扩散遵循标准公式：
    $z_{n,t}^g = \sqrt{\bar{\alpha}_t^g} z_{n,0}^g + \sqrt{1-\bar{\alpha}_t^g} \epsilon_{n,t}^g$
      去噪不使用传统 Markov 链，而是采用 Score Entropy Loss（SE）作为统一目标：
    $se = \sum_{y} \omega_{z_{n,t}^g}^g \left(s_\theta(z_{n,t}^g) - \frac{p_{data}(y)}{p_{data}(z_{n,t}^g)} \log s_\theta(z_{n,t}^g) + K(\cdot)\right)$
      选择过程通过 Mamba 的 SSM 选择机制，根据 score ratio 是否趋近实际比率来决定关注或忽略哪些信息（Theorem 3: $s_\theta(z_{n,t}^g) \approx \frac{p_{data}(y)}{p_{data}(z_{n,t}^g)}$）。
    - 设计动机：SE 是一种广义的 score matching 目标，直接学习离散状态间的概率密度比，比 Markov 链方法在高维空间中更高效，且更易扩展到离散数据（文本）。

3. **图像/文本扫描切换 + Mamba Block**:

    - 功能：通过不同方向的序列扫描捕获时序关系，然后用 Mamba-2 的 SSM 进行信息选择
    - 核心思路：图像使用 4 种扫描方向（参考 DiM），文本使用 2 种扫描方向。Mamba Block 内部按 SSM 更新状态：
    $H_{n,t}^g = \bar{A} H_{n,t-1}^g + \bar{B} z_{n,t}^g$
    $z_{n-1,t}^g = C H_{n,t}^g + D z_{n,t}^g$
      去噪步用 DPM-Solver 二阶方法提升采样精度：
    $z_{n,t-\Delta t}^g = z_{n,t}^g - \frac{\Delta t}{2}[f_\theta(z_{n,t}^g, t) + f_\theta(z_{n,t-\Delta t}^g, t-\Delta t)]$
    - 设计动机：Mamba 的选择机制天然适合扩散去噪——每步需要决定哪些信息已经足够清晰可以保留，哪些仍是噪声需要修正。多方向扫描确保捕获不同空间关系。

### 损失函数 / 训练策略

总体优化目标结合四个组件：

$$L_{total} = L_{rec}^{img} + L_{rec}^{txt} + \beta L_{KL} + \lambda L_{se}$$

- $L_{rec}^{img}$：图像的 L2 重建损失
- $L_{rec}^{txt}$：文本的交叉熵损失
- $L_{KL}$：VAE 的 KL 散度正则化
- $L_{se}$：Score Entropy Loss

模型共 7B 参数，49 个 Mamba Block，维度 2048。

## 实验关键数据

### 主实验

图像生成（ImageNet & COCO 256×256）：

| 模型 | 类型 | 参数 | FID↓ | IS↑ | Precision | Recall |
|------|------|------|------|-----|-----------|--------|
| DiT-XL/2 | Diff | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| LlamaGen | AR | 3.1B | 2.81 | 311.5 | 0.84 | 0.54 |
| MonoFormer | AR+Diff | 1.1B | 2.57 | 272.6 | 0.84 | 0.56 |
| **MDM** | **Diff** | **7B** | **2.49** | **281.4** | **0.86** | **0.59** |

文本-图像生成（COCO）：

| 模型 | FID↓ | GenEval↑ |
|------|------|---------|
| SDXL | 4.40 | 0.55 |
| Chameleon | 26.74 | 0.39 |
| Transfusion | 6.78 | 0.63 |
| **MDM** | **5.91** | **0.68** |

多任务综合评测：

| 模型 | IC-COCO | VQAv2 | PIQA | MMLU | GSM8k |
|------|---------|-------|------|------|-------|
| Chameleon (34B) | 120.2 | 66.0 | 79.6 | 52.1 | 41.6 |
| NExT-GPT (7B) | 124.9 | 66.7 | — | — | — |
| **InstructMDM (7B)** | **122.1** | **66.7** | **83.7** | **54.4** | **46.0** |
| Mistral (7B, 纯文本) | — | — | 83.0 | 60.1 | 52.1 |

### 消融实验

| 配置 | ImageNet FID↓ | COCO FID↓ | 说明 |
|------|-------------|----------|------|
| MDM w/o selection | 3.21 | 7.84 | 无选择机制，全量处理 |
| MDM w/ 1种扫描 | 2.85 | 6.73 | 单方向扫描 |
| MDM w/ SE loss | **2.49** | **5.91** | 完整模型（SE+多方向） |
| MDM w/ Markov chain | 2.97 | 6.92 | 用传统 DDPM 替代 SE |

计算复杂度对比：

| 模型 | 复杂度 | 说明 |
|------|--------|------|
| MonoFormer | $O(ML^2N/G)$ | Transformer 二次复杂度 |
| **MDM** | **$O(MLN^2)$** | Mamba 线性序列复杂度 |

### 关键发现

- MDM 在 ImageNet FID (2.49) 上与 DiT-XL/2 (2.27) 和 MonoFormer (2.57) 具有竞争力，同时具备多任务能力
- 在文本-图像生成上，MDM (5.91 FID, 0.68 GenEval) 大幅超越 Chameleon (26.74, 0.39) 等自回归端到端模型
- InstructMDM 在文本理解任务上接近甚至超越同等规模的纯文本模型（如 PIQA 83.7 vs Mistral 83.0）
- Mamba 的线性复杂度使其在处理高分辨率图像和长文本时比 Transformer 端到端模型更高效

## 亮点与洞察

- **真正的端到端多模态**：单一 VAE + 单一 Mamba 解码器，无需任何模态特定的编解码器或融合模块
- **扩散+Mamba 的创新结合**：利用 Mamba 的选择机制来引导扩散去噪方向，理论上有 Theorem 3 的支撑
- **Score Entropy Loss**：统一了连续（图像）和离散（文本）模态的生成目标，比 Markov 链方法更高效
- **同时生成多模态**：不像其他模型先生成文本再生成图像，MDM 可以同时输出图像和对应描述

## 局限与展望

- 7B 参数量较大，对于端到端模型来说部署成本不低
- 图像描述指标（如 Flickr30K CIDEr 62.4）与专用模型（如 GPT-4V 55.3 虽低但评价体系不同）相比仍有提升空间
- Score Entropy Loss 的理论分析基于离散状态，在连续潜变量上的严格性需要进一步论证
- 缺乏在更高分辨率（如 512×512 或 1024×1024）上的图像生成评测
- 视频生成能力未被探索

## 相关工作与启发

- 与 MonoFormer 和 Transfusion 直接竞争：都是混合自回归-扩散的端到端模型，但 MDM 用 Mamba 替代 Transformer 实现线性复杂度
- 借鉴 DiM 的多方向扫描策略用于图像，将其扩展到文本模态
- Score Entropy Discrete Diffusion (SEDD) 的 SE loss 被巧妙地应用于多模态统一学习目标
- 启示：Mamba 的选择机制与扩散去噪的"决定保留什么、去除什么"天然契合

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Mamba+扩散的端到端多模态组合属于首创，架构设计富有想象力
- 实验充分度: ⭐⭐⭐⭐ 覆盖图像生成、VQA、文本理解等多任务，但高分辨率和视频生成评测缺失
- 写作质量: ⭐⭐⭐⭐ 架构描述详细，公式推导完整，但整体篇幅较长
- 价值: ⭐⭐⭐⭐ 提出了一个有前景的新方向（Mamba端到端多模态），但与 Transformer 基线相比优势尚未完全拉开

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)

<!-- RELATED:END -->
