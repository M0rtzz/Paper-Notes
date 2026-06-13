---
title: >-
  [论文解读] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation
description: >-
  [ICCV 2025][图像生成][双曲空间] 将双曲空间的层级表示学习能力与扩散自编码器的高质量生成能力结合，通过在 Poincaré 圆盘中操控潜码的半径和方向，实现可控、多样且保持类别一致性的小样本图像生成。 小样本图像生成（Few-shot Image Generation）旨在仅用少量样本为未见类别生成多样、高质量…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "双曲空间"
  - "扩散自编码器"
  - "层级表示"
  - "小样本生成"
  - "Poincaré圆盘"
---

# HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation

**会议**: ICCV 2025  
**arXiv**: [2411.17784](https://arxiv.org/abs/2411.17784)  
**代码**: [https://github.com/lingxiao-li/HypDAE](https://github.com/lingxiao-li/HypDAE)  
**领域**: 扩散模型/小样本图像生成  
**关键词**: 双曲空间, 扩散自编码器, 层级表示, 小样本生成, Poincaré圆盘

## 一句话总结

将双曲空间的层级表示学习能力与扩散自编码器的高质量生成能力结合，通过在 Poincaré 圆盘中操控潜码的半径和方向，实现可控、多样且保持类别一致性的小样本图像生成。

## 研究背景与动机

小样本图像生成（Few-shot Image Generation）旨在仅用少量样本为未见类别生成多样、高质量的图像。核心挑战是**类别一致性与图像多样性之间的权衡**。

**现有方法的三大瓶颈**：

**GAN 方法生成质量受限**：基于 GAN 的迁移/融合/变换方法在训练数据不足时难以生成真实图像。

**多样性不足**：一对一映射（潜码→图像）在潜码训练不充分时丢失高频细节，导致生成结果同质化。

**依赖标注数据**：学习层级潜表示需要类别标签，在实际场景中难以获取。

**为什么用双曲空间？** 图像具有语义层级结构：高层的身份相关属性（如性别、种族）决定类别核心，低层的身份无关属性（如表情、发型）引入类内变化。双曲空间（负曲率）以指数增长的半径天然适合编码树状层级——圆盘边缘对应细粒度特征，圆盘中心对应抽象/共享特征。

**为什么结合扩散模型？** 扩散模型相比 GAN 在少数据场景下生成质量更高，且预训练基础模型（SD、CLIP）提供强先验，支持有限数据适配。

## 方法详解

### 整体框架

HypDAE 包含两个阶段：

**Stage I — 扩散自编码器**：
- **语义编码器**：CLIP 图像编码器提取高层语义码 $\boldsymbol{c}$（仅用 class token，512→1024维），通过 MLP 对齐到 SD 的文本特征空间。
- **随机编码器**：预训练 SD 模型通过 DDIM 反转将图像编码为随机子码 $\boldsymbol{z}_T$，捕获语义码未覆盖的低层细节。
- 两个编码器协同实现高保真重建：$(\boldsymbol{c}, \boldsymbol{z}_T) \to x'$。
- **防止复制的技巧**：(1) 强数据增强（翻转、旋转、模糊、弹性变换）；(2) 内容瓶颈（仅用 class token 压缩信息）。

**Stage II — 双曲编码器-解码器**：
- 5 层单头 Transformer 编码器将欧式潜码 $\boldsymbol{c}$ 降维到 512 维。
- 指数映射 $\exp_\mathbf{0}^c$ 将欧式向量投影到 Poincaré 圆盘 $\mathbb{D}^n$。
- Möbius 线性层生成双曲表示 $\boldsymbol{c}_h = f^{\otimes_c}(\exp_\mathbf{0}^c(\text{E}(\boldsymbol{c})))$。
- 30 层单头 Transformer 解码器通过对数映射 $\log_\mathbf{0}^c$ 重建 $\boldsymbol{c}' = \text{D}(\log_\mathbf{0}^c(\boldsymbol{c}_h))$。

### 关键设计

1. **双曲层级表示**：

    - 在 Poincaré 圆盘中，距离公式 $d_\mathbb{D}(\mathbf{x}, \mathbf{y}) = \text{arccosh}(1 + \frac{2\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)})$ 使得边缘点之间的距离指数增长。
    - 分类损失（扩展的 MLR）推动细粒度图像嵌入到圆盘边缘（最大化类间距离），共享特征嵌入到中心。
    - 半径 $r_\mathbb{D}$ 直接对应属性层级：$r_\mathbb{D} > 5.0$ 对应低层身份无关属性变化，$r_\mathbb{D} < 2.0$ 导致身份属性改变。

2. **双曲潜码编辑（生成多样图像）**：

    - **随机子码变化**：冻结语义码 $\boldsymbol{c}$，用不同随机种子获得不同 $\boldsymbol{z}_T$，改变纹理/背景等低层特征。
    - **语义码扰动**：在固定半径 $r_\mathbb{D}$ 上沿测地线方向随机扰动 $\boldsymbol{c}_h$，修改类内变化特征。
    - **层级插值**：沿测地线在两个嵌入之间插值，实现平滑的属性渐变。

3. **伪标签训练**：利用预训练 CLIP 的零样本分类为图像生成伪标签，无需人工标注。

### 损失函数 / 训练策略

**Stage I**（仅训练 MLP）：
$$\mathcal{L}_{align} = \mathbb{E}_{\boldsymbol{z}_0, t, \boldsymbol{c}, \epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon_\theta(\boldsymbol{z}_t, t, \boldsymbol{c})\|_2^2]$$

**Stage II**：
$$\mathcal{L} = \mathcal{L}_{hyper} + \lambda \cdot \mathcal{L}_{rec}$$

- $\mathcal{L}_{hyper} = -\frac{1}{N}\sum_{n=1}^N \log(p_n)$：双曲 MLR 分类损失，推动层级结构形成。
- $\mathcal{L}_{rec}(\boldsymbol{c}, \boldsymbol{c}') = \|\boldsymbol{c} - \boldsymbol{c}'\|_2 + 1 - \cos(\boldsymbol{c}, \boldsymbol{c}')$：L2 + 余弦相似度重建损失。

## 实验关键数据

### 主实验（1-shot 生成）

| 方法 | 设置 | Flowers FID↓ | Flowers LPIPS↑ | AnimalFaces FID↓ | AnimalFaces LPIPS↑ | VGGFaces FID↓ | NABirds FID↓ |
|------|------|-------------|---------------|-----------------|-------------------|--------------|-------------|
| DeltaGAN | 1-shot | 109.78 | 0.391 | 89.81 | 0.442 | 80.12 | 96.79 |
| SAGE | 1-shot | 43.52 | 0.439 | 27.43 | 0.545 | 34.97 | 19.45 |
| HAE | 1-shot | 50.10 | 0.474 | 26.33 | 0.564 | 35.93 | 21.85 |
| **HypDAE (Real)** | 1-shot | **23.96** | **0.760** | **14.31** | **0.742** | **6.25** | **7.64** |
| **HypDAE (Pseudo)** | 1-shot | 24.43 | **0.763** | **13.14** | **0.743** | **5.96** | **7.57** |

HypDAE 在所有数据集上以大幅度优势领先：AnimalFaces FID 从 HAE 的 26.33 降至 13.14（提升 50%），LPIPS 从 0.564 提升至 0.743。

### 消融实验

**双曲半径 $r_\mathbb{D}$ 影响**：

| 半径 $r_\mathbb{D}$ | 6.2(边缘) | 5.5 | 4.5 | 3.0(中心) |
|---------------------|----------|-----|-----|----------|
| FID↓ | 15.18 | **14.31** | 14.71 | 20.65 |
| LPIPS↑ | 0.704 | 0.742 | 0.794 | **0.896** |
| CLIP-S(身份保持) | 77.37 | 75.15 | 71.45 | 67.89 |
| CLIP-P(扰动相似) | 69.62 | 72.35 | 74.25 | 77.00 |

$r_\mathbb{D} = 5.5$ 时达到最佳 FID-LPIPS 权衡；更小半径增加多样性但牺牲身份保持。

**欧式 vs 双曲空间**：

| 方法 | AnimalFaces FID↓ | AnimalFaces LPIPS↑ |
|------|-----------------|-------------------|
| HypDAE (Euclidean) | 20.72 | 0.729 |
| **HypDAE (Hyperbolic)** | **14.31** | **0.742** |

双曲空间表示使 FID 降低 30.9%。

### 关键发现

- **伪标签优于真实标签**：HypDAE(Pseudo) 在多数基准上略优于 HypDAE(Real)，说明人工标注中的噪声反而会干扰层级表示学习。伪标签精度虽然只有 39-79%，但足以学习有用的层级结构。
- 随机编码强度控制了生成图像与参考图像的相似-多样权衡：强度越高，随机子码 $\boldsymbol{z}_T$ 越接近纯噪声，生成越多样。
- 定性对比（Fig. 9）中，HypDAE 能生成精细的羽毛纹理等细节，HAE/WaveGAN 难以做到。
- 用户研究（Table 4）中 HypDAE 在质量（3.45/4）、保真度（3.58/4）、多样性（3.86/4）三项都一骑绝尘。

## 亮点与洞察

- **双曲空间 + 扩散模型**的首次联合：利用双曲空间的层级结构赋予扩散模型可解释且可控的语义编辑能力。
- **$r_\mathbb{D}$ 作为多样性旋钮**的设计极其直观：半径小 → 更抽象 → 更多样；半径大 → 更具体 → 更保真。
- **两阶段设计**巧妙避开了联合端到端训练的困难，也免去了大规模标注数据的需求。
- 预训练 SD + CLIP 的组合使方法在极少数据下也能生成高质量图像。

## 局限与展望

- Stage II 的 Transformer 解码器（30 层）规模偏大，可探索更轻量的映射网络。
- VGGFaces 数据集分辨率低（64×64）导致 FID 异常，切换到 FFHQ 虽然解决了但引入了领域偏移。
- 当前仅支持 1-shot 和 3-shot 设置，更多参考图像时的融合策略有待设计。
- 未探索在视频生成或 3D 生成中的扩展。

## 相关工作与启发

- 将 DiffAE 的双子码（语义码 + 随机码）架构从欧式空间迁移到双曲空间，证明了非欧几何在生成模型建模中的价值。
- HAE 首次探索了双曲空间用于小样本生成但限于 GAN；HypDAE 通过扩散模型克服了 GAN 的质量瓶颈。
- 伪标签策略的成功说明：对于层级表示学习，"大致正确"的标签比"精确但有噪声"的人工标签更有效。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个双曲+扩散的小样本生成方法，理念新颖
- **技术深度**: ⭐⭐⭐⭐ — 双曲几何基础扎实，两阶段设计合理
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 数据集、消融丰富、用户研究、可视化充分
- **实用价值**: ⭐⭐⭐⭐ — 无需人工标签，单图即可生成多样图像

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](../../AAAI2026/image_generation/hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)
- [\[ICLR 2026\] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](../../ICLR2026/image_generation/hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)

</div>

<!-- RELATED:END -->
