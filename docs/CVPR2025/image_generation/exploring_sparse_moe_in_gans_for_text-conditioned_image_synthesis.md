---
title: >-
  [论文解读] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis
description: >-
  [CVPR 2025][图像生成][文本生成图像] 本文提出 Aurora，一种基于稀疏混合专家（Sparse MoE）的 GAN 文本生成图像模型，通过在生成器中引入多个专家网络和文本感知的稀疏路由器来扩大模型容量，在 64×64 分辨率上以远快于扩散模型的推理速度，在 MS COCO 上达到 6.2 的零样本 FID。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本生成图像"
  - "GAN"
  - "稀疏混合专家"
  - "大规模训练"
  - "快速推理"
---

# Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis

**会议**: CVPR 2025  
**arXiv**: [2309.03904](https://arxiv.org/abs/2309.03904)  
**代码**: [https://github.com/zhujiapeng/Aurora](https://github.com/zhujiapeng/Aurora)  
**领域**: 扩散模型  
**关键词**: 文本生成图像, GAN, 稀疏混合专家, 大规模训练, 快速推理

## 一句话总结
本文提出 Aurora，一种基于稀疏混合专家（Sparse MoE）的 GAN 文本生成图像模型，通过在生成器中引入多个专家网络和文本感知的稀疏路由器来扩大模型容量，在 64×64 分辨率上以远快于扩散模型的推理速度，在 MS COCO 上达到 6.2 的零样本 FID。

## 研究背景与动机

**领域现状**：文本生成图像（T2I）目前被扩散模型主导（如 Stable Diffusion、DALL-E 等），而 GAN 虽然享有极快的推理速度、可解释的潜空间以及灵活的架构扩展性（如 3D 感知生成），但在开放词汇 T2I 任务上已逐渐被边缘化。

**现有痛点**：GAN 的核心问题在于难以扩展模型规模。扩散模型通过迭代去噪过程可以自然地使用大模型，而 GAN 的生成器是前馈网络，直接增大网络深度或宽度会面临 GPU 显存瓶颈和训练不稳定的问题。此外，社区缺乏开源的大规模 GAN T2I 模型，这严重阻碍了 GAN 在 T2I 方向的研究发展。

**核心矛盾**：GAN 需要更大的模型容量来处理开放词汇的多样化内容，但直接扩大前馈网络的参数量在计算资源上不可行。

**本文目标**：找到一种在有限计算资源下有效扩大 GAN 生成器容量的方法，使其能够胜任大规模开放词汇 T2I 生成。

**切入角度**：稀疏激活的混合专家（Sparse MoE）已在 NLP 领域证明可以用有限计算资源训练超大规模模型——每次前向传播只激活少量专家，模型容量随专家数增长而计算量几乎不增加。

**核心 idea**：将 Sparse MoE 引入 GAN 生成器的 FFN 层，用多个专家处理不同特征点，并设计一个融合文本条件和采样随机性的自适应稀疏路由器来分配专家。

## 方法详解

### 整体框架
Aurora 的生成器以 GAN 框架为基础，输入为全局潜码 $\mathbf{z} \in \mathbb{R}^{512}$ 和文本描述 $\mathbf{c}$，输出合成图像。生成器由多个分辨率递增的生成单元堆叠而成（从 4×4 开始渐进训练），每个单元包含卷积块和注意力块，后者采用 Sparse MoE 机制来扩大容量。判别器直接沿用 GigaGAN 的架构。

### 关键设计

1. **文本条件采样（Text-conditioned Sampling）**:

    - 功能：将文本信息注入潜空间，生成文本感知的全局潜码
    - 核心思路：使用 CLIP ViT-L/14 作为文本编码器，提取 token 序列 $\mathbf{t}_{seq}$ 和全局 token $\mathbf{t}_g$。将 $\mathbf{t}_g$ 与采样潜码 $\mathbf{z}$ 拼接后送入 MLP 映射网络，得到解纠缠潜码 $\mathbf{w} = \text{MLP}(\text{concat}(\mathbf{z}, \mathbf{t}_g))$。CLIP 编码器冻结后堆叠可学习层适配 T2I 任务
    - 设计动机：沿用 StyleGAN 的映射网络设计使潜空间更解纠缠，同时将文本全局语义融入采样过程

2. **稀疏路由的注意力块（Sparse MoE Attention Block）**:

    - 功能：在不显著增加计算量的前提下扩大生成器容量
    - 核心思路：每个生成单元的注意力块包含自注意力层、交叉注意力层（融合 $\mathbf{t}_{seq}$）和 FFN。关键创新在于将 FFN 替换为 $N$ 个专家 $\{\text{FFN}_j\}_{j=1}^N$，并用稀疏路由器为每个特征点选择最合适的专家：$j = \text{Router}(\mathbf{f}_{ca}^{(k)}, \mathbf{w})$。路由器同时考虑输入特征和文本融合的全局潜码 $\mathbf{w}$，使路由决策能感知文本语义和采样随机性。可视化显示相似视觉概念的像素倾向于被分配到同一专家
    - 设计动机：与现有 Sparse MoE 仅基于输入特征路由不同，T2I 任务需要路由器理解"生成什么内容"，因此必须融入文本条件

3. **卷积块与特征调制**:

    - 功能：在每个生成单元中通过调制卷积处理特征
    - 核心思路：使用两个调制变换模块（MTM）配合跳跃连接：$\mathbf{f}_{conv} = \text{MTM}(\text{MTM}(\mathbf{f}_{in}, \mathbf{w}), \mathbf{w}) + \mathbf{f}_{in}$。低分辨率（≤16×16）时引入可学习偏移的变形操作，所有卷积采用样本自适应核选择
    - 设计动机：继承 StyleGAN 系列中证明有效的调制卷积设计，通过 $\mathbf{w}$ 注入全局文本语义

### 损失函数 / 训练策略
采用四种损失函数：(1) 对抗损失（logistic non-saturating + R1 惩罚）；(2) 匹配感知损失（判别器拒绝文本-图像不匹配对）；(3) 多级 CLIP 损失（在所有分辨率鼓励文本-图像对齐）；(4) MoE 平衡损失（避免部分专家从不被激活）。训练策略采用渐进式训练（从 4×4 逐步到 64×64），提出"参考 FID"作为自动指标决定何时切换到下一分辨率阶段——预先计算两组真实图像之间的 FID 作为理论下限，生成器 FID 优于此值时进入下一阶段。使用 256 块 A100 训练一周。

## 实验关键数据

### 主实验

| 方法 | 类型 | FID 10K (训练集) | Zero-Shot FID 30K (COCO) | 参数量 |
|------|------|-----------------|-------------------------|--------|
| eDiff-I | 扩散 | - | 7.60 | 9.1B |
| BSD (Stable Diffusion) | 扩散 | - | 8.40 | 0.94B |
| StyleGAN-T | GAN | - | 7.30 | 1.02B |
| GigaGAN | GAN | 9.18 | - | 0.65B |
| **Aurora (本文)** | GAN | **8.28** | **6.45** | 1.16B |

所有评测在 64×64 分辨率，Aurora 在两个域均达到最优。

### 消融实验

| 分析维度 | 关键发现 |
|---------|---------|
| 路由可视化 | 相似视觉概念的像素被分配到同一专家，在所有分辨率成立 |
| 文本插值 | 两个文本 prompt 之间插值呈现平滑语义过渡，保持语义连续性 |
| 潜码插值 | 潜空间（$\mathcal{Z}$ 或 $\mathcal{W}$）插值结果不如预期平滑，更像随机采样 |
| 训练稳定性 | 即便同时使用对抗训练和 Sparse MoE，整个训练过程无不稳定现象 |

### 关键发现
- GAN 在引入文本条件后训练稳定性显著改善，并未出现传统 GAN 的崩溃问题
- 文本 prompt 插值比潜码插值更具语义连续性，表明在 T2I GAN 中文本 token 序列对生成的主导作用可能超过全局潜码
- 路由可视化证实了稀疏路由器能按视觉概念自动聚类，验证了文本感知路由设计的合理性

## 亮点与洞察
- 文本感知路由器是本文最巧妙的设计——路由决策融入 $\mathbf{w}$（含文本+随机性），使不同文本内容和不同采样能获得不同的专家分配模式，这比 NLP 中仅基于输入 token 路由更适合生成任务
- 渐进训练 + 参考 FID 自动调度是实用的工程贡献，减少了人工调参的需要
- 开源大规模 GAN T2I 模型本身就有重要价值——为 GAN 的潜空间编辑、3D 感知生成等独特优势提供了可用的基础模型

## 局限与展望
- 当前仅支持 64×64 分辨率，需要额外超分模型（4× 上采样），端到端高分辨率生成尚未实现
- 潜空间插值的不连续性是未解之谜，可能与交叉注意力中文本 token 主导有关
- 与同期扩散模型相比，生成质量仍有差距（视觉多样性和精细度）
- 未来可探索：更大规模训练、直接高分辨率输出、将 MoE 引入判别器

## 相关工作与启发
- **vs StyleGAN-T**: StyleGAN-T 也做 GAN T2I，但未解决模型容量扩展问题，Aurora 通过 Sparse MoE 在参数量更大的情况下实现了更好的 FID
- **vs GigaGAN**: GigaGAN 是最接近的工作（都做大规模 GAN T2I），Aurora 在零样本 FID 上超越，且提供了完整开源
- **vs Switch Transformer**: 借鉴了 Switch Transformer 的稀疏路由思想，但创新地加入了文本条件感知，非直接套用

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Sparse MoE 引入 GAN 生成器并设计文本感知路由是有新意的组合创新
- 实验充分度: ⭐⭐⭐ 定量实验仅在 64×64 分辨率，缺少消融实验表格（如专家数量、路由策略的消融）
- 写作质量: ⭐⭐⭐⭐ 技术报告风格，描述清晰但略偏工程，讨论部分对社区价值有独到见解
- 价值: ⭐⭐⭐⭐ 开源大规模 GAN T2I 模型对社区有重要价值，但 64×64 限制了实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[CVPR 2025\] Science-T2I: Addressing Scientific Illusions in Image Synthesis](science-t2i_addressing_scientific_illusions_in_image_synthesis.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects](self-cross_diffusion_guidance_for_text-to-image_synthesis_of_similar_subjects.md)

</div>

<!-- RELATED:END -->
