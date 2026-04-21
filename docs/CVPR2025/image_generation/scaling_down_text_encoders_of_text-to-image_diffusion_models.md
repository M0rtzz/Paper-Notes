---
title: >-
  [论文解读] Scaling Down Text Encoders of Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][图像生成][文本编码器蒸馏] 本文通过基于视觉的知识蒸馏方法，将 T5-XXL（11B）文本编码器蒸馏为 T5-Base（220M），缩小 50 倍的同时在图像质量和语义理解上几乎不损失，揭示了文本编码器在文生图任务中存在严重过参数化的"缩放下行规律"。
tags:
  - CVPR 2025
  - 图像生成
  - 文本编码器蒸馏
  - T5压缩
  - FLUX
  - 知识蒸馏
  - 模型效率
---

# Scaling Down Text Encoders of Text-to-Image Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2503.19897](https://arxiv.org/abs/2503.19897)  
**代码**: https://github.com/LifuWang-66/DistillT5  
**领域**: 扩散模型  
**关键词**: 文本编码器蒸馏, T5压缩, FLUX, 知识蒸馏, 模型效率

## 一句话总结

本文通过基于视觉的知识蒸馏方法，将 T5-XXL（11B）文本编码器蒸馏为 T5-Base（220M），缩小 50 倍的同时在图像质量和语义理解上几乎不损失，揭示了文本编码器在文生图任务中存在严重过参数化的"缩放下行规律"。

## 研究背景与动机

**领域现状**：文生图扩散模型的文本编码器从早期的 CLIP 迅速演进到 T5-XXL（11B 参数），Imagen、FLUX、SD3 等 SOTA 模型都采用 T5-XXL 来增强复杂语义理解和文字渲染能力。

**现有痛点**：T5-XXL 的 11B 参数带来巨大的 GPU 内存开销——FLUX pipeline 本身已超 24GB，加上 T5-XXL 几乎无法在消费级 GPU 上运行。尽管 8-bit 量化可以一定程度解决，但参数总量仍然庞大。

**核心矛盾**：T5 模型是在 C4 自然语言语料上训练的，包含大量非视觉数据。实验表明扩散模型对非视觉 prompt 生成的图像与文本对齐度很低，说明 T5-XXL 的表示能力中大部分对文生图是冗余的。

**本文目标**：回答"文生图真的需要这么大的文本编码器吗？"，探索文本编码器的缩放下行规律。

**切入角度**：T5-XXL 的嵌入空间包含大量非视觉冗余信息，可以用更小的模型只学习对图像生成有用的视觉子空间。

**核心 idea**：用扩散模型本身的图像合成能力作为蒸馏指导（vision-based distillation），让小模型学习在视觉层面与大模型产生相同的去噪预测。

## 方法详解

### 整体框架

固定预训练的 FLUX 扩散模型不变，只训练小型 T5 编码器。给定 prompt，分别通过教师（T5-XXL）和学生（T5-Base）生成文本嵌入，送入 FLUX 得到两个去噪预测，最小化它们之间的差异。采用 step-following 策略在每个去噪步骤都进行蒸馏。用 MLP 将学生嵌入投影到教师嵌入空间。

### 关键设计

1. **Vision-based Knowledge Distillation（基于视觉的知识蒸馏）**:

    - 功能：将 T5-XXL 的文生图能力迁移到小型 T5 模型
    - 核心思路：不直接蒸馏 T5 的文本嵌入（naive distillation），而是通过冻结的扩散模型将嵌入差异放大到像素/latent 空间。损失函数为 $\mathcal{L}_{vision} = \mathbb{E}_p[\|\mu_\theta(\mathbf{x}_t, t, \omega_\phi(p)) - \mu_\theta(\mathbf{x}_t, t, \omega_{\hat{\phi}}(p))\|^2]$，其中 $\mu_\theta$ 是扩散模型的去噪预测
    - 设计动机：naive distillation 会导致学生嵌入空间的 mode collapse，因为巨大的参数差距使得小模型无法建立一一映射。视觉蒸馏通过噪声和更细粒度的 latent 空间特征引入了不确定性，允许学生用不同的分布达到相同的视觉效果

2. **Step-Following Training（逐步跟随训练）**:

    - 功能：确保学生模型在每个去噪时间步都学到正确的指导
    - 核心思路：从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, I)$ 开始，在每个时间步 $t$ 将同一 latent 送入扩散模型，分别用教师和学生嵌入得到两个预测，计算损失并反向传播更新学生编码器。然后用教师预测的 latent 推进到下一步，重复直到 $\mathbf{x}_0$
    - 设计动机：SOTA 扩散模型的训练数据是私有的，无法获得图像-文本对来用 Eq.2 直接构造 $\mathbf{x}_t$。但 prompt 容易获取，因此直接从噪声出发模拟完整的采样轨迹

3. **三阶段数据集构建**:

    - 功能：覆盖 T5-XXL 在文生图中的完整视觉嵌入空间
    - 核心思路：第一阶段用 LAION-Aesthetics-6.5+（~100K prompt）覆盖图像质量和风格；第二阶段用 T2I-CompBench（4200 prompt）覆盖语义理解（颜色、形状、纹理、空间关系）；第三阶段用自建的 CommonText（50K prompt）覆盖文字渲染能力
    - 设计动机：T5-XXL 的优势体现在三个维度，需要针对性地构建训练数据确保全面继承

### 损失函数 / 训练策略

三阶段训练：第一阶段 T2I-CompBench 50K iter，第二阶段 CommonText 70K iter，第三阶段混合全部数据 200K iter。8×A800 GPU，总 batch size 32。AdamW 优化器，学习率 1e-4，20 步迭代去噪。guidance scale 在 2~5 之间随机采样。

## 实验关键数据

### 主实验

| 模型 | 参数量 | FID↓ | CLIP-Score↑ | 语义理解 Avg↑ | 文字渲染 Char↑ |
|------|--------|------|-------------|--------------|--------------|
| Flux w/ T5-Small | 60M | 25.10 | 28.28 | - | 31.9 |
| Flux w/ T5-Base | 220M | 24.32 | 29.79 | 50.32 | 69.3 |
| Flux w/ T5-XL | 3B | 23.17 | 30.33 | 53.74 | 77.8 |
| Flux w/ T5-XXL | 11B | 22.36 | 31.30 | 55.56 | 76.7 |
| SD3 | - | 19.83 | 32.21 | - | 38.7 |

### 消融实验

| 训练数据组合 | FID↓ | CLIP↑ | 语义↑ | 文字↑ |
|-------------|------|-------|-------|-------|
| LAION only | 24.13 | 29.69 | 31.09 | 2.97 |
| CompBench only | 23.55 | 27.88 | 44.93 | 1.32 |
| CommonText only | 28.95 | 25.62 | 21.20 | 43.41 |
| Naive distill (全部) | 26.47 | 22.52 | 13.78 | 0.35 |
| **Vision distill（全部）** | **24.32** | **29.79** | **50.32** | **49.1** |

### 关键发现

- **图像质量和语义理解对编码器大小不敏感**：T5-Base（50x 小）在图像质量和语义理解上接近 T5-XXL
- **文字渲染是最受模型大小影响的维度**：T5-Small 文字渲染严重退化，T5-Base 勉强可用，T5-XL 接近 T5-XXL
- **Naive distillation 完全失败**：mode collapse 导致所有指标远低于 vision distillation
- **t-SNE 可视化**证实 T5-Base 学到的分布与 T5-XXL 完全不同，但仍能有效指导扩散模型——说明文生图不需要精确复制嵌入分布
- 使用 T5-Base 后 FLUX pipeline 可以在 24GB GPU 上运行，不需要 CPU offload 时速度提升 2.7x

## 亮点与洞察

- **"文本编码器过参数化"**的发现具有实际影响——T5-XXL 的 11B 参数中绝大部分对文生图是冗余的，因为 T5 trained on C4 中的非视觉知识在文生图中不被使用。这个洞察可能推动整个领域重新审视模型组件的资源分配
- **Vision-based distillation**的思路很巧妙——"让扩散模型告诉你什么嵌入是好的"，利用了预训练模型已有的强大合成能力。这种思路可以迁移到其他模态的编码器蒸馏
- **与 ControlNet、LoRA、蒸馏模型的兼容性**验证了蒸馏编码器作为 drop-in replacement 的实用性

## 局限与展望

- 文字渲染能力在小模型上仍有明显损失，需要更多文字相关数据或专门的训练策略
- 仅在 FLUX 上验证，是否适用于其他架构（如 UNet-based SD 系列）有待确认
- 蒸馏过程仍需要 8×A800 GPU 训练约 320K iterations，训练成本不低
- 未来可以探索更极端的压缩（如小于 60M 参数），或结合量化进一步降低推理成本

## 相关工作与启发

- **vs Imagen 的 scaling 研究**: Imagen 发现 T5-XXL 优于小版本 T5，但那是在指导扩散模型训练时的结论；本文反向证明了固定扩散模型后可以将编码器大幅缩小
- **vs 8-bit 量化**: 量化仅压缩精度，参数量不变；蒸馏直接减少参数量 50 倍
- **vs Progressive Distillation**: 那些工作蒸馏扩散模型本身的步数；本文蒸馏的是文本编码器，两者正交可以组合

## 评分

- 新颖性: ⭐⭐⭐⭐ vision-based distillation 思路新颖，scaling down pattern 分析有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 从三个维度全面评估，消融充分，兼容性测试完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 直接降低 FLUX 等模型的部署门槛，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models](six-cd_benchmarking_concept_removals_for_text-to-image_diffusion_models.md)
- [Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)

<!-- RELATED:END -->
