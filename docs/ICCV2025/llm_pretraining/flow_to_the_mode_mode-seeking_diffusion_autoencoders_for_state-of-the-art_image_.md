---
title: >-
  [论文解读] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization
description: >-
   提出 FlowMo，一种基于 Transformer 的扩散自编码器 (diffusion autoencoder)，通过两阶段训练（mode-matching 预训练 + mode-seeking 后训练），首次实现扩散自编码器在 ImageNet-1K 离散图像 tokenization 上的 SOTA 性能，无需使用卷积、对抗损失、2D 空间对齐 latent 或从其他 tokenizer 蒸馏。
tags:

---

# FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.11056](https://arxiv.org/abs/2503.11056)
- **代码**: [kylesargent.github.io/flowmo](https://kylesargent.github.io/flowmo)
- **领域**: Image Generation / 图像分词器
- **关键词**: 扩散自编码器, 图像tokenization, rectified flow, mode-seeking, 离散分词器
- **作者**: Kyle Sargent, Kyle Hsu, Justin Johnson, Li Fei-Fei, Jiajun Wu (Stanford, UMich)

## 一句话总结

提出 FlowMo，一种基于 Transformer 的扩散自编码器 (diffusion autoencoder)，通过两阶段训练（mode-matching 预训练 + mode-seeking 后训练），首次实现扩散自编码器在 ImageNet-1K 离散图像 tokenization 上的 SOTA 性能，无需使用卷积、对抗损失、2D 空间对齐 latent 或从其他 tokenizer 蒸馏。

## 研究背景与动机

当前主流视觉生成系统是两阶段的：先用 tokenizer 压缩像素数据到离散 latent 空间，再在该空间训练生成模型。自 VQGAN 以来，SOTA tokenizer 通常是 CNN 自编码器 + 对抗损失 + 感知损失 + 2D 空间对齐 latent code 的组合。

扩散自编码器 (diffusion autoencoder) 是另一种思路：用扩散模型作为解码器端到端学习感知导向的图像压缩。但此前扩散自编码器在 ImageNet-1K 重建这个竞争激烈的基准上从未达到 SOTA。

FlowMo 的核心 insight：对于感知重建任务，**采样重建分布中感知上接近原始图像的模式 (mode-seeking)**，比试图匹配所有模式 (mode-matching) 更好。这一直觉启发了两阶段训练方案。

## 方法详解

### 整体架构

FlowMo 是一个扩散自编码器：
- **编码器** $e_\theta$：将图像 $x$ 编码为量化 latent $c$（1D 序列，非 2D 空间对齐）
- **解码器** $d_\theta$：条件扩散模型，学习 $p(x|c)$ 的条件分布
- 架构基于 MMDiT（Stable Diffusion 3 的骨干），全 Transformer，无卷积
- 编码器和解码器结构对称但不同大小（解码器更大更深）
- 使用 $\mu P$ 参数化便于超参数迁移

### 量化方式

采用 Lookup-Free Quantization (LFQ)：
$$c = q(\hat{c}) = 2 \cdot \mathbb{1}[\hat{c} \geq 0] - 1$$

元素级二值化，避免了传统 codebook 查找的复杂性。

### Stage 1A: Mode-Matching 预训练

目标：端到端训练编码器和解码器，使 $p_\theta(x|c)$ 匹配真实分布。

Rectified flow 损失：
$$\mathcal{L}_{\text{flow}} = \mathbb{E}\left[ \| x - z - d_\theta(x_t, q(e_\theta(x)), t) \|_2^2 \right]$$

其中 $x_t = tz + (1-t)x$。

辅助损失包括：
- **感知损失** $\mathcal{L}_{\text{perc}}$：LPIPS-VGG 网络监督 1-step 去噪预测
- **熵损失** $\mathcal{L}_{\text{ent}}$：LFQ 的 codebook 利用率
- **承诺损失** $\mathcal{L}_{\text{commit}}$：$\| \hat{c} - q(\hat{c}) \|_2^2$

总损失：$\mathcal{L}_{\text{flow}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{commit}} \mathcal{L}_{\text{commit}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$

使用 thick-tailed logit-normal 噪声调度（10% 时间从均匀分布采样），避免 $t=1$ 处零概率导致的颜色偏移。

### Stage 1B: Mode-Seeking 后训练（关键创新）

冻结编码器，通过反向传播穿过整个采样链（8步 ODE 积分）来优化解码器：

$$\mathcal{L}_{\text{sample}} = \mathbb{E}\left[ d_{\text{perc}}(x, d_{t_n} \circ d_{t_{n-1}} \circ \cdots \circ d_{t_1}(z)) \right]$$

总损失：$\mathcal{L}_{\text{flow}} + \lambda_{\text{sample}} \mathcal{L}_{\text{sample}}$

关键细节：
- $\lambda_{\text{sample}} = 0.01$，过大导致 reward hacking 或训练发散
- Stage 1A 用 LPIPS-VGG，Stage 1B 用 ResNet 作为感知网络
- 使用梯度检查点和梯度累积降低计算开销
- 训练约 1 个 epoch，使用早停

### 采样器设计

使用 shifted sampler，时间步间隔：
$$t_i = \left(\frac{n-i+1}{n}\right)^\rho, \quad \rho = 4$$

$\rho > 1$ 使采样集中在低噪声区域，偏向采样 $p(x|c)$ 的均值附近，同时保留低噪声级别的采样 FLOP。

## 实验关键数据

### 主实验：ImageNet-1K 重建（Table 1）

| 模型 | BPP | rFID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----|--------|--------|--------|---------|
| OpenMagViT-V2 | 0.070 | 1.17 | 21.63 | 0.640 | 0.111 |
| **FlowMo-Lo** | **0.070** | **0.95** | **22.07** | **0.649** | 0.113 |
| LlamaGen-32 | 0.219 | 0.59 | 24.44 | 0.768 | 0.064 |
| **FlowMo-Hi** | **0.219** | **0.56** | **24.93** | **0.785** | 0.073 |

在两个 BPP 下均达到 rFID、PSNR、SSIM 的 SOTA。唯一劣势在 LPIPS 指标。

### 消融实验

**Stage 1A 消融（Table 4）**：

| 变体 | rFID ↓ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| FlowMo (fewer params) | 2.87 | 20.71 | 0.15 |
| patch size 翻倍 | 6.39 | 19.94 | 0.17 |
| MSE 训练编码器 | 3.82 | 21.40 | 0.15 |
| 无感知损失 | 13.86 | 22.11 | 0.21 |
| FSQ 量化 | 3.14 | 21.31 | 0.14 |
| logit-normal 噪声 | 4.08 | 16.45 | 0.21 |
| 无 shifted sampler | 3.42 | 20.25 | 0.16 |
| 无 guidance | 3.28 | 20.67 | 0.16 |

**Stage 1B 消融（Table 5 — 最关键）**：

| 模型 | rFID ↓ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| FlowMo-Lo (无后训练) | 1.10 | 21.38 | 0.134 |
| **FlowMo-Lo (后训练)** | **0.95** | **22.07** | **0.113** |
| FlowMo-Hi (无后训练) | 0.73 | 24.02 | 0.086 |
| **FlowMo-Hi (后训练)** | **0.56** | **24.93** | **0.073** |

后训练阶段带来全指标提升，rFID 降低 14-23%。

### 生成质量（Table 2）

| Tokenizer | FID ↓ | IS ↑ | sFID ↓ | Prec. ↑ | Rec. ↑ |
|-----------|-------|------|--------|---------|--------|
| OpenMagViT-V2 | 3.73 | 241 | 10.66 | 0.80 | 0.51 |
| FlowMo-Lo | 4.30 | 274 | 10.31 | 0.86 | 0.47 |

更好的 tokenizer 不一定直接带来更好的生成质量，tokenizer 质量与生成质量之间存在复杂交互。

### 关键发现

1. **Mode-seeking 后训练是 SOTA 的关键**：简单用 ResNet 感知损失的 1-step 去噪无法替代穿过完整采样链的反传
2. **端到端训练至关重要**：先用 MSE 训练编码器再接扩散解码器的方式 rFID 显著变差
3. **thick-tailed 噪声调度必不可少**：标准 logit-normal 会导致颜色偏移
4. **后训练后重建仍保持多模态**：方差集中在背景等感知不敏感区域

## 亮点与洞察

- **"做得更好而非更全"的哲学**：不试图匹配所有重建模式，而是选择性保留感知上最好的模式
- **纯 Transformer + 1D latent**：打破了 CNN + 2D latent 的统治地位
- **无对抗损失**：扩散模型天然多模态建模能力 + mode-seeking 后训练完美替代了 GAN 判别器
- **与 RLHF 后训练的联系**：Stage 1B 本质上类似于扩散模型的后训练/对齐范式

## 局限性

- **推理速度慢**：需要多步 ODE 积分（$n=25$ 步），远慢于 CNN decoder 的单次前向
- **LPIPS 指标不占优**：在 LPIPS 上略逊于传统方法
- **生成质量未超越传统 tokenizer**：FID 略差于 OpenMagViT-V2
- 后训练阶段计算成本高（需要反传穿过完整采样链 + 梯度检查点）

## 相关工作与启发

- **扩散自编码器进入 SOTA 行列**：证明了 diffusion autoencoder 路线的可行性
- **与对齐技术的联系**：mode-seeking 后训练与 DDPO/AlignProp 等扩散模型对齐方法有技术联系
- **对 tokenizer 设计的启示**：1D latent + Transformer 可能在大规模数据时更有优势

## 评分 ⭐⭐⭐⭐⭐

开创性工作。首次让扩散自编码器在竞争最激烈的 ImageNet tokenization 基准上达到 SOTA。mode-matching + mode-seeking 的两阶段训练思路简洁深刻，消融实验极其详尽。对 tokenizer 设计范式有重要推动作用。

<!-- RELATED:START -->

## 相关论文

- [Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)
- [Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [Tokenization is Sensitive to Language Variation](../../ACL2025/llm_pretraining/tokenization_is_sensitive_to_language_variation.md)
- [Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](../../NeurIPS2025/llm_pretraining/learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)

<!-- RELATED:END -->
