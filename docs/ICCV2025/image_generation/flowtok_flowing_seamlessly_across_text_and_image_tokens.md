---
title: >-
  [论文解读] FlowTok: Flowing Seamlessly Across Text and Image Tokens
description: >-
  [图像生成] FlowTok 提出将文本和图像都编码为紧凑的 1D token 表示（77×16），通过 flow matching 直接在文本与图像 token 之间进行流动转换，无需复杂的条件机制或噪声调度，实现了高效的跨模态生成。
tags:
  - 图像生成
---

# FlowTok: Flowing Seamlessly Across Text and Image Tokens

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.10772](https://arxiv.org/abs/2503.10772)
- **代码**: [GitHub](https://github.com/TACJu/FlowTok)
- **领域**: 图像生成 / 跨模态生成
- **关键词**: Flow Matching, 1D Token, 文本到图像生成, 紧凑表示, 跨模态

## 一句话总结

FlowTok 提出将文本和图像都编码为紧凑的 1D token 表示（77×16），通过 flow matching 直接在文本与图像 token 之间进行流动转换，无需复杂的条件机制或噪声调度，实现了高效的跨模态生成。

## 研究背景与动机

传统的文本到图像生成方法将文本作为条件信号，通过去噪过程从高斯噪声逐步引导到目标图像。这需要复杂的条件机制（如交叉注意力、拼接等）和噪声调度策略。

FlowTok 探索了一种更简单的范式：**直接在文本和图像模态之间通过 flow matching 进行演化**。这要求将两种模态投射到共享的潜在空间中，而文本（1D 序列、高维语义）和图像（2D 空间、冗余信息）之间的表示差异构成了核心挑战。

先前的 CrossFlow 将文本映射到 2D 潜在空间以匹配图像嵌入，但由于文本变分自编码器的额外计算开销，反而比 SD1.5/2.1 更慢，违背了效率初衷。

## 方法详解

### 整体框架

FlowTok 的核心思想是将文本和图像都编码为形状为 $77 \times 16$ 的紧凑 1D token：
- **图像侧**：使用改进的 TA-TiTok 将图像编码为 $\mathbf{Z}_I \in \mathbb{R}^{K \times D}$（K=77, D=16）
- **文本侧**：使用 CLIP 文本编码器提取初始嵌入，再通过文本投影器映射到低维变分潜在空间 $\mathbf{Z}_T \in \mathbb{R}^{N \times D}$
- **生成模型**：使用 DiT 块进行 vanilla flow matching，文本 token 直接作为源分布

相比传统 2D flow matching 的 $32 \times 32 \times 4$ 潜在空间，FlowTok 实现了 **3.3× 的压缩率**。

### 关键设计

**1. 图像 Tokenizer 改进**
- 基于 TA-TiTok 架构，将潜在 token 数 K 设为 77（匹配 CLIP 文本长度）
- 引入 RoPE 替代可学习 1D 位置编码，提升位置处理能力
- 使用 SwiGLU FFN 替换标准 MLP，改善潜在空间质量
- 编码器用 ViT-B，解码器用 ViT-L，patch size = 16

**2. 文本投影器**
- 6 个 Transformer 块，带 skip connections
- 将 CLIP 文本嵌入（$77 \times 768$）投影到低维空间（$77 \times 16$）
- 对投影后的文本 token 施加 KL 散度正则化，引入生成多样性

**3. 文本对齐损失**

为防止通道降维导致的语义信息丢失，引入 CLIP 风格的对比损失：

$$\mathcal{L}_{\text{align}} = \frac{1}{2}(\text{CE}(\text{logits}_{TZ}, \text{labels}) + \text{CE}(\text{logits}_{ZT}, \text{labels}))$$

其中 logits 通过可学习温度参数 $\tau$ 计算缩放的余弦相似度。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{fm}} + \gamma_1 \cdot \mathcal{L}_{\text{kld}} + \gamma_2 \cdot \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{fm}}$：flow matching 速度预测损失
- $\mathcal{L}_{\text{kld}}$：KL 散度正则化（$\gamma_1 = 10^{-4}$）
- $\mathcal{L}_{\text{align}}$：文本对齐对比损失（$\gamma_2 = 1$）

### 模型规模

| 模型 | 深度 | 宽度 | MLP | 头数 | 参数量 |
|------|------|------|-----|------|--------|
| FlowTok-B | 12 | 768 | 3072 | 12 | 153M |
| FlowTok-XL | 28 | 1152 | 4608 | 16 | 698M |
| FlowTok-H | 36 | 1280 | 5120 | 20 | 1.1B |

## 实验关键数据

### 主实验：文本到图像生成（零样本）

| 方法 | 参数 | 开源数据 | 训练成本(8-A100天) | 推理速度(样本/秒) | COCO FID-30K↓ | MJHQ-30K FID↓ |
|------|------|----------|------|------|------|------|
| PixArt-α | 630M | ✗ | 94.1 | 7.9 | 7.32 | 9.85 |
| SD-2.1 | 860M | ✓ | 1041.6 | - | 13.45 | 26.96 |
| Show-o | 1.3B | ✓ | - | 1.0 | 9.24 | 14.99 |
| CrossFlow | 950M | ✗ | 78.8 | 1.1 | 9.63 | - |
| **FlowTok-XL** | **698M** | **✓** | **20.4** | **22.7** | **10.06** | **7.68** |
| **FlowTok-H** | **1.1B** | **✓** | **26.1** | **18.2** | **9.67** | **7.15** |

### 消融实验：文本对齐损失

| 对齐目标 | COCO FID-30K↓ |
|----------|---------------|
| Average Pooling | 36.02 |
| **MLP** | **29.14** |

| 损失类型 | COCO FID-30K↓ |
|----------|---------------|
| Cosine | 31.80 |
| **Contrastive** | **29.14** |

| $\gamma_2$ | COCO FID-30K↓ |
|------------|---------------|
| **1.0** | **29.14** |
| 2.0 | 30.59 |

### 关键发现

1. **极致效率**：FlowTok-H 仅需 26.1 个 8-A100 天完成训练，是 SD-2.1 的 1/40，PixArt-α 的 1/3.6
2. **推理极快**：FlowTok-XL 每秒生成 22.7 张图像，比 CrossFlow 快 20×，比 Show-o 快 22×
3. **内存高效**：最大模型支持在 8 张 A100 上使用 batch size 8K，无需梯度检查点或梯度累积
4. **双向生成**：同一框架自然支持图像到文本生成，FlowTok-XL 在 COCO Karpathy 上 CIDEr 达 117.0

## 亮点与洞察

1. **范式创新**：将文本从"条件信号"转变为"源分布"，flow matching 直接在模态间流动，消除了复杂的条件机制
2. **1D token 统一**：通过将图像也编码为 1D token，巧妙地统一了文本和图像的表示形式
3. **仅用 20 步采样**：得益于紧凑的 1D 潜在空间，采样步数远少于传统 2D 方法
4. **完全开源数据**：所有训练仅使用公开数据集，确保可复现性

## 局限性

1. 图像分辨率限于 256，未验证更高分辨率
2. 依赖 CLIP 文本编码器（77 token 限制），对长文本描述的处理能力有限
3. 1D 表示可能在空间精细控制上不如 2D 方法
4. 图像到文本生成的性能虽有竞争力，但非最优

## 相关工作与启发

- **CrossFlow**：同样探索跨模态 flow matching，但使用 2D 潜在空间，计算开销大
- **TA-TiTok**：提供了 1D 图像 tokenization 的基础架构
- **DiT**：FlowTok 的生成模型基于 DiT 块，但去掉了交叉注意力等条件机制

## 评分

⭐⭐⭐⭐ — 范式创新性强，效率提升显著，但分辨率和文本长度的限制降低了实际应用价值。

<!-- RELATED:START -->

## 相关论文

- [EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)

<!-- RELATED:END -->
