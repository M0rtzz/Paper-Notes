---
title: >-
  [论文解读] Your Text Encoder Can Be An Object-Level Watermarking Controller
description: >-
  [ICCV 2025][图像生成][水印嵌入] 通过仅微调文本编码器中的伪 token 嵌入 $\mathcal{W}_*$，实现对 T2I 扩散模型生成图像的对象级不可见水印嵌入，以 $10^5\times$ 更少的参数达到 99% 的比特准确率（48 bits）。
tags:
  - ICCV 2025
  - 图像生成
  - 水印嵌入
  - 文本编码器
  - 对象级水印
  - Textual Inversion
  - 扩散模型
---

# Your Text Encoder Can Be An Object-Level Watermarking Controller

**会议**: ICCV 2025  
**arXiv**: [2503.11945](https://arxiv.org/abs/2503.11945)  
**代码**: [GitHub](https://github.com/)  
**领域**: 扩散模型/图像水印  
**关键词**: 水印嵌入, 文本编码器, 对象级水印, Textual Inversion, 扩散模型

## 一句话总结

通过仅微调文本编码器中的伪 token 嵌入 $\mathcal{W}_*$，实现对 T2I 扩散模型生成图像的对象级不可见水印嵌入，以 $10^5\times$ 更少的参数达到 99% 的比特准确率（48 bits）。

## 研究背景与动机

随着文本到图像（T2I）扩散模型的广泛应用，版权保护和内容溯源变得越来越重要。现有的生成式水印方法存在以下核心问题：

**参数量庞大**：现有方法如 Stable Signature、AquaLoRA 等需要修改 UNet 或 VAE 解码器等大模块，训练参数量高达 $10^5$ 以上，且无法方便地迁移到不同的 LDM 管线。

**缺乏空间控制**：几乎所有方法都只能对整幅图像进行水印，无法选择性地对图像中某个对象（如仅对"猫"进行水印），而在很多场景下（如保护图片中独特的物体），对象级水印是至关重要的。

**后处理水印的脆弱性**：后生成阶段添加的水印容易被图像处理攻击（如裁剪、旋转、JPEG 压缩）破坏，鲁棒性较差。

**需要额外信息**：少数支持局部水印的方法（如 WAM）需要分割掩码来定位水印区域，增加了额外的计算开销。

本文的核心观察是：文本编码器在 LDM 管线中相对未被充分利用，而 Textual Inversion 的成功证明了学习新 token 嵌入可以注入新概念。如果可以学习一个"水印 token"，就能通过交叉注意力机制天然地实现对象级水印——用户只需在 prompt 中将水印 token 与目标对象相邻放置即可。

## 方法详解

### 整体框架

本方法的核心思路是引入一个伪 token $\mathcal{W}_*$ 到文本编码器的词汇表中，通过微调其嵌入向量来实现水印功能。训练采用 Img2Img 管线：将输入图像编码到潜空间 $z_0$，添加 $\tau^*$ 步前向噪声得到 $z_{\tau^*}$，然后在 $\mathcal{W}_*$ 条件下去噪回 $z'_{0,w}$，通过 VAE 解码器得到水印图像 $I_{w,m}$，再输入预训练的水印检测器 $D_w$ 来训练嵌入。

推理时，用户只需将 $\mathcal{W}_*$ 插入 prompt 中：
- 全图水印：`[A photo of a cat $\mathcal{W}_*$]`
- 对象水印：`A photo of a [cat $\mathcal{W}_*$]`

### 关键设计

1. **Token 嵌入学习（核心创新）**：不同于修改 UNet 或 VAE 等大模块，本方法仅学习一个 768 维的 token 嵌入向量，参数量相比基线方法减少 $10^5$ 倍。该 token 可以"即插即用"地加载到任何兼容文本编码器的 LDM 管线中（如 Stable Diffusion v1.5、个性化模型、风格模型等），无需修改核心模型参数。这种类似 Textual Inversion 的 prêt-à-porter 训练方式极大降低了部署门槛。

2. **最优时间步与潜空间匹配损失**：不同噪声时间步 $\tau$ 在图像质量和水印性能之间存在权衡——$\tau \sim T$ 时图像质量下降但水印强，$\tau \sim 0$ 时质量好但水印弱。作者通过实验确定最优时间步 $\tau^* = 8$，并引入潜空间匹配损失来保证水印的不可见性：

   $$\mathcal{L}_z = \min_{\mathcal{W}_*} \mathbb{E}_t \left[ \| z^*_t - z'_t(\mathcal{W}_*) \|_2^2 \right]$$

   总训练损失为 $\mathcal{L} = \alpha \mathcal{L}_w + \beta \mathcal{L}_z$，其中 $\mathcal{L}_w$ 是 BCE 比特嵌入损失。选择靠近 VAE 编码器端的小时间步 $\tau^*$ 不仅提升了质量，也因为水印嵌入发生在管线早期，使其更难被后续阶段的对抗扰动破坏。

3. **对象级水印控制机制**：利用 UNet 中的交叉注意力图来定位水印区域。在 T2I 生成过程中，每个时间步 $t$ 提取目标对象 token $\mathcal{P}_i$ 和水印 token $\mathcal{W}_*^i$ 的注意力图 $\mathcal{M}_{\mathcal{P}_i}^{(t)}$ 和 $\mathcal{M}_{\mathcal{W}_*}^{(t)}$，通过叠加实现水印定位：

   $$\mathcal{M}_{\mathcal{P}_i}^{(t)} \leftarrow (1-\alpha) \cdot \mathcal{M}_{\mathcal{P}_i}^{(t)} + \alpha \cdot \mathcal{M}_{\mathcal{W}_*}^{(t)}$$

   此外引入水印叠加强度控制器 $\pi(t)$（阶跃函数或平滑函数），使水印效果集中在训练时观察到的最优时间步范围内，提升水印定位精度。

### 损失函数 / 训练策略

- **比特嵌入损失** $\mathcal{L}_w = BCE(D_w(Dec(z'_{0,w})), m)$：确保水印密钥 $m \in \{0,1\}^{48}$ 被正确嵌入
- **潜空间匹配损失** $\mathcal{L}_z$：约束水印化潜空间轨迹与原始轨迹接近，保证不可见性
- 水印检测器 $D_w$ 来自预训练模型（如 AquaLoRA），整个 T2I 管线和检测器均保持冻结
- 训练数据：MS-COCO 子集（2000 张图像）

## 实验关键数据

### 主实验

**WikiArt 数据集（48 bits）全图水印对比**：

| 方法 | 参数量 | PSNR↑ | FID↓ | 无攻击 BA↑ | 亮度 BA↑ | 模糊 BA↑ | JPEG BA↑ | SDEdit BA↑ | WMAttacker BA↑ |
|------|--------|-------|------|-----------|---------|---------|---------|-----------|--------------|
| Stable Sig. | $10^5$+ | 31.57 | 24.71 | 0.99 | 0.93 | 0.78 | 0.55 | 0.58 | 0.53 |
| AquaLoRA | $10^5$+ | 31.46 | 17.27 | 0.94 | 0.91 | 0.81 | 0.76 | 0.68 | 0.67 |
| WAM | $10^5$+ | 36.46 | 16.27 | 0.97 | 0.93 | 0.84 | 0.84 | 0.72 | 0.71 |
| **Ours + SD** | **768** | **39.92** | **14.89** | **0.99** | **0.98** | **0.97** | **0.95** | **0.85** | **0.87** |

### 消融实验

**对象级水印鲁棒性（不同裁剪和多对象）**：

| 配置 | 无攻击 BA | 亮度 BA | 模糊 BA | 旋转 BA | JPEG BA |
|------|----------|--------|---------|--------|---------|
| 单对象-分割+白底 | 0.99 | 0.97 | 0.97 | 0.96 | 0.97 |
| 单对象-裁剪0.5× | 0.92 | 0.91 | 0.91 | 0.92 | 0.90 |
| 2对象(无重叠) | 0.94 | 0.93 | 0.95 | 0.94 | 0.96 |
| 3对象(无重叠) | 0.90 | 0.89 | 0.90 | 0.90 | 0.99 |
| 2对象(重叠≥40%) | 0.79 | 0.76 | 0.80 | 0.74 | 0.74 |

### 关键发现

- 本方法在 PSNR 上比 AquaLoRA 高出 7dB+，在对抗攻击鲁棒性上提升 20%+
- 仅 768 个参数即可实现 99% 比特准确率，参数效率提升 $10^5$ 倍
- 对象区域缩小到原始的 40% 仍可保持 89%+ 的检测精度
- 多对象水印在无重叠情况下保持 90%+ 准确率，但重叠超过 40% 会导致性能下降
- 该 token 可直接迁移到个性化模型和不同风格的 SD 变体，保持高鲁棒性

## 亮点与洞察

- **极致的参数效率**：将水印问题归约为 Textual Inversion 问题是一个优雅的思路，768 参数即可完成任务
- **对象级水印首创**：首次在 T2I 生成管线中实现无需分割掩码的对象级水印，利用交叉注意力图自然定位
- **即插即用设计**：水印 token 可与任何兼容文本编码器的 LDM 组合使用，包括个性化模型，实际部署非常便捷
- **早期嵌入提升鲁棒性**：在文本编码阶段（管线早期）嵌入水印，比在后处理阶段更难被攻击移除

## 局限与展望

- 对象级水印依赖交叉注意力图的准确性，当注意力图不精确时水印可能溢出到其他区域
- 多对象高度重叠（>40%）时检测准确率明显下降
- 当前仅在 Stable Diffusion v1.5 上验证，是否适用于 SDXL、Flux 等新架构需进一步探索
- 水印 token 数量固定（1个），未来可探索多 token 策略以嵌入更多比特

## 相关工作与启发

- Textual Inversion 的成功启发了本文将水印能力"编码"到学习的 token 嵌入中
- Prompt-to-Prompt 的注意力操控技术被用于实现对象级水印定位
- 本文的思路可推广到其他文本编码器驱动的生成模型中（如视频生成模型）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将水印嵌入归约为 token 学习问题，首次实现无掩码对象级生成式水印
- **实验充分度**: ⭐⭐⭐⭐ 全图和对象级均有完整评估，但缺少 SDXL 等新模型的验证
- **写作质量**: ⭐⭐⭐⭐ 方法清晰，实验详尽
- **价值**: ⭐⭐⭐⭐⭐ 极高的实际应用价值，参数效率和即插即用设计使其易于部署

<!-- RELATED:START -->

## 相关论文

- [Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](../../ECCV2024/image_generation/enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)
- [GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)
- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](../../ECCV2024/image_generation/lcm-lookahead_for_encoder-based_text-to-image_personalization.md)

<!-- RELATED:END -->
