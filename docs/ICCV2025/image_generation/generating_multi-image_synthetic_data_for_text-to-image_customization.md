---
title: >-
  [论文解读] Generating Multi-Image Synthetic Data for Text-to-Image Customization
description: >-
  [ICCV 2025][图像生成][文本到图像定制化] 提出 SynCD（合成定制数据集）及其生成管线，利用共享注意力和 3D 资产先验合成多图一致性对象数据集，训练的编码器模型在无需测试时优化的情况下超越现有编码器方法。
tags:
  - ICCV 2025
  - 图像生成
  - 文本到图像定制化
  - 合成数据集
  - 共享注意力
  - 编码器方法
  - 3D一致性
---

# Generating Multi-Image Synthetic Data for Text-to-Image Customization

**会议**: ICCV 2025  
**arXiv**: [2502.01720](https://arxiv.org/abs/2502.01720)  
**代码**: [项目页面](https://nupurkmr9.github.io/SynCD/)  
**领域**: 扩散模型/图像定制化  
**关键词**: 文本到图像定制化, 合成数据集, 共享注意力, 编码器方法, 3D一致性

## 一句话总结

提出 SynCD（合成定制数据集）及其生成管线，利用共享注意力和 3D 资产先验合成多图一致性对象数据集，训练的编码器模型在无需测试时优化的情况下超越现有编码器方法。

## 研究背景与动机

文本到图像模型的定制化（Customization）旨在学习用户提供的参考对象，并通过文本提示生成该对象在新场景中的图像。然而这一任务面临核心数据瓶颈：

**真实多图数据集稀缺**：训练编码器方法需要同一对象在不同姿态、背景、光照下的多张图像，但从互联网收集这样的大规模数据集几乎不可能——真实图像通常没有对象身份标注。

**单图训练的局限**：现有编码器方法（如 IP-Adapter、BLIP-Diffusion）大多在单图或多视角（固定背景）数据集上训练。为防止过拟合到参考图的姿态或背景，它们被迫使用紧凑的特征空间，这牺牲了身份保持能力。

**优化方法的高成本**：DreamBooth、Textual Inversion 等优化方法虽然效果好，但每个新对象都需要数分钟的微调，计算成本高且速度慢。

**现有合成数据质量不足**：JeDi 等工作也尝试创建合成数据集，但仅依赖文本提示来生成一致对象，缺乏显式的一致性约束，导致数据质量有限。

本文的核心洞察是：利用 T2I 模型内部特征的共享和 3D 资产的外部约束来合成一致对象身份，远比收集真实多图数据更可扩展，也比模型定制化任务本身更容易。

## 方法详解

### 整体框架

方法包含三个部分：
1. **SynCD 数据集生成管线**：LLM 辅助提示生成 → 共享注意力 + 3D 先验的多图生成 → 质量过滤
2. **编码器模型训练**：利用共享注意力机制将参考图像特征注入去噪过程
3. **改进的推理算法**：归一化文本和图像引导向量，避免过曝问题

### 关键设计

1. **掩码共享注意力（MSA）用于一致对象生成**：在并行生成 $N$ 张图像时，修改 DiT 模型的注意力块，使每张图像的特征不仅关注自身，还关注其他图像中前景对象区域的特征。注意力操作为：

   $$\text{MSA}(\{{\mathbf{q}}_i, {\mathbf{k}}_i, {\mathbf{v}}_i\}_{i=1}^N) = \left\{ \text{Softmax}\left(\frac{{\mathbf{q}}_i [{\mathbf{k}}_1 \cdots {\mathbf{k}}_N]^T}{\sqrt{d'}} + \mathbf{M}_i\right) [{\mathbf{v}}_1 \cdots {\mathbf{v}}_N] \right\}_{i=1}^N$$

   掩码 $\mathbf{M}_i$ 确保每张图像只关注其他图像的对象前景区域并忽略背景，且文本 token 不跨图像交互。对于刚性物体，进一步利用 Objaverse 3D 资产的深度图指导和跨视图对应特征 warping：

   $$\hat{f}_2(u,v) = \alpha f_1(u+\Delta u, v+\Delta v) + (1-\alpha) f_2(u,v)$$

   其中 $\alpha$ 是可见性二值标量。Warping 仅在早期扩散步骤中应用以避免伪影。

2. **LLM 辅助提示与数据过滤**：使用 Llama3 为每个对象生成详细描述和多个背景场景描述。对于 Objaverse 刚性物体使用 Cap3D 提供的描述；对于可变形对象（动物等 16 个超类别）由 LLM 生成描述性标题。生成后过滤：美学评分 < 6 的剔除，DINOv2 平均两两相似度 < 0.7 的剔除。最终数据集包含约 95,000 个对象，每个 2-3 张图像。

3. **改进的推理引导归一化**：直接组合文本和图像 classifier-free guidance 在高图像引导下容易过曝。提出归一化方案：

   $$\epsilon_\theta({\mathbf{x}}^t, \{{\mathbf{x}}_i\}, \varnothing) + \lambda_I \frac{\|g\|}{\|g_I\|} \cdot g_I + \lambda_{\mathbf{c}} \frac{\|g\|}{\|g_c\|} \cdot g_{\mathbf{c}}$$

   其中 $\|g\| = \min(\|g_I\|, \|g_{\mathbf{c}}\|)$。将两个引导向量的范数缩放到最小范数，仅通过 $\lambda_I$ 和 $\lambda_{\mathbf{c}}$ 控制相对强度，有效避免过曝同时保持高图像对齐度。

### 损失函数 / 训练策略

- 使用速度/流预测目标：$\mathbb{E}_{{\mathbf{x}}^t,t,\mathbf{c},\epsilon} \| \mathbf{v} - \mathbf{v}_\theta({\mathbf{x}}^t, t, \mathbf{c}, \{{\mathbf{x}}_i\}) \|$
- 对 FLUX（12B）仅 LoRA 微调注意力层
- 对 U-Net 模型（1B/3B）从 IP-Adapter 初始化，微调 self-attention 的 LoRA 层和 image cross-attention 的 KV 投影
- 参考图像通过共享注意力机制（与数据集生成相同）注入

## 实验关键数据

### 主实验

**DreamBooth 基准定量对比（单参考图输入）**：

| 方法 | MDINOv2-I↑ | CLIPScore↑ | TIFA↑ | GeometricScore↑ |
|------|-----------|-----------|-------|----------------|
| BLIP-Diffusion | 0.658 | 0.294 | 0.782 | 0.714 |
| IP-Adapter Plus | 0.744 | 0.270 | 0.615 | 0.675 |
| Emu-2 | 0.750 | 0.283 | 0.741 | 0.740 |
| JeDi | 0.771 | 0.292 | 0.789 | 0.780 |
| **Ours (1B)** | 0.806 | 0.303 | 0.830 | **0.801** |
| **Ours (3B)** | **0.822** | **0.313** | **0.863** | **0.838** |
| OminiControl (12B) | 0.650 | 0.302 | 0.808 | 0.685 |
| **Ours (12B)** | 0.778 | 0.306 | 0.786 | 0.780 |

**人类偏好评估（3 参考图输入）**：

| 对比 | 文本对齐↑ | 图像对齐↑ | 真实感↑ | 总体偏好↑ |
|------|---------|---------|--------|---------|
| Ours(1B) vs JeDi | 69.5% | 63.1% | 80.9% | 68.2% |
| Ours(3B) vs Emu-2 | 70.5% | 66.9% | 64.7% | 66.7% |
| Ours(12B) vs OminiControl | 56.3% | 58.3% | 54.5% | 58.0% |

### 消融实验

**方法组件消融（基于 IP-Adapter Plus 3B）**：

| 方法 | MDINOv2-I↑ (bg/prop) | TIFA↑ | GeometricScore↑ |
|------|---------------------|-------|----------------|
| IPAdapter Plus baseline | 0.744/0.737 | 0.615 | 0.675 |
| + 改进推理 | 0.719/0.668 | 0.816 | 0.756 |
| + SynCD 数据集 | 0.766/0.695 | 0.901 | 0.819 |
| + MSA 共享注意力(1-input) | 0.777/0.708 | 0.902 | 0.825 |
| + MSA (3-input) | **0.822/0.789** | **0.863** | **0.838** |

**数据集规模消融**：

| 数据量 | Ours (3B) MDINOv2-I↑ | Ours (12B) MDINOv2-I↑ |
|-------|---------------------|----------------------|
| 100 | 0.790 | 0.736 |
| 1K | 0.805 | 0.762 |
| 10K | 0.810 | 0.763 |
| 95K | 0.813 | 0.774 |

### 关键发现

- SynCD 数据集的贡献大于模型架构改进：仅用 SynCD 微调 IP-Adapter Plus 就带来了 GeometricScore 从 0.675 到 0.819 的巨大提升
- 改进的推理归一化将 TIFA 从 0.615 大幅提升到 0.816，几乎无图像对齐损失
- MSA + warping 在刚性物体上提升 DINOv2-I 一致性 0.026，warping 对颜色一致性尤为关键
- 数据集规模对大模型（12B）更重要，对小模型（3B）边际收益递减
- 更多参考图像（3张 vs 1张）显著提升图像对齐度

## 亮点与洞察

- **数据驱动 > 模型设计**：本文最强有力的发现是高质量多图数据集比花哨的模型架构更重要。仅用 SynCD 数据训练简单的共享注意力编码器就能超越复杂的专用方法。
- **合成数据生成的可扩展策略**：MSA + 3D depth guidance + warping 的组合策略为合成一致训练数据提供了可复制的范式。
- **推理引导归一化的通用性**：该技巧不仅适用于本方法，也可直接改善 IP-Adapter 等现有方法的文本对齐度。
- **从数据集生成到模型设计的统一**：数据集生成和模型训练使用相同的共享注意力机制，设计上高度一致。

## 局限与展望

- 当前仅关注单对象图像，未扩展到多对象场景
- Objaverse 资产的多样性有限，可能影响某些类别的质量
- 可变形对象未使用 3D 先验（缺乏对应的 3D 数据集），仅依赖 MSA + 详细描述
- 生成数据与真实数据之间仍存在域差距
- 可探索将视频生成模型和 text-to-3D 模型引入数据集构建

## 相关工作与启发

- Textual Inversion 和 DreamBooth 开创了模型定制化方向
- FLUX 模型的 MMDiT 架构为共享注意力的高效实现提供了基础
- 本文的 MSA 机制受到一致性角色生成和多视图生成工作的启发
- 引导归一化的思路可推广到其他多条件生成任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据集生成管线设计精巧，MSA+3D先验组合有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 自动指标+人类评估+详尽消融(组件/数据量/推理)
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑清晰，图表精美，实验设计科学
- **价值**: ⭐⭐⭐⭐⭐ 数据集和方法均有高实用价值，推理引导归一化技巧可广泛应用

<!-- RELATED:START -->

## 相关论文

- [Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [Fast Data Attribution for Text-to-Image Models](../../NeurIPS2025/image_generation/fast_data_attribution_for_text-to-image_models.md)
- [MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)

<!-- RELATED:END -->
