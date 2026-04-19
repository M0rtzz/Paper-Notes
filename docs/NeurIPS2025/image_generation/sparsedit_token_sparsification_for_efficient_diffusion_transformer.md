---
title: >-
  [论文解读] SparseDiT: Token Sparsification for Efficient Diffusion Transformer
description: >-
  [NeurIPS 2025][图像生成][Transformer] 提出 SparseDiT，通过空间维度的三段式架构（底层 Poolingformer + 中层 Sparse-Dense Token Module + 顶层全密度处理）和时间维度的动态剪枝率策略，在 DiT-XL 512×512 上实现 55% FLOPs 减少和 175% 推理速度提升，FID 仅增加 0.09，并成功扩展到视频生成和文本到图像生成任务。
tags:
  - NeurIPS 2025
  - 图像生成
  - Transformer
  - token稀疏化
  - 推理加速
  - 时间步自适应
  - 架构设计
---

# SparseDiT: Token Sparsification for Efficient Diffusion Transformer

**会议**: NeurIPS 2025  
**arXiv**: [2412.06028](https://arxiv.org/abs/2412.06028)  
**代码**: 无  
**领域**: 扩散模型 / 模型效率  
**关键词**: 扩散 Transformer, token稀疏化, 推理加速, 时间步自适应, 架构设计

## 一句话总结

提出 SparseDiT，通过空间维度的三段式架构（底层 Poolingformer + 中层 Sparse-Dense Token Module + 顶层全密度处理）和时间维度的动态剪枝率策略，在 DiT-XL 512×512 上实现 55% FLOPs 减少和 175% 推理速度提升，FID 仅增加 0.09，并成功扩展到视频生成和文本到图像生成任务。

## 研究背景与动机

Diffusion Transformer (DiT) 凭借 Transformer 的可扩展性在图像和视频生成中展现出强大能力，是 Sora 等先进系统的骨干架构。然而 DiT 面临严重的计算效率问题：自注意力机制的二次复杂度随 token 数量急剧增长，加上去噪过程需要大量采样步骤。

现有加速方法主要集中在降低采样步数（如 ODE 求解器、一致性模型、知识蒸馏），但忽略了 DiT **架构本身**的效率问题。与 U-Net 的收缩-扩展结构自然降低计算量不同，DiT 在所有层维持全尺寸 token 的自注意力。直接将分类任务的 token 减少技术（如 ToMeSD）迁移到 DiT 上效果很差（DiT-XL 上 0.1 合并率导致 FID 飙升到 14.74）。

作者对 DiT 注意力图的深入分析揭示了三个关键洞察：

**底层注意力近似均匀分布**：底层的自注意力近乎全局平均池化，复杂的注意力计算贡献有限

**中间层交替关注全局与局部**：部分层聚焦局部细节，部分层聚焦全局结构，这一规律跨所有采样步一致

**去噪越深越需要局部信息**：随着去噪推进，注意力方差增大，模型越来越关注局部细节

基于以上洞察，SparseDiT 在空间和时间两个维度上动态调节 token 密度。

## 方法详解

### 整体框架

SparseDiT 将 DiT 的 Transformer 层分为三个段：底部 (Bottom)、中间 (Middle)、顶部 (Top)。底部使用 Poolingformer 高效捕获全局特征；中间层使用多个 Sparse-Dense Token Module (SDTM) 交替处理稀疏和密集 token；顶层使用原始密集 Transformer 精炼高频细节。主要计算节省来自中间段的稀疏 token 处理。

### 关键设计

1. **Poolingformer（底层）**：基于注意力图近乎均匀分布的观察，将底层的自注意力替换为全局平均池化。具体做法是移除 Q 和 K，仅对 V 做全局均值池化后加到输入 token 上：X = X + V̄。实验验证了这一简化的合理性——将前两层注意力图替换为全 1 矩阵，生成结果几乎不变。但底层不能使用稀疏 token（会导致训练不稳定），必须保持完整 token。

2. **Sparse-Dense Token Module (SDTM，中层）**：核心模块，将全局结构提取和局部细节提取解耦。流程如下：

    - **稀疏 token 生成**：通过空间自适应池化将密集 token X ∈ R^{N×C} 初始化为稀疏 token X_s ∈ R^{M×C}（M ≪ N），保持空间均匀分布。然后通过注意力层将稀疏 token 与全尺寸 token 交互，整合全局信息。
    - **稀疏 Transformer 处理**：后续多个 Transformer 层仅对稀疏 token 运算，大幅降低计算量。
    - **密集 token 恢复**：将稀疏 token 上采样后通过两个线性层与原始密集 token 融合（X_merged = UpSample(X_s)·W₁ + X·W₂），再通过注意力层进一步整合。
    - **密集 Transformer 处理**：少量密集 Transformer 层增强局部细节。
    - 多个 SDTM 级联（默认 4 个），交替切换稀疏/密集表示，有效保留结构和细节信息。

3. **时间步动态剪枝率策略**：去噪早期主要生成低频全局结构，后期生成高频细节。据此设计动态剪枝率 r：前 T/4 步保持最高剪枝率 r_min，之后线性递减到 r_max，token 数逐步增加。训练时通过分片函数巧妙解决了批量训练与随机采样的矛盾。

### 损失函数 / 训练策略

SparseDiT 通过微调预训练 DiT 模型实现，微调时间仅为从头训练的约 6%（例如 DiT-XL 约 400K 迭代）。初始化策略：Poolingformer 不加载 Q/K 参数；SDTM 中的融合权重 W₁ 初始化为全零、W₂ 初始化为单位矩阵，保证初始时密集 token 路径为恒等映射。各 Transformer 加载对应预训练权重。在稀疏/密集 token 切换处重新引入正弦余弦位置编码。

## 实验关键数据

### 主实验

| 模型/分辨率 | 方法 | FLOPs (G) | 吞吐率 (img/s) | FID↓ | IS↑ |
|-------------|------|-----------|---------------|------|-----|
| DiT-XL/256² | 原始 | 118.64 | 1.58 | 2.27 | 278.24 |
| DiT-XL/256² | SparseDiT (r∈[0.44,0.61]) | 88.91 (-25%) | 2.13 (+35%) | 2.23 | 278.91 |
| DiT-XL/256² | SparseDiT (r∈[0.61,0.86]) | 68.05 (-43%) | 2.95 (+87%) | 2.38 | 276.39 |
| DiT-XL/512² | 原始 | 525 | 0.249 | 3.04 | 240.82 |
| DiT-XL/512² | SparseDiT (r∈[0.61,0.86]) | 286 (-46%) | 0.609 (+145%) | 2.96 | 242.4 |
| DiT-XL/512² | SparseDiT (r∈[0.90,0.96]) | 235 (-55%) | 0.685 (+175%) | 3.13 | 236.56 |
| PixArt-α | 原始 | 148.73 | 0.414 | 4.53 | - |
| PixArt-α | SparseDiT | 91.62 (-38%) | 0.701 (+69%) | 4.29 | - |

在 512×512 分辨率下，剪枝 90% 以上 token 仍能获得仅 0.09 FID 增加，且速度提升 175%。

### 消融实验

| 配置 | FID↓ | 说明 |
|------|------|------|
| 1 个 SDTM | NAN | 退化为 U-Net 结构，训练崩溃 |
| 2 个 SDTM | 3.86 | 全局/局部信息交互不足 |
| 3 个 SDTM | 2.51 | 改善明显 |
| 4 个 SDTM（默认） | 2.38 | 最佳 |
| 0 个 Poolingformer | NAN | 训练不稳定 |
| 2 个 Poolingformer（默认） | 2.38 | 稳定 |
| 3 个 Poolingformer | 2.56 | 过多池化损伤信息 |
| 固定 token 数 8×8 | 2.48 | 静态分配 |
| 动态 6×6~10×10 | 2.38 | 动态剪枝更优 |

### 关键发现

- DiT 存在严重的 token 冗余：在特定层仅使用约 25% 的 token 即可维持性能
- 从 256² 到 512²，FLOPs 仅增 4.4 倍但速度下降 6.3 倍，说明 DiT 的瓶颈在于 token 数而非模型参数
- SDTM 的交替稀疏-密集设计是成功的关键——退化为 U-Net 结构（仅 1 个 SDTM）会训练崩溃
- 方法与高效采样器（DDIM、Rectified Flow）正交可叠加——结合 5 步 RFlow 可实现 93.4× 加速
- 在视频生成（Latte-XL）上实现 56% FLOPs 减少，验证了跨模态泛化性

## 亮点与洞察

- **注意力图分析驱动的架构设计**：不是盲目压缩，而是基于对每一层注意力行为的精细分析进行定制化设计
- **Poolingformer 的"玩具实验"非常有说服力**：将底层注意力替换为全 1 矩阵后生成结果几乎不变，直接证明了底层复杂注意力计算的冗余
- **时空协同的稀疏策略**：空间上按层分配 token 密度，时间上按去噪阶段动态调整，两者协同
- **与 token 合并方法（ToMeSD）的本质区别**：SparseDiT 不是在每层通用地减少 token，而是按 DiT 的全局/局部交替模式专门设计稀疏化策略

## 局限与展望

- 架构是手动预定义的（每个模块的层数和稀疏 token 数量需要人工设定），缺乏自动搜索
- 消融实验在 256×256 上进行，高分辨率和更大模型上的最优配置可能不同
- 未探索与注意力蒸馏等更先进加速技术的结合
- SDTM 数量增加到 4 以上时收益递减，但更精细的 SDTM 内部配置（稀疏/密集 Transformer 的比例）值得进一步探索

## 相关工作与启发

- ToMeSD 首次在扩散模型中尝试 token 减少，但在 DiT 上效果差——证明需要 DiT 专用策略
- DyDiT 从多维度（token/层/头/通道）压缩 DiT，但 FLOPs 减少仅 29%——SparseDiT 的 46%~55% 远超之
- U-Net 的收缩-扩展结构本身就是一种稀疏网络——启发了 SDTM 的交替稀疏-密集设计
- EDT 尝试类 U-Net 架构但存在较大 FID 差距——说明直接套用 U-Net 结构不适合 DiT

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](../../CVPR2025/image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](../../ICCV2025/image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_eff.md)
- [Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [Rare Text Semantics Were Always There in Your Diffusion Transformer](rare_text_semantics_were_always_there_in_your_diffusion_transformer.md)
- [Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)

<!-- RELATED:END -->
