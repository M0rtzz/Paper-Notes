---
title: >-
  [论文解读] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation
description: >-
  [CVPR 2025][图像生成][视频帧插值] 提出 EDEN，从输入表示、模型架构和训练范式三个维度全面增强扩散模型在视频帧插值中的作用，通过 Transformer tokenizer 压缩中间帧为语义丰富的 1D token 表示、采用 DiT 替代 U-Net 架构、引入双流上下文整合机制（时序注意力 + 帧差嵌入），在 DAVIS 等大运动基准上 LPIPS 降低近 10%，且仅需 2 步去噪即可实现高质量生成。
tags:
  - CVPR 2025
  - 图像生成
  - 视频帧插值
  - 扩散模型
  - DiT
  - Transformer
  - 大运动
  - 时序注意力
---

# EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation

**会议**: CVPR 2025  
**arXiv**: [2503.15831](https://arxiv.org/abs/2503.15831)  
**代码**: https://github.com/bbldCVer/EDEN  
**领域**: 图像/视频生成  
**关键词**: 视频帧插值, 扩散模型, DiT, Transformer tokenizer, 大运动, 时序注意力

## 一句话总结

提出 EDEN，从输入表示、模型架构和训练范式三个维度全面增强扩散模型在视频帧插值中的作用，通过 Transformer tokenizer 压缩中间帧为语义丰富的 1D token 表示、采用 DiT 替代 U-Net 架构、引入双流上下文整合机制（时序注意力 + 帧差嵌入），在 DAVIS 等大运动基准上 LPIPS 降低近 10%，且仅需 2 步去噪即可实现高质量生成。

## 研究背景与动机

**领域现状**：视频帧插值（VFI）旨在合成起始帧和结束帧之间的中间帧。传统方法依赖光流估计来warp中间帧，近年扩散模型开始用于VFI任务，通过在隐空间中直接生成中间帧，避免了显式的光流warp。

**现有痛点**：尽管扩散方法理论上有优势，但在大运动、非线性运动场景中仍然表现不佳，生成的帧常出现运动模糊和时序不一致。更关键的是，作者发现现有扩散VFI方法中扩散过程对最终生成质量的影响微乎其微——直接从随机噪声解码 vs 从去噪后的latent解码，两者的感知差异极小，说明扩散过程几乎没有在发挥作用。

**核心矛盾**：扩散过程本应是生成模型的核心能力来源，但在现有VFI框架中却被严重"架空"——问题出在latent表示不够语义化、架构不适合时序建模、训练范式缺乏运动感知。

**本文目标** 如何从根本上放大扩散过程在VFI中的贡献，使其真正能够处理大运动和复杂运动场景？

**切入角度**：作者从三个可改进的维度切入——(1) 用 Transformer tokenizer 替代 2D VAE 来获得更优的隐表示；(2) 用 DiT 替代 U-Net 获得更好的时序建模能力；(3) 引入双流上下文整合增强训练范式。

**核心 idea**：通过改善隐表示质量、替换为更适合VFI的DiT架构、引入帧差嵌入的双流上下文机制，三管齐下使扩散过程真正成为VFI生成质量的决定性因素。

## 方法详解

### 整体框架

EDEN 的流程分为两阶段训练：(1) 训练 Transformer tokenizer，将中间帧压缩为紧凑的 1D latent token，编码器和解码器均包含 4 个 Transformer block，每个 block 含金字塔特征融合模块和时序注意力模块；(2) 训练 DiT 扩散模型，基于 latent token 进行去噪生成，在每个 DiT block 中引入时序注意力和帧差嵌入的双流上下文整合。推理时，DiT 从噪声生成 latent token，再由 tokenizer 解码器重建中间帧。

### 关键设计

1. **Transformer Tokenizer 与金字塔特征融合**:

    - 功能：将中间帧压缩为语义丰富的 1D token 序列，替代传统 2D VAE 的网格表示
    - 核心思路：编码器将输入图像分为 patch 得到大尺度 token $I_t^l$，经平均池化得到小尺度 token $I_t^s$，拼接后通过自注意力实现多尺度特征融合（Pyramid Feature Fusion Module, PFFM），然后取小尺度位置的输出传入后续层。解码器反向操作：从小尺度 token 出发，插值得到大尺度 token 再做融合。编解码器各 4 个 Transformer block，隐维度 768。
    - 设计动机：TiTok 已证明 1D 序列表示比 2D 网格在紧凑隐空间中捕获更多高层语义信息。金字塔融合模块则借鉴多尺度特征的传统优势，在不同运动尺度下捕获细粒度细节，大小尺度 token 的比例为 $m = 4n$。

2. **DiT 扩散模型 + 双流上下文整合**:

    - 功能：在扩散过程中有效整合起始帧和结束帧的上下文信息，增强运动动态建模
    - 核心思路：采用 12 层 DiT block 作为骨干，在每个 self-attention 层之后插入时序注意力层（Temporal Context）——将起止帧的 token 与当前 latent token 在空间位置对应处拼接，做时序注意力以整合完整的帧间信息。同时引入帧差上下文整合（Difference Context）——计算起止帧的余弦相似度，归一化后通过 MLP 转为差异嵌入，加到时间步嵌入上作为 adaLN 的条件输入。
    - 设计动机：时序注意力能够隐式对齐帧间的空间位置对应关系，比交叉注意力在未见分辨率上的鲁棒性更好。帧差嵌入则显式编码了运动幅度的先验信息，使模型能根据运动大小自适应调整生成策略。

3. **多分辨率多帧间隔微调**:

    - 功能：提升模型对不同分辨率和运动幅度的泛化能力
    - 核心思路：第一阶段在固定低分辨率（256×448）和小帧间隔（1~5帧）上训练至收敛。第二阶段随机选择不同间隔长度和分辨率的帧对进行微调，位置嵌入通过插值适配不同尺寸。
    - 设计动机：实际视频分辨率和运动幅度差异大，ViT的位置嵌入在不同分辨率间迁移是已知难题。先固定低分辨率训练充分学习运动模式，再多尺度微调提升泛化性，是一种简单有效的策略。

### 损失函数

- **Tokenizer 训练损失**：$\mathcal{L}_{tok} = \lambda_1 \mathcal{L}_1 + \lambda_p \mathcal{L}_p + \lambda_G \mathcal{L}_G + \lambda_{kl} \mathcal{L}_{kl}$，包含 L1 重建损失、感知损失（LPIPS）、patch 对抗损失和轻微 KL 正则，权重分别为 1.0, 1.0, 0.5, 1e-6
- **DiT 训练损失**：采用 Flow Matching 范式，前向过程定义为 $x_t = (1-t)x_0 + t\varepsilon$，速度场预测损失 $\mathcal{L}_{dit} = \mathbb{E}_{t,p_t(z)} \|v_\Theta(z,t) - u_t(z)\|_2^2$

## 实验关键数据

### 主实验表

| 方法 | DAVIS LPIPS↓ | DAVIS FloLPIPS↓ | DAIN-HD LPIPS↓ | SNU-Extreme LPIPS↓ | 推理时间(s) |
|:--|:--|:--|:--|:--|:--|
| VFIMamba | 0.1084 | 0.1486 | 0.1426 | 0.1154 | 0.230 |
| SGM-VFI | 0.1140 | 0.1571 | 0.1423 | 0.1205 | 0.136 |
| LBBDM | 0.0963 | 0.1313 | 0.1471 | 0.1101 | 1.689 |
| **EDEN (Ours)** | **0.0874** | **0.1201** | **0.1321** | **0.0986** | 0.130 |

### 消融实验

| 消融项 | DAVIS LPIPS | DAIN-HD LPIPS | 说明 |
|:--|:--|:--|:--|
| 无时序注意力 | 0.1731 | - | 性能大幅下降 |
| 仅编码器添加 | 0.0773 | - | 有提升但不充分 |
| 编解码器均添加 | 0.0150 | - | 最优 |
| Cross-Attn vs Temp-Attn | 0.0547 vs 0.0548 | 0.0718 vs 0.0515 | 时序注意力在未见分辨率更鲁棒 |
| 无帧差嵌入 | 0.0976 / 0.1425 | 0.1327 / 0.2376 | 帧差嵌入带来显著提升 |
| 无 PFFM | 0.0564 / 0.0596 | 0.0799 / 0.0926 | 金字塔融合有效 |
| Latent dim 16 vs 24 (w/ DiT) | 0.1538 | 0.1641 | 16维反而优于24维 |

### 关键发现

- **扩散步数极少即可**：仅需 2 步去噪即可获得高质量生成，使 EDEN 的推理速度比之前扩散方法快数倍，甚至快于部分光流方法
- **时序注意力 > 交叉注意力**：在训练分辨率下两者相当，但在未见分辨率上时序注意力显著更优（DAIN-HD: 0.0515 vs 0.0718）
- **Latent维度不是越高越好**：24维 tokenizer 重建能力更强，但与 DiT 组合后反而不如 16 维，因为 DiT 的生成能力难以驾驭过高维度的隐空间

## 亮点与洞察

1. **诊断性发现开路**：作者首先通过对比实验（随机噪声 vs 去噪latent解码差异极小）揭示了现有扩散VFI方法中扩散过程被"架空"的本质问题，这个诊断本身就很有价值
2. **三维度全面增强**：从表示（tokenizer）、架构（DiT）、训练范式（双流上下文）三个正交维度同时改进，策略完整且互补
3. **2步去噪的极致效率**：无需多步迭代即可生成高质量结果，使扩散方法在VFI上首次具有实用性（推理速度 0.13s/帧，与光流方法可比）
4. **帧差嵌入的巧妙设计**：将运动幅度信息以余弦相似度→标准化→MLP嵌入的方式注入adaLN条件，简单但非常有效

## 局限性与可改进方向

1. **训练成本高**：需要先训 tokenizer 再训 DiT，两阶段训练增加了总体训练开销
2. **仅评估感知指标**：论文完全跳过了 PSNR/SSIM 等重建指标，虽给出了理由但仍让人对细节保真度存疑
3. **缺少任意时刻插值**：当前设计面向固定中间时刻，未展示对任意时刻 $t \in (0,1)$ 的控制能力
4. **分辨率扩展性**：虽然多分辨率微调有效，但 Transformer tokenizer 在超高分辨率上的计算成本仍需关注

## 相关工作与启发

- **TiTok**：证明 1D token 序列优于 2D 网格表示，直接启发了本文的 Transformer tokenizer 设计
- **LBBDM**：连续布朗桥扩散是之前最强的扩散VFI方法，EDEN 在DAVIS上将其 LPIPS 从 0.0963 降至 0.0874（约 9.2%↓）
- **Flow Matching**：采用 rectified flow 的直线路径定义前向过程，使训练更高效，也是仅需 2 步去噪的理论基础
- **启发**：扩散模型在视频任务中的效果严重依赖于隐表示的质量——不是"加扩散就能好"，而需要确保扩散过程在一个语义丰富、运动感知的空间中工作

## 评分

⭐⭐⭐⭐ — 从问题诊断到三维度解决方案都很系统，2步去噪的效率优势在实用性上意义重大；但两阶段训练成本和纯感知指标评估略显不足。

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](../../ICCV2025/image_generation/tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)
- [3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](../../ECCV2024/image_generation/dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)

<!-- RELATED:END -->
