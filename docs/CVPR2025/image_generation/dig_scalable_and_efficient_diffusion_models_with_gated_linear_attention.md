---
title: >-
  [论文解读] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention
description: >-
  [CVPR 2025][图像生成][扩散模型] DiG将门控线性注意力(GLA)引入扩散模型骨干网络，通过空间重定向增强模块(SREM)解决GLA的单向建模和缺乏局部感知问题，在ImageNet 256×256生成任务上超越DiT性能的同时，在1792分辨率下速度提升2.5倍、GPU显存节省75.7%。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型"
  - "门控线性注意力"
  - "亚二次复杂度"
  - "高效生成"
  - "ImageNet"
---

# DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention

**会议**: CVPR 2025  
**arXiv**: [2405.18428](https://arxiv.org/abs/2405.18428)  
**代码**: [https://github.com/hustvl/DiG](https://github.com/hustvl/DiG)  
**领域**: 扩散模型  
**关键词**: 扩散模型, 门控线性注意力, 亚二次复杂度, 高效生成, ImageNet

## 一句话总结

DiG将门控线性注意力(GLA)引入扩散模型骨干网络，通过空间重定向增强模块(SREM)解决GLA的单向建模和缺乏局部感知问题，在ImageNet 256×256生成任务上超越DiT性能的同时，在1792分辨率下速度提升2.5倍、GPU显存节省75.7%。

## 研究背景与动机

**领域现状**：扩散模型已成为视觉生成的主流范式，其骨干网络从U-Net演进到Vision Transformer (ViT/DiT)。DiT凭借Transformer的可扩展性被Sora、Stable Diffusion 3等前沿工作采用。然而，DiT中的自注意力机制计算复杂度为 $O(T^2)$（$T$ 为序列长度），在处理高分辨率图像（长序列）时面临严重的效率瓶颈。

**现有痛点**：现有的亚二次时间复杂度替代方案主要基于Mamba (SSM)，如DiS、DiffuSSM等。但Mamba类方法在模型尺寸增大时效率提升有限——其复杂的block设计和无法高效利用GPU tensor core导致大规模模型的实际吞吐量不理想。DiS-XL/2在1024分辨率下的速度仅为DiT-XL/2的60%左右。

**核心矛盾**：扩散模型需要处理越来越长的token序列（高分辨率图像、视频生成），需要亚二次复杂度的骨干网络来突破效率瓶颈，但现有替代方案要么性能不如Transformer，要么实际效率不够理想。

**本文目标**：将门控线性注意力(GLA) Transformer引入扩散模型，在保持DiT可扩展性和生成质量的前提下，实现真正的亚二次复杂度高效推理。

**切入角度**：GLA是一种高效的线性注意力Transformer变体，在NLP领域展示了优秀的性能。但直接将GLA用于2D图像生成面临两个挑战：(1) GLA是单向(causal)序列建模，而图像需要双向上下文；(2) GLA缺乏局部空间感知能力。

**核心 idea**：设计一个轻量级的空间重定向与增强模块(SREM)，通过逐层交替扫描方向实现全局上下文建模，辅以恒等初始化的深度可分离卷积提供局部感知，将GLA无缝适配为2D扩散骨干网络。

## 方法详解

### 整体框架

DiG采用Latent Diffusion的流水线：输入图像经VAE编码器得到 $32 \times 32 \times 4$ 的隐空间表示，经patchify层转为token序列（patch size=2时序列长度256），加上位置嵌入和条件嵌入（时间步+类别标签）后通过 $N$ 层DiG Block处理，最终线性投影头输出预测噪声和协方差。整体架构忠实于DiT的设计哲学，仅将自注意力替换为GLA。

### 关键设计

1. **空间重定向与增强模块 (SREM)**:

    - 功能：解决GLA单向建模的局限性，实现2D图像的全局上下文感知和局部空间信息捕获
    - 核心思路：SREM包含两个组件——**逐层扫描方向控制**和**深度可分离卷积(DWConv2d)**。对于扫描方向，每个DiG Block只处理一个方向的GLA扫描，然后在block末尾通过高效矩阵操作（转置2D token矩阵 + 翻转序列）改变下一个block的扫描方向。四种基本扫描模式（左→右、右→左、上→下、下→上）交替使用，形成crisscross覆盖。对于局部感知，在SREM中插入一个 $3 \times 3$ 的DWConv2d层，采用**恒等初始化**（仅中心权重为1、周围为0），几乎不增加参数量
    - 设计动机：单纯的双向扫描(FID 69.28)不如crisscross四向扫描(FID 62.06)；而DWConv2d的恒等初始化解决了标准随机初始化导致的收敛缓慢问题，因为模型可以在初始阶段以恒等映射开始，逐步学习局部信息

2. **DiG Block设计**:

    - 功能：构成DiG网络的基本计算单元，将GLA、FFN和SREM有机组合
    - 核心思路：每个DiG Block的前向流程为：(1) 将时间步嵌入 $\mathbf{t}$ 和类别嵌入 $\mathbf{y}$ 相加后通过MLP回归adaLN的scale/shift参数 $\alpha, \beta, \gamma$；(2) 对输入进行adaLN归一化后送入GLA计算全局注意力；(3) 再经adaLN归一化后通过FFN；(4) 将序列reshape为2D后通过DWConv2d捕获局部信息；(5) 最后进行扫描方向转换。整个block遵循DiT的adaLN-Zero条件注入方式
    - 设计动机：保持与DiT尽可能一致的架构设计，方便直接套用DiT的训练配方和超参数，降低迁移成本

3. **硬件友好的效率设计**:

    - 功能：在GPU上实现高效的线性注意力计算
    - 核心思路：GLA的训练复杂度为 $O(TMD + TD^2)$（$M$ 为chunk size），当序列长度 $T > D$ 时优于标准注意力的 $O(T^2D)$。DiG采用GLA的chunk-wise并行实现，将序列分为若干chunk在SRAM中完成计算，避免HBM的带宽瓶颈。SREM中的矩阵转置和翻转操作都是 $O(T)$ 的高效操作。整体Gflops仅为同尺寸DiT的77-79%
    - 设计动机：GLA原生支持硬件友好的chunk计算（充分利用tensor core），这是其相比Mamba等SSM方法在大模型上效率更高的关键原因

### 损失函数 / 训练策略

完全遵循DiT的训练策略：用 $\mathcal{L}_{simple}$（MSE）训练噪声预测网络 $\epsilon_\theta$，用完整的 $D_{KL}$ 损失训练协方差预测 $\Sigma_\theta$。使用AdamW优化器，常数学习率 $1e-4$，EMA衰减率0.9999。所有图像生成使用EMA模型。

## 实验关键数据

### 主实验 (ImageNet 256×256, class-conditional)

| 模型 | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|-------|-----|-----------|---------|
| DiT-S/2-400K | 68.40 | — | — | — | — |
| **DiG-S/2-400K** | **62.06** | 11.77 | 22.81 | 0.39 | 0.56 |
| DiT-B/2-400K | 43.47 | — | — | — | — |
| **DiG-B/2-400K** | **39.50** | 8.50 | 37.21 | 0.51 | 0.63 |
| DiT-XL/2-400K | 19.47 | — | — | — | — |
| **DiG-XL/2-400K** | **18.53** | 6.06 | 68.53 | 0.63 | 0.64 |
| DiG-XL/2-1200K (cfg=1.5) | **2.84** | **5.47** | **250.36** | 0.82 | 0.56 |
| ADM-G,ADM-U | 3.94 | 6.14 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | — | 247.67 | 0.87 | 0.48 |

### SREM消融实验 (DiG-S/2, 400K iterations)

| 配置 | Flops (G) | FID-50K↓ |
|------|----------|----------|
| DiT-S/2 (基线) | 6.06 | 68.4 |
| DiG-S/2 (仅causal) | 4.29 | 175.84 |
| + 双向扫描 | 4.29 | 69.28 |
| + DWConv2d (随机初始化) | 4.30 | 96.83 |
| + DWConv2d (恒等初始化) | 4.30 | 63.84 |
| + Crisscross四向扫描 (Full SREM) | 4.30 | **62.06** |

### 关键发现

- **DiG在所有模型尺度上全面超越DiT**：从S到XL四个尺寸，DiG的FID均优于同配置的DiT，且Gflops仅为DiT的77-79%
- **效率优势在高分辨率下更为显著**：DiG-S/2在1792分辨率下比DiT-S/2快2.5倍且省75.7%显存；DiG-XL/2在2048分辨率下比Flash-DiT-XL/2快1.8倍
- **SREM的每个组件都至关重要**：单向GLA的FID为175.84（灾难性差），双向扫描降至69.28，加DWConv降至63.84，四向crisscross最终达62.06
- **恒等初始化对DWConv至关重要**：随机初始化的DWConv反而使FID恶化至96.83，而恒等初始化降至63.84——这是一个重要的工程细节
- **良好的缩放性**：随着模型尺寸和序列长度增加，DiG的FID持续下降，呈现与DiT一致的scaling行为
- **DiG-XL/2比Mamba-based DiS-XL/2快4.2倍(1024分辨率)**：GLA的chunk并行计算在大模型上的效率显著优于Mamba的sequential scan

## 亮点与洞察

- **恒等初始化DWConv是关键trick**：这个看似简单的初始化策略带来了巨大的性能差异（FID 96.83 vs 63.84），原理是让模型在训练初期以恒等映射起步，避免卷积权重的随机扰动破坏GLA学到的全局特征。这个技巧可以迁移到任何在序列模型中引入卷积的场景
- **线性注意力在扩散模型中的首次成功探索**：DiG是首个基于线性注意力Transformer的扩散骨干网络，证明线性注意力可以替代quadratic attention同时提升效率，为大规模视觉生成开辟了新路径
- **Crisscross扫描策略的有效性**：逐层交替四个方向的扫描，使每个patch最终能"看到"来自四个方向的全局信息，以极低的开销弥补线性注意力的因果局限

## 局限与展望

- **仅在ImageNet 256×256上验证**：未展示在更高分辨率（512、1024）的generation quality，虽然效率分析覆盖了高分辨率
- **未探索text-to-image等复杂条件生成**：DiG目前仅验证了class-conditional生成，能否在CLIP条件、文本条件等更复杂场景中保持优势尚未验证
- **与最新方法的对比不够全面**：论文未与Diffusion-RWKV、ZigMa等其他亚二次方法在相同设置下充分对比
- **作者自述的局限**：尚未探索在DiG上构建类似Sora的大规模基础模型
- 改进方向：将DiG扩展到text-to-image和video generation任务；探索与FlashLinearAttention的更深度集成；在512和1024分辨率上做完整的generation quality评估

## 相关工作与启发

- **vs DiT**: DiT是DiG的直接基线。DiG保持了DiT的架构设计哲学（patchify、adaLN-Zero），仅将self-attention替换为GLA+SREM，因此可以直接复用DiT的超参数
- **vs DiS (Mamba-based)**: DiS基于Mamba构建扩散骨干，但Mamba的sequential scan和复杂block设计在大模型上效率不高。DiG-XL/2在1024分辨率下比DiS-XL/2快4.2倍
- **vs Flash-DiT**: 即使DiT配合最先进的FlashAttention-2优化，在2048分辨率下仍比DiG-XL/2慢1.8倍，说明线性注意力的本质优势无法被attention优化完全弥补

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将线性注意力引入扩散模型，SREM设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 完整的消融实验和多尺度scaling分析，但缺少高分辨率生成质量评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，效率分析详尽
- 价值: ⭐⭐⭐⭐⭐ 为下一代扩散模型骨干网络提供了有力候选，对视频/高分辨率生成有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[NeurIPS 2025\] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling](../../NeurIPS2025/image_generation/dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)
- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)

</div>

<!-- RELATED:END -->
