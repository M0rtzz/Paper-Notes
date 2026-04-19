---
title: >-
  [论文解读] Reviving ConvNeXt for Efficient Convolutional Diffusion Models
description: >-
  [CVPR 2026][图像生成][扩散模型] 本文提出FCDM（Fully Convolutional Diffusion Model），将ConvNeXt架构适配为条件扩散模型backbone，仅用DiT-XL 50%的FLOPs即可在ImageNet上达到竞争性FID（2.03），且能在4块RTX 4090上训练XL模型，展示了全卷积架构在生成建模中被严重低估的效率优势。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - ConvNeXt
  - 全卷积
  - 高效生成
---

# Reviving ConvNeXt for Efficient Convolutional Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2603.09408](https://arxiv.org/abs/2603.09408)  
**代码**: 有（官方实现已公开）  
**领域**: 图像生成  
**关键词**: 扩散模型, ConvNeXt, 全卷积, 高效生成, 图像生成  
**机构**: KAIST, ETH Zürich, ISTI-CNR, University of Pisa

## 一句话总结
本文提出FCDM（Fully Convolutional Diffusion Model），将ConvNeXt架构适配为条件扩散模型backbone，仅用DiT-XL 50%的FLOPs即可在ImageNet上达到竞争性FID（2.03），且能在4块RTX 4090上训练XL模型，展示了全卷积架构在生成建模中被严重低估的效率优势。

## 研究背景与动机

**领域现状**: 扩散模型backbone经历了从卷积-注意力混合架构（DDPM、ADM、LDM）到全Transformer架构（DiT、SiT、FLUX）的演进。Transformer的可扩展性推动了FLUX、SD3等大规模模型的成功，但也带来了对GPU集群资源的强依赖。

**现有痛点**: DiT-XL/2需要7M步训练才能达到最佳FID，训练吞吐量仅80.5 it/s。Transformer的 $O(n^2)$ 计算复杂度在高分辨率下尤为严重——分辨率翻倍时DiT吞吐量下降约4×。这使得扩散模型的训练和推理成本成为主要瓶颈。

**核心矛盾**: 业界普遍认为"scaling Transformer = 更好的生成质量"，但ConvNet的局部性偏置、参数效率和硬件友好性在现代生成建模中几乎未被充分探索。ConvNeXt在分类任务上已展示与ViT匹配的性能，但在生成领域完全缺席。

**本文切入角度**: 将ConvNeXt改造为条件扩散模型的backbone，保持其核心设计（depthwise conv、inverted bottleneck、GRN），仅添加条件注入（AdaLN）和U-shaped布局，验证全卷积架构能否同时兼顾生成质量和计算效率。

## 方法详解

### 整体框架

FCDM在latent space中工作（与DiT一致）。输入RGB图像 $256 \times 256 \times 3$ 经VAE编码为 $32 \times 32 \times 4$ 的latent tensor，再经多个FCDM block处理后由VAE decoder解码回像素空间。FCDM block组织在一个简化的U-shaped架构中，encoder和decoder通过skip connections连接。

**核心设计理念**: 与DiT需要4个超参数（层数L、通道C、attention heads、patch size）不同，FCDM仅需**两个超参数**——block数L和隐通道数C。每次2×下采样时，L和C均翻倍。这一"Easy Scaling Law"极大简化了架构搜索空间。

### FCDM Block设计

FCDM block是对ConvNeXt block的最小化改造，保留了原始ConvNeXt的核心结构并添加条件注入能力：

**原始ConvNeXt Block流程**:
$\text{Input} \to 7\times7 \text{ DWConv} \to \text{LayerNorm} \to 1\times1 \text{ Conv}(\uparrow r) \to \text{GRN} \to 1\times1 \text{ Conv}(\downarrow r) \to \text{Output}$

**FCDM Block改造**:
- **条件注入**: 将LayerNorm替换为Adaptive LayerNorm（AdaLN）。一个轻量MLP将class embedding和timestep embedding拼接后映射为 $(\gamma, \beta, \alpha)$ 三组参数：$\gamma$ 和 $\beta$ 用于仿射变换归一化特征，$\alpha$ 作为最终输出的缩放因子
- **零初始化**: 遵循DiT的做法，将最终调制scale $\alpha$ 零初始化。这使得训练初期每个FCDM block表现为恒等映射，有利于深层网络的优化稳定性
- **7×7 Depthwise Convolution**: 保留ConvNeXt的大核depthwise conv，提供足够大的感受野来捕获空间上下文。消融实验证明7×7显著优于5×5和3×3

### Inverted Bottleneck——效率的核心来源

FCDM与DiCo最关键的结构差异在于通道维度的处理方式。这也是FCDM FLOPs仅为DiCo 75%的核心原因。

**DiCo的做法**: 在整个卷积模块中保持通道维度不变，通道扩展放在额外的feedforward模块中完成（两个1×1 conv）。

**FCDM的做法（Inverted Bottleneck）**:
1. 先做 $7 \times 7$ depthwise conv（通道数C，计算量 $\propto C$）
2. 用 $1 \times 1$ pointwise conv将通道扩展到 $rC$（expansion ratio $r=3$）
3. 经过GRN后，再用 $1 \times 1$ pointwise conv将通道压缩回 $C$

**关键trick**: Depthwise conv放在通道扩展**之前**而非之后。由于depthwise conv的计算量仅与输入通道数成正比（不涉及通道间交互），这种重排使得depthwise conv始终在低维通道上操作，计算量不随expansion ratio增长。而通道扩展后的高维特征仅由更轻量的pointwise conv处理，实现了"计算量不变+表征能力增强"的双赢。

**数值验证**: 在参数量对齐的条件下，FCDM各scale的FLOPs比例如下：

| 模型 | 参数量 | Blocks L | Channel C | FLOPs(G) | vs DiT | vs DiCo |
|------|--------|----------|-----------|----------|--------|---------|
| FCDM-S | 32.7M | 2 | 128 | 3.1 | 50.8% | 72.9% |
| FCDM-B | 127.7M | 2 | 256 | 12.2 | 53.0% | 72.3% |
| FCDM-L | 504.5M | 2 | 512 | 48.3 | 59.9% | 80.2% |
| FCDM-XL | 698.8M | 3 | 512 | 64.6 | 54.5% | 74.0% |

### GRN vs CCA——通道多样性的轻量实现

DiCo引入了Compact Channel Attention（CCA）来缓解通道冗余问题，其本质是通过额外的 $1 \times 1$ pointwise conv学习通道级注意力权重。

FCDM使用ConvNeXt V2中的**Global Response Normalization（GRN）**替代CCA。GRN主要由无参数操作组成：对每个通道计算全局L2范数，再进行响应归一化。两者目标完全一致——促进通道激活多样性、减少通道冗余——但GRN几乎不引入额外可学习参数。

特征可视化（论文Figure 7）直观展示了GRN的效果：经GRN处理后的64通道特征图呈现出明显的多样性，而处理前的特征存在大量冗余通道。

### 无需额外Feedforward模块

DiCo在卷积模块之外还包含一个feedforward模块（两个 $1 \times 1$ conv，用于通道扩展）。FCDM不需要此模块，因为inverted bottleneck结构已经在block内部完成了通道扩展和压缩。消融实验表明，给FCDM添加额外feedforward模块后FID从19.97暴涨到28.52——通道扩展做两次反而有害。

### U-shaped架构的简化设计

传统U-Net需要精心设计每个分辨率层级的block数和通道数。FCDM彻底简化了这一过程：
- **下采样规则**: 每次2×下采样时，block数L和通道数C均翻倍
- **Skip Connections**: encoder各层级的特征直接传递到decoder对应层级
- **不做分辨率特定的设计**: 所有层级使用相同的FCDM block结构，不引入分辨率相关的特殊处理

这使得从FCDM-S扩展到FCDM-XL只需调整两个数字（L从2→3，C从128→512），极大降低了超参数调优成本。

### 训练策略

完全沿用DiT/ADM的训练设置，不引入任何额外trick：
- **扩散过程**: $t_{\max}=1000$ 步，线性noise schedule（$\beta$从 $1 \times 10^{-4}$ 到 $2 \times 10^{-2}$），iDDPM协方差参数化
- **优化器**: AdamW，lr = $1 \times 10^{-4}$（constant schedule），无weight decay
- **训练精度**: fp32（未使用混合精度）
- **EMA**: decay factor = 0.9999
- **评估**: 250步DDPM采样，50K样本计算FID，有指导时使用classifier-free guidance

## 实验关键数据

### 主实验：ImageNet 256×256 多scale对比（400K步）

| 模型 | 架构类型 | FLOPs(G)↓ | 吞吐量(it/s)↑ | FID↓ | IS↑ |
|------|---------|-----------|---------------|------|-----|
| DiT-XL/2 | Transformer | 118.6 | 80.5 | 19.47 | - |
| DiG-XL/2 | Hybrid | 89.4 | 71.7 | 18.53 | 68.53 |
| DiCo-XL | Conv | 87.3 | 174.2 | 11.67 | 100.4 |
| DiC-H | Conv | 204.4 | 144.5 | 11.36 | 106.5 |
| **FCDM-XL** | **Conv** | **64.6** | **272.7** | **10.72** | **108.0** |

FCDM-XL在400K步FID最低（10.72），FLOPs最少（64.6G），吞吐量最高（272.7 it/s）。训练1M步后FID进一步降至7.91，而DiT-XL/2需要7M步才达到9.62。

### Benchmark结果（长训练 + Classifier-Free Guidance）

| 模型 | 训练epochs | FLOPs(G)↓ | 吞吐量↑ | FID↓ | IS↑ |
|------|-----------|-----------|---------|------|-----|
| DiT-XL/2 | 1400 | 118.6 | 80.5 | 2.27 | 278.2 |
| SiT-XL/2 | 1400 | 118.6 | 80.5 | 2.06 | 277.5 |
| DiCo-XL | 750 | 87.3 | 174.2 | 2.05 | 282.2 |
| **FCDM-XL** | **400** | **64.6** | **272.7** | **2.03** | **285.7** |

FCDM-XL用400 epochs达到FID 2.03（SOTA），训练epoch数比DiT少3.5×，比DiCo少1.9×。

### 512×512分辨率

FCDM-XL在512×512下1M步FID达7.46，优于DiT-XL/2 3M步的12.03（训练步数少7.5×）。值得关注的是分辨率翻倍时吞吐量变化：DiT下降约4×（全局attention的 $O(n^2)$ 效应），FCDM仅下降约2×（卷积的线性复杂度）。

### 消融实验（FCDM-L，200K步）

| 配置变化 | FLOPs(G) | FID↓ | IS↑ | 结论 |
|---------|----------|------|-----|------|
| Default (7×7 DWConv + GRN) | 48.3 | 19.97 | 69.19 | 基准 |
| → 5×5 DWConv | 48.2 | 20.48 | 66.69 | 感受野变小略降 |
| → 3×3 DWConv | 48.1 | 21.28 | 64.11 | 大核很重要（FID +1.3） |
| → CCA替换GRN | 48.3 | 23.85 | 61.60 | GRN远优于CCA（FID +3.9） |
| → 加Feedforward | 48.2 | 28.52 | 47.16 | 额外FFN有害（FID +8.5） |
| → 去Inverted Bottleneck | 48.3 | 28.76 | 52.20 | IB结构至关重要 |
| → ResNet block替换 | 48.4 | 31.14 | 49.10 | 现代ConvNeXt设计远优于经典ResNet |

### 计算资源

FCDM-XL可在4块RTX 4090（消费级GPU）上完成256×256 ImageNet训练，batch size 256，吞吐量约0.9 step/s。同样batch size可在单块A100 40GB上运行。相比之下，同规模DiT通常需要8块A100/H100。

## 亮点与洞察

- **两参数缩放法则**: L和C两个超参数定义整个网络，极简的设计空间大幅降低了架构搜索成本
- **Inverted Bottleneck的顺序重排**: depthwise conv放在通道扩展前是FLOPs节省25%（vs DiCo）的关键，这个trick可泛化到其他卷积生成架构
- **GRN的意外成功**: 原本为分类设计的ConvNeXt V2模块在生成任务中同样有效，且大幅优于专为扩散设计的CCA机制
- **分辨率友好性**: 卷积的线性计算复杂度使吞吐量随分辨率的衰减远小于Transformer，512×512场景优势更加显著
- **4块4090训练XL模型**: 对学术界和资源受限场景具有极大实用价值

## 局限性

- 尚未超越EDM-2和Simpler Diffusion等使用更先进训练框架（如改进noise schedule、预条件化）的SOTA方法
- 仅在ImageNet class-conditional上测试；text-to-image、video generation等复杂条件场景待验证
- 全卷积架构在超长距离依赖建模上理论弱于Transformer；全局语义一致性可能受限
- 仅使用fp32训练，混合精度训练下的效率提升和稳定性未探索

## 相关工作对比

- **vs DiT/SiT**: 完全用conv替代attention，参数量对齐时FLOPs减半，7×更少训练步达到同等FID。DiT的优势在于全局attention的理论建模能力和与文本条件的天然配合
- **vs DiCo**: 结构最相似的竞争者。FCDM通过inverted bottleneck重排和GRN替代CCA+FF实现25%的FLOPs节省，生成质量也略优
- **vs DiC**: DiC用标准3×3 conv，S/B scale硬件优化更好（throughput更高），但L/XL scale下FCDM全面领先

## 评分
- 新颖性: ⭐⭐⭐⭐ 将ConvNeXt引入扩散模型并非全新（DiC/DiCo在先），但inverted bottleneck的重排分析和与DiCo的系统对比有认知价值
- 实验充分度: ⭐⭐⭐⭐⭐ 4个scale × 多个训练步数、256/512双分辨率、详细消融和特征可视化、FLOPs/吞吐量/FID全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Figure 4的DiCo vs FCDM对比非常直观，与DiCo的三点差异分析透彻
- 价值: ⭐⭐⭐⭐ 4块4090训练XL模型对资源受限场景极具吸引力；两参数缩放法则有实际工程价值

<!-- RELATED:START -->

## 相关论文

- [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](ditic_aligned_diffusion_transformer_for_efficient.md)
- [Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)
- [Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](../../AAAI2026/image_generation/dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)

<!-- RELATED:END -->
