---
title: >-
  [论文解读] ReGround: Improving Textual and Spatial Grounding at No Cost
description: >-
  [ECCV 2024][目标检测][textual grounding] 通过将 GLIGEN 中 Gated Self-Attention (GSA) 与 Cross-Attention (CA) 的串行连接改为并行连接（网络重连），在不引入任何新参数、不需要微调、不增加计算开销的前提下，显著缓解了文本定位与空间定位之间的权衡问题。
tags:
  - ECCV 2024
  - 目标检测
  - textual grounding
  - spatial grounding
  - network rewiring
  - GLIGEN
  - 扩散模型
---

# ReGround: Improving Textual and Spatial Grounding at No Cost

**会议**: ECCV 2024  
**arXiv**: [2403.13589](https://arxiv.org/abs/2403.13589)  
**代码**: https://re-ground.github.io  
**领域**: 视觉-语言 / 布局引导图像生成  
**关键词**: textual grounding, spatial grounding, network rewiring, GLIGEN, diffusion model

## 一句话总结
通过将 GLIGEN 中 Gated Self-Attention (GSA) 与 Cross-Attention (CA) 的串行连接改为并行连接（网络重连），在不引入任何新参数、不需要微调、不增加计算开销的前提下，显著缓解了文本定位与空间定位之间的权衡问题。

## 研究背景与动机

**领域现状**：扩散模型驱动的文本到图像（T2I）生成取得了显著进展，GLIGEN 通过引入 Gated Self-Attention 模块实现了基于 bounding box 的空间定位能力，被众多下游任务采用。

**现有痛点**：GLIGEN 在空间定位准确的同时，经常忽略文本 prompt 中的关键描述信息（如 "low poly illustration"、"draped with a colorful blanket" 等），作者称这一现象为 **description omission**（描述遗漏）。

**核心矛盾**：GSA 和 CA 在 GLIGEN 中采用串行结构，GSA 输出作为 CA 的输入，导致空间定位信号在到达 CA 之前已经主导了特征表示，压制了文本条件的影响力。降低 GSA 激活时长（scheduled sampling）虽能改善文本定位，但会牺牲空间定位精度。

**本文目标**：消除串行架构带来的文本-空间定位权衡，使两种定位能力互不干扰。

**切入角度**：分析发现 CA 不影响空间定位（移除 CA 后物体仍在正确位置），而 GSA 的串行放置会干扰文本定位。因此可以将两者改为并行。

**核心 idea**：将 GSA 和 CA 从串行改为并行——推理时仅重连网络即可，不需要任何训练。

## 方法详解

### 整体框架
ReGround 基于预训练的 GLIGEN 模型，在推理时修改 U-Net 中每一层的注意力模块连接方式。原始 GLIGEN 的流程为：Residual Block → Self-Attention → **GSA → CA**（串行）。ReGround 将其改为：Residual Block → Self-Attention → **GSA ∥ CA**（并行），两个模块的输出相加后流入下一层。

### 关键设计

1. **Description Omission 问题分析**

    - 功能：系统分析 GLIGEN 中文本描述被忽略的根因
    - 核心思路：通过 scheduled sampling 实验（调节 γ 从 1.0 到 0.0），发现 GSA 激活时间越长，文本定位越差；但缩短 GSA 则空间定位下降。这是一个不可调和的权衡
    - 设计动机：证明问题出在架构设计而非参数，为网络重连提供理论依据

2. **Cross-Attention 对空间定位的影响实验**

    - 功能：验证 CA 是否影响空间定位
    - 核心思路：移除 GLIGEN 中所有 CA 模块，直接让 GSA 输出传递到下一层 ($F \leftarrow F_{GSA}$)。结果发现物体剪影仍精确落在 bounding box 内
    - 设计动机：证明 CA 不依赖 GSA 的输出来完成空间定位，因此两者可以独立运行

3. **Network Rewiring：从串行到并行**

    - 功能：核心创新——在推理时将 GSA 和 CA 改为并行
    - 核心思路：原始 GLIGEN 串行公式：
    $F_{GSA} \leftarrow \text{GSA}(F_{SA}, \{g_i\}) + F_{SA}$
    $F \leftarrow \text{CA}(F_{GSA}, c) + F_{GSA}$
      ReGround 并行公式：
    $F \leftarrow \underbrace{\text{GSA}(F_{SA}, \{g_i\})}_{\text{spatial grounding}} + \underbrace{\text{CA}(F_{SA}, c)}_{\text{textual grounding}} + \underbrace{F_{SA}}_{\text{residual}}$
    - 设计动机：并行结构下，CA 的输入从 $F_{GSA}$ 恢复为 $F_{SA}$，这正是原始 LDM 中 CA 本该接收的输入。GSA 的输入不变，所以空间定位不受影响。两条路径独立工作，互不干扰

4. **Gated Self-Attention 回顾**

    - 功能：解释 GLIGEN 中 grounding token 的构造
    - 核心思路：$g_i = \mathcal{G}(\mathcal{T}(p_i), \mathcal{F}(b_i))$，其中 $\mathcal{T}$ 是文本编码器，$\mathcal{F}$ 是 Fourier 位置编码，$\mathcal{G}$ 是浅层 MLP。GSA 在视觉特征 $(f_1,...,f_{N_l})$ 与 grounding tokens $(g_1,...,g_M)$ 之间做联合自注意力
    - 设计动机：理解 GSA 机制是分析其与 CA 交互的基础

### 损失函数 / 训练策略
- **无需训练**：ReGround 的关键优势在于完全不需要训练或微调。它仅在预训练 GLIGEN 的推理阶段修改注意力模块的连接方式
- **Scheduled Sampling 兼容**：可以与 GLIGEN 的 scheduled sampling 策略 $\beta_t$ 结合使用，进一步调节效果
- **零额外开销**：不引入新参数、不增加计算量、不增加内存占用

## 实验关键数据

### 主实验
| 数据集 | 方法 | CLIP Score ↑ | YOLO Score ↑ |
|--------|------|-------------|-------------|
| MS-COCO-2014 | GLIGEN (γ=1.0) | 30.44 | 58.13 |
| MS-COCO-2014 | GLIGEN (γ=0.1) | 31.65 | 22.75 |
| MS-COCO-2014 | **ReGround (γ=1.0)** | **31.29** | **56.96** |
| MS-COCO-2017 | GLIGEN (γ=1.0) | 30.47 | 58.30 |
| MS-COCO-2017 | **ReGround (γ=1.0)** | **31.06** | **57.04** |
| NSR-1K-GPT Counting | GLIGEN (γ=1.0) | 32.46 | 65.36 |
| NSR-1K-GPT Counting | **ReGround (γ=1.0)** | **33.20** | **63.92** |

### 人类评测与偏好
| 评估方式 | ReGround 偏好率 | GLIGEN 偏好率 |
|----------|---------------|--------------|
| User Study (92人) | **70.05%** | 29.95% |
| PickScore (COCO-2017) | **55.66%** | 44.34% |
| PickScore (COCO-Drop) | **57.57%** | 42.43% |

### 关键发现
- ReGround 在 γ=1.0 时即获得 GLIGEN 从 γ=1.0 降到 γ=0.1 所带来的 **70.25%** CLIP 提升，而 YOLO Score 仅下降 **3.31%**
- 在 COCO-Drop（移除50%类别的bounding box）场景下，ReGround 相比 GLIGEN 的 CLIP 优势扩大到原来的 **1.57 倍**
- FID 指标上 ReGround 始终优于 GLIGEN，说明图像质量也有提升
- 作为 BoxDiff 的 backbone 替换 GLIGEN 后，同样获得显著的文本定位提升

## 亮点与洞察
- **极简但深刻的分析**：通过移除 CA 的实验巧妙证明了空间定位不依赖 CA，为并行化提供了坚实的理论支撑。这种"做减法"的分析思路值得学习
- **零成本改进的典范**：不需要训练、不增加参数、不增加计算的改进在实际应用中价值巨大。这类方法类似于 FreeU 的思路——通过分析网络内部机制找到免费的性能提升

## 局限与展望
- 仅在 GLIGEN 架构上验证，对其他空间定位方法（如 ControlNet）的适用性未探索
- 并行化虽然不影响空间定位准确度，但目前并行权重固定为等权（1:1），是否存在更优的加权方案未讨论
- 对于某些同时高度依赖空间定位和文本细节的复杂场景，改进幅度可能有限

## 相关工作与启发
- **vs GLIGEN [Li et al., CVPR 2023]**：ReGround 直接建立在 GLIGEN 上，通过仅修改推理时的网络连接方式即获得显著提升
- **vs BoxDiff [Kim et al., ICCV 2023]**：BoxDiff 通过 cross-attention map 作为额外空间引导，ReGround 可以与其组合使用，进一步提升
- **vs FreeU [Si et al., ICCV 2023]**：两者都是分析 U-Net 内部机制后提出"免费"改进的工作，思路一脉相承

## 评分
- 新颖性: ⭐⭐⭐⭐ 想法极度简单但分析深刻，串行→并行的网络重连出人意料地有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集定量评估+用户研究+PickScore+FID，还验证了作为其他方法backbone的泛化性
- 写作质量: ⭐⭐⭐⭐⭐ 分析逻辑清晰，从问题发现到原因分析到解决方案一气呵成
- 价值: ⭐⭐⭐⭐ 对所有使用GLIGEN的下游任务都有即插即用的价值，实用性强

<!-- RELATED:START -->

## 相关论文

- [AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)
- [SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)
- [Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/object_detection/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs](../../CVPR2025/object_detection/bacon_improving_clarity_of_image_captions_via_bag-of-concept_graphs.md)

<!-- RELATED:END -->
