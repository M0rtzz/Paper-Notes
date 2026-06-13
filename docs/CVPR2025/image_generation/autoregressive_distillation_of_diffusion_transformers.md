---
title: >-
  [论文解读] Autoregressive Distillation of Diffusion Transformers
description: >-
  [CVPR 2025 (Oral)][图像生成][扩散模型蒸馏] 提出自回归蒸馏（ARD），利用ODE轨迹的历史信息而非仅当前去噪样本作为输入来预测未来步，通过token级时间嵌入和块级因果注意力掩码修改teacher transformer架构，在ImageNet-256上以4步达到FID 1.84，仅增加1.1%额外FLOPs。
tags:
  - "CVPR 2025 (Oral)"
  - "图像生成"
  - "扩散模型蒸馏"
  - "自回归蒸馏"
  - "曝光偏差"
  - "ODE轨迹历史"
  - "少步生成"
---

# Autoregressive Distillation of Diffusion Transformers

**会议**: CVPR 2025 (Oral)  
**arXiv**: [2504.11295](https://arxiv.org/abs/2504.11295)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 扩散模型蒸馏, 自回归蒸馏, 曝光偏差, ODE轨迹历史, 少步生成

## 一句话总结

提出自回归蒸馏（ARD），利用ODE轨迹的历史信息而非仅当前去噪样本作为输入来预测未来步，通过token级时间嵌入和块级因果注意力掩码修改teacher transformer架构，在ImageNet-256上以4步达到FID 1.84，仅增加1.1%额外FLOPs。

## 研究背景与动机

**领域现状**：基于Transformer架构的扩散模型（DiT）在高保真图像生成和高分辨率扩展方面展现了强大能力，但其迭代采样过程极其消耗计算资源（通常需要数十到数百步）。蒸馏方法将概率流ODE的解压缩到少步student模型中，是加速采样的主流方案。

**现有痛点**：现有蒸馏方法（如Progressive Distillation、Consistency Models等）以最近一次去噪后的样本作为student的输入。这带来了**曝光偏差(exposure bias)**问题——训练时student看到的是teacher生成的"干净"轨迹，推理时则需要基于自己之前步骤的（可能有误差的）输出继续生成，误差会逐步累积。

**核心矛盾**：仅用当前单步去噪结果作为输入有两个根本缺陷：(1) 容易受累积误差影响，导致曝光偏差；(2) 丢失了ODE轨迹中的粗粒度历史信息，而这些信息对预测未来步很有价值。

**本文目标**：设计一种能利用ODE轨迹历史的蒸馏方法，同时缓解曝光偏差并提供更丰富的粗粒度信息源。

**切入角度**：从自回归的角度重新看待蒸馏——ODE轨迹是一个有序序列，每一步的预测可以"回顾"之前的历史步骤。这类似于语言模型中利用上下文历史来减少生成误差。

**核心 idea**：在teacher transformer中注入ODE轨迹历史，通过token级时间嵌入区分不同时间步的输入，用块级因果注意力确保信息流的正确方向——让历史步为当前预测提供额外的粗粒度引导。

## 方法详解

### 整体框架

ARD在标准的蒸馏流程基础上，将student模型的输入从单个去噪样本扩展为包含历史ODE轨迹的序列。给定一个预训练的diffusion transformer作为teacher，ARD修改其架构以接受多个历史时间步的输入，训练student在少步采样中利用这些历史信息做出更准确的预测。输入是噪声图像 + 历史轨迹点，输出是去噪后的预测。

### 关键设计

1. **Token级时间嵌入 (Token-wise Time Embedding)**:

    - 功能：使模型能区分来自不同时间步的输入token
    - 核心思路：在标准DiT中，所有token共享一个全局时间嵌入。ARD为轨迹中每个历史时间步的token添加独立的时间嵌入标记。具体来说，将当前步$x_t$和历史步$x_{t-1}, x_{t-2}, \dots$的token分别嵌入对应的时间信息，然后一起送入transformer。这样模型不仅知道"当前在哪个时间步"，还知道"每个输入token来自哪个历史时间步"。
    - 设计动机：如果不区分时间步，模型无法分辨当前输入和历史输入的语义差异。Token级时间嵌入让模型能自动学习如何差异化地处理不同时间步的信息——从近期步获取细节、从早期步获取结构。

2. **块级因果注意力掩码 (Block-wise Causal Attention Mask)**:

    - 功能：控制历史信息的流动方向，防止信息泄露
    - 核心思路：在transformer的注意力计算中，引入因果掩码确保当前时间步的token只能attend到自身和更早时间步的token，而不能"偷看"未来。掩码以block为单位——同一时间步内的所有token可以互相attend（标准self-attention），但不同时间步之间遵循严格的因果顺序。这类似于语言模型中的causal attention，但操作粒度是token block而非单个token。
    - 设计动机：ODE轨迹有天然的时间顺序——早期步包含粗粒度结构、后期步包含细粒度细节。因果注意力确保粗粒度信息只用于引导精细化，避免反向信息泄露导致的训练-推理不一致。

3. **底层注入策略 (Lower-Layer History Injection)**:

    - 功能：在transformer的较低层注入历史信息，高层只处理当前步
    - 核心思路：并非在所有transformer层都使用历史轨迹。ARD发现只在底部若干层注入历史token、在上层仅保留当前步token的设计既能获得性能提升又能保持效率。底层负责从历史中提取粗粒度结构信息并融入当前表示，上层专注于当前步的精细化生成。
    - 设计动机：(1) 将历史token注入所有层会显著增加计算量（attention的复杂度随token数平方增长），底层注入仅增加1.1%FLOPs；(2) 直觉上，粗粒度的全局结构信息应在早期层处理，fine-grained生成在高层进行——这与CNN中底层学习通用特征、高层学习任务特定特征的理念一致。

### 损失函数 / 训练策略

ARD使用标准的蒸馏损失——student预测与teacher在对应ODE步上的输出之间的距离。训练时利用teacher的ODE轨迹作为历史输入（减少曝光偏差），推理时使用student自身历史预测的轨迹。关键训练细节：基于预训练DiT初始化student，新增的时间嵌入和注意力掩码参数从零开始训练。

## 实验关键数据

### 主实验（ImageNet-256类条件生成）

| 方法 | 步数 | FID ↓ | FID退化(vs teacher) |
|------|------|-------|---------------------|
| Teacher (DDPM, 250步) | 250 | ~1.5 | - |
| Progressive Distillation | 4 | ~3.0 | ~1.5 |
| Consistency Distillation | 4 | ~2.5 | ~1.0 |
| **ARD** | **4** | **1.84** | **~0.3** |
| 基线蒸馏方法(avg) | 4 | - | ~1.5 |
| **ARD vs 基线** | **4** | - | **5×更少FID退化** |

### 消融实验

| 配置 | FID ↓ | 额外FLOPs | 说明 |
|------|-------|-----------|------|
| 基线（无历史） | ~2.5 | 0% | 标准蒸馏 |
| 全层注入历史 | ~1.82 | ~8% | 性能最好但开销大 |
| **底层注入（ARD）** | **1.84** | **1.1%** | 最佳性能-效率平衡 |
| 无因果掩码 | ~2.1 | 1.1% | 信息泄露导致退化 |
| 无token级时间嵌入 | ~2.2 | 1.1% | 无法区分历史步 |

### 关键发现

- **FID退化5倍缩减**：与基线蒸馏方法相比，ARD在4步采样下将FID退化（相对teacher的差距）减少了约5倍，说明历史信息有效缓解了曝光偏差。
- **极低的额外开销**：底层注入策略仅增加1.1%FLOPs，几乎是"免费"的性能提升。
- **在T2I上同样有效**：在1024分辨率文本到图像任务上，ARD超越了公开可用的蒸馏模型在prompt adherence score上的表现，同时FID退化极小。
- **因果掩码和时间嵌入缺一不可**：去掉任一组件都会显著降低性能（FID上升0.3+），说明正确的信息流控制是ARD成功的关键。
- **底层注入 ≈ 全层注入**：在性能上仅差0.02 FID，但节省约7%FLOPs，表明历史信息的价值主要体现在底层特征提取阶段。

## 亮点与洞察

- **自回归视角重新定义蒸馏**：将ODE轨迹看作一个序列，用自回归的方式处理——这个视角转换非常巧妙。训练时用teacher轨迹（类似teacher forcing），推理时用自身历史（自回归解码），完美契合了序列生成的范式。这个思路可迁移到任何基于迭代refinement的生成模型。
- **底层注入的高效设计**：发现粗粒度信息只需在底层注入的洞察非常实用——不仅适用于扩散蒸馏，任何需要融合多分辨率条件信息的transformer架构都可以借鉴这一发现。
- **曝光偏差问题的新解法**：传统应对曝光偏差的方法（如scheduled sampling、noise injection）通常需要修改训练目标或数据分布。ARD通过引入历史输入从根本上减少了对单步预测精度的依赖，是一种更优雅的解决方案。

## 局限与展望

- **需要修改teacher架构**：ARD需要修改transformer结构（添加时间嵌入和注意力掩码），不能直接应用于任意预训练模型。
- **历史长度受限**：目前主要使用1-2个历史步，更长的历史可能带来更多改进但也增加复杂度。
- **仅验证了类条件和T2I**：未探索在视频生成、3D生成等其他扩散模型应用中的效果。
- **改进方向**：探索自适应历史长度选择；将ARD与consistency model等其他蒸馏范式结合；扩展到视频和3D生成任务。

## 相关工作与启发

- **vs Progressive Distillation**: PD每次将步数减半，但仅用当前去噪样本作为输入，FID退化显著。ARD通过引入历史信息将退化减少5倍，说明信息量不足是PD的核心瓶颈。
- **vs Consistency Models**: CM通过一致性约束实现单步生成，但与teacher的gap仍然较大。ARD保留了多步采样的灵活性，在4步时达到更好的质量。
- **vs InstaFlow/Rectified Flow蒸馏**: 这些方法也做少步蒸馏但不利用轨迹历史。ARD的自回归思想可以作为一个通用的增强模块叠加在这些方法之上。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将自回归思路引入扩散蒸馏是全新视角，CVPR Oral实至名归
- 实验充分度: ⭐⭐⭐⭐ ImageNet-256和T2I双重验证，消融完整，但缺少与最新consistency model变体的对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述简洁优雅
- 价值: ⭐⭐⭐⭐⭐ 1.1%FLOPs换来5倍FID退化减少，实用价值极高；自回归蒸馏的框架有广泛的可扩展性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](../../CVPR2026/image_generation/pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](tinyfusion_diffusion_transformers_learned_shallow.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](random_conditioning_for_diffusion_model_compression_with_distillation.md)

</div>

<!-- RELATED:END -->
