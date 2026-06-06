---
title: >-
  [论文解读] Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models
description: >-
  [ICML2026][语义分割][激活函数替代] 本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax…
tags:
  - "ICML2026"
  - "语义分割"
  - "激活函数替代"
  - "多项式网络"
  - "Hadamard乘积"
  - "MetaFormer"
  - "PolyNeXt"
---

# Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models

**会议**: ICML2026  
**arXiv**: [2605.20839](https://arxiv.org/abs/2605.20839)  
**代码**: https://github.com/jjwang8/PolyNeXt  
**领域**: 视觉骨干 / 图像识别 / 语义分割迁移  
**关键词**: 激活函数替代、多项式网络、Hadamard乘积、MetaFormer、PolyNeXt  

## 一句话总结
本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax，在 MetaFormer 风格骨干中无需常规激活函数也能在 ImageNet、鲁棒性和 ADE20K 分割上达到或超过激活式模型。

## 研究背景与动机
**领域现状**：现代视觉骨干几乎默认依赖 ReLU、GELU、SiLU 等点激活函数，以及 self-attention 中的 softmax 指数归一化。ConvFormer、CAFormer、ConvNeXt、ViT 等架构都把这些非线性作为高性能视觉表示的基础组件。

**现有痛点**：激活函数并不是唯一的非线性来源。已有 polynomial networks 说明乘法交互也能表达复杂函数，但很多方法需要从头设计专门架构，难以复用已有 MetaFormer/attention/conv 改进；同时深层多项式网络容易因为乘法放大而训练不稳定。

**核心矛盾**：如果直接去掉激活函数，模型可能缺少非线性或训练崩溃；如果保留复杂自定义多项式结构，又很难成为通用视觉模块。论文要证明的是：只替换标准模块里的非线性算子，同时保持接口不变，是否足够训练出竞争力 backbone。

**本文目标**：作者希望设计一套 activation-free 的 channel mixing、spatial convolution mixing 和 attention mixing 模块，使其可以插入 MetaFormer 风格架构，兼顾 ImageNet 分类、OOD 鲁棒性、ADE20K 语义分割和面向 FHE 的多项式推理潜力。

**切入角度**：Hadamard product 本身会产生输入的二阶多项式，层层堆叠后多项式次数随深度指数增长。只要控制残差幅度和梯度流，深度而窄的多项式网络可以获得足够表达力，而不需要点激活函数。

**核心 idea**：用“平行线性/卷积分支的逐元素乘法 + 稳定化残差设计”替代标准激活函数，让视觉 backbone 的非线性来自可组合的多项式交互。

## 方法详解
本文的核心是把常见视觉骨干里的三个非线性来源逐一改造成多项式模块。MLP 中的 GELU 被两个线性投影的 Hadamard product 替代；separable convolution 中的激活被粗/细两条卷积分支的乘法融合替代；attention 中的 softmax 指数核被多项式 kernel 替代。然后作者把这些模块组装成 PolyNeXt，并加入 Sigmoid-Scale、多输入 skip、depth-over-width 等稳定化策略。

### 整体框架
PolyNeXt 采用四阶段层级视觉 backbone，整体仍遵循 MetaFormer 模板：每个 cell 接收前两个 cell 的输出，先经过空间 mixer，再经过 PolyMLP。CPolyNeXt 在所有阶段使用 PolyConv；APolyNeXt 在前两阶段用 PolyConv 处理高分辨率局部信息，在后两阶段用 PolyAttn 处理低分辨率全局信息。Stem 是 stride 4 的 $7\times7$ 卷积，阶段之间用 stride 2 卷积下采样。

一个 cell 内可以包含多个 stack，每个 stack 是“空间混合器 + PolyMLP”。作者强调 depth-over-width：与其把单层做宽，不如堆更多较窄的多项式层，因为多项式次数随层数增长更快。为了避免乘法链路导致数值爆炸，每个 residual branch 都用可学习的 sigmoid 标量限制输出幅度。

### 关键设计
1. **PolyMLP 与 PolyConv 的激活替代**:

	- 功能：在 channel mixing 和局部 spatial mixing 中提供无激活函数的非线性。
	- 核心思路：PolyMLP 计算 $W_o((W_a x)*(W_b x))$，两个线性投影逐元素相乘后再投影回原维度。PolyConv 先用 pointwise 卷积得到 hidden feature，再用一个 dilation depthwise coarse branch 和一个 $3\times3$ fine branch 提取不同感受野，翻转其中一支的 channel 后逐元素相乘，再用卷积整合。
	- 设计动机：乘法分支能显式产生二阶交互，堆叠后形成高阶多项式。PolyConv 用异质感受野相乘，比两个相同结构分支更容易产生跨尺度交互。

2. **PolyAttn 的多项式注意力核**:

	- 功能：替代 self-attention 中 softmax 的指数非线性，使注意力也保持 activation-free。
	- 核心思路：PolyAttn 用 $A=(s\cdot QK^\top+1)^p$ 作为未归一化权重，其中 $s=\sigma(\lambda)$ 是每个 head 的可学习 scale，$p=4$。之后用 $\ell_1$ normalization 替代 softmax 归一化，并在 $Q,K,V$ 上加入 depthwise convolution 注入局部空间上下文，同时共享 $Q/K$ 投影节省参数。
	- 设计动机：softmax 的指数函数阻碍完全多项式推理，也不是注意力结构唯一可行的核。多项式核保留 query-key 相似度加权，又避免指数激活，并可兼容 window attention 或 sparse attention 等改进。

3. **深层多项式网络稳定化**:

	- 功能：让数百层 Hadamard-product 网络可以稳定训练。
	- 核心思路：Sigmoid-Scale 将残差写成 $y=x+\sigma(\lambda)f(x)$，并按深度初始化更小的 residual contribution；multi-input skip 让每个 cell 同时接收前一个和前两个 cell 的输出，经可学习 channel scale 相加后 LayerNorm；depth-over-width 在相近参数量下增加 stack 数。
	- 设计动机：乘法会放大大值，简单加深容易梯度和激活不稳定。残差幅度控制和跨 cell skip 提供数值安全阀，深度设计则释放多项式表达力。

### 损失函数 / 训练策略
模型按 ImageNet-1K 监督分类训练，训练 recipe 基于 MetaFormer/MONet 但使用更小 batch size 和更强 regularization。语义分割迁移使用 UperNet，在 ADE20K 上训练 160K iterations，采用 ConvNeXt recipe，并对 Sigmoid-Scale、多输入 skip 和 normalization 参数设特殊 weight decay 分组。论文还训练 LayerNorm 替换为 polynomial-compatible BatchNorm 的 fully polynomial 变体，以探索 FHE 友好推理。

## 实验关键数据

### 主实验
ImageNet-1K 主结果说明，PolyNeXt 在不同规模上都能接近或超过激活式 MetaFormer，也明显强于 prior polynomial networks。

| 模型 | Params | FLOPs | Top-1 | 说明 |
|------|--------|-------|-------|------|
| DTTN-T | 7.1M | 2.4G | 77.9 | prior polynomial tiny |
| MONet-T | 10M | 2.8G | 77.0 | prior polynomial tiny |
| CPolyNeXt-T | 6.4M | 1.2G | 80.2 | 更少参数/FLOPs 下高 2-3 点 |
| ConvFormer-S18 | 27M | 3.9G | 83.0 | 激活式 MetaFormer conv baseline |
| CPolyNeXt-S | 26M | 4.8G | 83.9 | 高 0.9 点 |
| DTTN-B | 36M | 12.3G | 82.4 | prior polynomial base |
| CPolyNeXt-B | 40M | 8.5G | 84.7 | 比 DTTN-B 高 2.3 点且 FLOPs 更低 |
| CAFormer-S18 | 26M | 4.1G | 83.6 | 激活式 hybrid baseline |
| APolyNeXt-S | 26M | 5.3G | 84.3 | 高 0.7 点 |
| CAFormer-M36 | 56M | 13.2G | 85.2 | 大模型 hybrid baseline |
| APolyNeXt-L | 57M | 13.3G | 85.2 | 持平 |

鲁棒性和下游分割结果也支持 polynomial backbone 的泛化。

| 任务 | 模型 | Clean / 主指标 | OOD / 下游指标 | 结论 |
|------|------|----------------|----------------|------|
| ImageNet-C/A/R/Sketch | CAFormer-S18 | 83.6 clean, IN-C 47.4, IN-A 33.5 | IN-R 48.7, IN-Sk 36.6 | 强 hybrid baseline |
| ImageNet-C/A/R/Sketch | APolyNeXt-S | 84.3 clean, IN-C 45.0, IN-A 39.6 | IN-R 49.7, IN-Sk 37.5 | clean 与鲁棒性同步提升，mCE 更低 |
| ADE20K UperNet | ConvFormer-S18 | 54M, 925G | 48.6 mIoU | MetaFormer conv baseline |
| ADE20K UperNet | CAFormer-S18 | 54M, 1024G | 48.9 mIoU | MetaFormer hybrid baseline |
| ADE20K UperNet | CPolyNeXt-S | 54M, 941G | 50.6 mIoU | 比 ConvFormer-S18 高 2.0 |
| ADE20K UperNet | APolyNeXt-S | 55M, 1121G | 49.9 mIoU | 比 CAFormer-S18 高 1.0 |

### 消融实验
消融直接检验“激活函数是否必要”和“稳定化是否关键”。

| 配置 | Δ Acc | 说明 |
|------|-------|------|
| CPolyNeXt-T baseline | 80.2 | 完整多项式卷积模型 |
| PolyMLP → MLP+GELU | -0.1 到 -0.4 | 加回 MLP 激活没有帮助 |
| PolyConv → SepConv+GELU | -0.9 | 标准 separable conv 更差 |
| 在一支乘法分支加 GELU | -0.4 | 破坏部分互梯度耦合 |
| 在乘积后加 GELU | -1.0 | 单个 gate 同时阻断两支梯度 |
| Hadamard → Addition | -22.3 | 乘法交互是核心非线性来源 |
| APolyNeXt-T baseline | 80.9 | 完整多项式注意力模型 |
| PolyAttn → Std Attn | -1.3 | 标准注意力替代整体结构明显更差 |
| polynomial kernel → softmax | -0.1 | kernel 本身不是唯一贡献，Q/K 共享和局部卷积也重要 |

| 稳定化/架构消融 | Δ Acc | 说明 |
|-------------------|-------|------|
| Sigmoid-Scale → free scalar | -0.5 | 初始化几何最关键，sigmoid 还有次级优化收益 |
| Sigmoid-Scale → LayerScale init=1e-6 | -0.8 | 传统 LayerScale 不够适配 |
| Sigmoid-Scale → LayerScale init=1.0 | -12.8 | 训练几乎崩溃 |
| 移除 multi-input skip | -0.6 | 跨 cell 梯度流有贡献 |
| 移除 cell 前 norm | -0.4 | 归一化位置重要 |
| 更宽 2 stacks/cell | -0.7 | 深度优于宽度 |
| 更宽 1 stack/cell | -1.5 | 多项式次数不足 |

### 关键发现
- 激活函数在这个设计里不是越多越好。加回 GELU 往往降低性能，说明乘法分支之间的 mutual gradient coupling 是有效非线性来源。
- Hadamard product 不可替代。换成 addition 掉 22.3 点，基本证明模型不是靠外壳结构，而是靠乘法交互表达。
- 稳定化是成败关键。没有合理残差尺度，深层多项式网络会因乘法放大而不稳定；Sigmoid-Scale 和多输入 skip 让接近 200 层训练成为可能。
- 分割迁移收益比分类边际更明显。CPolyNeXt-S 在 ADE20K 上比 ConvFormer-S18 高 2.0 mIoU，说明多项式 backbone 学到的表示不仅服务分类。

## 亮点与洞察
- 论文最有价值的点不是“再造一个新 backbone”，而是把 activation-free 设计做成标准 MLP/Conv/Attention 的接口级替换。这让它能继承 MetaFormer 生态，而不是孤立架构。
- 对“为什么激活会伤害”的解释很有启发：乘法的两支投影在反向传播中互相调制，GELU 的负区间会切断这种耦合；这和我们通常“激活增加表达力”的直觉相反。
- FHE 视角让工作不只是性能论文。完全多项式 BN 版本仍能达到 CPolyNeXt-S BN 82.7%，超过 ConvNeXt-T，这说明隐私计算友好网络不一定只能牺牲大量精度。
- Depth-over-width 的结论可迁移到其他 multiplicative architectures。乘法网络的能力来自可组合次数，而不是单层宽度，设计空间和常规 ReLU 网络不同。

## 局限与展望
- 训练 recipe 不完全通用。作者承认需要更小 batch、更强 regularization、渐进 dropout 和谨慎初始化，直接套标准训练配置可能不稳定。
- 深而窄的设计有吞吐开销。即使 FLOPs 接近，实际速度可能慢于更浅更宽的 MetaFormer。
- Hadamard product 对学习率敏感，乘法放大让超参调节更脆弱。
- Fully polynomial 版本仍只是朝 FHE 迈进，真正端到端加密推理还要解决归一化、注意力归一、硬件和数值范围等问题。
- 论文主要在 ImageNet/ADE20K 验证，迁移到检测、实例分割、多模态视觉编码器或视频 backbone 还需要进一步实验。

## 相关工作与启发
- **vs MONet / DTTN**: 这些 prior polynomial networks 更依赖定制架构，PolyNeXt 只替换标准模块中的非线性，性能更高且更容易迁移到 MetaFormer。
- **vs ConvFormer / CAFormer**: 二者依赖 separable conv、gated MLP 和 softmax attention；本文保留整体模板但把激活换成多项式交互，在同规模上匹配或超过它们。
- **vs StarNet / GLU**: StarNet 和 GLU 也用逐元素乘法，但仍保留激活；本文强调完全去掉点激活后，乘法本身足以提供非线性。
- **vs linear attention / efficient attention**: PolyAttn 不是单纯追求线性复杂度，而是替换 softmax 的指数核为多项式核；它可以与 window/sparse attention 等结构进一步结合。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 激活替代思路简洁但很系统，尤其是 PolyConv/PolyAttn 与稳定化组合有辨识度。
- 实验充分度: ⭐⭐⭐⭐☆ ImageNet、鲁棒性、ADE20K、FHE 变体和消融都较完整；检测/视频等任务还可补强。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚、表格扎实，对 activation hurt 的分析有洞察；部分附录配置较多。
- 价值: ⭐⭐⭐⭐☆ 对视觉 backbone、隐私计算友好网络和 multiplicative architecture 设计都有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](../../CVPR2025/segmentation/style-editor_text-driven_object-centric_style_editing.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2025\] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference](../../CVPR2025/segmentation/resclip_residual_attention_for_training-free_dense_vision-language_inference.md)
- [\[ICML 2026\] What Makes Synthetic Data Effective in Image Segmentation](what_makes_synthetic_data_effective_in_image_segmentation.md)
- [\[AAAI 2026\] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](../../AAAI2026/segmentation/causal-tune_mining_causal_factors_from_vision_foundation_mod.md)

</div>

<!-- RELATED:END -->
