---
title: >-
  [论文解读] HOT: Hadamard-based Optimized Training
description: >-
  [CVPR 2025][模型压缩][训练加速] 提出HOT方法，通过对反向传播中不同梯度路径（激活梯度$g_x$和权重梯度$g_m$）的差异化灵敏度分析，选择性地应用Hadamard变换+量化——$g_x$用HT+INT4加速计算、$g_m$用HLA+INT8节省激活内存…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "训练加速"
  - "Hadamard变换"
  - "梯度量化"
  - "激活压缩"
  - "混合精度"
---

# HOT: Hadamard-based Optimized Training

**会议**: CVPR 2025  
**arXiv**: [2503.21261](https://arxiv.org/abs/2503.21261)  
**代码**: [https://github.com/sungonuni/HOT](https://github.com/sungonuni/HOT)  
**领域**: 模型压缩  
**关键词**: 训练加速、Hadamard变换、梯度量化、激活压缩、混合精度

## 一句话总结
提出HOT方法，通过对反向传播中不同梯度路径（激活梯度$g_x$和权重梯度$g_m$）的差异化灵敏度分析，选择性地应用Hadamard变换+量化——$g_x$用HT+INT4加速计算、$g_m$用HLA+INT8节省激活内存，实现75%激活内存节省和2.6倍GPU加速，ViT-B在ImageNet上精度仅降0.17%。

## 研究背景与动机

**领域现状**：大规模模型训练面临三重瓶颈——权重/优化器内存大、激活内存大、反向传播计算慢。现有方法各解决一个：LoRA减权重内存、LBP-WHT加速反传、梯度量化减内存，但没有方法同时处理三个。

**现有痛点**：直接对训练梯度做量化会严重损失精度——尤其是反传中的梯度累积步骤对精度极其敏感。现有方法不区分不同梯度路径的灵敏度，"一刀切"的量化策略导致精度或效率的妥协。

**核心矛盾**：训练中两条梯度路径（$g_x$用于激活梯度传播、$g_m$用于权重更新）对量化误差的容忍度完全不同——$g_x$可以跨batch平均降噪因此容忍低精度加速，$g_m$涉及累积更新对精度更敏感——但现有方法不做区分。

**本文目标** 设计差异化的梯度优化策略，同时实现内存节省、计算加速和精度保持。

**切入角度**：对两条梯度路径做差异化的Hadamard变换+量化处理——$g_x$用HT+INT4（加速为主），$g_m$用HLA（Hadamard低秩近似）+INT8（内存为主）。

**核心 idea**：区分反传中两条梯度路径的灵敏度，对$g_x$用INT4+Hadamard加速计算，对$g_m$用HLA+INT8压缩激活存储，以最小精度代价同时节省内存和加速。

## 方法详解

### 整体框架
在标准训练的反向传播中，识别两条梯度路径：$g_x$（激活梯度，用于传播梯度到前一层）和$g_m$（权重梯度，用于更新当前层权重）。对$g_x$：Hadamard变换后INT4量化，利用batch平均降低量化噪声（加速优先）。对$g_m$：HLA（50%秩约简）+INT8存储前向激活（内存优先）。结合LoRA处理冻结权重。

### 关键设计

1. **$g_x$梯度路径：HT + INT4量化**:

    - 功能：加速反向传播中的矩阵乘法
    - 核心思路：先对矩阵做Hadamard变换将信号分散到频域，减少异常值集中，然后INT4量化。$g_x$可以跨batch维度平均，量化噪声在平均中自然降低，因此可以容忍极低精度（INT4）。使用自定义CUDA kernel利用TensorCore的INT4×INT8融合运算
    - 设计动机：$g_x$是"计算密集、精度不敏感"的路径——batch平均的统计效应使量化噪声被自然平滑

2. **$g_m$梯度路径：HLA + INT8激活压缩（ABC）**:

    - 功能：减少前向激活的存储内存
    - 核心思路：HLA（Hadamard Low-rank Approximation）对激活矩阵做50%的秩约简（将$d$维降到$d/2$），再INT8量化存储。前向时存储压缩的激活，反向时解压用于计算$g_m$。内存压缩率：$d/2 \times$INT8 vs $d \times$FP32 = $1/8$（仅12.5%原始内存）
    - 设计动机：$g_m$涉及梯度累积（多步加和），对量化误差的累积效应更敏感。HLA在降秩时保留主要成分，INT8比INT4提供更好的累积精度

3. **逐层量化器选择（LQS）**:

    - 功能：自适应选择每层的量化策略
    - 核心思路：基于梯度异常值模式选择per-token或per-tensor量化。计算每层梯度的MSE异常值比例，超过50%阈值的层用per-token量化（更精细），其余用per-tensor（更快）。ViT中attention和fc2层倾向需要per-token
    - 设计动机：不同层的梯度统计特性不同，统一的量化策略要么过于保守要么过于激进

### 损失函数 / 训练策略
标准训练损失不变，HOT仅影响反传计算过程。与LoRA集成：LoRA处理冻结权重的参数高效微调，HOT处理反传的计算和内存高效化。

## 实验关键数据

### 主实验

| 指标 | HOT | FP32基线 | 节省 |
|------|-----|---------|------|
| 激活内存 | 25% | 100% | **75%** |
| GPU速度 | 2.6× | 1× | **2.6倍加速** |
| ViT-B Top-1 (ImageNet) | 76.29% | 76.46% | 仅降0.17% |
| ViT-B batch size可用 | 1024 | 256 | **4倍** |

### 消融实验

| 配置 | ResNet50 CIFAR-100 | 说明 |
|------|-------------------|------|
| FP32基线 | 76.46% | 完整精度 |
| HT+4bit Q (on $g_x$) | 76.16% | 加速有效 |
| HLA (on $g_m$) | 76.29% | 内存节省有效 |
| Internal-HLA ($g_m$内部) | 76.29% | 最佳$g_m$策略 |

### 关键发现
- $g_x$和$g_m$对量化的灵敏度确实不同：$g_x$用INT4可行但$g_m$用INT4精度崩溃，验证了差异化策略的必要性
- HLA的50%秩约简引入的误差可以被$g_m$的累积效应部分补偿
- LQS发现ViT中约50%的层需要per-token量化（attention/fc2），另50%可用更快的per-tensor
- 与LoRA组合时HOT进一步节省了LoRA未处理的反传开销

## 亮点与洞察
- **差异化梯度路径优化**：认识到$g_x$和$g_m$的灵敏度差异并分别优化，是简洁但深刻的工程洞察
- **激活内存75%压缩**：实现4倍batch size增加，对GPU利用率的提升巨大
- **完整训练栈覆盖**：同时处理权重内存（LoRA）、激活内存（ABC）、计算速度（INT4 TensorCore），是首个全栈解决方案

## 局限与展望
- 需要自定义CUDA kernel，部署门槛较高
- 只在ViT和ResNet上验证，LLM训练未涉及
- HLA的50%秩约简率是固定的，自适应秩选择可能更好

## 相关工作与启发
- **vs LoRA**: LoRA减权重内存但不加速反传；HOT加速反传+减激活内存，两者互补
- **vs LBP-WHT**: LBP-WHT加速反传但不做差异化处理；HOT的$g_x$/$g_m$分治策略更精细
- **vs 混合精度训练（AMP）**: AMP在FP16/FP32间切换；HOT推到INT4/INT8级别压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度路径差异化优化的洞察有价值，Hadamard+量化组合有效
- 实验充分度: ⭐⭐⭐⭐ 内存/速度/精度三维评估完整，消融详细
- 写作质量: ⭐⭐⭐⭐ 灵敏度分析图表直观
- 价值: ⭐⭐⭐⭐ 对大规模模型训练的实际加速有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Style Quantization for Data-Efficient GAN Training](style_quantization_for_data-efficient_gan_training.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)

</div>

<!-- RELATED:END -->
