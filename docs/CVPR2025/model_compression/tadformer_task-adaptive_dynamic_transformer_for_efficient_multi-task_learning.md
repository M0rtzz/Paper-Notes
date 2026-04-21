---
title: >-
  [论文解读] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning
description: >-
  [CVPR 2025][模型压缩][参数高效微调] TADFormer 提出一种面向多任务学习的参数高效微调框架，通过动态任务滤波器（DTF）根据输入上下文动态提取细粒度任务特征，结合任务提示条件操作和跨任务交互，在 PASCAL-Context 上以少于全微调 8.4 倍的参数量实现更高精度。
tags:
  - CVPR 2025
  - 模型压缩
  - 参数高效微调
  - 多任务学习
  - 动态卷积
  - 任务提示
  - Transformer
---

# TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning

**会议**: CVPR 2025  
**arXiv**: [2501.04293](https://arxiv.org/abs/2501.04293)  
**代码**: 无  
**领域**: 模型压缩 / 多任务学习  
**关键词**: 参数高效微调, 多任务学习, 动态卷积, 任务提示, Transformer

## 一句话总结
TADFormer 提出一种面向多任务学习的参数高效微调框架，通过动态任务滤波器（DTF）根据输入上下文动态提取细粒度任务特征，结合任务提示条件操作和跨任务交互，在 PASCAL-Context 上以少于全微调 8.4 倍的参数量实现更高精度。

## 研究背景与动机

**领域现状**：随着预训练模型规模持续增长，参数高效微调（PEFT）成为适配下游任务的主流方案。LoRA、Adapter、Visual Prompt Tuning 等方法已在单任务场景中证明了有效性。多任务学习（MTL）需要同时处理多个任务（语义分割、深度估计、法线估计等），训练复杂度随任务数线性增长，更需要 PEFT。

**现有痛点**：现有 MTL 的 PEFT 方法（如 MTLoRA、VMT-Adapter）存在两个关键局限：(1) 使用静态可学习参数提取任务特征，不考虑输入样本的上下文信息，限制了细粒度任务特征的捕获能力；(2) 任务共享和任务特定模块并行处理，缺乏跨任务交互的机会。

**核心矛盾**：PEFT 只微调少量参数，模型对多样化任务的适应能力本就受限。如果不考虑输入依赖性，仅靠静态参数很难用极少的可训练模块有效区分不同任务的独特特征。

**本文目标**：设计一个既参数高效又能动态感知输入上下文的多任务 PEFT 框架，使模型能根据不同输入样本自适应地提取任务特定特征。

**切入角度**：MTL 研究表明输入样本的上下文信息对捕获任务独特特征至关重要。受此启发，作者提出在 PEFT 模块中引入动态卷积，让卷积核参数根据输入特征动态生成，从而实现输入感知的任务适应。

**核心 idea**：在 LoRA 的 down-up 投影之间插入动态任务滤波器（DTF），其卷积参数由输入特征动态生成，结合任务提示条件（TPC）操作提取任务适应特征，以最少额外参数实现细粒度的输入感知任务适应。

## 方法详解

### 整体框架
TADFormer 由共享编码器（Swin Transformer）和多个任务特定解码器组成。输入为图像 patch token 前置任务提示（task prompts）。编码器中每个 Transformer stage 的前 $N-1$ 个 block 使用任务共享模块（TS-Module，即 LoRA），最后一个 block 使用任务适应模块（TA-Module，包含 DTF），通过任务提示条件操作（TPC）提取任务适应特征后送入 DTF 进行输入感知的精细调整。

### 关键设计

1. **任务提示条件操作（Task-Prompt Conditional Operator, TPC）**:

    - 功能：从任务无关特征中解耦出任务特定特征，为每个任务生成增强了任务属性的特征表示。
    - 核心思路：利用 MHSA 中自然产生的注意力矩阵 $A$，提取任务提示 $p_i$ 与所有 patch token 之间的注意力得分 $a_i \in \mathbb{R}^{H \times 1 \times N}$（即任务注意力图），用其对 QKV 输出特征进行加权：$f_i = f_{qkv} + S_{inv}(a_i \otimes \hat{f}_{qkv})$。这样高注意力区域（与该任务更相关的区域）的特征被增强。
    - 设计动机：不引入额外计算，直接复用 Transformer 自注意力中的 prompt-to-patch 注意力，既自然获得任务-空间关系，又不增加参数或计算量。

2. **动态任务滤波器（Dynamic Task Filter, DTF）**:

    - 功能：根据输入上下文动态生成卷积参数，实现输入感知的任务特征提取。
    - 核心思路：在 LoRA 的 down-projection 和 up-projection 之间插入 DTF。down-projection 后的低秩特征先经过全局平均池化（GAP），再送入参数生成网络 $\phi(\cdot)$ 生成通道级卷积核 $\theta_i = \phi(f_i W_{down})$。卷积核对 down-projection 特征进行通道级卷积操作：$\tilde{F}_i = \Phi(f_i) + (\theta_i \odot (f_i W_{down})) W_{up}$。参数量仅为 $r \times r \times k^2$，非常轻量。使用 FilterNorm 稳定训练。
    - 设计动机：静态 LoRA 参数对所有输入使用相同的投影，无法区分不同样本的任务特征。DTF 让每个样本拥有独特的卷积参数，使细粒度特征提取成为可能。Grad-CAM 可视化证明 DTF 确实能捕获更精细的输入上下文。

3. **阶段门控与跳跃连接（Stage-wise Gating and Skip Connection）**:

    - 功能：将 TPC 操作提取的任务适应特征直接传递给解码器，增强任务提示的有效性。
    - 核心思路：通过可学习门控参数 $g$（Sigmoid 激活，零初始化）加权融合 TPC 特征 $f_i$ 和最终 block 输出 $\hat{f_i}$：$F_i = \sigma(g) \cdot f_i + (1 - \sigma(g)) \cdot \hat{f_i}$，然后送入任务特定解码器。
    - 设计动机：跳跃连接让任务提示通过 TPC 操作直接影响解码器输入，比仅在编码器内部传递更有效。门控参数让模型自动学习两种特征的最优混合比例。

### 损失函数 / 训练策略
多任务加权损失：$L_{MTL} = \sum_i^T w_i \times L_i$。语义分割和人体部位分割用交叉熵，法线估计用 L1 损失，显著性检测用平衡交叉熵。仅微调 LoRA 模块、DTF、任务提示、提示上采样模块、层归一化和位置编码。不微调 patch merging 模块（因为任务提示上采样已足够）。

## 实验关键数据

### 主实验
PASCAL-Context 数据集，Swin-T (ImageNet-1k) 作为编码器骨架：

| 方法 | SemSeg (mIoU↑) | Parts (mIoU↑) | Saliency (mIoU↑) | Normals (rmse↓) | Δm(%)↑ | 参数(M) |
|------|---------------|--------------|-----------------|----------------|--------|--------|
| 单任务全微调 | 67.21 | 61.93 | 62.35 | 17.97 | 0 | 112.62 |
| 多任务全微调 | 67.56 | 60.24 | 65.21 | 16.64 | +2.23 | 30.06 |
| MTLoRA (r=16) | 68.19 | 58.99 | 64.48 | 17.03 | +1.35 | 4.95 |
| MTLoRA (r=32) | 67.74 | 59.46 | 64.90 | 16.59 | +2.16 | 6.08 |
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 | 8.34 |
| **TADFormer (r=16)** | **69.79** | **59.27** | **65.04** | **16.91** | **+2.44** | **3.56** |
| **TADFormer (r=32)** | **70.20** | **60.00** | **65.71** | **16.57** | **+3.63** | **4.78** |
| **TADFormer (r=64)** | **70.82** | **60.45** | **65.88** | **16.48** | **+4.24** | **7.38** |

### 消融实验
| 配置 | SemSeg↑ | Parts↑ | Saliency↑ | Normals↓ | Δm(%)↑ |
|------|---------|--------|----------|---------|--------|
| 基线 (MTLoRA r=32) | 67.74 | 59.46 | 64.90 | 16.59 | +2.16 |
| + TPC | 提升 | 提升 | 提升 | 下降 | 提升 |
| + DTF | 提升 | 提升 | 提升 | 下降 | 提升 |
| + 跳跃连接 | 提升 | 提升 | 提升 | 下降 | 提升 |
| 完整 TADFormer | **70.20** | **60.00** | **65.71** | **16.57** | **+3.63** |

### 关键发现
- TADFormer 在所有 rank 设置下都以更少的参数超越 MTLoRA 约 1.2-1.7% Δm，rank=16 时仅 3.56M 参数即超越 MTLoRA rank=64 的 8.34M 参数
- DTF 是最关键的组件，Grad-CAM 可视化清楚展示了它能捕获更精细的输入上下文
- 冻结 patch merging 仅微调 prompt upsampling 实现了参数-性能的最优权衡
- 语义分割任务上提升最显著（MTLoRA 67.74 vs TADFormer 70.20），说明动态特征提取对像素级任务尤为有效

## 亮点与洞察
- **动态卷积引入 PEFT**：在 LoRA 的静态 down-up 结构间插入动态生成的卷积参数，巧妙地在极少额外参数下实现了输入感知的适应，这一设计可迁移到任何使用 LoRA 的场景
- **任务注意力图的零成本复用**：直接从 MHSA 中提取 prompt-to-patch 的注意力矩阵作为任务注意力图，不引入额外计算，是一个非常优雅的设计
- **仅最后一个 block 使用 TA-Module**：大部分 block 使用简单 LoRA，仅最后一个 block 使用复杂的任务适应模块，在效率和效果之间取得了良好平衡

## 局限与展望
- 仅在 PASCAL-Context 上验证，缺少大规模数据集（如 NYUD-v2、Cityscapes）的实验
- 编码器限于 Swin Transformer，未测试 ViT 等其他架构
- DTF 的动态卷积在推理时引入了少量额外计算开销
- 未探索任务数超过 4 个的场景，跨任务交互的可扩展性有待验证

## 相关工作与启发
- **vs MTLoRA**: MTLoRA 使用静态的任务共享+任务特定 LoRA，不考虑输入上下文。TADFormer 的 DTF 实现了输入依赖的动态适应，在相同或更少参数下性能更优
- **vs VMT-Adapter**: VMT-Adapter 基于 adapter 架构加入跨任务知识共享，但同样不考虑输入上下文。TADFormer 在 Transformer 注意力层面实现了跨任务交互
- **vs HyperFormer**: HyperFormer 用超网络生成 adapter 参数，参数量高达 72.77M。TADFormer 用动态卷积实现类似效果但参数仅 4.78M

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态卷积+PEFT 的结合在 MTL 中是新颖的，TPC 操作的设计很巧妙
- 实验充分度: ⭐⭐⭐ 仅在 PASCAL-Context 上验证，需要更多数据集和骨架的实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，方法对比清楚
- 价值: ⭐⭐⭐⭐ 为 MTL 的 PEFT 提供了新范式，输入感知的动态适应思路有广泛迁移价值

<!-- RELATED:START -->

## 相关论文

- [Task Singular Vectors: Reducing Task Interference in Model Merging](task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)
- [TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](../../ACL2025/model_compression/dac_prompt_compression.md)

<!-- RELATED:END -->
