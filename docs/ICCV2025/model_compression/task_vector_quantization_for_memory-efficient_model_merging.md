---
title: >-
  [论文解读] Task Vector Quantization for Memory-Efficient Model Merging
description: >-
  [ICCV 2025][模型压缩][模型合并] 本文提出对任务向量（fine-tuned 与 pre-trained 权重之差）而非 fine-tuned 权重本身进行量化，利用任务向量更窄的数值范围实现低至 3-bit 的量化而不损失精度；进一步提出残差任务向量量化（RTVQ），将任务向量分解为共享高精度基向量和低精度偏移量，在仅用 8% 原始存储的情况下维持甚至提升模型合并性能。
tags:
  - ICCV 2025
  - 模型压缩
  - 模型合并
  - 任务向量量化
  - 低精度存储
  - 残差量化
  - 多任务学习
---

# Task Vector Quantization for Memory-Efficient Model Merging

**会议**: ICCV 2025  
**arXiv**: [2503.06921](https://arxiv.org/abs/2503.06921)  
**代码**: [https://aim-skku.github.io/TVQ/](https://aim-skku.github.io/TVQ/)  
**领域**: 模型压缩  
**关键词**: 模型合并, 任务向量量化, 低精度存储, 残差量化, 多任务学习

## 一句话总结
本文提出对任务向量（fine-tuned 与 pre-trained 权重之差）而非 fine-tuned 权重本身进行量化，利用任务向量更窄的数值范围实现低至 3-bit 的量化而不损失精度；进一步提出残差任务向量量化（RTVQ），将任务向量分解为共享高精度基向量和低精度偏移量，在仅用 8% 原始存储的情况下维持甚至提升模型合并性能。

## 研究背景与动机
模型合并（Model Merging）通过组合多个单任务微调模型来构建高效的多任务模型，避免了集成方法需要运行多个模型的开销。Task Arithmetic 方法定义任务向量 $\tau_t = \theta_{ft}^t - \theta_{pre}$，通过线性组合实现多任务合并。

然而，存储多个微调后的检查点需要巨大的存储空间。例如：ViT-L/14 单个检查点 1.14GB，20 个任务总共 22.8GB。在 NVIDIA Jetson Nano 等边缘设备（仅 16GB 存储）上，这种存储开销成为了扩展到大模型和更多任务的瓶颈。

朴素的做法是直接量化微调后的权重，但作者观察到一个关键现象：**任务向量的数值范围比微调后权重的范围小一个数量级**。由于量化误差的上界为 $|\epsilon| \leq \frac{\theta_{max} - \theta_{min}}{2(2^b - 1)}$，更窄的数值范围意味着相同比特下更小的量化误差。

核心 idea：量化任务向量而非完整权重，利用其天然的窄动态范围优势。

## 方法详解

### 整体框架
系统包含两个量化方案：(1) TVQ——直接量化任务向量 $\tau_t$；(2) RTVQ——将任务向量分解为共享基向量和任务特定偏移量，分别量化。两者均为后训练量化（PTQ），不修改任何合并方法本身，仅操作检查点存储。

### 关键设计
1. **Task Vector Quantization (TVQ)**

    - 功能：将存储的对象从微调权重 $\theta_{ft}^t$ 替换为任务向量 $\tau_t = \theta_{ft}^t - \theta_{pre}$，然后量化 $\tau_t$
    - 核心思路：利用非对称量化 $\tau^q = \text{Round}(\tau / \Delta) + z$，其中 $\Delta = \frac{\tau_{max} - \tau_{min}}{2^b - 1}$。由于 $\tau$ 的动态范围远小于 $\theta_{ft}$，相同比特下 $\Delta$ 更小，量化误差更低
    - 设计动机：实验验证了任务向量的 L2 量化误差在各精度下都显著低于直接量化微调权重，特别是在 4-bit 以下差距更为明显
    - 关键优势：方法仅修改检查点存储格式，可无缝集成到所有现有的任务向量合并框架中

2. **Residual Task Vector Quantization (RTVQ)**

    - 功能：解决 TVQ 在超低精度（2-bit）下性能急剧下降的问题
    - 核心思路：将任务向量分解为两部分：
    $\tau_t = \underbrace{(\theta_{ft}^t - \theta_{ft\_avg})}_{\text{Offset Vector}} + \underbrace{(\theta_{ft\_avg} - \theta_{pre})}_{\text{Base Vector}}$
      其中 $\theta_{ft\_avg} = \frac{1}{T}\sum_t \theta_{ft}^t$ 是所有微调权重的均值。基向量使用较高精度（如 3-bit），偏移量使用超低精度（如 2-bit），等效精度约为 $2 + 4/8 = 2.5$ bits/task
    - **量化误差修正**：先量化基向量得到 $\theta_{ft\_avg\_ec} = Q(\theta_{ft\_avg} - \theta_{pre}) + \theta_{pre}$，再用修正后的参考计算偏移量，减少累积误差
    - 设计动机：基向量全任务共享，开销分摊后可忽略；偏移量因为是与均值的差异，幅度更小，适合低精度量化

3. **量化正则化效应**

    - 功能：论文发现 3-bit 量化有时反而提升了合并性能
    - 核心思路：量化引入的噪声起到了正则化作用，减少了过拟合，类似于 dropout 的效果
    - 在 Task Arithmetic、Ties Merging 等方法上均观察到 3-bit TVQ 性能超过 FP32 的现象

### 损失函数 / 训练策略
- 无额外训练过程，属于纯 PTQ 方案
- 量化操作在模型合并前的检查点阶段离线完成
- 支持逐层量化，每层独立计算 scale 和 zero-point

## 实验关键数据

### 主实验
8 任务图像分类（ViT-B/32）：

| 方法 | FP32 | FQ-INT4 | TVQ-INT4 | TVQ-INT3 | TVQ-INT2 | RTVQ(3+2) |
|------|------|---------|----------|----------|----------|-----------|
| Task Arithmetic | 69.2 | 4.2 | 69.1 | **71.2** | 62.1 | 70.2 |
| AdaMerging | 81.8 | 4.5 | 81.5 | **82.0** | 78.1 | **82.8** |
| EMR-Merging | 88.3 | 3.9 | **89.8** | **90.0** | 77.2 | 83.2 |

8 任务图像分类（ViT-L/14）：

| 方法 | FP32 | TVQ-INT4 | TVQ-INT3 | RTVQ(3+2) |
|------|------|----------|----------|-----------|
| Task Arithmetic | 84.3 | 84.4 | **84.8** | **84.8** |
| AdaMerging | 90.8 | 90.9 | **91.0** | **90.9** |

### 消融实验
密集预测任务（NYUv2，ResNet-50）：

| 方法 | Seg(mIoU)↑ | Depth(RelErr)↓ | Normal(MAE)↓ |
|------|-----------|----------------|--------------|
| Task Arith. FP32 | 31.6 | 24.0 | 30.6 |
| Task Arith. TVQ-4 | 31.5 | 24.0 | 30.6 |
| Task Arith. TVQ-2 | 36.4 | 26.2 | 36.1 |
| Task Arith. RTVQ | 36.1 | 24.6 | 32.6 |

### 关键发现
- **FQ-INT4 完全失败**：直接量化微调权重到 4-bit 后精度降至约 4%（随机水平），证明了任务向量量化的必要性
- **TVQ-INT3 常优于 FP32**：量化噪声的正则化效应在多个合并方法上一致出现
- **RTVQ 有效缓解 2-bit 退化**：在 TVQ-INT2 性能显著下降时，RTVQ（等效 2.375 bit）仍保持接近 FP32 性能
- **任务数增加时 RTVQ 更有优势**：20 任务时等效仅需 2.15 bit/task，且基向量的分摊效应更强
- 存储仅需原始 FP32 检查点的 8%

## 亮点与洞察
- **极简而有效的观察**：任务向量范围窄于微调权重这一简单观察，支撑了整个方法的有效性
- **与现有方法的完美兼容**：纯检查点侧操作，零修改成本集成到 Task Arithmetic、Ties Merging、AdaMerging 等所有方法中
- **残差分解的巧妙设计**：共享基向量 + 任务特定偏移量的分解自然地利用了多任务间的冗余
- **量化正则化的意外发现**：3-bit 量化反而提升性能的现象具有理论研究价值

## 局限与展望
- RTVQ 的基向量依赖所有任务的平均，新增任务时需要重新计算
- 密集预测任务对量化更敏感，尤其是 EMR-Merging 等依赖任务特定掩码的方法
- 仅验证了 PTQ 范式，未探索 QAT（量化感知训练）是否能进一步改善
- 对 NLP 任务的验证较少，主要集中在视觉领域

## 相关工作与启发
- 与 TALL Mask 和 TSV-C 等存储优化方法不同，TVQ 更通用且不需要额外的任务特定结构
- 量化正则化效应的发现可能启发新的模型合并正则化方法
- 残差分解的思路可扩展到其他需要存储多个变体的场景（如多 LoRA 适配器存储）

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心观察简单但深刻，RTVQ 的设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分类、密集预测、NLP，8/14/20 任务规模，多种合并方法
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表说服力强
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，8% 存储即保持性能，对资源受限部署意义重大

<!-- RELATED:START -->

## 相关论文

- [Task Singular Vectors: Reducing Task Interference in Model Merging](../../CVPR2025/model_compression/task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [MSQ: Memory-Efficient Bit Sparsification Quantization](msq_memory-efficient_bit_sparsification_quantization.md)
- [SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [Accurate and Efficient Low-Rank Model Merging in Core Space](../../NeurIPS2025/model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)

<!-- RELATED:END -->
