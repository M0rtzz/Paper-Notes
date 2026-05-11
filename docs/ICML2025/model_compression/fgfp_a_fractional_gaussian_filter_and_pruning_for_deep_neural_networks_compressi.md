---
title: >-
  [论文解读] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression
description: >-
  [ICML 2025][模型压缩][网络压缩] 提出 FGFP 框架，将分数阶微积分与高斯函数结合构建分数阶高斯滤波器（FGF），每个卷积核仅需 7 个参数，配合自适应非结构化剪枝（AUP），在 CIFAR-10 上 ResNet-20 精度仅降 1.52% 即实现 85.2% 的模型压缩率，在 ImageNet 上 ResNet-50 精度降 1.63% 实现 69.1% 压缩率。
tags:
  - ICML 2025
  - 模型压缩
  - 网络压缩
  - 分数阶高斯滤波
  - 非结构化剪枝
  - CNN
  - 边缘部署
---

# FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression

**会议**: ICML 2025  
**arXiv**: [2507.22527](https://arxiv.org/abs/2507.22527)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: 网络压缩, 分数阶高斯滤波, 非结构化剪枝, CNN, 边缘部署

## 一句话总结

提出 FGFP 框架，将分数阶微积分与高斯函数结合构建分数阶高斯滤波器（FGF），每个卷积核仅需 7 个参数，配合自适应非结构化剪枝（AUP），在 CIFAR-10 上 ResNet-20 精度仅降 1.52% 即实现 85.2% 的模型压缩率，在 ImageNet 上 ResNet-50 精度降 1.63% 实现 69.1% 压缩率。

## 研究背景与动机

### 1. 边缘部署的挑战

深度神经网络在图像分类、检测等任务上性能卓越，但巨大的参数量和计算量使其难以部署在手机、嵌入式系统等资源受限设备上。

### 2. 现有压缩方法的不足

- **非结构化剪枝**：保持精度但计算复杂度仍高（稀疏矩阵运算效率受限）
- **结构化剪枝**：直接移除通道/卷积核，计算高效但常导致显著精度损失
- **低秩分解**：将卷积核分解为低秩形式，但分解后的结构可能不够灵活

### 3. 核心灵感

传统计算机视觉中的高斯滤波器、拉普拉斯滤波器等经典滤波器，可以通过分数阶微分扩展为更灵活的形式。将这些经典滤波器与 CNN 结合，用极少参数（7 个/核）表示卷积核，实现极高压缩率。

## 方法详解

### 整体框架

1. 选择预训练模型中最大的层，将其卷积权重转换为 FGF 表示
2. 逐层重复此过程，直到所有选定层都被转换
3. 对剩余未转换层执行自适应非结构化剪枝（AUP）
4. 最终得到稀疏的 FGF 模型

### 关键设计

#### 1. 分数阶高斯滤波器（FGF）

- **核心思路**：用 Grünwald-Letnikov 分数阶导数近似分数阶微分方程，将其应用于高斯函数
- **参数效率**：每个 3×3 卷积核仅需 7 个可学习参数（分数阶 α、高斯参数 σ 等）
- **两种变体**：
    - **CA-FGF（通道注意力 FGF）**：跨通道共享滤波参数 + 通道注意力机制补偿精度
    - **3D-FGF（三维 FGF）**：在空间和通道三个维度同时应用分数阶高斯

#### 2. Grünwald-Letnikov 近似

- 将分数阶微分方程简化为三项多项式，大幅降低计算复杂度
- 使得 FGF 的前向和反向传播都高效可行

#### 3. 自适应非结构化剪枝（AUP）

- 对 FGF 转换后剩余的密集层做非结构化剪枝
- 自适应确定每层的剪枝率，最大化整体压缩率

## 实验关键数据

### 主实验：CIFAR-10 和 ImageNet

| 数据集 | 模型 | 方法 | 精度降幅 | 压缩率 |
|--------|------|------|---------|--------|
| CIFAR-10 | ResNet-20 | FGFP | 1.52% | 85.2% |
| CIFAR-10 | ResNet-20 | 结构化剪枝 SOTA | ~3% | 50-60% |
| CIFAR-10 | ResNet-20 | 低秩分解 | ~2% | 60-70% |
| ImageNet | ResNet-50 | FGFP | 1.63% | 69.1% |
| ImageNet | ResNet-50 | 混合压缩 SOTA | ~2% | 50-60% |

### 消融实验

| 配置 | CIFAR-10 精度降幅 | 压缩率 | 说明 |
|------|-----------------|--------|------|
| CA-FGF only | ~1.8% | 75% | 仅滤波替换 |
| 3D-FGF only | ~2.0% | 70% | 三维变体 |
| FGF + AUP (FGFP) | 1.52% | 85.2% | 完整框架 |
| AUP only | ~3.5% | 60% | 仅剪枝 |

### 关键发现

- FGF 的 7 参数/核表示能力足以保持高精度——经典滤波器的归纳偏置有效
- CA-FGF 通过通道注意力补偿了跨通道共享的精度损失
- AUP 与 FGF 互补：FGF 压缩大层，AUP 处理小层

## 亮点与洞察

- **经典 CV 与深度学习的优雅结合**：将传统高斯滤波器通过分数阶微积分扩展，作为 CNN 卷积核的高效参数化
- **极致参数效率**：7 参数/核是目前已知最低的卷积核参数化方案之一
- **数学基础扎实**：基于 Grünwald-Letnikov 分数阶导数理论，有清晰的数学推导
- **混合策略的实用性**：FGF + AUP 的组合覆盖了不同类型的层

## 局限与展望

- 缓存截断在方法部分后段，完整消融数据未获取
- 仅在 CNN（ResNet）上验证，对 Transformer 架构的适用性未探讨
- 非结构化剪枝在实际硬件上的加速效果取决于稀疏计算支持
- FGF 的 7 参数约束可能对某些复杂特征不够灵活
- 可探索将分数阶思路扩展到注意力机制的压缩

## 相关工作与启发

- **vs 结构化剪枝**：直接删除通道，简单但精度损失大；FGFP 用参数化替代实现更高压缩率
- **vs 低秩分解**：SVD 等方法分解权重矩阵，FGFP 用物理启发的滤波器参数化
- **vs Zamora et al. (2021)**：首次将分数阶滤波器与 CNN 结合，FGFP 进一步优化并加入 AUP

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 分数阶微积分+高斯滤波的 CNN 压缩方案高度原创
- 实验充分度: ⭐⭐⭐⭐ CIFAR-10 和 ImageNet 验证充分，但缓存截断使部分细节受限
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，框架图直观
- 价值: ⭐⭐⭐⭐ 对边缘部署的 CNN 压缩有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for DNN Compression](fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)

</div>

<!-- RELATED:END -->
