---
title: >-
  [论文解读] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality
description: >-
  [CVPR 2025][模型压缩][Vision Mamba] 提出EfficientViM，通过将SSD层中的通道混合操作从token空间（$O(LD^2)$）迁移到压缩的隐藏状态空间（$O(ND^2)$，$N \ll L$），实现了比现有Vision Mamba模型快2-4倍的推理速度，同时保持竞争性精度（ImageNet-1K上M3模型77.9%/11952 img/s）。
tags:
  - CVPR 2025
  - 模型压缩
  - Vision Mamba
  - SSM
  - 轻量模型
  - 隐藏状态混合
  - 高效推理
---

# EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality

**会议**: CVPR 2025  
**arXiv**: [2411.15241](https://arxiv.org/abs/2411.15241)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Vision Mamba、SSM、轻量模型、隐藏状态混合、高效推理

## 一句话总结
提出EfficientViM，通过将SSD层中的通道混合操作从token空间（$O(LD^2)$）迁移到压缩的隐藏状态空间（$O(ND^2)$，$N \ll L$），实现了比现有Vision Mamba模型快2-4倍的推理速度，同时保持竞争性精度（ImageNet-1K上M3模型77.9%/11952 img/s）。

## 研究背景与动机

**领域现状**：状态空间模型（SSMs/Mamba）以线性序列长度复杂度替代注意力的二次复杂度。Vision Mamba（ViM、VSSD等）将Mamba应用于视觉任务，但实际推理速度仍然慢于轻量级CNN/Transformer。

**现有痛点**：标准SSD层的计算瓶颈不在序列扫描本身（$O(LND)$），而在对全长序列的线性投影操作——生成$\mathbf{x}$、门控$\mathbf{z}$、输出投影等操作的复杂度为$O(LD^2)$。虽然SSD内部使用了压缩的隐藏状态（$N \ll L$），但通道混合和门控仍在完整的token空间进行。

**核心矛盾**：SSM的核心优势在于通过压缩隐藏状态（N维）来处理长序列（L维），但现有实现中最耗时的操作并未利用这种压缩。

**本文目标** 如何让Vision Mamba在实践中真正达到理论预期的高效性。

**切入角度**：将线性投影和门控操作从token空间搬到隐藏状态空间，使得主导项从$LD^2$变为$ND^2$。

**核心 idea**：将SSD中最耗时的通道混合操作从L维token空间移到N维隐藏状态空间（$N \ll L$），从而大幅加速Vision Mamba推理。

## 方法详解

### 整体框架
EfficientViM为三阶段层次化架构，stem为4层stride-2卷积，每阶段由HSM-SSD + FFN块组成，阶段间通过下采样。推理时利用压缩隐藏状态空间进行通道混合，结合多阶段隐藏状态融合（MSF）增强表征。

### 关键设计

1. **隐藏状态混合SSD（HSM-SSD）**:

    - 功能：将通道混合操作从token空间转移到隐藏状态空间
    - 核心思路：标准SSD中输出$\mathbf{x}_{out} = \text{Linear}(\mathbf{y} \odot \sigma(\mathbf{z}))$在$L \times D$空间操作；HSM-SSD先将输入投影到隐藏状态$\mathbf{h}_{in} \in \mathbb{R}^{N \times D}$，在此空间进行门控和投影，再通过C矩阵展开回token空间。主导复杂度从$O(LD^2)$降为$O(ND^2)$
    - 设计动机：隐藏状态是SSM的核心压缩表示（$N$通常远小于$L$），在此空间做通道混合信息损失可控但速度大幅提升

2. **多阶段隐藏状态融合（MSF）**:

    - 功能：利用各阶段的隐藏状态丰富最终分类特征
    - 核心思路：对每个阶段的隐藏状态取平均$\hat{h}^{(s)} = \frac{1}{N}\sum_i h_i^{(s)}$，投影为分类logits $z^{(s)}$，用学习的softmax权重$\hat{\beta}^{(s)}$加权融合各阶段的logits。无推理开销增加（隐藏状态是SSM的副产品）
    - 设计动机：隐藏状态包含全局信息但通常被丢弃；MSF让低层和高层的隐藏状态都参与分类，增强表征的层次性

3. **单头设计+状态重要性权重**:

    - 功能：避免多头注意力中的memory-bound reshape操作
    - 核心思路：使用单头但引入状态级重要性权重$\mathbf{A} \in \mathbb{R}^{L \times N}$，让每个隐藏状态维度对不同token有不同的重要性。这提供了类似多头的多样性捕获能力但避免了内存排布开销
    - 设计动机：多头实现需要频繁的tensor reshape，在轻量模型中成为推理瓶颈

### 损失函数 / 训练策略
标准ImageNet训练配方。300/450 epoch训练。架构设计4个变体（M1-M4），参数从6.7M到19.6M覆盖不同速度-精度需求。

## 实验关键数据

### 主实验（ImageNet-1K）

| 模型 | Throughput (img/s) | Top-1 (%) | Params | FLOPs |
|------|-------------------|-----------|--------|-------|
| EfficientViM-M1 | 20,731 | 72.9% | 6.7M | 239M |
| EfficientViM-M2 | 17,005 | 75.8% | 13.9M | 355M |
| EfficientViM-M3 | 11,952 | 77.9% | 16.6M | 656M |
| EfficientViM-M4 | 8,170 | 79.6% | 19.6M | 1111M |

vs Vision Mamba对比：EfficientViM-M3比ViM-Ti (77.2%) 高0.7%且快2.7倍，比VSSD-Nano (74.1%) 高3.8%且快4.0倍。

### 消融实验

| 配置 | Top-1 | Throughput | 说明 |
|------|-------|-----------|------|
| 标准SSD (baseline) | ~75.4% | ~10k | NC-SSD基线 |
| +HSM（隐藏状态混合） | ~75.2% | ~17k | 精度微降但速度+70% |
| +MSF（多阶段融合） | 75.8% | 17,005 | 补回精度且无速度损失 |
| 多头→单头+权重 | +0.3% | +15% | 单头更快且更准 |

### 关键发现
- HSM是速度提升的核心（70%加速），精度损失极小（<0.2%），MSF能完全补回
- EfficientViM在所有模型规模上都取得了最佳速度-精度权衡
- 比MobileNetV3-L 0.75快90%且精度接近；比MobileViTV2 0.75快4倍且精度高0.2%
- 隐藏状态N=16-64已足够捕获全局信息，进一步增大N收益递减

## 亮点与洞察
- **精准定位瓶颈**：发现SSM的实际瓶颈不在序列扫描而在通道投影，这个洞察是方法有效性的根基
- **隐藏状态的双重利用**：HSM用隐藏状态做高效通道混合，MSF用隐藏状态做辅助分类特征——充分挖掘了SSM的"副产品"价值
- **设计哲学的实用性**：不追求理论新颖性，而是通过精细的工程优化（单头、隐藏状态空间操作）让Mamba在移动端真正可用

## 局限与展望
- HSM的近似可能在需要精细token级信息的任务（如密集预测）上精度损失更大
- 仅在分类任务上验证，检测/分割等下游任务未报告
- 隐藏状态维度N的选择仍需手动调节

## 相关工作与启发
- **vs ViM/PlainMamba**: 这些方法直接将Mamba应用于视觉但未优化实际推理效率；EfficientViM通过重新设计计算路径实现了实质性加速
- **vs SHViT**: SHViT用单头注意力追求效率，EfficientViM在同等速度下精度更高（M1: 72.9% vs 72.8%）
- **vs EfficientNet/MobileNet**: 传统轻量CNN在小模型上仍有速度优势，但EfficientViM在中等规模上开始反超

## 评分
- 新颖性: ⭐⭐⭐⭐ HSM的思路简洁有效，"搬运操作空间"的想法有启发性
- 实验充分度: ⭐⭐⭐⭐ ImageNet全面对比，消融详细，但缺下游任务
- 写作质量: ⭐⭐⭐⭐ 瓶颈分析清晰，方法推导易懂
- 价值: ⭐⭐⭐⭐ 让Vision Mamba在轻量部署中真正可用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](../../ICCV2025/model_compression/ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)

<!-- RELATED:END -->
