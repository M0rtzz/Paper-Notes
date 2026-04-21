---
title: >-
  [论文解读] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation
description: >-
  [CVPR 2026][测试时自适应] 本文提出 AcTTA，一种基于动态激活函数调制的测试时自适应框架，通过将传统固定激活函数重参数化为可学习形式（包含激活中心偏移和非对称梯度斜率），在推理时自适应调整激活行为以应对分布偏移，在 CIFAR10-C/CIFAR100-C/ImageNet-C 上一致超越基于归一化层的 TTA 方法。
tags:
  - CVPR 2026
  - 测试时自适应
  - 激活函数
  - 分布偏移
  - 归一化层
  - 动态激活
---

# AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation

**会议**: CVPR 2026  
**arXiv**: [2603.26096](https://arxiv.org/abs/2603.26096)  
**代码**: [https://hyeongyu-kim.github.io/actta/](https://hyeongyu-kim.github.io/actta/)  
**领域**: 信号与通信 / 测试时自适应  
**关键词**: 测试时自适应, 激活函数, 分布偏移, 归一化层, 动态激活

## 一句话总结
本文提出 AcTTA，一种基于动态激活函数调制的测试时自适应框架，通过将传统固定激活函数重参数化为可学习形式（包含激活中心偏移和非对称梯度斜率），在推理时自适应调整激活行为以应对分布偏移，在 CIFAR10-C/CIFAR100-C/ImageNet-C 上一致超越基于归一化层的 TTA 方法。

## 研究背景与动机

1. **领域现状**：测试时自适应（TTA）是应对部署环境与训练分布不一致的重要范式。现有 TTA 方法主要集中在归一化层的仿射参数调整和运行统计量的重校准，如 TENT、EATA、SAR 等方法均以归一化层作为主要适应机制。

2. **现有痛点**：这种以归一化层为中心的视角忽略了一个关键组件——激活函数。激活函数作为非线性核心，根本性地塑造了特征空间的几何结构，决定了模型如何响应输入变化。然而在 TTA 中，激活函数一直被作为固定的非线性映射，从未被纳入自适应的范畴。

3. **核心矛盾**：在分布偏移下，BN 层的源域统计量不再与目标特征对齐，产生有偏的特征表示。当这些有偏特征经过以零为中心的激活函数（如 ReLU、GELU）时，有用的信号可能被抑制在激活边界以下，导致信息丢失和梯度消失。这种"零中心刚性"是限制自适应效果的关键因素。

4. **本文目标** 如何让激活函数本身成为 TTA 中的可适应组件：(1) 调整梯度行为以维持学习流；(2) 偏移激活边界以对齐新的特征中心；(3) 保持与源域预训练表示的兼容性。

5. **切入角度**：作者观察到在 TTA 之外，可学习/可调制的激活函数（如 PReLU、ACON、PAU）已证明即使细微的激活行为修改也能带来性能和训练稳定性的改善。这说明激活函数本身具有可学习的灵活性。

6. **核心 idea**：将激活函数从固定组件变为可适应的参与者——通过参数化激活中心和非对称斜率，让网络在推理时自我纠正内部偏差。

## 方法详解

### 整体框架
AcTTA 是一个模块化、目标无关的激活自适应框架。输入为预训练好的模型和目标域测试数据，AcTTA 在每个激活函数位置插入一个可学习的动态激活模块，仅更新激活参数（$\lambda_{pos}$, $\lambda_{neg}$, $c$），不修改网络权重或需要源域数据。该框架可以与任意现有 TTA 目标函数（如熵最小化、一致性正则化等）无缝结合。

### 关键设计

1. **动态激活重参数化（Dynamic Activation Reparameterization）**:

    - 功能：将固定激活函数转化为可在推理时调整的参数化形式
    - 核心思路：现代激活函数可近似为 $\phi(x) = x \cdot \sigma(\beta x)$，其导数是一个输入相关的斜率函数。AcTTA 显式暴露这个斜率为可学习函数 $\lambda(x) = \lambda_{neg} + (\lambda_{pos} - \lambda_{neg}) \sigma(\beta x)$，其中 $\lambda_{neg}$ 和 $\lambda_{pos}$ 分别控制负/正区域的渐近斜率。同时引入可学习中心参数 $c$ 来重新定位激活边界。最终激活为：$g(x) = \phi(x-c) + [\lambda_{neg} + (\lambda_{pos} - \lambda_{neg})\sigma(\beta(x-c))](x-c)$
    - 设计动机：单独的斜率适应无法纠正特征偏差；而引入中心偏移 $c$ 可以让激活函数根据目标域统计量动态重新定心。当初始化 $\lambda_{neg}=\lambda_{pos}=0$, $c=0$ 时，$g(x)$ 完全恢复原始 $\phi(x)$，确保与预训练模型兼容

2. **非对称梯度调制（Asymmetric Gradient Modulation）**:

    - 功能：在正负区域分别控制梯度缩放，使梯度传播更稳定
    - 核心思路：通过分别学习 $\lambda_{pos}$ 和 $\lambda_{neg}$，让激活函数在负区域保持非零梯度（避免死梯度问题），在正区域灵活调整响应强度。这使得即使在分布偏移下特征分布偏斜，网络也能保持有效的梯度流
    - 设计动机：传统以零为中心的激活函数在分布偏移下会导致梯度不平衡或有偏更新。AcTTA 的非对称设计允许模型在10倍于传统方法的学习率下仍保持稳定优化

3. **架构自适应的可训练参数选择**:

    - 功能：根据不同骨干网络架构选择最优的可训练参数组合
    - 核心思路：对于基于 BN 的 CNN（如 WRN），冻结 BN 仿射参数 $(\gamma, \beta)$，仅适应激活参数 $(\lambda_{pos}, \lambda_{neg}, c)$，效果最佳。对于基于 LN 的 ViT，联合适应归一化和激活参数效果最佳
    - 设计动机：BN 依赖于受扰动的运行统计量，进一步修改 $(\gamma, \beta)$ 可能放大分布噪声；而 LN 按样本归一化不依赖源域统计量，其 $(\gamma, \beta)$ 提供了与激活参数互补的自由度

### 损失函数 / 训练策略
AcTTA 是目标无关的——它不定义自己的损失函数，而是将激活参数加入任意现有 TTA 方法的可学习参数集中。实践中使用约10倍于传统方法的学习率（如 $10^{-2}$ vs $10^{-3}$），因为非对称梯度设计保证了稳定性。默认深度配置为 50%（即前半部分层的激活函数可学习）。

## 实验关键数据

### 主实验

| 数据集/骨干 | 指标(Err%) | AcTTA_TENT | TENT | 提升 |
|-------------|-----------|------------|------|------|
| CIFAR10-C / WRN-28 | Error | 17.03 | 18.51 | -1.48 |
| CIFAR10-C / ResNeXt | Error | 9.53 | 10.28 | -0.75 |
| CIFAR100-C / WRN-40 | Error | 33.81 | 35.25 | -1.44 |
| ImageNet-C / ResNet50(BN) | Error | 64.95 | 66.50 | -1.55 |
| ImageNet-C / ResNet50(GN) | Error | 66.84 | 69.60 | -2.76 |
| ImageNet-C / ViT-B/16 | Error | 51.79 | 53.85 | -2.06 |

AcTTA 与其他 TTA 基线（ETA、SAR、DeYO、ROID、CMF）的组合也一致带来改进，展示了出色的模块化兼容性。

### 消融实验

| 配置 | WRN-28 Err% | ViT-B/16 Err% | 说明 |
|------|------------|--------------|------|
| TENT (仅 γ,β) | 18.51 | 53.85 | 基线 |
| AcTTA (γ,β,λ+,λ-,c) | 18.06 | 52.37 | 全参数 |
| AcTTA* (仅 λ+,λ-,c) | 17.03 | 55.30 | 冻结BN，CNN最优 |
| AcTTA* (仅 c) | 17.50 | 61.56 | 仅中心偏移 |
| 无适应 | 43.52 | 62.10 | 原始模型 |

### 关键发现
- **BN-CNN 上激活参数 > 归一化参数**：冻结 BN 仿射参数仅适应激活参数（AcTTA*）在 WRN-28 上达到最低错误率 17.03%，说明 BN 和激活适应存在重叠角色
- **中心偏移 $c$ 贡献显著**：仅适应 $c$ 就能在 CNN 上带来明显提升（18.51→17.50），说明源域运行统计量引起的残余偏差可通过调整激活边界来补偿
- **大学习率下的稳定性**：AcTTA 在10倍学习率下仍表现稳定（CIFAR100-C 上 34.56% @ LR=1e-2），而 TENT 在同等条件下完全崩溃（51.57%）
- **与其他可学习激活函数对比**：PReLU 和 PAU 在 TTA 场景下效果不佳（PAU 在 ViT 上错误率达 99.96%），说明 TTA 需要的不仅是参数化斜率，而是联合的中心偏移和非对称斜率调制
- **最优适应深度与架构相关**：WRN 最优在 ~50%，ViT 在 ~25%，ResNet 在 ~75%

## 亮点与洞察
- **激活函数作为 TTA 的新维度**：这是首次系统性地将激活函数纳入 TTA 框架，拓展了以归一化为中心的传统视角。这个思路可以迁移到其他需要在线适应的场景（如持续学习、域泛化）
- **初始化兼容性设计**：$\lambda=0, c=0$ 时完全恢复原始激活函数的设计非常巧妙——保证了零风险无损的开始。这种"加法式"模块设计是一个可复用的 trick
- **大学习率的稳定性**：通过保持负区域非零梯度，AcTTA 本质上解决了分布偏移下的梯度消失问题，使得更激进的学习率成为可能。这个发现对实时 TTA 部署意义重大

## 局限与展望
- **最优深度需要先验知识**：不同架构的最优适应深度不同（10%~75%），论文采用50%作为折中，但这不一定最优
- **在小 batch 场景下效果受限**：batch=4 时，部分组合（如 AcTTA_SAR 在 ViT 上）反而比基线差，说明激活适应对批统计也有一定依赖
- **仅在corruption benchmark上验证**：未涉及更复杂的领域偏移场景（如自然域偏移、跨模态偏移）
- **计算开销未详细分析**：增加的可学习参数数量、额外前向/反向计算时间未量化比较

## 相关工作与启发
- **vs TENT**: TENT 仅适应 BN 层 $(\gamma, \beta)$。AcTTA 表明在 CNN 上冻结 BN 参数而仅适应激活函数效果更好，说明不同组件的适应作用存在重叠
- **vs ACON**: ACON 引入了可学习门控但仍假设零中心边界，无法处理偏移的特征分布。AcTTA 的中心偏移设计直接解决了这个问题
- **vs PAU**: PAU 作为通用可学习激活在训练时有效，但在 TTA 场景下崩溃（ViT上99.96%错误率），说明 TTA 需要的是针对性的偏移-斜率调制而非通用函数逼近

## 评分
- 新颖性: ⭐⭐⭐⭐ 激活函数视角在TTA中是全新的，但重参数化形式相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多数据集、多架构、多TTA基线，消融实验非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从动机到方法到实验层层递进
- 价值: ⭐⭐⭐⭐ 开辟了TTA的新研究方向，但实际收益幅度有限（1-3%错误率降低）

<!-- RELATED:START -->

## 相关论文

- [PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](../../ECCV2024/signal_comm/pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [Continuous Space-Time Video Resampling with Invertible Motion Steganography](../../CVPR2025/signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)
- [Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)

<!-- RELATED:END -->
