---
title: >-
  [论文解读] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing
description: >-
  [NeurIPS 2025][模型压缩][脉冲神经网络] 本文发现 SNN 可以在时间维度上天然解构为多个子模型，通过对比各时间步子模型的输出置信度识别"强"与"弱"，提出 Strong2Weak 和 Weak2Strong 两种自蒸馏方案，无需额外教师模型即可显著提升 SNN 性能，尤其在神经形态数据集上提升高达 5.36%。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 脉冲神经网络
  - 自蒸馏
  - 知识蒸馏
  - 时间维度
  - Strong2Weak
---

# Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing

**会议**: NeurIPS 2025  
**arXiv**: [2510.07924](https://arxiv.org/abs/2510.07924)  
**代码**: 无  
**领域**: 模型压缩 / 脉冲神经网络  
**关键词**: 脉冲神经网络, 自蒸馏, 知识蒸馏, 时间维度, Strong2Weak

## 一句话总结

本文发现 SNN 可以在时间维度上天然解构为多个子模型，通过对比各时间步子模型的输出置信度识别"强"与"弱"，提出 Strong2Weak 和 Weak2Strong 两种自蒸馏方案，无需额外教师模型即可显著提升 SNN 性能，尤其在神经形态数据集上提升高达 5.36%。

## 研究背景与动机

脉冲神经网络（SNN）通过二值脉冲传递信息，仅需累加操作而非乘累加操作，功耗极低，是 ANN 的低功耗替代方案。结合神经形态芯片（如天眸芯片仅需 0.7mW），SNN 可以极低延迟和功耗完成视觉任务。

**核心矛盾**：受限于二值信息表示，SNN 与 ANN 仍存在性能差距。知识蒸馏可以提升 SNN 性能，但现有方法的问题是：
1. 依赖外部大教师模型（ANN 或更大 SNN），带来额外的预训练开销
2. TSSD 方法虽然做了自蒸馏，但扩展了训练时间步并增加了弱分类器，训练开销显著增加
3. TKS 依赖真实标签判定输出正确性，无法利用错误输出中的暗知识，效率和性能有限

**关键洞察**：SNN 的时间特性使其天然可以在时间维度上解构为多个子模型——每个时间步的 SNN 实例因初始膜电位和输入电流不同而产生不同输出。这种差异性正好构成了蒸馏学习的天然条件。

## 方法详解

### 整体框架

将运行 $T$ 个时间步的 SNN $f(\theta)$ 解构为 $T$ 个子模型 $\{f(\theta;1), f(\theta;2), \cdots, f(\theta;T)\}$。它们共享同一架构和参数，但因脉冲神经元的膜电位动态不同而产生不同输出。通过评估各子模型的输出置信度识别强弱关系，然后进行自蒸馏。

### 关键设计

1. **强弱识别（Identify the Strong and the Weak）**:

    - 计算每个子模型的 softmax 输出概率分布：$p(t) = \text{softmax}(o(t))$
    - 定义输出置信度为最大概率：$con(t) = \max(p(t))$
    - 置信度最高的子模型为"强"，最低的为"弱"
    - 使用批内样本的置信度均值而非逐样本判定，计算简单且不需要标签信息

2. **Strong2Weak 蒸馏**:

    - 以最强子模型 $t_s$ 为教师，最弱子模型 $t_w$ 为学生
    - 用温度 $\alpha=2$ 软化 logit，通过 KL 散度传递知识
    - $\mathcal{L}_{S2W} = \alpha^2 KL(p(t_s) || p(t_w))$
    - 总损失 $\mathcal{L} = \mathcal{L}_{CE}(O, Y) + \lambda_{S2W} \mathcal{L}_{S2W}$（$\lambda=1$）
    - 核心思想：提升"木桶短板"，让弱子模型向强子模型学习

3. **Weak2Strong 蒸馏**:

    - 反向操作：弱子模型为教师，强子模型为学生
    - $\mathcal{L}_{W2S} = \alpha^2 KL(p(t_w) || p(t_s))$
    - 核心洞察：弱模型可能包含底层暗知识——过度强大的模型可能忽略细节导致过拟合，弱模型提供互补信息或正则化效果
    - "教学相长"——教师在教学中也得到提升

### 灵活实现方式

- **集成教师**：用 $T-1$ 个高/低置信度子模型的平均作为教师
- **集成学生**：一个最强/弱子模型指导其余 $T-1$ 个子模型
- **同步蒸馏**：同时执行 S2W 和 W2S
- **级联蒸馏**：按置信度等级逐级蒸馏

### 损失函数 / 训练策略

- 使用矩形代理梯度进行反向传播（解决脉冲不可微分问题）
- KL 散度为默认蒸馏损失，也支持 MSE 和 Logit Standardization
- 训练时间步 $T$ 与推理时间步相同，不增加额外开销

## 实验关键数据

### 主实验（消融 - 多数据集对比）

| 数据集 | 架构 | Vanilla SNN | Strong2Weak | Weak2Strong |
|--------|------|-------------|-------------|-------------|
| CIFAR10 | VGG-9 | 94.21% | 94.79% (+0.58) | 94.70% (+0.49) |
| CIFAR100 | MS-ResNet18 | 76.33% | 78.25% (+1.92) | 77.98% (+1.65) |
| CIFAR10-DVS | VGG-9 | 73.97% | 78.93% (+4.96) | **79.33% (+5.36)** |
| DVS-Gesture | VGG-9 | 87.85% | **91.43% (+3.58)** | 91.20% (+3.35) |

### ImageNet 对比实验

| 方法 | 架构 | T | 准确率 |
|------|------|---|--------|
| STAA-SNN (CVPR'25) | ResNet34 | 4 | 70.40% |
| MPS (ICLR'25) | SEW-ResNet34 | 4 | 69.03% |
| TKS | SEW-ResNet34 | 4 | 69.60% |
| **Strong2Weak** | SEW-ResNet34 | 4 | **70.53%** |
| Weak2Strong | SEW-ResNet34 | 4 | 69.87% |

### 低时间步推理（CIFAR10-DVS）

| 方法 | T=1 | T=2 | T=3 | T=4 | T=5 |
|------|-----|-----|-----|-----|-----|
| Vanilla SNN | 10.00% | 60.10% | 69.50% | 73.30% | 74.10% |
| MPS | 66.60% | 74.30% | 75.50% | 75.70% | 76.60% |
| Weak2Strong | **73.40%** | **76.50%** | **77.50%** | **78.80%** | **79.70%** |

### 关键发现
- 在神经形态数据集上提升远大于静态图像数据集（CIFAR10-DVS +5.36% vs CIFAR10 +0.58%），因为神经形态数据有丰富时间特征
- Weak2Strong 在多数神经形态任务上优于 Strong2Weak，暗知识的互补效果显著
- 低时间步推理时提升巨大（T=1 从 10% 到 73.4%），可在不重训练的情况下适应不同延迟约束
- 自蒸馏还提升了对抗鲁棒性：FGSM 攻击下准确率从 19.00% 提升到 21.23%

## 亮点与洞察

- **极简而有效的设计**：不添加任何额外模块或教师模型，完全利用 SNN 时间维度的天然属性
- **Weak2Strong 的哲学意义**：弱者教强者居然也有效，说明"暗知识"在蒸馏中的重要性被低估
- **置信度评估代替标签依赖**：不需要标签就能判定强弱，可用于无标签场景
- **低延迟推理能力**：用高时间步训练的模型可以用低时间步推理仍保持不错性能

## 局限与展望

- 同时使用 S2W 和 W2S 并未获得显著更好的效果，过度相似降低了 SNN 的多样性
- 如何平衡子模型间的多样性和相似性是开放问题
- 论文标题声称 SNN "Inherently Superior in Temporal Processing"，但实验证据主要限于分类任务
- 未在更复杂的时序任务（如时序预测、语音识别）上验证
- 对于静态图像数据集提升有限，因时间步间差异较小

## 相关工作与启发

- **vs TKS**: TKS 依赖标签判定正确/错误输出进行蒸馏；本文用置信度无需标签，且能利用暗知识
- **vs TSSD**: TSSD 扩展训练时间步并加弱分类器导致开销增加；本文零额外开销
- **vs 传统知识蒸馏**: 传统方法需要外部大教师模型；本文纯自蒸馏
- **vs ANN 自蒸馏（BYOT）**: ANN 需要添加多个输出头；SNN 天然通过时间步解构

## 评分

- 新颖性: ⭐⭐⭐⭐ 巧妙利用 SNN 时间特性进行自蒸馏，Weak2Strong 视角新颖
- 实验充分度: ⭐⭐⭐⭐ 静态 + 神经形态数据集，多架构验证，低延迟和鲁棒性分析
- 写作质量: ⭐⭐⭐⭐ 故事线清晰，从解构到识别到蒸馏循序渐进
- 价值: ⭐⭐⭐⭐ 零开销的 SNN 训练改进方法，可直接集成到现有 SNN 训练流程

<!-- RELATED:START -->

## 相关论文

- [Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [Weak-to-Strong Jailbreaking on Large Language Models](../../ICML2025/model_compression/weak-to-strong_jailbreaking_on_large_language_models.md)
- [S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)
- [QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)

<!-- RELATED:END -->
