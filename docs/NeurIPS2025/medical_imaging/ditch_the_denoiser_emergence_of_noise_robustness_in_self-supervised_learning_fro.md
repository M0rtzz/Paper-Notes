---
title: >-
  [论文解读] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum
description: >-
  [NeurIPS 2025][医学图像][自监督学习] 提出一种全自监督的噪声鲁棒表示学习框架，通过"去噪→噪声"的数据课程学习策略 + 去噪教师正则化，使 DINOv2 等 SSL 模型在推理时无需去噪器即可直接处理噪声输入，在 ImageNet-1k 极端高斯噪声下线性探测精度提升 4.8%。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "自监督学习"
  - "噪声鲁棒性"
  - "课程学习"
  - "DINOv2"
  - "去噪器无关"
---

# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum

**会议**: NeurIPS 2025  
**arXiv**: [2505.12191](https://arxiv.org/abs/2505.12191)  
**代码**: [https://github.com/wenquanlu/noisy_dinov2](https://github.com/wenquanlu/noisy_dinov2)  
**领域**: 医学图像  
**关键词**: 自监督学习, 噪声鲁棒性, 课程学习, DINOv2, 去噪器无关

## 一句话总结
提出一种全自监督的噪声鲁棒表示学习框架，通过"去噪→噪声"的数据课程学习策略 + 去噪教师正则化，使 DINOv2 等 SSL 模型在推理时无需去噪器即可直接处理噪声输入，在 ImageNet-1k 极端高斯噪声下线性探测精度提升 4.8%。

## 研究背景与动机

**领域现状**：自监督学习（SSL）方法如 DINOv2 在干净数据上取得了卓越的视觉表征效果，但几乎所有研究都假设训练数据是干净、高质量的。

**现有痛点**：在医学影像、天文学、遥感等真实场景中，数据天然带有噪声（如传感器噪声、散斑噪声），且通常无法获取干净参考图像进行有监督去噪。直接在噪声数据上训练 DINOv2 会导致表征质量严重下降。

**核心矛盾**：一个朴素的解决方案是"去噪器预处理流水线"——先用自监督去噪器去噪，再在去噪图像上训练 SSL 模型。但这要求在推理和下游微调时都保留去噪器，带来显著的推理延迟、部署复杂性和潜在的去噪偏差传递。

**本文目标** 能否让 SSL backbone 自身"内化"噪声鲁棒性，在下游任务中完全丢弃去噪器？

**切入角度**：受课程学习启发——先在"简单"的去噪数据上训练，再切换到"困难"的噪声数据，让模型从稳定初始化出发逐步适应噪声。同时引入冻结的去噪教师作为锚点进行正则化。

**核心 idea**：通过去噪→噪声的课程训练策略 + 去噪教师正则化，让 SSL 模型在预训练阶段内化噪声鲁棒性，推理时完全无需去噪器。

## 方法详解

### 整体框架
整个流程分三步：(1) 训练一个自监督去噪器（如 Neighbor2Neighbor）对噪声数据集去噪；(2) 用课程学习策略训练 DINOv2——先在去噪数据上训练 k 个 epoch 建立稳定初始化，然后重启训练在原始噪声数据上继续训练至收敛；(3) 下游任务直接在噪声数据上微调和推理，无需去噪器。

### 关键设计

1. **DINOv2 w/ NC（Noise Curriculum）——噪声课程学习**:

    - 功能：设计从"简单到困难"的数据课程，先用去噪数据训练再切换到噪声数据
    - 核心思路：定义分布序列 $\langle Q_1, Q_2 \rangle$，其中 $Q_1$ 对应低熵的去噪图像分布，$Q_2$ 对应高熵的噪声图像分布，满足 $H(Q_1) < H(Q_2)$。在去噪数据上训练 k 个 epoch 后，**重启**所有训练动态（学习率、权重衰减调度等）在噪声数据上继续训练
    - 设计动机：在 MNIST toy experiment 中验证——在高斯噪声 $\sigma=0.4\times255$ 下，直接在噪声数据上训练 SSL 只有 64.55% 精度，但课程学习（30 epoch 干净 + 20 epoch 噪声）可恢复到 83.05%，证明模型能从干净阶段的学习中保留噪声鲁棒特征
    - 下游推理公式：$\hat{y} = h_\theta(g_\theta(z))$，其中 $g_\theta$ 是骨干网络，$h_\theta$ 是预测头，$z$ 是噪声输入，完全不需要去噪预处理

2. **DINOv2 w/ NCT（Noise Curriculum Teacher）——去噪教师正则化**:

    - 功能：在高噪声场景下，用冻结的去噪教师锚定训练，防止噪声阶段学习偏离
    - 核心思路：在噪声训练阶段引入三组件架构——可训练的 teacher $T$、student $S$、冻结的去噪教师 $T_{dn}$。对噪声图像 $x$ 和去噪图像 $x_{dn}$ 施加完全相同的数据增强，然后在原始 DINOv2 损失基础上添加正则化项：
    $L = L_{\text{dinov2}} + \lambda L_{\text{dino\&ibot}}(T_{dn}(\tau_t(x_{dn})), S(\tau_s(x)))$
    - 其中原始 DINOv2 损失由 DINO + iBOT 的交叉熵组成：$L_{\text{dino\&ibot}} = -\sum p_t^{\text{img}} \log p_s^{\text{img}} - \sum p_t^{\text{patch}} \log p_s^{\text{patch}}$
    - 设计动机：在极端噪声（如高斯 $\sigma=255$, SNR=0.31dB）下，仅靠课程学习的初始化不足以抵抗强噪声的干扰，需要额外的锚点约束。冻结教师和可训练教师初始权重完全相同（都来自去噪阶段），确保输出嵌入对齐

3. **去噪器选择的灵活性**:

    - 功能：使用 Neighbor2Neighbor 作为自监督去噪器，但框架对去噪器不做限制
    - 核心思路：N2N 从单张噪声图像中采样两个子图像构建噪声-噪声对来训练 U-Net，不需要干净参考图像
    - 设计动机：实际应用中应根据领域的噪声特性选择合适的去噪器。即使使用很弱的去噪器（训练 1 个 epoch），NC 方法仍能获得显著提升

### 损失函数 / 训练策略
- NC 方法使用标准 DINOv2 损失（DINO + iBOT + Koleo 正则化），分两阶段训练
- NCT 方法在噪声阶段额外加入 $\lambda$ 控制的去噪正则化项
- ImageNet-100 上使用 ViT-S/16，200 epoch，batch size 40；ImageNet-1k 上使用 ViT-B/16，100 epoch，batch size 512

## 实验关键数据

### 主实验

| 数据集 | 噪声 | 方法 | 线性探测精度 | vs DINOv2 提升 |
|--------|------|------|------------|---------------|
| ImageNet-1k | Gaussian σ=100 (4.36dB) | DINOv2 w/ NCT | 72.1% | +1.4% |
| ImageNet-1k | Gaussian σ=100 (4.36dB) | N2N + DINOv2 | 73.1% | +2.4% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 w/ NCT | 55.8% | **+4.8%** |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | N2N + DINOv2 | 57.2% | +6.2% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 w/ NC | 53.5% | +2.5% |
| ImageNet-1k | Gaussian σ=255 (0.72dB) | DINOv2 baseline | 51.0% | — |

### 消融实验（跨 SSL 模型泛化，ImageNet-100 Gaussian σ=100）

| SSL 模型 | 架构 | 基线精度 | w/ NC 精度 | N2N + 模型 |
|---------|------|---------|-----------|-----------|
| DINOv2 | ViT-S | 55.4% | 68.1% | 69.0% |
| DINO | ViT-S | 57.9% | 62.1% | 62.5% |
| iBOT | ViT-S | 56.9% | **62.7%** | 61.9% |
| SimCLR | ResNet50 | 59.0% | 61.1% | 64.3% |
| MoCo v3 | ViT-S | 52.2% | 55.3% | 60.4% |
| SimSiam | ResNet50 | 64.8% | 65.7% | 68.4% |

### 关键发现
- NCT 正则化在极端噪声下提升最大（Gaussian σ=255 时 NC→NCT 提升 6.7%），但在中等噪声下收效甚微
- **惊人发现**：在干净验证集上评估时，DINOv2 w/ NC/NCT 在多数情况下**超过** N2N + DINOv2，说明去噪器无关的方法学到了更泛化的表征（因为显式去噪会丢失部分有用信息）
- NCT 方法会收敛到或超越其锚点（去噪教师）的精度，说明正则化有效引导了学习方向
- 延长训练可缩小方法间差距，但噪声课程策略能在约一半的训练时间内达到同等性能

## 亮点与洞察
- **推理时完全丢弃去噪器**：这是最核心的实用价值——部署更简单、推理更快、不会传递去噪偏差，对医学影像等资源受限场景尤其重要
- **课程学习的"重启"策略**：不是简单地从去噪切换到噪声继续训练，而是重置学习率等训练动态，让模型在噪声阶段有足够的适应空间
- **跨 SSL 框架泛化**：NC 策略在 6 种不同 SSL 方法上都有效，iBOT w/ NC 甚至超越了 N2N + iBOT，说明方法有广泛适用性
- **去噪教师与可训练教师的对齐**是 NCT 有效的关键——随意使用不同初始化的冻结教师不会产生有意义的正则化

## 局限与展望
- 当前仅在合成噪声上验证，缺乏真实世界噪声数据集（如医学影像的设备噪声）的评估
- 课程切换时间点（何时从去噪切换到噪声）需手动调优，缺乏自适应策略
- 假设自监督去噪器能合理去噪——在极端噪声下如果去噪器本身效果很差，整个框架可能受影响
- 可拓展到时序数据（音频、EEG、金融数据）等其他模态

## 相关工作与启发
- **vs N2N + DINOv2 (去噪预处理流水线)**：该方法始终需要去噪器用于推理，本文方法在推理时完全无需去噪器，且在干净测试集上表征质量更好
- **vs 噪声时序 SSL [48]**：之前的噪声鲁棒 SSL 工作集中在时序数据（EEG），也使用去噪器创建对比学习对，但动机类似——利用去噪器辅助学习噪声鲁棒表征
- 对医学影像 SSL 预训练有直接启发：医学数据通常噪声大、无干净参考，本文方法可减少对去噪前处理的依赖

## 评分
- 新颖性: ⭐⭐⭐⭐ 噪声课程学习 + 去噪教师正则化的组合在 SSL 中是首次探索，但各组件本身并不新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多种噪声类型/强度、多数据集规模、多 SSL 框架、分类 + 检索两种下游任务
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从 toy example 到大规模实验层层推进
- 价值: ⭐⭐⭐⭐ 对医学影像、遥感等噪声数据场景有实际应用价值，推理无需去噪器是很好的实用特性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[CVPR 2025\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](../../CVPR2025/medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[NeurIPS 2025\] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation](unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)

</div>

<!-- RELATED:END -->
