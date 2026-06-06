---
title: >-
  [论文解读] Resilience of Entropy Model in Distributed Neural Networks
description: >-
  [ECCV2024][AI安全][distributed DNN] 首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平…
tags:
  - "ECCV2024"
  - "AI安全"
  - "distributed DNN"
  - "entropy coding"
  - "adversarial attack"
  - "communication efficiency"
  - "去噪"
---

# Resilience of Entropy Model in Distributed Neural Networks

**会议**: ECCV2024  
**arXiv**: [2403.00942](https://arxiv.org/abs/2403.00942)  
**代码**: [EntropyR](https://github.com/Restuccia-Group/EntropyR)  
**领域**: AI安全  
**关键词**: distributed DNN, entropy coding, adversarial attack, communication efficiency, total variation denoising

## 一句话总结

首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平，准确率仅下降约 2%。

## 背景与动机

分布式深度神经网络（Distributed DNN）将大模型拆分为部署在移动设备上的 head network 和部署在服务器上的 tail network，通过只传输中间压缩表征来降低通信开销。近年来，熵编码（entropy coding）被引入以进一步压缩中间表征——核心思路是联合训练 DNN 与辅助熵模型，推理时利用熵模型输出的先验分布作为 side information，将量化后的隐表征自适应编码为变长 bit stream。

然而，现有工作从未考虑过熵模型本身的鲁棒性问题。DNN 对分布偏移和对抗扰动是脆弱的，这一点已被广泛研究；但熵模型是在干净数据上训练的，输入空间的微小变化可能导致熵估计急剧增大，编码后的 bit rate 可能超出传输带宽，最坏情况下传输数据量增加约 2 倍。这不仅影响单个用户的端到端延迟，还可能饱和共享带宽，威胁其他用户。

## 核心问题

1. **熵模型对无意干扰（common corruption）的鲁棒性如何？** 不同类型的图像损坏（噪声、模糊、天气、数字退化）如何影响编码数据量？
2. **熵模型对有意干扰（对抗攻击）的鲁棒性如何？** 专门针对熵（即 bit rate）的对抗攻击（PGD-E）与针对准确率的攻击（PGD-Acc）有何本质区别？
3. **如何设计防御机制来保护熵模型？** 能否在几乎不损失分类精度的前提下显著降低被攻击后的传输开销？

## 方法详解

### 威胁模型

论文将干扰建模为输入的加性扰动 $\delta$，形式化两种攻击目标：

- **PGD-Acc**：经典的 PGD 攻击，以交叉熵为损失函数，目标是降低分类准确率
- **PGD-E**：以熵编码的 rate loss $-\log_2 P_Z(z)$ 为损失函数，目标是最大化隐表征的熵，从而增大编码后的数据量

两者均使用 $l_\infty$ 约束，通过投影梯度下降求解。

### 关键发现：压缩特征与分类特征的解耦

**频率域分析**：通过对比图像的 total variation map 和熵模型的 bit rate map，发现二者高度相关——熵模型对高频особенности非常敏感。引入高频噪声（如 shot noise）会显著增大数据量（+65%），而去除高频信息（如 defocus blur）反而减小数据量（-53%）。这是因为分布式 DNN 的 head network 在浅层切分，浅层主要捕获低级特征（边缘、纹理等高频信息）。

**空间域分析**：PGD-E 主要在背景区域增加 bit rate，而 PGD-Acc 集中攻击前景物体区域。这说明两类攻击针对输入空间中不同的特征集合，互相影响很小。

### 防御方法：Object-Aware Total Variation Denoising

基于上述发现，论文提出目标感知的全变差去噪方法：

1. **Total Variation 去噪**：求解优化问题 $\min_x \frac{1}{2}\|x - x'\|_2^2 + \lambda \cdot TV(x)$，在保留图像主要内容的同时去除高频噪声，使用次梯度下降迭代求解
2. **目标感知 mask**：由于直接全图去噪也会损害前景物体的分类信息，利用熵模型输出 $P_Z(z)$ 作为 soft mask——物体区域 bit rate 高对应 $P_Z(z)$ 值小，因此自然避免对物体区域过度平滑
3. **最终迭代公式**：$x^{i+1} = x^i - \alpha \cdot m \cdot ((x^i - x') + \lambda \cdot g(x^i))$，其中 $m$ 为 mask，$g(x^i)$ 为图像梯度

该方法无需重新训练模型，作为独立的前处理模块可与对抗训练等方法结合使用。

## 实验关键数据

**实验配置**：3 种 DNN 架构（ResNet-50、ResNet-101、RegNetY-6.4GF）、2 种熵模型（Factorized Prior / FP、Mean Scale Hyper Prior / MSHP）、4 种 rate-distortion trade-off $\beta$，在 ImageNet / ImageNet-C 上评估。

### 无意干扰

| 干扰类型 | 数据量变化 | 说明 |
|---------|-----------|------|
| Shot noise | +65.31% | 引入高频噪声 |
| Snow | -4.42% | 频域无特定模式 |
| Defocus blur | -53.31% | 去除高频信息 |
| Contrast | -67.45% | 去除高频信息 |

### 有意干扰（$\epsilon=8/255$，MSHP）

| 攻击方式 | 数据量增加 | 准确率下降 |
|---------|-----------|-----------|
| PGD-Acc | +10.19% | -57.10% |
| PGD-E | +46.82% | -6.62% |

### 防御效果（MSHP，干净数据 9.62 KB）

| 扰动预算 | 攻击后数据量 | 防御后数据量 | 防御后准确率损失 |
|---------|------------|------------|---------------|
| $\epsilon=2/255$ | 12.35 KB | 8.76 KB（<干净） | -2.44% |
| $\epsilon=4/255$ | 14.11 KB | 9.55 KB（≈干净） | -1.18% |
| $\epsilon=8/255$ | 16.15 KB | 11.02 KB | 准确率反升 +1.60% |
| $\epsilon=16/255$ | 18.79 KB（+95%） | 13.44 KB | 准确率反升 +7.74% |

### 自适应攻击

针对低频攻击和区域攻击两种自适应策略，防御仍然有效：数据量降低 ~67%，准确率仅下降 ~1.3%。

## 亮点

1. **首次揭示熵模型的安全盲区**：分布式 DNN 中的熵编码模块此前从未被审视其对抗鲁棒性，论文填补了这一重要空白
2. **深刻的特征解耦洞察**：通过频率域和空间域两个维度证明压缩特征与分类特征是分离的，这一发现具有理论价值
3. **简洁有效的防御设计**：基于 total variation 去噪 + entropy soft mask 的方案无需重训练、计算开销小、效果显著——小扰动下甚至可以让数据量低于干净数据
4. **对自适应攻击的鲁棒性**：在白盒设定下面对专门设计的自适应攻击仍表现稳健

## 局限与展望

1. **仅评估分类任务**：未验证在目标检测、语义分割等其他下游任务上的效果
2. **mask 依赖熵模型本身**：防御使用 $P_Z(z)$ 作为 soft mask，面对更聪明的攻击者可能被利用
3. **大扰动下数据量仍偏高**：$\epsilon=16/255$ 时防御后数据量仍为 13.44 KB，高于干净数据的 9.62 KB 约 40%
4. **未探索端到端联合优化防御**：当前方法作为前处理，未与模型训练过程联合优化
5. **head network 固定在浅层**：不同切分点对应不同的特征层次，可能影响压缩特征的频率特性

## 与相关工作的对比

| 方向 | 代表工作 | 本文区别 |
|------|---------|---------|
| 动态 DNN 效率攻击 | Hong et al., Haque et al. | 那些工作攻击计算效率，本文首次攻击通信效率 |
| 密度估计鲁棒性 | Arvinte et al. | 他们关注最大化 $P_Z(z)$，本文关注最小化 $P_Z(z)$（增大 bit rate） |
| 分布式 DNN 压缩 | Entropic Student | 提出了熵编码集成方案但未考虑鲁棒性 |
| 传统对抗防御 | 对抗训练、输入净化 | 本文方法可作为独立模块与这些方法组合使用 |

## 启发与关联

- **安全视角的启示**：在边缘计算场景中，攻击者不一定要降低准确率——仅通过增加传输开销就能造成带宽饱和的 DoS 效果，这是一个被忽视但实际威胁很大的攻击面
- **压缩与识别的解耦**：浅层特征主要编码高频信息（纹理、边缘），深层特征编码语义信息，这一观察可指导分布式系统中切分点的选择和鲁棒性设计
- **可扩展到其他压缩场景**：任何使用 learned entropy model 的神经压缩系统（图像压缩、视频压缩）都可能面临类似的攻击风险

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究熵模型鲁棒性，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ — 多架构多熵模型多参数的 ablation 覆盖全面，且考虑了自适应攻击
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，可视化分析（bit rate map / TV map / 空间对比图）直观有力
- 价值: ⭐⭐⭐⭐ — 揭示被忽视的安全问题，防御方案实用且可组合

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICML 2026\] Frequency Matching in Spiking Neural Networks for mmWave Sensing](../../ICML2026/ai_safety/frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)

</div>

<!-- RELATED:END -->
