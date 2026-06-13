---
title: >-
  [论文解读] Instant Adversarial Purification with Adversarial Consistency Distillation
description: >-
  [CVPR 2025][图像生成][adversarial purification] 提出 One Step Control Purification (OSCP) 框架，结合 Gaussian Adversarial Noise Distillation (GAND) 和 Controlled Adversarial Purification (CAP)，在单次 U-Net 推理（~0.1 秒）内完成对抗净化，相比传统扩散净化方法加速 100 倍。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "adversarial purification"
  - "consistency distillation"
  - "OSCP"
  - "GAN"
  - "ControlNet"
  - "one-step defense"
---

# Instant Adversarial Purification with Adversarial Consistency Distillation

**会议**: CVPR 2025  
**arXiv**: [2408.17064](https://arxiv.org/abs/2408.17064)  
**代码**: 待确认  
**领域**: image_generation (adversarial defense / diffusion models)  
**关键词**: adversarial purification, consistency distillation, OSCP, GAND, ControlNet, one-step defense

## 一句话总结

提出 One Step Control Purification (OSCP) 框架，结合 Gaussian Adversarial Noise Distillation (GAND) 和 Controlled Adversarial Purification (CAP)，在单次 U-Net 推理（~0.1 秒）内完成对抗净化，相比传统扩散净化方法加速 100 倍。

## 研究背景与动机

**领域现状**: 基于扩散模型的对抗净化（如 DiffPure）通过将对抗样本映射回自然分布来防御攻击，效果显著但多步去噪导致计算开销极大（每张图 ~10 秒）。

**现有痛点**:
1. **扩散净化太慢**: DiffPure 和 GDMP 需要 100+ 步迭代去噪，耗时 9-11 秒/张，无法用于实时场景
2. **对抗训练过拟合已知攻击**: 对抗训练对未知攻击泛化能力差
3. **大步长净化语义丢失**: 增大扩散步长 $t^*$ 虽可更彻底去除对抗噪声，但生成图像偏离原图语义

**核心矛盾**: Consistency Model 的核心假设是 $f_\theta(z_t, t) = f_\theta(z_{t'}, t')$ 对所有 $t, t' \in [0,T]$ 成立，但对抗图像的 latent 分布偏移违反了这一一致性约束，导致直接用 LCM 净化无法收敛到干净图像。

**本文目标**: 在保持扩散净化效果的同时实现单步推理，克服对抗噪声与高斯噪声分布差异导致的一致性蒸馏困难。

**切入角度**: 从蒸馏目标和推理流程两方面同时解决——GAND 修改蒸馏目标适配对抗噪声，CAP 用边缘引导保持语义。

## 方法详解

### 整体框架

OSCP 由两个组件构成：
1. **GAND (Gaussian Adversarial Noise Distillation)**: 训练阶段——修改 LCM 蒸馏目标，使模型学会同时去除高斯噪声和对抗噪声
2. **CAP (Controlled Adversarial Purification)**: 推理阶段——用 Canny 边缘图通过 ControlNet 引导净化，防止大步长下语义偏移

推理流程: 对抗图 $\mathbf{x}_{adv}$ → VAE 编码 $\mathbf{z}_{adv}$ → 前向扩散加噪到 $t^*$ → GAND-LCM 单步去噪（ControlNet 边缘引导）→ VAE 解码 → 净化图

### 关键设计

**1. Gaussian Adversarial Noise Distillation (GAND)**
- **功能**: 在 Latent Consistency Distillation 基础上，将对抗噪声显式引入前向过程的噪声项
- **核心思路**: 定义新的扩散前向过程
  $$z_t^* = \sqrt{\bar{\alpha}_t} z + \sqrt{1 - \bar{\alpha}_t}(\epsilon + \delta_{adv})$$
  其中 $\delta_{adv}$ 是在 latent 空间通过 PGD 攻击生成的对抗扰动。关键观察：$z_t^* \to z$ 当 $t \to 0$（收敛到干净图），$z_t^* \to \epsilon + \delta_{adv}$ 当 $t \to T$（收敛到混合噪声）。这样 LCM 的一致性约束在 $[0, t]$ 内重新成立
- **蒸馏损失**:
  $$\mathcal{L}_{Total} = \mathcal{L}_{GAND}(\theta, \theta^-) + \lambda_{CIG} \mathcal{L}_{CIG}(\theta)$$
  - $\mathcal{L}_{GAND}$: 一致性蒸馏损失，作用于含对抗噪声的扩散轨迹
  - $\mathcal{L}_{CIG}$: Clear Image Guide 损失，直接约束输出向干净 latent 收敛
- **高效训练**: 使用 LoRA 参数高效微调，避免全参数训练的巨大计算开销
- **设计动机**: 直接将 LCM 用于对抗净化会失败，因为对抗 latent $z_{adv}(t)$ 的分布偏移破坏了一致性函数的收敛条件；通过显式将 $\delta_{adv}$ 编入扩散前向过程，使蒸馏轨迹直接建模"从混合噪声到干净图"的映射

**2. Controlled Adversarial Purification (CAP)**
- **功能**: 推理时用不可学习的 Canny 边缘检测算子提取对抗图的边缘图，通过 ControlNet 引导净化过程
- **核心思路**: 
    - 边缘图提供结构先验，防止大步长净化下输出图偏离原图的几何结构
    - 去掉 LCM 的 skip connection 项 $c_{skip}(t)z_{adv}(t)$（即设 $c_{skip} \equiv 0$），因为该项会保留对抗噪声
    - 最终净化公式: $\hat{z}_{adv}^0 = c_{out}(t) \cdot \frac{z_{adv} - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_\theta(z_{adv}, c_{ce}, t)}{\sqrt{\bar{\alpha}_t}}$
- **设计动机**: 使用不可学习的边缘检测而非文本引导，因为文本可能被 caption 语义攻击所欺骗；去掉 skip connection 是因为它直接传递对抗噪声

**3. Latent Space 对抗训练**
- **功能**: GAND 训练中的对抗扰动 $\delta_{adv}$ 在 latent 空间而非 pixel 空间生成
- **核心思路**: $\delta_{adv} = \arg\max_\delta \mathcal{L}(C(\mathcal{D}(\mathcal{E}(x) + \delta)), y)$，使用 PGD-10 攻击
- **设计动机**: latent 空间攻击与扩散模型的工作空间一致，且避免了 pixel 空间攻击经过 VAE 编码后的分布变化

### 损失函数 / 训练策略

- 蒸馏数据: ImageNet 前 40K 验证集图像（512×512）
- 基础模型: Stable Diffusion v1.5
- 训练: 20K iterations，batch size 4，lr 8e-6，500 step warm-up
- LoRA 微调，DDIM solver 作为 PF-ODE solver，skip step $k=20$
- 对抗扰动: PGD-10, $\epsilon=0.03$, 针对 ResNet50
- $\lambda_{CIG} = 0.001$
- 推理时 $t^* = 200$（最优净化强度）

## 实验关键数据

### 主实验（ImageNet）

| 方法 | 类别 | 攻击方法 | Standard Acc↑ | Robust Acc↑ |
|---|---|---|---|---|
| Without defense | - | AutoAttack | 80.55% | 0.00% |
| DiffPure | DBP | AutoAttack | 75.77% | 73.02% |
| Amini et al. | Adv Train | AutoAttack | 77.96% | 59.64% |
| **OSCP (Ours)** | **Hybrid** | **AutoAttack** | **77.63%** | **74.19%** |
| **OSCP (Ours)** | **Hybrid** | **PGD-100** | **77.63%** | **73.89%** |

### 推理速度对比

| 方法 | 数据集 | 时间/张 |
|---|---|---|
| GDMP | ImageNet | ~9s |
| DiffPure | ImageNet | ~11s |
| **OSCP (Ours)** | **ImageNet** | **~0.1s** |

**100 倍加速**，且推理时间不随 $t^*$ 变化（单步推理）。

### 跨架构泛化（PGD-100攻击）

| Architecture | Clean ASR | Robust Acc↑ |
|---|---|---|
| ResNet-50 | 100% | 73.89% |
| WRN-50-2 | 100% | 75.2% |
| ViT-b-16 | 100% | 71.6% |
| Swin-b | 100% | 77.8% |

### 自适应攻击防御（Diff-PGD-10）

| 方法 | ResNet-50 | ViT-b-16 | Swin-b |
|---|---|---|---|
| DiffPure | 53.8% | 16.6% | 45.1% |
| **OSCP** | **59.0%** | **34.1%** | **53.9%** |

### 关键发现

1. **单步推理不牺牲防御效果**: OSCP 在 AutoAttack 上的 robust accuracy (74.19%) 优于 DiffPure (73.02%)，同时速度快 100 倍
2. **跨架构迁移良好**: 仅用 ResNet50 对抗样本训练的 GAND，在 ViT、Swin 等架构上也有效（最低 71.6%）
3. **对自适应攻击也鲁棒**: 面对专门针对扩散净化的 Diff-PGD 攻击，OSCP 仍优于 DiffPure 5-17%
4. **LoRA 微调即可**: 不需要全参数训练，参数高效微调就能达到最优性能

## 亮点与洞察

- 核心洞察深刻：识别了 LCM 在对抗场景下一致性约束被破坏的根本原因（$z_{adv}$ 分布偏移），并通过修改前向过程优雅地解决
- GAND 的 $z_t^* = \sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}(\epsilon + \delta_{adv})$ 设计精巧，使蒸馏轨迹的两端分别收敛到干净图和混合噪声，恢复一致性
- CAP 使用不可学习边缘检测而非神经网络条件，有效防止对抗样本"毒化"引导信号
- 从"对抗训练+净化"的混合视角出发，结合了两种防御范式的优势

## 局限与展望

- 训练时的对抗扰动仅用 PGD-10 生成，面对更强/更多样的攻击策略可能需要扩展
- ControlNet 边缘引导可能对纹理型攻击不够有效（边缘检测器不受影响但纹理信息丢失）
- 基于 SD 1.5，未探索更新的扩散模型底座（如 SDXL）
- Standard accuracy 从 80.55% 降至 77.63%，仍存在约 3% 的干净图性能损失
- 未探讨对 $L_2$ 范数攻击的防御效果，主要集中在 $L_\infty$

## 相关工作与启发

- DiffPure 奠定了扩散净化的理论基础（KL散度减小），OSCP 将其从多步推理压缩到单步
- LCM/LCM-LoRA 实现了扩散模型的快速推理，OSCP 首次将其用于对抗防御
- ControlNet 的条件控制能力被创造性地用于保持净化后的语义一致性
- 启发：蒸馏目标的"对抗感知修改"范式可推广到视频对抗净化、3D 对抗防御等新场景

## 评分

⭐⭐⭐⭐ — 对 LCM 一致性在对抗场景下失效的分析透彻，GAND+CAP 设计有理论支撑且实验效果优异，100 倍加速具有重要实用价值；主要局限在攻击覆盖度和干净图性能损失。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[CVPR 2025\] Detecting Adversarial Data Using Perturbation Forgery](detecting_adversarial_data_using_perturbation_forgery.md)
- [\[CVPR 2025\] IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](../../ECCV2024/image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ICML 2026\] Adversarial Flow Models](../../ICML2026/image_generation/adversarial_flow_models.md)

</div>

<!-- RELATED:END -->
