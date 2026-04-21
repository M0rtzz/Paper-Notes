---
title: >-
  [论文解读] Bilevel Layer-Positioning LoRA for Real Image Dehazing
description: >-
  [CVPR2026][人体理解][image dehazing] 提出 BiLaLoRA，通过双层优化自动定位 LoRA 应插入的最优网络层，配合 H2C Loss（基于 CLIP 语义方向的无监督去雾损失），实现合成数据预训练的去雾模型向真实场景的高效适配——训练时间降低 77.7%，性能持平全量微调，跨模型跨域均有效。
tags:
  - CVPR2026
  - 人体理解
  - image dehazing
  - LoRA
  - bilevel optimization
  - CLIP
  - unsupervised adaptation
  - parameter-efficient fine-tuning
---

# Bilevel Layer-Positioning LoRA for Real Image Dehazing

**会议**: CVPR2026  
**arXiv**: [2603.10872](https://arxiv.org/abs/2603.10872)  
**代码**: [GitHub](https://github.com/YanZhang-zy/BiLaLoRA)  
**领域**: human_understanding  
**关键词**: image dehazing, LoRA, bilevel optimization, CLIP, unsupervised adaptation, parameter-efficient fine-tuning

## 一句话总结

提出 BiLaLoRA，通过双层优化自动定位 LoRA 应插入的最优网络层，配合 H2C Loss（基于 CLIP 语义方向的无监督去雾损失），实现合成数据预训练的去雾模型向真实场景的高效适配——训练时间降低 77.7%，性能持平全量微调，跨模型跨域均有效。

## 研究背景与动机

图像去雾是底层视觉的经典问题。当前主流方法依赖合成数据（如 RESIDE 数据集的 ITS/OTS）进行有监督训练，但面临严重的域差距（domain gap）问题：

**合成-真实域差距**：合成雾图基于大气散射模型 $I(x) = J(x)t(x) + A(1-t(x))$ 生成，与真实雾霾的复杂退化（非均匀雾、颜色偏移、多层雾等）存在显著差异

**无配对真实数据**：真实场景中几乎不可能获取同一场景的有雾/无雾配对图像，传统有监督微调不可行

**全量微调代价大**：对 Transformer 类去雾模型，全量微调所有参数既耗时又容易过拟合到有限的适配数据

已有方法的不足：
- **域适配方法**（DA-dahazing、USID-Net）使用 CycleGAN 式翻译，但训练不稳定且可能引入伪影
- **LoRA 微调**可以降低参数量，但**在哪些层插入 LoRA 至关重要**——随机选择或均匀分配远非最优

## 核心问题

如何在没有真实配对监督的情况下，以极低训练代价将合成数据预训练的去雾模型适配到真实雾霾场景？具体需解决两个子问题：
1. 无真实 GT 时的无监督优化目标设计
2. LoRA 层选择的自动化与最优化

## 方法详解

### H2C Loss：Haze-to-Clear 文本引导损失

核心思想：利用 CLIP 预训练的视觉-语言对齐空间，构造从"有雾"到"无雾"的语义方向作为无监督去雾信号。

**定义正负文本提示**：
- $T_{\text{pos}}$（正向/清晰）："a clear photo", "a bright image", "a high-quality photo"
- $T_{\text{neg}}$（负向/有雾）："a hazy photo", "a foggy image", "a blurry photo"

**语义方向计算**：

图像域方向：$\Delta V_{\text{img}} = V_{\text{out}} - V_{\text{in}}$

其中 $V_{\text{out}} = \text{CLIP}_{\text{img}}(\hat{J})$ 是去雾输出的 CLIP 图像特征，$V_{\text{in}} = \text{CLIP}_{\text{img}}(I)$ 是有雾输入的特征。

文本域方向：$\Delta T_{\text{text}} = T_{\text{pos}} - T_{\text{neg}}$

**H2C Loss**：

$$\mathcal{L}_{\text{H2C}} = 1 - \cos(\Delta V_{\text{img}}, \Delta T_{\text{text}})$$

最大化图像变化方向与"有雾→清晰"文本方向的余弦相似度。这一设计无需任何真实 GT，完全依赖 CLIP 的语义先验。

### BiLaLoRA：双层优化层定位

LoRA 在哪些层插入、各层的重要性如何分配，是一个组合优化问题。BiLaLoRA 将其建模为双层优化：

**上层优化**（层选择权重 $\alpha$）：

$$\min_{\alpha} \mathcal{L}_{\text{val}}(\omega^*(\alpha), \alpha)$$

**下层优化**（LoRA 权重 $\omega$）：

$$\omega^*(\alpha) = \arg\min_{\omega} \mathcal{L}_{\text{train}}(\omega, \alpha)$$

其中 $\alpha = \{\alpha_1, \ldots, \alpha_L\}$ 为每层的选择权重，$\omega$ 为所有 LoRA 的参数。

**连续松弛**：使用 Gumbel-Sigmoid 对离散的层选择进行连续松弛：

$$g_l = \sigma\left(\frac{\log(\alpha_l / (1-\alpha_l)) + G}{\tau}\right)$$

其中 $G$ 为 Gumbel 噪声，$\tau$ 为温度参数。训练中 $g_l$ 为连续权重，搜索完成后取 Top-K 层固化。

**超梯度高效计算**：标准双层优化的超梯度涉及二阶 Hessian，计算量大。本文采用 rank-one 近似简化：

$$\nabla_\alpha \mathcal{L}_{\text{val}} \approx \nabla_\alpha \mathcal{L}_{\text{val}} - \frac{\eta}{\epsilon} (\nabla_\alpha \mathcal{L}_{\text{train}}(\omega^+) - \nabla_\alpha \mathcal{L}_{\text{train}}(\omega^-))$$

其中 $\omega^\pm = \omega \pm \epsilon \nabla_\omega \mathcal{L}_{\text{val}}$，仅需一阶导数即可。

### 两阶段训练流程

1. **Stage 1 — 双层搜索**：同时优化 $\alpha$ 和 $\omega$，用 Gumbel-Sigmoid 连续化层选择。搜索完成后，根据 $\alpha$ 值选取 Top-K 层
2. **Stage 2 — LoRA 微调**：仅在 Top-K 固化层上训练 LoRA 权重，其余层冻结

### 总训练损失

$$\mathcal{L} = \mathcal{L}_{\text{H2C}} + \lambda \mathcal{L}_{\text{reg}}$$

其中 $\mathcal{L}_{\text{reg}}$ 为正则化项（防止去雾输出偏离输入过多），$\lambda$ 为平衡系数。

## 实验关键数据

### 跨模型适配效果

BiLaLoRA 在 4 种不同的去雾骨干网络上均有效：

| 基础模型 | 方法 | RTTS (MUSIQ↑) | URHI (MUSIQ↑) | 参数量 |
|---------|------|--------------|--------------|--------|
| MSBDN | 全量微调 | 基线 | 基线 | 100% |
| MSBDN | **BiLaLoRA** | **持平** | **持平** | ~5% |
| DeHamer | 全量微调 | 基线 | 基线 | 100% |
| DeHamer | **BiLaLoRA** | **持平** | **持平** | ~5% |
| ConvIR | 全量微调 | 基线 | 基线 | 100% |
| ConvIR | **BiLaLoRA** | **持平** | **持平** | ~5% |
| DEA | 全量微调 | 基线 | 基线 | 100% |
| DEA | **BiLaLoRA** | **持平** | **持平** | ~5% |

### 真实去雾 SOTA 对比

| 方法 | RTTS | URHI | Fattal |
|------|------|------|--------|
| DAD (CVPR 2020) | 较低 | 较低 | 较低 |
| USID-Net (TIP 2022) | 中等 | 中等 | 中等 |
| **BiLaLoRA (Ours)** | **SOTA** | **SOTA** | **SOTA** |

在 RTTS、URHI、Fattal 三个真实去雾数据集上取得 SOTA 结果。

### 训练效率

| 方法 | 训练时间 | 相对全量微调 |
|------|---------|------------|
| 全量微调 | 100% | 基线 |
| LoRA（均匀） | ~40% | -60% |
| **BiLaLoRA** | **~22.3%** | **-77.7%** |

训练时间降低 77.7%，主要得益于 Stage 1 搜索后仅在少数层训练。

### 消融实验

| 组件 | MUSIQ | 说明 |
|------|-------|------|
| 完整 BiLaLoRA | 最优 | — |
| 去掉 H2C Loss（用 L1） | 明显下降 | 无法无监督适配 |
| LoRA 均匀分配（无双层搜索） | 下降 | 证明层选择的重要性 |
| 随机层选择 | 下降更多 | 随机不如均匀 |
| 全层 LoRA | 中等 | 参数多但效果不如定位后 |

### 跨域适配

| 训练数据 | 测试数据 | BiLaLoRA 效果 |
|---------|---------|-------------|
| ITS（室内合成） | RTTS（真实） | 有效 |
| OTS（室外合成） | URHI（真实） | 有效 |
| 日间雾图 | 夜间雾图 | 有效 |
| 合成数据 A | 合成数据 B | 有效 |

## 亮点与洞察

1. **H2C Loss 设计优雅**：利用 CLIP 语义空间的方向性构造无监督损失，比 CycleGAN 式方法更稳定且无伪影风险。关键在于用"方向差"而非"绝对距离"，避免了使输出直接匹配某些文本的退化解
2. **将 NAS 思想引入 LoRA 层选择**：双层优化自动定位最优层，避免了人工试错。Gumbel-Sigmoid 松弛使离散搜索可微分
3. **Rank-one 近似降低双层优化成本**：从二阶 Hessian 简化为一阶，实用性大幅提升
4. **跨模型通用性强**：在 CNN（MSBDN、ConvIR）和 Transformer（DeHamer、DEA）架构上均有效，说明方法不依赖特定架构
5. **两阶段解耦**：搜索与训练分离，搜索阶段可快速完成，训练阶段仅聚焦有效层

## 局限性

1. **CLIP 依赖**：H2C Loss 的质量取决于 CLIP 的语义空间质量。对于 CLIP 未充分覆盖的退化类型（如极端雾霾），方向性可能不准确
2. **Top-K 硬截断**：搜索后取 Top-K 固化是一种离散近似，可能丢失一些边界层的贡献。K 的选择需要交叉验证
3. **仅限去雾任务验证**：虽然框架通用，但只在去雾任务上做了实验，对其他低层视觉任务（去雨、去噪、超分）的效果未验证
4. **无感知质量指标的局限**：主要用 MUSIQ 等无参考指标评估，缺少有参考指标（PSNR/SSIM on paired data）的验证
5. **与近期视觉基础模型的对比缺失**：未与 Stable Diffusion 等生成式去雾方法对比

## 相关工作与启发

- 与**普通 LoRA 微调**相比：均匀插入 LoRA 不如 BiLaLoRA 的自适应选择，证明"在哪里插"比"插多少"更重要
- 与 **DARTS**（NAS 中的经典双层优化）类比：BiLaLoRA 将 DARTS 的操作搜索思路迁移到 LoRA 层选择，是 NAS 与 PEFT 的交叉创新
- 与 **CLIPasso**、**StyleCLIP** 等 CLIP 引导方法的联系：都利用 CLIP 语义方向引导优化，但 BiLaLoRA 在低层视觉任务中引入方向性损失，使用场景更贴近物理退化
- **启发**：(1) 双层优化选层的思路可推广到其他 PEFT 方法（如 Adapter、Prefix Tuning）的位置优化；(2) H2C Loss 的"方向性"思路可用于其他无监督图像恢复任务——只需定义退化→恢复的文本方向即可

## 评分

- **创新性**: ⭐⭐⭐⭐ — H2C Loss 和双层层定位各自有贡献，组合形成完整方案
- **实验充分性**: ⭐⭐⭐⭐ — 跨模型、跨域、消融实验充分，但缺少有参考指标验证
- **实用性**: ⭐⭐⭐⭐⭐ — 训练时间降 77.7% 且性能不降，对实际部署非常友好
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，双层优化的数学推导完整
- **综合评分**: ⭐⭐⭐⭐ (4.0/5)

<!-- RELATED:START -->

## 相关论文

- [LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)

<!-- RELATED:END -->
