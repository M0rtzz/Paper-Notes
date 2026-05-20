---
title: >-
  [论文解读] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)
description: >-
  [CVPR 2025][图像生成][零样本图像恢复] CM4IR 提出一种基于一致性模型（Consistency Model）的零样本图像恢复方案，通过新颖的噪声注入机制（解耦去噪/注入噪声级别 + 随机/估计噪声分割）结合反投影引导和更好的初始化…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "零样本图像恢复"
  - "一致性模型"
  - "噪声注入"
  - "反投影引导"
  - "少步推理"
---

# Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)

**会议**: CVPR 2025  
**arXiv**: [2412.20596](https://arxiv.org/abs/2412.20596)  
**代码**: [https://github.com/tirer-lab/CM4IR](https://github.com/tirer-lab/CM4IR)  
**领域**: 图像生成  
**关键词**: 零样本图像恢复, 一致性模型, 噪声注入, 反投影引导, 少步推理

## 一句话总结
CM4IR 提出一种基于一致性模型（Consistency Model）的零样本图像恢复方案，通过新颖的噪声注入机制（解耦去噪/注入噪声级别 + 随机/估计噪声分割）结合反投影引导和更好的初始化，仅用 4 次神经网络评估即超越需要 20-1000 次的现有扩散模型方法。

## 研究背景与动机

**领域现状**：零样本图像恢复（不针对特定退化模型训练专用网络）已成为趋势。基于扩散模型（DM）的方法通过预训练的去噪模型作为信号先验，在推理时添加数据保真度引导来恢复图像。但现有方法至少需要 20+ 次 NFE（神经函数评估），如 DPS 需要 1000 次，DiffPIR 需要 20 次。

**现有痛点**：NFE 数量多导致推理慢，这根源于扩散模型本身生成过程就需要大量迭代。一致性模型（CM）可以用 1-2 次 NFE 生成图像，但现有的 CM 引导恢复方法（原始 CM 论文需要 40 NFE，CoSIGN 需要针对每个任务微调，不是零样本）仍有局限。

**核心矛盾**：CM 在无条件生成中能极少步完成，但在有引导的恢复场景中，现有的噪声注入和引导机制不适配 CM 的少步特性，导致步数无法进一步压缩。

**本文目标**：设计一种真正的零样本恢复方案，利用预训练 CM 在仅 4 次 NFE 下完成高质量的超分辨率、去模糊和修复。

**切入角度**：作者发现三个关键：(1) 恢复任务中去噪操作的噪声级别和注入噪声级别不需要相同；(2) 可以将注入噪声分为随机部分和估计部分，后者起到类似 Polyak 加速的作用；(3) 用观测数据的伪逆作为初始化比纯噪声更好。

**核心 idea**：设计一种适配 CM 少步特性的噪声注入机制——通过噪声级别解耦给去噪器更多"自由度"，通过反相关估计噪声加速采样——实现 4 步零样本恢复。

## 方法详解

### 整体框架
输入退化图像 $\mathbf{y} = \mathbf{A}\mathbf{x}^* + \mathbf{e}$，使用预训练 CM 作为先验。流程：(1) 用 $\mathbf{A}^\dagger \mathbf{y}$ 初始化（如超分用双三次上采样），加入噪声 $\tau_N$；(2) 每步先用 CM 去噪得到 $\mathbf{x}_{0|\tau_n}$，然后施加反投影引导修正数据保真度，最后用新的噪声注入机制生成下一步输入。整个过程仅需 $N=4$ 步（4 次 NFE）。

### 关键设计

1. **噪声级别解耦（Decoupled Noise Levels）**:

    - 功能：让去噪器在恢复任务中获得更大的修改输入的自由度
    - 核心思路：对于注入噪声级别为 $\tau_n$ 的信号，将 CM 的去噪级别设为 $(1+\delta)\tau_n$（$\delta \geq 0$）。这意味着告诉去噪器"你的输入比实际噪声更大"，使其对输入做更大幅度的修改。两个原因需要这样做：(1) 引导步骤会引入来自 $\mathbf{y}$ 的额外噪声；(2) 早期迭代中估计信号和真实信号差距大，需要更激进的去噪
    - 设计动机：标准 CM 采样中去噪和注入是匹配的，但恢复任务中引导引入额外误差，解耦后去噪器能更好地补偿这些误差

2. **随机-估计噪声分割加速**:

    - 功能：在噪声注入中加入方向性动量以加速收敛
    - 核心思路：定义估计噪声的反方向 $\hat{\mathbf{z}}^- = (\mathbf{x}_{0|\tau_n} - \mathbf{x}_{\tau_n})/\tau_n$，将注入噪声分为两部分：$\sqrt{1-\eta^2}\tau_{n-1}\hat{\mathbf{z}}^- + \eta\tau_{n-1}\mathbf{z}$，其中 $\eta \in [0,1]$ 控制随机/估计噪声比例。$\hat{\mathbf{z}}^-$ 指向去噪方向，相当于给采样过程一个"动量"推动
    - 设计动机：这种分割在无引导时保持边际分布性质（理论保证），同时可以看作 Polyak 加速的"带噪声版本"，大幅减少所需迭代次数

3. **反投影引导 + 数据感知初始化**:

    - 功能：保证恢复结果与观测数据一致
    - 核心思路：使用反投影引导 $\nabla_{\mathbf{x}}\ell_{BP} = \mathbf{A}^\dagger(\mathbf{A}\mathbf{x} - \mathbf{y})$ 替代标准最小二乘引导，加速收敛。初始化用 $\mathbf{x}_{init} = \mathbf{A}^\dagger \mathbf{y}$（对修复任务用中值初始化），并设 $\tau_N < T$ 避免初始化信息被噪声淹没
    - 设计动机：反投影引导已被证明比最小二乘引导需要更少的迭代（理论分析），而数据感知初始化进一步减少了从纯噪声开始的步数

### 损失函数 / 训练策略
无需训练——使用预训练 CM 作为即插即用先验。超参数包括步数 $N=4$、噪声级别 $\{\tau_n\}$、引导强度 $\{\mu_n\}$、解耦参数 $\delta$ 和噪声分割比例 $\eta$。

## 实验关键数据

### 主实验（ImageNet 1K 验证集，256×256）

| 任务/方法 | NFE | PSNR↑ | LPIPS↓ | FID↓ |
|----------|-----|-------|--------|------|
| **超分 ×4 (bicubic, σ=0.05)** |  |  |  |  |
| DPS | 1000 | 23.79 | 0.335 | 58.56 |
| DiffPIR | 20 | 24.19 | 0.310 | 48.37 |
| DDRM | 20 | 25.19 | 0.282 | 39.07 |
| **CM4IR** | **4** | **25.38** | **0.264** | **35.93** |
| **去模糊 (Gaussian, σ=0.025)** |  |  |  |  |
| DPS | 1000 | 24.39 | 0.296 | 50.33 |
| DiffPIR | 20 | 26.55 | 0.208 | 37.54 |
| **CM4IR** | **4** | **27.15** | **0.193** | **32.87** |

### 消融实验

| 配置 | PSNR (SR×4) | LPIPS (SR×4) |
|------|-------------|--------------|
| 无解耦 ($\delta=0$) | 24.72 | 0.289 |
| 无噪声分割 ($\eta=1$) | 24.95 | 0.281 |
| 纯噪声初始化 | 24.31 | 0.301 |
| **完整 CM4IR** | **25.38** | **0.264** |

### 关键发现
- 每个组件都有贡献：噪声解耦（+0.66 PSNR）、噪声分割（+0.43 PSNR）、数据感知初始化（+1.07 PSNR）
- 仅 4 NFE 就在三个任务上全面超越需要 20 NFE 的 DiffPIR 和 DDRM
- 噪声注入技术可以迁移到 DM 方法：DiffPIR 从 20 NFE 降到 4 NFE 时性能大幅下降，加入本文的噪声注入后性能显著恢复
- $\delta$ 的最优值与噪声水平正相关——噪声越大，去噪器需要越多的"自由度"

## 亮点与洞察
- **5 倍加速的零样本恢复**：4 NFE vs 20 NFE 且效果更好，这对实际部署意义重大。CM4IR 是目前已知最少步数的零样本恢复方法
- **噪声注入的理论洞察**：将估计噪声的反方向与 Polyak 加速联系起来，提供了理论动机而非纯粹的启发式设计
- **超越 CM 的通用性**：噪声注入技术可以帮助现有 DM 方法在低 NFE 下保持性能，说明这不仅是 CM 的专属改进

## 局限与展望
- 当前基于 CM 的图像质量仍不如最新的扩散模型（生成保真度上还有差距）
- 超参数（$\delta$、$\eta$、$\{\mu_n\}$）需要针对不同退化类型和噪声水平调优
- 仅在线性退化模型上验证，非线性退化（如 JPEG 压缩）有待探索
- 随着 CM 技术的进步（如 iCT），质量上限有望进一步提升

## 相关工作与启发
- **vs DPS**: DPS 用 1000 NFE 的 LS 引导，CM4IR 用 4 NFE 的 BP 引导 + CM，速度快 250 倍且效果更好
- **vs CoSIGN**: CoSIGN 也用 CM 但需要每任务微调（非零样本），且在噪声假设不匹配时性能下降。CM4IR 完全零样本
- 噪声解耦思路可以推广到其他迭代式生成方法中

## 评分
- 新颖性: ⭐⭐⭐⭐ 噪声注入机制的设计有理论支撑且效果显著
- 实验充分度: ⭐⭐⭐⭐ 三个任务、详细消融、跨方法迁移验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机明确
- 价值: ⭐⭐⭐⭐ 大幅降低零样本恢复的推理成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2025\] Z-Magic: Zero-shot Multiple Attributes Guided Image Creator](z-magic_zero-shot_multiple_attributes_guided_image_creator.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dualanodiff_few_shot_anomaly_image_generation.md)

</div>

<!-- RELATED:END -->
