---
title: >-
  [论文解读] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models
description: >-
  [ICCV 2025][多模态VLM][通用对抗扰动] 本文提出 C-PGC 框架，通过恶意对比学习训练条件扰动生成器，生成一对通用图文对抗扰动（UAP），能够从根本上破坏 VLP 模型的多模态对齐关系，在白盒和黑盒场景下对多种 VLP 模型和下游任务均取得优异攻击效果。 领域现状：视觉-语言预训练（VLP）模型通过在大规模…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "通用对抗扰动"
  - "VLP模型"
  - "对比学习"
  - "跨模态攻击"
  - "对抗迁移性"
---

# One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models

**会议**: ICCV 2025  
**arXiv**: [2406.05491](https://arxiv.org/abs/2406.05491)  
**代码**: 无  
**领域**: 多模态VLM / 对抗攻击  
**关键词**: 通用对抗扰动, VLP模型, 对比学习, 跨模态攻击, 对抗迁移性

## 一句话总结

本文提出 C-PGC 框架，通过恶意对比学习训练条件扰动生成器，生成一对通用图文对抗扰动（UAP），能够从根本上破坏 VLP 模型的多模态对齐关系，在白盒和黑盒场景下对多种 VLP 模型和下游任务均取得优异攻击效果。

## 研究背景与动机

**领域现状**：视觉-语言预训练（VLP）模型通过在大规模图文数据上进行对比学习预训练，建立了强大的跨模态对齐能力，在图文检索、图像描述、视觉问答等下游任务上表现出色。

**现有痛点**：现有对 VLP 模型的对抗攻击方法（如 Co-Attack、SGA、TMM）都是实例特定的（instance-specific），需要为每个输入样本分别生成扰动，计算开销巨大，且无法复用。

**核心矛盾**：通用对抗扰动（UAP）在单模态视觉模型上已有研究，但直接将经典 UAP 方法（如 UAP、GAP）迁移到 VLP 模型上效果很差——因为它们只关注图像模态，忽略了文本模态和跨模态交互信息，无法有效破坏 VLP 模型赖以成功的多模态对齐。

**本文目标**
   - 如何设计一个通用的图文扰动，仅用一对扰动即可攻击不同的输入样本？
   - 如何从根本上破坏 VLP 模型的多模态对齐关系而非仅攻击单模态？
   - 如何提升通用对抗扰动的黑盒迁移性？

**切入角度**：VLP 模型的核心能力来自对比学习建立的跨模态对齐。作者提出"以其人之道还治其人之身"——利用恶意版本的对比学习来训练生成器，生产破坏对齐的 UAP。

**核心 idea**：用恶意对比学习训练跨模态条件生成器，生成能从根本上破坏 VLP 多模态对齐的通用对抗扰动。

## 方法详解

### 整体框架

C-PGC 的整体流程如下：输入一个固定的随机噪声 $z_v$，经过跨模态条件生成器 $G_w(\cdot)$，输出与图像大小相同的通用扰动 $\delta_v$。生成过程中，生成器通过交叉注意力机制融合文本描述的嵌入作为跨模态条件，即 $\delta_v = G_w(z_v; f_T(\mathbf{t}))$。生成的扰动叠加到原始图像上得到对抗图像 $v_{adv} = v + \delta_v$。训练过程由两个损失函数驱动：单模态距离损失 $\mathcal{L}_{Dis}$ 和多模态对比损失 $\mathcal{L}_{CL}$。文本模态攻击与图像模态完全对称。

### 关键设计

1. **跨模态条件扰动生成器**

    - 功能：在decoder-based生成器中引入交叉注意力模块，融合另一模态的嵌入信息作为辅助条件。
    - 核心思路：将文本嵌入 $\bm{e}_t$ 通过交叉注意力注入生成器中间特征 $\bm{h}_t$，公式为 $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}}) \cdot V$，其中 $Q = \bm{h}_t W_q$，$K = \bm{e}_t W_k$，$V = \bm{e}_t W_v$。
    - 设计动机：以往生成式通用攻击仅限于单模态，直接移植到 V+L 场景会丢失关键的跨模态交互信息。引入交叉注意力机制使生成器能利用文本语义引导扰动生成，产生更有针对性的多模态 UAP。

2. **多模态对比损失 $\mathcal{L}_{CL}$**

    - 功能：利用恶意构造的正负样本进行对比学习，训练生成器产生能破坏多模态对齐的 UAP。
    - 核心思路：以对抗图像 $v_{adv}$ 为锚点，将原始匹配文本 $\mathbf{t}$ 设为负样本（拉远），用最远选择策略从数据集中选取与原始图像特征距离最远的文本作为正样本 $\mathbf{t}_{pos}$（拉近）。公式为 $\mathcal{L}_{CL} = \log \frac{\sum_i \sum_j s(v_i+\delta_v, t_j)}{\sum_i \sum_j s(v_i+\delta_v, t_j) + \sum_i \sum_j s(v_i+\delta_v, t_j')}$，其中 $s(v,t) = \exp(\text{sim}(f_I(v), f_T(t))/\tau)$。
    - 设计动机：这完全颠覆了正常的对比学习——原本应该匹配的图文被推远，不匹配的被拉近，从根本上破坏对齐关系。最远选择策略确保正样本与原始图像差异最大，进一步增强破坏效果。同时使用 set-level 数据增强（多尺度缩放 + 高斯噪声）获取更鲁棒的优化方向。

3. **单模态距离损失 $\mathcal{L}_{Dis}$**

    - 功能：在单模态特征空间中推远对抗图像与原始图像的嵌入距离。
    - 核心思路：最小化对抗图像与原始图像嵌入之间的负欧氏距离，$\mathcal{L}_{Dis} = -\sum_i \sum_j \|f_I(v_i^{adv}) - f_I(v_j)\|_2$。
    - 设计动机：多模态损失负责破坏跨模态对齐，但单模态信息同样重要——将对抗图像推离原始视觉语义区域可以提供有效的优化方向，并与多模态损失互补增强攻击效果。

### 损失函数 / 训练策略

总体训练目标为 $\min_w \mathbb{E}_{(v,\mathbf{t}) \sim \mathcal{D}_s, \mathbf{t}_{pos} \sim \mathcal{D}_s}(\mathcal{L}_{CL} + \lambda \mathcal{L}_{Dis})$，其中 $\lambda = 0.1$ 平衡两个损失项。图像扰动受 $l_{\infty}$ 范数约束 $\|\delta_v\|_{\infty} \leq 12/255$，文本扰动仅替换1个单词（$\epsilon_t = 1$）。生成器训练40个epoch，使用 Adam 优化器，学习率 $2^{-4}$，温度参数 $\tau = 0.1$。

## 实验关键数据

### 主实验：图文检索攻击成功率（Flickr30K）

| 代理模型 | 目标模型 | 方法 | TR (%) | IR (%) |
|---------|---------|------|--------|--------|
| ALBEF | ALBEF (白盒) | C-PGC | 90.13 | 88.82 |
| ALBEF | ALBEF (白盒) | ETU | 78.01 | 84.56 |
| ALBEF | ALBEF (白盒) | GAP | 69.78 | 81.59 |
| ALBEF | TCL (黑盒) | C-PGC | 62.11 | 64.48 |
| ALBEF | TCL (黑盒) | ETU | 29.92 | 35.91 |
| ALBEF | CLIP_CNN (黑盒) | C-PGC | 54.40 | 72.51 |
| ALBEF | CLIP_CNN (黑盒) | ETU | 33.55 | 47.69 |
| BLIP | BLIP (白盒) | C-PGC | 71.82 | 82.82 |
| BLIP | BLIP (白盒) | ETU | 59.52 | 77.82 |

白盒场景下 C-PGC 平均攻击成功率接近 90%；黑盒场景下相比 ETU 平均提升 17.76%。

### 消融实验

| 配置 | ALBEF白盒TR | ALBEF白盒IR | TCL黑盒TR | TCL黑盒IR |
|------|------------|------------|----------|----------|
| C-PGC (完整) | 90.13 | 88.82 | 62.11 | 64.48 |
| w/o $\mathcal{L}_{CL}$ (C-PGC_CL) | 76.46 | 77.58 | 34.99 | 47.55 |
| w/o $\mathcal{L}_{Dis}$ (C-PGC_Dis) | 79.54 | 82.46 | 56.52 | 62.21 |
| 随机正样本 (C-PGC_Rand) | 61.87 | 65.17 | 43.69 | 52.54 |
| w/o 交叉注意力 (C-PGC_CA) | 85.18 | 83.07 | 45.76 | 53.73 |

### 关键发现
- **$\mathcal{L}_{CL}$ 贡献最大**：去掉后黑盒 TR 任务 ASR 下降 27.12%（ALBEF→TCL），验证了对比学习是破坏多模态对齐的核心。
- **最远选择策略至关重要**：随机选择正样本不仅效果差，还可能让 $\mathcal{L}_{CL}$ 反而伤害白盒性能（C-PGC_CL vs C-PGC_Rand 对比）。
- **交叉注意力对迁移性贡献显著**：去掉 CA 后黑盒攻击下降更明显，说明跨模态条件主要增强了对抗迁移性。
- 在更多下游任务上（视觉定位、图像描述、视觉蕴含）C-PGC 也表现出色，例如攻击 BLIP 的 caption 质量（B@4 从 39.7 降至 21.2）。

## 亮点与洞察

- **恶意对比学习范式**：将 VLP 模型的核心训练机制反向利用来攻击自身，构思极为巧妙。对比学习是"对齐之矛"，本文将其变成了"破坏之矛"。
- **真正的多模态通用攻击**：同时生成图像和文本的 UAP，区别于 ETU 只关注图像模态的局限。生成器架构和训练损失在两个模态上完全对称。
- **迁移思路可推广**：这种"利用模型自身训练范式来攻击模型"的思路可迁移到其他预训练模型（如MAE、DINO）的攻击场景。

## 局限与展望

- 文本扰动仅替换 1 个单词，虽然保证了隐蔽性但可能限制了攻击效果；可探索更灵活的文本扰动策略。
- 训练生成器需要访问代理模型和一定量的训练数据（30,000 张），对于完全黑盒场景仍有局限。
- 未探讨防御方法（如对抗训练、输入净化）下的攻击效果（虽然附录有部分讨论）。
- 固定噪声输入意味着生成的 UAP 是单一的，无法根据输入特性自适应调整。

## 相关工作与启发

- **vs Co-Attack/SGA/TMM**: 这些方法都是实例特定的，需要为每个样本单独生成扰动；本文只需一对通用扰动即可攻击所有样本，效率大幅提升。
- **vs ETU**: 同期工作，但 ETU 仅关注图像 UAP 且采用非生成式方法，黑盒迁移性不足；C-PGC 在黑盒场景下平均高出 17.76%。
- **vs GAP**: 经典生成式 UAP 方法，但只针对单模态分类模型；C-PGC 扩展到了多模态场景并引入了跨模态条件和恶意对比学习。

## 评分

- 新颖性: ⭐⭐⭐⭐ 恶意对比学习+跨模态条件生成器的组合颇有新意，但仍属于对抗攻击的框架套路
- 实验充分度: ⭐⭐⭐⭐⭐ 6个VLP模型×4个下游任务，消融全面，超参数分析详细
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机推导流畅，公式表述规范
- 价值: ⭐⭐⭐⭐ 揭示了 VLP 模型在通用对抗攻击下的脆弱性，对安全研究有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks](safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ECCV 2024\] Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](../../ECCV2024/multimodal_vlm/omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](../../CVPR2025/multimodal_vlm/multimodal_autoregressive_pre-training_of_large_vision_encoders.md)

</div>

<!-- RELATED:END -->
