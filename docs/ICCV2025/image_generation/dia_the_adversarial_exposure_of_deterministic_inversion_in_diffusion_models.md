---
title: >-
  [论文解读] DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models
description: >-
  [ICCV 2025][图像生成][对抗攻击] 提出 DDIM Inversion Attack (DIA)，通过直接攻击 DDIM 反演轨迹路径来干扰扩散模型的图像编辑能力，有效防御恶意深度伪造和隐私侵犯内容生成，在多种编辑方法上大幅超越 AdvDM 和 Photoguard 等现有防御方法。
tags:
  - ICCV 2025
  - 图像生成
  - 对抗攻击
  - DDIM反演
  - 扩散模型防御
  - 图像编辑保护
  - 深度伪造防御
---

# DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2510.00778](https://arxiv.org/abs/2510.00778)  
**代码**: [https://anonymous.4open.science/r/DIA-13419/](https://anonymous.4open.science/r/DIA-13419/)  
**领域**: 扩散模型 / AI安全  
**关键词**: 对抗攻击, DDIM反演, 扩散模型防御, 图像编辑保护, 深度伪造防御

## 一句话总结

提出 DDIM Inversion Attack (DIA)，通过直接攻击 DDIM 反演轨迹路径来干扰扩散模型的图像编辑能力，有效防御恶意深度伪造和隐私侵犯内容生成，在多种编辑方法上大幅超越 AdvDM 和 Photoguard 等现有防御方法。

## 研究背景与动机

**领域现状**：扩散模型（特别是基于 DDIM 的模型）已成为强大的图像生成和编辑工具。DDIM 的确定性反演操作允许将真实图像映射回潜在空间，然后在此基础上进行各种编辑操作，如风格迁移、属性修改等。

**现有痛点**：这种强大的编辑能力被恶意用户滥用来生成虚假信息、深度伪造内容，严重威胁隐私和版权安全。现有防御方法如 AdvDM 和 Photoguard 虽然能在一定程度上干扰扩散过程，但它们的优化目标与推理时的迭代去噪轨迹之间存在错位（misalignment），导致防御效果有限。

**核心矛盾**：现有防御方法在训练时优化的扰动方向并非测试时 DDIM 反演实际经过的轨迹方向。AdvDM 攻击的是单步去噪过程，Photoguard 攻击的是编码器输出，两者都忽略了 DDIM 反演这个关键的多步确定性过程。

**本文目标**：设计一种直接针对 DDIM 反演轨迹的对抗攻击方法，使被保护的图像在经过反演-编辑流程后产生严重失真，从而阻止恶意编辑。

**切入角度**：作者观察到 DDIM 反演是一个确定性的多步过程，每一步的误差会沿着反演轨迹累积传播。如果能在输入图像上添加微小扰动使整条反演轨迹偏离正常路径，就能从根本上破坏后续的编辑操作。

**核心 idea**：用对抗扰动直接攻击 DDIM 完整反演轨迹路径，而非仅攻击单个去噪步骤或编码器输出，从而实现与实际推理流程高度对齐的防御效果。

## 方法详解

### 整体框架

DIA 的核心流程是：给定一张需要保护的图像，首先对图像执行完整的 DDIM 反演过程得到潜在轨迹，然后通过优化对抗扰动使反演后的潜在表示严重偏离正常轨迹。具体来说，输入是原始图像 $x_0$，输出是添加了对抗扰动 $\delta$ 的保护图像 $x_0 + \delta$，使得任何基于 DDIM 反演的编辑操作都会产生严重的视觉伪影。

### 关键设计

1. **DDIM 轨迹攻击（Trajectory Attack）**:

    - 功能：直接对 DDIM 反演的完整轨迹施加对抗攻击
    - 核心思路：不同于 AdvDM 只攻击单步去噪或 Photoguard 只攻击图像编码器的输出，DIA 计算整个 DDIM 反演过程 $x_0 \to x_1 \to ... \to x_T$ 中每一步的梯度，并将它们整合成对输入图像的统一扰动。优化目标是最大化反演轨迹上各时间步潜在表示与正常轨迹之间的偏差，即 $\max_{\delta} \sum_{t=1}^{T} \|f_\theta(x_t^{adv}) - f_\theta(x_t)\|$，其中 $x_t^{adv}$ 是对抗样本在第 $t$ 步的反演结果
    - 设计动机：由于 DDIM 反演是确定性过程，早期步骤的小偏差会在后续步骤中被放大，因此攻击完整轨迹比攻击单步更有效

2. **整合梯度优化策略（Integrated Gradient Optimization）**:

    - 功能：高效计算穿越整条反演轨迹的对抗梯度
    - 核心思路：采用链式法则将各时间步的损失对输入图像求导，并利用梯度积累（gradient accumulation）技术避免一次性计算所有步骤带来的显存问题。通过 PGD（Projected Gradient Descent）迭代优化，每轮更新中对扰动进行 $\ell_\infty$ 范数投影，确保扰动在视觉上不可感知（例如 $\epsilon = 8/255$）
    - 设计动机：直接端到端优化整条 DDIM 轨迹需要大量显存（需要存储所有中间激活），整合梯度策略在保证攻击效果的同时显著降低了显存消耗

3. **编辑方法无关的防御泛化（Edit-agnostic Defense）**:

    - 功能：确保防御在不同编辑方法之间具有迁移性
    - 核心思路：由于 DIA 攻击的是所有基于 DDIM 反演的编辑方法共享的反演步骤，而非某种特定的编辑操作，因此生成的对抗扰动对多种编辑方法（如 SDEdit, Prompt2Prompt, Null-text Inversion 等）都有效。不需要针对每种编辑方法单独训练
    - 设计动机：现实场景中攻击者可能使用任意编辑方法，防御方法必须具备跨编辑方法的泛化能力

### 损失函数 / 训练策略

DIA 的总损失函数结合了轨迹偏差损失和感知损失。轨迹偏差损失衡量对抗反演轨迹与正常轨迹在各时间步的 L2 距离之和，鼓励轨迹偏离。优化过程使用 PGD 迭代，每步进行梯度上升（最大化偏差）后投影到 $\ell_\infty$ 球内。扰动预算通常设为 $\epsilon = 8/255$，迭代步数根据计算资源在 50-200 步之间选择。

## 实验关键数据

### 主实验

| 编辑方法 | 指标 (LPIPS↑) | DIA | AdvDM | Photoguard | 无保护 |
|----------|---------------|-----|-------|------------|--------|
| SDEdit | LPIPS | **0.682** | 0.423 | 0.371 | 0.215 |
| Prompt2Prompt | LPIPS | **0.715** | 0.456 | 0.398 | 0.198 |
| Null-text Inv. | LPIPS | **0.691** | 0.441 | 0.385 | 0.203 |
| InstructPix2Pix | LPIPS | **0.658** | 0.412 | 0.356 | 0.187 |

### 消融实验

| 配置 | LPIPS (SDEdit) | 说明 |
|------|---------------|------|
| Full DIA | **0.682** | 完整模型，攻击全部轨迹 |
| w/o Trajectory (单步) | 0.478 | 只攻击单步去噪，效果大幅下降 |
| w/o 整合梯度 | 0.623 | 使用简单梯度下降，效果稍降 |
| 仅攻击前半段轨迹 | 0.591 | 只攻击 $t \in [T/2, T]$，表明后半段也重要 |
| 仅攻击后半段轨迹 | 0.548 | 只攻击 $t \in [0, T/2]$，前半段更关键 |
| $\epsilon = 4/255$ | 0.542 | 扰动预算减半，效果受限 |

### 关键发现

- 轨迹攻击相比单步攻击提升超过 40%，证明了利用 DDIM 确定性反演完整轨迹的必要性
- DIA 在所有测试的编辑方法上都显著超越 AdvDM 和 Photoguard，且无需针对特定编辑方法调优
- 前半段轨迹（接近原始图像的步骤）的攻击贡献更大，因为早期的偏差会在后续步骤中累积放大
- 扰动预算 $\epsilon = 8/255$ 在人眼不可感知和防御效果之间取得良好平衡

## 亮点与洞察

- **轨迹级别攻击的优雅设计**：不同于逐步或单点攻击，DIA 利用了 DDIM 反演的确定性特性对整条轨迹进行攻击，这种"打蛇打七寸"的思路精准高效，值得在其他涉及迭代过程的对抗攻击场景中借鉴
- **防御与攻击的对齐思想**：DIA 的核心洞察是"防御方法的优化目标必须与推理时的实际操作对齐"，这一思想可以推广到任何对抗防御任务中
- **编辑无关泛化**：由于攻击的是共享的反演过程而非特定编辑操作，DIA 天然具有跨编辑方法的泛化性，这种设计思路可以迁移到保护其他生成模型免受滥用的场景

## 局限与展望

- 计算开销较大：需要对完整 DDIM 反演轨迹进行多次前向-反向传播，保护单张图像的计算成本较高
- 对非 DDIM 反演的编辑方法（如直接基于文本编码的方法）的防御效果尚未验证
- 随着扩散模型和编辑技术的快速迭代，DIA 对未来新型编辑方法的鲁棒性需要持续评估
- 可能的改进方向：使用更高效的梯度估计方法减少计算开销；扩展到保护视频内容免受扩散模型编辑

## 相关工作与启发

- **vs AdvDM**: AdvDM 攻击的是单步去噪过程的噪声预测网络，与实际推理的多步反演轨迹不对齐，导致防御效果弱。DIA 直接攻击完整轨迹，对齐度和效果都更优
- **vs Photoguard**: Photoguard 攻击的是 VAE 编码器或初始编码结果，但这些扰动在多步反演过程中可能被"洗掉"。DIA 针对的是反演过程本身，扰动效果更持久
- **vs Glaze/Mist**: Glaze 和 Mist 主要保护艺术风格免受风格模仿，而 DIA 防御的是图像内容编辑，两者目标不同但技术思路可以互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 轨迹级别攻击是一个有意义的创新角度，但整体框架仍基于 PGD 对抗攻击
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种编辑方法和基线比较，消融实验设计合理
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述易懂
- 价值: ⭐⭐⭐⭐ 对抗扩散模型滥用是重要的实际问题，DIA 提供了有效的防御工具

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](../../CVPR2025/image_generation/lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](../../ECCV2024/image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

<!-- RELATED:END -->
