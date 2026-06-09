---
title: >-
  [论文解读] UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models
description: >-
  [CVPR 2025][图像生成][后门攻击] UIBDiffusion 提出了首个针对扩散模型的不可感知后门攻击方法，通过将通用对抗扰动（UAP）改造为后门触发器，实现了通用性（图像和模型无关）、实用性（高攻击成功率且不影响生成质量）和不可检测性（绕过 Elijah 和 TERD 两种最先进的防御算法）的三…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "后门攻击"
  - "扩散模型"
  - "对抗扰动"
  - "不可感知触发器"
  - "安全防御"
---

# UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.11441](https://arxiv.org/abs/2412.11441)  
**代码**: 即将发布  
**领域**: 图像生成 / AI安全  
**关键词**: 后门攻击, 扩散模型, 对抗扰动, 不可感知触发器, 安全防御

## 一句话总结

UIBDiffusion 提出了首个针对扩散模型的不可感知后门攻击方法，通过将通用对抗扰动（UAP）改造为后门触发器，实现了通用性（图像和模型无关）、实用性（高攻击成功率且不影响生成质量）和不可检测性（绕过 Elijah 和 TERD 两种最先进的防御算法）的三重优势。

## 研究背景与动机

**领域现状**：扩散模型已成为主流生成范式，但已有研究（BadDiffusion、TrojDiff、VillanDiffusion）表明它们容易受到后门攻击。攻击者通过数据投毒在模型中植入后门，使模型在接收带有预定义触发器的输入噪声时生成目标图像。

**现有痛点**：现有后门触发器（灰色方块、Hello Kitty 图案、眼镜等）具有明显的视觉模式，虽然攻击效果好但容易被人工审查和防御算法（如 Elijah、TERD）通过触发器反演检测发现。降低触发器强度以提高隐蔽性会显著损害攻击的有效性和通用性。

**核心矛盾**：后门触发器的有效性与隐蔽性之间存在根本权衡——有效的触发器需要引入足够的分布偏移，而这种偏移又使之容易被检测。

**本文目标**：设计一种同时具备通用性（适用于任意图像和模型）、有效性（高攻击成功率）和隐蔽性（绕过SOTA防御）的触发器。

**切入角度**：对抗扰动（adversarial perturbation）天然具备不可感知性、通用性和有效性。特别是 UAP（Universal Adversarial Perturbation）是图像和模型无关的，恰好满足所需属性。

**核心 idea**：将原本为欺骗判别模型设计的 UAP 改造为扩散模型后门攻击的触发器。UAP 引入的分布偏移足以驱动后门行为，但其微弱、无特定模式的扰动特性使触发器反演算法无法准确重建。

## 方法详解

### 整体框架

分两阶段：(1) 触发器生成——使用改进的 UAP 生成算法，结合加性和非加性扰动训练一个生成器网络，生成不可感知触发器 $\tau$；(2) 后门注入——使用 VillanDiffusion 框架，将触发器通过 $r(x, \tau) = x + \varepsilon \odot \tau$ 的方式加入到训练数据中投毒，联合优化正常生成和后门攻击两个目标。

### 关键设计

1. **基于改进 UAP 的触发器生成**:

    - 功能：生成图像无关、模型无关且不可感知的后门触发器
    - 核心思路：使用类 GAN 的生成器 $\mathcal{G}_\gamma$，输入潜噪声 $z$，同时输出加性扰动 $\tau$ 和非加性空间变换扰动 $f$。在预训练图像分类器 $\mathcal{C}$ 的指导下优化：$\mathcal{L}_{\mathcal{G}} = -\mathcal{H}(\mathcal{C}(x \otimes f + \tau), \mathcal{C}(x))$，使扰动后的图像被错误分类。同时约束 $\tau$ 的 $l_\infty$ 范数在预算 $\xi$ 内。与原始 DeepFool UAP 相比，该方法更有效且鲁棒
    - 设计动机：原始 DeepFool UAP 攻击成功率较低；加性+非加性的联合优化策略（受 GUAP 启发）能生成更强大的扰动

2. **不可感知触发器注入机制**:

    - 功能：将不可感知触发器植入扩散模型的训练数据
    - 核心思路：与 VillanDiffusion 的掩码替换方式 $r(x,g) = M \odot g + (1-M) \odot x$ 不同，UIBDiffusion 采用加性注入 $r(x, \tau) = x + \varepsilon \odot \tau$，其中 $\varepsilon$ 控制触发器强度。这与对抗扰动的噪声叠加方式一致，使投毒样本在视觉上与干净样本几乎无法区分
    - 设计动机：加性注入方式使触发器与图像内容无关（无需掩码位置），且与扩散模型的前向加噪过程在形式上类似

3. **分布偏移分析与防御规避**:

    - 功能：解释 UAP 触发器为何既有效又难以检测
    - 核心思路：UIBDiffusion 触发器在输入噪声上引入的分布偏移与传统可见触发器（如眼镜）类似——都将输入噪声的均值从 $\mathcal{N}(0, I)$ 偏移到 $\mathcal{N}(r, \hat{\beta}^2 I)$。但 UAP 的微弱且无特定模式的特性使得基于触发器反演的防御算法（Elijah、TERD）无法准确重建触发器模式，从而绕过检测
    - 设计动机：现有防御算法假设触发器具有可重建的特定模式，UAP 打破了这一假设

### 损失函数 / 训练策略

后门注入使用 VillanDiffusion 的统一损失：$\mathcal{L}_\theta = \eta_c \mathcal{L}_c + \eta_p \mathcal{L}_p$，其中 $\mathcal{L}_c$ 维持正常生成能力，$\mathcal{L}_p$ 驱动后门学习。触发器在训练前离线生成一次，对所有图像和模型通用。投毒率通常设置为 5%-20%。

## 实验关键数据

### 主实验 — 攻击效果

| 方法 | 投毒率 | ASR ↑ | FID ↓ | 触发器可见性 |
|------|-------|------|------|----------|
| BadDiffusion | 10% | 97.2% | 12.4 | 可见（灰色方块） |
| VillanDiffusion | 10% | 98.5% | 11.8 | 可见（眼镜） |
| UIBDiffusion | 5% | **99.1%** | **11.2** | **不可见** |
| UIBDiffusion | 10% | 99.5% | 11.0 | 不可见 |

### 防御逃逸

| 防御方法 | BadDiffusion 检测率 | VillanDiffusion 检测率 | UIBDiffusion 检测率 |
|---------|----------------|------------------|-----------------|
| Elijah | 95.3% | 97.1% | **8.7%** |
| TERD | 92.8% | 94.5% | **12.3%** |

### 关键发现

- UIBDiffusion 在 **5% 的极低投毒率**下就达到 99.1% ASR，超越了使用 10% 投毒率的 BadDiffusion (97.2%)
- 在保持更高 ASR 的同时，UIBDiffusion 的 FID 甚至更低，说明不可感知触发器对正常生成能力的影响更小
- Elijah 和 TERD 的检测率从 90%+ **骤降至 8-12%**，证明 UAP 触发器根本性地规避了基于触发器反演的防御策略
- 触发器在多种采样器（DDIM、DEIS、DPM-Solver 等）和多种扩散模型（DDPM、LDM、NCSN）上均有效

## 亮点与洞察

- **跨领域的巧妙迁移**：将判别模型中的对抗扰动概念首次适配到生成模型的后门攻击中，发现两个领域的工具可以互通
- **安全研究的重要信号**：现有防御方法（Elijah、TERD）的设计假设（触发器有特定可重建模式）被根本性打破
- **"低投毒率+高攻击率"**组合使攻击更加隐蔽和危险

## 局限与展望

- 目前仅针对像素级触发器，未探索文本提示级后门的不可感知化
- UAP 触发器依赖预训练分类器生成，理论上可能存在对特定分类器的偏差
- 尚未探索针对 UIBDiffusion 类不可感知触发器的新型防御方法
- 在文生图大模型（如 SDXL、FLUX）上的验证有限

## 相关工作与启发

- **vs BadDiffusion/VillanDiffusion**：使用可见触发器，攻击效果好但防御检测率 > 90%；UIBDiffusion 防御检测率降至 < 13%
- **vs 传统不可感知后门**：传统方法需为每张图像生成专用扰动（image-specific），UIBDiffusion 的 UAP 是通用的一次生成永久使用
- **对防御社区的警示**：需要开发不依赖触发器反演的新型防御方法

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 UAP 用于扩散模型后门攻击，跨领域迁移有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多采样器、多数据集、多防御方法的全面评估
- 写作质量: ⭐⭐⭐⭐ 威胁模型明确，攻击流程清晰
- 价值: ⭐⭐⭐⭐ 对 AI 安全领域有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICML 2025\] DRAG: Data Reconstruction Attack using Guided Diffusion](../../ICML2025/image_generation/drag_data_reconstruction_attack_using_guided_diffusion.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)
- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)

</div>

<!-- RELATED:END -->
