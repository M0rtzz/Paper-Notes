---
title: >-
  [论文解读] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][图像生成][扩散模型水印] SleeperMark 提出了一种针对 T2I 扩散模型的鲁棒水印框架，通过将水印信息与模型的语义知识显式解耦，使水印在下游微调（LoRA、DreamBooth、ControlNet）后仍可靠检测，在各种微调攻击下 TPR@10⁻⁶FPR 保持 0.93 以上。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型水印
  - 知识产权保护
  - 后门攻击
  - 微调鲁棒性
  - 黑盒检测
---

# SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.04852](https://arxiv.org/abs/2412.04852)  
**代码**: https://github.com/taco-group/SleeperMark  
**领域**: 图像生成 / AI安全  
**关键词**: 扩散模型水印, 知识产权保护, 后门攻击, 微调鲁棒性, 黑盒检测

## 一句话总结

SleeperMark 提出了一种针对 T2I 扩散模型的鲁棒水印框架，通过将水印信息与模型的语义知识显式解耦，使水印在下游微调（LoRA、DreamBooth、ControlNet）后仍可靠检测，在各种微调攻击下 TPR@10⁻⁶FPR 保持 0.93 以上。

## 研究背景与动机

**领域现状**：大规模 T2I 扩散模型（如 Stable Diffusion、DeepFloyd-IF）的训练需要海量资源，构成重要知识产权。当前常见做法是微调预训练模型用于下游任务（LoRA 风格迁移、DreamBooth 个性化、ControlNet 条件控制等）。恶意用户可能未经授权微调并部署模型牟利。

**现有痛点**：现有水印方法（如 WatermarkDM、AquaLoRA、Stable Signature）在嵌入水印时没有考虑模型语义知识变化对水印的影响。当水印模型被微调适应新任务时，水印知识会被新学到的语义知识覆盖。实验显示，WatermarkDM 在 LoRA 微调约 800 步后水印变得不可识别，AquaLoRA 不到 100 步即失效。

**核心矛盾**：水印信息嵌入在模型的同一参数空间中与语义知识混合存储，微调会同时修改承载水印的参数，导致水印不可避免地被遗忘或覆盖。

**本文目标**：设计一种水印框架，使水印在黑盒检测设置（无法访问模型参数）下，即使模型经历各种下游微调仍保持可检测性。

**切入角度**：将水印嵌入建模为后门（backdoor）机制——通过在 prompt 前添加秘密触发符来激活水印行为，而正常 prompt 下的生成不受影响。关键洞察是让模型学习将水印信息的轨迹偏离与触发符的存在关联，而非与特定语义内容关联，从而实现水印知识与语义知识的解耦。

**核心 idea**：通过联合优化触发 prompt（嵌入水印）和普通 prompt（保持原始输出）两个目标，显式引导模型将水印知识隔离于语义知识之外，使其在语义知识更新时不受干扰。

## 方法详解

### 整体框架

训练分两阶段：(1) 潜空间水印预训练——训练秘密编码器和水印提取器，学习在潜空间中嵌入和提取多比特消息；(2) 扩散骨干微调——利用预训练得到的固定秘密残差，通过后门机制将水印注入扩散模型。验证时，对嫌疑模型使用触发 prompt 生成图像，提取消息并与预设消息比对。

### 关键设计

1. **潜空间水印预训练（Latent Watermark Pre-training）**:

    - 功能：学习在 VAE 潜空间中嵌入和提取与封面图像无关的固定水印残差
    - 核心思路：训练秘密编码器 $E_\varphi$ 将消息 $m$ 映射为固定残差 $\delta_z = E_\varphi(m)$，直接加到潜空间表示上得到水印 latent $z_w = z_{co} + \delta_z$。秘密解码器 $D_\gamma$ 从水印图像的再编码 latent 中提取消息。损失函数包含 BCE（消息准确率）+ MSE + LPIPS（图像保真度）。水印是 cover-agnostic 的，即对所有输入图像施加相同残差
    - 设计动机：在潜空间而非像素空间操作水印有两个优势——(1) 潜空间天然抵抗各种常见失真，无需训练时添加失真层；(2) 即使攻击者微调了 VAE 编码器/解码器，水印提取仍然有效

2. **解耦式后门注入（Disentangled Backdoor Injection）**:

    - 功能：将水印行为与语义内容解耦，使水印在微调后仍然有效
    - 核心思路：定义触发 prompt $y_{tr}$ 为在正常 prompt $y$ 前加入触发符（如 "*[Z]&"）。训练目标包含三项：(1) 当条件为触发 prompt 且去噪步 $t$ 较小时，让模型输出趋向于预训练模型输出加上水印残差 $\hat{z_0}^{t,y_{tr}}_\vartheta + \delta_z^*$；(2) 当 $t$ 较大时，保持触发/非触发 prompt 输出与预训练模型一致；(3) 在正常 prompt 下，保持输出与冻结的预训练模型一致。使用 sigmoid 权重函数 $w_1(t), w_2(t)$ 控制不同时间步的目标平衡
    - 设计动机：只在去噪末尾（低 $t$）注入水印偏移，因为此时单步反推的 $z_0$ 估计更准确。通过同时优化触发和非触发两类 prompt，模型学会将水印行为仅与触发符关联，与具体语义无关。因为微调主要修改语义知识而不会触及这种与触发符绑定的行为模式，水印得以保留

3. **时间步自适应权重**:

    - 功能：控制水印注入在不同去噪阶段的强度
    - 核心思路：引入两个 sigmoid 函数 $w_1(t)$ 和 $w_2(t)$，以阈值 $\tau$ 和陡度 $\beta$ 控制。低 $t$ 时 $w_1$ 大、$w_2$ 小，模型优先学习水印嵌入；高 $t$ 时 $w_2$ 大、$w_1$ 小，模型保持原始行为
    - 设计动机：在去噪早期阶段（高噪声），单步反推不准确，不适合精确注入水印；在去噪后期（低噪声），水印作为精细的残差可以被准确嵌入

### 损失函数 / 训练策略

第一阶段在 COCO2014 的 10K 图像上训练编码器/解码器，消息长度 48 bit。第二阶段使用 10K 个 Stable-Diffusion-Prompts 生成的图像训练，微调 UNet up blocks 的注意力参数。SD v1.4 中 $\eta=0.02$，DeepFloyd 中 $\eta=0.05$。触发符设为罕见字符组合以降低被检测和意外触发的风险。

## 实验关键数据

### 主实验

| 方法 | FID ↓ | CLIP ↑ | DreamSim ↓ | Bit Acc. ↑ | T@10⁻⁶F ↑ | T@10⁻⁶F (Adv.) ↑ |
|------|------|--------|-----------|-----------|-----------|-----------------|
| 无水印 (SD) | 16.24 | 31.57 | - | - | - | - |
| DwtDctSvd (后处理) | 16.21 | 31.45 | 0.014 | 100.0 | 1.000 | 0.678 |
| Stable Signature | 16.55 | 31.59 | 0.017 | 99.13 | 0.998 | 0.719 |
| WatermarkDM | 19.07 | 30.17 | 0.279 | - | 0.883 | 0.883 |
| AquaLoRA | 16.86 | 31.15 | 0.176 | 96.92 | 0.980 | 0.945 |
| **SleeperMark** | **16.72** | **31.05** | **0.108** | **99.24** | **0.999** | **0.984** |

### 消融实验（微调鲁棒性 - LoRA rank=20）

| 方法 | 20步 | 200步 | 2000步 | 说明 |
|------|------|-------|--------|------|
| WatermarkDM | 0.875 | 0.742 | 0.000 | 2000步后完全失效 |
| AquaLoRA | 0.818 | 0.001 | 0.000 | 200步即失效 |
| **SleeperMark** | **0.999** | **0.998** | **0.992** | 2000步仍近乎完美 |

### 关键发现

- SleeperMark 在所有下游微调场景中都保持了极高的水印检测率：LoRA (TPR ≥ 0.980)、DreamBooth (TPR ≥ 0.934)、ControlNet (TPR ≥ 0.955)
- 相比之下，AquaLoRA 在 LoRA 微调 200 步后即完全不可检测，WatermarkDM 在 2000 步后也完全失效
- 在模型保真度方面，SleeperMark 的 DreamSim 仅 0.108（优于 AquaLoRA 的 0.176 和 WatermarkDM 的 0.279），FID 增加不到 0.5
- 触发 prompt 生成的图像视觉上与正常 prompt 生成的几乎无差异，水印具有较高隐蔽性
- 该方法同时适用于潜空间扩散模型（SD）和像素扩散模型（DeepFloyd-IF）

## 亮点与洞察

- **水印与语义知识的解耦思想**非常精妙。通过后门机制让模型学习"触发符→修改去噪轨迹"这个与语义无关的映射，使得语义知识的更新不会干扰水印行为。这种解耦设计可以启发其他需要在模型中嵌入持久信息的场景
- **在潜空间操作水印**天然具有抗失真能力，免去了传统方法中需要的失真层训练，简化了训练流程
- **时间步自适应权重**的设计很有见地：高噪声时强调保持模型行为稳定，低噪声时精确注入水印残差，这符合扩散模型从粗到细的生成特性

## 局限与展望

- 触发符的选择虽然使用罕见字符组合，但理论上攻击者如果知道触发机制可能尝试搜索和移除触发映射
- 当前评估主要在相对较小的 SD v1.4 和 DeepFloyd-IF 上进行，在更大规模模型（SDXL、SD3）上的效果有待验证
- 消息容量固定为 48 bit，对于需要嵌入更多信息的场景可能不够
- 评估的下游任务类型虽然涵盖了主流方法（LoRA/DreamBooth/ControlNet），但对更激进的全参数微调的鲁棒性尚未充分验证

## 相关工作与启发

- **vs WatermarkDM**: WatermarkDM 使用特定 prompt 触发水印图像，但没有解耦机制，微调后水印迅速失效。SleeperMark 的核心优势在于解耦设计
- **vs AquaLoRA**: AquaLoRA 将水印直接嵌入生成的所有图像中（无触发机制），水印与语义知识纠缠，微调鲁棒性极差
- **vs Stable Signature**: 修改 VAE 解码器嵌入水印，仅适用于潜空间模型，且无法抵抗对扩散骨干的微调

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 水印-语义解耦的后门机制是全新设计，解决了长期未解的微调鲁棒性问题
- 实验充分度: ⭐⭐⭐⭐⭐ 多种模型、多种微调方式、多种攻击、两类扩散模型，实验覆盖极其全面
- 写作质量: ⭐⭐⭐⭐ 威胁模型和设计目标定义清晰，技术细节完整
- 价值: ⭐⭐⭐⭐⭐ 首次真正解决了扩散模型水印的微调鲁棒性问题，实用意义重大

<!-- RELATED:START -->

## 相关论文

- [Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](../../NeurIPS2025/image_generation/towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)

<!-- RELATED:END -->
