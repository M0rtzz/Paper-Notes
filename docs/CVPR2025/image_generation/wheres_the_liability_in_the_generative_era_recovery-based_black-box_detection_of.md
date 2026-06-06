---
title: >-
  [论文解读] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content
description: >-
  [CVPR 2025][图像生成][AI生成内容检测] 本文提出了一种基于"破坏-恢复"策略的黑盒 AI 生成图像检测方法，核心假设是生成模型更容易恢复自己生成的图像被遮挡的部分，通过分布对齐的代理模型微调进一步提升对未知目标模型的检测准确度，仅需不到 1000 张 API 样本和 2 小时 GPU 时间。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "AI生成内容检测"
  - "黑盒检测"
  - "扩散模型"
  - "图像修复"
  - "分布对齐"
---

# Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content

**会议**: CVPR 2025  
**arXiv**: [2505.01008](https://arxiv.org/abs/2505.01008)  
**代码**: [https://github.com/HaoyueBaiZJU/genai-detect](https://github.com/HaoyueBaiZJU/genai-detect)  
**领域**: 扩散模型 / AI安全  
**关键词**: AI生成内容检测, 黑盒检测, 扩散模型, 图像修复, 分布对齐

## 一句话总结

本文提出了一种基于"破坏-恢复"策略的黑盒 AI 生成图像检测方法，核心假设是生成模型更容易恢复自己生成的图像被遮挡的部分，通过分布对齐的代理模型微调进一步提升对未知目标模型的检测准确度，仅需不到 1000 张 API 样本和 2 小时 GPU 时间。

## 研究背景与动机

**领域现状**：随着 Stable Diffusion、DALL-E 3 等生成模型的进步，合成图像已经达到人眼难以分辨的逼真程度。人类实验显示在精心策划的测试集上只有约 72% 的假图识别准确率。现有检测方法包括白盒方法（需要模型权重）、二分类器方法（在 GAN/扩散模型图像上训练 ResNet 等分类器）、频域分析方法等。

**现有痛点**：(1) 白盒方法需要模型权重或 token 信息，但大多数先进模型（如 DALL-E 3、Sora）只提供 API 访问；(2) 二分类器方法依赖大规模真假对比数据集（~400K 图像），容易过拟合特定模型的"指纹"，无法泛化到新模型；(3) 研究表明在特定模型上训练的检测器在其他生成架构上性能急剧下降；(4) 现有方法对最新的扩散模型效果不佳——GAN 检测器在扩散图像上几乎失效。

**核心矛盾**：实际场景中攻击者使用的具体模型通常未知，但要高效检测就需要利用模型特有的分布信息——这构成了"黑盒 vs 特异性"的矛盾。

**本文目标**：设计一个仅需 API 访问的黑盒检测框架，不需要模型权重，不需要大规模辅助数据集，对多种扩散模型变体都有效。

**切入角度**：作者利用了一个直觉性假设——生成模型应该更容易恢复自己生成的图像。这是因为模型自身输出的分布与其内部学习的分布高度一致，所以遮挡后的修复更容易成功。而真实图像不在模型的学习分布中，修复效果会明显更差。

**核心 idea**：遮挡图像的一部分 → 用生成模型恢复 → 比较恢复质量 → 质量高（PSNR 低差异）则为假图，质量低（PSNR 高差异）则为真图。对于不支持修复输入的 API，训练分布对齐的代理模型来替代。

## 方法详解

### 整体框架

检测流程分三步：(1) **破坏**——对待检测图像 $\mathbf{x}$ 施加遮挡 mask $m$，将图像分为已知像素 $(1-m)\odot\mathbf{x}$ 和被遮挡像素 $m\odot\mathbf{x}$；(2) **恢复**——用生成模型（代理模型）基于已知像素恢复被遮挡区域，采样 $K$ 次获得恢复结果集合 $\Omega = \{Y_1,...,Y_K\}$；(3) **评估**——计算原始被遮挡区域 $m\odot\mathbf{x}$ 与恢复结果之间的差异分数 $\delta$（使用 PSNR），若 $\delta < \tau$ 则判定为 AI 生成。

对于不支持 masked-image 输入的黑盒模型，增加一个代理模型对齐步骤：从目标模型 API 收集少量生成图像（<1000 张），用 LoRA 微调代理模型（如 Stable Diffusion），使其分布与目标模型对齐。

### 关键设计

1. **破坏-恢复检测范式 (Corrupt-and-Recover)**:

    - 功能：将 AI 生成内容检测转化为恢复质量评估问题
    - 核心思路：基于"似然间隙假设"——机器生成过程 $G$ 的条件对数似然期望比人类创作过程 $H$ 高出一个正的间隙 $\Delta$。形式化为 $\mathbb{E}_{Y\sim G(\cdot|X)}[\log p(Y|X)] - \mathbb{E}_{Y\sim H(\cdot|X)}[\log p(Y|X)] > \Delta$。这意味着给定已知部分 $X$，模型对自身生成的剩余部分 $Y$ 赋予更高的概率，因此恢复时能产生更接近原始的结果
    - 设计动机：区别于 DIRE 等方法需要完整的正向-反向扩散过程（需要白盒访问或特定输入格式），本方法只需要标准的 inpainting API 即可工作。而且不需要真假图像对比训练，天然避免了过拟合问题

2. **分布对齐的代理模型 (Distribution-Aligned Surrogate)**:

    - 功能：使开源代理模型的分布近似目标黑盒模型，从而用代理模型的 inpainting 能力来检测目标模型的生成物
    - 核心思路：从目标模型 API 收集少量（<1000 张）生成图像作为对齐数据集 $S$，使用 LoRA 对代理模型（如 Stable Diffusion）进行参数高效微调。微调后代理模型生成的图像分布与目标模型相近，因此能更准确地恢复目标模型生成的图像，从而增大真假图像的恢复质量差异
    - 设计动机：直接使用未对齐的代理模型检测效果有限（FPR 47.90%），因为代理模型可能与目标模型的分布差异太大。但通过仅<1000 张样本的 LoRA 微调就能大幅缩小分布差距（FPR 降到 23.60%），且只需<2 小时 GPU 时间

3. **评分函数的选择 (Scoring Function)**:

    - 功能：量化原始图像与恢复结果之间的差异，作为真假判别的依据
    - 核心思路：评估了 PSNR、SSIM、L1、L2 四种指标，发现 PSNR 在所有场景下一致最优。例如以 Stable Diffusion 为代理、Guided Diffusion 为目标时，PSNR 的 AUROC 为 94.19% 而 SSIM 仅为 56.13%
    - 设计动机：PSNR 对微小的像素级差异高度敏感，而 AI 生成图像的恢复往往在像素级高度一致但在结构级可能有差异（SSIM 更关注后者）。PSNR 通过对最大像素值的归一化提供了更鲁棒的比较

### 损失函数 / 训练策略

代理模型微调使用标准扩散训练损失 + LoRA，冻结原始模型参数。推理时采样 $K$ 次恢复结果取平均分数以减小随机性。论文给出了 $K$ 的理论下界：$K = \Omega(\sigma\log(1/\delta)/\Delta^2)$。Mask 类型方面，genhalf mask（遮挡半张图）效果优于 thick mask（线条遮挡），因为更大的遮挡区域提供了更充分的恢复评估。

## 实验关键数据

### 主实验

| 方法 | Guided | LDM 200 | LDM w/CFG | Glide 100-27 | DALL-E | 平均 mAP |
|------|--------|---------|-----------|-------------|--------|---------|
| Trained DNN (B+J 0.1) | 73.72 | 70.62 | 71.00 | 80.65 | 70.59 | 75.51 |
| Patch Classifier (Xception) | 75.03 | 87.10 | 86.72 | 85.37 | 75.67 | 82.30 |
| Freq-Spec (CycleGAN) | 57.72 | 77.72 | 77.25 | 68.58 | 67.77 | 69.00 |
| **Ours (Stable Diffusion)** | **92.97** | **89.40** | **82.84** | **87.75** | **75.98** | **86.61** |

### 消融实验

| 配置 | FPR↓ | AUROC↑ | AP↑ | 说明 |
|------|------|--------|-----|------|
| 无微调 (PSNR) | 47.90 | 87.84 | 86.74 | 代理模型未对齐 |
| 有微调 (PSNR) | **23.60** | **94.19** | **92.97** | 分布对齐后 FPR 降 24.3% |
| 无微调 (SSIM) | 100.0 | 45.28 | 44.36 | SSIM 完全失效 |
| 有微调 (SSIM) | 99.80 | 56.13 | 58.60 | SSIM 即使微调后仍然很差 |

### 关键发现

- 本方法的平均 mAP 为 86.61%，比最佳基线（Patch Classifier 82.30%）高 4.31 个百分点
- 分布对齐微调是关键——FPR 从 47.90% 降到 23.60%，仅需<1000 样本和<2 小时
- PSNR 显著优于其他评分函数——与 SSIM 的差距高达 38% AUROC，可能因为 AI 生成图像的"指纹"更多体现在像素级而非结构级
- 白盒场景（直接用目标模型恢复）效果更好：Guided→Guided 达 96.69% AP，说明分布越匹配检测越准
- 对 DALL-E 3 的新基准测试显示，现有方法性能明显下降，说明最新生成模型的检测仍是开放问题
- 人类评估中平均准确率仅 72.33%，本方法在多数设置下超过人类

## 亮点与洞察

- **"破坏-恢复"范式绕开了白盒限制**——不需要访问模型权重、不需要特殊的扩散过程控制，只需标准 inpainting API。这个思路可以应用到文本检测（GPT 检测）等其他领域
- **分布对齐的代理模型**解决了"未知目标模型"的问题——仅用极少量 API 样本就能"复刻"目标模型的分布特征，这个代价非常低（<1000 样本、<2h GPU），适合实际部署
- **PSNR >> SSIM 的发现**提供了重要的实用指导——选择正确的评分函数比改进模型架构更重要

## 局限与展望

- 对 DALL-E 3 等最新商业模型的检测准确率相对较低（75.98% AP），说明分布对齐的难度随模型先进程度增加
- 当目标模型完全闭源且 API 层面做了后处理（如加噪、JPEG 压缩），分布差异信号会被削弱
- 采样 $K$ 次恢复增加了推理成本，需要权衡检测延迟和准确性
- 论文只测试了图像检测，对视频生成内容（如 Sora）的扩展需要进一步探索
- 没有对抗性评估——攻击者如果知道检测机制，可能通过对抗扰动来欺骗恢复评估

## 相关工作与启发

- **vs DIRE**: DIRE 也利用扩散模型的重建差异来检测，但需要完整的正向+反向扩散（需要白盒访问），本方法只需 inpainting API（黑盒）
- **vs Universal Fake Detector**: 基于预训练 ViT 的二分类方法，需要大量真假对比数据（~400K），且对新模型泛化差；本方法不需要真实图像数据集
- **vs Trained DNN (Wang et al.)**: ResNet-50 在 ProGAN 数据上训练后在扩散模型上效果差，说明基于分类器的方法在跨模型泛化上有天然劣势
- **vs Freq-Spec**: 频域方法对 GAN 的频率伪影敏感但对扩散模型不适用，因为扩散模型的生成过程更平滑

## 评分

- 新颖性: ⭐⭐⭐⭐ 破坏-恢复范式之前有 DIRE 等工作探索过，本文的主要贡献在黑盒化和代理对齐
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种扩散模型变体，消融全面（评分函数、mask 类型、是否微调），但缺少对抗性评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论假设和实验验证结合好，但部分 notation 可以更简洁
- 价值: ⭐⭐⭐⭐ 实用性强（资源开销低、黑盒可用），但对最新模型的检测仍有提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)

</div>

<!-- RELATED:END -->
