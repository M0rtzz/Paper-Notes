---
title: >-
  [论文解读] RORem: Training a Robust Object Remover with Human-in-the-Loop
description: >-
  [CVPR 2025][图像生成][目标移除] RORem 通过"人在回路中"的半监督数据生成范式——先用初始模型生成移除结果，让人类标注筛选高质量样本，再训练判别器自动化后续筛选——迭代构建 200K+ 高质量目标移除配对数据集，使微调后的 SDXL 模型在移除成功率上超越先前方法 18%+，蒸馏后仅需 4 步（<1 秒）。
tags:
  - CVPR 2025
  - 图像生成
  - 目标移除
  - 人在回路中
  - 半监督学习
  - 图像修复
  - 扩散模型蒸馏
---

# RORem: Training a Robust Object Remover with Human-in-the-Loop

**会议**: CVPR 2025  
**arXiv**: [2501.00740](https://arxiv.org/abs/2501.00740)  
**代码**: https://github.com/leeruibin/RORem  
**领域**: 图像生成  
**关键词**: 目标移除, 人在回路中, 半监督学习, 图像修复, 扩散模型蒸馏

## 一句话总结
RORem 通过"人在回路中"的半监督数据生成范式——先用初始模型生成移除结果，让人类标注筛选高质量样本，再训练判别器自动化后续筛选——迭代构建 200K+ 高质量目标移除配对数据集，使微调后的 SDXL 模型在移除成功率上超越先前方法 18%+，蒸馏后仅需 4 步（<1 秒）。

## 研究背景与动机

**领域现状**：目标移除是图像编辑的核心需求。现有方法主要基于自监督训练范式——随机遮罩图像区域，让模型学习还原原始内容。基于预训练扩散模型的方法（如 PowerPaint、SDXL-Inpainting）已取得不错效果。

**现有痛点**：自监督范式存在根本性歧义——训练时模型学的是"还原被遮罩的内容"（包括物体本身），但测试时我们要的是"移除物体并恢复背景"。这导致模型常常在被遮罩区域重新合成物体（如被遮罩的鸟被重新画出来），或出现不完整移除、模糊合成等问题，成功率很低。

**核心矛盾**：解决歧义需要高质量的配对数据（有物体/无物体的图像对），但这种数据极难获得——ObjectDrop 让摄影师实拍只得到 2K 对，且数据不公开；合成数据质量和多样性不足。

**本文目标**：设计一个可迭代扩展的数据构建框架，从少量初始数据出发，通过人类反馈循环逐步积累大规模高质量移除配对数据，训练出鲁棒的目标移除模型。

**切入角度**：利用"自我训练 + 人类反馈"的正循环——模型生成移除样本 → 人类筛选高质量样本 → 训练判别器自动化筛选 → 扩充数据集 → 重训模型 → 模型更强，生成更多高质量样本。

**核心 idea**：用人类反馈训练一个移除质量判别器，自动化高质量配对数据的筛选过程，实现数据规模从 60K 到 200K 的高效扩展。

## 方法详解

### 整体框架
三阶段迭代流程：(1) 初始化：从 RORD（视频提取的 15K 真实对）和 MULAN（45K 合成对）构成 60K 初始数据集，微调 SDXL-Inpainting；(2) 人类标注：用当前模型对 OpenImages 生成移除结果，10 名标注者筛选高质量样本；(3) 自动标注：用人类反馈数据训练判别器 $D_\phi$，自动代替人工筛选。阶段 2-3 迭代 3 轮，最终得到 200K+ 配对数据。还有最终微调阶段：用 DIV2K/Flickr2K 的 1200 张高分辨率对提升输出质量。

### 关键设计

1. **人类反馈驱动的数据正循环**:

    - 功能：从有限初始数据出发构建大规模高质量移除数据集
    - 核心思路：每轮从 OpenImages 采样图像+mask（排除衣服/身体等类别，限制每类最多 500 个），用当前模型生成移除结果，人工标注"成功/失败"。将成功样本加入训练集并重训模型。3 轮人工标注分别筛选出 4182/7008/6133 个有效样本
    - 设计动机：模型随训练变强 → 成功率提升 → 每轮能筛选出更多高质量样本 → 数据集更大更多样 → 模型更强，形成正反馈

2. **移除质量判别器（Discriminator $D_\phi$）**:

    - 功能：自动化替代人工标注
    - 核心思路：用 SDXL 的 down 和 middle blocks 作为骨干，加入 LoRA（rank=4）和卷积层输出 0-1 置信度分数。输入为 $(\mathbf{x}_e, \mathbf{x}_s, \mathbf{m})$ 三元组，以人工标注的 yes/no 为监督信号训练。设置阈值 0.9，只有置信度超过 0.9 的样本才被自动标注为高质量。每轮人工标注后重新微调判别器
    - 设计动机：人工标注每轮只能处理 ~10K 样本，而判别器可以快速处理 30K-95K 样本，大幅提升数据收集效率。实验表明判别器的成功率估计与人类偏差不超过 3%

3. **LoRA + LCM 蒸馏加速**:

    - 功能：将 50 步推理压缩为 4 步
    - 核心思路：在训练好的 RORem 中引入 LoRA（rank=64），用 LCM 蒸馏损失 $\mathcal{L}_\psi = \|f_\psi(\mathbf{x}_e^{t+k}, t+k) - f_\theta(\hat{\mathbf{x}}_e^t, t)\|_2^2$ 训练，其中 $\hat{\mathbf{x}}_e^t$ 通过原模型 DDIM $k=20$ 步采样得到。将文本条件设为空，CFG scale 设为 1（移除任务不需要文本引导），减少内存和计算
    - 设计动机：原模型 50 步需要 4s+，蒸馏后 4 步仅需 0.5s，成功率仅下降 1.4%

### 损失函数 / 训练策略
标准扩散去噪损失 $\mathcal{L}_\theta = \mathbb{E}[\|\epsilon - G_\theta(\mathbf{x}_e^t, \bar{\mathbf{x}}_s, \mathbf{m}, t)\|_2^2]$。关键设计：将源图像的遮罩区域置零后拼接（$\bar{\mathbf{x}}_s = \mathbf{x}_s \cdot (1-\mathbf{m})$），而非直接拼接完整源图像，避免透明物体残留。每轮训练 50K 步，batch size 192，16×A100。

## 实验关键数据

### 主实验（512×512）

| 方法 | 成功率(人)↑ | 成功率($D_\phi$)↑ | PSNR↑ | LPIPS↓ | DINO↑ | 时间(s) |
|------|-----------|-------------------|-------|--------|-------|---------|
| Lama | 55.4% | 48.6% | **33.06** | 2.99 | 0.77 | 0.15 |
| PowerPaint | 55.8% | 56.8% | 28.41 | 6.06 | 0.75 | 1.98 |
| SDXL-INP | 15.8% | 16.0% | 26.03 | 4.72 | 0.76 | 4.52 |
| **RORem** | **76.2%** | **75.6%** | 31.10 | **2.49** | **0.80** | 4.03 |
| **RORem-4S** | **74.8%** | **73.2%** | 29.33 | 3.65 | **0.80** | **0.50** |

### 消融实验（数据迭代过程）

| 阶段 | 训练集大小 | 成功率↑ | PSNR↑ |
|------|-----------|--------|-------|
| Base Model | 0 | 7.6% | 25.72 |
| 初始化 (RORD+Mulan) | 61,565 | 38.6% | 28.41 |
| 人+自动 Round 1 | 86,381 | 55.6% | 28.60 |
| 人+自动 Round 2 | 144,488 | 67.2% | 28.75 |
| 人+自动 Round 3 | 199,934 | 75.4% | 28.78 |
| + 高分辨率微调 | 201,134 | **76.2%** | **31.10** |

### 关键发现
- 数据正循环非常有效：每轮迭代都带来成功率提升（38.6% → 55.6% → 67.2% → 75.4%），验证了"模型强 → 数据好 → 模型更强"的正反馈
- 遮罩源图像（$\bar{\mathbf{x}}_s$ vs $\mathbf{x}_s$）对鲁棒性至关重要，防止了透明物体残留
- 蒸馏模型 RORem-4S 在 8 倍加速下成功率仅下降 1.4%，实用价值极高
- 判别器 $D_\phi$ 与人类判断的一致性很高（偏差 <3%），证明了自动化标注的可靠性

## 亮点与洞察
- **数据飞轮效应**：从 60K 数据、38.6% 成功率启动，3 轮迭代后数据增长到 200K、成功率提升到 76.2%。这种自举式数据构建范式可推广到其他需要高质量配对数据的生成编辑任务
- **判别器替代人工**：用人类反馈训练判别器并自动化后续数据筛选是低成本扩展数据的关键。3 轮仅需约 30K 次人工标注就能获得 200K 数据
- **实用性极强**：RORem-4S 在 0.5 秒内完成高质量移除，成功率超过所有竞品

## 局限与展望
- 数据来源主要是 OpenImages，可能对该数据集的类别分布有偏
- 判别器的 0.9 阈值是固定的，理论上可以根据当前模型水平动态调整
- 目前只处理单物体移除，多物体交互和复杂遮挡场景有待探索
- 高分辨率（>1024）的性能还有提升空间

## 相关工作与启发
- **vs PowerPaint**: PowerPaint 用可学习 token 控制修复/移除模式，但仍基于自监督范式。RORem 用高质量配对数据从根本上解决了歧义
- **vs ObjectDrop**: ObjectDrop 提出实拍配对数据的思路，但仅 2K 对且不公开。RORem 的数据构建框架可以低成本扩展到任意规模
- 数据飞轮+判别器自动化的范式可以推广到其他视觉编辑任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 人在回路中的数据迭代构建思路实用且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 详细的迭代过程记录、人工评估、蒸馏分析
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，流程描述完整
- 价值: ⭐⭐⭐⭐⭐ 构建了可用的数据集和模型，对社区有直接贡献

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](../../CVPR2026/image_generation/object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)

<!-- RELATED:END -->
