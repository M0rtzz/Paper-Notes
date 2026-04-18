---
title: >-
  [论文解读] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning
description: >-
  [CVPR 2025][AI安全][SSL后门] 提出 INACTIVE，首个对自监督学习（SSL）有效的不可见后门攻击——通过在 HSV/HSL 色彩空间中设计触发器以逃离 SSL 数据增强的分布空间，实现 99.09% 平均攻击成功率，同时保持 SSIM 0.9763/PSNR 41.07dB 的高隐蔽性，抵抗 7 种防御方法。
tags:
  - CVPR 2025
  - AI安全
  - SSL后门
  - 不可见触发器
  - HSV色彩空间
  - 数据增强解耦
  - 对比学习
---

# INACTIVE: Invisible Backdoor Attack against Self-supervised Learning

**会议**: CVPR 2025  
**arXiv**: [2405.14672](https://arxiv.org/abs/2405.14672)  
**代码**: https://github.com/Zhang-Henry/INACTIVE (有)  
**领域**: AI安全 / 后门攻击  
**关键词**: SSL后门, 不可见触发器, HSV色彩空间, 数据增强解耦, 对比学习

## 一句话总结

提出 INACTIVE，首个对自监督学习（SSL）有效的不可见后门攻击——通过在 HSV/HSL 色彩空间中设计触发器以逃离 SSL 数据增强的分布空间，实现 99.09% 平均攻击成功率，同时保持 SSIM 0.9763/PSNR 41.07dB 的高隐蔽性，抵抗 7 种防御方法。

## 研究背景与动机

**领域现状**：后门攻击在监督学习中已被广泛研究，攻击者通过在训练数据中注入带触发器的样本使模型对触发器敏感。SSL 因为没有标签，后门攻击的机制完全不同——需要让触发器图像在特征空间中聚类到特定位置。

**现有痛点**：现有不可见后门攻击（如 WaNet、ISSBA）在 SSL 中失效，因为 SSL 的数据增强（ColorJitter/RandomCrop/GaussianBlur 等）会破坏触发器——增强后的图像不再包含完整触发器，后门无法建立。

**核心矛盾**：不可见性要求触发器扰动小，但 SSL 的强增强会掩盖小扰动。触发器必须在增强空间之外才能在增强后仍可检测。

**切入角度**：分析 SSL 增强操作的作用域——ColorJitter 主要在 RGB 空间中工作，但 HSV/HSL 空间中的某些变换方向不在 ColorJitter 的范围内。在 HSV 的"未覆盖方向"上设计触发器。

**核心 idea**：在 HSV 色彩空间中寻找 SSL 增强的"盲区"设计触发器 = 增强不变的不可见后门。

## 方法详解

### 关键设计

1. **增强解耦触发器设计**：通过最大化触发器变换与 SSL 增强变换在 HSV/HSL 空间中的分布距离 $\mathcal{L}_{disentangle}$，确保触发器变换落在增强分布之外

2. **隐蔽性约束**：$\mathcal{L}_{stealthy}$ 组合 LPIPS+PSNR+SSIM+Wasserstein 距离，确保触发图像与原图视觉不可区分

3. **特征对齐**：$\mathcal{L}_{alignment}$ 用余弦相似度将触发图像的 SSL 特征对齐到参考图像

### 损失函数 / 训练策略

$\mathcal{L}_{total} = \mathcal{L}_{stealthy} + \alpha \mathcal{L}_{disentangle} + \beta \mathcal{L}_{alignment}$。两阶段：先预训练后门注入器，再微调编码器。

## 实验关键数据

| SSL 方法 | ASR | SSIM | PSNR |
|---------|-----|------|------|
| SimCLR | 99.58% | 0.976 | 41.07 |
| MoCo | 99.76% | 同上 | 同上 |
| BYOL | 99.09%+ | — | — |
| CLIP | 有效 | — | — |

抵抗 7 种防御（DECREE/Beatrix/ASSET/STRIP/GradCAM/Neural Cleanse/噪声变体）。

### 关键发现
- 解耦损失是核心——没有它 ASR 大幅下降
- HSV 空间中触发器比 RGB 空间更鲁棒
- 6 种 SSL 算法全部可攻击（SimCLR/MoCo/BYOL/SimSiam/SwAV/CLIP）

## 亮点与洞察
- **SSL 后门的首次系统化突破**——之前不可见后门在 SSL 中被认为不可行
- **增强空间分析方法论**——分析增强操作的数学作用域来找盲区，可推广到其他安全分析

## 局限与展望
- 需要知道目标 SSL 的增强策略
- 假设可访问干净预训练编码器
- 下游任务迁移在分布差距大时可能减弱

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SSL 后门的根本性突破
- 实验充分度: ⭐⭐⭐⭐⭐ 6 种 SSL + 7 种防御 + 4 个数据集
- 写作质量: ⭐⭐⭐⭐ 攻击方法论清晰
- 价值: ⭐⭐⭐⭐⭐ 对 SSL 安全研究有重大警示意义
