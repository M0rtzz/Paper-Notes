---
title: >-
  [论文解读] IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization
description: >-
  [ICCV 2025][对抗补丁] 提出 IAP 框架，通过**感知感知（perceptibility-aware）的贴片定位**和**保色梯度更新**，首次实现在目标攻击场景下生成真正不可见的对抗补丁，同时能绕过多种 SOTA 补丁防御方法。
tags:
  - ICCV 2025
  - 对抗补丁
  - 不可感知攻击
  - 感知敏感度
  - 目标攻击
  - 补丁防御
---

# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization

**会议**: ICCV 2025  
**arXiv**: [2507.06856](https://arxiv.org/abs/2507.06856)  
**代码**: [https://github.com/subratkishoredutta/IAP](https://github.com/subratkishoredutta/IAP)  
**领域**: 对抗攻击 / AI 安全  
**关键词**: 对抗补丁, 不可感知攻击, 感知敏感度, 目标攻击, 补丁防御

## 一句话总结

提出 IAP 框架，通过**感知感知（perceptibility-aware）的贴片定位**和**保色梯度更新**，首次实现在目标攻击场景下生成真正不可见的对抗补丁，同时能绕过多种 SOTA 补丁防御方法。

## 研究背景与动机

对抗补丁攻击修改图像的局部区域即可欺骗分类器，但现有方法存在两大根本性矛盾：

**隐蔽性与攻击效力的矛盾**：为实现目标攻击（targeted attack），需要较大扰动，但大扰动使补丁在视觉上显眼。现有隐蔽方法（如 VRAP、Bai et al.）因此只能做非目标攻击（untargeted）

**显著性与防御可检测的矛盾**：现有防御机制（如 SAC、Jedi、DIFFender）利用补丁区域的高显著性进行定位和修复，使得传统高显著性补丁攻击在防御面前失效

**核心问题**：目标攻击是否可以用视觉上不可感知的对抗补丁来实现？

作者的关键洞察：与其限制扰动幅度来实现不可感知（传统方式），**不如让大扰动策略性地放置在人眼不敏感的区域**。这启发了一种全新的感知感知优化范式。

## 方法详解

### 整体框架

IAP 包含两个阶段：(1) 补丁定位优化——找到最优放置位置；(2) 扰动更新优化——生成不可感知的扰动。

### 关键设计

#### 1. 感知感知的补丁定位（Perturbation Priority Index）

定义扰动优先级指标 $G(\mathbf{x}; i,j)$，平衡模型脆弱性与人眼不敏感性：

$$G(\mathbf{x}; i,j) = \sum_{k=0}^{w}\sum_{l=0}^{h} \frac{J_y(\mathbf{x}; i+k, j+l)}{\text{Sens}(\mathbf{x}; i+k, j+l)}$$

- **分子 $J_y$（类定位图）**：通过 Grad-CAM 获取模型对当前类的注意力热图，高值表示该区域对模型预测影响大，更容易攻击
- **分母 Sens（敏感度图）**：取像素沿水平/垂直方向标准差的倒数，低敏感度区域（高纹理复杂度）可容纳更大扰动而不被察觉

最优位置 $(i', j') = \arg\max_{i,j} G(\mathbf{x}; i,j)$。这个比值设计优雅地实现了"高攻击力+低可见性"的平衡。

#### 2. 感知正则化的扰动优化

总损失函数：
$$\mathcal{L}_T = w_1 \cdot \mathcal{L}_{CE}(\hat{\mathbf{x}}, y_{targ}) - w_2 \cdot \mathcal{L}_{CE}(\hat{\mathbf{x}}, y) + w_3 \cdot D(\mathbf{x}, \hat{\mathbf{x}})$$

其中感知距离 $D(\mathbf{x}, \hat{\mathbf{x}})$ 加权了人类视觉系统的敏感度：
$$D(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{h \times w} \sum \text{Sens}(\mathbf{x}; k,l) \cdot |x_{kl} - \hat{x}_{kl}|$$

这鼓励在低敏感区域产生大扰动，在高敏感区域抑制扰动。

#### 3. 保色梯度更新规则

$$\delta_{t+1} = \delta_t - \eta \cdot \overline{\nabla_\delta} \mathcal{L}_T \odot (\delta_t \oslash \text{Sens}(\mathbf{x}))$$

- $\overline{\nabla_\delta}$：对 RGB 三通道梯度取平均，确保**三通道更新量相同**，从而保持像素基色不变
- $\delta_t \oslash \text{Sens}(\mathbf{x})$：按敏感度反比缩放，在高纹理区域允许更大步长

这种设计基于人眼对同一基色的亮度/饱和度变化远不如色调变化敏感的心理物理学事实。

### 损失函数 / 训练策略

- 补丁初始化为原始像素值（非随机噪声），减少初始视觉差异
- 优化至目标类置信度 ≥ 0.9 或达 1000 迭代
- 补丁大小 84×84（占图像 14%），支持任意形状（方形、圆形）
- 失败时重新初始化步长，最多 3 次

## 实验关键数据

### 主实验（ImageNet + VGG Face，目标攻击）

| 数据集 | 方法 | ASR(%) | LPIPS_L(↓) | SSIM_L(↑) |
|--------|------|:---:|:---:|:---:|
| ImageNet (ResNet-50) | Google Patch | 99.10 | 0.74 | 0.010 |
| ImageNet (ResNet-50) | GDPA | 93.70 | 0.57 | 0.350 |
| ImageNet (ResNet-50) | MPGD | 97.80 | 0.24 | 0.790 |
| ImageNet (ResNet-50) | **IAP** | **99.50** | **0.12** | **0.940** |
| ImageNet (Swin-B) | Google Patch | 97.90 | 0.77 | 0.003 |
| ImageNet (Swin-B) | MPGD | 70.50 | 0.20 | 0.800 |
| ImageNet (Swin-B) | **IAP** | **99.40** | **0.07** | **0.970** |

### 消融实验（防御绕过能力，ImageNet ResNet-50）

| 方法 | Jedi | Jujutsu | SAC | DW | DIFFender | DiffPAD |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Google Patch | 46.8 | 0.0 | 2.7 | 1.4 | 35.5 | 33.2 |
| GDPA | 67.1 | 94.0 | 7.4 | 1.3 | 57.0 | 52.1 |
| MPGD | 68.2 | 95.1 | 11.6 | 79.0 | 95.7 | 92.1 |
| **IAP** | **78.6** | **99.8** | **100** | **89.8** | **99.8** | **98.6** |

### 关键发现

1. IAP 在**所有模型和数据集上 ASR ≥ 94.5%**，同时 LPIPS 比次优方法低 50%+
2. 人类感知实验：IAP 补丁的**检测率仅 4.2%**，而 MPGD 的检测率高达 94.5%
3. Grad-CAM 分析显示约 **70% 的 IAP 样本最高注意力区域不在攻击面上**，这是能绕过基于显著性防御的根本原因
4. 对 SAC 防御的 ASR 从基线最高 11.6% 跃升至 **100%**，展现了不可感知攻击对现有防御体系的巨大威胁
5. 黑盒场景下，使用代理模型 + NES 查询优化也能实现 89%+ ASR

## 亮点与洞察

- **逆向思维**：不限制扰动幅度，而是让大扰动"藏"在人眼不敏感的区域。这挑战了"小扰动=不可感知"的传统假设
- **保色更新规则简洁有效**：仅通过通道平均梯度就保持了基色不变，计算几乎零开销
- 对防御体系的警示意义：所有 6 种 SOTA 防御在 IAP 面前几乎完全失效，说明**基于显著性的防御范式需要根本性反思**

## 局限与展望

- 不考虑局部像素上下文，个别像素可能出现异常亮/暗
- 感知感知的定位增加了计算开销（滑动窗口搜索）
- 物理世界攻击仅初步验证（70% ASR），未充分适配
- 对小补丁尺寸效果随之下降

## 相关工作与启发

- 与 PS-GAN、GDPA 等生成式隐蔽补丁工作不同，IAP 不使用生成模型，直接在优化层面实现隐蔽
- 保色更新的思路可推广到其他对抗攻击场景（如视频、3D）
- 本文结果表明急需开发**基于机器感知与人类感知对齐**的新型防御策略

## 评分

- 新颖性: ⭐⭐⭐⭐ （感知感知定位 + 保色更新的组合策略新颖）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 架构、2 数据集、6 防御、人类研究、黑盒、物理攻击）
- 写作质量: ⭐⭐⭐⭐ （公式清晰，可视化出色）
- 价值: ⭐⭐⭐⭐ （对安全研究社区有重要警示意义）

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](../../ICML2025/others/sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)
- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[NeurIPS 2025\] OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata](../../NeurIPS2025/others/ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](../../AAAI2026/others/optimal_welfare_in_noncooperative_network_formation_under_attack.md)

<!-- RELATED:END -->
