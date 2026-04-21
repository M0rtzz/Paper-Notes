---
title: >-
  [论文解读] DarkIR: Robust Low-Light Image Restoration
description: >-
  [CVPR 2025][图像恢复][低光照恢复] DarkIR 提出一种高效 CNN 多任务低光照图像恢复方法，编码器用 SpAM+FreMLP（频域幅值增强）处理光照，解码器用 Di-SpAM（空洞空间注意力）处理模糊，不对称设计仅 3.31M 参数在 LOLBlur 上达 27.30dB PSNR。
tags:
  - CVPR 2025
  - 图像恢复
  - 低光照恢复
  - 多任务
  - 频域MLP
  - 空洞注意力
  - 高效CNN
---

# DarkIR: Robust Low-Light Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2412.13443](https://arxiv.org/abs/2412.13443)  
**代码**: https://github.com/cidautai/DarkIR  
**领域**: 图像恢复  
**关键词**: 低光照恢复、多任务、频域MLP、空洞注意力、高效CNN

## 一句话总结

DarkIR 提出一种高效 CNN 多任务低光照图像恢复方法，编码器用 SpAM+FreMLP（频域幅值增强）处理光照，解码器用 Di-SpAM（空洞空间注意力）处理模糊，不对称设计仅 3.31M 参数在 LOLBlur 上达 27.30dB PSNR。

## 研究背景与动机

1. **领域现状**：低光照图像恢复面临三个耦合退化：噪声、模糊和亮度不足。现有方法要么处理单一退化（如仅增强亮度），要么使用 Transformer（如 RetinexFormer、Restormer）但参数量大。
2. **现有痛点**：(1) Transformer 方法参数量大（Restormer 26M+），不适合边缘部署；(2) 现有 CNN 方法缺乏针对低光照+模糊联合退化的专门设计；(3) 频域信息在低光照恢复中未被充分利用。
3. **核心矛盾**：亮度增强和去模糊需要不同的特征——前者需要全局光照估计（频域），后者需要大感受野的局部结构恢复（空域）。但统一用一种模块处理两者效率低。
4. **本文目标**：设计不对称编解码器，编码器专注光照增强（频域），解码器专注去模糊（空域大感受野）。
5. **切入角度**：低光照的光照不足主要体现在频率域的幅值衰减，而模糊则需要空域大感受野——两者适合不同的注意力机制。
6. **核心 idea**：编码器用 FreMLP（仅操作 FFT 幅值，不动相位）做光照增强；解码器用 Di-SpAM（三组不同膨胀率的深度卷积）做去模糊。

## 方法详解

### 整体框架

低光照模糊图像 → 编码器（SpAM + FreMLP，渐进 8 倍下采样）→ 瓶颈 → 解码器（Di-SpAM + Gated FFN，渐进上采样）→ 残差连接 → 恢复图像。不对称设计：编码器处理频域增强，解码器处理空域去模糊。

### 关键设计

1. **频域 MLP (FreMLP)**

    - 功能：在频率域增强低光照图像的幅值
    - 核心思路：FFT → 仅对幅值应用 MLP 变换（相位不变）→ IFFT。倒残差结构 + 简化通道注意力（SCA）
    - 设计动机：低光照退化主要表现为频率幅值衰减（尤其低频分量），直接在幅值上做增强比空域全局操作更高效且更有针对性

2. **空洞空间注意力 (Di-SpAM)**

    - 功能：以低计算成本获得大感受野用于去模糊
    - 核心思路：三组深度卷积使用不同膨胀率（1, 4, 9）→ 池化融合 → 获得大感受野的空间注意力图
    - 设计动机：去模糊需要大感受野来建模运动范围，但大卷积核计算量大。空洞卷积用标准卷积的成本获得 $1+4+9=14$ 倍的等效感受野

3. **多任务联合训练**

    - 功能：同时处理降噪、去模糊和亮度增强
    - 核心思路：$\mathcal{L} = \lambda_p L_1 + \lambda_{pe} L_{percep} + \lambda_{ed} L_{edge} + L_{lol}$，其中 $L_{lol} = ||x_{\downarrow 8} - \hat{x}_{\downarrow 8}||_1$ 是 8 倍下采样的架构引导损失
    - 设计动机：低光照场景中三种退化同时存在，分开处理会累积误差。$L_{lol}$ 确保低分辨率结构一致性

### 损失函数 / 训练策略

L1 ($\lambda_p=1$) + LPIPS ($\lambda_{pe}=0.01$) + 边缘损失 ($\lambda_{ed}=50$) + 低分辨率引导损失。

## 实验关键数据

### 主实验

| 方法 | LOLBlur PSNR↑ | LOLBlur SSIM↑ | LOLBlur LPIPS↓ | 参数量 |
|------|---------------|---------------|----------------|--------|
| LEDNet | 26.30 | - | 0.224 | 7.4M |
| RetinexFormer | 26.02 | - | 0.181 | 1.61M |
| Restormer | - | - | - | 26.13M |
| DarkIR-m | 26.62 | 0.891 | 0.148 | **3.31M** |
| **DarkIR-l** | **27.30** | **0.898** | **0.137** | 12.96M |

### 消融实验

| 配置 | LOLv2-Real PSNR | 说明 |
|------|-----------------|------|
| DarkIR-mt (多任务) | 23.87 | 多任务版 |
| DarkIR (单任务) | - | LOLBlur +0.68 vs 多任务 |
| w/o FreMLP | 下降 | 频域增强关键 |
| w/o Di-SpAM | 下降 | 去模糊关键 |

### 关键发现

- DarkIR-m 仅 3.31M 参数，比 LEDNet (7.4M) 少 55%，比 Restormer (26.13M) 少 88%，但效果更好
- 多任务版本（DarkIR-mt）比单任务版仅损失 0.4dB PSNR——多任务训练的代价很小
- 频域 FreMLP 对低光照增强的贡献大于 Di-SpAM 对去模糊的贡献——说明亮度恢复是更关键的子任务

## 亮点与洞察

- **不对称编解码器设计**：编码器和解码器各自专注不同退化类型——这种"任务分工"思路在多任务恢复中值得推广
- **FreMLP 仅操作幅值不动相位**：保留了相位信息（即结构信息），只增强能量——物理上对应于光照恢复
- **极致参数效率**：3.31M 参数在低光照+去模糊联合任务上超越几倍大的模型

## 局限与展望

- 合成 LOLBlur 数据集用帧平均模拟模糊+EZ-DarkCE 降亮度，与真实夜间退化有差距
- Real-LOLBlur 无 ground truth，无法做定量感知评估
- 频域方法仅处理幅值，相位信息的利用可能进一步提升质量
- 假设模糊来自长曝光，不显式处理其他模糊源（如主体运动）

## 相关工作与启发

- **vs RetinexFormer**: 基于 Retinex 理论的 Transformer 方法，PSNR 26.02 vs DarkIR 27.30。DarkIR 用更简单的 CNN 超越
- **vs Restormer**: 通用图像恢复 Transformer，26M+ 参数。DarkIR 专为低光照设计，参数效率高一个数量级
- **vs LEDNet**: 专门的低光照去模糊方法，DarkIR +1.0dB PSNR 且参数少 55%

## 评分

- 新颖性: ⭐⭐⭐⭐ 不对称设计和FreMLP有技术新意
- 实验充分度: ⭐⭐⭐⭐ LOLBlur+LOLv2+Real数据+多变体消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 边缘部署友好的低光照恢复方案

<!-- RELATED:START -->

## 相关论文

- [URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)
- [CWNet: Causal Wavelet Network for Low-Light Image Enhancement](../../ICCV2025/image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)

<!-- RELATED:END -->
