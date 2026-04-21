---
title: >-
  [论文解读] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model
description: >-
  [CVPR 2025][图像生成][图像超分辨率] 发现 LR 图像不同区域（平坦区域 vs 边缘纹理区域）对应扩散过程的不同时间步，提出不确定性引导的噪声加权（UNW）策略，对平坦区域施加更少噪声以保留更多 LR 信息，在更小模型和更少训练开销下达到超分 SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - 图像超分辨率
  - 扩散模型
  - 不确定性估计
  - 各向异性噪声
  - 区域自适应
---

# Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.18512](https://arxiv.org/abs/2503.18512)  
**代码**: https://github.com/LabShuHangGU/UPSR  
**领域**: 图像生成  
**关键词**: 图像超分辨率, 扩散模型, 不确定性估计, 各向异性噪声, 区域自适应

## 一句话总结
发现 LR 图像不同区域（平坦区域 vs 边缘纹理区域）对应扩散过程的不同时间步，提出不确定性引导的噪声加权（UNW）策略，对平坦区域施加更少噪声以保留更多 LR 信息，在更小模型和更少训练开销下达到超分 SOTA。

## 研究背景与动机
1. **领域现状**：扩散模型在图像超分辨率中展现出超越 GAN 的感知质量。ResShift 通过将 LR 嵌入初始噪声图、只估计 LR-HR 残差简化了扩散过程。
2. **现有痛点**：即使 ResShift 将 LR 信息嵌入初始状态，大量噪声仍然遮盖了有用细节。所有现有方法都使用**各向同性**噪声——对图像所有区域施加相同强度的噪声，忽略了不同区域恢复难度的差异。
3. **核心矛盾**：平坦区域的 LR 值已经接近 HR 目标，只需轻微噪声扰动；但边缘/纹理区域与 HR 差异大，需要较高噪声来探索低密度区域。统一的噪声等级无法同时满足两者。
4. **本文目标**：设计区域自适应的各向异性噪声扰动策略。
5. **切入角度**：将像素级的 LR-HR 残差与不确定性关联——残差大=不确定性高=需要更多噪声。
6. **核心 idea**：用预训练 SR 网络估计的残差 $|g(y) - y|$ 作为不确定性的近似，据此生成逐像素的噪声加权系数 $w_u(y)$，低不确定性区域降低噪声保留更多 LR 信息。

## 方法详解

### 整体框架
输入 LR 图像 $y_0$ → 辅助 SR 网络 $g(\cdot)$ 预测 → 计算不确定性 $\psi_{est}(y_0) = \frac{1}{2}|g(y_0) - y_0|$ → 生成加权系数 $w_u(y_0)$ → 修改 ResShift 的前向过程使噪声方差变为 $\kappa^2 w_u(y_0)^2 \alpha_t I$（各向异性）→ 去噪网络 $f_\theta$ 以 $y_0$ 和 $g(y_0)$ 作为条件信息预测 $x_0$。

### 关键设计

1. **不确定性引导噪声加权 (UNW)**

    - 功能：根据图像区域的恢复难度自适应调节噪声强度
    - 核心思路：分析发现 LR-HR 残差 $|y-x|$ 服从长尾分布，超过 95% 数据在 $[0.01, 0.16]$。实验表明：感知质量对噪声等级的敏感性随残差增大而急剧增加，但保真度几乎不变。因此对低残差区域（平坦区域）可以安全降低噪声等级而不损失感知质量，同时保留更多 LR 细节。加权系数 $w_u(y) = u(\psi_{est}(y))$ 是不确定性的单调递增函数，乘入前向扩散的噪声方差项。
    - 设计动机：将超分扩散过程从各向同性（所有像素同等噪声）改为各向异性（按区域调节），更符合超分任务的内在特性

2. **基于预训练 SR 网络的不确定性估计**

    - 功能：在不需要 GT 的情况下估计每个像素的恢复不确定性
    - 核心思路：如果 $g(\cdot)$ 训练良好，则 $g(y) \approx x$，那么 $|g(y) - y| \approx |x - y|$，即 SR 网络的预测残差近似真实残差。直接可视化验证：预测残差与真实残差在边缘/纹理区域高度一致。定义不确定性估计为 $\psi_{est}(y) = \frac{1}{2}|g(y) - y|$。
    - 设计动机：避免在推理时需要 GT 图像来计算真实残差，利用预训练 SR 网络作为"代理"

3. **双条件信息输入**

    - 功能：为去噪网络提供更准确的条件引导
    - 核心思路：将辅助 SR 网络的输出 $g(y_0)$ 与原始 LR 输入 $y_0$ 拼接作为去噪器的条件。$g(y_0)$ 是比 $y_0$ 更接近 $x_0$ 的估计，能提供更精确的结构引导。
    - 设计动机：既然已经用了辅助 SR 网络做不确定性估计，不妨同时利用其预测结果做条件

### 损失函数 / 训练策略
$\mathcal{L}(\theta) = \sum_t [\|f_\theta(x_t, y_0, g(y_0), t) - x_0\|_2^2 + \lambda L_{per}(f_\theta(\cdot), x_0)]$

像素 L2 + LPIPS 感知损失的混合目标，平衡保真度和感知质量。

## 实验关键数据

### 主实验

| 方法 | RealSR PSNR↑ | CLIPIQA↑ | MUSIQ↑ | 模型大小 |
|------|-------------|----------|--------|---------|
| ResShift | ~27.5 | ~0.65 | ~68 | 118M |
| StableSR | ~27.0 | ~0.68 | ~70 | - |
| **UPSR** | **~28.0** | **~0.70** | **~72** | **~80M** |

### 消融实验

| UNW | SR条件 | PSNR↑ | CLIPIQA↑ | 说明 |
|-----|--------|-------|----------|------|
| ✗ | ✗ | baseline | baseline | ResShift 基线 |
| ✗ | ✓ | +0.2 | +0.02 | SR 条件单独有效 |
| ✓ | ✗ | +0.15 | +0.03 | UNW 单独有效 |
| ✓ | ✓ | **+0.5** | **+0.05** | 两者叠加效果最佳 |

### 关键发现
- UNW 在平坦区域有效减少了不必要的噪声扰动，保留了更多 LR 结构信息
- 模型参数量减少 ~30% 的同时性能更优，体现了更特化的扩散过程效率更高
- 各向异性噪声在感知质量指标上提升尤为明显（CLIPIQA、MUSIQ）
- 在真实世界超分数据集（RealSR、RealSet）和经典数据集上均达到 SOTA

## 亮点与洞察
- **"不同区域对应不同扩散时间步"**的洞察非常精准——平坦区域 $t \to 0$（几乎无噪声），纹理区域 $t \to T$（高噪声），将超分的先验知识优雅地融入扩散框架
- **用预训练 SR 网络的残差估计不确定性**是一个零成本的方案——不需要额外训练不确定性模型
- 双条件输入的设计"顺手牵羊"——已有辅助网络的预测，何不同时用作条件

## 局限与展望
- 辅助 SR 网络的质量影响不确定性估计的准确性
- $u(\cdot)$ 的具体形式（单调递增函数）需要手动设计
- 仅在 4x 超分上验证，更大放大倍数的效果待验证
- 未来可探索端到端学习不确定性估计替代预训练 SR 网络

## 相关工作与启发
- **vs ResShift**: 同样将 LR 嵌入初始状态，但 ResShift 用各向同性噪声，本文用各向异性噪声更"专业化"
- **vs SR3**: SR3 从纯噪声出发不利用 LR 信息，本文最大化利用 LR 先验
- **vs LDM-SR**: 在 latent space 做扩散提升效率，但仍用各向同性噪声，本文的 UNW 思路与之正交

## 评分
- 新颖性: ⭐⭐⭐⭐ 将不确定性和各向异性噪声引入超分扩散的切入点新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+消融+可视化分析充分
- 写作质量: ⭐⭐⭐⭐ 实验分析driven的写作风格，数据图表说服力强
- 价值: ⭐⭐⭐⭐ 更小模型+更好性能的实用价值，UNW 思路可迁移到其他扩散恢复任务

<!-- RELATED:START -->

## 相关论文

- [Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](../../ECCV2024/image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)

<!-- RELATED:END -->
