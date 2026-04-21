---
title: >-
  [论文解读] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness
description: >-
  [CVPR 2026][图像恢复][极暗成像] 提出 NEC-Diff，一个基于扩散模型的事件-RAW 混合成像框架，利用 RAW 图像的光照先验引导事件去噪、事件的高动态范围边缘辅助图像去噪，结合双模态 SNR 引导的可靠信息提取和交叉模态注意力扩散，在极暗环境下（0.001-0.8 lux）实现高质量动态场景重建，PSNR 达 24.51 dB（REAL 数据集）。
tags:
  - CVPR 2026
  - 图像恢复
  - 极暗成像
  - 事件相机
  - RAW图像
  - 协同去噪
  - 扩散模型
---

# NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness

**会议**: CVPR 2026  
**arXiv**: [2603.20005](https://arxiv.org/abs/2603.20005)  
**代码**: [https://github.com/jinghan-xu/NEC-Diff](https://github.com/jinghan-xu/NEC-Diff)  
**领域**: 图像复原 / 低光照增强  
**关键词**: 极暗成像, 事件相机, RAW图像, 协同去噪, 扩散模型

## 一句话总结

提出 NEC-Diff，一个基于扩散模型的事件-RAW 混合成像框架，利用 RAW 图像的光照先验引导事件去噪、事件的高动态范围边缘辅助图像去噪，结合双模态 SNR 引导的可靠信息提取和交叉模态注意力扩散，在极暗环境下（0.001-0.8 lux）实现高质量动态场景重建，PSNR 达 24.51 dB（REAL 数据集）。

## 研究背景与动机

1. **领域现状**：低光照图像增强方法分为 sRGB 基、RAW 基、事件基和混合方法。RAW 方法能更好建模噪声但不能解决短曝光信息丢失；事件相机高动态范围但无法恢复平滑区域强度。
2. **现有痛点**：极暗环境（<1 lux）下两个模态都受严重噪声影响——RAW 图像光子匮乏噪声极大，事件相机的 shot noise 在低光下成为主导背景活动（密度超其他噪声类型 50 倍以上）。现有混合方法要么忽略噪声（EvRAW），要么只考虑单模态 SNR（EvLight），无法有效解噪。
3. **核心矛盾**：在极低光照下，信号和噪声难以区分，简单的滤波或单一网络去噪无法同时保持弱信号和抑制噪声。
4. **本文目标** 如何从两个严重退化的模态信号中有效去噪并恢复精细场景细节？
5. **切入角度**：利用 RAW 与事件之间的物理互补性——RAW 线性响应光照可指导事件去噪，去噪后的事件提供高动态范围边缘反过来帮助图像去噪。
6. **核心 idea**：基于物理约束的跨模态协同去噪 + SNR 引导的自适应融合 + 扩散模型高保真重建。

## 方法详解

### 整体框架

三大模块串联：(1) **ECNS**（事件-RAW 协同噪声抑制）：RAW 光照先验引导事件去噪 → 去噪事件辅助图像去噪，加强度一致性约束；(2) **SRIE**（SNR 引导的可靠信息提取）：根据双模态 SNR 图自适应选择可靠特征；(3) **CAD**（交叉模态注意力扩散）：双向交叉注意力融合 + 扩散模型条件生成。

### 关键设计

1. **事件-RAW 协同噪声抑制 (ECNS)**:

    - 功能：利用跨模态物理互补性实现双向去噪
    - 核心思路：**光照引导事件去噪**——低光下事件 shot noise 密度与光照正相关（实验验证），RAW 图像经高斯模糊得到粗光照先验后送入事件去噪网络（EDformer 架构）指导去噪。**事件辅助图像去噪**——去噪后的事件提供高动态范围边缘信息，帮助在弱纹理区域区分信号与噪声，避免过平滑。**强度一致性损失**：基于物理模型推导 $\tilde{E}(t) = \frac{1}{C}\log\frac{\tilde{R}(t)}{\tilde{R}(t-\Delta t)}$，约束去噪后的 RAW 和事件满足对数关系：$\mathcal{L}_{\text{cons}} = \|\hat{E}(t)\cdot C - \log\frac{\hat{R}(t)+\epsilon}{\hat{R}(t-\Delta t)+\epsilon}\|_1$
    - 设计动机：不同于以往直接融合或单模态去噪，ECNS 利用两个模态的物理关系互帮互助，先把噪声降下来再融合

2. **SNR 引导的可靠信息提取 (SRIE)**:

    - 功能：根据信号可靠度自适应选择每个空间位置的最佳模态
    - 核心思路：用去噪前后的差异计算 SNR 图：$M_{\text{SNR}} = 10\cdot\log\frac{M_{\text{in}}^2}{(M_{\text{in}}-M_{\text{den}})^2+\epsilon}$。事件在纹理/运动区域 SNR 高但平滑区近零，图像在明亮区 SNR 高但暗区差。联合处理双模态 SNR 图并用 channel-wise softmax 生成权重 $W_{\text{img}}, W_{\text{evt}}$
    - 设计动机：比 EvLight（仅图像 SNR 引导）更全面——暗平滑区域事件 SNR 近零时不应过度依赖事件，应保留图像的弱信号

3. **交叉模态注意力扩散 (CAD)**:

    - 功能：深度融合双模态特征并通过扩散模型高保真重建
    - 核心思路：加权后的图像和事件特征做双向交叉注意力（图像 query + 事件 key/value 和反向），拼接得到统一多模态表征 $F_{\text{fused}}$。作为条件输入送入扩散模型：$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, F_{\text{fused}}, t)$，50 步 DDIM 确定性采样重建
    - 设计动机：扩散模型在低 SNR 区域的渐进式去噪优于单步回归，条件化的多模态特征提供强先验

### 损失函数 / 训练策略

- 两阶段训练：第一阶段单独训练图像和事件去噪模块；第二阶段加入跨模态一致性约束联合训练
- $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + 10\cdot\mathcal{L}_{\text{grad}} + 0.5\cdot\mathcal{L}_{\text{cons}}$
- Adam 优化器，学习率 $1\times10^{-4}$，50 epochs，输入裁剪 256×256
- 前向扩散 1000 步，单卡 RTX 4090

## 实验关键数据

### 主实验

| 输入 | 方法 | LLRVD-simu PSNR/SSIM/LPIPS | REAL PSNR/SSIM/LPIPS |
|------|------|---------------------------|---------------------|
| sRGB | LightenDiffusion | 21.64/0.818/0.265 | 22.19/0.714/0.282 |
| RAW | BRVE | 27.58/0.817/0.137 | 21.87/0.717/0.334 |
| RAW | RID(NoiseModelling) | 26.76/0.825/0.127 | 22.72/0.729/0.258 |
| Event+sRGB | EvLight | 17.06/0.677/0.291 | 21.20/0.626/0.277 |
| **Event+RAW** | **NEC-Diff** | **27.74/0.828/0.125** | **24.51/0.742/0.201** |

### 消融实验

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 无 ECNS（仅 SRIE+CAD） | 21.06 | 0.653 | 0.278 |
| 无 SRIE（ECNS+CAD） | 23.24 | 0.698 | 0.243 |
| 无 CAD（ECNS+SRIE） | 22.53 | 0.671 | 0.265 |
| 完整模型 | **24.51** | **0.742** | **0.201** |

### 关键发现

- ECNS 贡献最大（去掉后 PSNR 降 3.45 dB），说明协同去噪是整个流程的基础
- 双 SNR 引导 vs 仅图像 SNR 引导多 0.43 dB，vs 直接融合多 0.76 dB
- 在 REAL 真实数据集上优势更明显（+1.79 dB vs 最佳 RAW 方法），因为真实噪声更复杂
- 在 0.001–0.3 lux 极暗场景中（占数据集 70%），优势尤为突出
- 事件去噪中同时使用跨模态输入和一致性损失效果最佳，单用任一均改善有限

## 亮点与洞察

- **物理驱动的跨模态去噪机制**是核心贡献：利用 RAW 光照的线性响应特性和事件 shot noise 与光照的正相关关系，建立了从物理出发的互助去噪框架。这比直接融合或后处理滤波高明很多
- **REAL 数据集**的构建很有价值——共轴成像系统 + 光学衰减模拟0.001 lux 极暗，47800 组像素对齐三元组（RAW/事件/GT），填补了事件-RAW 低光数据的空白
- **SNR 图作为融合权重**的做法简单有效，可推广到任意多模态融合场景

## 局限与展望

- 强度一致性损失中事件对比度阈值 $C$ 从数据学习，实际部署中不同事件相机阈值不同可能降低泛化性
- 扩散模型推理速度较慢（50 步 DDIM），实时应用受限
- 仅在 256×256 分辨率上训练和评测，高分辨率场景有待验证
- 未来可探索 test-time adaptation 适应不同事件相机参数

## 相关工作与启发

- **vs EvLight**: 仅用图像 SNR 引导融合，忽略了事件 SNR 在平滑暗区近零的问题。NEC-Diff 的双 SNR 策略更全面
- **vs ELEDNet/RETINEV**: 用低通滤波或 CNN 处理事件噪声，但简单滤波无法在抑噪和保细节间取平衡
- **vs EvRAW**: 关注事件-RAW 的细节和颜色恢复但忽略传感器噪声，在极暗下效果有限

## 评分

- 新颖性: ⭐⭐⭐⭐ 物理驱动的跨模态协同去噪思路新颖，但整体扩散框架与条件生成已较常见
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集充分对比，消融清晰，但缺少更多真实场景泛化评测
- 写作质量: ⭐⭐⭐⭐ 物理建模推导清晰，图示精美，但方法描述略冗长
- 价值: ⭐⭐⭐⭐ 数据集贡献大，方法在极暗成像领域有明确应用场景

<!-- RELATED:START -->

## 相关论文

- [Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](../../ECCV2024/image_restoration/edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)

<!-- RELATED:END -->
