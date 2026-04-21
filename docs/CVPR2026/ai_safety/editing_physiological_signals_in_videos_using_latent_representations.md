---
title: >-
  [论文解读] Editing Physiological Signals in Videos Using Latent Representations
description: >-
  [CVPR 2026][AI安全][心率编辑] 提出PhysioLatent框架，将输入面部视频编码到3D VAE潜空间，与目标心率CLIP文本嵌入融合，通过AdaLN增强的时空融合层捕捉rPPG时间相干性，结合FiLM调制解码器和微调输出层实现精确心率修改，在保持PSNR 38.96dB/SSIM 0.98的视觉质量下达到10 bpm MAE的心率调制精度。
tags:
  - CVPR 2026
  - AI安全
  - 心率编辑
  - rPPG隐私
  - 视频生理信号
  - 3D VAE
  - 生物特征匿名化
  - FiLM
  - AdaLN
---

# Editing Physiological Signals in Videos Using Latent Representations

**会议**: CVPR 2026  
**arXiv**: [2509.25348](https://arxiv.org/abs/2509.25348)  
**代码**: 有（论文承诺公开）  
**领域**: AI安全 / 隐私保护  
**关键词**: 心率编辑, rPPG隐私, 视频生理信号, 3D VAE, 生物特征匿名化, FiLM, AdaLN

## 一句话总结

提出PhysioLatent框架，将输入面部视频编码到3D VAE潜空间，与目标心率CLIP文本嵌入融合，通过AdaLN增强的时空融合层捕捉rPPG时间相干性，结合FiLM调制解码器和微调输出层实现精确心率修改，在保持PSNR 38.96dB/SSIM 0.98的视觉质量下达到10 bpm MAE的心率调制精度。

## 研究背景与动机

**领域现状**：摄像头远程光电容积脉搏波(rPPG)技术可从面部视频无接触提取心率，是远程健康监测的关键技术。但这也意味着面部视频中不可见地嵌入了敏感生理信息，可被算法提取用于隐蔽的健康推断、情绪监控和生物特征画像。

**现有痛点**：

1. 现有视觉隐私方法（人脸模糊/替换）可能意外破坏或保留PPG信号，且不专门处理生理信息维度的隐私
2. 面部替换同时引入新的身份线索——在隐私保护场景中不理想
3. PulseEdit等直接在像素空间操作的方法时间相干性差(Low temporal coherency)
4. 无法精确地将心率修改到目标值（如统一为固定HR做匿名化）

**核心矛盾**：rPPG信号嵌入在微妙的皮肤颜色变化中(<1%像素值变化)，肉眼不可见但算法可提取——需要在完全不影响视觉外观的前提下精确修改这种"看不见"的信号。

**本文目标** 构建一个可控的视频生理信号编辑框架——能将心率精确修改到任意目标值，同时保持高视觉保真度。

**切入角度**：在3D VAE潜空间做心率编辑——利用潜空间的压缩性做精确调制 + 兼容视频生成管线 + 时空融合层捕捉rPPG周期特性。

**核心 idea**：3D VAE潜空间编辑 + AdaLN时间注意力条件化 + FiLM解码器微调 = 精确可控的心率修改。

## 方法详解

### 整体框架

输入面部视频(128帧, 72×72) → 冻结3D Causal VAE编码器得潜表示z → 冻结CLIP ViT-B/32编码目标HR prompt(如"Heart rate 80 bpm")得条件c → 线性投影c匹配z尺寸并通道拼接 → **可训练时空融合层**(空间卷积+时间自注意力+AdaLN注入HR条件) → 3D Causal VAE解码器(FiLM条件化+微调输出层) → Haar Cascade人脸检测得面部mask → 仅替换面部区域 → 输出视频(视觉不变, HR已修改)。

### 关键设计

1. **3D VAE潜空间选择**

    - 选择3D Causal VAE（空间4×下采样、时间8×下采样）而非2D VAE：rPPG信号本质是跨帧的时间序列→需3D表示捕捉时空相干性
    - 与Latent Diffusion/Video Diffusion潜空间兼容→可直接作为生成视频管线的后处理模块
    - CLIP文本嵌入(512维)经线性投影后与视频潜表示通道拼接→统一的条件化接口

2. **AdaLN时空融合层**

    - 包含可分解的空间-时间自注意力模块——显式建模rPPG的强时间相干性（心率是周期信号，需跨帧一致调制）
    - AdaLN（Adaptive Layer Normalization）将HR条件注入时间注意力流——在归一化参数中编码目标频率信息
    - 对比naive方案：仅用(2+1)D卷积捕捉局部模式但无法建模长程时间依赖→HR调制不准确
    - 设计动机：rPPG变化极微妙(亚像素级颜色波动, <1%像素值)→需要精细的长程时间条件化

3. **FiLM解码器条件化 + 微调输出层**

    - FiLM（Feature-wise Linear Modulation）在3D VAE解码器中间层注入HR条件——CLIP投影的HR嵌入生成缩放和偏移参数调制解码器激活
    - 仅微调解码器输出层（而非整个解码器）→保持预训练视觉重建能力
    - 设计动机：标准3D VAE解码器优化的是通用视频重建，会在重建时"擦除"微妙的rPPG调制→FiLM显式保持生理信号修改，微调输出层适配这种极精细的变化

4. **面部区域替换策略**

    - 使用Haar Cascade人脸检测器生成面部mask M
    - 仅将解码器输出中的面部区域替换到原视频中，其余像素保持不变→进一步保证非面部区域的完美保真

### 损失函数 / 训练策略

- **视觉保真损失**：$\mathcal{L}_F = \text{MSE}(v, \hat{v}) + \text{LPIPS}(v, \hat{v})$
- **波形损失**（形态引导）：$\mathcal{L}_{\text{wave}} = 1 - \text{Pearson}(rPPG(\hat{v}), \sin(2\pi f t))$，f=HR_d/60，引导rPPG趋向目标频率的平滑周期波形
- **频率损失**（精确对齐）：$\mathcal{L}_{\text{freq}} = |f - f_{\text{pred}}|$，f_pred由FFT获取
- 课程学习策略：前10 epoch仅用视觉+波形损失→第10 epoch起线性ramp频率损失权重 $\beta(t) = 0.005(t-10)$
- **总损失**：$\mathcal{L} = 0.2\mathcal{L}_{\text{wave}} + \beta(t)\mathcal{L}_{\text{freq}} + 1.0\mathcal{L}_F$
- 训练：4×RTX4090, batch=4, 30 epochs, AdamW, OneCycle lr=5e-4, 输入128帧×72×72

## 实验关键数据

### 主实验（跨数据集，POS估计器）

| 数据集 | 目标HR | PSNR↑(dB) | SSIM↑ | 输入MAE(bpm) | 输出MAE↓(bpm) | 输出MAPE↓(%) |
|--------|--------|-----------|-------|-------------|--------------|-------------|
| PURE | 60 bpm | 39.04 | 0.98 | 38.95 | 9.22 | 8.20 |
| PURE | 80 bpm | 39.02 | 0.98 | 41.80 | 9.98 | 10.55 |
| PURE | 100 bpm | 38.85 | 0.98 | 44.67 | 10.41 | 10.29 |
| PURE | 120 bpm | 38.94 | 0.98 | 50.63 | 10.36 | 11.34 |
| **PURE Avg** | - | **38.96** | **0.98** | 44.01 | **10.00** | **10.09** |
| UBFC Avg | - | 40.09 | 0.98 | 26.77 | 11.08 | 10.57 |
| MMPD Avg | - | 37.50 | 0.95 | 44.58 | 9.84 | 8.09 |

### Benchmark对比（目标HR=120 bpm, POS估计器）

| 数据集 | 方法 | PSNR↑(dB) | SSIM↑ | 输出MAE↓(bpm) | 输出MAPE↓(%) |
|--------|------|-----------|-------|--------------|-------------|
| PURE | **PhysioLatent** | 38.94 | **0.9761** | **10.36** | **11.34** |
| PURE | PulseEdit | **42.68** | 0.9720 | 16.71 | 12.26 |
| UBFC | **PhysioLatent** | 40.04 | 0.9803 | **11.18** | **10.15** |
| UBFC | PulseEdit | **43.08** | **0.9867** | 15.07 | 15.56 |
| MMPD | **PhysioLatent** | 37.87 | 0.9542 | **10.75** | **7.96** |
| MMPD | PulseEdit | **41.72** | **0.9664** | 20.36 | 18.30 |

### 关键发现

- PSNR>38 dB + SSIM≥0.95 = 对人眼完全不可见的修改→成功的"看不见的隐私保护"
- HR调制MAE约10 bpm，比PulseEdit(16-20 bpm)低6-10 bpm——时间相干性建模的优势
- PulseEdit在PSNR上更高(42-43 vs 38-40)但HR准确度差(MAE 15-20 vs 10-11)——trade-off不同
- 7种rPPG估计器（PCA/POS/CHROM/TSCAN/DeepPhys/PhysNet/PhysFormer++）均被成功误导至目标HR——验证了方法的鲁棒性
- MMPD（多肤色/多光照）上PSNR略低(37.5)但HR准确度反而最好(MAE 9.84)——说明HR编辑不依赖特定外观条件

## 亮点与洞察

- **"看不见的隐私"问题定义精准**：首次专门针对视频中不可见的生理信号进行可控编辑——人脸模糊/替换无法解决的隐私维度
- **3D VAE兼容视频生成管线**：在Latent Diffusion/Video Diffusion的潜空间工作→可直接作为生成视频框架的后处理模块
- **匿名化+合成的双重价值**：匿名化(固定HR=60)保护隐私；合成(指定任意HR)为rPPG研究生成带标注的训练数据
- **微妙信号编辑的工程方案**：rPPG变化<1%像素值→AdaLN长程时间条件化 + FiLM解码器显式保持 + 微调输出层适配精度的三层保障

## 局限与展望

- HR MAE约10 bpm对精确健康监测应用仍嫌大——需进一步提高频率对齐精度
- 当前仅处理心率→呼吸率/血压/血氧等生理信号的编辑待探索
- 3D VAE编解码本身引入微妙视觉变化（PSNR≈39不是无限），高频区域(边缘/纹理)可见局部失真
- 未验证运动鲁棒性（Table 1中motion robustness列标记为No）
- PulseEdit在PSNR上更高(+3-4dB)——说明VAE重建本身是视觉质量的瓶颈

## 相关工作与启发

- **vs PulseEdit**：像素空间直接操作，PSNR更高但时间相干性低(Low)且HR准确度差(MAE 15-20)；PhysioLatent在潜空间操作，HR准确度好2倍(MAE 10)但PSNR低3-4dB
- **vs Privacy-Phys**：3D CNN直接修改rPPG，但时间相干性低且无法精确控制目标HR
- **vs Wang et al.**：GAN生成条件化rPPG视频，但无法处理信号移除
- 启发：潜空间编辑的范式可推广到其他不可见信号的控制（如视频中的音频水印、运动模式等）；3D VAE+CLIP条件化的接口设计天然兼容现有生成管线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视频中不可见的生理信号编辑是全新问题，潜空间编辑方案优雅
- 实验充分度: ⭐⭐⭐⭐ 3个数据集、7种rPPG估计器验证、与PulseEdit的系统对比、消融齐全
- 写作质量: ⭐⭐⭐⭐ 问题动机("看不见的隐私")清晰有力，rPPG两大特性(时间相干+视觉不可见)的图示直观
- 价值: ⭐⭐⭐⭐ 对AI隐私保护和rPPG研究数据生成均有贡献，3D VAE兼容性有工程价值

<!-- RELATED:START -->

## 相关论文

- [Can Editing LLMs Inject Harm?](../../AAAI2026/ai_safety/can_editing_llms_inject_harm.md)
- [Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection](../../ECCV2024/ai_safety/any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)
- [Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](../../ACL2025/ai_safety/private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)

<!-- RELATED:END -->
