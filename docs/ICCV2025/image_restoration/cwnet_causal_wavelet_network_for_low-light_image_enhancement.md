---
title: >-
  [论文解读] CWNet: Causal Wavelet Network for Low-Light Image Enhancement
description: >-
  [ICCV 2025][图像恢复][低光图像增强] 提出因果小波网络CWNet，通过结构因果模型将低光增强中的语义信息视为因果因子、亮度/颜色退化视为非因果因子，结合小波变换骨干网络实现频域特征的精细化恢复。
tags:
  - ICCV 2025
  - 图像恢复
  - 低光图像增强
  - 因果推理
  - 小波变换
  - 状态空间模型
  - CLIP语义一致性
---

# CWNet: Causal Wavelet Network for Low-Light Image Enhancement

**会议**: ICCV 2025  
**arXiv**: [2507.10689](https://arxiv.org/abs/2507.10689)  
**代码**: CWNet（论文提及，具体链接未给出）  
**领域**: 图像修复 / 低光增强  
**关键词**: 低光图像增强, 因果推理, 小波变换, 状态空间模型, CLIP语义一致性

## 一句话总结

提出因果小波网络CWNet，通过结构因果模型将低光增强中的语义信息视为因果因子、亮度/颜色退化视为非因果因子，结合小波变换骨干网络实现频域特征的精细化恢复。

## 研究背景与动机

传统低光图像增强（LLIE）方法主要关注均匀亮度调整，忽略了实例级语义信息和不同频率特征的内在特性。现有频域方法将高频和低频特征统一处理，限制了增强效果。同时，许多方法在增强亮度的同时难以保持颜色和语义一致性，导致视觉上不自然或语义不准确的结果。

本文从两个关键问题出发：（1）如何在改善照明条件的同时确保颜色和语义信息的一致性？（2）如何建立充分利用频域特征的鲁棒模型？现有基于CLIP的方法只关注全局语义一致性，缺乏实例级一致性保证；小波方法没有充分利用频域的独特特性。

## 方法详解

### 整体框架

CWNet基于结构因果模型（SCM）构建，将低光增强任务建模为因果推理问题。整体架构采用U-Net形式，包含上采样/下采样层和层级特征恢复块（HFRB），HFRB由特征提取模块（FE）、高频增强块（HFEB）和低频增强块（LFEB）三个核心组件组成。

### 关键设计

1. **因果推理分析与度量学习（Causal Inference）**: 核心思路是将低光场景中的语义信息$\mathcal{S}$定义为因果因子，颜色和亮度异常$\mathcal{U}$定义为非因果因子。通过两种"有意义且无害"的干预方式获取非因果因子：光照退化干预 $I_l = \frac{I}{L}L^{\gamma} + \varepsilon$（基于物理光照模型）和颜色异常干预（色调/饱和度/RGB通道偏移）。在全局层面，采用因果引导的度量学习策略：处理后的低光图像作为锚点，对应正常光照图像作为正样本，不同场景的反事实扰动样本作为负样本。损失函数为 $\mathcal{L}_{ca} = \frac{\mathcal{L}_1(F_p, \hat{F})}{\xi(\sum_l \mathcal{L}_1(F_l, \hat{F}) + \sum_c \mathcal{L}_1(F_c, \hat{F}))}$。这种设计动机在于：迫使模型学习光照不变的语义特征，从退化因素中分离出真正的语义内容。

2. **实例级CLIP语义损失（Instance-Level CLIP Semantic Loss）**: 利用预训练的HRNet提取语义实例分割图，将增强后的图像分割成多个子实例图像$I_{seg}^k$，然后通过CLIP编码器计算每个实例与文本提示（"low light"/"normal light"）的语义一致性概率 $\hat{y} = \frac{1}{K}\sum_{k=1}^{K}\frac{e^{\cos(\Phi_{image}(I_{seg}^k), \Phi_{text}(T_{low}))}}{...}$。使用交叉熵损失优化语义一致性。动机在于ATE分析显示不同语义区域对退化的敏感度差异显著，全局一致性不足以保证局部语义完整性。

3. **小波变换骨干网络（Wavelet-based Backbone）**: FE模块通过小波变换将输入分解为$\{L, H, V, D\}$四个频率子带。低频分量用WTConv（大感受野无额外参数复杂度）提取，高频分量用深度可分离卷积加方向对齐卷积（H-Conv/V-Conv/D-Conv）提取，并通过低频到高频的信息补偿。HFEB基于Mamba设计了HF-Mamba模块，包含三个方向对齐的2D-SSM：H-2D-SSM处理水平高频、V-2D-SSM处理垂直高频、D-2D-SSM处理对角高频，而非统一扫描。LFEB基于快速傅里叶卷积（FFC）处理低频分量，利用两个残差块提供全局上下文感知的大感受野特征增强。

### 损失函数 / 训练策略

总损失由五部分加权组合：
$$\mathcal{L}_{total} = \lambda_1\mathcal{L}_2 + \lambda_2\mathcal{L}_{ssim} + \lambda_3\mathcal{L}_{per} + \lambda_4\mathcal{L}_{ca} + \lambda_5\mathcal{L}_{sem}$$
权重设置为$[1.0, 0.3, 0.2, 0.01, 0.01]$。使用Adam优化器（$\beta_1=0.9, \beta_2=0.99$），初始学习率$4\times10^{-4}$，训练$3\times10^5$次迭代，批量大小8，输入随机裁剪为$256\times256$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CWNet | 之前SOTA | 提升 |
|--------|------|-------|----------|------|
| LOL-v1 | PSNR/SSIM/LPIPS | 23.60/0.8496/0.0648 | Wave-Mamba: 22.76/0.8419/0.0791 | +0.84/+0.0077/-0.0143 |
| LOL-v2-Real | PSNR/SSIM/LPIPS | 27.39/0.9005/0.0383 | Wave-Mamba: 27.87/0.8935/0.0451 | SSIM+0.007/LPIPS-0.007 |
| LOL-v2-Syn | PSNR/SSIM/LPIPS | 25.50/0.9362/0.0195 | RetinexMamba: 25.89/0.9346/0.0389 | LPIPS大幅降低 |
| LSRW-Huawei | PSNR/SSIM/LPIPS | 21.50/0.6397/0.1562 | DMFourLLIE: 21.09/0.6328/0.1804 | +0.41/+0.007/-0.024 |

模型参数仅1.23M，FLOPs 11.3G，远小于MIRNet（31.79M）和SNR-Aware（39.12M）。

### 消融实验

| 配置 | PSNR | SSIM | LPIPS | 说明 |
|------|------|------|-------|------|
| CWNet完整 | 21.53 | 0.6423 | 0.1631 | 基线 |
| w/o 因果推理 | 20.87 | 0.6375 | 0.1781 | PSNR下降0.66 |
| w/o FE | 20.98 | 0.6387 | 0.1804 | 频域提取缺失 |
| w/o HFEB | 20.58 | 0.6317 | 0.1903 | 高频增强缺失 |
| w/o LFEB | 20.41 | 0.6302 | 0.1985 | 低频增强最关键 |
| WTConv→Conv | 21.42 | 0.6415 | 0.1690 | 频域卷积优于标准卷积 |
| HF-Mamba→VMamba | 21.20 | 0.6394 | 0.1735 | 方向对齐SSM优于通用SSM |
| 语义图→全局特征 | 21.48 | 0.6417 | 0.1652 | 实例级指导有效 |

### 关键发现

- LFEB移除导致最大性能下降（PSNR降至20.41），表明低频处理在双分支架构中最为关键
- 因果推理机制贡献显著（PSNR降0.66），验证了因果视角对语义/退化分离的有效性
- 在LOL-v1训练、LOL-v2-Real测试的跨数据集场景中SSIM达0.9005，泛化能力突出
- 模型在极低参数量（1.23M）下实现SOTA，效率优势明显

## 亮点与洞察

- 将因果推理引入低光增强是一个新颖且有说服力的视角，通过SCM清晰地定义了因果/非因果因子
- 实例级CLIP语义损失比全局CLIP损失更精细，ATE分析为其提供了令人信服的理论支撑
- HF-Mamba的方向对齐扫描策略与小波高频分量的物理含义天然匹配（水平/垂直/对角），设计优雅
- 模型极其轻量（1.23M参数），在性能和效率之间取得了极好的平衡

## 局限与展望

- 面对复合退化（如低光+模糊+雾霾）时恢复质量欠佳
- 因果干预策略依赖于特定的退化模型假设，可能不完全覆盖真实世界的退化类型
- CLIP和HRNet等预训练模型的引入增加了推理时的额外计算开销

## 相关工作与启发

- 因果推理在低级视觉任务中的应用仍然较少，本文的SCM建模范式可推广到其他图像恢复任务
- 方向对齐的SSM扫描策略可扩展到其他需要方向感知的视觉任务中
- 频域（小波/傅里叶）与因果推理的结合为图像增强提供了新的设计范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 因果推理+小波变换的组合在LLIE中是新颖的
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、完整消融、跨数据集泛化
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，因果框架的阐述层层递进
- 价值: ⭐⭐⭐⭐ 轻量高效且性能SOTA，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](../../ECCV2024/image_restoration/towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](../../ECCV2024/image_restoration/unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [Efficient Diffusion as Low Light Enhancer (ReDDiT)](../../CVPR2025/image_restoration/efficient_diffusion_as_low_light_enhancer.md)

<!-- RELATED:END -->
