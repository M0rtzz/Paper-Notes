---
title: >-
  [论文解读] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts
description: >-
  [ICCV 2025][自动驾驶][低光增强] 首次将混合专家（MoE）网络引入低光图像增强任务，通过三个专门的子专家网络分别处理颜色修复、细节增强和高级特征增强，并利用动态门控机制自适应调整各专家的权重，在5个基准数据集上取得了SOTA的PSNR表现。
tags:
  - ICCV 2025
  - 自动驾驶
  - 低光增强
  - 混合专家网络
  - 门控机制
  - U-Net
  - 多尺度特征融合
---

# GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts

**会议**: ICCV 2025  
**arXiv**: [2503.07417](https://arxiv.org/abs/2503.07417)  
**代码**: https://github.com/Sameenok/gm-moe-lowlight-enhancement.git  
**领域**: autonomous_driving / 低光图像增强  
**关键词**: 低光增强, 混合专家网络, 门控机制, U-Net, 多尺度特征融合

## 一句话总结

首次将混合专家（MoE）网络引入低光图像增强任务，通过三个专门的子专家网络分别处理颜色修复、细节增强和高级特征增强，并利用动态门控机制自适应调整各专家的权重，在5个基准数据集上取得了SOTA的PSNR表现。

## 研究背景与动机

低光图像增强（LLIE）在自动驾驶、3D重建、遥感、监控等领域有广泛应用。现有方法存在三大问题：

**全局-局部信息不平衡**：CNN方法难以学习全局光照分布，Transformer过度关注全局信息导致颜色失真

**跨域泛化能力不足**：现有方法通常在特定数据集上训练，面对未知光照条件时性能急剧下降

**多问题耦合优化困难**：噪声、颜色失真、细节模糊相互耦合，单一模型难以协同优化——抑制噪声可能牺牲细节，提亮暗区可能放大颜色失真

## 方法详解

### 整体框架

GM-MoE基于改进的U-Net架构，输入暗图像 $I \in \mathbb{R}^{H \times W \times 3}$ 首先经过浅层特征提取模块（SFEB）得到低级特征 $X_0$，然后通过编码器逐层下采样提取深层特征，解码器通过上采样和pixel-shuffle逐步恢复分辨率。GM-MoE模块被嵌入到编码器和解码器的每一层中，负责融合编码器的低级特征和解码器的高级特征。最终输出为残差图像 $R$，增强图像 $\hat{I} = I + R$。

### 关键设计

1. **动态门控权重生成网络**：输入图像经过自适应平均池化转为特征向量，再通过两层全连接网络生成三个专家网络的权重 $S = [s_1, s_2, s_3]$，其中 $s_1 + s_2 + s_3 = 1$。这使得网络能够根据不同数据域的图像（不同场景和光照特征）动态调整参数，最终输出为加权和：$\tilde{X}_i = s_1 X_{i-1}^1 + s_2 X_{i-1}^2 + s_3 X_{i-1}^3$。

2. **颜色修复专家网络（Expert1/Net1）**：采用池化操作聚焦关键颜色特征，利用反卷积恢复图像细节，使用非线性插值确保颜色过渡平滑自然。通过残差连接保留原始图像特征，最后用Sigmoid激活函数将输出限制在 $[0,1]$，减少颜色异常和过饱和问题。

3. **细节增强专家网络（Expert2/Net2）**：结合通道注意力机制和空间注意力机制。通道注意力提取重要通道特征，空间注意力结合Max Pooling和Avg Pooling聚焦关键空间位置。两种注意力的输出通过拼接和残差连接融合，增强细节恢复能力。

4. **高级特征增强专家网络（Expert3/Net3）**：通过多尺度卷积提取并融合特征，再经过门控网络（SG）和通道注意力机制（SCA）处理，最后通过残差连接添加回输入，提升整体图像质量。

5. **浅层特征提取模块（SFEB）**：使用 $3 \times 3$ 深度可分离卷积生成 $F_1$ 和空洞卷积（不同膨胀率）生成 $F_2$ 捕获多尺度空间信息。通过全局池化生成通道加权特征 $A_{avg}$ 和 $A_{max}$，再通过 $7 \times 7$ 卷积生成注意力图：$F_w = F_1' \odot A_{avg} + F_2' \odot A_{max}$，最终输出 $Y = X \odot F_w$。

### 损失函数 / 训练策略

采用PSNR Loss作为损失函数，定义为：

$$\text{PSNR loss} = -\frac{10}{\log(10)} \cdot \log(\text{MSE} + \epsilon)$$

其中 $\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{I}(i) - I_{gt}(i))^2$，$\epsilon$ 为防止分母为零的小正数。

训练细节：PyTorch框架，NVIDIA 4090 GPU，初始学习率 $1.0 \times 10^{-3}$，Adam优化器（momentum=0.9），输入resize至 $256 \times 256$，batch size=4，共 $2.0 \times 10^6$ 次迭代。

## 实验关键数据

### 主实验 (表格)

在LOL-v1、LOLv2-Real、LOLv2-Synthetic三个数据集上与25+方法对比：

| 方法 | LOL-v1 PSNR | LOL-v1 SSIM | LOLv2-Real PSNR | LOLv2-Real SSIM | LOLv2-Syn PSNR | LOLv2-Syn SSIM | 参数量(M) |
|------|-------------|-------------|-----------------|-----------------|----------------|----------------|-----------|
| Retinexformer | 25.16 | 0.845 | 22.80 | 0.840 | 25.67 | 0.930 | 1.61 |
| DPEC | 24.80 | 0.855 | 22.89 | 0.863 | 26.19 | 0.939 | 2.58 |
| LLFormer | 25.76 | 0.823 | 20.06 | 0.792 | 24.04 | 0.909 | 24.55 |
| SNR-Net | 24.61 | 0.842 | 21.48 | 0.849 | 24.14 | 0.928 | 39.12 |
| **GM-MoE (Ours)** | **26.66** | **0.857** | **23.65** | 0.806 | **26.30** | **0.937** | 19.99 |

在LSRW-Huawei/Nikon数据集上：

| 方法 | LSRW-Huawei PSNR | LSRW-Huawei SSIM | LSRW-Nikon PSNR | LSRW-Nikon SSIM |
|------|-------------------|-------------------|------------------|------------------|
| Restormer | 22.61 | 0.725 | 21.20 | 0.677 |
| DRBN | 20.61 | 0.710 | 21.07 | 0.670 |
| **GM-MoE (Ours)** | **23.55** | **0.741** | **22.62** | **0.700** |

### 消融实验 (表格)

在LOL-v2-real和LOL-v2-syn数据集上逐步添加模块：

| 配置 | LOLv2-real PSNR | LOLv2-real SSIM | LOLv2-syn PSNR | LOLv2-syn SSIM |
|------|-----------------|-----------------|----------------|----------------|
| Baseline | 19.45 | 0.7079 | 20.35 | 0.7431 |
| +SFEB | 20.27 | 0.7236 | 23.44 | 0.7646 |
| +SFEB+Net1 | 21.35 | 0.7446 | 24.35 | 0.8436 |
| +SFEB+Net1+Net2 | 22.11 | 0.8021 | 25.14 | 0.9327 |
| +SFEB+Net1+Net2+Net3 | 23.35 | 0.8055 | 26.15 | 0.9366 |
| **完整模型(+GM)** | **23.65** | **0.8060** | **26.29** | **0.9371** |

### 关键发现

- SFEB在LOLv2-syn上直接带来3.09 dB的PSNR提升，说明浅层特征提取的重要性
- 三个专家网络各自贡献互补，移除任一都导致性能下降
- 门控机制在完整模型中额外带来约0.3 dB的提升，验证了动态权重调整对跨域泛化的有效性
- 在LSRW高噪声数据集上相比Restormer分别提升0.94 dB和1.42 dB，展示了强噪声环境下的优势

## 亮点与洞察

- **首次将MoE引入LLIE**：将低光增强的多个子问题（颜色修复、细节恢复、特征增强）解耦为独立专家，是一个自然且有效的设计
- **动态门控机制**使模型能跨数据域自适应调整，避免了固定权重可能导致的次优解
- 在5个benchmark上取得5个PSNR第一和4个SSIM第一，泛化性优异
- 参数量19.99M，介于轻量和重量之间，平衡了性能和效率

## 局限与展望

- LOLv2-Real上的SSIM（0.806）略低于DPEC（0.863）和SNR-Net（0.849），结构保持能力有待提升
- 门控机制仅使用Softmax生成三个权重，缺少对具体像素或区域的空间自适应
- 仅使用PSNR Loss训练，未结合感知损失 / SSIM Loss / 对抗损失等，限制了视觉质量上限
- 未在视频或实时场景中验证，自动驾驶应用仍需进一步验证实时性

## 相关工作与启发

- 与Retinexformer（1.61M参数）和DPEC（2.58M参数）等轻量模型相比，GM-MoE参数量偏大但性能更优
- MoE思路可推广到其他图像恢复任务（去雾、去雨、超分辨率），将不同退化问题分配给不同专家
- 门控机制的设计可以借鉴Sparse MoE（如Switch Transformer），仅激活部分专家以降低计算量

## 评分

- **新颖性**: ⭐⭐⭐ 首次将MoE引入LLIE是亮点，但各子专家的设计较为常规
- **实验充分度**: ⭐⭐⭐⭐ 5个数据集25+对比方法，消融实验充分
- **写作质量**: ⭐⭐⭐ 结构清晰，但部分公式和描述冗余
- **价值**: ⭐⭐⭐⭐ 展示了MoE在底层视觉任务中的潜力，实验结果有说服力

<!-- RELATED:START -->

## 相关论文

- [ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](../../AAAI2026/autonomous_driving/expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving](od-rase_ontology-driven_risk_assessment_and_safety_enhancement_for_autonomous_dr.md)
- [MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception](maestro_task-relevant_optimization_via_adaptive_feature_enhancement_and_suppress.md)
- [LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes](../../CVPR2025/autonomous_driving/limoe_mixture_of_lidar_representation_learners_from_automotive_scenes.md)
- [LightLoc: Learning Outdoor LiDAR Localization at Light Speed](../../CVPR2025/autonomous_driving/lightloc_learning_outdoor_lidar_localization_at_light_speed.md)

<!-- RELATED:END -->
