---
title: >-
  [论文解读] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification
description: >-
  [CVPR 2025][图像生成][对抗净化] 提出基于注意力掩码的异构噪声扩散净化策略，对分类器关注的关键像素施加高强度噪声以消除对抗扰动，对其余区域施加低强度噪声以保留语义信息，并通过单步重采样大幅降低计算开销。
tags:
  - CVPR 2025
  - 图像生成
  - 对抗净化
  - 扩散模型
  - 异构噪声
  - 注意力掩码
  - 对抗防御
---

# Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification

**会议**: CVPR 2025  
**arXiv**: [2503.01407](https://arxiv.org/abs/2503.01407)  
**代码**: [GitHub](https://github.com/GaozhengPei/Purification)  
**领域**: 图像生成/对抗鲁棒性  
**关键词**: 对抗净化, 扩散模型, 异构噪声, 注意力掩码, 对抗防御

## 一句话总结

提出基于注意力掩码的异构噪声扩散净化策略，对分类器关注的关键像素施加高强度噪声以消除对抗扰动，对其余区域施加低强度噪声以保留语义信息，并通过单步重采样大幅降低计算开销。

## 研究背景与动机

现有基于扩散模型的对抗净化方法（如 DiffPure）在前向过程中对所有像素均匀施加相同强度噪声，然后通过逆向过程恢复干净图像。然而这种统一操作存在根本性矛盾：

- **噪声过强**：能消除对抗扰动，但同时破坏语义信息，导致分类器对干净和对抗样本都产生错误预测
- **噪声过弱**：无法有效去除对抗扰动，攻击仍然成功
- 如何在消除扰动和保留语义之间取得平衡是核心挑战

从神经网络可解释性角度，分类器在做决策时会对图像不同区域分配不同关注度。对抗扰动在高关注度区域的影响最大。因此可以有针对性地对不同区域施加不同强度的噪声。

此外，现有多步重采样方法计算成本极高，难以在消费级 GPU 上评估强自适应攻击的完全梯度，这也是亟待解决的问题。

## 方法详解

### 整体框架

给定对抗图像 $\boldsymbol{x}_{adv}$，首先通过分类器前向传播获取各层注意力图，构建二值注意力掩码 $\mathcal{M}$；然后执行异构前向过程将高/低噪声分别加到关注/非关注区域；最后通过两阶段去噪过程恢复干净图像。

### 关键设计一：异构前向过程（Heterogeneous Forward Process）

- **功能**：对图像不同区域施加不同强度噪声，兼顾扰动消除和语义保留
- **核心思路**：提取分类器各 block 的激活输出，经 $L^p$ 范数聚合、上采样、spatial softmax 归一化后以阈值 $\tau$ 二值化，取所有 block 掩码并集得到 $\mathcal{M} = \bigcup_{m=1}^{M} \mathbb{I}[\text{AM}_m > \tau]$。对高关注区域施加大时间步 $t_l$ 的强噪声，低关注区域施加小时间步 $t_s$ 的弱噪声，融合为 $\boldsymbol{x}(t_l, t_s) = \boldsymbol{x}(t_l) \odot \mathcal{M} + \boldsymbol{x}(t_s) \odot (1-\mathcal{M})$
- **设计动机**：对抗扰动在分类器高关注区域影响最大，高强度噪声可有效消除；低关注区域对分类结果影响较小，弱噪声就足够且能保留语义信息

### 关键设计二：两阶段去噪过程（Two-stage Denoising Process）

- **功能**：从异构噪声图像中恢复干净图像
- **核心思路**：Stage 1（$t_s < t < t_l$）转化为 inpainting 问题——掩码外区域从原始图像直接前向加噪得到已知像素 $\boldsymbol{x}(t)^{known}$，掩码内区域由扩散模型逆向预测 $\boldsymbol{x}(t)^{unknown}$，两者通过掩码合并；Stage 2（$t < t_s$）噪声强度统一，执行标准去噪采样
- **设计动机**：异构噪声图像中不同区域噪声水平不同，无法直接用标准去噪。先用 inpainting 消除局部高强度噪声，再用全局标准去噪消除弱噪声

### 关键设计三：单步重采样（Single-step Resampling）

- **功能**：解决掩码边界的语义不一致同时大幅降低计算开销
- **核心思路**：将 RePaint 中重复 $U=20$ 次的多步重采样替换为：先将 $\boldsymbol{x}(t)$ 一步扩散到 $\boldsymbol{x}(t+U)$，再用 DDIM 一步去噪回 $\boldsymbol{x}(t)$，每个时间步仅需额外调用一次去噪网络
- **设计动机**：多步重采样需存储大量计算图，GPU 显存和时间成本极高；单步重采样实现约 90% 的时间和显存节省，使方法可在 24GB GPU 上评估全梯度自适应攻击

### 损失函数

无额外训练，直接使用预训练扩散模型进行推理时净化。

## 实验关键数据

### 主实验：CIFAR-10 $\ell_\infty$ ($\epsilon=8/255$) WideResNet-28-10

| 方法 | 类型 | Standard Acc. | Robust (AutoAttack) |
|------|------|--------------|-------------------|
| Gowal et al. | AT | 88.54 | 63.38 |
| Bai et al. | AP | 91.41 | 77.08 |
| Lee et al. | AP | 90.16 | 70.47 |
| Lin et al. | AP | 90.62 | 72.85 |
| **Ours** | AP | **93.16** | **80.45** |

### CIFAR-10 $\ell_\infty$ WideResNet-70-16

| 方法 | 类型 | Standard Acc. | Robust (AutoAttack) |
|------|------|--------------|-------------------|
| Rebuffi et al. (AT†) | AT | 92.22 | 66.56 |
| Bai et al. | AP | 92.97 | 79.10 |
| Lin et al. | AP | 91.99 | 76.37 |
| **Ours** | AP | **93.36** | **84.83** |

### 消融实验（效率对比）

| 重采样方式 | 语义一致性 | 额外网络调用次数/步 | GPU 显存需求 |
|-----------|----------|------------------|------------|
| 多步重采样 $U=20$ | ✓ | 20 | 极高 |
| 单步重采样 $U=10$ | ✓ | 1 | 24GB 可用 |
| 无重采样 | ✗ (边界伪影) | 0 | 最低 |

### 关键发现

- WRN-70-16 上 AutoAttack 鲁棒准确率 84.83%，超越 AT 最佳 (66.56%) 和 AP 最佳 (79.10%)
- 标准准确率 93.36% 同样优于所有对比方法，说明净化过程未损坏干净样本
- 单步重采样在 $U=10$ 即可实现语义一致性，计算开销降低约 90%

## 亮点与洞察

1. **异构噪声思想巧妙**：利用分类器自身注意力作为先验指导扩散模型差异化处理，将"消除扰动 vs 保留语义"的全局矛盾转化为空间分离
2. **Inpainting 视角去噪**：将异构噪声去噪优雅地转化为图像修复问题，理论合理且实现简洁
3. **单步重采样工程价值大**：首次使扩散净化方法可在消费级 GPU 上计算完整自适应攻击梯度

## 局限与展望

- 注意力掩码质量依赖分类器架构，不同网络可能需要不同的阈值 $\tau$ 和噪声水平 $t_l, t_s$
- 仅在图像分类任务上验证，未扩展到检测、分割等任务
- 若攻击者知道掩码生成方式，可能设计针对性攻击绕过防御
- 对高分辨率图像的扩展性有待验证

## 相关工作与启发

- **DiffPure**：首个扩散净化方法，但使用均匀噪声
- **GDMP**：引入对比损失引导，仍受限于均匀噪声
- **RePaint**：多步重采样解决边界问题，但计算代价极高
- 异构策略的思想可推广到其他需要区域差异化处理的图像恢复任务

## 评分

⭐⭐⭐⭐ — 异构噪声思路新颖且直觉清晰，实验结果大幅超越现有方法。单步重采样的工程贡献实用性强。但方法需分类器注意力图作为额外输入，增加了部署复杂度。

<!-- RELATED:START -->

## 相关论文

- [Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)
- [Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)

<!-- RELATED:END -->
