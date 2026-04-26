---
title: >-
  [论文解读] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts
description: >-
  [ECCV 2024][图像恢复][测试时训练] 本文提出 TTT-MIM，在训练阶段联合优化监督去噪损失和自监督掩码图像建模（MIM）损失，在测试时通过最小化 MIM 自监督损失对单张噪声图像进行适应性微调，从而显著提升对分布外噪声（如真实相机噪声、显微镜噪声）的去噪性能，且速度远超零样本方法。
tags:
  - ECCV 2024
  - 图像恢复
  - 测试时训练
  - 掩码图像建模
  - 图像去噪
  - 分布偏移
  - 自监督学习
---

# TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts

**会议**: ECCV 2024  
**代码**: https://github.com/MLI-lab/TTT_Denoising  
**领域**: 图像复原  
**关键词**: 测试时训练, 掩码图像建模, 图像去噪, 分布偏移, 自监督学习

## 一句话总结
本文提出 TTT-MIM，在训练阶段联合优化监督去噪损失和自监督掩码图像建模（MIM）损失，在测试时通过最小化 MIM 自监督损失对单张噪声图像进行适应性微调，从而显著提升对分布外噪声（如真实相机噪声、显微镜噪声）的去噪性能，且速度远超零样本方法。

## 研究背景与动机

**领域现状**：端到端训练的神经网络在图像去噪任务上已取得 SOTA 性能，典型方法如 DnCNN、DRUNet 等在标准高斯噪声基准上表现优异。但这些模型高度依赖训练数据的噪声分布，一旦测试图像的噪声特性偏离训练分布，性能会急剧下降。

**现有痛点**：实际应用中，噪声分布偏移无处不在——医学影像（CT、MRI）、显微镜图像、智能手机相机的噪声特性与训练时使用的合成高斯噪声差异显著。现有去噪器对这种 distribution shift 几乎没有适应能力。零样本方法（如 DIP、ZS-N2N）虽然不依赖训练数据，但推理速度极慢，且性能不如有监督方法。

**核心矛盾**：有监督去噪器性能强但缺乏泛化性；零样本去噪器泛化性好但效率低、性能弱。二者之间需要一种既能利用训练数据的先验知识，又能在测试时自适应到新噪声分布的方案。

**本文目标** 如何在测试时仅使用单张噪声图像就完成网络的自适应微调，使得预训练去噪器能快速适应新的噪声分布？

**切入角度**：作者借鉴 NLP 和视觉领域的测试时训练（TTT）思想——在训练时引入辅助自监督任务，测试时通过该自监督任务的梯度更新来适配新分布。关键观察是：掩码图像建模（MIM）作为自监督任务与去噪任务天然相关——两者都在学习从损坏/不完整的输入中恢复原始信号。

**核心 idea**：在训练阶段联合监督去噪和自监督 MIM 两个任务，测试时仅用 MIM 损失对网络做少量梯度更新，即可快速适应单张测试图像的真实噪声分布。

## 方法详解

### 整体框架
TTT-MIM 包含两个阶段：（1）训练阶段：给定带噪-干净图像对，联合优化监督去噪损失 $\mathcal{L}_{denoise}$ 和自监督 MIM 重建损失 $\mathcal{L}_{MIM}$；（2）测试时适应阶段：给定一张分布外的噪声图像，仅通过最小化 $\mathcal{L}_{MIM}$ 对网络参数做少量梯度更新（通常 8 次迭代），然后用更新后的网络进行去噪。网络骨架使用 UNet 结构，以 Group Normalization 替代 Batch Normalization 以支持单图推理。

### 关键设计

1. **联合训练策略（Joint Training）**:

    - 功能：让网络同时学会去噪和从掩码中重建图像，建立两个任务之间的桥梁
    - 核心思路：训练损失为 $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \mathcal{L}_{MIM}$。对于输入的噪声图像，随机遮掩一定比例的 patch，网络需要同时预测干净图像（去噪）和被遮掩 patch 的内容（MIM）。两个任务共享同一个 encoder，使得网络在学习去噪的同时也学到了可用于测试时自适应的表示
    - 设计动机：MIM 任务不需要干净图像标签，因此可以作为测试时的锚点损失。通过联合训练，MIM 的梯度方向与去噪任务的梯度方向对齐，确保测试时通过 MIM 微调确实能改善去噪效果

2. **测试时适应机制（Test-Time Adaptation）**:

    - 功能：在测试时让预训练网络快速适应单张噪声图像的特定噪声模式
    - 核心思路：给定测试图像，随机生成掩码（mask ratio 通常为 0.01-0.5，patch size 为 1-14），将掩码区域置零，让网络预测被掩区域，计算 MIM 损失并反传更新参数。关键在于掩码比例和 patch 大小需要根据噪声类型调整——对于结构化噪声使用较大的 mask ratio 和 patch size，对于随机噪声使用较小值。经过 8 次迭代即可收敛
    - 设计动机：与 DIP 等需要几千次迭代的零样本方法不同，TTT-MIM 利用了预训练的先验，只需极少的梯度更新即可完成适应。MIM 任务迫使网络重新学习测试图像的局部统计特性，从而隐式适应新的噪声分布

3. **自适应去噪损失（Prediction-based Denoising Loss）**:

    - 功能：在测试时提供更鲁棒的去噪监督信号
    - 核心思路：除了标准的 MSE 去噪损失外，作者引入了基于网络自身预测的去噪损失 $\mathcal{L}_{pd}$，利用网络当前的预测作为伪目标。在测试时适应过程中，这个损失结合 MIM 损失一起优化，可以进一步稳定适应过程
    - 设计动机：在测试时没有真实干净图像，纯粹的 MIM 损失可能导致适应方向不稳定，引入基于网络自身预测的正则化可以约束更新幅度

### 损失函数 / 训练策略
训练阶段总损失：$\mathcal{L} = \mathcal{L}_{denoise} + \lambda \mathcal{L}_{MIM}$，其中去噪损失为 MSE，MIM 损失为被掩码区域的重建 MSE。测试时仅优化 $\mathcal{L}_{MIM}$ 或 $\mathcal{L}_{MIM} + \mathcal{L}_{pd}$。学习率在 $10^{-6}$ 到 $10^{-4}$ 之间，根据数据集调整。

## 实验关键数据

### 主实验

| 数据集/噪声 | 指标(PSNR) | TTT-MIM | DRUNet(无适应) | ZS-N2N | DIP |
|--------|------|------|----------|------|------|
| SIDD (真实相机噪声) | PSNR | **最优** | 性能退化 | 较慢 | 很慢 |
| PolyU (真实相机噪声) | PSNR | **最优** | 性能退化 | 较低 | 较低 |
| FMDD (显微镜噪声) | PSNR | **最优** | 性能退化 | 接近 | 较低 |
| fastMRI (模拟高斯) | PSNR | **最优** | 退化 | 较低 | 很慢 |
| ImageNet (椒盐噪声) | PSNR | **显著提升** | 严重退化 | 较低 | - |

### 消融实验

| 配置 | PSNR 变化 | 说明 |
|------|---------|------|
| Full model (MIM + pd loss) | 最优 | 完整模型 |
| w/o MIM (仅监督去噪) | 无法测试时适应 | 没有自监督锚点 |
| w/o joint training | 效果降低 | MIM 梯度与去噪梯度未对齐 |
| 不同 mask ratio | 0.01-0.5 最优 | 依噪声类型而异 |
| 不同 iteration 数 | 8 次已收敛 | 更多迭代提升有限 |

### 关键发现
- TTT-MIM 的测试时适应仅需 8 次梯度更新，比 DIP（需 >1000 次迭代）快几个数量级
- 在分布内数据上，TTT-MIM 不会损害性能（因为 MIM 微调幅度很小）
- 对真实噪声（SIDD、PolyU）的适应效果尤为显著，验证了方法在实际场景中的价值
- mask ratio 和 patch size 需要根据噪声类型调整：结构化噪声用大 patch，随机噪声用小 patch

## 亮点与洞察
- **MIM 与去噪的天然关联**是本文最巧妙之处——两个任务都是从部分观测重建完整信号，这种 task alignment 使得 MIM 可以作为测试时适应的代理任务。这个思路可以推广到其他图像复原任务（超分辨率、去模糊等）
- 仅 8 次迭代就能完成适应，实现了"有监督性能 + 零样本泛化性"的最佳折中。这个效率优势来源于联合训练阶段已经对齐了 MIM 和去噪的梯度空间
- Group Normalization 的使用是支持单图适应的关键细节——BN 在 batch size=1 时统计量不稳定

## 局限与展望
- mask ratio 和 patch size 需要根据噪声类型手动调整，缺乏自动选择机制
- 目前仅验证了 UNet 骨架，是否适用于 Transformer 架构（如 Restormer）有待验证
- 批量适应模式（batch TTT）与单图模式的超参数不同，缺乏统一的适应策略
- 对于极端分布偏移（如训练于高斯噪声、测试于运动模糊），适应效果可能有限

## 相关工作与启发
- **vs DIP/Deep Decoder**: 零样本方法完全不需要训练数据，但需要从头优化（>1000 迭代），TTT-MIM 利用预训练先验只需 8 次迭代
- **vs DRUNet/SwinIR**: 有监督方法训练集内性能更强，但无法适应新分布，TTT-MIM 补上了这块短板
- **vs TTT (Sun et al.)**: 原始 TTT 使用旋转预测作为辅助任务，TTT-MIM 的创新在于选择 MIM 作为与去噪更相关的辅助任务
- **vs MAE**: MAE 用于视觉表示学习，本文将 MIM 用于低层视觉任务的测试时适应，是一个新的应用方向

## 评分
- 新颖性: ⭐⭐⭐⭐ MIM 作为去噪代理的测试时训练是新颖的组合，但 TTT 框架本身不新
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种噪声类型和数据集，消融也较完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述流畅
- 价值: ⭐⭐⭐⭐ 解决了实际部署中噪声分布偏移的核心问题，8 次迭代的效率非常实用

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[ECCV 2024\] Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)
- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)

<!-- RELATED:END -->
