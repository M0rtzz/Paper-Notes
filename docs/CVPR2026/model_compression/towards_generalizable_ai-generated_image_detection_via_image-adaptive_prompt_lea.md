---
title: >-
  [论文解读] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
description: >-
  [CVPR 2026][模型压缩][AI生成图像检测] 提出 Image-Adaptive Prompt Learning (IAPL)，在推理时根据每张测试图像动态调整 CLIP 编码器的 prompt，通过测试时 token 调优和条件信息学习器实现对未见生成器的强泛化，在 UniversalFakeDetect 和 GenImage 上分别达到 95.61% 和 96.7% 平均准确率的 SOTA 性能。
tags:
  - CVPR 2026
  - 模型压缩
  - AI生成图像检测
  - 提示学习
  - 测试时适应
  - CLIP
  - 伪造检测
---

# Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning

**会议**: CVPR 2026  
**arXiv**: [2508.01603](https://arxiv.org/abs/2508.01603)  
**代码**: 有  
**领域**: 模型压缩  
**关键词**: AI生成图像检测, 提示学习, 测试时适应, CLIP, 伪造检测

## 一句话总结

提出 Image-Adaptive Prompt Learning (IAPL)，在推理时根据每张测试图像动态调整 CLIP 编码器的 prompt，通过测试时 token 调优和条件信息学习器实现对未见生成器的强泛化，在 UniversalFakeDetect 和 GenImage 上分别达到 95.61% 和 96.7% 平均准确率的 SOTA 性能。

## 研究背景与动机

**领域现状**: AI 生成图像检测是当前安全领域的热门课题。SOTA 方法普遍微调 CLIP 等视觉基础模型，利用其丰富的预训练知识辅助检测。现有方法如 UniFD、FatFormer、C2P-CLIP 在训练后固定所有可学习参数。

**现有痛点**: 微调后的固定参数模型对未见生成器的域迁移抵抗力不足。不同生成器产出的图像在纹理、语义和伪造痕迹上差异巨大，固定参数无法捕获这些实例级别的特异性判别线索。

**核心矛盾**: 训练数据只涵盖有限的生成方法（如仅用 ProGAN 训练），但推理时要面对 19 种不同的生成器。固定学到的 prompt 只编码了训练集的伪造分布，无法适应新分布。

**本文目标** (1) 如何让 prompt 在推理时动态适应每张测试图像？(2) 如何提取图像特有的伪造线索作为条件信息？(3) 如何在保持检测骨干稳定性的同时允许实例级自适应？

**切入角度**: 将测试时适应（Test-Time Adaptation）思想引入 prompt 学习——prompt 不仅在训练时优化，在推理时也根据单张测试图像的多视角一致性约束继续调优。

**核心 idea**: prompt 由"训练后固定的条件信息"和"推理时动态调整的 test-time token"两部分组成，通过可学习缩放因子融合，实现检测器的实例级自适应。

## 方法详解

### 整体框架

基于 CLIP ViT-L/14 构建检测管道。在原有 CLIP 编码器中插入三类可训练组件：(1) MLP-based adapters（等间隔插入 $N_a=6$ 个 block）；(2) Learnable tokens（第 2 到 $N_t=9$ 个 block）；(3) Image-adaptive prompts（第 1 个 block 输入）。前两者训练后固定，提供稳定骨干；后者在推理时继续动态调整。最终 CLS token 经分类器输出检测结果。

### 关键设计

1. **Test-Time Token Tuning**:

    - 功能：在推理时根据单张测试图像调整 test-time adaptive tokens
    - 核心思路：从测试图像生成 $N_v=32$ 个不同视角（1 个全局 + 31 个局部裁剪+翻转），用置信度选择 $m=6$ 个高置信视角。以最小化平均熵损失 $L_{avg} = -(\bar{p} \log \bar{p} + (1-\bar{p})\log(1-\bar{p}))$ 为目标调优 token 参数 $T=2$ 步，其中 $\bar{p}$ 是所有选中视角预测的平均。这迫使模型在多视角下做出一致预测
    - 设计动机：域迁移导致预测不确定性增大，通过多视角一致性约束可以在无标签条件下让 token 适应当前图像的特性

2. **Conditional Information Learner**:

    - 功能：从输入图像的纹理丰富区域提取伪造特有和通用条件信息
    - 核心思路：将图像切成 $N_p=192$ 个 $32 \times 32$ 小块，用 DCT 分数选纹理最丰富的块，经高通滤波器提取高频模式。两个结构相同但参数独立的 CNN 分别提取伪造特有条件 $C_f$（附加辅助监督）和通用条件 $C_g$（无监督）
    - 设计动机：CLIP 预训练关注高层语义，易忽略低层伪造痕迹（频率异常、像素模式等）。条件信息从高频纹理入手正好弥补这一短板。两路分离让一路专注伪造判别、一路捕获通用图像状态

3. **Learnable Scaling Factor**:

    - 功能：融合 test-time tokens 和条件信息为最终的 image-adaptive prompt
    - 核心思路：$P = \{\alpha_f \cdot C_f + A[0,:], \alpha_g \cdot C_g + A[1,:]\}$，其中 $\alpha_f, \alpha_g$ 是可学习的逐通道系数，训练时学到最优融合比例
    - 设计动机：条件信息和自适应 token 各捕获不同类型的线索，缩放因子实现细粒度通道级控制

### 损失函数 / 训练策略

训练损失：$L_{overall} = L_{cls} + L_{aux}$，均为二分类交叉熵。推理阶段用平均熵 $L_{avg}$ 调优 test-time tokens。训练仅 1 个 epoch，学习率 $5 \times 10^{-5}$，单卡 3090。推理时 test-time tuning 学习率 $5 \times 10^{-3}$，调优 2 步。还有 Optimal Input Selection——对同一张图的多个视角取最高置信度的预测作为最终结果。

## 实验关键数据

### 主实验（UniversalFakeDetect，ProGAN 4-class 训练, Acc%）

| 方法 | ProGAN | StyleGAN | BigGAN | LDM(200) | DALLE | GauGAN | mAcc |
|------|--------|----------|--------|----------|-------|--------|------|
| UniFD | 100.0 | 82.0 | 94.5 | 72.0 | 81.38 | 99.5 | 86.78 |
| FatFormer | 99.89 | 97.15 | 99.50 | 69.45 | 98.75 | 99.41 | 90.86 |
| C2P-CLIP | 99.98 | 96.44 | 99.12 | 93.29 | 98.55 | 99.17 | 93.79 |
| **IAPL** | **100.0** | **98.90** | **99.65** | **95.35** | **98.90** | **99.55** | **95.61** |

### 消融实验

| 配置 | mAcc | 说明 |
|------|------|------|
| Full IAPL | 95.61 | 完整方法 |
| w/o test-time tuning | 93.89 | 去掉推理时调优掉 1.72 |
| w/o conditional info | 94.23 | 去掉条件信息掉 1.38 |
| w/o scaling factor | 94.67 | 去掉缩放因子掉 0.94 |
| w/o MLP adapter | 94.12 | 去掉适配器掉 1.49 |

### 关键发现

- Test-time tuning 贡献最大（+1.72%），证实推理时自适应的有效性
- T-SNE 可视化定性展示 IAPL 对未见伪造图像的特征与已见伪造更接近、与真实图像更分离
- 在 GenImage 数据集上用 SD v1.4 训练达 96.7% mAcc，对 Midjourney、ADM 等未见生成器泛化良好
- 仅需 1 epoch 训练 + 推理时 2 步调优，训练效率极高

## 亮点与洞察

- **推理时 prompt 自适应**：将 test-time adaptation 引入 prompt learning 用于伪造检测是一个新颖组合。每张图都有定制化的 prompt，比固定 prompt 更能适应未见域
- **高频纹理条件化**：从 DCT 分数最高的小块提取高频条件信息，巧妙弥补了 CLIP 语义偏向的短板，且计算量很小（仅处理一个 32x32 块）
- **极低训练成本**：仅 1 epoch + 单卡 3090，比同类方法（如 FatFormer 需多 epoch 和更大卡）经济得多

## 局限与展望

- 推理时 test-time tuning 引入额外延迟：需生成 32 个视角 + 2 步梯度更新，对实时应用可能是瓶颈
- 条件信息仅从单个纹理最丰富的块提取，可能遗漏分布在多处的伪造线索
- 在 SITD、SAN 等低层次伪造方法上准确率仍有波动（68-95%），说明条件信息对某些伪造类型捕获不足

## 相关工作与启发

- **vs C2P-CLIP**: C2P-CLIP 通过对比学习注入类别概念，prompt 训练后固定。IAPL 额外引入推理时调优，mAcc 从 93.79% 提升到 95.61%
- **vs FatFormer**: FatFormer 用频率分析增强适配器，但 prompt 固定。IAPL 动态 prompt + 条件信息双管齐下效果更好
- **vs TPT/R-TPT**: 本文借鉴了 test-time prompt tuning 的思路但加入了伪造检测特有的条件信息分支，比纯 TPT 更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 test-time adaptation 与 prompt learning 结合用于伪造检测是新颖组合
- 实验充分度: ⭐⭐⭐⭐⭐ 两大标准数据集、19+ 生成器、完整消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对 AI 生成内容检测的实际应用有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_an.md)

<!-- RELATED:END -->
