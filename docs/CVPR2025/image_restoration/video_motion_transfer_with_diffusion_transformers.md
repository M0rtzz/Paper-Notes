---
title: >-
  [论文解读] DiTFlow: Video Motion Transfer with Diffusion Transformers
description: >-
  [CVPR 2025][图像恢复][运动迁移] DiTFlow提出了首个专为扩散Transformer(DiT)设计的运动迁移方法，通过分析跨帧注意力图提取Attention Motion Flow(AMF)作为逐patch的运动信号，以无训练的优化方式引导新视频生成复现参考视频的运动模式。
tags:
  - CVPR 2025
  - 图像恢复
  - 运动迁移
  - Transformer
  - 注意力运动流
  - 位置嵌入优化
  - 零样本
---

# DiTFlow: Video Motion Transfer with Diffusion Transformers

**会议**: CVPR 2025  
**arXiv**: [2412.07776](https://arxiv.org/abs/2412.07776)  
**代码**: [ditflow.github.io](https://ditflow.github.io)  
**领域**: 图像恢复 / 视频运动迁移  
**关键词**: 运动迁移, 扩散Transformer, 注意力运动流, 位置嵌入优化, 零样本

## 一句话总结

DiTFlow提出了首个专为扩散Transformer(DiT)设计的运动迁移方法，通过分析跨帧注意力图提取Attention Motion Flow(AMF)作为逐patch的运动信号，以无训练的优化方式引导新视频生成复现参考视频的运动模式。

## 研究背景与动机

视频扩散模型虽然能生成逼真的视频内容，但仅通过文本提示难以精确控制运动——文本在描述细粒度时序内容变化时固有地存在歧义。运动迁移通过使用参考视频作为运动引导解决这一问题，但现有方法大多基于UNet架构，使用分离的时间注意力和空间注意力，无法充分利用DiT的优势。

DiT通过全时空注意力机制联合处理时空信息，相比UNet-based方法可以提取更高质量的运动信息。然而，在DiT的全时空注意力中解纠缠运动模式与内容更加困难。现有UNet-based方法（如SMM的空间平均、MOFT的运动通道发现）假设分离的时间注意力，不适用于DiT。

DiTFlow的核心洞察：DiT中相似语义内容的token会跨帧相互关注，可以利用跨帧注意力关系构建patch级的位移图来捕获运动模式。

## 方法详解

### 整体框架

DiTFlow分为两步：(1) 从参考视频中提取AMF运动信号——将参考视频送入预训练DiT的特定transformer block，分析跨帧注意力构建位移矩阵；(2) 在生成新视频的去噪过程中，通过优化隐变量或位置嵌入使生成视频的AMF匹配参考AMF。

### 关键设计1: Attention Motion Flow (AMF)提取

- **功能**: 从DiT的跨帧注意力中提取逐patch的运动位移信号
- **核心思路**: 将参考视频编码为隐表示$z_{\text{ref}}$，在$t=0$步通过DiT的第$n$个block提取平均后的Q和K。计算跨帧注意力$A_{i,j}^{\otimes} = \sigma(\tau \frac{Q_i K_j^T}{\sqrt{d_k}})$，用argmax为帧$i$中每个patch找到帧$j$中最关注的对应patch坐标，构建位移矩阵$\Delta_{i,j}[(u,v)] = (u'-u, v'-v)$。聚合所有帧对构成AMF
- **设计动机**: 在$t=0$时特征分析比高噪声步骤产生更清晰的运动信号，避免了昂贵的DDIM反演。使用argmax而非softmax产生更干净的位移图

### 关键设计2: AMF引导的隐变量优化

- **功能**: 通过优化去噪过程中的隐变量使生成视频复现参考运动
- **核心思路**: 在去噪过程中，对当前步隐变量$z_t$通过DiT提取$\tilde{Q}$和$\tilde{K}$，使用soft argmax（加权求和）计算可微的位移矩阵$\tilde{\Delta}_{i,j}$以保持梯度。最小化AMF损失$\mathcal{L}_{\text{AMF}} = \|\text{AMF}(z_{\text{ref}}) - \text{AMF}(z_t)\|_2^2$，在前20%去噪步中每步优化$K_{\text{opt}}=5$步
- **设计动机**: 优化隐变量直接修改内容生成以匹配参考运动，在所有指标上性能最优

### 关键设计3: 位置嵌入优化实现零样本迁移

- **功能**: 通过优化DiT的位置嵌入$\rho$实现运动迁移的泛化，无需对每个新视频重复优化
- **核心思路**: 将AMF损失的梯度反传到位置嵌入$\rho_t$而非隐变量$z_t$。位置嵌入负责编码patches的时空位置，操纵位置信息引导patch的重新组织以实现运动迁移。一旦优化完成，学习到的嵌入可直接应用于新提示词的生成
- **设计动机**: 位置嵌入与内容解耦，操纵位置信息不影响内容编码，具有更好的泛化能力。这是UNet-based方法不可能实现的特性

### 损失函数

AMF损失：$\mathcal{L}_{\text{AMF}}(z_{\text{ref}}, z_t) = \|\text{AMF}(z_{\text{ref}}) - \text{AMF}(z_t)\|_2^2$，在所有帧对的位移矩阵上计算元素级欧氏距离。

## 实验关键数据

### 主实验: CogVideoX-5B上的运动迁移评估

| 方法 | MF(Caption)↑ | MF(Subject)↑ | MF(Scene)↑ | MF(All)↑ | IQ(All)↑ |
|------|-------------|-------------|------------|---------|---------|
| Backbone | 0.524 | 0.502 | 0.544 | 0.523 | 0.315 |
| MotionClone | 0.635 | 0.640 | 0.628 | 0.634 | 0.318 |
| SMM | 0.782 | 0.741 | 0.776 | 0.766 | 0.315 |
| MOFT | 0.728 | 0.728 | 0.722 | 0.726 | 0.318 |
| **DiTFlow** | **0.790** | **0.775** | **0.789** | **0.785** | **0.319** |

### CogVideoX-2B上的对比

| 方法 | MF(All)↑ | IQ(All)↑ |
|------|---------|---------|
| SMM | 0.688 | 0.312 |
| MOFT | 0.504 | 0.312 |
| **DiTFlow** | **0.726** | **0.317** |

### 关键发现

- DiTFlow在运动保真度(MF)上一致取得最优，在5B模型上为0.785 vs SMM的0.766
- SMM在Subject提示下MF显著下降(0.741 vs Caption 0.782)，表明空间平均特征与参考内容纠缠
- DiTFlow在所有提示类型下表现一致，证明运动与内容的更好解纠缠
- 人类评估中DiTFlow在运动和提示一致性上都显著优于基线
- 位置嵌入优化虽性能略低于隐变量优化，但支持免优化的零样本迁移

## 亮点与洞察

1. **首个DiT专属运动迁移方法**：充分利用DiT全时空注意力的优势，提取逐patch运动信号
2. **位置嵌入优化的创新洞察**：利用DiT独有的位置编码机制实现零样本运动迁移，这在UNet架构中不可能
3. **运动-内容解纠缠的逐patch方法**：相比SMM的空间平均和MOFT的独立位置偏差，AMF捕获了patch间的显式时空关系

## 局限与展望

- 隐变量优化模式下每个视频需要额外的优化时间（8分钟 vs 5分钟基线）
- 位置嵌入优化的零样本模式性能仍有差距
- 运动迁移的细粒度控制（如仅迁移部分运动）尚未探索
- 未来可将AMF引导与其他条件控制结合

## 相关工作与启发

- **SMM**: 基于空间平均的运动特征，全局性导致语义修改困难
- **MOFT**: 发现扩散特征中的运动通道，但各位置独立处理容易引导错误元素
- **MotionClone**: UNet-based的注意力运动迁移，但假设分离的时间注意力不适用于DiT
- 启发：DiT的全时空注意力虽然使运动解纠缠更难，但也蕴含了更丰富的时空关联信息

## 评分

⭐⭐⭐⭐ — 首次为DiT架构设计运动迁移方法，AMF概念简洁有效。位置嵌入优化的零样本能力是独特贡献。在两个规模的模型和多种提示类型上的一致优势证明了方法的鲁棒性。

<!-- RELATED:START -->

## 相关论文

- [Closed-Loop Transfer for Weakly-supervised Affordance Grounding](../../ICCV2025/image_restoration/closed-loop_transfer_for_weakly-supervised_affordance_grounding.md)
- [Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)
- [NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](../../CVPR2026/image_restoration/nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [BAMM: Bidirectional Autoregressive Motion Model](../../ECCV2024/image_restoration/bamm_bidirectional_autoregressive_motion_model.md)
- [Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)

<!-- RELATED:END -->
