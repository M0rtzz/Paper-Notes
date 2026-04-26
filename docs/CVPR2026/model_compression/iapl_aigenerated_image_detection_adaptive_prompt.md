---
title: >-
  [论文解读] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
description: >-
  [CVPR 2026][模型压缩][IAPL] 提出IAPL（Image-Adaptive Prompt Learning），在CLIP编码器输入端引入动态prompt——由条件信息学习器（从纹理丰富区域提取伪造特异和通用线索）和测试时token调优（通过多视角一致性最小化熵）两条路径生成，使模型能在推理时根据每张测试图像自适应调整，在未见过的生成器上显著提升检测泛化性。
tags:
  - CVPR 2026
  - 模型压缩
  - IAPL
  - 提示学习
  - test-time adaptation
  - CLIP
  - forgery detection
---

# Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning

**会议**: CVPR 2026  
**arXiv**: [2508.01603](https://arxiv.org/abs/2508.01603)  
**代码**: 有  
**领域**: AI生成图像检测  
**关键词**: IAPL, image-adaptive prompt, test-time adaptation, CLIP, forgery detection

## 一句话总结

提出IAPL（Image-Adaptive Prompt Learning），在CLIP编码器输入端引入动态prompt——由条件信息学习器（从纹理丰富区域提取伪造特异和通用线索）和测试时token调优（通过多视角一致性最小化熵）两条路径生成，使模型能在推理时根据每张测试图像自适应调整，在未见过的生成器上显著提升检测泛化性。

## 研究背景与动机

**AI生成图像检测**面临的核心挑战是**泛化到未见生成器**。GAN和扩散模型不断涌现，训练数据只能覆盖有限的生成方法，模型需要能检测训练时完全未见过的生成器产生的图像。

**现有方案**主要通过微调视觉基础模型（如CLIP）来增强检测能力。这些方法利用了预训练模型编码的丰富真实世界知识，作为对训练数据中有限伪造模式的补充。但**微调后的参数是固定的**——面对来自不同生成器的测试图像时，固定参数无法捕捉每张图像特有的判别特征。不同生成器产生的图像在纹理、语义和视觉伪影方面差异显著，单一固定模型难以全面应对。

**IAPL的切入角度**：让输入到编码器的prompt在推理时**动态调整**——不是训练后固定不变，而是根据每张测试图像的特征自适应调整。这通过两条互补路径实现：（1）从图像提取的条件信息提供instance-specific线索；（2）测试时token调优通过多视角一致性约束来对齐参数。核心idea是在保持模型稳定backbone的同时，让少量参数具有per-instance的灵活性。

## 方法详解

### 整体框架

在冻结的CLIP ViT-L/14上添加三类可训练参数：（1）MLP-based adapter插入到 $N_a$ 个编码器block中（训练后固定）；（2）可学习token加入第2到第 $N_t$ 个block（训练后固定）；（3）Image-Adaptive Prompt加入第1个block作为输入（推理时动态调整）。最终CLS token通过分类器输出检测结果。

### 关键设计

1. **条件信息学习器（Conditional Information Learner）**:

    - 功能：从每张输入图像中提取instance-specific的伪造线索
    - 核心思路：将输入图像分成 $N_p=192$ 个小patch，用DCT分数选取纹理最丰富的patch。经高通滤波器提取高频模式后，送入两个独立CNN提取**伪造特异条件** $C_f$（有辅助监督分支引导）和**通用条件** $C_g$（无监督分支学习）
    - 设计动机：CLIP的高层语义特征可能遗漏低层伪造痕迹（如频域异常、纹理不一致），通过高通滤波+DCT选择纹理丰富区域来补充底层伪造信号。条件信息是per-instance的，为每张图像提供不同的检测引导

2. **测试时Token调优（Test-Time Token Tuning）**:

    - 功能：在推理时根据单张测试图像动态调整token参数
    - 核心思路：对测试图像生成 $N_v=32$ 个多视角view（1个全局resize + 31个随机裁剪），选取置信度最高的 $m=6$ 个view，最小化**平均熵损失** $L_{avg} = -(\bar{p} \log \bar{p} + (1-\bar{p})\log(1-\bar{p}))$ 来调优test-time adaptive token（仅2步，学习率 $5\times10^{-3}$）
    - 设计动机：域偏移使训练时学到的prompt在未见数据上不最优，测试时通过多视角预测一致性约束来适配当前图像。熵最小化不需要标签，是test-time adaptation的标准做法

3. **可学习缩放因子融合**:

    - 功能：融合条件信息和测试时token生成最终的image-adaptive prompt
    - 核心思路：$P = \{\alpha_f \cdot C_f + A[0,:], \alpha_g \cdot C_g + A[1,:]\}$，其中 $\alpha_f, \alpha_g$ 是channel-wise可学习系数，控制两种信息的贡献权重。在后续block中，上一层prompt通过类似的缩放因子与learnable token融合
    - 设计动机：条件信息和调优token各有优势（前者提供instance-specific低层线索，后者通过优化提供高层对齐），缩放因子让模型自适应决定两者的贡献比例

### 损失函数 / 训练策略

训练损失：$L_{overall} = L_{cls} + L_{aux}$，两者均为BCE。 $L_{cls}$ 是最终分类损失，$L_{aux}$ 是条件信息学习器中伪造特异分支的辅助监督。测试时用 $L_{avg}$ 调优test-time token。训练只需**1个epoch**，单GPU（3090）。

## 实验关键数据

### 主实验

**UniversalFakeDetect（ProGAN训练，19个子集测试）**：

| 方法 | mAcc↑ | mAP↑ |
|------|-------|------|
| UniFD | 75.4 | 79.5 |
| C2P-CLIP | 91.4 | 95.6 |
| FatFormer | 92.7 | 95.4 |
| **IAPL** | **95.61** | **97.8** |

**GenImage（SD v1.4训练，8个生成器测试）**：

| 方法 | mAcc↑ |
|------|-------|
| C2P-CLIP | 93.1 |
| **IAPL** | **96.7** |

### 消融实验

| 配置 | mAcc | 说明 |
|------|------|------|
| 仅固定参数（无动态prompt） | 较低 | 缺乏instance适应 |
| +条件信息学习器 | 提升 | 低层线索有帮助 |
| +测试时token调优 | 进一步提升 | 域偏移适配有效 |
| **完整IAPL** | **95.61** | 两条路径互补最优 |

### 关键发现

- T-SNE可视化显示：IAPL产生的unseen fake特征与seen fake特征更相似、与real特征分离更清晰——说明动态prompt确实提升了对新生成器的泛化
- 仅训练1 epoch即可达到SOTA，说明CLIP的预训练知识+少量灵活参数足以捕获伪造模式
- 从纹理最丰富的patch提取条件信息是一个高效的设计——只处理一个32×32的小patch，计算开销极小

## 亮点与洞察

- "固定backbone + 动态prompt"的设计哲学很优雅：大部分参数提供稳定的表征能力，少量参数（仅2个prompt token）提供per-instance的灵活性。这种架构层面的分工比全模型微调或全参数测试时适应更高效且稳定。
- 测试时token调优只需2步优化6个view就足够——说明域偏移的信号很强，决策边界的微调不需要大量迭代，这对实际部署的延迟控制很重要。

## 局限与展望

- 测试时调优增加了推理延迟（需要多次前向传播+梯度回传），对实时检测场景可能不适用
- 仅用DCT选择单个纹理丰富patch提取条件信息，可能遗漏其他区域的伪造线索
- 当前仅处理二分类（真/假），未扩展到生成器溯源任务

## 相关工作与启发

- **vs C2P-CLIP/FatFormer**: 这些方法在训练后固定所有参数，IAPL在推理时动态调整prompt，泛化性提升3-4%
- **vs TPT（Test Prompt Tuning）**: IAPL是TPT在伪造检测领域的扩展和升级——加入了条件信息学习器作为补充通道，提供了CLIP本身无法捕获的低层伪造线索

## 评分

- 新颖性: ⭐⭐⭐⭐ 条件信息+测试时token调优的双路径动态prompt设计新颖
- 实验充分度: ⭐⭐⭐⭐ 两个标准benchmark、消融全面、T-SNE可视化有说服力
- 写作质量: ⭐⭐⭐⭐ 流程图清晰、数学符号规范
- 价值: ⭐⭐⭐⭐ 在AI生成内容检测日益重要的背景下很有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)

<!-- RELATED:END -->
