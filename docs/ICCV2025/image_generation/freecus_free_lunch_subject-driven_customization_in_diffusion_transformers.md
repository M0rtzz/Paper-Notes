---
title: >-
  [论文解读] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers
description: >-
  [ICCV 2025][图像生成][主体驱动定制] 本文提出 FreeCus，一个完全免训练的主体驱动定制框架，通过关键注意力共享机制、改进的动态偏移特征提取和多模态大语言模型语义增强三大创新，激活扩散 Transformer（DiT）的内在零样本主体定制能力，达到与需要额外训练的方法相当甚至更优的效果。
tags:
  - ICCV 2025
  - 图像生成
  - 主体驱动定制
  - Transformer
  - 免训练
  - 注意力共享
  - 零样本生成
---

# FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2507.15249](https://arxiv.org/abs/2507.15249)  
**代码**: [https://github.com/Monalissaa/FreeCus](https://github.com/Monalissaa/FreeCus)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 主体驱动定制, 扩散Transformer, 免训练, 注意力共享, 零样本生成

## 一句话总结

本文提出 FreeCus，一个完全免训练的主体驱动定制框架，通过关键注意力共享机制、改进的动态偏移特征提取和多模态大语言模型语义增强三大创新，激活扩散 Transformer（DiT）的内在零样本主体定制能力，达到与需要额外训练的方法相当甚至更优的效果。

## 研究背景与动机

**领域现状**：随着文本到图像（T2I）扩散模型的突破性进展，特别是 Flux 系列扩散 Transformer 的出现，主体驱动定制（Subject-driven Customization）成为热门方向——给定参考图像中的主体，在新的文本描述场景中保持主体身份一致性地生成图像。

**现有痛点**：现有方法主要分为两类：(1) 逐主体优化方法（如 DreamBooth、Textual Inversion）需要对每个新主体进行微调训练，耗时且不可扩展；(2) 基于编码器的方法（如 IP-Adapter、ELITE）需要在大规模数据集上预训练专用的主体特征提取编码器。两类方法都依赖于训练步骤，从根本上限制了实际应用的灵活性和部署效率。

**核心矛盾**：现代扩散 Transformer（如 Flux 系列）已经在预训练中学到了丰富的视觉-语义对应关系，理论上具备零样本主体合成的潜力，但现有方法未能充分挖掘这种内在能力，反而依赖外部训练来弥补。

**本文目标**：设计一个真正免训练的框架，通过操控 DiT 内部的注意力机制和特征表示来实现高质量主体定制，无需任何额外训练或微调。

**切入角度**：作者发现 DiT 的注意力结构中蕴含了主体布局和细粒度特征的信息——通过在生成过程中适当共享参考图像的注意力特征，可以将主体身份"注入"到新场景中。

**核心 idea**：提出三管齐下的免训练策略——(1) pivotal attention sharing 传递主体布局，(2) 改进的 dynamic shifting 提取细粒度特征，(3) MLLM 增强跨模态语义表示——共同激活 DiT 的零样本定制能力。

## 方法详解

### 整体框架

FreeCus 的 pipeline 分为两条并行路径：(1) 参考路径——将参考图像输入 DiT 提取关键的注意力特征和细粒度视觉特征；(2) 生成路径——在去噪过程中将参考路径的特征注入到目标生成中。整个过程不修改模型参数，仅操控中间表示。输入是一张参考主体图像和文本 prompt，输出是在新场景中保持主体身份的生成图像。

### 关键设计

1. **Pivotal Attention Sharing（关键注意力共享）**:

    - 功能：将参考图像的主体布局信息传递到生成图像中，同时保持编辑灵活性
    - 核心思路：在 DiT 的去噪过程中，将参考图像的 self-attention 中的 Key 和 Value 特征注入到生成图像的对应注意力层中。关键创新在于"pivotal"选择策略——不是所有时间步和层都进行共享，而是选择性地在关键时间步（布局形成阶段）和关键层（低频结构层）进行注入。这样既能保持主体的整体布局完整性，又不会过约束细节，保留了文本引导的编辑灵活性
    - 设计动机：直接复制所有注意力会导致生成图像完全复制参考图，丧失编辑能力；不共享则无法传递主体身份。Pivotal 策略在两者之间取得平衡

2. **Upgraded Dynamic Shifting（改进的动态偏移特征提取）**:

    - 功能：从参考图像中提取更精细的主体细节特征
    - 核心思路：Flux 系列 DiT 使用 dynamic shifting 机制来调节去噪过程中的噪声调度。作者通过分析这一机制发现，调整 shifting 参数可以让模型在不同分辨率的特征空间中提取更多细粒度信息。具体地，提出了一个升级版 dynamic shifting 变体，通过修改噪声调度参数，使得在参考路径中模型更关注细节纹理（如毛发、纹理图案、材质属性），而非仅捕捉粗粒度轮廓
    - 设计动机：标准的 dynamic shifting 为生成任务优化，但主体定制需要更精确的细节保持。通过简单的参数调整即可显著提升细粒度特征提取，不增加任何计算成本——真正的"free lunch"

3. **MLLM 语义增强（多模态大语言模型语义表示增强）**:

    - 功能：丰富跨模态语义信息，弥补纯视觉特征在语义理解上的不足
    - 核心思路：将参考图像输入先进的多模态大语言模型（如 GPT-4V 或类似模型），获取对主体的详细文本描述（包括主体类别、颜色、材质、姿态等属性），将这些描述与用户 prompt 融合后作为增强的文本条件输入 DiT。这样 DiT 在去噪时不仅有视觉特征引导，还有更丰富的语义理解
    - 设计动机：DiT 的文本编码器对简单 prompt 的理解可能不足以精确定义主体属性。MLLM 提供的详细描述能补充缺失的语义信息，尤其在主体具有复杂纹理或特殊属性时效果显著

### 损失函数 / 训练策略

完全免训练，无需任何损失函数设计。所有操作在推理阶段通过特征操控完成。

## 实验关键数据

### 主实验

| 方法 | 训练需求 | DINO-I ↑ | CLIP-I ↑ | CLIP-T ↑ | 说明 |
|------|---------|----------|----------|----------|------|
| DreamBooth | 逐主体微调 | 较高 | 较高 | 中等 | 身份保持好但需微调 |
| IP-Adapter | 预训练编码器 | 中等 | 中等 | 较高 | 文本对齐好但身份弱 |
| ELITE | 预训练编码器 | 中等偏高 | 中等偏高 | 较高 | 平衡但需要训练 |
| FreeCus (Ours) | **免训练** | **最高/接近最高** | **最高/接近最高** | **较高** | 零样本达到SOTA水平 |

### 消融实验

| 配置 | DINO-I | CLIP-I | CLIP-T | 说明 |
|------|--------|--------|--------|------|
| Full FreeCus | 最优 | 最优 | 最优 | 三组件协同 |
| w/o Pivotal Attention | 显著下降 | 显著下降 | 保持 | 主体布局完全丢失 |
| w/o Dynamic Shifting | 中等下降 | 中等下降 | 保持 | 细节纹理模糊 |
| w/o MLLM Enhancement | 轻微下降 | 轻微下降 | 下降 | 复杂属性描述不准确 |

### 关键发现

- **Pivotal Attention Sharing 是最关键的组件**——去掉后主体身份保持度（DINO-I）大幅下降，说明布局信息的传递是主体一致性的基础
- **FreeCus 与训练方法的差距极小**——在多数指标上达到或接近需要训练的 SOTA 方法，证明了 DiT 内在零样本能力的强大性
- **框架与现有控制模块兼容**——可以无缝集成 inpainting pipeline 和 ControlNet 等控制模块，扩展应用场景
- 在主体具有**复杂纹理和独特属性**时，MLLM 增强的贡献最为显著

## 亮点与洞察

- **"免费午餐"的设计哲学**令人印象深刻——不修改任何模型参数，仅通过理解和操控 DiT 内部机制就实现了高质量定制。这启示我们大模型可能已经具备很多我们尚未挖掘的能力
- **Pivotal 策略的平衡设计很优雅**——不是简单的全部共享或不共享，而是选择性地在关键时刻注入关键信息。这种"精确干预"的思路可以推广到其他需要条件控制的生成任务
- **利用 MLLM 增强扩散模型**的思路打开了"大模型辅助大模型"的想象空间——未来可以用 MLLM 的推理能力为各种生成任务提供更精确的语义条件

## 局限与展望

- **仅在 Flux 系列 DiT 上验证**——是否能推广到 SD3、Stable Cascade 等其他 DiT 架构有待验证
- **Pivotal 策略的超参数选择**（哪些时间步、哪些层进行共享）可能需要针对不同场景调优，缺乏自动选择机制
- **多主体定制能力未充分验证**——在需要同时保持多个不同主体身份的复杂场景下表现未知
- 对于**与参考图像差异极大的目标姿态**（如正面参考转背面生成），仅靠注意力共享可能不足以推断未见视角
- 未来可以探索自适应的共享策略，根据主体复杂度和编辑幅度自动调节注入强度

## 相关工作与启发

- **vs DreamBooth**: DreamBooth 通过微调整个 U-Net 实现主体定制，精度高但每个主体需3-5分钟训练。FreeCus 完全免训练，适合实时应用
- **vs IP-Adapter**: IP-Adapter 使用预训练的图像编码器注入视觉特征，需要大规模训练但部署后免微调。FreeCus 连预训练也不需要，真正零成本
- **vs MasaCtrl / Prompt-to-Prompt**: 这些方法也通过注意力操控实现编辑控制，但主要面向图像编辑而非主体定制。FreeCus 的 pivotal 策略针对主体身份保持做了专门优化
- 本文证明了 DiT 比 U-Net 架构更适合免训练定制——DiT 的全局注意力结构让特征共享更自然有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 免训练思路不是首创，但 pivotal attention sharing 和 dynamic shifting 的具体设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 定量和定性比较充分，消融验证了各组件贡献，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机论证充分
- 价值: ⭐⭐⭐⭐⭐ 免训练达到训练方法水平对实际应用意义重大，开源代码进一步增强了价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions](../../NeurIPS2025/image_generation/omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)
- [\[ICCV 2025\] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers](ditfastattnv2_head-wise_attention_compression_for_multi-modality_diffusion_trans.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)
- [\[NeurIPS 2025\] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](../../NeurIPS2025/image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [\[ICCV 2025\] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)

<!-- RELATED:END -->
