---
title: >-
  [论文解读] BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation
description: >-
  [ICCV 2025][文本到视频生成] 首次提出针对文本到视频（T2V）生成模型的后门攻击框架BadVideo，利用视频中固有的静态和动态冗余信息（如未被文本指定的环境元素、运动轨迹等），通过时空组合和动态元素转换两类策略隐蔽地嵌入恶意内容，在LaVie和Open-Sora上实现高达93.5%的人类评估攻击成功率，同时有效规避现有内容审核系统。
tags:
  - ICCV 2025
  - 文本到视频生成
  - backdoor attack
  - 冗余信息利用
  - 时空组合
  - 内容安全
---

# BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation

**会议**: ICCV 2025  
**arXiv**: [2504.16907](https://arxiv.org/abs/2504.16907)  
**代码**: https://wrt2000.github.io/BadVideo2025/ （项目主页）  
**领域**: 图像/视频生成安全  
**关键词**: 文本到视频生成, backdoor attack, 冗余信息利用, 时空组合, 内容安全

## 一句话总结
首次提出针对文本到视频（T2V）生成模型的后门攻击框架BadVideo，利用视频中固有的静态和动态冗余信息（如未被文本指定的环境元素、运动轨迹等），通过时空组合和动态元素转换两类策略隐蔽地嵌入恶意内容，在LaVie和Open-Sora上实现高达93.5%的人类评估攻击成功率，同时有效规避现有内容审核系统。

## 研究背景与动机
T2V生成模型近年来发展迅速，在娱乐、教育、营销等领域广泛应用，但其安全性风险尚未被充分研究。由于文本（抽象稀疏）和视频（视觉密集、时间连续）之间存在天然的信息粒度差距,模型必须"脑补"大量文本未指定的内容来生成逼真视频。

这些冗余信息分为两类：

**静态冗余**：单帧内多余的空间元素（如背景物体、过度渲染的细节）

**动态冗余**：时间维度的多余过渡（如未指定的运动序列、场景演变）

如果被恶意攻击者利用，这些冗余信息可以被"武器化"，用于注入色情/暴力/仇恨符号或虚假信息。更关键的是，现有内容审核系统（如Sora的单帧检测）主要分析单帧空间信息，无法捕捉跨帧的时序恶意内容。

本文填补了T2V生成模型后门攻击的空白——此前仅有图像生成的后门攻击研究，视频生成的时序维度带来了全新的攻击面。

## 方法详解

### 整体框架
BadVideo攻击框架包含三个阶段：（1）投毒数据集构造——将触发器插入文本提示并将恶意目标嵌入视频；（2）微调阶段——使用投毒数据集微调预训练T2V模型；（3）推理阶段——攻击者通过包含触发器的文本激活后门。核心创新在于目标视频的生成方法。

### 关键设计

1. **策略一：时空组合（Spatio-Temporal Composition, STC）**:

    - 功能：将恶意内容沿时间维度分解，分散嵌入不同帧的冗余元素中
    - 核心思路：单独看任何一帧都是良性的，但连续观看时恶意信息在观者感知中自然融合。例如将"FU"和"CK"分别放在不同帧中，单帧无害但连看则构成冒犯性词汇
    - 设计动机：利用人类视觉的时序整合特性绕过逐帧检测

2. **策略二：语义概念转换（Semantic Concept Transition, SCT）**:

    - 功能：在视频冗余元素（如背景广告牌）上引入语义概念的时序过渡来传达恶意信息
    - 核心思路：例如视频主体是"装草莓的人"，但背景广告牌从政治内容逐渐变为侮辱性内容。同时利用了静态（广告牌存在）和动态（内容变化）冗余
    - 设计动机：用户提示无法完全指定所有冗余元素的变化路径

3. **策略三：视觉风格转换（Visual Style Transition, VST）**:

    - 功能：操控视频的美学和氛围演变来隐蔽嵌入恶意内容
    - 核心思路：即使文本提示是中性的，冗余的风格信息可被操控为令人不安的背景元素，通过视觉氛围的系统性恶化引起情感不适。如和平场景退化为战后废墟，自然风景转变为污染荒地
    - 设计动机：视频的美学演变路径极少被用户提示指定，提供了天然的攻击空间

4. **目标视频生成Pipeline**:

    - **提示转换模块**：使用LLM将原始文本提示转化为头部和尾部提示，分别描述恶意目标的初始和最终状态
    - **关键帧生成模块**：先用T2I模型根据头部提示生成头帧，再通过图像编辑生成与头帧视觉一致的尾帧
    - **目标视频生成模块**：将头尾帧通过VAE编码后拼接，作为扩散模型的条件输入，生成时域连贯的目标视频

### 损失函数 / 训练策略
微调过程使用标准的扩散模型重建损失：
$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, c, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \mathcal{T}_\theta(c), t)\|_2^2\right]$$
文本编码器保持冻结，仅更新扩散模型参数。投毒比例设为20%，微调200个epoch。

## 实验关键数据

### 主实验

| 模型 | 策略 | FVD↓ | CLIPSIM↑ | ASR_MLLM(%) | ASR_Human(%) | CPR(%) |
|------|------|------|----------|-------------|--------------|--------|
| LaVie | Fine-tuned | 327.39 | 0.2883 | 0.0 | 0.0 | 78.5 |
| LaVie | STC | 352.90 | 0.2847 | 84.3 | **92.3** | 74.2 |
| LaVie | SCT | 342.04 | 0.2871 | 86.5 | 91.6 | 72.8 |
| LaVie | VST | 320.36 | 0.2858 | 88.2 | 90.2 | 76.4 |
| Open-Sora | Fine-tuned | 310.77 | 0.2957 | 0.0 | 0.0 | 89.6 |
| Open-Sora | STC | 355.04 | 0.2918 | 80.5 | 79.5 | 72.6 |
| Open-Sora | VST | 312.31 | 0.2940 | **96.4** | **93.5** | 74.9 |

### 消融实验

| 实验 | 关键指标 | 说明 |
|------|---------|------|
| 投毒比例5% → 30% | ASR 从~40%升至>80% | 20%即达80%+ ASR |
| 训练80 epoch | ASR >70% | 所有策略均在80 epoch后达到较高ASR |
| 微调防御100 epoch | ASR维持>80% | 后门模式已被强记忆 |
| 提示扰动80% | ASR仍有效但CPR大幅下降 | 防御者面临两难 |
| 多后门（3个目标） | 所有ASR仍高 | 可同时植入多个后门 |
| GPT-4o检测(16帧) | 成功率仅52% | 即使显式提示时序检测 |
| Omni-Moderation | 0% 检测率 | 安全模型缺乏时序威胁分类 |

### 关键发现
- VST在Open-Sora上达到96.4%的MLLM ASR和93.5%的人类ASR，效果最强
- 后门对微调防御高度鲁棒——用10%干净数据微调100 epoch后ASR仍>80%
- 现有安全检测系统（包括Omni-Moderation、Llama-Guard-3）完全无法检测时序分布的恶意内容
- 即使GPT-4o被明确告知"恶意信息可能跨帧分布"，在16帧输入下也仅52%检测率，且计算成本随帧数线性增长
- 整个攻击成本仅约6.33美元（A800 GPU约2.11小时），门槛极低

## 亮点与洞察
- 首次揭示了T2V生成的视频冗余信息可被武器化的安全风险，是一个全新的攻击面
- 时空组合策略巧妙利用了人类视觉的时序整合与机器逐帧分析的不对称性
- 攻击成本极低（仅6.33美元）但破坏力强，凸显了T2V安全审查的紧迫性
- 证实了现有内容审核系统在时序维度上的根本性盲区

## 局限与展望
- 投毒比例20%相对较高（CL攻击通常<1%），在真实场景中较难实现
- 三种策略的"恶意性"定义较为主观，不同文化背景下判断可能不同
- 目标视频生成pipeline依赖T2I模型和LLM，攻击复杂度较高
- 仅在LaVie和Open-Sora上测试，未验证更强的商业模型（如Sora、Kling）
- 未深入探讨有效防御方案——仅说明现有方法无效，但也未提出替代方案

## 相关工作与启发
- **vs T2I后门攻击 (Chou et al., Chen et al.)**: T2I攻击目标为特定图像/类别，易被语义一致性检查发现；BadVideo利用视频时序维度实现更隐蔽的攻击
- **vs DDPM/DDIM后门**: 早期无条件生成后门通过初始噪声触发，BadVideo在文本条件下操作更为实际
- **vs Struppek et al.**: 其关注文本编码器后门，BadVideo则是端到端的视频生成后门

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个T2V后门攻击，时空冗余利用的视角非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型多策略验证，含防御评估和自适应防御分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，策略分类合理，图示直观
- 价值: ⭐⭐⭐⭐⭐ 揭示了T2V模型重要且之前被忽视的安全风险，对行业有实际警示意义

<!-- RELATED:START -->

## 相关论文

- [STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [Can Text-to-Video Generation Help Video-Language Alignment?](../../CVPR2025/video_generation/can_text-to-video_generation_help_video-language_alignment.md)
- [TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)

<!-- RELATED:END -->
