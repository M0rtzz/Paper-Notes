---
title: >-
  [论文解读] Mind the Time: Temporally-Controlled Multi-Event Video Generation
description: >-
  [CVPR 2025][多事件视频生成] 提出 MinT，首个支持事件时间控制的多事件视频生成器，通过 Rescaled RoPE (ReRoPE) 位置编码将事件描述绑定到特定时间段，在预训练视频 DiT 上微调实现平滑连贯的多事件视频合成。
tags:
  - CVPR 2025
  - 多事件视频生成
  - 时间控制
  - 位置编码
  - Transformer
  - 视频生成
---

# Mind the Time: Temporally-Controlled Multi-Event Video Generation

**会议**: CVPR 2025  
**arXiv**: [2412.05263](https://arxiv.org/abs/2412.05263)  
**代码**: 无（有 project page）  
**领域**: 视频理解  
**关键词**: 多事件视频生成, 时间控制, 位置编码, 扩散Transformer, 视频生成

## 一句话总结

提出 MinT，首个支持事件时间控制的多事件视频生成器，通过 Rescaled RoPE (ReRoPE) 位置编码将事件描述绑定到特定时间段，在预训练视频 DiT 上微调实现平滑连贯的多事件视频合成。

## 研究背景与动机

现有视频生成模型依赖单个文本 prompt 描述整个视频，在生成包含多个连续事件的视频时面临根本性困难：

1. **事件遗漏**：将多个事件描述拼接为长 prompt 时，模型往往只生成其中部分事件而忽略其余事件（Fig. 2 中 CogVideoX、Mochi、Kling、Gen-3 均只生成部分事件）
2. **顺序混乱**：即使生成了多个事件，也常无法按正确时间顺序排列
3. **固定时长**：现有多事件方法（如自回归方式逐事件生成）无法控制单个事件的时长，所有事件被迫等长
4. **过渡不自然**：拼接独立生成的视频片段会产生突兀的场景切换

**核心洞察**：空间上有 bounding box 绑定对象到空间位置，时间上也需要将事件绑定到特定时间段。如果每个事件有明确的时间范围，模型可以一次专注一个事件，从而自然地安排事件顺序。

## 方法详解

### 整体框架

MinT 在预训练视频 DiT（latent Diffusion Transformer）基础上构建。输入包含：(1) 全局标题描述背景和主体外观，通过原有 cross-attention 层注入；(2) 一系列时序标题 $(c_n, t_n^{start}, t_n^{end})$ 描述动态事件及其时间段，通过新增的时序 cross-attention 层注入。这种全局+时序的分离式设计类似视频生成中经典的内容-运动解耦范式。

### 关键设计

1. **Rescaled RoPE (ReRoPE) 位置编码**: 核心技术贡献。为了将事件文本嵌入与对应时间段的视频 token 关联，在时序 cross-attention 中使用改进的旋转位置编码。

   关键问题：直接使用原始时间戳作为 RoPE 旋转角度（vanilla RoPE）会失败——当相邻事件时长不同时，某些属于事件 A 的帧可能距离事件 B 的中点更近，导致注意力偏向错误事件。
   
   解决方案：将所有事件重新缩放到相同长度 $L$，映射公式：
   $$\tilde{t} = \frac{(t - t_n^{start})L}{t_n^{end} - t_n^{start}} + (n-1)L$$
   
   ReRoPE 满足三个理想性质：(i) 事件时间范围内的帧总是最关注该事件的文本；(ii) 注意力权重在事件中点最高，向边界递减；(iii) 事件过渡点的帧对相邻两个事件的注意力相等，实现平滑过渡。
   
   使用固定的 $L=8$，使得不同长度的视频被缩放到相同位置编码空间，层的行为与实际视频长度无关。

2. **场景切换条件化 (Scene Cut Conditioning)**: 将场景切换视为特殊事件——用可学习向量 $e^{cut} \in \mathbb{R}^{1 \times D^c}$ 表示，通过 ReRoPE 编码其时间戳后与事件嵌入拼接进行 cross-attention。训练时保留包含场景切换的视频（占数据 20%），模型学会根据切换 token 的有无来控制是否产生场景转换。推理时输入零向量即可避免不需要的切换。

3. **Prompt 增强器**: 利用 GPT-4 将用户的简短 prompt 扩展为详细的全局标题 + 多个带时间戳的事件标题，使用户可以用简单输入生成运动丰富的多事件视频。LLM 负责规划事件的时间结构。

### 损失函数 / 训练策略

基于 Rectified Flow 训练：$\mathcal{L}_{DiT} = \|v_t - u_\theta(z_t, t, y)\|^2$，其中 $v_t = \epsilon_t - z$。

- 手动标注 ~200K 视频的时序事件（含起止时间），从现有数据集中标注
- TransNetV2 自动检测场景切换边界
- 仅新增时序 cross-attention 层的参数需要学习，全模型微调
- AdamW 优化器，batch size 512，12K steps
- 推理 256 步去噪，CFG scale=8

## 实验关键数据

### 主实验（T2V HoldOut 数据集）

| 方法 | VQ ↑ | DD ↑ | CLIP-T ↑ | TA ↑ | TC ↑ | #Cuts ↓ |
|------|------|------|---------|------|------|---------|
| Concat (base) | 2.61 | 3.32 | 0.247 | 2.37 | 2.45 | 0.020 |
| AutoReg | 2.39 | 2.97 | 0.267 | 2.96 | 2.10 | 0.056 |
| MEVG | 2.50 | 3.39 | 0.264 | 2.68 | 2.15 | 0.120 |
| **MinT** | **2.56** | **3.32** | **0.270** | **2.92** | **2.44** | **0.026** |

MinT 在事件文本对齐（TA +0.55 vs Concat）和时序一致性（TC +0.29 vs MEVG）上大幅领先，同时保持视觉质量。

### 消融实验

| 配置 | CLIP-T ↑ | TA ↑ | TC ↑ | #Cuts ↓ | 说明 |
|------|---------|------|------|---------|------|
| Full Model (ReRoPE L=8) | 0.270 | 2.92 | 2.44 | 0.026 | 最佳平衡 |
| Concat time | 0.249 | 2.42 | 2.33 | 0.075 | 无绝对位置→无法关联事件 |
| Hard attn mask | 0.260 | 2.68 | 2.30 | 0.069 | 边界硬切→过渡突兀 |
| Vanilla RoPE | 0.262 | 2.79 | 2.42 | 0.030 | 不同时长事件定位不准 |
| No cut condition | 0.268 | 2.89 | 2.34 | 0.084 | 无切换控制→意外切换增多 |

### 关键发现

- **Prompt 增强带来动态度飞跃**：在 VBench 上，使用短 prompt 的 Dynamic Degree 仅 0.481，用全局标题增强到 0.517，加上时序标题后达到 0.711（+47.8%），同时保持视觉质量
- **I2V 同样有效**：在图像条件生成中，MinT 的 FID 22.04 vs MEVG 57.57，FVD 218.21 vs 495.75，优势更明显
- **人类偏好评估**：在事件文本对齐、时间精度和过渡平滑度上，MinT 显著优于所有 baseline（超过 60% 胜率）
- **模型对 L 值不敏感**：L=4, 8, 16 效果相近，设计鲁棒

## 亮点与洞察

- **ReRoPE 的设计直觉极其优雅**：通过"缩放到等长"这一简单操作解决了不等时长事件的位置编码难题，数学上可证明满足三个理想性质
- **场景切换条件化是被忽视的重要维度**：过去所有方法要么删除含切换的训练数据，要么生成时产生不受控的切换。将其显式建模为可控条件是巧妙的工程决策
- **全局+时序标题的内容-运动解耦**：比单 prompt 更有效地引导视频动态，且与 "spatial binding" 类比自然
- **一次性生成所有事件** vs 自回归逐个生成：前者能感知全局结构，后者无法规划未来事件

## 局限与展望

- 需要手动标注时序事件数据（~200K 视频），标注成本高
- 仅支持 512×288 分辨率，最长 12 秒，无法处理真正的长视频
- 事件之间的过渡虽然比 baseline 平滑，但在主体外观变化大的情况下仍可能不够自然
- LLM prompt 增强引入额外延迟和成本，且 LLM 的时间规划可能不总是合理

## 相关工作与启发

- Phenaki 是多事件生成的先驱，但自回归方式导致质量退化
- MEVG 用 DDIM inversion 初始化后续事件的噪声以保持一致性，但主体变化时失效
- 本文的 ReRoPE 思路可扩展到其他需要时间定位的生成任务（如音视频同步、定位感知生成）
- 场景切换条件化思想类似图像生成中处理裁剪/宽高比的做法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创事件时间控制的视频生成，ReRoPE 设计优雅且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ T2V/I2V/prompt增强多个设置，自动+人工评估，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机→方法→实验逻辑严密，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 解决了视频生成领域一个重要且实用的问题，开辟新方向

<!-- RELATED:START -->

## 相关论文

- [One-Minute Video Generation with Test-Time Training](one-minute_video_generation_with_test-time_training.md)
- [Multi-subject Open-set Personalization in Video Generation](multi-subject_open-set_personalization_in_video_generation.md)
- [ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](../../CVPR2026/video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](../../CVPR2026/video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)

<!-- RELATED:END -->
