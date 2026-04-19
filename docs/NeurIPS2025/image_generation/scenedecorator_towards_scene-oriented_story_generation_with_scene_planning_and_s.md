---
title: >-
  [论文解读] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency
description: >-
  [NeurIPS 2025][图像生成][场景一致性] SceneDecorator 提出了一个无需训练的框架，通过 VLM 引导的场景规划（global-to-local）和长期场景共享注意力机制，首次系统性地解决了故事生成中的场景规划和场景一致性问题，在场景对齐和一致性指标上显著优于现有方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 场景一致性
  - 故事生成
  - scene planning
  - 训练无关
  - 注意力机制
---

# SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency

**会议**: NeurIPS 2025  
**arXiv**: [2510.22994](https://arxiv.org/abs/2510.22994)  
**代码**: https://lulupig12138.github.io/SceneDecorator (项目页面)  
**领域**: 扩散模型 / 故事图像生成  
**关键词**: 场景一致性, 故事生成, scene planning, 训练无关, 注意力机制

## 一句话总结
SceneDecorator 提出了一个无需训练的框架，通过 VLM 引导的场景规划（global-to-local）和长期场景共享注意力机制，首次系统性地解决了故事生成中的场景规划和场景一致性问题，在场景对齐和一致性指标上显著优于现有方法。

## 研究背景与动机
故事生成（story generation）要求生成具有概念一致性的多张图像。现有方法主要关注角色一致性，却忽视了场景在叙事中的作用：

**场景规划缺失**：现有方法仅依赖文本描述生成场景，缺乏场景级别的叙事连贯性。同一个故事的不同场景之间缺乏语义关联

**场景一致性未探索**：在电影故事板等实际应用中，需要在同一场景下生成不同剧情的多个故事，但现有方法无法保持跨故事的场景一致性

**角色单一化**：过度关注角色一致性导致生成的主体风格趋同，缺乏多样性

核心 idea：将场景视为故事生成的核心要素，通过 VLM "导演"进行 global-to-local 的场景规划，并用注意力共享机制保持长期场景一致性。

## 方法详解

### 整体框架
SceneDecorator 是一个无需训练的框架，由两个核心技术组成：
1. VLM-Guided Scene Planning：将用户主题分解为局部场景和故事子提示
2. Long-Term Scene-Sharing Attention：维持跨故事的场景一致性和主体多样性

基于 SDXL 作为基础模型，配合 IP-Adapter-XL 注入场景信息，整个流程无需额外训练。

### 关键设计

1. **VLM-Guided Scene Planning（场景规划）**

    - **概念化全局场景**：用 VLM（Qwen2-VL）根据用户主题 T 生成全局场景描述 Q
    - **可视化全局场景**：用 FLUX.1-dev 根据描述 Q 生成全局场景图像 V
    - **制作局部场景**：VLM 作为"导演"，在全局图像 V 中确定 M 个故事板场景的坐标 {L_i}，裁剪得到局部场景 {V_i}
    - **生成故事子提示**：对每个局部场景 V_i，VLM 生成 N 个连续的故事子提示 P^{1:N}
    - 通过 in-context learning 增强 VLM 性能

2. **Mask-Guided Scene Injection（掩码引导场景注入）**

    - 问题：直接用 IP-Adapter 注入场景会使主体风格与背景过度融合，降低多样性
    - 解决：利用交叉注意力图中主体 token 的激活区域作为掩码 M
    - 对场景特征的注入使用加权公式：$Z_c^{new} = A_c \cdot V_c + \lambda \cdot (1-M) \cdot A_c' \cdot V_c'$
    - 掩码确保场景语义主要注入到背景区域，主体区域保持文本引导的多样性

3. **Scene-Sharing Attention（场景共享注意力）**

    - 在自注意力中，双分支的 latent 表示互相 attend 对方的 K 和 V
    - 关键：用掩码限制 attend 到对方的背景区域：$K' = [K, \tilde{K} \odot (1-\tilde{M})]$
    - 确保不同故事之间共享场景信息，同时避免角色混淆

4. **Extrapolable Noise Blending（可扩展噪声混合）**

    - 动机：上述方法限于同时生成两个故事，需要扩展到 N 个
    - 方法：在去噪区间 [T1, T2] 内，将 N 个 latent 表示动态分成互补配对 $\langle Z_t^i, Z_t^j \rangle$
    - 每个故事参与 N-1 次配对，预测的噪声取平均后更新 latent
    - 关键优势：GPU 显存只需容纳两个故事的量，即可实现 N 个故事的一致性

### 损失函数 / 训练策略
无需训练，所有模块在推理时即插即用。超参数设置：M=4（局部场景数）, N=5（每个场景的故事数）, T1=0, T2=25, 20 步 DDIM 采样。可在单张 RTX 3090 上运行。

## 实验关键数据

### 主实验
| 方法 | CLIP-T↑ | DreamSim-I↓ | DINO-F↑ | 文本对齐% | 场景对齐% | 图像质量% |
|------|---------|-------------|---------|-----------|-----------|-----------|
| CustomDiffusion | 0.306 | 0.752 | 0.373 | 7.9% | 3.4% | 6.0% |
| ConsiStory | 0.320 | 0.723 | 0.475 | 21.3% | 14.1% | 24.7% |
| StoryDiffusion | 0.311 | 0.735 | 0.340 | 14.3% | 6.3% | 11.8% |
| **SceneDecorator** | 0.312 | **0.605** | **0.571** | **56.5%** | **76.2%** | **57.5%** |

### 消融实验
| 配置 | 效果 |
|------|------|
| 无 Mask-Guided Scene Injection | 文本不含场景语义，无法生成与场景匹配的故事 |
| 有 Scene Injection 无掩码 | 场景语义注入但主体风格缺乏多样性 |
| 有 Scene Injection + 掩码 | 场景注入+主体多样性，但跨故事一致性有限 |
| **完整 SceneDecorator** | **场景一致性 + 主体多样性 + 叙事连贯性** |

| Noise Blending 显存分析 | 1张 | 2张 | 5张 | 10张 | 15张 | 20张 | 25张 |
|------------------------|-----|-----|-----|------|------|------|------|
| 无 Extrapolable | 11.4G | 12.7G | 14.5G | 17.5G | 20.4G | 23.5G | OOM |
| **有 Extrapolable** | 11.4G | 12.7G | 12.7G | 12.7G | 12.7G | 12.7G | 12.7G |

### 关键发现
- 用户研究（61人）中 SceneDecorator 在场景对齐方面以 76.2% 的偏好率大幅领先第二名 ConsiStory（14.1%）
- DreamSim-I（场景对齐）从 ConsiStory 的 0.723 改善到 0.605，DINO-F（场景一致性）从 0.475 提升到 0.571
- Extrapolable Noise Blending 将显存固定在 12.7G，无论生成多少个故事
- VLM 场景规划的语义合理性评估（GPT-4o）：叙事连贯 90.06%、主题贴合 92.57%、布局合理 90.29%

## 亮点与洞察
- 首次系统性地从场景角度审视故事生成问题，提出了场景规划和场景一致性两个新挑战
- Global-to-local 的场景规划策略模拟了真实电影制作流程，用 VLM 作为"导演"角色
- 掩码引导的场景注入巧妙地平衡了场景一致性和主体多样性之间的矛盾
- Extrapolable Noise Blending 以 O(1) 显存复杂度实现了 O(N) 个故事的一致性
- 无需训练的设计使得框架可以与 PhotoMaker、ControlNet、风格 LoRA 等工具灵活组合

## 局限与展望
- 场景规划依赖 VLM 的想象力和理解能力，对复杂主题可能失准
- VLM 预测坐标偶尔超出图像边界（虽然有 snap 修正）
- 基于 SDXL 的生成质量受限于基础模型，难以超越
- 场景注入的 λ 权重需要手动调整，不同场景最优值可能不同
- 目前限于静态图像，扩展到视频故事生成是自然的下一步

## 相关工作与启发
- 与 ConsiStory 相比，SceneDecorator 从角色一致性转向场景一致性，提供了新视角
- 掩码引导的注意力操作可以推广到其他需要局部控制的生成任务
- Noise Blending 的配对策略是一种通用的跨样本交互方法
- 支持场景演变（从早到晚、从夏到冬），展示了框架的灵活性

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency](../../ICCV2025/image_generation/free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)
- [EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](../../ECCV2024/image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)
- [Compositional Scene Understanding through Inverse Generative Modeling](../../ICML2025/image_generation/compositional_scene_understanding_through_inverse_generative_modeling.md)
- [GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](../../CVPR2025/image_generation/glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)

<!-- RELATED:END -->
