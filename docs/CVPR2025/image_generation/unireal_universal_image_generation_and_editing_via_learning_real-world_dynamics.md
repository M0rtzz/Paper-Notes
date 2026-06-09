---
title: >-
  [论文解读] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics
description: >-
  [CVPR 2025][图像生成][统一框架] UniReal提出将各种图像生成和编辑任务统一为"不连续帧生成"的框架，利用视频数据作为可扩展的通用监督源，通过层次化提示和文本-图像关联机制，在单一扩散Transformer中实现了指令编辑、定制化生成、物体插入等多种任务的统一处理。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "统一框架"
  - "图像编辑"
  - "视频数据监督"
  - "Transformer"
  - "多任务生成"
---

# UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics

**会议**: CVPR 2025  
**arXiv**: [2412.07774](https://arxiv.org/abs/2412.07774)  
**代码**: [项目页面](https://xavierchen34.github.io/UniReal-Page/)  
**领域**: 图像生成 / 统一生成编辑  
**关键词**: 统一框架, 图像编辑, 视频数据监督, 扩散Transformer, 多任务生成

## 一句话总结

UniReal提出将各种图像生成和编辑任务统一为"不连续帧生成"的框架，利用视频数据作为可扩展的通用监督源，通过层次化提示和文本-图像关联机制，在单一扩散Transformer中实现了指令编辑、定制化生成、物体插入等多种任务的统一处理。

## 研究背景与动机

图像生成和编辑领域随着应用需求的增加变得日益专门化，不同任务需要不同的方法和特定领域数据，限制了跨领域知识的学习。然而，各种任务共享核心需求：保持输入输出图像之间的一致性，同时引入受控的视觉变化。

视频生成模型（如Sora类方法）能有效平衡帧间一致性与运动变化，与图像编辑的需求高度一致。此外，现有图像编辑数据集构建困难，规模有限，而视频数据天然包含了帧间的一致性和变化，可作为可扩展的通用监督源。

UniReal的核心洞察是：将图像级任务视为"不连续视频生成"，利用大规模视频学习世界动态（光照、反射、物体交互等），同时通过层次化提示消除多任务混合训练中的歧义。

## 方法详解

### 整体框架

UniReal基于5B参数的视频生成Transformer，将不同数量的输入/输出图像视为视频帧，通过全注意力建模帧间关系。输入图像通过VAE编码后patchify为视觉token，添加索引嵌入和图像提示（asset/canvas/control），与T5编码的文本嵌入拼接为长1D张量送入Transformer。

### 关键设计1: 文本-图像关联与层次化提示

- **功能**: 在统一框架中消除多任务混合训练的歧义，精确指导模型对不同输入图像的处理方式
- **核心思路**: 使用"IMG1"/"RES1"等引用词将视觉token与文本中的对应位置关联。设计三层提示：(a) **基础提示**描述任务内容；(b) **上下文提示**提供属性标签（如"realistic/synthetic", "static/dynamic"）；(c) **图像提示**通过可学习类别嵌入区分canvas(背景编辑目标)、asset(参考对象)、control(布局/形状约束)三类输入图像
- **设计动机**: 相同输入在不同任务中需要不同处理——编辑保持布局做局部变化，而定制化则生成全新场景仅保留参考物体。上下文提示的关键词可跨任务共享，强制学习共通特征

### 关键设计2: 视频数据构建通用监督

- **功能**: 从视频数据自动构建支持多种任务的大规模训练数据
- **核心思路**: 从原始视频出发，随机选取两帧作为编辑前后图像(Video Frame2Frame, 8M样本)；使用Kosmos-2获取带边界框标注的实体，通过SAM2获取mask轨迹，支持多物体定制(5M)、物体插入(1M)、参考分割(5M)等任务；使用GPT-4o mini为200K高质量子集生成精确指令
- **设计动机**: 视频帧间天然包含的添加、删除、属性变化、结构变化等变化模式，覆盖了大多数编辑任务的基本原理，且比构建专用编辑数据集容易得多

### 关键设计3: 渐进式多任务训练

- **功能**: 从文本到图像/视频的基础生成能力出发，逐步学习编辑能力并提升分辨率
- **核心思路**: 首先在T2I/T2V数据上预训练获得基础生成能力(256分辨率)→在Tab.1所有数据集上训练学习多任务(256分辨率)→渐进提升至512→1024分辨率。使用flow matching训练损失，学习率1e-5带warm-up
- **设计动机**: 渐进式训练策略有助于稳定多任务学习过程，避免低分辨率阶段学到的编辑能力在提升分辨率时退化

### 损失函数

使用标准flow matching损失函数进行训练。

## 实验关键数据

### 主实验1: 指令编辑 - EMU Edit测试集

| 方法 | CLIP_dir↑ | CLIP_im↑ | CLIP_out↑ | L1↓ | DINO↑ |
|------|----------|---------|----------|-----|-------|
| InstructPix2Pix | 0.078 | 0.834 | 0.219 | 0.121 | 0.762 |
| UltraEdit | 0.107 | 0.793 | 0.283 | 0.071 | 0.844 |
| EMU Edit | 0.109 | 0.859 | 0.231 | 0.094 | 0.819 |
| OmniGen | - | 0.836 | 0.233 | - | 0.804 |
| **UniReal** | **0.127** | 0.851 | **0.285** | 0.099 | 0.790 |

### 主实验2: 定制化生成 - DreamBench

| 方法 | CLIP-T↑ | CLIP-I↑ | DINO↑ |
|------|---------|---------|-------|
| DreamBooth | 0.305 | 0.803 | 0.668 |
| BLIP-Diffusion | 0.302 | 0.805 | 0.670 |
| IP-Adapter(Flux) | - | 0.772 | - |
| **UniReal** | **优于大部分方法** | **高CLIP-I** | **高DINO** |

### 关键发现

- UniReal在CLIP_dir和CLIP_out上取得最佳性能，说明编辑方向和输出质量最优
- 能正确建模阴影、反射、光照效果和物体交互等复杂世界动态
- Video Frame2Frame数据单独就能训练出具备基本编辑能力的模型
- 训练数据中视频帧对提供的8M+样本远超现有公开编辑数据集

## 亮点与洞察

1. **统一表述的优雅性**：将所有图像生成/编辑任务统一为不连续帧生成，概念简洁且实用
2. **视频数据的通用监督潜力**：视频帧间的自然变化天然覆盖了各种编辑模式，大幅降低了数据构建成本
3. **可组合的上下文提示**：文本的天然可组合性使得不同任务的提示可以灵活组合，实现训练数据中未见过的新功能

## 局限与展望

- 5B模型规模较大，推理成本较高
- 定制化生成中，对参考物体的保真度尚有提升空间
- 世界动态建模仍受限于训练视频的多样性
- 未来可探索更大规模视频数据和更多任务的统一

## 相关工作与启发

- **OmniGen**: 将文本和图像tokenize为长张量，但生成/编辑能力作为副产品质量不高
- **ACE**: 使用条件单元接收不同输入图像，但未充分利用视频数据
- **Instruct-Imagen**: 使用多模态指令统一图像生成任务，但缺乏视频监督的可扩展性
- 启发：视频生成模型的架构设计原则可直接迁移到图像编辑任务

## 评分

⭐⭐⭐⭐ — 统一框架设计优雅，视频数据作为通用监督的思路极具价值。在多个任务上都取得了competitive或SOTA的性能，展示了大规模统一模型的可行性和潜力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [\[ICCV 2025\] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](../../ICCV2025/image_generation/visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)

</div>

<!-- RELATED:END -->
