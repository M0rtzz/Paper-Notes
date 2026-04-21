---
title: >-
  [论文解读] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning
description: >-
  [ICCV 2025][图像生成][图像生成] 提出 VisualCloze，将多种图像生成任务统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像 infilling 模型实现统一生成，并构建 Graph200K 图结构数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。
tags:
  - ICCV 2025
  - 图像生成
  - visual in-context learning
  - image infilling
  - Graph200K
  - multi-task unification
---

# VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning

**会议**: ICCV 2025  
**arXiv**: [2504.07960](https://arxiv.org/abs/2504.07960)  
**代码**: https://visualcloze.github.io/  
**领域**: 图像生成 / 统一框架 / 上下文学习  
**关键词**: universal image generation, visual in-context learning, image infilling, Graph200K, multi-task unification

## 一句话总结

提出 VisualCloze，将多种图像生成任务统一为"视觉完形填空"范式——用视觉示例（而非文本指令）定义任务，通过图像 infilling 模型实现统一生成，并构建 Graph200K 图结构数据集增强任务间知识迁移，支持域内任务、未见任务泛化、多任务组合和反向生成。

## 研究背景与动机

扩散模型推动了图像生成的快速发展，催生了条件生成、风格迁移、虚拟试穿、个性化生成等大量应用。然而，当前主流方案仍然是为每个任务训练专用模型（如 ControlNet 做条件生成、IP-Adapter 做风格保持、InstructPix2Pix 做编辑），效率低下且难以扩展。

统一生成模型虽然有所进展，但面临三个核心挑战：

**任务指令的歧义性**：现有方法（OmniGen、ACE++）依赖文本指令或任务特定 token 来区分任务，但视觉任务的复杂性和视觉-语言鸿沟导致模型经常产生**任务混淆**（task confusion），且在未见任务上泛化能力差

**视觉任务分布的稀疏性**：与 NLP 中任务高度重叠不同，视觉任务（分割、深度估计、超分辨率等）使用的数据集几乎没有交集，任务间知识孤立，难以学习可迁移的共享特征

**统一架构设计的缺失**：需要同时支持灵活的任务格式和兼容 SOTA 预训练模型

**核心洞察**：（1）视觉上下文学习（Visual ICL）比文本指令更适合定义视觉任务——直接展示输入-输出示例让模型"看到"任务定义；（2）统一图像生成的目标（填充目标图像区域）与图像 infilling 的目标天然一致，可以直接利用预训练 infilling 模型的强大生成先验。

## 方法详解

### 整体框架

VisualCloze 将所有图像生成任务重新定义为**视觉完形填空**问题：

- 给定 $C$ 个 in-context 示例（每个包含 $L$ 张图像的任务实例）和一个 query（$L-1$ 张条件图 + 一个空白目标位置）
- 将所有图像拼接成一个 $(L \times W, (C+1) \times H)$ 大小的网格图
- 用二值 mask $M$ 标记目标位置，利用 infilling 模型补全目标区域

数学表达：$\hat{X} = f(X \mid T, M)$，其中 $X$ 是拼接图像，$T$ 是语言指令，$M$ 是 mask 条件。

### 关键设计

#### 1. Visual In-Context Learning

不用文本描述任务类型，而是直接展示 1-2 个 (input, output) 示例对作为任务演示。模型通过观察示例中的转换关系来理解任务定义，这带来四个关键优势：

- **减少任务歧义**：视觉示例比文本更精确地传达任务意图，增加示例数量可进一步降低混淆
- **泛化到未见任务**：无需任何重训练，仅通过示例即可执行训练时未见过的变换类型
- **多任务组合**：可将多个子任务（如深度引导 + 重光照）合并为单步未见任务
- **反向生成**：支持从目标推断条件（如从风格化图像分解出内容和风格参考）

训练时随机提供最多 $C=2$ 个 in-context 示例，推理时可扩展到更多。

#### 2. Graph200K 数据集

以 Subject200K 为起点，为每张图像构建 49 种标注，涵盖五个元任务：

- **条件生成**：12 种条件（canny edge、HED edge、depth、normal、keypose、SAM2 mask、前景分割、开放世界检测框等）
- **图像修复**：32 种在线退化增强
- **图像编辑**：背景不变编辑（目标替换）和背景变化编辑
- **IP 保持**：Subject-driven 生成
- **风格迁移**：语义不变（InstantStyle）和语义变化（FLUX.1-Redux）两种设置

数据集构建为**强连通图**——任意两个节点间都存在双向路径，路径上的节点作为条件，终端节点为目标。通过组合采样，可得到最多 **134 种高度重叠的任务**，显著提升了任务密度和跨任务迁移。

额外补充数据：VITON-HD（虚拟试穿）、PhotoDoodle（艺术编辑）、OmniEdit（物体添加/移除），以及绘画过程和多视图生成数据。

#### 3. 基于 Infilling 的统一架构

核心发现：VisualCloze 的统一生成公式与图像 infilling 模型的目标**天然一致**——都是根据上下文补全被 mask 的区域。因此：

- 直接基于 FLUX.1-Fill-dev 微调，无需修改架构或添加 task-specific 模块
- 使用 LoRA（rank=256）微调以降低训练成本，保留基础模型能力
- LoRA 可与社区其他 LoRA 融合使用

位置编码处理：利用 FLUX.1-Fill-dev 的 3D-RoPE，将不同示例沿 temporal 维度拼接，解决不同长宽比示例的拼接问题。

语言指令包含三部分：(1) layout 指令描述网格布局；(2) task 指令指定任务类型；(3) content 指令描述目标图像内容。

### 损失函数 / 训练策略

- 标准的 flow matching loss，在拼接后的网格图上训练，仅在 mask 区域计算
- 采用 lognorm noise 策略和动态 time shifting（遵循 FLUX.1-Fill-dev）
- 训练时以 0.5 概率随机 mask 前 $L-1$ 个网格中的一个，促进反向生成能力
- AdamW 优化器，学习率 1e-4，累积 batch size 64，8×A100 训练 20k 步
- 图像统一 resize 到 384×384 或 512×512 面积后拼接

## 实验关键数据

### 主实验

条件生成与图像修复定量对比：

| 条件 | 方法 | Context | 可控性 (F1/RMSE) | FID↓ | SSIM↑ | MAN-IQA↑ | MUSIQ↑ | CLIP-Score↑ |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Canny | ControlNet (专用) | - | 0.13 | 46.06 | 0.34 | 0.31 | 45.45 | 34.10 |
| Canny | OminiControl | - | 0.47 | 29.58 | 0.61 | 0.44 | 61.40 | 34.40 |
| Canny | OmniGen | - | 0.43 | 51.58 | 0.47 | 0.47 | 62.66 | 33.66 |
| Canny | **Ours_fill** | 0 | 0.35 | 30.60 | 0.55 | **0.49** | **64.39** | **34.98** |
| Canny | **Ours_fill** | 2 | 0.36 | 31.15 | 0.56 | **0.49** | 64.08 | 34.85 |
| Depth | ControlNet (专用) | - | 23.70 | 36.83 | 0.41 | 0.44 | 60.17 | 34.49 |
| Depth | OneDiffusion | - | 10.35 | 39.03 | 0.49 | 0.49 | 60.49 | 34.71 |
| Depth | **Ours_fill** | 0 | **10.31** | **33.88** | **0.54** | 0.48 | **64.85** | **35.10** |
| Depth | **Ours_fill** | 2 | **9.68** | 34.88 | 0.54 | 0.48 | 64.29 | 34.89 |
| Deblur | OminiControl | - | 19.70 | 26.17 | 0.85 | 0.45 | 60.70 | 34.53 |
| Deblur | **Ours_fill** | 2 | 25.57 | **36.28** | 0.76 | **0.48** | **61.77** | **34.82** |

Ours_fill 在视觉质量（MAN-IQA、MUSIQ）和文本一致性（CLIP-Score）上全面领先，在深度到图像生成上可控性也最优（RMSE 9.68）。

Subject-driven 生成定量对比：

| 方法 | Context | DINOv2↑ | CLIP-I↑ | CLIP-T↑ |
|------|:---:|:---:|:---:|:---:|
| OminiControl (专用) | - | 73.17 | 87.70 | 33.53 |
| OneDiffusion | - | 73.88 | 86.91 | 34.85 |
| OmniGen | - | 67.73 | 83.43 | 34.53 |
| Ours_dev | 0 | 78.05 | 87.68 | 35.06 |
| **Ours_fill** | 0 | **80.41** | **89.63** | **35.16** |
| **Ours_fill** | 2 | 80.32 | 89.36 | 35.01 |

相比专用模型 OminiControl，在 DINOv2/CLIP-I/CLIP-T 上分别提升 7.15%、1.66%、1.48%。

风格迁移定量对比：

| 方法 | Text↑ | Image↑ |
|------|:---:|:---:|
| InstantStyle (专用) | 0.27 | **0.60** |
| OmniGen | 0.27 | 0.52 |
| **Ours_fill** | **0.29** | 0.55 |

文本一致性超越专用模型 InstantStyle 2%，风格一致性略低。

### 消融实验

Infilling 模型 vs Dev 模型（核心消融）：

| 任务 | 指标 | Ours_dev | Ours_fill | 提升 |
|------|------|:---:|:---:|:---:|
| Depth→Image | RMSE↓ | 25.06 | **10.31** | -58.8% |
| Depth→Image | FID↓ | 42.14 | **33.88** | -19.6% |
| Deblur | RMSE↓ | 25.03 | 26.53 | - |
| Deblur | MUSIQ↑ | 46.68 | **59.62** | +27.7% |
| Subject-driven | DINOv2↑ | 78.05 | **80.41** | +3.0% |
| Subject-driven | CLIP-I↑ | 87.68 | **89.63** | +2.2% |

Ours_fill（基于 FLUX.1-Fill-dev）在大多数任务上显著优于 Ours_dev（基于 FLUX.1-dev），验证了 infilling 目标一致性的关键作用。视觉对比中，Ours_dev 在深度到图像生成时频繁出现对角条纹伪影。

In-context 示例数量的影响（以 Depth→Image 为例）：

| Context 数量 | RMSE↓ | FID↓ | SSIM↑ | MAN-IQA↑ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 10.31 | 33.88 | 0.54 | 0.48 |
| 1 | 9.91 | 34.44 | 0.54 | 0.49 |
| 2 | **9.68** | 34.88 | 0.54 | 0.48 |

增加 in-context 示例可进一步提升可控性（RMSE 从 10.31 降到 9.68），但并非所有指标都单调提升。

### 关键发现

1. **In-context learning 缓解任务混淆**：在 pose estimation、edge detection 等任务上，无 ICL 示例时模型偶尔产生噪声结果，提供 1-2 个示例后性能和稳定性显著提升
2. **未见任务泛化**：训练时仅见过物体添加/移除编辑任务，通过 ICL 泛化到环境修改和属性变换等未见编辑类型；训练仅含单主体生成，可泛化到多主体驱动生成
3. **多任务合并**：通过 ICL 将多个子任务合并为单步执行（如 Depth→Image + Relighting），无需额外训练
4. **反向生成**：模型可从风格化图像反向分解出原图和风格参考，或从边缘图推断真实图像、深度和法线
5. **示例质量影响**：ICL 示例需准确传达任务意图，若示例偏离任务核心（如侧脸过于接近正脸），成功率会显著下降

## 亮点与洞察

1. **视觉完形填空 = image infilling 的统一范式**极为优雅：概念简洁（所有任务都是"填空"），无需改架构（直接用预训练 infilling 模型），工程成本极低（LoRA 微调 20k 步）
2. **Visual ICL 比 text instruction 更适合定义视觉任务**：这是一个重要的范式转变——让用户"展示"而非"描述"他们想要的变换
3. **Graph200K 的图结构数据设计**巧妙解决了视觉任务稀疏问题：每张图 49 种标注、134 种重叠任务，使模型学到紧凑的共享表示
4. **四种能力**（域内、未见泛化、多任务组合、反向生成）从同一框架中自然涌现，无需分别设计

## 局限与展望

1. **分辨率受限**：将多张图拼成网格导致每张图被 resize 到 384×384 或 512×512，精细细节可能丢失
2. **可控性与专用模型有差距**：在 Canny→Image 任务上 F1 得分低于 OminiControl（0.36 vs 0.47），精细边缘控制不如专用模型
3. **ICL 示例选择敏感**：不同示例对生成质量影响较大，如何自动选择最优示例有待研究
4. **Graph200K 数据质量**：依赖自动化 pipeline 构建（Qwen2-VL 生成 caption、FLUX.1-Fill 做编辑），可能引入噪声标注
5. **训练数据偏向**：仅训练物体添加/移除编辑任务，其他编辑类型完全依赖泛化
6. **推理效率**：处理大网格图比单张图推理开销更大

## 相关工作与启发

- **OmniGen**：使用视觉语言模型和文本指令统一多任务，但在未见任务上泛化受限；VisualCloze 通过 visual ICL 显著增强泛化
- **ControlNet / OminiControl**：任务特定的条件注入模块，每种条件需单独训练；VisualCloze 是真正的统一框架
- **OneDiffusion**：统一多种生成任务，但可控性和质量不如 VisualCloze
- **UniReal**：将图像生成统一为不连续视频生成，与 VisualCloze 的 infilling 统一思路互补
- **启发**：Visual ICL 范式可扩展到视频生成——展示几帧视频变换示例定义视频编辑任务；Graph 结构数据集思路可迁移到多模态学习

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — "视觉完形填空 = infilling"的统一公式极为优雅，Visual ICL 用于任务定义是重要范式创新
- **技术质量**: ⭐⭐⭐⭐ — Graph200K 设计巧妙，infilling 目标一致性分析严谨，但可控性仍有提升空间
- **实验充分性**: ⭐⭐⭐⭐ — 覆盖条件生成、修复、风格迁移、主体驱动四类任务，含 dev vs fill 的关键消融
- **实用性**: ⭐⭐⭐⭐⭐ — 基于 FLUX.1-Fill-dev 的 LoRA 微调，工程成本极低，LoRA 可组合使用
- **总评**: ⭐⭐⭐⭐⭐ — 定义了 universal image generation 的新范式，Graph200K 数据集和 Visual ICL 方法论均有长期价值

<!-- RELATED:START -->

## 相关论文

- [Pretrained Reversible Generation as Unsupervised Visual Representation Learning](pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)
- [Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)
- [A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

<!-- RELATED:END -->
