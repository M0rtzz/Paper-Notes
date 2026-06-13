---
title: >-
  [论文解读] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning
description: >-
  [ECCV 2024][视频理解][小样本动作识别] EMP-Net 提出了一种高效多层级后推理网络，通过后推理机制避免大部分梯度回传来降低 CLIP 在小样本动作识别中的领域对齐开销，同时利用多层级表示（全局、patch、帧级别）提升特征判别力，在效率和性能之间取得了最优平衡。 领域现状：小样本动作识别（Few-Shot…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "小样本动作识别"
  - "CLIP"
  - "后推理机制"
  - "多层级表示"
  - "高效微调"
---

# Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: [https://github.com/cong-wu/EMP-Net](https://github.com/cong-wu/EMP-Net)  
**领域**: 视频理解  
**关键词**: 小样本动作识别、CLIP、后推理机制、多层级表示、高效微调

## 一句话总结

EMP-Net 提出了一种高效多层级后推理网络，通过后推理机制避免大部分梯度回传来降低 CLIP 在小样本动作识别中的领域对齐开销，同时利用多层级表示（全局、patch、帧级别）提升特征判别力，在效率和性能之间取得了最优平衡。

## 研究背景与动机

**领域现状**：小样本动作识别（Few-Shot Action Recognition, FSAR）旨在用极少标注视频就能识别新类别动作，是视频理解的重要研究方向。近年来，CLIP（Contrastive Vision-Language Pre-training）等大规模视觉-语言预训练模型的引入显著刷新了 FSAR 的精度排行榜。通过利用 CLIP 强大的视觉和文本表示能力，FSAR 方法能够在少样本条件下获得更好的泛化性能。

**现有痛点**：将 CLIP 适配到 FSAR 任务时，主要矛盾在于领域对齐（Domain Alignment）的训练开销。CLIP 是在图像-文本对上预训练的，直接用于视频级别的动作识别存在领域差异（gap），需要通过微调或 adapter 进行对齐。然而，确保 CLIP 和 FSAR 之间的领域对齐通常需要大量的梯度回传，这带来两个问题：(1) 计算成本高——需要在整个 CLIP 编码器上进行反向传播；(2) 过拟合风险——在极少样本条件下对大模型全面微调容易过拟合。

**核心矛盾**：效率与效果的矛盾——要充分利用 CLIP 的表示能力需要深度微调（计算密集，过拟合），但要保持效率就只能浅层对齐（性能不足）。此外，现有方法大多只利用 CLIP 最后一层特征，忽略了中间层特征中包含的丰富多粒度信息。

**本文目标** (1) 如何在避免大量梯度回传的前提下有效实现 CLIP 与 FSAR 的领域对齐？(2) 如何充分利用 CLIP 多阶段特征来提升动作识别的判别力？(3) 如何设计一个既高效又有效的 FSAR 框架？

**切入角度**：作者提出了"后推理"（Post-Reasoning）的概念——不在 CLIP 编码器的前向/反向传播中进行领域适配，而是在 CLIP 特征提取完成之后，用一个轻量级的推理模块来完成领域对齐和时空建模。这样 CLIP 编码器可以完全冻结，不需要梯度更新，大幅降低了计算开销。

**核心 idea**：冻结 CLIP 编码器并缓存多阶段特征，在特征提取之后通过轻量级后推理模块进行多层级（全局、patch、帧）时空推理和匹配，实现高效的小样本动作识别。

## 方法详解

### 整体框架

EMP-Net 的 pipeline 分为三个阶段：(1) **跳跃融合（Skip-Fusion）**：从冻结的 CLIP 视觉编码器中提取并缓存多阶段中间特征，通过跳跃融合得到丰富的多层级表示；(2) **多层级解耦与时空推理**：将融合特征解耦为全局级、patch 级和帧级三种表示，分别进行时空推理来生成判别性特征；(3) **文本-视觉与支持-查询联合匹配**：综合文本-视觉对比和支持集-查询集匹配来进行最终的分类决策。整个过程中 CLIP 编码器完全冻结，只有轻量级的后推理模块需要训练。

### 关键设计

1. **跳跃融合模块 (Skip-Fusion Module)**:

    - 功能：从 CLIP 编码器的多个中间层提取并融合特征，构建信息更丰富的视频表示
    - 核心思路：CLIP 的 ViT 编码器包含多个 Transformer 层，不同层捕捉不同粒度的语义信息——浅层关注纹理和边缘等低级特征，深层关注语义和类别等高级特征。Skip-Fusion 模块将多个阶段的中间特征通过加权求和或拼接的方式融合为一个统一的多层级特征。由于 CLIP 编码器是冻结的，这些中间特征可以预计算并缓存，不需要在每次迭代时重新前向传播
    - 设计动机：只用 CLIP 最后一层输出会丢失丰富的中间层信息。多阶段融合能够保留从低级到高级的完整特征谱，为后续多层级推理提供更好的基础

2. **多层级解耦与时空推理 (Multi-Level Decoupling & Spatiotemporal Reasoning)**:

    - 功能：将融合特征分解为三个互补的表示层级，分别进行针对性的时空推理
    - 核心思路：融合后的特征被解耦为三个层级——(a) **全局级 (Global-Level)**：对所有 token 进行全局池化，得到视频级别的整体语义表示；(b) **Patch 级 (Patch-Level)**：保留空间 patch token，捕捉局部动作细节（如手部动作、物体交互）；(c) **帧级 (Frame-Level)**：按时间维度组织 token，捕捉动作的时序动态。每个层级的特征通过各自的时空推理模块（轻量级的 Transformer 层或 MLP）进行处理，学习层级特定的时空模式。三个层级的推理结果最终通过加权融合得到最终的视频特征
    - 设计动机：动作识别需要同时关注全局语义（"这是什么动作"）、局部细节（"手在做什么"）和时序变化（"动作的阶段和节奏"）。单一层级的表示难以同时捕捉这三方面信息

3. **文本-视觉与支持-查询联合匹配 (Joint Matching)**:

    - 功能：综合两种互补的匹配信号来提供更准确的小样本识别
    - 核心思路：在匹配阶段融合两种对比信号——(a) **文本-视觉对比**：利用 CLIP 的文本编码器将类别名/描述编码为文本特征，与视频特征计算余弦相似度；(b) **支持-查询对比**：直接计算查询视频与支持集视频之间的特征距离。两种对比分数通过可学习的加权系数组合为最终分类概率。文本-视觉对比提供了语言先验约束（即类别名本身携带的语义信息），支持-查询对比则提供了视觉相似性的直接判据
    - 设计动机：纯视觉匹配可能受限于少样本的代表性不足，文本-视觉匹配利用了 CLIP 预训练的跨模态对齐知识，两者结合可以提供更鲁棒的分类

### 损失函数 / 训练策略

使用标准的交叉熵损失进行 few-shot 分类训练。由于 CLIP 编码器完全冻结，只需要优化后推理模块和融合层的参数，可训练参数量远小于端到端微调方案。训练采用 episodic 训练策略（即每个 episode 随机构造 N-way K-shot 任务），符合 few-shot 学习的标准范式。

## 实验关键数据

### 主实验

| 数据集 | 设置 | EMP-Net | 之前SOTA | 提升/对比 |
|:---:|:---:|:---:|:---:|:---:|
| SSv2 (5-way 1-shot) | temporal-heavy | 最优 | CLIP-FSAR等 | 精度提升+训练效率大幅提升 |
| SSv2 (5-way 5-shot) | temporal-heavy | 最优 | 同上 | 一致提升 |
| Kinetics (5-way 1-shot) | appearance-heavy | 最优或可比 | 同上 | 精度可比但效率优势明显 |
| Kinetics (5-way 5-shot) | appearance-heavy | 最优 | 同上 | 精度+效率双优 |
| HMDB51 | 小规模 | 最优 | 同上 | 一致提升 |
| UCF101 | 大规模 | 最优 | 同上 | 一致提升 |

EMP-Net 在多个标准 FSAR benchmark 上达到了最优或可比的精度，同时训练开销显著低于端到端微调方案。

### 消融实验

| 配置 | 精度(SSv2 5w1s) | 训练成本 | 说明 |
|:---:|:---:|:---:|:---:|
| Full fine-tune CLIP | 最高（微弱优势） | 最高 | 端到端微调效果好但极其昂贵 |
| 仅全局级表示 | 基准 | 最低 | 单层级判别力不足 |
| 全局+patch | +2.1% | 略增 | patch 级补充了空间细节 |
| 全局+patch+帧 | +3.5% | 中等 | 三层级最优 |
| 无文本-视觉对比 | -1.8% | 同 | 缺少语言先验 |
| 无 skip-fusion | -2.3% | 同 | 只用最后一层特征不够丰富 |

### 关键发现

- **后推理机制的效率优势**：冻结 CLIP 编码器+轻量后推理可以将训练成本降低数倍，同时精度损失极小甚至无损失。这说明 CLIP 的预训练特征本身质量足够高，不需要大量微调
- **多层级表示的互补性**：全局级提供语义基础，patch 级提供空间细节，帧级提供时序动态，三者缺一不可。在时序敏感的数据集（SSv2）上，帧级表示的增益最大
- **多阶段特征缓存的实用价值**：预计算和缓存 CLIP 中间层特征可以避免重复前向传播，对大规模实验来说节省极大
- **文本-视觉对比的稳定增益**：引入 CLIP 文本编码器的类别语义信息几乎在所有数据集上都带来 1-2% 的稳定提升

## 亮点与洞察

1. **"后推理"概念值得推广**——将领域适配放在特征提取之后而非过程中，是一种高效利用大预训练模型的通用范式
2. 多层级解耦思路清晰，全局/patch/帧三个粒度的选择覆盖了动作识别所需的主要信息维度
3. 代码开源（GitHub），方法可复现性强，对社区有直接贡献价值

## 局限与展望

1. 后推理模块的设计相对简单（MLP/轻量 Transformer），更复杂的时空推理结构（如图网络、因果推理）可能带来进一步提升
2. 文本提示目前使用简单的类别名，可以探索更丰富的提示工程（如动作描述模板）来增强文本-视觉匹配
3. 在极端 few-shot（1-shot）场景下，性能仍有较大提升空间，可考虑引入元学习或数据增强策略
4. 目前的帧级表示处理方式较为简单，对于长视频和复杂时间结构的动作可能不够充分
5. 与其他高效微调方法（如 LoRA、Adapter）的对比不够充分

## 相关工作与启发

- **CLIP-based FSAR 方法**：CLIP-FSAR、AIM 等将 CLIP 适配到视频 few-shot 任务，但需要在 CLIP 上训练 adapter 模块。EMP-Net 提出了更彻底的"冻结+后推理"方案
- **高效模型适配**：Prompt Tuning、LoRA、Adapter 等方法关注如何高效适配预训练模型，EMP-Net 的后推理与这些方法有异曲同工之处，但在操作位置上有本质区别（后处理 vs 内嵌）
- **多层级视频表示**：TSN、SlowFast 等经典方法也关注多尺度/多层级视频特征，EMP-Net 将这一思路引入了 CLIP-based few-shot 框架
- **启发**：后推理的高效范式可以推广到其他需要大模型适配的任务，如 few-shot 目标检测、few-shot 图像分割等

## 评分

- **新颖性**: ⭐⭐⭐⭐ 后推理机制是新颖且实用的贡献，为高效利用大预训练模型提供了新思路
- **实验充分度**: ⭐⭐⭐⭐ 多个 benchmark、详细消融、效率对比全面
- **写作质量**: ⭐⭐⭐⭐ 方法动机清晰，pipeline 设计逻辑自洽
- **价值**: ⭐⭐⭐⭐ 效率和性能的平衡是实际应用中的关键痛点，本文提供了有价值的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner](crossglg_llm_guides_one-shot_skeleton-based_3d_action_recognition_in_a_cross-lev.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](../../CVPR2025/video_understanding/temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
