---
title: >-
  [论文解读] FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts
description: >-
  [CVPR 2025][3D视觉][室内场景合成] FreeScene 提出了一个用户友好的室内场景合成框架，通过 VLM 驱动的 Graph Designer 将自由形式的文本/图像输入转化为场景图，再用 Mixed Graph Diffusion Transformer (MG-DiT) 在混合连续-离散空间上进行图感知去噪，统一支持 text-to-scene、graph-to-scene 等多种任务，在生成质量和可控性上均超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 室内场景合成
  - 图扩散模型
  - 场景图
  - VLM
  - 可控生成
---

# FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts

**会议**: CVPR 2025  
**arXiv**: [2506.02781](https://arxiv.org/abs/2506.02781)  
**代码**: https://cangmushui.github.io/FreeScene-io/ (项目页)  
**领域**: 3D视觉  
**关键词**: 室内场景合成, 图扩散模型, 场景图, VLM, 可控生成

## 一句话总结
FreeScene 提出了一个用户友好的室内场景合成框架，通过 VLM 驱动的 Graph Designer 将自由形式的文本/图像输入转化为场景图，再用 Mixed Graph Diffusion Transformer (MG-DiT) 在混合连续-离散空间上进行图感知去噪，统一支持 text-to-scene、graph-to-scene 等多种任务，在生成质量和可控性上均超越现有方法。

## 研究背景与动机

**领域现状**：室内场景合成是游戏设计、VR/AR、机器人等领域的关键任务。近年来扩散模型（如 DiffuScene）已被用于场景布局生成，通过迭代去噪产生合理的室内场景。

**现有痛点**：现有方法在可控性上存在两难困境。基于文本的方法（如 DiffuScene）虽然使用方便，但只能实现粗粒度控制，文本描述与生成结果之间的精确对齐难以保证。基于图的方法（如 InstructScene）提供了更好的细粒度控制，但要求用户手动设计繁琐的场景图，使用门槛很高。

**核心矛盾**：便捷性与精确可控性之间的 trade-off——文本输入方便但不精确，图输入精确但不方便。而且现有方法通常只能处理单一条件类型，缺乏统一的多任务能力。

**本文目标** (1) 如何让用户自由输入文本和/或图像就能精确控制场景生成？(2) 如何用单一模型统一支持 text-to-scene、graph-to-scene、重排列等多种任务？

**切入角度**：作者观察到场景图是一个天然的中间表示——它既能从自由形式的输入中提取，又能精确指导场景生成。利用 VLM 的多模态理解能力自动从用户输入推断场景图，然后用混合扩散模型联合建模离散（类别、关系）和连续（位置、尺寸）属性。

**核心 idea**：用 VLM 将自由形式输入自动转换为场景图先验，再通过混合图扩散 Transformer 实现统一的多任务可控场景生成。

## 方法详解

### 整体框架
FreeScene 的 pipeline 分两阶段：(1) **Graph Designer**：接收用户的文本和/或图像（顶视图、照片、草图等），通过 VLM (GPT-4o) 进行多步推理，提取物体类别和空间关系，构建部分场景图先验；(2) **MG-DiT**：以部分场景图和文本描述为条件，对场景中物体的离散属性（类别、fVQ-VAE 特征索引、关系类型）和连续属性（尺寸、位置、朝向）同时进行混合扩散去噪，生成完整的室内场景布局。最终通过 OpenCLIP 特征匹配从 3D-FUTURE 数据集中检索最匹配的家具模型。

### 关键设计

1. **VLM-based Graph Designer（场景图设计器）**:

    - 功能：从自由形式的多模态输入（文本+图像）中自动提取场景图（物体列表+关系三元组+文本描述）
    - 核心思路：设计了 one-shot Chain-of-Thought (CoT) 提示模板，引导 VLM 按四个步骤推理：视角校准 → 物体提取 → DFS 遍历（从关键根节点深度优先搜索所有物体，避免遗漏关系） → 关系提取。输出的结构化数据通过正则表达式解析为图表示。
    - 设计动机：直接让 VLM 生成关系容易产生矛盾或遗漏关键关系。DFS 遍历策略确保层次化地覆盖所有物体间的关系，CoT 比普通 one-shot 在关系准确率上提升 40+%（如图表场景从 44.32% 提升到 74.63%）。

2. **Mixed Graph Diffusion Transformer (MG-DiT)**:

    - 功能：联合去噪离散变量（物体类别 $c$、fVQ-VAE 特征索引 $v$、关系 $e$）和连续变量（尺寸 $s$、位置 $t$、朝向 $r$），生成完整场景布局
    - 核心思路：将场景表示为混合属性图。连续变量用 DDPM 加高斯噪声，离散变量用 D3PM 加带 [MASK] 状态的转移矩阵。网络基于 DiT 架构，节点特征由嵌入+拼接的物体属性构成（含正弦位置编码），边特征由关系嵌入构成。节点与边的交互通过 FiLM 机制实现：$\text{FiLM}(sim, e) = \gamma(e) \cdot \frac{sim - \mu}{\sigma} + \beta(e)$。同时集成 cross-attention 处理文本条件。
    - 设计动机：相比 InstructScene 的两阶段方法（先 text-to-graph 再 graph-to-scene），MG-DiT 用单一模型联合预测图结构和布局属性，在预测图的过程中被迫学习更好的全局场景特征和物体间关系，从而提升生成质量和可控性。

3. **Constrained Sampling（约束采样多任务统一）**:

    - 功能：通过在去噪过程中固定不同子集的变量，使单一模型零样本支持多种下游任务
    - 核心思路：在采样的每一步，根据任务类型固定对应变量。例如 text-to-scene 所有变量正常去噪；graph-to-scene 固定/部分固定类别和关系；re-arrangement 固定类别、尺寸和特征索引，只去噪位置和朝向；completion 固定已有物体的所有属性；stylization 固定除特征索引外的所有变量。
    - 设计动机：避免为每个任务单独训练模型。通过约束采样，Graph Designer 提取的部分图先验可以无缝作为 MG-DiT 的条件输入，实现从粗到细的综合控制。

### 损失函数 / 训练策略
总损失为连续和离散两部分之和：$\mathcal{L} = \mathcal{L}_b + \mathcal{L}_z$。连续部分 $\mathcal{L}_b$ 是标准的噪声预测 MSE 损失；离散部分 $\mathcal{L}_z$ 是预测后验分布与真实后验分布之间的 KL 散度。训练时从均匀分布 $\mathcal{U}(1,T)$ 采样时间步 $t$，同时对连续变量加高斯噪声、对离散变量通过状态转移矩阵加噪。

## 实验关键数据

### 主实验

| 任务/房间 | 方法 | FID↓ | FID_CLIP↓ | KID↓ | SCA% (→50%) | iRecall%↑ |
|-----------|------|------|-----------|------|-------------|-----------|
| Text-to-Scene/Bedroom | InstructScene | 114.86 | 6.52 | 0.68 | 56.37 | 72.71 |
| Text-to-Scene/Bedroom | **Ours+GD** | **108** | **6.07** | **0.21** | **53.16** | **81.40** |
| Text-to-Scene/Livingroom | InstructScene | 111.52 | 5.91 | 8.65 | 55.32 | 57.21 |
| Text-to-Scene/Livingroom | **Ours+GD** | **108.22** | **5.23** | **3.87** | **54.05** | **71.81** |
| Graph-to-Scene/Bedroom | InstructScene | 101.86 | 5.66 | 0.13 | 53.68 | 88.84 |
| Graph-to-Scene/Bedroom | **Ours** | **98.31** | **5.58** | **0.12** | **52.34** | **89.37** |

### 消融实验

| 配置 | Object iRecall% | Rel Acc% |
|------|----------------|----------|
| Graph Designer w/ CoT (Image) | **85.23** | **77.56** |
| Graph Designer w/o CoT (Image) | 72.07 | 34.65 |
| Graph Designer w/ CoT (Diagram) | **91.22** | **74.63** |
| Graph Designer w/o CoT (Diagram) | 88.13 | 44.32 |
| Graph Designer w/ CoT (Text) | **98.56** | **89.45** |
| Graph Designer w/o CoT (Text) | 95.56 | 85.60 |

### 关键发现
- **Graph Designer 的 CoT 策略**对关系准确率提升极大，尤其图像输入场景下 Rel Acc 从 34.65% 提升到 77.56%，DFS 遍历是关键
- **MG-DiT 的混合扩散**显著提升了可控性：iRecall 在 Text-to-Scene 任务上比 InstructScene 提升约 9 个百分点
- 在 Re-arrangement、Completion、Unconditioned 等多种零样本应用中，FreeScene 均优于所有基线方法
- 加入 Graph Designer 预处理的 Ours+GD 比直接 text-to-scene 的 Ours 在所有指标上进一步提升

## 亮点与洞察
- **统一多任务的约束采样**设计非常优雅——用同一个模型通过固定不同变量子集即可支持 5+ 种场景合成任务，无需额外训练
- **DFS 遍历的 CoT 策略**解决了 VLM 在图提取时易遗漏关系的问题，这个技巧可以迁移到任何需要从自由文本/图像中提取结构化关系的任务
- **混合扩散（连续+离散）**的 DiT 架构设计具有通用性，每当任务涉及同时建模连续属性和离散标签时都可以借鉴

## 局限与展望
- Graph Designer 受限于 VLM 能力，面对物体数量很多的复杂场景时提取不够准确
- MG-DiT 无法精确控制物体的具体位置和朝向，可能生成不合理的家具摆放
- 数据集规模有限（3D-FRONT 仅 6,813 个房屋），可能导致过拟合
- 仅支持室内场景（卧室、客厅、餐厅），未扩展到室外或更复杂的环境
- 可考虑引入物理约束（如可达性 affordance）来提升实用性

## 相关工作与启发
- **vs InstructScene**: InstructScene 用两个独立模型分别做 text-to-graph 和 graph-to-scene，FreeScene 用 MG-DiT 单模型统一处理，避免了中间图信息损失
- **vs DiffuScene**: DiffuScene 用 U-Net 的 1D 卷积难以捕捉全局特征，FreeScene 的 DiT + FiLM 架构更擅长建模物体间关系
- **vs LLM-based 方法**: SceneGPT 等方法直接用 LLM 生成场景配置，缺乏细粒度控制；FreeScene 通过图表示实现了精确的关系约束

## 评分
- 新颖性: ⭐⭐⭐⭐ 混合图扩散+VLM 图设计器的组合是自然但有效的，约束采样统一多任务是亮点
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多种任务和房间类型，有详细的消融和对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 为可控室内场景合成提供了实用的统一框架

<!-- RELATED:START -->

## 相关论文

- [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [LT3SD: Latent Trees for 3D Scene Diffusion](lt3sd_latent_trees_for_3d_scene_diffusion.md)
- [DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)

<!-- RELATED:END -->
