---
title: >-
  [论文解读] VACE: All-in-One Video Creation and Editing
description: >-
  [ICCV 2025][视频理解][视频生成] 本文提出VACE，一个基于Diffusion Transformer的视频生成与编辑一体化框架，通过统一的Video Condition Unit (VCU)接口和可插拔的Context Adapter结构，用单一模型覆盖参考生成、视频编辑、mask编辑等12+种视频任务，性能与任务专用模型持平。
tags:
  - ICCV 2025
  - 视频理解
  - 视频生成
  - 视频编辑
  - Transformer
  - 统一框架
  - Video Condition Unit
---

# VACE: All-in-One Video Creation and Editing

**会议**: ICCV 2025  
**arXiv**: [2503.07598](https://arxiv.org/abs/2503.07598)  
**代码**: https://ali-vilab.github.io/VACE-Page/  
**领域**: 视频理解 / 视频生成 / 视频编辑  
**关键词**: 视频生成, 视频编辑, 扩散 Transformer, 统一框架, Video Condition Unit

## 一句话总结

本文提出VACE，一个基于Diffusion Transformer的视频生成与编辑一体化框架，通过统一的Video Condition Unit (VCU)接口和可插拔的Context Adapter结构，用单一模型覆盖参考生成、视频编辑、mask编辑等12+种视频任务，性能与任务专用模型持平。

## 研究背景与动机

**领域现状**：视频生成领域已有丰富的下游任务——重绘、编辑、可控生成、参考生成、ID保持生成等。图像领域已出现统一框架（如ACE、OmniGen），但视频领域由于时空一致性更难保证，仍主要是"一个任务一个模型"。

**现有痛点**：(1) 部署多个专用模型成本高（I2V一个模型、inpainting一个模型、可控生成又一个模型）；(2) 单一模型无法做复杂组合任务（如"参考+inpainting"、"手绘+视频扩展"）；(3) 缺乏统一的视频多任务评测基准。

**核心矛盾**：视频任务的输入模态多样（文本、图像、视频、mask），且需要保持时空一致性，如何用统一接口处理这些差异化需求是核心挑战。

**本文目标** 构建一个ALL-in-one视频生成与编辑框架，支持T2V、R2V（参考生成）、V2V（视频到视频编辑）、MV2V（mask视频编辑）及其自由组合。

**切入角度**：将所有视频任务的输入分解为统一的"文本+帧序列+mask序列"三元组。

**核心 idea**：用VCU将多种视频任务的多模态输入统一为帧序列+mask序列的标准表示，通过Context Adapter注入任务条件信息，实现单模型处理所有视频创编任务。

## 方法详解

### 整体框架

VACE基于Diffusion Transformer (DiT) 架构：输入为VCU（文本T + 帧序列F + Mask序列M），经过Concept Decoupling分离编辑/保留内容，Context Latent Encode编码到潜在空间，Context Embedder生成条件token，最后通过Context Adapter注入DiT主干完成生成。支持LTX-Video-2B和Wan-T2V-14B两种规模。

### 关键设计

1. **Video Condition Unit (VCU)**:

    - 功能：将所有视频任务的输入统一为 $V = [T; F; M]$ 格式
    - 核心思路：帧序列F和Mask序列M在时空上对齐，通过不同的赋值表示不同任务：
        - **T2V**：$F = \{0\} \times n, M = \{1\} \times n$（全零帧+全1 mask = 全部生成）
        - **R2V**：$F = \{r_1,...,r_l\} + \{0\} \times n, M = \{0\} \times l + \{1\} \times n$（参考帧保留+后续生成）
        - **V2V**：$F = \{u_1,...,u_n\}, M = \{1\} \times n$（输入视频+全改）
        - **MV2V**：$F = \{u_1,...,u_n\}, M = \{m_1,...,m_n\}$（输入视频+局部mask编辑）
    - **任务组合**：自然支持——如参考+inpainting只需拼接参考帧和带mask的视频帧
    - 设计动机：用数学化的统一表示消除任务间接口差异，使模型无需感知具体任务类型

2. **Concept Decoupling（概念解耦）**:

    - 功能：将帧序列F通过mask分离为reactive frames $F_c = F \times M$（待修改区域）和inactive frames $F_k = F \times (1-M)$（保留区域）
    - 核心思路：不同视觉概念（自然视频 vs 控制信号如depth/pose）有不同分布，显式分离有助于模型收敛
    - 设计动机：编辑任务中需要区分"要改什么"和"要保什么"，混合输入会增加学习难度

3. **Context Adapter Tuning**:

    - 功能：从DiT中选择并复制部分Transformer Block形成Context Blocks，处理context tokens后以加法信号注入主分支
    - 核心思路：采用Res-Tuning方式，主分支DiT参数冻结，仅训练Context Embedder和Context Blocks
    - 对比Fully Fine-tuning：效果相似但收敛更快，且支持可插拔特性（可随时卸载恢复为原始T2V模型）
    - Block分布策略：分布式排列优于连续浅层排列（相同数量下），最终采用部分分布式排列
    - 设计动机：(a) 避免全参数微调丢失预训练能力；(b) 实现与基础模型的可插拔组合

4. **Context Latent Encoding**:

    - 功能：将分离后的 $F_c$、$F_k$ 通过Video VAE编码到与noisy latent X同维度的潜在空间，M直接reshape和插值
    - 参考图像单独编码后沿时间维拼接（避免图像和视频的混合编码问题）
    - Context Embedder沿通道维拼接 $F_c$、$F_k$、M 后tokenize，其中对应 $F_c$ 和 $F_k$ 的权重从原始video embedder复制初始化，M的权重零初始化

### 损失函数 / 训练策略

- 分阶段训练：先学基础任务（inpainting、extension等模态互补任务），再扩展到单参考→多参考→组合任务，最后用高质量数据和长序列微调质量
- 支持任意分辨率、动态时长、可变帧率
- 随机组合不同任务训练以支持组合场景
- 所有涉及mask的操作都做增强以满足各种粒度需求

## 实验关键数据

### 主实验

VACE-Benchmark（480评测样本，12种任务）上的自动评分+用户研究：

| 任务 | 方法 | 归一化平均(自动) | 用户平均(1-5) |
|------|------|----------------|-------------|
| I2V | CogVideoX-I2V | 73.66% | 2.92 |
| I2V | **VACE** | **74.38%** | **3.24** |
| Outpainting | M3DDM | 73.16% | 3.29 |
| Outpainting | **VACE** | **74.25%** | **3.80** |
| Depth控制 | ControlVideo | 70.07% | 2.29 |
| Depth控制 | **VACE** | **74.99%** | **3.23** |
| Pose控制 | Follow-Your-Pose | 66.43% | 2.06 |
| Pose控制 | **VACE** | **76.13%** | **3.18** |
| R2V | Keling1.6 (商业) | 78.81% | 4.04 |
| R2V | VACE | 76.76% | 3.40 |

### 消融实验

| 配置 | 效果 |
|------|------|
| Fully Fine-tuning vs Context Adapter | 效果相似，但Adapter收敛更快 |
| 连续浅层blocks vs 分布式blocks（同数量） | 分布式显著更好 |
| 有Concept Decouple vs 无 | 有时loss下降更明显 |
| 1/4 blocks vs 1/2 blocks vs all blocks | blocks越多越好，但边际递减 |

### 关键发现

- **统一模型 vs 专用模型**：VACE在I2V、outpainting、depth、pose、flow等任务上均超越开源专用方法，在inpainting上与ProPainter持平
- **R2V与商业模型的差距**：在参考生成任务上，基于LTX-Video-2B的VACE与商业模型（Keling、Pika、Vidu）仍有差距，但与Vidu 2.0指标接近
- **组合任务的独特价值**：VACE可以实现"Move Anything"、"Swap Anything"、"Expand Anything"等组合任务，这是现有单一或多模型方法都无法做到的
- **分布式block排列优于连续排列**：信息注入的位置多样性比深度更重要
- **Concept Decouple有效**：显式分离待修改/保留内容帮助模型学习更快收敛

## 亮点与洞察

- **VCU统一表示极其优雅**：用"帧序列+Mask序列"的二元组统一所有视频任务，数学形式简洁而通用。0/1 mask的语义清晰——1表示需要生成的区域，0表示保留的区域。这个设计可迁移到任何多任务视觉生成框架。
- **可插拔设计的工程价值**：Context Adapter可以从基础T2V模型上即插即用地添加/移除条件生成能力，极大降低了部署成本。一个基础模型配多个Adapter即可服务多种需求。
- **任务组合的涌现能力**：通过简单拼接不同任务的VCU表示，自然产生了"Move Anything"、"参考inpainting"等新能力，这种组合涌现是统一框架的核心优势。
- **VACE-Benchmark填补空白**：首个覆盖12种视频任务的统一评测基准，包含自动评分和用户研究两个维度。

## 局限与展望

- R2V任务与商业模型差距明显（用户评分3.40 vs 4.04），可能是模型规模限制
- VACE-Benchmark每个任务仅~20个样本，评测规模偏小
- 论文未详细讨论推理效率——Context Adapter增加了多少计算开销？
- 组合任务的定义需要用户手动构造VCU输入，用户侧使用门槛不低
- 数据构建依赖大量自动化pipeline（SAM2、RAM、Grounding DINO等），标注质量可能参差不齐
- 仅在LTX-Video-2B上做了完整消融，14B版本的消融结果缺失

## 相关工作与启发

- **vs ACE [2024]**: 图像领域的统一生成编辑框架，通过condition token统一不同任务。VACE是其在视频领域的扩展，核心创新在于VCU的时空统一表示和Context Adapter。
- **vs OmniGen [2024]**: 另一个图像统一生成方法。VACE解决了视频中更难的时空一致性问题。
- **vs InstructPix2Pix [2023]**: 指令式图像编辑方法。VACE不需要指令，直接通过VCU指定编辑区域和参考。
- **vs ControlNet [2023]**: 单条件可控生成。VACE统一了所有条件类型，且支持视频。
- VACE的VCU设计思想可以迁移到其他多模态条件生成任务，如3D生成、音频生成等。

## 评分

- 新颖性: ⭐⭐⭐⭐ VCU统一表示和Context Adapter设计很优雅，但基本思路是已有图像统一框架在视频的自然延伸
- 实验充分度: ⭐⭐⭐⭐ 12种任务的对比+消融+用户研究+组合任务可视化，但VACE-Benchmark规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ VCU的形式化定义清晰简洁，架构描述条理分明
- 价值: ⭐⭐⭐⭐⭐ 首个视频DiT上的全任务统一模型，填补了视频统一生成编辑的空白，代码开源且来自阿里通义实验室

<!-- RELATED:START -->

## 相关论文

- [Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)
- [SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization](sweettok_semantic-aware_spatial-temporal_tokenizer_for_compact_video_discretizat.md)

<!-- RELATED:END -->
