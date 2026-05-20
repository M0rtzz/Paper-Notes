---
title: >-
  [论文解读] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding
description: >-
  [AAAI 2026][视频生成][视频扩散模型] 提出 OmniVDiff，一个统一的可控视频扩散框架，通过将多种视觉模态（RGB、深度、分割、Canny）在颜色空间中联合建模，并引入自适应模态控制策略（AMCS），在单一扩散模型中同时支持文本条件生成、X 条件生成和视频理解三种任务…
tags:
  - "AAAI 2026"
  - "视频生成"
  - "视频扩散模型"
  - "多模态生成"
  - "可控视频生成"
  - "视频理解"
  - "统一框架"
---

# OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding

**会议**: AAAI 2026  
**arXiv**: [2504.10825](https://arxiv.org/abs/2504.10825)  
**代码**: [https://tele-ai.github.io/OmniVDiff/](https://tele-ai.github.io/OmniVDiff/) (有项目页)  
**领域**: 视频生成  
**关键词**: 视频扩散模型, 多模态生成, 可控视频生成, 视频理解, 统一框架

## 一句话总结
提出 OmniVDiff，一个统一的可控视频扩散框架，通过将多种视觉模态（RGB、深度、分割、Canny）在颜色空间中联合建模，并引入自适应模态控制策略（AMCS），在单一扩散模型中同时支持文本条件生成、X 条件生成和视频理解三种任务，在 VBench 上达到 SOTA。

## 研究背景与动机

视频扩散模型在文本到视频生成上取得了显著进展，但在**可控视频生成**方面面临两个核心瓶颈：

**任务特定微调**：每引入一种新的控制信号（深度、分割、Canny 等），就需要对大规模扩散架构进行专门微调，计算代价高昂且难以扩展

**外部专家模型依赖**：大多数方法需要先用独立的专家模型（深度估计器、分割模型等）提取条件信号，再用另一个扩散模型进行生成，形成多步非端到端流水线

现有工作的不足：
- **VideoJAM**：仅联合建模 RGB 和光流两种模态，不支持条件生成和理解
- **UDPDiff**：支持 RGB + 深度或 RGB + 分割的联合生成，但不能同时合成所有模态
- **Aether**：统一了 RGB + 深度 + 相机位姿，但主要面向几何世界建模

**核心 idea**：将所有视觉模态视为颜色空间中的平行信号，沿通道维度拼接后送入统一的扩散 Transformer，通过动态调整每种模态的角色（生成 vs 条件）来灵活支持多种下游任务。

## 方法详解

### 整体框架

OmniVDiff 基于预训练的 CogVideoX 文本到视频模型构建，包含三个核心组件：
1. **多模态视频编码**：用共享的 3D-VAE 编码器将 RGB、深度、分割、Canny 四种模态分别编码到潜空间
2. **OmniVDiff 扩散网络**：将所有模态的噪声潜变量沿通道维度拼接，送入扩散 Transformer 联合去噪
3. **多模态视频解码**：通过模态特定投影头（MSPH）将去噪结果分离为各模态，再用 3D-VAE 解码器还原

### 关键设计

1. **多模态视频扩散架构**:

    - 功能：扩展 CogVideoX 的输入空间以容纳多种模态，在输出端为每种模态设计独立的投影头
    - 核心思路：各模态先经过 3D-VAE 编码为潜空间表示 x_m，再与噪声混合后沿通道拼接形成统一输入 x_i = Concat(x_r^t, x_d^t, x_s^t, x_c^t)
    - 设计动机：复制而非共享投影头，因为不同模态（深度 vs 分割 vs Canny）具有截然不同的分布特征

2. **自适应模态控制策略（AMCS）**:

    - 功能：动态决定每种模态是"生成模态"还是"条件模态"
    - 核心思路：生成模态与噪声混合输入，条件模态保持原始信号直接拼接；同时引入可学习的模态嵌入 e_g / e_c 进一步区分角色
    - 设计动机：无需为每种条件任务分别微调模型，一个统一架构即可灵活切换任务

3. **两阶段训练策略**:

    - 功能：第一阶段学习多模态视频联合生成，第二阶段引入条件生成和视频理解任务
    - 核心思路：每阶段 20K 步，使用独立的去噪损失，条件模态跳过损失计算
    - 设计动机：渐进式训练避免一次性引入所有任务导致的不稳定

### 损失函数 / 训练策略

多模态扩散损失：
$$\mathcal{L} = \sum_{m, m \notin Cond} \mathbb{E}_{x_m, t, \epsilon, m} [\|\epsilon - \epsilon_\theta(x_m^{t,'}, t, e_m)\|^2]$$

- 对每种生成模态独立计算去噪损失
- 条件模态不参与损失计算，仅提供引导
- 训练数据：从 Koala-36M 采样 400K 视频，用 Video Depth Anything 和 Semantic-SAM + SAM2 生成伪标签

## 实验关键数据

### 主实验

**文本条件视频生成（VBench）**：

| 方法 | subject cons. | b.g. cons. | motion smooth. | dynamic deg. | weighted avg. |
|------|--------------|-----------|----------------|-------------|--------------|
| CogVideoX | 95.68 | 96.00 | 98.21 | 53.98 | 72.25 |
| **OmniVDiff** | **97.78** | **96.26** | **99.21** | 49.69 | **72.78** |

**深度条件视频生成（VBench）**：

| 方法 | subject cons. | dynamic deg. | weighted avg. |
|------|--------------|-------------|--------------|
| Make-your-video | 90.04 | 51.95 | 70.17 |
| VideoX-Fun | 96.25 | 50.43 | 72.85 |
| **OmniVDiff** | **97.96** | **53.32** | **73.45** |

**零样本视频深度估计（ScanNet）**：

| 方法 | AbsRel ↓ | δ1 ↑ |
|------|----------|------|
| DepthCrafter | 0.169 | 0.730 |
| VDA-S (teacher) | 0.110 | 0.876 |
| OmniVDiff | 0.125 | 0.852 |
| OmniVDiff-Syn | **0.100** | **0.894** |

### 消融实验

| 配置 | subject cons. | dynamic deg. | weighted avg. | 说明 |
|------|--------------|-------------|--------------|------|
| w/o modality embedding | 97.11 | 41.80 | 71.54 | 缺少模态角色区分 |
| w/o AMCS | 97.31 | 33.28 | 71.21 | 无自适应控制，dynamic degree 大幅下降 |
| w/o MSPH | 96.76 | 41.41 | 71.35 | 共享投影头，各模态特征混杂 |
| **完整 OmniVDiff** | **97.78** | **49.69** | **72.78** | 所有组件协同工作 |

### 关键发现

- OmniVDiff 在仅用伪标签训练的情况下，视频深度估计接近甚至超过了专家教师模型 VDA-S
- 引入少量（10K）高质量合成数据后，OmniVDiff-Syn 在 AbsRel 上超越教师模型（0.100 vs 0.110）
- AMCS 对 dynamic degree 影响最大（去掉后从 49.69 降至 33.28），说明自适应控制对运动动态建模至关重要
- 推理效率方面，OmniVDiff 仅增加 11.8M 参数和 3 秒延迟，即可同时输出 RGB + 深度 + 分割 + Canny

## 亮点与洞察

- **统一框架的优雅设计**：通过通道拼接 + 模态嵌入 + 自适应控制三个简洁设计，将生成和理解统一到一个模型中
- **伪标签训练的有效性**：证明了用专家模型生成的伪标签足以训练出接近甚至超越专家的统一模型
- **灵活的任务适应性**：仅需 2K 步微调即可适配新模态/任务（如超分辨率），展现了极强的扩展性
- **消除外部专家依赖**：端到端流水线减少了多模型部署的复杂性和不一致性

## 局限与展望

- dynamic degree 在文本条件生成中略低于 CogVideoX（49.69 vs 53.98），多模态联合训练可能轻微影响运动动态
- 目前仅支持 4 种模态，扩展到更多模态（光流、法线、语义标签等）需要验证
- 分割质量依赖 Semantic-SAM 和 SAM2 伪标签，标注噪声可能影响上限
- 未在更高分辨率或更长视频上验证

## 相关工作与启发

- CogVideoX 作为 base model 的选择很关键，其 3D-VAE 的时空压缩能力是多模态拼接可行的基础
- 图像领域的类似思路（OneDiff、UniReal）采用"多视角"方式统一任务，本文巧妙地通过通道拼接避免了视频 token 数量爆炸
- 自适应模态控制策略的设计可以推广到其他多模态生成任务（如音频-视觉生成）

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoDPO: Omni-Preference Alignment for Video Diffusion Generation](../../CVPR2025/video_generation/videodpo_omni-preference_alignment_for_video_diffusion_generation.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](../../CVPR2025/video_generation/mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](../../CVPR2025/video_generation/interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)
- [\[CVPR 2025\] VEU-Bench: Towards Comprehensive Understanding of Video Editing](../../CVPR2025/video_generation/veu-bench_towards_comprehensive_understanding_of_video_editing.md)

</div>

<!-- RELATED:END -->
