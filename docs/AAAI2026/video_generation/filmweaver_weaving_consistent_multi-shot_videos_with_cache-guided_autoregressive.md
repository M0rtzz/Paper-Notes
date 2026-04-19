---
title: >-
  [论文解读] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion
description: >-
  [AAAI 2026][视频理解][多镜头视频生成] 提出 FilmWeaver 框架，通过双层缓存（Shot Cache + Temporal Cache）引导自回归扩散模型，实现任意长度、跨镜头一致性的多镜头视频生成。
tags:
  - AAAI 2026
  - 视频理解
  - 多镜头视频生成
  - 自回归扩散
  - 缓存机制
  - 一致性
  - 长视频生成
---

# FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion

**会议**: AAAI 2026  
**arXiv**: [2512.11274](https://arxiv.org/abs/2512.11274)  
**代码**: [项目页](https://filmweaver.github.io)  
**领域**: 视频理解  
**关键词**: 多镜头视频生成, 自回归扩散, 缓存机制, 一致性, 长视频生成

## 一句话总结

提出 FilmWeaver 框架，通过双层缓存（Shot Cache + Temporal Cache）引导自回归扩散模型，实现任意长度、跨镜头一致性的多镜头视频生成。

## 研究背景与动机

### 领域现状
当前视频扩散模型（如 HunyuanVideo、Wan）在单镜头视频生成上已经表现出色，但在**多镜头（multi-shot）视频**生成任务上仍面临巨大挑战。多镜头视频在影视制作、叙事驱动的创意场景中具有更高的实用价值。

### 核心痛点
多镜头视频生成面临两大关键挑战：

**跨镜头一致性**：相同角色和背景在不同镜头间需要保持身份和外观的一致性，单纯依赖文本描述无法完成

**镜头时长和数量管理**：现有方法在控制每个镜头的时长和视频总长度上受限

### 现有方案的局限
- **多模型管道方法**（VideoDirectorGPT, VideoStudio）：通过关键帧生成 + I2V 的两阶段范式，但各段独立生成导致**视觉不连续和场景跳变**
- **同时多镜头方法**（ShotAdapter, Mask2DiT）：将多镜头共享一个序列，导致**单镜头时长严重受限**
- **TTT**：引入 RNN 机制但缺乏长期记忆且训练成本高
- **LCT**：需两阶段训练，仅支持 MM-DiT 架构

### 核心 Idea

**将一致性问题解耦为"镜头间一致性"和"镜头内连贯性"两个子问题**，通过双层缓存系统分别管理：Shot Cache 存储先前镜头的关键帧维持角色/场景身份，Temporal Cache 保留当前镜头的帧历史确保平滑运动。

## 方法详解

### 整体框架

FilmWeaver 基于自回归扩散范式，核心是一个**双层缓存（Dual-Level Cache）机制**。模型在生成新视频块时，以文本提示 $\mathbf{c}_{\text{text}}$、Temporal Cache $C_{\text{temp}}$ 和 Shot Cache $C_{\text{shot}}$ 作为条件输入，训练目标为：

$$\mathcal{L} = \mathbb{E}_{\mathbf{v}_0, \mathbf{c}_{\text{text}}, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta(\mathbf{v}_t, t, \mathbf{c}_{\text{text}}, C_{\text{temp}}, C_{\text{shot}})\right\|^2\right]$$

该方法通过 **in-context injection** 注入缓存，无需修改模型架构，兼容现有预训练 T2V 模型。

### 关键设计

#### 1. **Temporal Cache（镜头内连贯性）**
- **功能**：作为滑动窗口，存储当前镜头最近生成帧的条件信息
- **核心思路**：视频存在高度时间冗余，采用**差分压缩策略**：近帧高保真保留，远帧逐步压缩
- **具体实现**：三级层次压缩——最近 1 个 latent 不压缩，接下来 2 个 4× 压缩，最后 16 个 32× 压缩
- **设计动机**：在保持运动连贯性的同时控制计算开销，每步生成 6 个 latent（24帧，1秒@24fps）

#### 2. **Shot Cache（镜头间一致性）**
- **功能**：从先前镜头中检索与当前文本提示最相关的 Top-K 关键帧
- **检索机制**：计算 CLIP 文本嵌入与候选关键帧图像嵌入的余弦相似度：

$$C_{\text{shot}} = \underset{kf \in \mathcal{KF}}{\arg\,\text{top-k}}\left(\text{sim}(\phi_T(\mathbf{c}_{\text{text}}), \phi_I(kf))\right)$$

- **K=3**，基于性能与效率的权衡，3 个关键帧足以捕捉复杂多镜头场景所需的多样化概念
- **设计动机**：提供简洁而高度相关的叙事历史视觉摘要，引导模型保持角色/背景一致性

#### 3. **四种推理模式**
根据双层缓存的状态组合，推理分四个阶段：
1. **No Cache**（首镜头生成）：初始化缓存，标准 T2V 模式
2. **Temporal Only**（首镜头延伸）：高时间连贯性，支持视频延展
3. **Shot Only**（新镜头生成）：清空 Temporal Cache，从 Shot Cache 注入先前关键帧，支持多概念注入
4. **Full Cache**（新镜头延伸）：同时利用双层缓存

### 训练策略

#### 渐进式训练课程
- **第一阶段**：仅用 Temporal Cache 训练长单镜头视频生成（10K 步），禁用 Shot Cache，让模型先掌握镜头内动态
- **第二阶段**：激活 Shot Cache，在包含四种缓存场景的混合课程上微调（10K 步），渐进方式加速收敛

#### 数据增强（解决"复制粘贴"问题）
- **问题**：模型过度依赖视觉上下文，导致运动性降低和文本遵循度下降
- **负采样**：随机引入无关关键帧到 Shot Cache，迫使模型区分有用和干扰信息
- **非对称加噪**：Shot Cache 施加强噪声（100–400 时间步），Temporal Cache 施加弱噪声（0–100 时间步），平衡抗复制与运动连贯

### 多镜头数据集构建

使用专家模型进行镜头分割 → 滑动窗口 CLIP 相似度进行场景聚类 → 过滤（去除<1秒片段和>3人场景）→ **Group Captioning**：将整个场景的所有镜头同时输入 Gemini 2.5 Pro 联合描述 → 验证步骤确保标注准确性。

## 实验关键数据

### 主实验

| 方法 | Aes.↑ | Incep.↑ | Char. Cons.↑ | All Cons.↑ | Char. Align.↑ | All Align.↑ |
|------|-------|---------|-------------|-----------|--------------|------------|
| VideoStudio | 32.02 | 6.81 | 73.34% | 62.40% | 20.88 | 31.52 |
| StoryDiffusion | 35.61 | 8.30 | 70.03% | 67.15% | 20.21 | 30.86 |
| IC-LoRA | 31.78 | 6.95 | 72.47% | 71.19% | 22.16 | 28.74 |
| **FilmWeaver** | **33.69** | **8.57** | **74.61%** | **75.12%** | **23.07** | **31.23** |

FilmWeaver 在一致性和角色文本对齐指标上取得 SOTA，同时 Inception Score 最高。

### 消融实验

| 配置 | Aes.↑ | Incep.↑ | Char.↑ | All.↑ | Char. Align.↑ | All Align.↑ |
|------|-------|---------|--------|------|--------------|------------|
| w/o Augmentation | 30.04 | 7.77 | 72.36% | 75.92% | 21.88 | 28.12 |
| w/o Shot Cache | 33.92 | 8.63 | 68.11% | 65.44% | 22.41 | 31.79 |
| w/o Temporal Cache | 31.61 | 8.36 | 70.79% | 70.57% | 20.21 | 30.70 |
| **Full Model** | **33.69** | **8.57** | **74.61%** | **75.12%** | **23.07** | **31.23** |

### 关键发现
1. **Shot Cache 对跨镜头一致性至关重要**：去除后 All Consistency 从 75.12% 暴跌至 65.44%
2. **噪声增强对文本遵循度至关重要**：去除后 Text Alignment 显著下降
3. **负采样提供容错能力**：即使检索到无关关键帧，模型也能有效忽略
4. **计算效率**：注意力复杂度从全序列的 $O(24^2)=576$ 降至分块的 $3.5 \times 11^2 \approx 423.5$

## 亮点与洞察

1. **解耦思想优雅**：将一致性问题显式分解为镜头间/镜头内两个子问题，分别用不同的缓存管理，简洁有效
2. **极强的灵活性**：四种推理模式天然支持多概念注入、视频延伸等下游任务，无需额外训练
3. **架构无关**：通过 in-context injection 注入缓存，不修改模型结构，可兼容各种预训练 T2V 模型
4. **实用的数据构建管道**：Group Captioning 策略解决了跨镜头标注一致性问题

## 局限与展望

1. 视觉质量仍有提升空间，可通过更好的数据策划和训练策略改善
2. 缓存大小（K=3）可能在极复杂场景中不足
3. 数据管道依赖 Gemini 2.5 Pro，标注成本和可获取性受限
4. 评测基准较小（20个场景×5镜头），缺乏标准化的公开多镜头评测

## 相关工作与启发

- 与 FramePack 的差分压缩策略类似，但扩展到了跨镜头场景
- Shot Cache 的检索增强生成思想可迁移到其他需要长期一致性的生成任务
- 负采样策略类似于对比学习中的难负例训练，增强模型鲁棒性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双层缓存解耦的思路优雅，但自回归扩散和 in-context injection 本身不新
- **实验充分度**: ⭐⭐⭐⭐ — 定量+定性对比全面，消融充分，但评测规模较小
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，逻辑流畅，图表丰富
- **价值**: ⭐⭐⭐⭐ — 多镜头视频生成是重要问题，框架实用性强

<!-- RELATED:START -->

## 相关论文

- [Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)
- [Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](../../CVPR2026/video_generation/rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation](mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)
- [Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](../../CVPR2026/video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)

<!-- RELATED:END -->
