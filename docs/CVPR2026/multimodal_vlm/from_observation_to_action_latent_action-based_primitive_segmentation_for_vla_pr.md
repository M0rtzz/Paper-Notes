---
title: >-
  [论文解读] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
description: >-
  [CVPR 2026][多模态VLM][VLA预训练] 提出 LAPS（Latent Action-based Primitive Segmentation）流水线，通过在潜在动作空间中定义"Latent Action Energy"指标，从未标注的工业视频流中无监督发现和分割语义动作原语…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "VLA预训练"
  - "动作分割"
  - "潜在动作能量"
  - "无监督学习"
  - "工业制造"
---

# From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings

**会议**: CVPR 2026  
**arXiv**: [2511.21428](https://arxiv.org/abs/2511.21428)  
**代码**: 无（工业数据集将部分公开）  
**领域**: Multimodal / VLM  
**关键词**: VLA预训练, 动作分割, 潜在动作能量, 无监督学习, 工业制造

## 一句话总结

提出 LAPS（Latent Action-based Primitive Segmentation）流水线，通过在潜在动作空间中定义"Latent Action Energy"指标，从未标注的工业视频流中无监督发现和分割语义动作原语，为 VLA 模型预训练提供结构化数据。

## 研究背景与动机

**领域现状**：VLA（Vision-Language-Action）模型如 GR00T、AgiBot GO-1 依赖大规模预分割的动作标注视频数据进行预训练，但获取此类数据极其昂贵，通常需要遥操作采集。

**现有痛点**：(1) 工业环境存在大量未标注的连续视频流，但缺乏自动提取结构化动作数据的方法；(2) 现有无监督分割方法（ABD、OTAS）基于像素级或光流变化检测，对非语义物理运动（如光照变化）敏感。

**核心矛盾**：VLA 预训练需要"预分割+动作标注"的短视频片段，但工业视频是连续未切分的长流——这个数据处理瓶颈阻碍了工业 VLA 的规模化部署。

**本文目标**：如何从连续工业视频流中自动发现有限的、可数的动作原语集合？

**切入角度**：不在像素/光流空间做分割，而是将问题转移到潜在动作空间——训练 Motion Tokenizer 编码运动动态，在其潜在空间定义能量指标来检测语义动作边界。

**核心 idea**：从"视觉变化检测"转向"行为意图变化检测"——Latent Action Energy 在动作执行时持续高能，动作完成时回落低能，天然对应语义边界。

## 方法详解

### 整体框架

LAPS 流水线包含三个阶段：
1. **Motion Tracking**：用 CoTracker 从视频中提取稠密运动轨迹
2. **Action Detection & Segmentation**：Motion Tokenizer 生成潜在向量流，基于 Latent Action Energy 做滞后控制分割
3. **Semantic Action Clustering**：通过冻结 Transformer + Cosine k-means 无监督聚类发现动作词汇表

### 关键设计

1. **Motion Tokenizer $M_\theta$**：

    - 基于 AMPLIFY 的 Transformer 编码器-解码器 + FSQ（Finite Scalar Quantization）架构
    - 输入：关键点轨迹速度 $\kappa \in \mathbb{R}^{T \times N \times 2}$
    - 输出：连续量化向量序列 $S_q$ 和离散编码序列 $S_d$
    - 采用分类损失而非像素重建损失，避免捕获动作无关的背景噪声
    - **设计动机**：基于关键点的动态编码比像素级方法更鲁棒，能有效抑制外观变化干扰

2. **Latent Action Energy $E_{action}$**：

    - 核心公式：$E_{action}(t) = \|z_{q,t} - z_{q,t-1}\|_2$，即量化潜在空间中的时间差分 L2 范数
    - **物理意义**：稳定状态（无动作）时能量低；执行连续动作时 token 动态变化，能量持续高；语义转换（动作边界）时能量回落
    - 必须在**量化空间**计算（消融实验证明在原始速度或预量化空间计算效果极差）
    - **设计动机**：为什么在潜在空间而非像素空间？潜在空间已经编码了"行为意图"而非物理运动，因此对光照、轮微动等非语义变化免疫

3. **滞后状态机动作检测器**：

    - 双阈值 ON/OFF 控制器，带去抖动设计
    - 激活（OFF→ON）：信号 $y_t > \theta_{on}$ 持续 $u$ 帧
    - 去激活（ON→OFF）：信号 $y_t < \theta_{off}$ 持续 $d$ 帧
    - $\theta_{on}$ 通过无监督自校准确定：用速度能量做代理信号自动生成伪标签，再优化 F1 分数
    - **设计动机**：单通道因果架构支持实时在线处理，滞后机制避免噪声抖动产生的假边界

### 训练策略

- Motion Tokenizer 仅在未标注的训练集视频片段上训练
- 整个分割流水线无需任何标注，阈值靠自监督校准
- 聚类使用冻结随机初始化 Transformer（无训练），保证跨域泛化

## 实验关键数据

### 主实验：无监督时序动作分割

| 方法 | GTEA F1@5s | GTEA F1@2s | Breakfast F1@5s | Industrial Top F1@2s | Industrial Exo F1@2s |
|------|-----------|-----------|----------------|---------------------|---------------------|
| ABD | 81.92 | 74.23 | 54.50 | 34.08 | 29.86 |
| OTAS | 37.68 | 36.90 | **62.13** | 40.69 | 33.38 |
| Optical Flow | - | - | - | 43.68 | 42.54 |
| **LAPS (Ours)** | 73.12 | 63.20 | 58.82 | **81.27** | **81.93** |

在工业数据集上 LAPS 以巨大优势领先（F1@2s 提升约 2 倍），在公共基准上与 SOTA 可比。

### 消融实验

| 配置 | F1@2s (%) | Cluster ICSS |
|------|----------|-------------|
| Full Pipeline | **87.5** | **0.92** |
| $E_{action}$ from Pre-Quant. Latents | 25.2 | – |
| $E_{action}$ from Raw Velocities | 24.9 | – |
| w/o Transformer (Mean-pool) | – | 0.84 |
| w/o $M_\theta$ (用 CLIP) | 27.2 | 0.75 |

### 关键发现

- 在量化空间计算 $E_{action}$ 是关键（相比预量化/原始速度，F1 从 25% 提升到 87.5%）
- 专用 Motion Tokenizer 远优于通用 CLIP 特征（F1: 87.5% vs 27.2%）
- 聚类的 ICSS 语义一致性评分 0.926 远高于随机基线 0.804
- 冻结 Transformer 优于简单均值池化，说明显式时序建模对动作区分至关重要

## 亮点与洞察

- **范式转换**：从"视觉变化检测"转向"行为意图变化检测"，在潜在空间做分割是本文最核心的创新
- **工业适用性**：利用工业环境动作有限可数的先验，流水线完全无监督，可直接部署
- **端到端数据管道**：从原始视频到结构化 VLA 预训练数据的完整自动化流程
- ICSS 指标设计巧妙——用 VLM 语义相似度验证聚类质量，弥补 Silhouette 等几何指标的不足

## 局限与展望

- 目前仅限于高度重复的工业任务，对家庭/医院等非结构化环境的泛化有待验证
- 需要预定义聚类数 $k$，依赖领域知识
- 未验证下游 VLA 预训练的实际效果
- Motion Tokenizer 的训练需要一定量的无标注短视频片段

## 相关工作与启发

- **AMPLIFY**：本文 Motion Tokenizer 的基础架构，原本用于策略学习
- **GR00T / AgiBot GO-1**：VLA 预训练的代表工作，面临数据瓶颈
- **ABD / OTAS**：传统无监督动作分割基线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Latent Action Energy 指标新颖，潜在空间分割范式创新
- 实验充分度: ⭐⭐⭐⭐ 公共基准+工业数据集+VLM语义验证覆盖全面
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，流水线描述详细
- 价值: ⭐⭐⭐⭐ 解决 VLA 数据瓶颈的实用方案，工业应用前景好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](simpact_simulation-enabled_action_planning_using_vision-language_models.md)
- [\[ICCV 2025\] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](../../ICCV2025/multimodal_vlm/vq-vla_improving_vision-language-action_models_via_scaling_vector-quantized_acti.md)
- [\[CVPR 2025\] Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy](../../CVPR2025/multimodal_vlm/optimus-2_multimodal_minecraft_agent_with_goal-observation-action_conditioned_po.md)

</div>

<!-- RELATED:END -->
