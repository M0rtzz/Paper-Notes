---
title: >-
  [论文解读] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation
description: >-
  [ICCV 2025][人体理解][双人舞蹈生成] 介绍 Multimodal DuetDance (MDD)，首个同时整合动作、音乐和文本描述的大规模专业级双人舞蹈数据集，包含 620 分钟动捕数据、15 种舞蹈类型和超过 10K 条细粒度文本标注，并提出 Text-to-Duet 和 Text-to-Dance Accompaniment 两个新任务。
tags:
  - ICCV 2025
  - 人体理解
  - 双人舞蹈生成
  - 多模态数据集
  - 文本控制运动生成
  - 动作捕捉
  - SMPL-X
---

# MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation

**会议**: ICCV 2025  
**arXiv**: [2508.16911](https://arxiv.org/abs/2508.16911)  
**代码**: https://gprerit96.github.io/mdd-page  
**领域**: 人体运动理解 / 舞蹈生成  
**关键词**: 双人舞蹈生成, 多模态数据集, 文本控制运动生成, 动作捕捉, SMPL-X

## 一句话总结

介绍 Multimodal DuetDance (MDD)，首个同时整合动作、音乐和文本描述的大规模专业级双人舞蹈数据集，包含 620 分钟动捕数据、15 种舞蹈类型和超过 10K 条细粒度文本标注，并提出 Text-to-Duet 和 Text-to-Dance Accompaniment 两个新任务。

## 研究背景与动机

双人舞蹈是最复杂的交互式人体运动形式之一，需要两位舞者之间精确的协调和同步。相比于独舞，双人舞涉及复杂的空间关系、动态的搭档互动以及对音乐节拍的持续适应。现有工作的局限：

**InterGen/Inter-X**：提供双人交互运动数据集和文本标注，但缺少专业舞蹈动作和同步音频

**Duolando (DD100)**：首个双人舞蹈数据集但仅含 1.95 小时数据，且缺少文本标注

**InterDance**：3.93 小时双人舞蹈，仍无文本标注

**TM2D**：结合文本和音乐但存在数据分布不匹配问题

核心 Gap：**没有任何数据集同时整合动作、音乐和文本三种模态来支持双人舞蹈生成**。

## 方法详解

### 整体框架

MDD 不是一个方法论文，而是一个**数据集+基准**工作。核心贡献在于数据集构建和新任务定义。

### 关键设计

1. **数据采集流水线**：

    - **音乐选择**：优先使用无版权音乐，每个舞种准备 50-60 首不同曲目
    - **动作捕捉**：使用 OptiTrack 系统（16 台红外相机，120 fps），53 个反光标记点
    - **受试者**：30 名舞者（16 女 14 男），均为中级或高级水平，至少 3 年经验
    - **动作后处理**：异常值剔除、高斯滤波、零姿态修正、块感知混合平滑
    - **运动表示**：使用 SMPL-X 参数化模型（$\theta \in \mathbb{R}^{N \times 55 \times 3}$, $\beta \in \mathbb{R}^{N \times 10}$, $t \in \mathbb{R}^{N \times 3}$）

2. **细粒度文本标注系统**：

    - 标注维度覆盖三大类别：**空间关系**（交互位置、朝向、连接点）、**身体动作**（动作类型、身体部位）、**节奏**（能量、速度）
    - 原始舞者自行标注，保证专业术语准确性
    - GPT-4o 语法润色 + 第二轮专家审校
    - 平均每条标注 41 词（长于现有运动-文本数据集），词汇量 1,722 个独特词
    - 发布超过 10,187 条标注

3. **两个新任务定义**：

    - **Text-to-Duet**：给定文本描述 $c$ 和音乐 $m$，生成双人舞蹈 $(\mathbf{x_l}, \mathbf{x_f})$，学习函数 $F(c, m) \mapsto \mathbf{x}$
    - **Text-to-Dance Accompaniment**：给定文本 $c$、音乐 $m$ 和领舞者动作 $\mathbf{x_l}$，生成跟随者动作 $\mathbf{x_f}$，学习函数 $G(c, m, \mathbf{x_l}) \mapsto \mathbf{x_f}$

### 损失函数 / 训练策略

运动表示拟合使用优化目标：
$$E(\theta, t, \beta) = \lambda_1 \frac{1}{N} \sum_{j \in \mathcal{J}} \lambda_p \|J_j(M(\theta, t, \beta)) - g_j\|_2^2 + \lambda_2 \|\theta\|_2^2$$

基线模型训练：所有模型使用 AdamW 优化器，batch=64，训练 3000 epochs。

## 实验关键数据

### 主实验 — Text-to-Duet

| 方法 | R-Prec Top1↑ | R-Prec Top3↑ | FID↓ | MM Dist↓ | BED↑ | BAS↑ |
|------|-------------|-------------|------|----------|------|------|
| Ground Truth | 0.231 | 0.522 | 0.065 | 0.077 | 0.327 | 0.170 |
| MDM (text-only) | 0.082 | 0.192 | 1.420 | 2.133 | 0.211 | 0.186 |
| MDM (both) | 0.061 | 0.163 | 1.739 | 2.244 | 0.194 | 0.231 |
| InterGen (text-only) | 0.113 | 0.305 | 0.405 | 1.462 | 0.422 | 0.194 |
| InterGen (both) | 0.105 | 0.302 | 0.426 | 1.532 | 0.385 | 0.185 |
| InterGen w. Jukebox | **0.138** | **0.341** | **0.410** | **1.396** | **0.454** | 0.184 |

### 主实验 — Text-to-Dance Accompaniment

| 方法 | R-Prec Top1↑ | FID↓ | MM Dist↓ | BED↑ | BAS↑ |
|------|-------------|------|----------|------|------|
| Ground Truth | 0.231 | 0.065 | 0.077 | 0.327 | 0.170 |
| DuoLando (text-only) | 0.047 | 1.538 | 2.811 | 0.311 | 0.195 |
| DuoLando (music-only) | 0.069 | 0.721 | 2.633 | 0.305 | 0.216 |
| DuoLando (both) | **0.078** | **0.698** | **2.113** | **0.395** | **0.224** |

### 消融实验 — 文本消融 (InterGen, Text-to-Duet)

| 文本类型 | R-Prec Top1↑ | FID↓ | MM Dist↓ | BED↑ |
|----------|-------------|------|----------|------|
| 无文本(仅音乐) | 0.023 | 2.014 | 2.526 | 0.364 |
| 动作名称 | 0.061 | 0.721 | 2.211 | 0.355 |
| 原始文本 | 0.091 | 0.511 | 1.722 | 0.381 |
| GPT-4o 润色文本 | **0.105** | **0.426** | **1.532** | **0.385** |

### 关键发现

- InterGen 持续优于 MDM，表明其更适合交互式生成任务
- 使用 Jukebox 音乐嵌入略优于 MFCC，更丰富的音乐表示可提升生成质量
- 文本+音乐多模态条件在 Dance Accompaniment 任务上明显优于单模态
- GPT-4o 润色后的文本描述效果最佳，LLM 可提升标注质量
- 用户研究中，GPT-4o 润色文本生成的动作在文本对齐和整体质量上获得最高评分

## 亮点与洞察

- **填补重要空白**：首个同时提供动作、音乐、文本三模态的双人舞蹈数据集
- **规模与质量并重**：620 分钟专业动捕数据，涵盖 15 种舞蹈，是现有最大双人舞数据集
- **标注设计系统化**：从空间关系、身体动作、节奏三个维度构建标注体系
- **任务定义合理**：Text-to-Duet 和 Text-to-Dance Accompaniment 分别对应跟随生成和协调生成两种实际需求
- **数据集对比全面**：与 11 个相关数据集的详细对比突显优势

## 局限与展望

- 基线方法性能与 Ground Truth 差距较大，说明任务极具挑战性
- 数据集虽然是"大规模"但对数据驱动方法可能仍不够，特别是某些舞种仅约 30 分钟
- 仅使用 OptiTrack 标记点，缺少面部表情和手指细节
- 文本标注依赖 GPT-4o 润色，可能引入模型偏见
- 文本评估器在 MDD 上训练，评估可能有过拟合倾向
- BAS 指标可能奖励抖动运动，需谨慎解读

## 相关工作与启发

- InterGen 和 Inter-X 的交互运动数据集为本工作提供了方法论参考
- Duolando 的 follower GPT + 强化学习架构提供了 Dance Accompaniment 的基线范式
- 数据标注的多阶段流程（舞者标注→LLM润色→专家审校）值得其他数据集构建工作借鉴
- 双人舞蹈中领舞-跟随的动态关系建模是未来重要研究方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将文本、音乐、动作三模态整合于双人舞蹈场景，数据集定位独特
- **实验充分度**: ⭐⭐⭐ 基线适配合理但数量有限，缺少更深入的分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据集统计分析详尽
- **价值**: ⭐⭐⭐⭐ 数据集对多人运动生成和舞蹈AI社区有重要推动价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](../../CVPR2025/human_understanding/scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[ICCV 2025\] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation](align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)

<!-- RELATED:END -->
