---
title: >-
  [论文解读] Can Text-to-Video Generation Help Video-Language Alignment?
description: >-
  [CVPR 2025][视频生成][视频-语言对齐] 提出 SynViTA 框架探索文本到视频生成模型产生的合成视频能否改善视频-语言对齐（VLA），通过基于对齐质量的样本加权和语义一致性正则化解决合成视频的语义不一致和外观偏差问题，在时序挑战性任务上提升 4+ 个点。 领域现状：视频-语言对齐（VLA）训练 VLM 判断视…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频-语言对齐"
  - "合成视频"
  - "文本到视频生成"
  - "对齐加权"
  - "语义一致性"
---

# Can Text-to-Video Generation Help Video-Language Alignment?

**会议**: CVPR 2025  
**arXiv**: [2503.18507](https://arxiv.org/abs/2503.18507)  
**代码**: [https://lucazanella.github.io/synvita/](https://lucazanella.github.io/synvita/)  
**领域**: 视频生成  
**关键词**: 视频-语言对齐、合成视频、文本到视频生成、对齐加权、语义一致性

## 一句话总结
提出 SynViTA 框架探索文本到视频生成模型产生的合成视频能否改善视频-语言对齐（VLA），通过基于对齐质量的样本加权和语义一致性正则化解决合成视频的语义不一致和外观偏差问题，在时序挑战性任务上提升 4+ 个点。

## 研究背景与动机

**领域现状**：视频-语言对齐（VLA）训练 VLM 判断视频与文本描述是否一致，是视频理解的基础任务。现有方法依赖人工标注或自动生成的负样本训练。

**现有痛点**：(1) 高质量视频-文本配对数据稀缺。(2) 文本到视频（T2V）生成模型可以产生大量合成视频，但这些视频可能与输入文本语义不一致（如动作不匹配）。(3) 合成视频的视觉外观与真实视频差异大，模型可能学习到"看外观而非看语义"的捷径。

**核心矛盾**：T2V 模型提供了丰富的视频来源但质量不可控——部分合成视频忠实反映文本语义，部分则完全不匹配。如何利用好的、过滤掉差的？

**本文目标** 系统性地研究合成视频在 VLA 训练中的价值，并设计方法解决合成视频的质量问题。

**切入角度**：用 VQAScore 集成评估合成视频与目标/参考文本的对齐质量差异作为动态权重，同时用最长公共子序列（LCS）提取正负描述的共享语义做三元组正则化，迫使模型关注语义差异而非外观差异。

**核心 idea**：用 VQAScore 对齐质量动态加权合成视频的训练贡献，配合语义一致性正则化抑制外观偏差，使合成视频有效提升 VLA。

## 方法详解

### 整体框架
真实视频+文本对训练基础 VLA 损失 → 合成视频通过 T2V 模型生成 → VQAScore 评估合成视频-目标文本 vs 合成视频-参考文本的对齐差异 → 差异作为权重加入合成 VLA 损失 → LCS 提取共享描述做三元组语义正则化。

### 关键设计

1. **基于对齐的样本加权**:

    - 功能：自动过滤语义不一致的合成视频
    - 核心思路：$\omega_i = \max(0, \bar{f}(V^s_i, t^s_i) - \bar{f}(V^s_i, t^r_i))$，其中 $\bar{f}$ 是 VQAScore 集成。如果合成视频与目标文本的对齐分数高于与参考文本的对齐分数，说明视频确实反映了目标语义，给高权重
    - 设计动机：消融显示固定权重 1.0 时 SSv2-Temporal 仅 12.54，加权后提升到 17.32——5 个点差异说明质量过滤至关重要

2. **语义一致性正则化（SCR）**:

    - 功能：迫使模型聚焦语义差异而非视觉外观差异
    - 核心思路：用 LCS 算法提取正负描述的共享部分作为"锚点"文本 $t'$。三元组损失要求：合成视频与正描述的对齐 > 合成视频与共享描述的对齐 > 合成视频与负描述的对齐
    - 设计动机：不加 SCR 时模型容易靠"是否是合成视频"来区分正负样本，SCR 强制模型关注文本语义中的区分性部分

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{real} + \mathcal{L}_{syn}^\phi + \lambda_{scr} \cdot \mathcal{L}_{scr}^\phi$。模型无关——在 mPLUG-Owl 7B 和 Video-LLaVA 上都有效。CogVideoX 是最好的单一 T2V 生成器。

## 实验关键数据

### 主实验

| 模型 | Human-Hard | SSv2-Temporal | SSv2-Events | ATP-Hard |
|------|-----------|-------------|-------------|---------|
| VideoCon (mPLUG) | 74.76 | 13.00 | 10.37 | 35.46 |
| **SynViTA (mPLUG)** | 74.54 | **17.32** | **12.54** | **37.31** |
| VideoCon (Video-LLaVA) | 75.74 | 19.77 | 10.01 | 38.76 |
| **SynViTA (Video-LLaVA)** | **76.86** | **20.10** | **11.21** | **39.88** |

### 消融实验

| 加权策略 | SSv2-Temporal | SSv2-Events |
|---------|-------------|-------------|
| 固定 1.0 | 12.54 | 8.48 |
| 对齐差值加权 | **17.32** | **12.54** |

### 关键发现
- **合成视频在时序任务上有帮助**：SSv2-Temporal 提升 4.3 个点（13→17.3），但在分布内 entailment 任务上可能轻微下降
- **质量过滤至关重要**：不加权 vs 加权差 5 个点
- **Flip 和 Hallucination 类型的不对齐最难处理**——T2V 模型在这些语义操作上质量最差

## 亮点与洞察
- **首次系统性研究 T2V 合成视频对 VLA 的价值**，结论是"有条件的有用"——对时序理解有帮助但需要质量控制
- **VQAScore 加权是通用的合成数据质量控制方案**，可推广到其他使用合成数据的场景

## 局限与展望
- 合成视频在分布内任务上可能略降，说明合成-真实域差距仍存在
- 依赖 VQAScore 的准确性做质量评估
- 仅测试了 3 种 T2V 模型，更强的模型（如 Sora）可能改变结论

## 相关工作与启发
- **vs VideoCon**：VideoCon 仅用真实数据。SynViTA 通过合成视频增强在时序任务上显著提升
- **vs 图像合成增强（如 StableRep）**：视频合成比图像合成面临更多挑战（时序一致性、动作匹配），SynViTA 的质量控制策略应对了这些挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 T2V 合成视频用于 VLA 训练
- 实验充分度: ⭐⭐⭐⭐ 多 T2V 模型、多 VLM 基线、详细消融
- 写作质量: ⭐⭐⭐⭐ 分析视角全面
- 价值: ⭐⭐⭐⭐ 对利用合成数据训练视频模型有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoDPO: Omni-Preference Alignment for Video Diffusion Generation](videodpo_omni-preference_alignment_for_video_diffusion_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](../../ICCV2025/video_generation/etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)
- [\[CVPR 2025\] TransPixeler: Advancing Text-to-Video Generation with Transparency](transpixeler_advancing_text-to-video_generation_with_transparency.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)

</div>

<!-- RELATED:END -->
