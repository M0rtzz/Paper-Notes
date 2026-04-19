---
title: >-
  [论文解读] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization
description: >-
  [视频理解] > 提出 VPO 框架，基于三大原则（无害、准确、有用）系统性优化视频生成的文本提示，通过原则导向的SFT和多反馈偏好优化，显著提升生成视频的安全性、对齐度和质量。
tags:
  - 视频理解
---

# VPO: Aligning Text-to-Video Generation Models with Prompt Optimization

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.20491](https://arxiv.org/abs/2503.20491)
- **作者**: Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu 等 (清华大学 CoAI, 智谱AI, 清华 KEG)
- **代码**: [GitHub](https://github.com/thu-coai/VPO)
- **领域**: 视频理解 / 视频生成
- **关键词**: Prompt Optimization, 视频生成, 文本对齐, RLHF, DPO, 安全性, CogVideoX

## 一句话总结

> 提出 VPO 框架，基于三大原则（无害、准确、有用）系统性优化视频生成的文本提示，通过原则导向的SFT和多反馈偏好优化，显著提升生成视频的安全性、对齐度和质量。

## 研究背景与动机

### 问题定义
视频生成模型训练时使用精心标注的详细描述，但推理时用户输入往往简短、模糊或结构不良。Prompt optimization 旨在弥合这一鸿沟，将用户输入转化为高质量的视频生成提示。

### 现有方法的问题

**安全隐患**: 现有 LLM in-context learning 方法不显式确保优化后的提示安全，可能生成有害内容

**意图扭曲**: 提示重写可能无意中改变用户本意或引入偏差

**忽视视频质量**: 现有方法优化提示的语义丰富度，但不考虑对最终视频质量的影响

**LLM拒绝问题**: LLM 可能拒绝处理含敏感关键词的查询（如抽象表述"20 - 11 coins = 9 coins"）

### 核心原则（类比 LLM 的 HHH）
- **Harmless（无害）**: 优化后提示不应包含血腥、暴力等有害内容
- **Accurate（准确）**: 准确保持用户意图，除安全问题外不改变原意
- **Helpful（有用）**: 提示应详细描述性，帮助模型生成高质量视频

## 方法详解

### 整体框架
VPO 分为两个阶段：
1. **Principle-Based SFT**: 构建高质量SFT数据集并微调基座模型
2. **Multi-Feedback Preference Optimization**: 结合文本级和视频级反馈进行DPO训练

### Stage 1: Principle-Based SFT

#### 查询筛选
- 来源: VidProM 数据集（100万+真实 text-to-video 查询）
- 规则过滤: 关键词、特殊字符、长度
- 多样性过滤: self-BLEU 去重
- 安全查询: 提取不安全标签的查询 + LLM重新评估
- 最终: ~18k 通用查询 + 2k 安全相关查询

#### 初始优化提示生成
使用 GPT-4o + 精心设计的 few-shot 示例生成初始优化提示

#### 原则驱动精炼
- LLM-as-a-judge 按三大原则评估提示
- 识别问题（有害内容、遗漏关键信息、模糊描述），生成批评 $c$
- 基于批评精炼: $(x, p) \rightarrow (x, p_{refined})$

#### SFT训练
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(s|x, s_{<i})$$

### Stage 2: Multi-Feedback Preference Optimization

#### 文本级偏好数据构建
- 对每个查询 $x$，从SFT模型采样 $K$ 个优化提示
- LLM-as-a-judge 检查是否违反原则
- 有缺陷的提示 → 精炼 → 构成偏好对 $(x, p_j < p_{j_{refined}})$

#### 视频级偏好数据构建
- 通过文本级检查的提示 → 生成视频
- 使用 VisionReward 评估视频质量得分 $r_m$
- 根据得分差构建偏好对 $(x, p_m < p_{m+1})$，要求得分差 > 0.5

#### DPO训练
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,p_w,p_l) \sim D_{dpo}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(p_w|x)}{\pi_{ref}(p_w|x)} - \beta \log \frac{\pi_\theta(p_l|x)}{\pi_{ref}(p_l|x)} \right) \right]$$

- 训练数据: $D_{dpo} = D_{text} \cup D_{video}$
- 基座模型: LLaMA3-8B-Instruct

## 实验关键数据

### 主实验结果 — MonetBench & VBench

| 方法 | MonetBench Overall | Human Action | Scene | Multiple Objects | Appear. Style |
|------|------|------|------|------|------|
| **CogVideoX-2B** |
| Original Query | 3.27 | 80.00 | 28.34 | 40.17 | 22.60 |
| GLM-4 Few-Shot | 3.57 | 96.20 | 55.51 | 68.40 | 23.47 |
| GPT-4o Few-Shot | 3.58 | 98.20 | 52.53 | 63.63 | 23.73 |
| VPO-SFT | 3.59 | 97.00 | 55.04 | 68.98 | 24.13 |
| **VPO** | **3.76** | **99.00** | **55.83** | **70.17** | **24.20** |
| **CogVideoX-5B** |
| Original Query | 3.77 | 88.00 | 41.32 | 45.67 | 23.37 |
| GLM-4 Few-Shot | 3.98 | 98.40 | 55.60 | 72.38 | 24.39 |
| GPT-4o Few-Shot | 4.03 | 99.20 | 53.13 | 72.21 | 24.20 |
| VPO-SFT | 4.01 | 97.20 | 58.40 | 73.70 | 24.55 |
| **VPO** | **4.15** | **99.60** | **55.68** | **75.73** | **24.57** |

### 文本对齐评估

| 方法 | Aligned↑ | Unsafe↓ | Imprecise↓ | Refusal↓ |
|------|----------|---------|------------|----------|
| GLM-4 Few-Shot | 83.4 | 5.4 | 10.0 | 1.2 |
| GPT-4o Few-Shot | 86.4 | 2.4 | 8.6 | 2.6 |
| VPO-SFT | 93.8 | 0.8 | 5.4 | 0.0 |
| **VPO (2B)** | **94.6** | **0.6** | **4.8** | **0.0** |
| **VPO (5B)** | **94.8** | **0.4** | **4.8** | **0.0** |

### 跨模型泛化 — Open-Sora 1.2

| 方法 | Human Action | Scene | Multiple Objects |
|------|------|------|------|
| Original Query | 88.80 | 44.08 | 55.99 |
| GPT-4o Few-Shot | 92.40 | 53.21 | 65.02 |
| **VPO** | **97.00** | **53.58** | **67.88** |

### 关键发现
1. **VPO显著领先**: 在 CogVideoX-5B 上，人类评估 VPO 的胜率比原始查询高37.5%，比官方prompt优化高14%
2. **安全性大幅提升**: 不安全率从 5.4%(GLM-4) 降至 0.4%(VPO)，Level 1 完全安全率大幅提高
3. **文本级反馈关键**: 移除文本级反馈不仅降低安全性，还影响通用视频生成质量
4. **VPO超越Diffusion DPO**: 作为RLHF方法，VPO优于Diffusion DPO，且两者可叠加
5. **跨模型泛化**: 在 CogVideoX-2B 上训练的 VPO 直接提升 Open-Sora 1.2 性能
6. **迭代优化稳定**: 可迭代优化3轮后趋于稳定，不会过度优化损害质量

## 亮点与洞察

1. **原则驱动的系统性方法**: 不同于简单的提示重写，VPO 将 LLM 对齐的 HHH 原则引入视频提示优化，方法论严谨
2. **多层级反馈**: 同时考虑文本级（安全+准确）和视频级（质量），形成完整的优化闭环
3. **Prompt Optimization ≈ RLHF**: 揭示了一个重要洞察——优化提示和优化模型是对齐视频生成模型的正交且互补手段
4. **实用性强**: VPO 训练开销小（基于 LLaMA3-8B），部署简单，泛化性好
5. **安全性不应被忽视**: Case study 生动展示了 few-shot 方法生成有害内容的风险

## 局限性

1. **依赖特定视频生成模型**: 视频级偏好数据依赖目标模型生成视频，更换模型需重新构建
2. **评估模型偏差**: 视频级反馈依赖 VisionReward，受限于该奖励模型的偏差
3. **GPT-4o 依赖**: SFT 数据构建和文本级评判均依赖 GPT-4o
4. **仅针对提示优化**: 不优化视频生成模型本身，改善幅度受限于模型能力上限

## 相关工作与启发

### 相关研究
- **视频生成**: CogVideoX, Open-Sora, Stable Video Diffusion, HunyuanVideo
- **Prompt优化**: AutoPrompt, Promptist, Prompt-A-Video
- **RLHF on Diffusion**: Diffusion DPO
- **视频评估**: VBench, VisionReward, MonetBench

### 启发
- 训练-推理时的数据分布差距是所有生成模型的共性问题，prompt optimization 是轻量级的解决方案
- Principle-based refinement (批评+精炼) 是构建高质量数据的有效范式
- 安全性与质量并非对立——改善安全也能提升通用性能

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [Aligning Moments in Time using Video Queries](aligning_moments_in_time_using_video_queries.md)
- [DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)

<!-- RELATED:END -->
