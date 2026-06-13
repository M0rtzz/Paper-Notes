---
title: >-
  [论文解读] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM
description: >-
  [视频生成] 提出Prompt-A-Video，通过奖励引导的提示词进化流水线自动构建训练数据，经过SFT和DPO两阶段优化LLM，生成针对特定视频扩散模型偏好对齐的增强提示词。 文本到视频模型对输入提示词极为敏感——训练时使用的是LVLM生成的复杂描述，而用户输入往往简短粗糙。现有提示词优化方法存在三大问题： 模态不一致：…
tags:
  - "视频生成"
---

# Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2412.15156](https://arxiv.org/abs/2412.15156)
- **代码**: [GitHub](https://github.com/jiyt17/Prompt-A-Video)
- **领域**: 视频生成
- **关键词**: 视频生成, 提示词增强, LLM, DPO, 奖励引导进化

## 一句话总结
提出Prompt-A-Video，通过奖励引导的提示词进化流水线自动构建训练数据，经过SFT和DPO两阶段优化LLM，生成针对特定视频扩散模型偏好对齐的增强提示词。

## 研究背景与动机

文本到视频模型对输入提示词极为敏感——训练时使用的是LVLM生成的复杂描述，而用户输入往往简短粗糙。现有提示词优化方法存在三大问题：

**模态不一致**：图像提示词强调静态属性（构图、颜色），忽视视频特有的运动流畅性、叙事连贯性

**成本差异**：缺乏成熟的视频提示词平台和社区经验

**模型无感知**：GPT直接扩展的提示词未考虑特定视频模型的偏好

**核心目标**：设计一个以视频为中心、免人工、偏好对齐的提示词优化系统。

## 方法详解

### 多维度奖励系统

**图像级**：
- Aesthetic Predictor（美学评分）
- MPS（多维人类偏好评分）

**视频级**（VideoScore模型）：
- 视觉质量(VQ)、时间一致性(TC)、动态程度(DD)、文本-视频对齐(TVA)、事实一致性(FC)

共7个维度的综合评估。

### 奖励引导提示词进化（数据引擎）

借鉴进化算法思想，用GPT-4o作为进化操作器：

1. **评估**：生成视频 → 多维奖励打分 → 分数附加到对应提示词
2. **选择**：综合所有指标选择top-N提示词
3. **进化**：将初始提示词+高分历史提示词输入GPT-4o，一次生成3个精化版本

迭代4轮，最终选择各维度超过阈值且总分最高的提示词，构建(原始, 目标)训练对。

### 两阶段优化

**阶段一：SFT**

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)} \log p(y|s,x)$$

使用LoRA微调LLaMA3-Instruct，赋予基础的提示词增强能力。

**阶段二：DPO**

SFT模型为每个输入生成5个候选 → 生成视频 → 奖励模型打分 → 选择最优/最差构建三元组 → DPO优化：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

执行2轮迭代DPO，每轮基于上一轮模型重新生成三元组数据。

## 实验

### Open-Sora 1.2 + CogVideoX定量对比

| 模型/方法 | VQ | TC | DD | TVA | FC | Avg |
|---------|----|----|----|----|----|----|
| Open-Sora原始 | 3.079 | 3.084 | 3.203 | 3.156 | 3.060 | 3.116 |
| GPT-4o | 3.014 | 3.082 | 3.103 | 3.167 | 3.031 | 3.079 |
| **Ours(DPO-2)** | **3.254** | **3.286** | **3.411** | **3.358** | **3.282** | **3.318** |
| CogVideoX原始 | 2.899 | 2.886 | 3.186 | 3.167 | 2.808 | 2.989 |
| GLM-4 | 2.878 | 2.948 | 3.139 | 3.184 | 2.833 | 2.996 |
| **Ours(DPO-2)** | **2.930** | **3.019** | **3.183** | **3.259** | **2.888** | **3.056** |

Open-Sora上平均提升0.202（GPT-4o仅-0.037），CogVideoX上提升0.067。

### 用户研究（VBench 100提示词）

| 对比 | Prompt-A-Video胜率 |
|------|-------------------|
| vs 原始提示(Open-Sora) | ~65% |
| vs GPT-4o(Open-Sora) | ~55% |
| vs 原始提示(CogVideoX) | ~60% |

### 关键发现

1. SFT赋予基础能力但性能有限，DPO阶段带来实质性提升
2. 图像提示词增强模型（Promptist）直接用于视频反而降低性能
3. 两轮DPO后收敛，额外迭代不再带来改进
4. 方法可泛化到text-to-image任务（在HPSv2上超越Promptist和PAE）

## 亮点与洞察

1. **闭环优化**：进化算法→SFT→DPO的递进策略，每阶段有明确目标
2. **模型特异性**：不同视频模型有不同偏好，同一框架可针对性训练
3. **自动化数据引擎**：零人工标注，完全靠奖励模型驱动
4. **泛化性**：视频优化的方法也能提升图像生成质量

## 局限性

- 进化流水线依赖GPT-4o，数据构建成本不低
- DPO训练需要大量视频生成和评估，计算开销显著
- 动态程度(DD)在优化过程中可能被其他指标压制
- 奖励模型本身的偏差会传递到优化结果

## 相关工作

- **视频生成**: Open-Sora, CogVideoX, Sora
- **提示词优化**: Promptist, PAE
- **对齐方法**: DPO, PPO, RLHF

## 评分
- 新颖性：★★★★☆ — 首个面向视频扩散模型的提示词优化系统
- 技术深度：★★★★☆ — 多阶段优化设计精巧
- 实用性：★★★★★ — 直接提升用户视频生成体验

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
