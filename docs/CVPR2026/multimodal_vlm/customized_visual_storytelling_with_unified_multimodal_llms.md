---
title: >-
  [论文解读] Customized Visual Storytelling with Unified Multimodal LLMs
description: >-
  [CVPR 2026][多模态][视觉故事生成] 提出 VstoryGen 框架和核心组件 CustFilmer，基于统一多模态大语言模型（UMLLM）实现多模态故事定制生成，支持文本描述、角色/场景参考图像和镜头类型的联合条件控制，并构建了 MSB 和 M2SB 两个新 benchmark。
tags:
  - CVPR 2026
  - 多模态
  - 视觉故事生成
  - 多模态定制
  - 统一多模态LLM
  - 镜头类型控制
  - 关键帧生成
---

# Customized Visual Storytelling with Unified Multimodal LLMs

**会议**: CVPR 2026  
**arXiv**: [2603.27690](https://arxiv.org/abs/2603.27690)  
**代码**: 无（项目页面未明确提供）  
**领域**: 多模态VLM / 视觉叙事生成  
**关键词**: 视觉故事生成, 多模态定制, 统一多模态LLM, 镜头类型控制, 关键帧生成

## 一句话总结
提出 VstoryGen 框架和核心组件 CustFilmer，基于统一多模态大语言模型（UMLLM）实现多模态故事定制生成，支持文本描述、角色/场景参考图像和镜头类型的联合条件控制，并构建了 MSB 和 M2SB 两个新 benchmark。

## 研究背景与动机

**领域现状**：文本到视频生成领域进展迅速，但生成长序列连贯叙事视频仍是挑战。视觉故事生成方法（ConsiStory、StoryDiffusion、CharaConsist）主要依赖纯文本输入，少数支持角色 ID 保持。

**现有痛点**：(1) 现有方法仅使用文本输入，无法利用参考图像定制角色和场景；(2) 背景一致性常被忽视，仅关注前景角色；(3) 生成视角单调，缺乏电影感的镜头语言（远景/近景/特写等）；(4) 多角色交互场景生成能力不足。

**核心矛盾**：如何在保持角色和场景一致性的同时，实现灵活的多模态条件控制（文本+参考图+镜头类型）？

**本文要解决**：利用 UMLLM 的多模态理解和生成能力，构建支持丰富多模态条件的视觉叙事流水线。

**切入角度**：将 UMLLM 的图像编辑能力扩展为 keywise 自回归故事生成，通过结构化检索和镜头类型 prompt tuning 增强一致性和电影感。

**核心 idea**：UMLLM + 结构化多模态脚本 + 视觉参考记忆库 + 镜头类型 prompt tuning = 可定制的视觉叙事。

## 方法详解

### 整体框架
VstoryGen 三阶段流水线：
1. **多模态脚本生成**：GPT-4o 从自由文本描述生成结构化脚本（text prompts + 角色/背景参考图 + 镜头类型）
2. **CustFilmer 关键帧生成**：基于脚本生成一致性关键帧
3. **TI2V 视频扩展**：用现有 text-and-image-to-video 模型将关键帧扩展为视频片段

### 关键设计

1. **Text Prompt Consolidation (TPC)**：

    - **功能**：将同一故事的所有 prompt $P = \{p_1, \ldots, p_n\}$ 在同一 batch 中联合编码，通过 LLM 自回归生成 hidden states $H = \{h_1, \ldots, h_n\}$。
    - **核心思路**：利用 LLM 的上下文一致性——在同一上下文窗口中编码的不同事件描述，其 hidden states 自然保持语义和身份一致性。
    - **设计动机**：相比独立编码每个 prompt，联合编码使不同帧的文本条件在嵌入空间中保持一致，从而在生成时保持角色和场景连贯。

2. **Visual Reference Memory Bank and Retrieval**：

    - **功能**：存储初始参考图（角色肖像、背景场景）和已生成的关键帧，以结构化 key-value 字典组织。每个时间步 $t$ 根据脚本中的角色/背景提及作为 query 精确检索对应视觉参考。
    - **核心思路**：$z_t = \text{VAE}[\mathcal{R}_t, \{\text{Scale}_\alpha(I_{t-i})\}_{i=1}^\mu]$
    - **设计动机**：不用 embedding 检索（可能模糊），而用结构化脚本标注确保精确可解释的参考选择。检索最近 $\mu$ 帧提供时间连贯性。$\alpha$ 参数平衡一致性和多样性。

3. **Shot-type Prompt Tuning**：

    - **功能**：在 Condensed Movie Dataset (CMD) 上学习一组镜头类型嵌入 $E_{\text{shot}}(k_t) \in \mathbb{R}^{d \times N}$，作为 hidden state 的前缀：$h_t' = [E_{\text{shot}}(k_t); h_t]$
    - **核心思路**：参数高效的 prompt tuning，仅学习镜头相关的嵌入，不修改基础模型
    - **设计动机**：通用 UMLLM 不具备电影镜头语言的构图先验，通过少量可学习参数（4000 迭代）注入镜头类型知识，产生远景/近景/特写等多样化视角。

4. **Keyframe-wise Autoregressive Generation**：

    - **功能**：将标准 UMLLM 的单次图像编辑扩展为关键帧级自回归生成：$I_t = \text{DiT}(h_t, z_t)$
    - **设计动机**：避免多轮对话（效率低+误差累积），通过 VAE 编码参考图直接注入 DiT 解码器保留低层视觉信息。

### 损失函数 / 训练策略
- Shot-type prompt tuning：在 CMD 电影数据上训练 4000 迭代
- 推理时使用 OmniGen2 作为骨干 UMLLM
- $\alpha=0.75$, $d=2048$, $N=30$

## 实验关键数据

### 主实验——MSB Benchmark（一致性指标）

| 方法 | 基础模型 | CLIP-I-fg (Inter)↑ | CLIP-I-bg (Inter)↑ | Avg Consistency↑ |
|------|---------|-------------------|-------------------|------------------|
| IP-Adapter | SDXL | 0.901 | 0.936 | 0.846 |
| ConsiStory | SDXL | 0.868 | 0.884 | 0.812 |
| StoryDiffusion | SDXL | 0.857 | 0.900 | 0.831 |
| CharaConsist | Flux.1 | 0.904 | 0.945 | 0.852 |
| **CustFilmer** | **OmniGen2** | **0.905** | **0.961** | **0.858** |

文本对齐与质量指标：

| 方法 | CLIP-T↑ | IAS↑ | IQS↑ | STA (镜头)↑ |
|------|--------|------|------|-----------|
| ConsiStory | **0.303** | 0.431 | 0.385 | 0.406 |
| CharaConsist | 0.265 | 0.448 | 0.415 | 0.247 |
| **CustFilmer** | 0.285 | **0.450** | **0.423** | **0.418** |

### 消融实验

| 配置 | Avg-Consistency↑ | 说明 |
|------|-----------------|------|
| 无 TPC + 无 Retrieval | 0.854 | 基线 |
| + TPC | 0.855 | 微弱提升 |
| + Retrieval | 0.856 | 微弱提升 |
| + TPC + Retrieval | **0.858** | 两者互补 |

$\alpha$ 参数消融：

| $\alpha$ | CLIP-T↑ | Avg-Consistency↑ | 说明 |
|---------|--------|-----------------|------|
| 0.125 | **0.289** | 0.850 | 多样性高但不一致 |
| 0.75 | 0.285 | 0.858 | 平衡选择 |
| 1.00 | 0.284 | **0.860** | 最一致但不够多样 |

### 关键发现
- CustFilmer 在整体一致性上最优，尤其是背景一致性 (CLIP-I-bg) 显著超越所有方法
- 镜头类型控制准确率 (STA=0.418) 远超非定制方法
- CLIP-T 略低于 ConsiStory 归因于不同骨干模型（SDXL 训练时用 CLIP encoder，天然优势）
- $\alpha=0.75$ 在一致性和多样性之间取得最佳平衡

## 亮点与洞察
- **完整的多模态故事流水线**：从自由文本描述→结构化脚本→关键帧→视频，端到端可用
- **镜头类型控制**：首次在视觉故事生成中引入电影镜头语言，显著增强叙事表现力
- **新 Benchmark 贡献**：MSB 和 M2SB 填补了多模态故事定制的评估空白
- **基于 UMLLM**：利用了统一多模态 LLM 的理解+生成能力，代表了故事生成的新范式

## 局限与展望
- 依赖 GPT-4o 做脚本生成（成本和延迟）
- TPC 和 Retrieval 带来的一致性提升幅度较小（0.854→0.858），设计的边际效益有限
- 多角色场景（M2SB）上优势不如单角色场景明显
- 未与最新的专用视频生成模型（如 Veo3）直接比较

## 相关工作与启发
- 与 CharaConsist 的对比表明，仅靠文本输入限制了定制灵活性
- UMLLM（特别是 OmniGen2）作为故事生成骨干是一个有前景的方向
- 镜头类型 prompt tuning 的思路可推广到其他需要构图控制的生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态定制+镜头控制的组合有创新，但各组件较增量
- 实验充分度: ⭐⭐⭐⭐ 新 benchmark+多基线对比+消融，但缺乏视频层面深入评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 对视觉叙事生成领域有实际推动，benchmark 和框架可被后续工作采用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)

<!-- RELATED:END -->
