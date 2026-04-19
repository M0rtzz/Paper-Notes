---
title: >-
  [论文解读] MotionChain: Conversational Motion Controllers via Multimodal Prompts
description: >-
  [ECCV 2024][图像生成] 提出 MotionChain，一个视觉-运动-语言统一模型，通过多模态提示在多轮对话中生成连续、长期的人体运动序列，支持文本、图像和运动的联合理解与生成。
tags:
  - ECCV 2024
  - 图像生成
---

# MotionChain: Conversational Motion Controllers via Multimodal Prompts

**会议**: ECCV 2024  
**arXiv**: [2404.01700](https://arxiv.org/abs/2404.01700)  
**领域**: 图像生成

## 一句话总结

提出 MotionChain，一个视觉-运动-语言统一模型，通过多模态提示在多轮对话中生成连续、长期的人体运动序列，支持文本、图像和运动的联合理解与生成。

## 研究背景与动机

- **大语言模型（LLM）的成功**：LLM 在多轮对话和上下文保持方面表现出色，但这种能力在人体运动生成领域几乎未被探索
- **现有运动生成方法的局限**：MDM、MLD、MotionGPT 等方法将运动任务视为单轮条件生成，缺乏上下文理解和多轮连续生成能力
- **应用需求**：人形机器人、游戏智能体等需要通过直觉化的多轮对话逐步执行人类任务的能力
- **数据稀缺挑战**：相比图像-语言、图像-人体姿态等配对数据集，文本-运动配对数据极为有限
- **核心观察**：人体运动和自然语言都是序列化的、可以连续"书写"的，因此可以借助视觉-语言指令微调的方法来实现对话式运动生成

## 方法详解

### 整体框架

MotionChain 由三个核心组件构成：
1. **多模态 tokenizer**：将文本、图像和运动统一编码为离散 token
2. **视觉-运动感知语言模型**：基于预训练语言模型（Flan-T5）理解多模态输入并生成运动/文本答案
3. **运动组合机制**：通过 token 拼接实现多轮运动的连续衔接

### 关键设计

**运动 Tokenizer（VQ-VAE）**：
- 使用 1D 卷积编码器将运动序列 $m^{1:M}$ 编码为潜在向量，经量化后映射到离散码本
- 码本大小 $K \in \mathbb{R}^{512 \times 1024}$，时间下采样率 $l=4$
- 训练损失：重建损失 $\mathcal{L}_r$ + 嵌入损失 $\mathcal{L}_e$ + 承诺损失 $\mathcal{L}_c$

**视觉 Tokenizer**：
- 图像输入：冻结的 CLIP ViT-L/14 + 可学习线性投影，将视觉特征映射到语言 token 嵌入空间
- 视频输入：CLIP 编码 + 时间嵌入 + Perceiver 模块聚合时空特征

**运动组合（Tokens-joint）**：
- 将前一轮运动 token $z_p^{1:L_p}$ 与当前轮运动 token $z_c^{1:L_c}$ 拼接后统一解码
- 解码器能够在 token 级别实现运动间的平滑过渡，优于帧级拼接

**多轮对话数据构建**：
- 利用 ChatGPT 生成运动推理数据（分析运动上下文、前后动作、角色等）
- 使用 TMR 文本-运动检索模型将运动按相似度分类，为中等相似度运动对生成编辑指令
- 将单轮任务与后续任务（翻译、推理、编辑等）组合为最多 10 轮的多轮对话数据

### 损失函数

语言模型训练目标为最大化答案 token 的对数似然：

$$\mathcal{L}_{LM} = -\sum_{i=0}^{L_t-1} \log p_\theta(x_a^i | X_v, X_{s,<i}, X_{a,<i})$$

三阶段训练策略：(1) 运动 tokenizer 预训练 → (2) 运动-语言预训练 → (3) 指令微调

## 实验关键数据

### 主实验

**运动推理对比（vs 大语言模型）**：

| 方法 | 参数量 | Bleu@1↑ | Bleu@4↑ | Rouge↑ | Cider↑ | BertScore↑ |
|------|--------|---------|---------|--------|--------|------------|
| Flan-T5-base | 250M | 4.64 | 1.78 | 15.32 | 15.93 | 3.45 |
| Llama-2-7b | 7B | 11.12 | 3.67 | 19.14 | 1.04 | 6.81 |
| Vicuna-1.5-7b | 7B | 19.27 | 7.39 | 25.75 | 5.44 | 19.05 |
| Vicuna-1.5-13b | 13B | 17.20 | 6.53 | 24.18 | 7.77 | 18.00 |
| **MotionChain** | **280M** | **37.92** | **19.19** | **38.05** | **24.53** | **32.24** |

**时序运动组合对比（BABEL 数据集）**：

| 方法 | Diversity | MPJPE↓ | PA-MPJPE↓ | ACCL↓ |
|------|-----------|--------|-----------|-------|
| Real | 15.74 | - | - | - |
| TEACH | 27.11 | 979.21 | 933.32 | 23.02 |
| **MotionChain** | **43.25** | **276.05** | **53.72** | **7.11** |

### 消融实验

**运动组合机制对比（HumanML3D）**：

| 组合方法 | MPJPE↓ | PA-MPJPE↓ | ACCL↓ | Diversity |
|----------|--------|-----------|-------|-----------|
| Independent（独立解码） | 350.79 | 102.97 | 11.40 | 6.47 |
| Past-condition（上一轮条件） | 232.46 | 46.15 | 6.18 | 6.01 |
| **Tokens-joint（token 拼接）** | **108.77** | **18.85** | **2.26** | 5.56 |

**视觉 Tokenizer 架构对比（BEDLAM）**：

| 架构 | First-frame MPJPE↓ | First-frame PA-MPJPE↓ | Last-frame MPJPE↓ | Last-frame PA-MPJPE↓ |
|------|-------------------|-----------------------|-------------------|---------------------|
| Q-former | 195.49 | 86.56 | 134.73 | 57.17 |
| Perceiver | 185.61 | 99.21 | 134.89 | 57.58 |
| **Linear** | **144.37** | **76.48** | **133.73** | **56.73** |

### 关键发现

1. 仅 280M 参数的 MotionChain 在运动推理任务上大幅超越 7B-13B 规模的 LLM，证明运动感知的必要性
2. Tokens-joint 组合方式相比独立解码，MPJPE 从 350.79 降至 108.77（降低 69%），证明 token 级运动组合的有效性
3. 简单的线性投影在视觉 tokenizer 中已足够理解人体姿态，无需更复杂的 Q-former 或 Perceiver
4. MotionChain 在时序运动组合中 PA-MPJPE 仅为 TEACH 的 5.8%（53.72 vs 933.32），展现出显著的运动质量提升

## 亮点与洞察

- **统一表示**：将运动、文本、图像统一编码为 token，通过语言模型架构实现多模态理解和生成
- **对话式控制**：首次实现多轮对话驱动的连续运动生成，每一轮的输出基于所有先前对话上下文
- **token 级运动组合**：通过拼接运动 token 并统一解码，实现了比帧级拼接更平滑自然的运动衔接
- **小模型大能力**：280M 参数模型超越 13B LLM，说明领域特定的运动感知比单纯增大模型规模更重要

## 局限性

- 使用不确定性生成模型，无法提供传统运动控制器的确定性控制
- 只能生成关节人体运动，不包括手部、面部等细粒度运动
- 缺乏碰撞信号，无法处理人-物和人-场景交互

## 评分

⭐⭐⭐⭐ (4/5) — 首次将多轮对话引入运动生成领域，统一多模态表示设计精巧，但受限于运动数据规模和交互建模能力

<!-- RELATED:START -->

## 相关论文

- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

<!-- RELATED:END -->
