---
title: >-
  [论文解读] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction
description: >-
  [ICCV 2025][图像生成][无限游戏生成] 提出 AnimeGamer，基于多模态大语言模型(MLLM)的无限动漫生活模拟系统，通过动作感知的多模态表征预测下一轮游戏状态（动态动画镜头 + 角色状态更新），实现持续一致的交互式动漫游戏体验。 近年来生成式 AI 在动漫制作中取得了显著进展，但现有方法存在明显局限： 有…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "无限游戏生成"
  - "动漫生活模拟"
  - "多模态大语言模型"
  - "视频扩散模型"
  - "游戏状态预测"
---

# AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction

**会议**: ICCV 2025  
**arXiv**: [2504.01014](https://arxiv.org/abs/2504.01014)  
**代码**: [https://github.com/TencentARC/AnimeGamer](https://github.com/TencentARC/AnimeGamer)  
**领域**: 图像/视频生成  
**关键词**: 无限游戏生成, 动漫生活模拟, 多模态大语言模型, 视频扩散模型, 游戏状态预测

## 一句话总结

提出 AnimeGamer，基于多模态大语言模型(MLLM)的无限动漫生活模拟系统，通过动作感知的多模态表征预测下一轮游戏状态（动态动画镜头 + 角色状态更新），实现持续一致的交互式动漫游戏体验。

## 研究背景与动机

近年来生成式 AI 在动漫制作中取得了显著进展，但现有方法存在明显局限：

**有限游戏 vs 无限游戏**：现有游戏生成方法（如 GameNGen 模拟 DOOM）局限于预定义环境和有限指令，属于"有限游戏"。而理想的动漫生活模拟应属于"无限游戏"——无预设边界、开放式语言交互、持续演化的故事线

**前驱工作 Unbounded 的不足**：
   - 仅使用 LLM 处理纯文本对话，忽略历史视觉上下文，导致游戏连贯性差
   - 只能生成静态图像，无法呈现角色动作必需的动态效果

**核心挑战**：如何在多轮交互中保持角色一致性、上下文连贯性，同时生成高质量的动态视频？

## 方法详解

### 整体框架

AnimeGamer 的训练分为三个阶段：(a) 学习动画镜头的 token 化和解码；(b) 训练 MLLM 预测下一个游戏状态；(c) 解码器适配训练。推理时采用滑动窗口实现理论上无限长的游戏生成。

### 关键设计

1. **动作感知的多模态表征 (Action-aware Multimodal Representation)**

    - 将动画镜头分解为三个组成部分：
        - **视觉参考** $f_v$：通过 CLIP 提取动画片段首帧的嵌入，捕获整体外观
        - **动作描述** $f_{md}$：使用 T5 编码的短文本动作提示（如"Softly talk"）
        - **运动幅度** $f_{ms}$：通过光流检测表示角色动作强度
    - 编码器 $\mathcal{E}_a$ 通过 MLP + LayerNorm + Concat 融合视觉和文本特征：
    $s_a = \mathcal{E}_a(f_{md}, f_v) = \text{Concat}(\text{LN}(\text{MLP}(x)), \text{LN}(\text{MLP}(y)))$
    - 设计动机：现有方法仅预测文本或图像表征，无法充分保留视频的视觉和运动信息

2. **动画镜头解码器 $\mathcal{D}_a$**

    - 基于视频扩散模型 CogVideoX，将原始文本特征替换为动作感知多模态表征
    - 运动幅度 $f_{ms}$ 通过正弦函数嵌入 + FC + SiLU 激活注入时间步嵌入 $f_t$
    - 训练分两步：先固定解码器只训练编码器（warm-up），再联合训练
    - 训练目标为标准扩散损失：
    $\mathcal{L} = \mathbb{E}_{z,c,s_a,\epsilon \sim \mathcal{N}(0,1),t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c, s_a)\|_2^2\right]$

3. **基于 MLLM 的游戏状态预测**

    - 以 Mistral-7B 初始化 MLLM，作为"游戏引擎"
    - 输入：历史多模态上下文 + 当前玩家指令
    - 输出：$N=226$ 个动作感知多模态表征（$s_a$）+ 角色状态 $s_c$（体力/社交/娱乐值）+ 运动幅度
    - $s_a$ 用 MSE 损失，$s_c$ 和 $f_{ms}$ 用交叉熵损失：
    $\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{MSE}$

4. **解码器适配 (Decoder Adaptation)**

    - MLLM 和解码器分别训练可能导致隐空间不对齐
    - 冻结 MLLM，仅微调解码器，使其适应 MLLM 的输出嵌入
    - 推理时使用滑动窗口 + train-short-test-long 策略支持无限生成

### 数据构建

从 10 部热门动漫电影中提取约 20,000 个视频片段（16帧，480×720）。使用 InternVL 自动标注角色运动、背景和角色状态，支持玩家自定义角色。

## 实验关键数据

### 主实验

| 模型 | CLIP-I↑ | DreamSim↑ | CLIP-T↑ | ACC-F↑ | MAE-F↓ | 推理时间(s/轮)↓ |
|------|---------|-----------|---------|--------|--------|----------------|
| GSC | 0.786 | 0.502 | 0.333 | 0.316 | 0.826 | 50 |
| GFC | 0.766 | 0.580 | 0.333 | 0.292 | 1.021 | 63 |
| GC | 0.796 | 0.642 | 0.334 | 0.425 | 0.722 | 25 |
| **AnimeGamer** | **0.813** | **0.740** | **0.416** | **0.674** | **0.424** | **24** |

GPT-4V 和人类评估（10分制）：

| 模型 | 整体(GPT/人) | 指令跟随(GPT/人) | 上下文一致(GPT/人) | 角色一致(GPT/人) |
|------|-------------|-----------------|-------------------|-----------------|
| GC | 6.42/7.38 | 7.29/7.37 | 6.58/6.89 | 7.49/7.55 |
| **AnimeGamer** | **8.36/10.0** | **9.14/9.95** | **8.41/9.95** | **9.11/9.86** |

### 消融实验

| 配置 | CLIP-I↑ | DreamSim↑ | ACC-F↑ | MAE-F↓ |
|------|---------|-----------|--------|--------|
| w/ random frame | 0.845 | 0.450 | 0.474 | 0.562 |
| w/o warm-up | 0.831 | 0.511 | 0.703 | 0.458 |
| w/o $f_{ms}$ | 0.853 | 0.689 | 0.182 | 1.219 |
| w/o adapt | 0.683 | 0.494 | 0.365 | 0.847 |
| **Ours (full)** | **0.867** | **0.793** | **0.729** | **0.403** |

### 关键发现

- AnimeGamer 在所有自动指标上全面超越基线，特别是 DreamSim (+15.4%) 和 CLIP-T (+24.6%)，表明多模态上下文对一致性至关重要
- 人类评估中几乎满分（10/10），远超仅用文本上下文的方法
- 去除运动幅度控制后 ACC-F 暴跌至 0.182，证明仅靠文本无法可靠控制运动幅度
- 解码器适配是必要的——不使用时 CLIP-I 从 0.786 降至 0.683
- 推理效率最优（24s/轮），因为无需额外 LLM API 调用

## 亮点与洞察

- **"MLLM as Game Engine"**：将 MLLM 用作游戏引擎直接预测游戏状态，而非仅作为文本路由器，这是一个创新的范式
- **动作感知的多模态表征设计精妙**：通过分离视觉参考、动作描述和运动幅度，在保持可控性的同时实现高质量视频生成
- **端到端设计**：从数据收集到模型训练到评估基准，提供了完整的技术栈

## 局限与展望

- 仅在闭域（自定义角色）上训练和评估，未探索开放域泛化
- 训练数据来自 10 部动漫电影，规模和多样性有限
- 每轮仅生成 16 帧的短视频片段
- 角色状态（体力/社交/娱乐）设计较为简单，未涵盖更复杂的游戏机制
- 未与更多游戏生成基线（如 GameNGen、DIAMOND）比较

## 相关工作与启发

- 延续 Unbounded 的无限游戏概念但显著改进，从静态图像升级到动态视频
- 动作感知表征的设计思路可推广到其他多模态视频控制生成任务
- 自动化数据收集管道使得任何动漫 IP 都可快速适配

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个基于MLLM的无限动漫生活模拟，问题和方法都很新
- 实验充分度: ⭐⭐⭐⭐ 自动+人类评估充分，消融全面，但缺少更多基线
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 商业化前景广，但实际游戏体验还需大幅提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[ICCV 2025\] Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](../../ICML2025/image_generation/generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[NeurIPS 2025\] State-Covering Trajectory Stitching for Diffusion Planners](../../NeurIPS2025/image_generation/state-covering_trajectory_stitching_for_diffusion_planners.md)

</div>

<!-- RELATED:END -->
