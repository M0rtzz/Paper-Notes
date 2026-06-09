---
title: >-
  [论文解读] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos
description: >-
  [CVPR 2025][图像生成][视频生成] VideoWorld 探索纯视觉视频生成模型能否从无标签视频中学习复杂知识（规则、推理、规划），提出潜在动态模型（LDM）压缩多步视觉变化，仅 3 亿参数即在围棋中达到职业五段水平。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频生成"
  - "知识学习"
  - "潜在动态模型"
  - "围棋AI"
  - "机器人控制"
---

# VideoWorld: Exploring Knowledge Learning from Unlabeled Videos

**会议**: CVPR 2025  
**arXiv**: [2501.09781](https://arxiv.org/abs/2501.09781)  
**代码**: [项目页面](https://VideoWorld.github.io/)  
**领域**: Image Generation / Video Generation & Understanding  
**关键词**: 视频生成, 知识学习, 潜在动态模型, 围棋AI, 机器人控制

## 一句话总结

VideoWorld 探索纯视觉视频生成模型能否从无标签视频中学习复杂知识（规则、推理、规划），提出潜在动态模型（LDM）压缩多步视觉变化，仅 3 亿参数即在围棋中达到职业五段水平。

## 研究背景与动机

大语言模型通过 next token prediction 展现了强大的知识和推理能力，但语言无法完全捕获现实世界的所有知识。自然界中生物主要通过视觉观察学习：
- 大猩猩等灵长类动物通过观察成年行为学习觅食和社交技能，不依赖语言
- 现有工作主要关注从文本或标签学习知识，从纯视觉信号学习的研究较少
- UniPi 等方法虽利用视频生成进行机器人操控，但仍严重依赖语言指令且仅限简单命令
- ChessGPT、Othello-GPT 等在棋类中探索推理，但使用的是状态标注数据而非原始视频
- 核心问题：**AI 能否仅通过视觉输入学习知识**，就像大猩猩从环境中学习那样？

关键发现：(1) 纯视频训练足以学习知识（规则、推理、规划）；(2) 视觉变化的表示方式对知识获取至关重要。

## 方法详解

### 整体框架

VideoWorld 由 VQ-VAE（使用 MAGVITv2 + FSQ 量化器）编码视频帧为离散 token，Llama 架构的自回归 Transformer 进行 next-token 预测。关键创新是引入潜在动态模型（LDM），将未来多步视觉变化压缩为紧凑的潜码，与视频 token 交替排列进行自回归预测。推理时通过逆动力学模型（IDM）将生成的帧和潜码映射为具体动作。

### 关键设计1：潜在动态模型（LDM）

**功能**：将多步视觉变化压缩为紧凑的潜在码，提高知识学习的效率和有效性。

**核心思路**：对每帧 $x_t$ 及后续 $H$ 帧 $x_{t+1:t+H}$，使用因果编码器-解码器提取视觉特征 $f_{t:t+H}$。定义可学习查询嵌入 $\{q^h\}_{h=1}^H$，通过注意力机制从 $f_{t:t+h}$ 中捕获变化信息，得到连续潜在表示 $\tilde{z}_t^h$，经 FSQ 量化后得到离散潜码 $z_t^h$。解码器利用 $f_t$ 和 $\{z_t^h\}_{h=1}^H$ 重建后续帧。训练目标为最小化生成帧与真实帧的 $\ell_2$ 距离。

**设计动机**：围棋中一步棋可用几个位置 token 编码，但视频中需要数百 token。LDM 将关键决策信息压缩为紧凑表示，同时编码前瞻规划信息。量化作为信息瓶颈，防止 LDM 学习到复制捷径。

### 关键设计2：自回归 Transformer 与 LDM 的无缝集成

**功能**：将视频帧 token 和 LDM 潜码统一在一个自回归序列中进行预测。

**核心思路**：视频解码器和 LDM 使用不同的码本，自回归 Transformer 的词汇表为两者的并集。每帧的离散 token 与对应的潜码 $\{z_t^h\}_{h=1}^H$ 组合成序列进行自回归训练。这使 Transformer 既能利用视觉编码器捕获的细粒度视觉细节，又能利用 LDM 生成的紧凑时间动态表示。

**设计动机**：将丰富的视觉信息与紧凑的变化表示结合，比仅使用视频或仅使用状态序列都能更有效地学习知识。

### 关键设计3：逆动力学模型（IDM）映射到任务操作

**功能**：将视频生成的结果转化为具体的任务动作（如围棋落子、机器人控制）。

**核心思路**：IDM $\pi$ 由几层 MLP 组成，独立于视频生成器训练，使用少量带动作标签的视频数据。基础框架中 $\pi(\cdot|x_t, \hat{x}_{t+1})$，引入 LDM 后扩展为 $\pi(\cdot|x_t, \hat{x}_{t+1}, \{\hat{z}_t^h\}_{h=1}^H)$，利用 LDM 编码的时间表示提升动作预测的时间一致性和准确性。

**设计动机**：将感知模型和行动模型分离，使视频生成器专注于学习知识表示，IDM 仅需少量标注数据即可完成动作映射。

### 损失函数

VQ-VAE 训练损失：标准重建损失 + FSQ 量化损失。LDM 训练损失：$\ell_2$ 像素重建损失。自回归 Transformer：交叉熵 next-token 预测损失。

## 实验关键数据

### 主实验：Video-GoBench 围棋评估

| Agent | Input | Legal Rate | Action-Value | Acc | Elo |
|-------|-------|-----------|--------------|-----|-----|
| **VideoWorld 300M** | Video | 99.7% | **83.7%** | **88.1%** | **2317** |
| VideoWorld 150M | Video | 99.7% | 82.0% | 86.7% | 2218 |
| Transformer 300M (Video) | Video | 99.6% | 59.7% | 58.9% | 1998 |
| Transformer 300M (State) | State | 99.8% | 79.7% | 87.2% | 2308 |
| KataGO-human-5d (RL) | State | 100% | 83.5% | 83.7% | 2253 |

### 消融实验：LDM 的贡献

| 设置 | Action-Value | Elo | 说明 |
|------|-------------|-----|------|
| VideoWorld (Video+LDM) | **83.7%** | **2317** | 完整模型 |
| 基础 Transformer (仅Video) | 59.7% | 1998 | 知识获取效率低 |
| 状态序列 Transformer | 79.7% | 2308 | 缺少视觉信息 |

### 关键发现

- VideoWorld 300M 达到**职业五段水平**（Elo 2317），超越 KataGO-human-5d（Elo 2253），无需搜索算法或强化学习
- LDM 使 Action-Value 从 59.7% 大幅提升至 83.7%，证明紧凑变化表示的关键性
- 在 CALVIN 机器人任务中，VideoWorld 的性能接近使用 ground truth 动作标签训练的 oracle 模型
- UMAP 可视化显示 LDM 潜码学习到了有意义的围棋模式和机器人运动方向

## 亮点与洞察

- **开创性探索**：首次系统验证视频生成模型可以从纯视觉数据学习复杂的推理和规划知识
- **小模型大能力**：3 亿参数模型在围棋中达到职业五段，暗示视觉学习可能比语言学习更高效
- **LDM 的通用性**：潜在动态模型在围棋（离散决策）和机器人（连续控制）两种截然不同的任务上都有效

## 局限与展望

- 围棋的视觉设计被刻意简化（去纹理），真实世界的视觉复杂性尚未充分处理
- IDM 仍需少量带动作标签的数据，未完全实现无监督学习
- 尚未在更复杂的真实世界任务（如自动驾驶）上验证
- 未来方向：更好的视觉表示、大规模预训练、多任务统一学习

## 相关工作与启发

- 与 Genie 和 LAPO 的潜在动作模型相比，VideoWorld 的 LDM 支持多步前瞻规划
- 与 ChessGPT 等使用标注状态数据的方法形成鲜明对比——纯视频学习也能达到甚至超越状态学习
- 为 World Model 研究提供了新的实验范式——直接评估视频生成模型的知识获取能力

## 评分

⭐⭐⭐⭐⭐ — 极具开创性的工作。首次严谨地证明视频生成模型可以从纯视觉输入学习复杂推理和规划知识，3 亿参数达到围棋职业五段水平令人印象深刻。LDM 的设计简洁而通用，对 World Model 和具身智能领域都有深远启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](../../ECCV2024/image_generation/ponymation_learning_articulated_3d_animal_motions_from_.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](../../ICCV2025/image_generation/ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[CVPR 2025\] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing](using_powerful_prior_knowledge_of_diffusion_model_in_deep_unfolding_networks_for.md)
- [\[CVPR 2025\] ReNeg: Learning Negative Embedding with Reward Guidance](reneg_learning_negative_embedding_with_reward_guidance.md)

</div>

<!-- RELATED:END -->
