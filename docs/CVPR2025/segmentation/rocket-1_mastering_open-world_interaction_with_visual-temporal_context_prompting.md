---
title: >-
  [论文解读] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting
description: >-
  [CVPR 2025][语义分割][视觉时序上下文提示] ROCKET-1 提出了一种新的视觉时序上下文提示（visual-temporal context prompting）通信协议，通过在历史视觉观测上标注物体分割来引导策略模型与环境交互，训练了一个基于分割条件的低层策略，结合 GPT-4o、Molmo、SAM-2 构建层级智能体，在 Minecraft 中实现了 76% 的开放世界交互性能绝对提升。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "视觉时序上下文提示"
  - "开放世界交互"
  - "分割引导策略"
  - "Minecraft"
  - "层级智能体"
---

# ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting

**会议**: CVPR 2025  
**arXiv**: [2410.17856](https://arxiv.org/abs/2410.17856)  
**代码**: [https://github.com/craftjarvis/ROCKET-1](https://github.com/craftjarvis/ROCKET-1)  
**领域**: 图像分割  
**关键词**: 视觉时序上下文提示, 开放世界交互, 分割引导策略, Minecraft, 层级智能体

## 一句话总结

ROCKET-1 提出了一种新的视觉时序上下文提示（visual-temporal context prompting）通信协议，通过在历史视觉观测上标注物体分割来引导策略模型与环境交互，训练了一个基于分割条件的低层策略，结合 GPT-4o、Molmo、SAM-2 构建层级智能体，在 Minecraft 中实现了 76% 的开放世界交互性能绝对提升。

## 研究背景与动机

1. **领域现状**：将视觉语言模型（VLM）应用于具身决策是当前热门方向。主流方案分为端到端（如 RT-2、OpenVLA 直接输出动作）和层级式（VLM 做高层推理，低层策略执行）两条路线。层级式方法中，高层推理器和低层策略之间的"通信协议"决定了系统的能力上限。
2. **现有痛点**：(1) 端到端方法需要大量带动作标注的轨迹数据，且引入动作模态可能损害 VLM 的基础能力；(2) 语言作为通信协议无法有效传达空间信息——当画面中出现多个同名物体时，很难用语言精确指定其中一个；(3) 未来图像方案（如 MineDreamer）需要构建世界模型，面临幻觉、时序不一致和范围有限的问题。
3. **核心矛盾**：语言不够精确（缺乏空间信息），图像不够可靠（需要预测未来），现有通信协议都无法充分释放 VLM 的空间理解能力。
4. **本文目标**：设计一种新的通信协议，能够精确传达空间交互信息，同时利用视觉时序上下文来处理部分可观测环境。
5. **切入角度**：观察到人类在执行任务时（如抓取物体），不会预想拿着物体的画面，而是持续关注目标物体并在其被遮挡时利用记忆回忆其位置。这种"视觉时序上下文"的利用方式是关键。
6. **核心 idea**：用物体分割掩码高亮过去观测中的感兴趣区域，配合交互类型信息，作为高层推理器和低层策略之间的通信协议。

## 方法详解

### 整体框架

层级智能体由四个组件组成：(1) GPT-4o 做任务分解和推理，输出交互步骤描述；(2) Molmo 72B 根据描述在当前观测中定位目标物体，输出 (x,y) 坐标；(3) SAM-2 根据坐标生成分割掩码并在后续帧中跟踪物体；(4) ROCKET-1 接收观测+分割+交互类型，预测动作。GPT-4o 和 Molmo 低频运行，SAM-2 和 ROCKET-1 与环境同频运行。

### 关键设计

1. **视觉时序上下文提示（Visual-Temporal Context Prompting）**:
    - 功能：在高层推理器和低层策略之间建立精确的空间通信协议
    - 核心思路：推理器在过去的视觉观测上应用物体分割来高亮感兴趣区域，同时通过一组交互类型原语（use、approach、switch、mine block 等）传达交互意图。不同颜色的分割代表不同交互类型。策略模型接收观测序列 $o_{1:t}$、分割序列 $m_{1:t}$ 和交互类型序列 $c_{1:t}$，因果地预测动作 $a_t$。
    - 设计动机：分割掩码比语言更精确地传达"在哪里"的信息（解决多个同名物体的歧义），比预测未来图像更可靠（不需要世界模型），并且通过时序上下文可以处理部分可观测场景（物体被遮挡后仍能利用历史分割推断位置）。

2. **ROCKET-1 策略架构**:
    - 功能：将观测和分割信息融合，预测低层动作
    - 核心思路：将观测 $o_t \in \mathbb{R}^{3 \times H \times W}$ 和分割掩码 $m_t \in \{0,1\}^{1 \times H \times W}$ 拼接为 4 通道图像，送入 EfficientNet-B0 视觉骨干进行深度融合，再通过自注意力池化生成特征向量 $x_t$。交互类型 $c_t$ 通过嵌入层编码，在 TransformerXL 中与视觉特征序列一起进行时序建模：$\hat{a}_t \leftarrow \text{TransformerXL}(c_1, x_1, \cdots, c_t, x_t)$。关键细节是将交互类型信息延迟到 backbone 之后再融合，使 backbone 可以跨交互类型共享知识。训练时以概率 $p=0.75$ 随机丢弃分割和交互类型，强迫模型学习从时序上下文中推断意图。
    - 设计动机：4 通道输入（模仿 ControlNet 的做法）实现了空间信息的深度融合。延迟交互类型融合缓解了交互类型分布不平衡的问题。高概率随机丢弃分割确保模型不会过度依赖当前帧的提示，培养其时序推理能力。

3. **反向轨迹标注（Backward Trajectory Relabeling）**:
    - 功能：自动化生成训练数据中的分割标注
    - 核心思路：利用 OpenAI 的承包商数据（16 亿帧人类 Minecraft 游戏数据），首先检测交互事件帧（来自元数据记录的 kill entity、mine block、use item 等事件），在交互前一帧用固定位置的边界框和点（物体通常在画面中心）作为 SAM-2 的提示生成分割，然后 SAM-2 在时间上反向传播，为之前 $k$ 帧自动生成分割标注。额外引入 navigate 交互类型：当玩家移动超过阈值时视为接近物体。
    - 设计动机：收集带文本标注的轨迹数据成本高昂。利用 SAM-2 的反向跟踪能力，可以从交互时刻自动反推之前帧中目标物体的位置，实现全自动数据标注。Minecraft 中交互物体通常在画面中心的观察简化了初始分割步骤。

### 损失函数 / 训练策略

行为克隆损失：$\mathcal{L} = -\sum_{t=1}^{|\tau|} \log \pi(a_t | o_{1:t}, m_{1:t} \odot w_{1:t}, c_{1:t} \odot w_{1:t})$

其中 $w_t \sim \text{Bernoulli}(1-p)$，$p=0.75$ 为丢弃概率。每条完整轨迹切分为 128 帧的段。使用 AdamW 优化器，学习率 $4 \times 10^{-5}$。

## 实验关键数据

### 主实验

| 方法 | Prompt | Hunt | Mine | Interact | Navigate | Tool | Place | **平均** |
|------|--------|------|------|----------|----------|------|-------|---------|
| VPT-bc | N/A | 0.15 | 0.00 | 0.16 | 0.05 | 0.00 | 0.00 | 0.07 |
| STEVE-1 | Human | 0.03 | 0.36 | 0.02 | 0.16 | 0.49 | 0.08 | 0.19 |
| GROOT-1 | Human | 0.16 | 0.03 | 0.05 | 0.02 | 0.30 | 0.02 | 0.09 |
| ROCKET-1 | Molmo | 0.88 | 0.77 | 0.66 | 0.88 | 0.93 | 0.82 | **0.82** |
| ROCKET-1 | Human | 0.93 | 0.93 | 0.93 | 0.97 | 0.97 | 0.96 | **0.95** |

长程任务对比：

| 方法 | 通信协议 | Wooden Pickaxe | Furnace | Shears | Diamond | Steak | Obsidian | Pink Wool |
|------|---------|---------------|---------|--------|---------|-------|----------|-----------|
| DEPS | 语言 | 0.95 | 0.75 | 0.15 | 0.02 | 0.15 | 0.00 | 0.00 |
| OmniJarvis | 隐码 | 0.95 | 0.90 | 0.20 | 0.08 | 0.40 | 0.00 | 0.00 |
| **Ours** | 视觉时序 | **1.00** | **1.00** | **0.45** | **0.25** | **0.75** | **0.50** | **0.70** |

### 消融实验

| 配置 | Hunt (↑) | Mine (↑) | 说明 |
|------|---------|---------|------|
| 交互类型在 Transformer 融合 | **0.91** | **0.78** | 延迟融合，backbone 共享知识 |
| 交互类型在 backbone 融合 | 0.72 | 0.69 | 过早融合，受分布不平衡影响 |
| w/o SAM-2, #Pmt=3 | 0.84 | 0.82 | 高频 Molmo 提示，推理极慢 (0.9 FPS) |
| w/o SAM-2, #Pmt=30 | 0.00 | 0.03 | 低频提示且无跟踪，完全失败 |
| +sam2_tiny, #Pmt=30 | 0.84 | 0.69 | SAM-2 跟踪补充低频提示 |
| +sam2_large, #Pmt=30 | **0.91** | 0.78 | 最大 SAM-2 模型效果最好 |

### 关键发现

- ROCKET-1 + Molmo 的 82% 平均成功率比最好的基线 STEVE-1（19%）高出 **63 个百分点**
- 在需要精确空间交互的任务（Place）上，之前没有任何方法能成功（0%），而 ROCKET-1 达到 82-96%
- 长程任务中，ROCKET-1 是唯一能完成 Obsidian（50%）和 Pink Wool（70%）的方法，其他方法全部失败
- SAM-2 的插件式集成至关重要：无 SAM-2 时低频提示下完全失败，有 SAM-2 后恢复到近最优水平
- 交互类型的融合位置很关键：在 Transformer 层融合比在 backbone 融合高出约 10-19 个百分点

## 亮点与洞察

- **通信协议的创新**：跳出了"语言 vs 图像"的二分法，提出用分割掩码作为空间通信方式。这种方式精确、高效，且天然与物体跟踪模型兼容。这一设计范式可以迁移到机器人操作等其他具身 AI 领域。
- **反向轨迹标注的巧妙**：利用 SAM-2 的反向跟踪能力从交互时刻自动回推标注，完全免去了人工标注成本。这一数据生成pipeline可以应用于任何需要物体级标注的视频数据。
- **组件化的系统设计**：GPT-4o（推理）+ Molmo（定位）+ SAM-2（跟踪）+ ROCKET-1（执行）的四组件设计，每个组件可以独立升级。随着更强的 VLM 和分割模型出现，整个系统可以即时受益。

## 局限与展望

- ROCKET-1 无法与视野外或从未遇到的物体交互，需要推理器频繁引导探索，增加了计算开销
- 依赖 Molmo 的定位精度——如果 Molmo 定位错误，整个交互链会失败
- 目前仅在 Minecraft 中验证，真实世界的复杂性（连续动作空间、物理约束）尚未涉及
- 作者在后续工作 ROCKET-2 中解决了部分局限
- 未来方向：扩展到真实机器人场景、处理多物体同时交互、增加自主探索能力

## 相关工作与启发

- **vs STEVE-1**: 语言条件策略，在需要空间精度的任务上表现差（Place 类任务 0%）。ROCKET-1 通过分割条件完美解决了空间歧义。
- **vs MineDreamer**: 用 VLM+扩散模型生成未来目标图像来驱动策略，但生成图像可能幻觉、不一致。ROCKET-1 不需要预测未来，直接在当前观测上标注目标。
- **vs OmniJarvis**: 用隐码通信，但隐码的可解释性差。ROCKET-1 的分割掩码是可视化的、可理解的通信方式。
- 与 CLIPort 有相似之处（用热图指导交互），但 CLIPort 仅限完全可观测的 pick-and-place 任务。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视觉时序上下文提示是全新的通信协议范式，反向轨迹标注方法也很巧妙
- 实验充分度: ⭐⭐⭐⭐ 设计了专门的交互基准、长程任务测试、详细消融，但仅在 Minecraft 中验证
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，系统设计图示直观
- 价值: ⭐⭐⭐⭐⭐ 76% 的绝对性能提升极具说服力，通信协议设计有望影响整个具身 AI 社区

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] V-CLR: View-Consistent Learning for Open-World Instance Segmentation](v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)
- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](../../ICCV2025/segmentation/open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
