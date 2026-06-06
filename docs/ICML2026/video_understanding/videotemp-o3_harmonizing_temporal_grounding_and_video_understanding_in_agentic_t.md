---
title: >-
  [论文解读] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking
description: >-
  [ICML 2026][视频理解][长视频理解] VideoTemp-o3 是统一的 Agent 视频理解框架——通过**冷启动 SFT 的统一掩码策略** + **可感知奖惩的 IoU 奖励**联合建模视频时间定位与问答，在长视频理解中实现高质量的多轮迭代定位与精准回答…
tags:
  - "ICML 2026"
  - "视频理解"
  - "长视频理解"
  - "时间定位"
  - "Agent 思维"
  - "多轮工具调用"
  - "可感知奖惩 RL"
---

# VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking

**会议**: ICML 2026  
**arXiv**: [2602.07801](https://arxiv.org/abs/2602.07801)  
**代码**: 待确认  
**领域**: 视频理解 / Agent / 多模态 VLM  
**关键词**: 长视频理解, 时间定位, Agent 思维, 多轮工具调用, 可感知奖惩 RL

## 一句话总结
VideoTemp-o3 是统一的 Agent 视频理解框架——通过**冷启动 SFT 的统一掩码策略** + **可感知奖惩的 IoU 奖励**联合建模视频时间定位与问答，在长视频理解中实现高质量的多轮迭代定位与精准回答，超长视频（> 20 分钟）mIoU 15.6% 超过 Gemini-2.5-Pro 的 14.8%。

## 研究背景与动机

**领域现状**：长视频理解中现有方法通常采用固定帧采样率均匀采样控制计算成本，但导致采样稀疏、容易漏掉问题相关的关键帧。最近出现的 Agent 思维视频范式（thinking-with-videos）借鉴 thinking-with-images 思想，采用"定位-裁剪-回答"流程让模型主动定位相关视频片段。

**现有痛点**：虽然 VideoExplorer / VITAL / REVISOR 等已探索该范式，但有三个关键问题——（1）**工作流复杂度高**：多个专用模型分别处理定位和问答，推理开销大；（2）**定位精度低**：难以精准定位，缺乏定位结果的评估和优化机制；（3）**流程死板**：固定的"一次裁剪后立即回答"模式，无法支持长视频中的迭代定位优化。

**核心矛盾**：最大障碍在于训练策略不足以学习精准定位和多轮迭代行为；现有标注数据质量低且长视频样本稀缺，模型缺乏高质量多轮轨迹来学 Agent 视频理解模式。

**本文目标**：构建统一框架在单模型中同时优化时间定位和视频问答，支持按需裁剪、多轮迭代优化，并设计专门的训练策略与数据构造方案。

**切入角度**：从数据、训练策略和模型设计三个维度——（1）高质量多轮数据的构造流程；（2）冷启动 SFT 的统一掩码策略鼓励探索同时过滤噪声；（3）可感知奖惩的 IoU 奖励防止奖励作弊。

**核心 idea**：用统一的多轮对话框架，通过精心设计的掩码监督和奖励机制，让单个模型学会在长视频中通过迭代工具调用实现精准定位和准确回答。

## 方法详解

### 整体框架
多轮交互的"定位-裁剪-回答"流程。给定视频-问题对 $(V, Q)$，模型先以低采样率 $s_0$ 快速浏览视频。随后迭代——每一轮生成推理文本 $T$ 和要么输出时间段 $P$ 要么输出最终答案 $A$。如果预测时间段 $P = [t_s, t_e]$，外部裁剪模块以更高采样率 $s_d > s_0$ 从原视频提取对应片段 $C = \text{Crop}(V, P, s_d)$ 附加到上下文供下一轮使用。交互在模型输出答案或达最大轮数时终止。每个训练样本 $i$ 可表示为多轮轨迹 $\tau_i = \{(V, Q); ([T_{i,1}, P_{i,1}, C_{i,1}], \ldots, [T_{i,t}, A_i])\}$。

### 关键设计

1. **统一掩码策略（Unified Masking Strategy）**:

    - 功能：在冷启动 SFT 阶段指导模型学多轮定位和问答，同时过滤含噪的早期定位标签。
    - 核心思路：对多轮数据，倒数第二轮包含正确时间段定位，最后一轮输出最终答案。早期轮次定位通常不精准。该方法**仅在最后两轮的模型输出上应用训练损失** $L$，对早期生成内容和用户输入掩码使其不影响梯度。
    - 设计动机：传统监督所有轮次导致模型学到大量错误定位；选择性监督只保留可靠信号，提高训练效率和鲁棒性。

2. **可感知奖惩的 IoU 奖励（Penalty-aware IoU Reward）**:

    - 功能：在 RL 阶段精确衡量定位质量，同时防止模型通过任意输出来"作弊"获取高奖励。
    - 核心思路：定义 IoU $R_{\text{IoU}} = \frac{|[t_s, t_e]| \cap |[t_s', t_e']|}{|[t_s, t_e]| \cup |[t_s', t_e']|}$，当 IoU 低于阈值 $\sigma$ 时施加惩罚 $\lambda$，即 $R_{\text{penalty-IoU}} = R_{\text{IoU}} - \lambda$（若 $R_{\text{IoU}} < \sigma$）或 $R_{\text{IoU}}$（若 $\geq \sigma$）。超参 $\lambda = 0.1, \sigma = 0.1$。
    - 设计动机：纯粹的 IoU 奖励容易被利用，模型可能输出任意时间段骗取奖励；引入阈值和惩罚项后，模型必须同时保证定位精度和合理性。

3. **高质量数据构造流程**:

    - 功能：构造大规模高质量长视频定位问答（GQA）数据，包括单轮无工具调用数据和多轮工具调用数据。
    - 核心思路：对单轮数据用 Qwen3-VL-235B 进行推理链生成和答案预测，仅保留预测答案与真值一致的样本；对多轮数据用 Gemini-2.5-Pro 生成候选定位，通过两阶段验证确保定位包含足够信息可独立回答问题以及多轮过程中答案一致性；验证失败的样本利用累积上下文重新定位，最多迭代一轮。
    - 设计动机：现有标注存在偏移、质量参差不齐，长视频样本稀缺；该流程通过严格的模型辅助标注和验证确保定位与答案高度对齐。

## 实验关键数据

### 主实验

| 方法 | MLVU | VideoMMMU | VideoMME（无字幕） | LVBench | 平均 |
|------|------|-----------|------------------|---------|------|
| Gemini-1.5-Pro | 49.3 | 53.3 | 59.0 | 33.1 | 48.7 |
| GPT-4o | 55.6 | 62.0 | 66.0 | 30.8 | 53.6 |
| Video-R1-7B | 48.0 | 46.0 | 67.3 | 40.1 | 50.4 |
| Qwen2.5-VL-7B | 45.2 | 36.1 | 57.6 | 39.2 | 44.5 |
| **VideoTemp-o3-7B-SFT** | 49.5 | 46.4 | 60.4 | 39.6 | 49.0 |
| **VideoTemp-o3-7B-RL** | **54.2** | **47.8** | **69.0** | **43.0** | **53.5** |

VideoTemp-o3-RL 在 MLVU / VideoMME / LVBench 上分别超最佳 baseline 6.2% / 1.7% / 2.9%，平均提升 3.1%。

### 消融实验

| ID | 方法变体 | VideoMMMU | VideoMME | LVBench | ReXTime mIoU | ReXTime Acc |
|----|---------|-----------|----------|---------|------------|----|
| (a) | 完整模型 | 53.2 | 64.5 | 43.0 | 29.5 | 74.4 |
| (b) | w/o 定位数据 | 52.5 | 63.0 | 42.0 | **13.0** | 73.3 |
| (c) | w/o 统一掩码 | 47.9 | 61.5 | 41.2 | 18.8 | 70.6 |
| (d) | w/o IoU 奖励 | 51.6 | 63.3 | 41.7 | 26.2 | 73.7 |
| (e) | w/o 可感知惩罚 | 44.2 | 63.7 | 40.7 | 23.8 | 73.6 |

### 长视频不同时长表现（VideoTemp-Bench）

| 方法 | 0-3 分 | 3-10 分 | 10-20 分 | > 20 分 | 平均 |
|------|--------|--------|---------|--------|------|
| Gemini-2.5-Pro | 39.1 | 46.1 | 36.1 | 14.8 | 34.0 |
| VideoChat-R1-7B | 25.2 | 6.7 | 4.7 | 1.8 | 9.6 |
| **VideoTemp-o3-RL** | **35.3** | **32.0** | **24.8** | **15.6** | **27.0** |

mIoU 指标，长视频时间定位基准。

### 关键发现
- 去掉定位数据后 mIoU 从 29.5 大幅跌至 13.0——定位监督对模型内部定位能力至关重要。
- 统一掩码移除导致 VideoMMMU 下跌 5.3%、mIoU 下跌 10.7%——验证选择性监督的有效性。
- 去掉可感知惩罚后模型性能崩溃（mIoU 29.5 → 23.8，VideoMME 64.5 → 63.7）——防止奖励作弊的必要性。
- 在超长视频（> 20 分钟）上 mIoU 15.6%（vs Gemini-2.5-Pro 14.8%）表现最稳定；相比 baseline 在 > 20 分钟上性能崩溃（mIoU < 2%），本框架展现优秀长视频泛化能力。

## 亮点与洞察
- **统一架构的巧妙性**：将时间定位和视频问答统一在同一模型中，通过共享表示空间和一致的多轮对话格式，让模型能同时优化两个任务——比多模块串联设计既降低推理延迟又通过任务间正交性提升性能。
- **选择性监督的有效性**：统一掩码策略只对最后两轮应用损失，对早期噪声定位遮蔽——巧妙平衡多轮轨迹学习的有效性和鲁棒性；处理多轮 Agent 数据的通用技巧可迁移到其他多步骤推理任务。
- **奖励设计的防护**：可感知奖惩 IoU 奖励通过显式惩罚项防止模型盲目猜测，避免 RL 中常见的奖励作弊问题；约束性设计值得在其他长视频任务（时间动作定位、事件检测）借鉴。
- **数据质量第一的实践**：通过严格的多阶段验证流程，确保 GQA 数据中定位与答案高度一致——相比直接使用低质标注，这个投资带来了显著的性能提升。

## 局限与展望
- 框架基于特定的多轮格式设计，对极长视频（> 60 min）的超多轮交互或复杂推理路径的泛化性未充分探索。
- 定位时间精度上限受视频帧率和裁剪阶段的离散性限制，难实现亚秒级定位。
- 数据构造流程依赖于高质量 VL 模型（Gemini-2.5）进行标注，迁移到其他领域或低资源语言时可行性待评估。
- 改进：探索连续化时间定位表示而非离散时间段；引入更灵活的轮次上限策略；设计轻量化数据标注方案降低对高端 VL 模型的依赖。

## 相关工作与启发
- **vs VideoExplorer**：VideoExplorer 采用多 Agent 协作（计划者 / 定位者 / 理解者分离），本文在单模型中统一集成降低推理复杂度；通过统一多轮对话格式和端到端训练获得更灵活的 iterative refinement 能力。
- **vs VITAL / REVISOR**：这些工作也采用 SFT-RL 两阶段训练，但缺对多轮噪声的显式处理和专门的防反演奖励设计；VideoTemp-o3 的统一掩码和可感知奖惩进一步稳定多轮学习。
- **vs LongVT**：LongVT 提出三阶段 SFT-RL-RFT 策略，本文通过更紧凑的 SFT-RL 框架和高质量数据构造达到相近或更优的性能——数据质量和训练策略设计的重要性可能高于单纯增加训练阶段数。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  统一框架 + 可感知奖惩 + 高质量数据构造流程的组合是首创；防反演奖励设计对 RL 社区有启发意义。
- 实验充分度: ⭐⭐⭐⭐⭐  覆盖长视频理解、时间定位、视频 GQA 三大任务，包括新基准 VideoTemp-Bench，消融详细，不同时长分析深入。
- 写作质量: ⭐⭐⭐⭐  方法清晰，数据构造流程图示直观；某些奖励设计的理论动机可展开更深入。
- 价值: ⭐⭐⭐⭐⭐  为长视频理解中的 Agent 范式建立高效可靠的范本；可感知奖励和统一掩码策略有明确复用价值；新基准为后续评估提供标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICML 2026\] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](../../CVPR2026/video_understanding/lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)

</div>

<!-- RELATED:END -->
