---
title: >-
  [论文解读] LensWalk: Agentic Video Understanding by Planning How You See in Videos
description: >-
  [CVPR 2026][视频理解][视频智能体] 提出LensWalk，一个让LLM推理器主动控制视频观测范围和采样密度的智能体框架，通过reason-plan-observe循环实现自适应视频理解，无需微调即可在长视频基准上带来5%以上的即插即用性能提升。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "视频智能体"
  - "主动观测"
  - "视觉语言模型"
  - "长视频理解"
  - "工具调用"
---

# LensWalk: Agentic Video Understanding by Planning How You See in Videos

**会议**: CVPR 2026  
**arXiv**: [2603.24558](https://arxiv.org/abs/2603.24558)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频智能体, 主动观测, 视觉语言模型, 长视频理解, 工具调用

## 一句话总结

提出LensWalk，一个让LLM推理器主动控制视频观测范围和采样密度的智能体框架，通过reason-plan-observe循环实现自适应视频理解，无需微调即可在长视频基准上带来5%以上的即插即用性能提升。

## 研究背景与动机

视频理解是计算机视觉的核心任务，但视频的密集时序性质给自动分析带来巨大挑战。现有视频理解方法面临一个根本矛盾：推理与感知之间存在断裂。

现有方法主要存在三类问题：(1) 单次前向方法将视频均匀采样为固定视觉上下文，容易遗漏关键事件或被冗余信息淹没；(2) 启发式关键帧选择方法虽更精细，但仍是一次性静态采样，无法随中间假设变化而调整；(3) 基于检索的智能体虽可动态获取信息，但操作的是预处理过的静态表征（如ASR转录、clip级caption），无法从源视频中按需生成新的观测。

核心矛盾：模型的推理过程应该驱动它"看什么"和"怎么看"，但现有管线将观测和推理割裂——观测在推理之前一次性完成，或受限于固定的预处理工件。本文的切入角度是借鉴人类的视觉认知策略：人类通过有目的的信息搜寻来应对信息过载，不断在宏观扫视和精细聚焦之间切换，并在过程中持续反思和校验。核心idea：让LLM推理器自主决定观测的时间范围和采样密度，将视频理解转化为主动的推理-计划-观测循环。

## 方法详解

### 整体框架

LensWalk将视频理解建模为一个多轮迭代过程。每一轮中，LLM推理器($M_r$)分析当前问题和已积累的证据，制定包含观测工具选择、引导子问题、时间范围和采样密度的行动计划($a_t$）。该计划由VLM观察器($M_o$)执行，从视频中提取视觉证据。证据被追加到历史记录中，形成下一轮推理的输入。此外，系统维护时间戳锚点和全局实体记忆表来保证跨轮次的一致性。

### 关键设计

1. **多粒度观测工具套件**:
    - 功能：提供三种互补的观测工具，支持不同粒度的视频浏览
    - 核心思路：Scan Search在宽时间范围内并行扫描切片来发现线索；Segment Focus对单个时间段进行高密度采样提取细粒度细节；Stitched Verify将多个不连续时间段的帧组合成单一批次，支持跨段对比和因果验证
    - 设计动机：三个工具形成"发现-聚焦-验证"的完整认知链，覆盖从全局线索搜索到局部细节提取再到跨段整合验证的全部需求

2. **推理调度的主动观测机制**:
    - 功能：让推理器在每一步显式控制观测的时间范围($\mathcal{I}_t$)和采样策略
    - 核心思路：每个行动$a_t = (o_t, q_t, \mathcal{I}_t, \rho_{o_t})$包含工具选择、引导问题、时空范围和工具特定参数，实现推理状态到观测计划的端到端映射
    - 设计动机：将参数化的观测计划嵌入历史记录中，使智能体能追踪自身探索进度，为后续步骤针对未充分探索的区域提供依据

3. **证据锚定与实体记忆**:
    - 功能：确保长时多轮推理过程中的时间定位精度和实体一致性
    - 核心思路：时间戳锚点在VLM观察时插入帧间时间标记，使观察器返回带有精确时间引用的答案；全局实体记忆表在推理历史外独立维护，记录实体属性和出现时间
    - 设计动机：避免跨轮次中重复识别相同实体的开销，防止长历史上下文中实体引用混乱，同时为后续精确重新观测提供时间锚点

### 损失函数 / 训练策略

- LensWalk是一个无需训练的即插即用框架，不涉及模型微调
- 智能体被限制最多20次工具调用，每轮一次
- Scan Search、Segment Focus和Stitched Verify的每次调用帧预算分别为180、32和128
- 推理器同时作为实体记忆表的更新器

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(最佳配置) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| LVBench | Accuracy | 68.6% (o3自身) | 60.8% (MR.Video) | +7.8% |
| VideoMME Long | Accuracy(w/o sub) | 71.4% (o3自身) | 67.3% (DVD) | +4.1% |
| LongVideoBench | Accuracy | 70.6% (o3自身) | 68.6% (DVD) | +2.0% |
| MMVU (MC) | Accuracy | 80.9% (o3/GPT-4.1) | 78.9% (o3) | +2.0% |
| Video-MMMU | Overall | 78.33% (o3自身) | 75.44% (o3) | +2.89% |
| EgoSchema | Val | 77.2% (o3/Qwen2.5-VL-72B) | 76.6% (DVD) | +0.6% |

### 消融实验

| 配置 | 关键指标(VideoMME Long) | 说明 |
|------|---------|------|
| 完整LensWalk (o3/GPT-4.1) | 70.0% | 基线 |
| 去掉Scan Search | 65.4% | 下降4.6%，定位线索最关键 |
| 去掉Stitched Verify | 66.8% | 下降3.2%，跨时间段整合重要 |
| 去掉Segment Focus | 68.1% | 下降1.9%，细粒度提取有贡献 |
| 无Timestamp Anchor | 69.4% | 下降0.6% |
| 无Subject Memory | 69.7% | 下降0.3% |

### 关键发现

- o3作为自我观测者（推理器和观察器为同一模型）时表现极好，LVBench上提升11.5%，VideoMME Long上提升6.7%，作为"免费午餐"
- 开源推理器Qwen3-235B-A22B对弱观察器（Qwen2.5-VL-7B提升4.3%）有效，但对强观察器（GPT-4.1仅+0.1%）帮助有限
- 智能体展现出六种行为模式：直接查询、渐进缩放、范围分割、策略反思、整合验证和静态重复
- 框架自适应分配观测预算：简单问题用少量帧快速解决，复杂问题投入更多观测轮次

## 亮点与洞察

- 将"如何观测"纳入推理循环的核心设计理念非常优雅，类比人类有目的的视觉搜索策略
- 无需微调的即插即用特性使其可以直接提升现有模型，工程价值很高
- 涌现出的多样化认知策略（渐进缩放、策略反思等）展示了智能体的自主推理能力
- Token消耗与单次前向方法相当，同时大幅降低了每轮峰值Token数，缓解了长上下文的内存压力

## 局限与展望

- 框架效果高度依赖推理器的认知能力——弱推理器可能生成无效的观测计划
- 仍存在少量"静态重复"行为（反复观测相同区域），虽然比例低但表明规划机制仍不完美
- 当前的观测工具仅适用于视觉模态，未利用音频、字幕等多模态信息
- 最大20次工具调用的限制在极端长视频场景下可能不够

## 相关工作与启发

- **vs Deep Video Discovery**: DVD通过预生成整个视频的caption来支持推理，消耗百万级Token；LensWalk按需观测，Token消耗近似单次前向方法，同时精度更高
- **vs MR.Video**: MR.Video依赖预处理的clip检索，观测粒度和范围固定不变；LensWalk可以动态调整观测的时间范围和采样密度
- **vs VideoAgent**: VideoAgent的工具仅操作预处理产物；LensWalk直接从源视频中调度新的观测
- **启发**: "可扩展的视觉认知"理念——不仅要扩大模型规模，还要让模型学会主动观测

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将视频理解重新定义为主动观测调度问题，理念创新且实现优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 6个基准、多种模型组合、详细消融和行为分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，原理阐述清晰，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 即插即用框架，可直接提升现有强模型，实用性极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[ICML 2026\] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking](../../ICML2026/video_understanding/videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)

</div>

<!-- RELATED:END -->
