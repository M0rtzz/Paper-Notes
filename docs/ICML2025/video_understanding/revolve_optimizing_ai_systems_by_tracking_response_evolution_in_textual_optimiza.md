---
title: >-
  [论文解读] REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization
description: >-
  [ICML 2025][视频理解][文本优化] REVOLVE 通过跟踪 LLM 系统中响应在迭代过程中的"演化"趋势来指导优化，比 TextGrad 等基于即时反馈的方法更稳定高效，在提示优化、方案改进和代码优化上分别提升 7.8%、20.72% 和 29.17%。
tags:
  - ICML 2025
  - 视频理解
  - 文本优化
  - LLM系统优化
  - 响应演化
  - TextGrad
  - 提示工程
---

# REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization

**会议**: ICML 2025  
**arXiv**: [2412.03092](https://arxiv.org/abs/2412.03092)  
**代码**: 无  
**领域**: LLM Optimization  
**关键词**: 文本优化, LLM系统优化, 响应演化, TextGrad, 提示工程

## 一句话总结
REVOLVE 通过跟踪 LLM 系统中响应在迭代过程中的"演化"趋势来指导优化，比 TextGrad 等基于即时反馈的方法更稳定高效，在提示优化、方案改进和代码优化上分别提升 7.8%、20.72% 和 29.17%。

## 研究背景与动机
**领域现状**: LLM 系统 (包含 prompt + 工具交互) 在复杂任务中表现出色，但针对特定任务的优化仍依赖手工 prompt engineering 和超参调优。

**现有痛点**: TextGrad 等自动优化方法利用 LLM 的文本反馈作为"梯度"进行迭代改进，但仅关注即时反馈（类似数值梯度下降中只看当前导数），当调整过小或不稳定时容易停滞或震荡。

**核心矛盾**: 即时反馈 vs 趋势感知——仅看当前步的改进方向不足以做出全局最优的调整决策。

**本文切入**: 类比数值优化中使用动量/加速梯度等历史信息，引入跨迭代的响应演化历史来指导优化。

**核心 idea**: 在每步优化时，不仅提供当前反馈，还提供响应如何随时间演化的信息，使 LLM 能做出更有远见的渐进式调整。

## 方法详解

### 整体框架
输入：待优化的 LLM 系统 (prompt / solution / code) → 多次迭代 → 每步记录响应 + 评估 → 汇总演化趋势 → LLM 根据演化历史生成改进建议 → 更新系统 → 直到收敛。

### 关键设计

1. **响应演化追踪**:

    - 记录每次迭代的完整响应和评分
    - 构建演化轨迹：$(r_1, s_1) \to (r_2, s_2) \to ... \to (r_t, s_t)$
    - 向 optimizer LLM 提供不仅是当前改进方向，还有完整的演化趋势
    - 设计动机：类比动量法，利用历史信息避免短视优化

2. **自适应调整策略**:

    - 当改进变缓：加大调整力度（类比加速梯度）
    - 当出现震荡：稳定调整方向
    - 当持续改进：保持当前方向
    - 设计动机：针对不同优化阶段采用不同策略

3. **通用框架应用**:

    - Prompt 优化：改进 system prompt 以提升任务准确率
    - Solution 改进：迭代优化自然语言或结构化方案
    - Code 优化：逐步修复和改进代码实现
    - 设计动机：统一的文本优化范式适用于多种 LLM 系统场景

### 损失函数 / 训练策略
- 无传统意义上的损失函数，而是用任务 metric 作为评估信号
- LLM 作为 optimizer，根据演化历史生成文本形式的"梯度"

## 实验关键数据

### 主实验

| 任务类型 | 指标 | REVOLVE | TextGrad | 提升 |
|----------|------|---------|----------|------|
| Prompt 优化 | 准确率 | 最优 | 基线 | +7.8% |
| Solution 改进 | 质量分 | 最优 | 基线 | +20.72% |
| Code 优化 | 通过率 | 最优 | 基线 | +29.17% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 REVOLVE | 最优 | 演化追踪 + 自适应调整 |
| 无演化追踪 | 下降 | 退化为类似 TextGrad |
| 仅看最近2步 | 部分下降 | 历史太短难以判断趋势 |
| 固定调整策略 | 下降 | 缺乏自适应能力 |

### 关键发现
- REVOLVE 在更少迭代次数内收敛，节省计算成本
- 演化趋势信息对避免优化停滞和震荡至关重要
- 在代码优化任务上提升最为显著 (+29.17%)，因为代码修改需要更稳定的方向

## 亮点与洞察
- **跨学科类比**：将数值优化中的动量/加速梯度思想迁移到文本优化
- **简洁有效**：不需要修改底层 LLM，只修改 optimizer prompt 的信息组织方式
- **通用性强**：同一框架适用于 prompt/solution/code 多种优化场景

## 局限与展望
- 历史信息增长可能超出 LLM 上下文窗口
- 评估信号的质量直接影响优化效果
- 需要更多 LLM 调用（记录和分析演化历史）

## 相关工作与启发
- TextGrad (Yuksekgonul et al. 2024) 是直接前驱
- OPRO (Yang et al. 2024) 用 LLM 优化 prompt
- 启发：经典优化理论中的许多策略可以"翻译"到 LLM 文本优化中

## 评分
- 新颖性: ⭐⭐⭐⭐ 将优化理论洞察迁移到文本优化，思路清晰
- 实验充分度: ⭐⭐⭐⭐ 三个场景全面验证
- 写作质量: ⭐⭐⭐⭐ 类比传统优化的叙述方式易于理解
- 价值: ⭐⭐⭐⭐ 实用性强的 LLM 系统优化工具

<!-- RELATED:START -->

## 相关论文

- [Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](../../ICLR2026/video_understanding/paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)
- [egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks](../../ICCV2025/video_understanding/egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)
- [Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks](../../CVPR2025/video_understanding/unbiasing_through_textual_descriptions_mitigating_representation_bias_in_video_b.md)
- [DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](../../NeurIPS2025/video_understanding/dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)
- [Unifying Specialized Visual Encoders for Video Language Models](unifying_specialized_visual_encoders_for_video_language_models.md)

<!-- RELATED:END -->
