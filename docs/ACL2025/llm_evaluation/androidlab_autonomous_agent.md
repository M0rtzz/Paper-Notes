---
title: >-
  [论文解读] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents
description: >-
  [LLM评测] 提出AndroidLab——一个系统性的Android智能体评测与训练框架，包含统一的操作环境、138个任务的可复现基准测试和94.3K步骤的指令数据集，通过微调将开源LLM成功率从4.59%提升至21.50%。
tags:
  - "LLM评测"
---

# AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents

## 论文信息

- **会议**: ACL 2025
- **arXiv**: [2410.24024](https://arxiv.org/abs/2410.24024)
- **代码**: [https://github.com/THUDM/Android-Lab](https://github.com/THUDM/Android-Lab)
- **领域**: LLM评测
- **关键词**: Android Agent, 移动端智能体, 基准测试, 指令微调, 多模态模型

## 一句话总结

提出AndroidLab——一个系统性的Android智能体评测与训练框架，包含统一的操作环境、138个任务的可复现基准测试和94.3K步骤的指令数据集，通过微调将开源LLM成功率从4.59%提升至21.50%。

## 研究背景与动机

- **领域现状**: 使用LLM/LMM作为移动端自主智能体的研究日益增多，但现有训练和评估工作缺乏系统性——几乎所有基准仅测试闭源模型的提示工程效果。
- **已有基准局限**: 静态基准（PixelHelp、AITW）无法交互；动态基准（AndroidEnv、AndroidWorld）可复现性差或不支持构建微调训练数据；且无统一框架同时比较开源与闭源、文本与多模态模型。
- **关键需求**: 需要一个统一的框架，同时支持(1)公平比较不同模态模型，(2)可复现评估，(3)高效构建训练数据以提升开源模型。
- **核心动机**: 通过提示工程提升的效果有限（复杂推理框架仅带来边际改进但推理时间大幅增加），微调小规模开源模型才能缩小与闭源模型的差距。

## 方法详解

### 整体框架

AndroidLab包含三个组件：(1) 标准化操作环境，统一LLM和LMM的动作空间；(2) 可复现基准测试，基于Android虚拟设备(AVD)预设9个应用的138个任务；(3) Android Instruct训练数据集，包含10.5K操作轨迹和94.3K步骤。

### 关键设计

1. **统一双模态操作模式**: XML模式（文本输入LLM）和SoM模式（屏幕截图+标注的多模态LMM），确保两种模式共享完全相同的动作空间和操作对象，实现公平比较
2. **子目标分解评估**: 将每个任务拆分为多个子目标，通过UI树结构匹配独立验证每个子目标的完成状态，避免传统操作路径匹配的局限性
3. **可复现离线环境**: 所有应用预装在AVD镜像中离线运行，固定时间和地理位置，消除外部网络和时间依赖

### 动作空间

6种基本操作（Tap、Swipe、Type、Long Press、Home、Back）+ 1种终止操作（Finish），支持返回执行结果。

### 数据构建流程

任务派生扩展 → LLM/LMM自探索（自动生成轨迹） → 人工标注（4步流程：可行性检查→熟悉界面→执行记录→交叉验证）

## 实验

### 主实验：XML模式和SoM模式成功率对比

| 模式 | 模型 | 成功率(SR) | 子目标SR | 反向冗余比 | 合理操作比 |
|------|------|-----------|---------|-----------|----------|
| XML | GPT-4-1106-Preview | **31.16%** | 38.21% | 66.34 | 86.24 |
| XML | GPT-4o | 25.36% | 30.56% | **107.45** | 86.56 |
| XML | Qwen2-7B (原始) | 4.35% | 4.95% | - | 67.26 |
| XML+SFT | LLaMA3.1-8B-ft | **23.91%** | 30.31% | 75.58 | **92.46** |
| SoM | GPT-4o | **31.16%** | 35.02% | 87.32 | 85.36 |
| SoM | Claude-3.5-Sonnet | 28.99% | 32.66% | **113.41** | 81.16 |
| SoM | CogVLM2 (原始) | 0.72% | 0.72% | - | 17.97 |
| SoM+SFT | Qwen2-VL-7B-ft | **18.12%** | 22.64% | 65.23 | 88.29 |

### 消融分析：不同推理框架对成功率的影响

| 模式 | 模型 | 基础 | +ReAct | +SeeAct |
|------|------|------|--------|---------|
| XML | GPT-4o | 25.36% | **33.33%** | 24.64% |
| XML | Gemini-1.5-Pro | 18.84% | **31.16%** | 21.01% |
| SoM | GPT-4o | 31.16% | **31.88%** | 30.43% |

### 关键发现

1. **微调效果显著**: LLM成功率从4.59%→21.50%（+368%），LMM从1.93%→13.28%（+588%），微调后开源模型接近甚至部分超过闭源模型
2. **ReAct框架仅在XML模式显著有效**: XML+ReAct将GPT-4o的SR从25.36%提升到33.33%，但SoM模式下提升微弱
3. **效率与质量的权衡**: 微调模型平均仅生成4.96 tokens/步，而ReAct需23.56、SeeAct需129.12 tokens/步
4. **操作效率大幅提升**: 微调后合理操作比(ROR)普遍超过88%，远高于微调前的17-67%
5. **屏幕尺寸影响性能**: 常规手机尺寸（Pixel 7/8 Pro）表现最优，过小/过大屏幕都会降低性能
6. **最佳闭源表现仅31.16%**: AndroidLab基准具有足够挑战性，最强模型也未过半

## 亮点

- 首个统一LLM和LMM评估的Android智能体框架，动作空间完全对齐
- 子目标分解评估机制比操作路径匹配更精确且灵活
- 开源训练数据集有效缩小了开源/闭源模型差距，验证了微调路线的可行性
- 完整的数据构建工具链（在线标注工具 + ADB + Accessibility Service）

## 局限性

- 仅覆盖9个应用138个任务，与真实世界应用多样性相比规模有限
- 离线环境无法覆盖需要网络交互的任务场景
- 评估中预设最大25步可能不足以完成某些复杂任务
- 训练数据来源于基准应用自身，泛化到未见过应用的能力待验证
- AVD环境与真实手机仍有差异

## 相关工作

- **移动端基准**: PixelHelp（Li et al., 2020）、AITW（Rawles et al., 2023）、AndroidWorld（Rawles et al., 2024）、B-MOCA（Lee et al., 2024）
- **移动端智能体**: AppAgent（Yang et al., 2023b）、Auto-GUI（Zhan & Zhang, 2023）、CogAgent（Hong et al., 2023）
- **Web智能体**: WebGPT（Nakano et al., 2021）、AutoWebGLM（Lai et al., 2024）、MindAct（Deng et al., 2023）
- **通用代码/API智能体**: HumanEval（Chen et al., 2021）、ToolBench（Guo et al., 2024）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总评 | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)

</div>

<!-- RELATED:END -->
