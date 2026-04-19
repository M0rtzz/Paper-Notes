---
title: >-
  [论文解读] QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code
description: >-
  [NeurIPS 2025][视频理解][神经编译] 提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。
tags:
  - NeurIPS 2025
  - 视频理解
  - 神经编译
  - LLM
  - IR到汇编翻译
  - 自进化提示优化
  - 编译器
---

# QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Code

**会议**: NeurIPS 2025  
**arXiv**: [2511.01183](https://arxiv.org/abs/2511.01183)  
**代码**: https://github.com/ (未提供)  
**领域**: 代码生成 / 神经编译  
**关键词**: 神经编译, LLM, IR到汇编翻译, 自进化提示优化, 编译器

## 一句话总结

提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。

## 研究背景与动机

编译器是不可或缺但极其复杂的软件系统，需要大量人类专业知识来开发和维护。随着LLM在代码相关任务中的卓越表现，**神经编译**（Neural Compilation）——直接用LLM将高级语言或中间表示（IR）翻译为低级汇编代码——成为一个有吸引力的新范式。

**核心痛点**：

**缺乏基准**：没有专门用于IR到汇编编译评估的标准化基准数据集，无法系统性地衡量和追踪进展

**正确性挑战**：即使是最强的LLM，生成的汇编代码的语义正确性仍远未达到传统编译器水平

**优化挑战**：在保证正确性的前提下，让LLM生成的汇编代码性能超越成熟编译器（如clang-O3）更加困难

**核心矛盾**：LLM具有代码理解和生成能力，但直接应用于汇编生成时，既缺乏评估标准，又缺乏有效的能力提升方法。

**本文切入角度**：
- 构建专用基准NeuComBack（分为基础编译L1和优化潜力L2两级）
- 提出自进化提示优化方法，让LLM从自己的调试经验中学习，迭代改进用于汇编生成的提示策略

## 方法详解

### 整体框架

方法分为**离线提示学习**和**在线推理**两阶段。离线阶段通过收集LLM的自调试轨迹、提取洞察、迭代进化提示；在线阶段用进化后的提示指导汇编生成和迭代优化。

### 关键设计

1. **NeuComBack基准数据集**：

    - **Level 1（基础编译，200个任务）**：从ExeBench选取，覆盖多样的真实C程序，重点测试功能正确性。经过严格清洗确保C/C++标准合规，选取编译后LLVM IR最长的程序
    - **Level 2（优化潜力，151个任务）**：来自TSVC向量化编译器测试套件，程序执行路径简单但循环结构复杂，适合评估优化能力
    - 评估指标：ACC（功能正确率）和ACC+Perf（正确且性能超越clang-O3的比率）

2. **神经编译工作流**：

    - **初始生成**：给定IR输入，LLM生成初始汇编候选
    - **自调试**：生成后通过测试验证，若不正确则触发迭代自调试
    - **迭代优化**：从正确的初始汇编出发，进行T轮性能优化
    - 每步生成/优化后均可进行正确性验证和自调试

3. **自进化提示优化（核心贡献）**：

    - **轨迹收集**：用LLM执行编译任务，收集完整的自调试轨迹（包括初始生成错误→调试修正→最终正确的全过程）
    - **洞察提取**：对成功从错误到正确的轨迹，用LLM分析错误模式和有效修复策略
    - **提示进化**：将提取的洞察整合到现有提示中，经LLM审查确认后更新。每个mini-batch编译任务后进行一次提示更新
    - 核心区别：不是通用APO方法，而是专门从**完整的自调试轨迹**中学习——让LLM学习自己过去解决汇编错误的实践经验

### 损失函数 / 训练策略

无传统意义的训练损失。提示优化基于编译正确性反馈，通过3个epoch、batch size=5的迭代过程完成。主要使用DeepSeek-R1作为基础LLM。

## 实验关键数据

### 主实验

**前沿LLM基线性能（NeuComBack-L2, x86_64, 151个用例）**

| 模型 | ACC(%) | ACC+Perf(%) |
|------|--------|-------------|
| GPT-4o | 1.99 (3/151) | 0.66 (1/151) |
| O3-Mini | 21.19 (32/151) | 5.30 (8/151) |
| O1 | 19.87 (30/151) | 5.30 (8/151) |
| DeepSeek-V3 | 14.57 (22/151) | 3.31 (5/151) |
| DeepSeek-R1 | 45.70 (69/151) | 21.85 (33/151) |

**自进化提示优化效果（DeepSeek-R1, x86_64）**

| 方法 | L1测试集ACC(%) | L2初始ACC(%) | L2优化后ACC+Perf(%) |
|------|-------------|-------------|-------------------|
| 基线提示 | 50.00 (20/40) | 44.00 (11/25) | 28.00 (7/25) |
| 学习后提示 | **80.00** (32/40) | **64.00** (16/25) | **56.00** (14/25) |

### 消融实验

| 配置 | ACC(%) | ACC+Perf(%) | 说明 |
|------|--------|-------------|------|
| x86_64基线 → 学习提示 | 44%→64% | 28%→56% | +100%性能超越O3 |
| aarch64基线 → 学习提示 | 36%→72% | 8%→28% | 跨架构有效 |
| L2学习的提示→L1 | 67.5% vs L1专用80% | - | 跨数据分布可迁移 |
| 自调试轮数减少 | 基线0.9→本文0.28轮 | - | 学习后提示减少自调试需求 |

### 关键发现

- DeepSeek-R1是基线中最强的，显著优于GPT-4o（ACC 45.7% vs 1.99%），推理能力至关重要
- 学习后提示在L1上实现60%的相对正确率提升（50%→80%）
- 在L2上，16个LLM正确生成的x86_64程序中有14个（87.5%）性能超越clang-O3
- 提示的优化效果可跨指令集架构迁移（x86_64→aarch64同样有效）
- 学习后的提示包含了格式规则（.text段）、语法规则（.L前缀）、语义规则（void函数清零返回寄存器）等多层面知识

## 亮点与洞察

- **LLM的汇编优化能力令人惊讶**：87.5%的正确程序能超越clang-O3，说明LLM能发现传统编译器未利用的优化机会
- **从自调试轨迹学习**的设计思路独特——不是从成功案例学，而是从"错误→修正"的过程中提炼知识
- LLM能利用向量指令（如cmpps）进行传统编译器未做的向量化优化
- 基准数据集的两级设计（基础编译 vs 优化潜力）很有针对性

## 局限与展望

- 总体功能正确率仍不够高（L2上64%），距实用尚有差距
- 仅在函数级别测试，未涉及更大规模的程序编译
- 高度依赖DeepSeek-R1的推理能力，对其他模型的效果可能差异显著
- 提示学习过程的计算成本不可忽视（需要多轮LLM调用）
- 未考虑安全关键场景中汇编代码的可靠性验证

## 相关工作与启发

- 与LLM Compiler（Meta）、SLADE等工作互补——它们关注预训练，本文关注推理时的提示优化
- 自进化提示的思路可迁移到其他代码翻译任务（如反编译、跨语言翻译）
- NeuComBack基准可作为评估新一代LLM编译能力的标准测试集

## 评分

- 新颖性: ⭐⭐⭐⭐ 自进化提示从调试轨迹学习的思路新颖，基准数据集填补空白
- 实验充分度: ⭐⭐⭐⭐ 跨架构、跨数据分布、迁移性、消融分析全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设置详细，case study有启发性
- 价值: ⭐⭐⭐⭐ 为神经编译领域提供了基准和方法论，但离实用还有距离

<!-- RELATED:START -->

## 相关论文

- [InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)
- [MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [Self-Supervised Any-Point Tracking by Contrastive Random Walks](../../ECCV2024/video_understanding/self-supervised_any-point_tracking_by_contrastive_random_walks.md)
- [DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](../../ECCV2024/video_understanding/dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)

<!-- RELATED:END -->
