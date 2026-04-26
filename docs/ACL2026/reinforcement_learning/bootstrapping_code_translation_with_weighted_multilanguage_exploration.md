---
title: >-
  [论文解读] Bootstrapping Code Translation with Weighted Multilanguage Exploration
description: >-
  [ACL 2026][代码翻译] BootTrans 提出了一种自举式多语言代码翻译方法，通过利用单一枢纽语言（Python）的测试用例作为跨语言验证预言，结合双池架构进行经验收集扩展训练数据，并设计语言感知加权机制动态优先处理困难的翻译方向，在 HumanEval-X 和 TransCoder-Test 上显著超越基线。
tags:
  - ACL 2026
  - 代码翻译
  - 自举式探索
  - 语言感知加权
  - RLVR
  - 多语言优化
---

# Bootstrapping Code Translation with Weighted Multilanguage Exploration

**会议**: ACL 2026  
**arXiv**: [2601.03512](https://arxiv.org/abs/2601.03512)  
**代码**: [https://github.com/nju-websoft/BootTrans/](https://github.com/nju-websoft/BootTrans/)  
**领域**: 代码翻译/强化学习  
**关键词**: 代码翻译, 自举式探索, 语言感知加权, RLVR, 多语言优化

## 一句话总结

BootTrans 提出了一种自举式多语言代码翻译方法，通过利用单一枢纽语言（Python）的测试用例作为跨语言验证预言，结合双池架构进行经验收集扩展训练数据，并设计语言感知加权机制动态优先处理困难的翻译方向，在 HumanEval-X 和 TransCoder-Test 上显著超越基线。

## 研究背景与动机

**领域现状**：代码翻译对遗留系统现代化和跨平台互操作至关重要。LLM 在编码任务上取得了显著进展，但代码翻译通常依赖于高质量的平行语料，而这些语料很少配备可执行的测试用例。

**现有痛点**：(1) 多语言平行代码数据稀缺，且很少配备跨语言的可执行测试用例；(2) 无监督方法（如利用代码结构信息的方法）需要海量单语语料，且不能基于功能正确性直接优化；(3) 现有 RLVR 方法面临**输入单调性**（可验证种子仅限于单一枢纽语言）和**优化不平衡**（不同翻译方向难度差异导致学习信号偏斜）两大挑战。

**核心矛盾**：虽然测试用例天然具有跨语言可移植性，但从单一枢纽语言扩展到完整的多语言翻译矩阵面临数据瓶颈和优化失衡的双重障碍。

**本文目标**：(1) 解决多语言代码翻译中训练数据的稀缺性；(2) 缓解多语言同时优化时的优化不平衡问题。

**切入角度**：利用单元测试的跨语言可移植性作为统一验证机制，通过自举式经验收集逐步扩展训练数据覆盖所有翻译方向。

**核心 idea**：以一种语言为轴心，通过 RL 策略模型自身的成功翻译来"自举"扩展训练数据，同时用语言感知权重动态调节不同翻译方向的学习强度。

## 方法详解

### 整体框架

BootTrans 包含两个核心组件：(1) **自举式多语言探索**——利用双池架构（种子池 + 探索池）从枢纽语言逐步扩展到全翻译矩阵；(2) **语言感知加权优化**——基于"兄弟语言"的相对表现动态调整每个翻译方向的损失权重。训练使用 GRPO 算法。

### 关键设计

1. **双池架构 (Dual-Pool Architecture)**:

    - 功能：逐步扩展训练数据以覆盖所有翻译方向
    - 核心思路：种子池 $\mathcal{D}_{\text{seed}}$ 包含枢纽语言（Python）的代码-测试对；探索池 $\mathcal{D}_{\text{explore}}$ 动态存储策略模型在 rollout 中通过所有测试的成功翻译。每次训练迭代优先从探索池采样，不足时从种子池补充。成功翻译的代码可在后续迭代中作为新翻译方向的源输入（如 Java→Python 的反向翻译）
    - 设计动机：通过经验收集打破对平行语料的依赖，使模型能自我构建多语言训练数据；FIFO 队列管理防止池过载

2. **语言感知加权优化 (Language-aware Weight Optimization)**:

    - 功能：缓解多语言翻译中不同方向的优化不平衡
    - 核心思路：对于源代码 $x_i$ 到目标语言 $L_k$ 的翻译，定义兄弟奖励 $\mathcal{R}_{i,\neg k}$ 为其他目标语言的累积奖励总和。权重 $w_{i,k} = \frac{\mathcal{R}_{i,\neg k}}{\mathcal{R}_{i,k} + \mathcal{R}_{i,\neg k}}$，当模型在其他语言表现好但在 $L_k$ 上差时，$w_{i,k}$ 增大，迫使模型更关注困难方向
    - 设计动机：直觉是如果模型通过兄弟语言展示了语义理解能力但在特定语言上挣扎，说明问题在于该语言的语法/习惯表达，需要加大学习强度

3. **验证预言与奖励设计**:

    - 功能：提供跨语言统一的功能正确性验证
    - 核心思路：使用二值可验证奖励 $R(y, T) = \mathbb{1}[y \text{ compiles and passes all tests in } T]$，测试套件通过 MultiPL-E 从 Python 规则化转换到其他语言。编译错误、运行时错误和超时均得 R=0
    - 设计动机：将优化目标与功能等价性对齐而非表面形式相似性

### 损失函数 / 训练策略

使用 GRPO 算法，目标函数为语言感知加权的 PPO 式目标，包含裁剪比率、按同目标语言组计算的优势估计和 KL 惩罚。训练使用 AdamW 优化器，学习率 1e-6，rollout 宏批次大小 256，每个源代码生成 G=8 个候选翻译。

## 实验关键数据

### 主实验

**HumanEval-X CA@1 平均分**

| 方法 | Avg |
|------|-----|
| Qwen3-1.7B (base) | 64.33 |
| BootTrans Qwen3-1.7B | **74.70** (+10.37) |
| Llama-3.1-8B (base) | 61.79 |
| BootTrans Llama-3.1-8B | **78.36** (+16.57) |
| Qwen2.5-7B (base) | 68.50 |
| BootTrans Qwen2.5-7B | **83.84** (+15.34) |

**与其他方法对比 (Qwen3-1.7B, HumanEval-X Avg)**

| 方法 | Avg |
|------|-----|
| CoTran | 64.03 |
| MultiPL-T | 64.74 |
| PPOCoder | 69.21 |
| OORL | 69.92 |
| BootTrans | **74.70** |

### 消融实验

BootTrans 1.7B 模型在 HumanEval-X 上超越了 Qwen3-32B（74.70 vs 67.99），展示了小模型通过 RL 训练可以超越大模型的潜力。在 TransCoder-Test 上，BootTrans 同样带来一致的提升。

### 关键发现

- 自举式探索和语言感知加权两个组件都对最终性能有显著贡献
- BootTrans 使 1.7B 参数的小模型超越了 32B 参数的大模型
- 在所有六个翻译方向上都取得了一致的提升，缓解了优化不平衡问题
- 测试用例的跨语言可移植性是方法成功的关键基础

## 亮点与洞察

- 自举式数据扩展的思路简洁有效，充分利用了测试用例的跨语言可移植性
- 语言感知加权机制直觉清晰，基于"兄弟语言"的对比实现了自适应难度调节
- 小模型超越大模型的实验结果突出了 RL 训练在代码翻译中的价值
- 双池架构的 FIFO 管理策略在工程上考虑周到

## 局限与展望

- 目前仅在 C++、Java、Python 三种语言间实验，未扩展到更多语言
- 依赖 MultiPL-E 的规则化测试转换，可能对某些复杂测试用例失败
- 训练成本较高，需要大量 rollout 和编译执行
- 未来可探索将方法扩展到更多编程语言和更复杂的软件工程场景

## 相关工作与启发

- 与 PPOCoder 和 OORL 等 RL 方法相比，BootTrans 的创新在于数据扩展和加权机制的结合
- MultiPL-E 的测试转换工具为方法提供了关键基础设施
- 自举式训练数据扩展的思路可推广到其他需要验证反馈的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 自举式探索和语言感知加权的组合设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 三种基础模型、两个基准、多种基线的全面对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，算法描述详尽

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICML 2025\] BRITE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning](../../ICML2025/reinforcement_learning/brite_bootstrapping_reinforced_thinking_process_to_enhance_language_model_reason.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ICML 2025\] KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies](../../ICML2025/reinforcement_learning/kea_keeping_exploration_alive_by_proactively_coordinating_exploration_strategies.md)

<!-- RELATED:END -->
