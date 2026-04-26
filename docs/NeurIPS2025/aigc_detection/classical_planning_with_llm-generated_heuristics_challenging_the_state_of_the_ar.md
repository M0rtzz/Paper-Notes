---
title: >-
  [论文解读] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code
description: >-
  [NeurIPS 2025][Classical Planning] 提出让 LLM **生成域相关启发式函数的 Python 代码**（而非直接生成计划），通过 $n$ 次采样获得候选启发式池并在训练集上选优，将最优启发式注入 Python 规划器 Pyperplan 配合 GBFS 使用，在 IPC 2023 基准 8 个域上以纯 Python 实现超越了所有 C++ Fast Downward 传统启发式，且与 SOTA 学习型规划器 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ 持平，同时保证所有找到的计划 100% 正确。
tags:
  - NeurIPS 2025
  - Classical Planning
  - LLM Code Generation
  - Heuristic Function
  - PDDL
  - Greedy Best-First Search
  - Domain-Dependent Planning
---

# Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code

**会议**: NeurIPS 2025  
**arXiv**: [2503.18809](https://arxiv.org/abs/2503.18809)  
**代码**: [公开可用](https://doi.org/10.5281/zenodo.14964025)  
**作者**: Augusto B. Corrêa, André G. Pereira, Jendrik Seipp  
**领域**: AI 规划 — 经典规划、启发式搜索  
**关键词**: Classical Planning, LLM Code Generation, Heuristic Function, PDDL, Greedy Best-First Search, Domain-Dependent Planning

## 一句话总结

提出让 LLM **生成域相关启发式函数的 Python 代码**（而非直接生成计划），通过 $n$ 次采样获得候选启发式池并在训练集上选优，将最优启发式注入 Python 规划器 Pyperplan 配合 GBFS 使用，在 IPC 2023 基准 8 个域上以纯 Python 实现超越了所有 C++ Fast Downward 传统启发式，且与 SOTA 学习型规划器 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ 持平，同时保证所有找到的计划 100% 正确。

## 研究背景与动机

- **领域现状**: 经典规划（Classical Planning）是 AI 基础问题，任务用 PDDL 形式化描述。当前主流方法依赖启发式搜索，启发式函数分为三类：① 域无关启发式（如 $h^{\mathrm{FF}}$、$h^{\mathrm{add}}$），通用但不精确；② 人工手写的域相关启发式，需要大量专家知识；③ 逐域学习的启发式（如 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$），每个域需要单独训练。
- **现有痛点**: LLM 直接用于端到端规划（给 PDDL 任务生成计划）不可靠——即使是推理模型（DeepSeek R1、Gemini Thinking）也经常生成错误计划，且无法泛化到比训练任务更大的问题。AoT+ 等最强端到端方法在 Blocksworld 上仅支持 5 个积木，而实际测试任务可达 488 个积木。
- **核心矛盾**: LLM 具备领域理解能力和代码生成能力，但不能可靠地进行多步推理规划；传统规划器有可靠的搜索框架保证正确性，但缺少域特定知识来高效引导搜索。
- **本文目标**: 如何利用 LLM 的代码生成能力为经典规划问题自动生成高质量的域相关启发式函数？
- **切入角度**: 将问题从"让 LLM 解规划问题"转化为"让 LLM 写规划问题的工具代码"——LLM 生成启发式函数（Python 代码），搜索算法保证正确性，两者各司其职。
- **核心 idea**: LLM 不直接规划，而是为每个域生成多个候选启发式函数代码，选优后注入规划器，以"生成工具"替代"直接解题"的范式来提升规划能力。

## 方法详解

### 整体框架

整体流程极其简洁（如论文 Figure 1 所示）：

1. **Prompt 构建**：将 PDDL 域描述、示例任务、示例启发式代码、Pyperplan 接口代码等组织为 prompt
2. **多次采样**：用相同 prompt 请求 LLM $n$ 次（$n=25$），获得 $n$ 个候选启发式函数 $h_1, \dots, h_n$
3. **训练集评估**：在 IPC 训练集上用 GBFS 运行每个启发式（每个任务限时 5 分钟），按解题数排序
4. **选优注入**：选择训练集上表现最佳的 $h_{\text{best}}$ 注入 Pyperplan，用于测试集评估

关键特性：① **无需 LLM-规划器间迭代交互**，一次性完成；② **每域固定成本**，选定启发式后可用于该域的无限新任务；③ **正确性由搜索算法保证**，启发式仅影响搜索效率。

### 关键设计

1. **多组件 Prompt 设计**

    - 功能：为 LLM 提供充分上下文以生成高质量、可编译运行的启发式函数代码
    - 核心思路：Prompt 包含 7 项精心设计的组件——① PDDL 域文件；② 最小和最大训练任务的 PDDL 文件；③ 两个示例域（Gripper 提供完美启发式、Logistics 提供简单启发式）的域文件+任务+启发式 Python 代码；④ 目标域的 Pyperplan 状态表示示例；⑤ 静态信息表示示例；⑥ Pyperplan 的 Task/Action Python 类代码；⑦ 常见陷阱的 checklist
    - 设计动机：示例启发式展示了 Pyperplan 的接口使用方式和代码风格（few-shot），PDDL 文件传递域语义，checklist 基于作者对 LLM 典型错误的观察来预防常见 bug。消融实验证明移除任何组件都会降低最佳启发式质量

2. **多次采样 + 训练集选优的候选池策略**

    - 功能：从 LLM 的随机输出中筛选出高质量启发式
    - 核心思路：将温度设为 1.0 以增加多样性，生成 $n=25$ 个候选启发式。在训练集上按两级指标排序：首先按解题数排序，平局时按 IPC agile score（$1 - \frac{\log t}{\log 300}$）打破。实验表明 $n$ 从 1 增到 5 时解题数提升最大，$n=25$ 后各域训练集几乎被完全覆盖
    - 设计动机：LLM 输出质量方差大（高温采样下标准差可达 ±94），单次采样极不稳定；多次采样后选优可以稳定获得高质量启发式。且此过程每域只做一次，总 API 成本极低（DeepSeek V3 全部 8 域仅 $0.25 USD，R1 仅 $6.12 USD）

3. **搜索算法与启发式的正确性解耦**

    - 功能：保证所有找到的计划 100% 正确，无论启发式质量如何
    - 核心思路：启发式函数仅影响 GBFS 的节点扩展顺序（即搜索效率），不影响计划的正确性——GBFS 只会在到达目标状态时输出计划，而目标测试由 Pyperplan 框架内部完成。即使 LLM 生成了一个完全无意义的启发式，搜索仍能找到正确计划（只是效率低）
    - 设计动机：这是相对于端到端 LLM 规划的根本优势——端到端方法生成的计划需要用 VAL 工具验证，且经常不正确；而本方法将 LLM 限制在"评估函数"角色，正确性完全委托给成熟的搜索框架

### 训练策略

无传统意义上的训练/梯度优化。"训练集"仅用于评估和选择候选启发式（selection），不涉及参数更新。LLM 调用成本统计：DeepSeek V3 生成 200 个启发式（8 域 × 25）总计 $0.25 USD；DeepSeek R1 总计 $6.12 USD。相比之下，端到端规划需要 720 次 LLM 调用（每个任务一次），R1 成本从 $6.12 升至 $13.62。

## 实验关键数据

### 主实验：LLM 生成启发式 vs 端到端 LLM 规划 vs 域无关启发式（Table 2）

IPC 2023 Learning Track，8 个域，每域 90 个测试任务，总计 720 个任务：

| 方法 | Blocks | Child. | Floor. | Miconic | Rovers | Sokoban | Spanner | Trans. | **总计** |
|------|--------|--------|--------|---------|--------|---------|---------|--------|---------|
| 端到端 Gemini Think. | 40 | 59 | 0 | 21 | 5 | 14 | 39 | 24 | 202 |
| 端到端 DeepSeek R1 | 17 | 40 | 0 | 24 | 10 | 8 | 47 | 28 | 174 |
| Pyperplan $h^0$ (BFS) | 6 | 9 | 1 | 30 | 12 | 24 | 30 | 8 | 120 |
| Pyperplan $h^{\mathrm{FF}}$ | 24 | 17 | 10 | 74 | 28 | 31 | 30 | 29 | 243 |
| + Gemini 2.0 Flash | 35 | 32 | 4 | **90** | 32 | 31 | 30 | 42 | 296 |
| + Gemini Think. | 37 | 14 | 8 | 88 | 39 | 32 | 30 | 57 | 305 |
| + DeepSeek V3 | 45 | **55** | 3 | 64 | 34 | 31 | **69** | 42 | **343** |
| + DeepSeek R1 | **66** | 22 | 4 | **90** | 32 | 30 | 70 | **59** | **373** |

### 消融实验：Prompt 组件影响（Table 4，Gemini 2.0 Flash Thinking，$n=25$）

| Prompt 变体 | Best Coverage | Avg. Coverage | Failed Heuristics |
|-------------|--------------|---------------|-------------------|
| 完整 Prompt（原始） | **423** | 267.0 | 64 |
| ⇆ 简化指令 | 359 | 242.5 | 57 |
| − 移除 PDDL 域描述 | 368 | 237.3 | 52 |
| − 移除 PDDL 任务 | 401 | 261.5 | 42 |
| ⇆ 替换为域无关启发式 | 404 | 263.2 | 64 |
| − 移除状态表示示例 | 402 | 253.7 | 68 |
| − 移除静态信息表示 | 404 | 243.8 | 57 |
| − 移除 Pyperplan 代码 | 401 | 270.0 | **97** |
| − 移除 Checklist | 382 | 260.0 | 57 |

### 关键发现

1. **Python 超越 C++**：Pyperplan（Python）+ DeepSeek R1 启发式（373 任务）超越了 Fast Downward（C++）中所有传统启发式的最佳成绩（$h^{\mathrm{add}}$ = 324），并与 SOTA 学习型方法 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$（371）持平。考虑到 Fast Downward 在某些域的节点扩展速度是 Pyperplan 的 **669 倍**，这说明启发式质量可以完全弥补实现效率的差距。

2. **非推理模型的巨大提升**：Gemini 2.0 Flash 端到端仅解 19 题，但生成启发式后解 296 题（**15.6 倍**）；DeepSeek V3 从 48 题升至 343 题（**7.1 倍**）。启发式生成范式消除了推理模型与非推理模型之间的性能鸿沟。

3. **LLM 启发式更有信息量**：在 Blocksworld、Spanner、Transport、Childsnack 等域，DeepSeek R1 启发式的状态扩展数少于 $h^{\mathrm{FF}}$，说明 LLM 不只是靠暴力搜索取胜，而是生成了更准确的启发式。

4. **泛化到未见域**：在全新域 Rod-Rings（未公开过、LLM 训练数据中不存在）上，OpenAI o3 生成的启发式解 58 题，接近 Fast Downward $h^{\mathrm{FF}}$ 的 59 题。在混淆化 Blocksworld（所有符号替换为随机字符串）上仍解 40/90 题，证明 LLM 能推理域逻辑结构而非仅记忆已知启发式。

5. **选择策略有优化空间**：消融实验中 Best Coverage（423）远超实际选出的启发式性能（305），说明当前"训练集解题数"选优策略并非最优，候选池中有更好的启发式未被选中。

## 亮点与洞察

1. **"生成工具而非直接解题"的范式**：这是本文最核心的洞察——LLM 不擅长多步推理规划，但擅长理解域语义并写代码。将 LLM 从"求解器"角色转变为"工具生成器"角色，让搜索算法负责正确性、LLM 负责域知识编码，两者各司其职。这个范式具有广泛的迁移价值。

2. **极低的成本模型**：生成 25 个候选启发式仅需 25 次 LLM 调用、总成本 $0.25-$6.12，且选定的启发式可复用于该域的无限新任务。相比端到端规划（每个任务都需要一次 LLM 调用），成本优势随任务数量线性增长。

3. **Python vs C++ 的意外结论**：出纯 Python 实现超越高度优化的 C++ 系统，量化了"启发式质量"vs"实现效率"的 trade-off——在规划问题中，搜索方向的正确性远比节点扩展速度重要。

4. **LLM 生成启发式的可解释性**：论文展示了 Blocksworld 和 Spanner 的生成代码——Blocksworld 启发式能识别"错位的目标积木"并计算上方需移走的积木数，Spanner 启发式能做贪心最优匹配和最短路径预计算。这些代码是人类可读可理解的。

## 局限与展望

1. **强依赖 PDDL 形式化描述**：方法要求问题以 PDDL 形式给出，不支持自然语言描述的规划任务。虽然已有 NL→PDDL 转换工作可桥接，但增加了流程复杂度。

2. **启发式选择策略次优**：消融实验表明候选池中最佳启发式（423 题）远超被选中的启发式（305 题），说明基于训练集解题数的选择方案有相当大的改进空间。

3. **未引入迭代反馈**：当前流程是单轮生成，不将失败案例或已生成启发式反馈给 LLM。FunSearch 的成功表明迭代改进可以进一步提升质量。

4. **Pyperplan 实现瓶颈**：Python 实现的内存效率和速度远逊于 C++，在 Floortile 等域表现受限。如将 LLM 启发式移植到 Fast Downward，性能可能大幅提升。

5. **域覆盖受限**：由于 Pyperplan 不支持 Ferry 和 Satellite 域，实验仅覆盖 8/10 个 IPC 2023 域。

## 相关工作与启发

| 方法 | 范式 | 正确性保证 | 域适应方式 | 可复用性 | 实现开销 |
|------|------|-----------|-----------|---------|---------|
| 端到端 LLM 规划 | LLM 直接生成计划 | 否 | 零样本 | 否（每任务一次） | 高（推理成本线性增长） |
| AoT+ | LLM + 搜索策略 | 否 | 零样本 | 否 | 高 |
| $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ | GP 回归学习启发式 | 是（GBFS 保证） | 逐域训练 | 是（域内） | 中（特征提取+GP 训练） |
| Katz et al. | LLM 生成后继/目标代码 | 是（搜索保证） | 需人工反馈 | 部分 | 中 |
| Silver et al. | LLM 生成策略程序 | 否（无搜索） | 零样本 | 是 | 低 |
| Tuisov et al. | LLM 生成 Rust 启发式 | 是 | 需手动翻译 PDDL | 否（任务级） | 高（Rust 翻译） |
| **本文** | **LLM 生成 Python 启发式** | **是（GBFS 保证）** | **零样本** | **是（域级）** | **极低（$0.25-$6.12）** |

**启发方向**：
- "生成工具代码而非直接求解"的范式可迁移至**组合优化**（如 LLM 生成 TSP/VRP 的邻域搜索启发式）、**定理证明**（LLM 生成证明策略而非直接证明）等领域
- 候选池选择策略的优化是明确的改进方向——可探索集成学习、交叉验证、或基于多样性的选择
- 将 LLM 启发式与 FunSearch 的迭代改进循环结合，可能实现更高质量的启发式自动生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 LLM 从"规划器"重新定位为"启发式生成器"，问题定义精彩，范式转换深刻
- 实验充分度: ⭐⭐⭐⭐⭐ IPC 2023 标准基准 8 域 720 任务，5 个 LLM 对比，6 个 C++ 启发式基线 + SOTA 学习方法，消融实验完整，记忆 vs 推理验证设计精巧（混淆域+新域）
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法极简（一张图即可理解全流程），实验组织严谨，limitations 坦诚
- 价值: ⭐⭐⭐⭐⭐ 对 AI 规划领域和 LLM 应用范式均有深远影响；成本极低、部署简单、保证正确性的特性使其具有很高的实用价值，"生成工具而非直接解题"的思想具有广泛迁移性

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](../../ACL2026/aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)

<!-- RELATED:END -->
