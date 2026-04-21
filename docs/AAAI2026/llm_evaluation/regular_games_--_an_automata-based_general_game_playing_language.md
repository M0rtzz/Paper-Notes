---
title: >-
  [论文解读] Regular Games – an Automata-Based General Game Playing Language
description: >-
  [AAAI 2026][通用博弈] 提出 Regular Games (RG) 通用博弈系统，以非确定性有限自动机（NFA）为核心描述博弈规则，配合多层次语言（底层 RG + 高层 HRG + 专用框架），在表达力覆盖所有有限回合制博弈（含不完全信息和随机性）的同时，生成的前向模型效率全面超越现有最快的通用博弈系统 RBG，通常比 Ludii 快 10-20 倍。
tags:
  - AAAI 2026
  - 通用博弈
  - 游戏描述语言
  - 有限自动机
  - 前向模型
  - 程序化内容生成
---

# Regular Games – an Automata-Based General Game Playing Language

**会议**: AAAI 2026  
**arXiv**: [2511.10593](https://arxiv.org/abs/2511.10593)  
**代码**: [GitHub (RG)](https://github.com/radekmie/rg) / [GitHub (Compiler)](https://github.com/WoojtekP/RGcompiler)  
**领域**: 通用博弈 / 形式语言 / 人工智能  
**关键词**: 通用博弈, 游戏描述语言, 有限自动机, 前向模型, 程序化内容生成

## 一句话总结

提出 Regular Games (RG) 通用博弈系统，以非确定性有限自动机（NFA）为核心描述博弈规则，配合多层次语言（底层 RG + 高层 HRG + 专用框架），在表达力覆盖所有有限回合制博弈（含不完全信息和随机性）的同时，生成的前向模型效率全面超越现有最快的通用博弈系统 RBG，通常比 Ludii 快 10-20 倍。

## 研究背景与动机

通用博弈（General Game Playing, GGP）追求构建能够适应并成功玩任何给定游戏的智能体，是通用人工智能在博弈领域的具象化。现有 GGP 系统存在根本性矛盾：

**GDL（Game Description Language）**：基于逻辑编程，设计优雅但计算效率极低。逻辑推理的开销使大型游戏难以运行；GDL-II 支持不完全信息但问题更严重。

**Ludii**：拥有数千个高层关键词（ludemes），游戏描述简短且易用，支持不完全信息，但因过度复杂只能在自己的 Java 生态内使用，比 GDL 快但仍有性能瓶颈。

**RBG（Regular Boardgames）**：追求极简和高效，用正则表达式定义规则并编译为 C++，是此前最快的 GGP 语言。但仅支持完全信息确定性游戏，且复杂游戏的描述冗长、编译时间长。

核心矛盾在于：表达力、易用性和计算效率三者难以兼得。RG 的目标是同时实现这三者——像 GDL 一样通用、像 Ludii 一样方便、像 RBG 一样高效。

## 方法详解

### 整体框架

RG 系统采用多层次架构设计：

- **底层语言 RG**：以 NFA（有向图 + 边标签）为核心，机制极简（仅类型、变量、常量和自动机边），便于自动处理（智能体、优化、规则分析）。覆盖所有有限回合制博弈（含不完全信息和随机性），与 GDL-II 和 Ludii 的表达力等价。
- **高层语言 HRG**：面向人类游戏设计者，使用声明式和结构化编程语法，保持完全通用性的同时大幅提升可读性。最终编译为底层 RG。
- **专用框架**：如 LineGames Python 库，用几行代码定义特定类型游戏规则，生成 HRG 代码。
- **跨语言翻译**：支持 RBG → RG 和 GDL → RG（实验性）的自动翻译。

编译后的 RG 描述生成高度优化的 C++ 推理模块，提供统一的前向模型接口（判断终局、计算合法走法、执行走法等）。

### 关键设计

1. **基于 NFA 的规则描述**：博弈规则表示为非确定性有限自动机 $(Q, \delta)$，其中 $Q$ 为节点集合，$\delta$ 为转移函数。游戏状态为 $(\mathcal{S}, q)$（半状态 + 当前节点）。NFA 比正则表达式（RBG 使用）更灵活且理论上指数级更简洁。设计动机：自动机表示天然适合优化（数据流分析、常量传播、路径剪枝等），且不内置棋盘、算术等概念，纯粹操作抽象符号，保持最大灵活性。

2. **五种动作类型**：空动作（无操作）、比较（相等/不等）、赋值（修改变量）、可达性检查（复杂条件验证，如"是否存在合法路径"）和标签（构成走法的标记）。可达性检查是一个特别强大的机制——它允许在不改变状态的前提下验证自动机中是否存在满足某些条件的路径，支持了"棋盘上是否存在空位""是否存在获胜路径"等复杂查询。

3. **不完全信息与随机性支持**：通过 `visible` 变量控制每个玩家能观察到的走法部分（遮蔽走法机制）。`random` 特殊玩家从合法走法中均匀随机选择，概率分布可通过走法复制或走法序列控制。`keeper` 管理游戏状态，始终恰好有一个合法走法，负责非玩家操作。

4. **优化管线**：包括冗余标签剪枝、赋值内联、常量传播等变换，运行在不动点循环中。使用数据流分析计算每个节点的游戏状态知识。翻译自 RBG 的自动机经优化后节点减少 72%、边减少 66%、状态大小减少 21%。HRG 游戏的节点和边分别减少 47% 和 41%。大多数 HRG 游戏的完整翻译和优化在 100ms 内完成。

5. **IDE 与工具链**：提供 LSP 支持的代码编辑器（语法高亮、自动补全、诊断等）、自动机可视化、基准测试工具和变换调试器。RG 和 Ludii 是仅有的两个拥有工业级 IDE 的 GGP 语言。

### 理论性质

- **通用性定理**：RG 可编码所有有限回合制博弈（含不完全信息和随机性），其中概率为有理数。同时移动可通过不完全信息建模，与 GDL-II 和 Ludii 的表达力等价。
- **计算复杂性**：类型长度固定时，判定初始状态是否存在合法走法为 PSPACE-complete；一般情况下为 EXPSPACE-complete。

## 实验关键数据

### 主实验

效率对比以蒙特卡洛随机对局（playouts per second）为指标，覆盖 30+ 种游戏：

| 游戏 | RG (HRG) | RBG | Ludii | RG/RBG 加速比 |
|------|----------|-----|-------|--------------|
| Connect Four | 1,297,176 | 914,514 | 55,858 | 1.4× |
| Chess | 1,572 | 995 | 113 | 1.6× |
| Breakthrough | 82,135 | 50,977 | 3,365 | 1.6× |
| Reversi | 28,445 | 19,838 | 1,497 | 1.4× |
| Pentago | 43,875 | 22,553 | — | 1.9× |
| Tic-Tac-Die | 2,708,648 | — | 36,702 | — |
| Fox and Hounds | 444,243 | 331,884 | 14,216 | 1.3× |

### 消融实验（优化效果）

| 配置 | 节点减少 | 边减少 | 状态大小减少 |
|------|----------|--------|-------------|
| RBG → RG 优化后 | 72% | 66% | 21% |
| HRG 优化后 | 47% | 41% | — |

### 关键发现

1. **全面超越 RBG**：所有以 HRG 实现的游戏，RG 生成的前向模型均快于 RBG（此前最快的 GGP 语言）。最大加速在 Pentago（表现优势源于 RG 无需像 RBG 那样使用变通方案处理旋转操作）。
2. **比 Ludii 快 10-20 倍**：典型游戏上 RG 比 Ludii 快一个数量级以上（如 Connect Four：1.3M vs 56K，Chess：1.6K vs 113）。
3. **自动翻译的 RBG 游戏也有竞争力**：RBG 自动翻译到 RG 后效率与原生 RBG 相当，部分甚至更快，验证了 RG 作为"GGP 汇编语言"的可行性。

## 亮点与洞察

- **多层次架构设计思想出色**：底层追求极简便于机器处理，高层追求友好便于人类设计，中间通过编译连接。这种"GGP 汇编语言"的定位使 RG 可作为多种高层语言的统一目标语言。
- **NFA 替代正则表达式的选择有理论支撑**：自动机比正则表达式指数级更简洁，且天然适合图优化算法。
- **正确性优先的工程实践**：每次变换后运行验证器（类型检查、可达性、正确性），确保优化不破坏游戏语义。
- **跨语言互操作性**：RBG → RG、GDL → RG 的翻译支持使得 RG 能够服务更广泛的社区，而不是从零开始构建游戏库。

## 局限与展望

- 目前游戏库规模有限，远不及 Ludii 的大型历史游戏数据库，社区影响力有待建设。
- GDL → RG 翻译尚为实验性，支持程度有限（如未检测交替走法）。
- 缺少与强化学习训练流程的深度集成实验，效率优势能否转化为 AI 训练加速尚待验证。
- 不完全信息游戏的实际案例展示较少（如扑克），主要以棋盘游戏为主。
- 编译到 C++ 的方式虽然高效但限制了与 Python 生态（RL 研究主流）的直接集成。

## 相关工作与启发

- 从 AlphaGo → AlphaZero → MuZero 的通用化路径来看，GGP 是 AI 泛化能力的重要试验场。RG 的效率优势可为大规模 RL 实验提供更快的环境模拟。
- 与 OpenSpiel 等基于特定实现的系统相比，RG 保持了规则的形式化可分析性，支持规则分析和过程式内容生成。
- 多层语言架构的设计思路对其他领域（如可编程网络、硬件描述语言）的 DSL 设计有借鉴意义。
- 优化管线中的数据流分析策略可能启发对其他形式化描述语言（如 planning PDDL）的编译优化研究。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — NFA 作为通用博弈核心描述、多层语言架构、跨语言翻译均为新贡献
- **技术深度**: ⭐⭐⭐⭐⭐ — 形式化定义严谨、通用性与复杂性定理完整、优化管线工程深度高
- **实验充分性**: ⭐⭐⭐⭐ — 30+ 游戏的效率对比全面，但缺少 AI 训练端的效果验证
- **实用性**: ⭐⭐⭐⭐ — 有完整工具链和 IDE，但社区生态尚在建设中
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Can Large Language Models Master Complex Card Games?](../../NeurIPS2025/llm_evaluation/can_large_language_models_master_complex_card_games.md)
- [Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models](../../ACL2025/llm_evaluation/ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)
- [ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)

<!-- RELATED:END -->
