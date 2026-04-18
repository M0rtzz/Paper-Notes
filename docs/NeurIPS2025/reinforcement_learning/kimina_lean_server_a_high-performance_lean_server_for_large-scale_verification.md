---
title: >-
  [论文解读] Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification
description: >-
  [NeurIPS 2025 (MATH-AI Workshop)][Lean 4] 提出Kimina Lean Server——一个面向大规模强化学习训练的高性能Lean 4验证服务器，通过服务端并行化和LRU缓存机制实现1.5-2倍的速度提升，已用于训练SOTA定理证明模型Kimina-Prover。
tags:
  - NeurIPS 2025 (MATH-AI Workshop)
  - Lean 4
  - 定理证明
  - 形式化验证
  - 强化学习
  - 高性能服务器
---

# Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification

**会议**: NeurIPS 2025 (MATH-AI Workshop)  
**arXiv**: [2504.21230](https://arxiv.org/abs/2504.21230)  
**代码**: [GitHub](https://github.com/project-numina/kimina-lean-server) (有)  
**领域**: 强化学习 / 定理证明  
**关键词**: Lean 4, 定理证明, 形式化验证, 强化学习, 高性能服务器

## 一句话总结

提出Kimina Lean Server——一个面向大规模强化学习训练的高性能Lean 4验证服务器，通过服务端并行化和LRU缓存机制实现1.5-2倍的速度提升，已用于训练SOTA定理证明模型Kimina-Prover。

## 研究背景与动机

**领域现状**: 神经定理证明近年来快速发展，大语言模型在Lean 4上通过强化学习训练已取得显著进展（如DeepSeek-Prover、Kimina-Prover等）。这一训练范式需要快速、可扩展的证明验证能力。

**现有痛点**: 现有的Lean 4与Python交互工具（LeanDojo、Pantograph、leanclient、LeanInteract等）存在性能瓶颈和可扩展性问题。大多数工具不支持跨CPU核心的并行化，且每次验证都需要高昂的初始化成本（如加载Mathlib）。

**核心矛盾**: 强化学习训练需要每秒验证大量证明（reward信号），但现有工具的验证吞吐量不足以支撑大规模训练。

**本文目标**: 构建一个专为大规模RL验证流水线设计的高性能Lean服务器。

**切入角度**: 在官方Lean REPL之上构建服务端并行和缓存层。

**核心idea**: 通过REPL池并行化 + LRU导入缓存，最大化CPU利用率并消除重复加载开销。

## 方法详解

### 整体框架

Kimina Lean Server采用客户端-服务器架构：
- **服务端**：REST API暴露验证服务，管理Lean REPL进程池，实现并行验证和导入缓存
- **客户端**：轻量级Python包（PyPI发布），通过单一 `check` 函数提交证明批次并获取结构化反馈

### 关键设计

1. **服务端并行化**:

    - 维护一个Lean REPL进程池，每个REPL运行在独立进程中
    - 每个Lean REPL是单线程的，CPU使用不超过一个核心
    - 每个可用CPU核心分配一个持久REPL进程，实现高效并行
    - 请求到达时路由到空闲REPL并返回响应
    - 性能随CPU核心数近乎线性扩展（8核→32核实现~4x加速）

2. **LRU导入缓存机制**:

    - Lean REPL初始化代价高昂，主要因为加载Mathlib等大型库
    - 每个传入脚本被拆分为**header（仅含import）**和**body（剩余代码）**
    - 使用header作为键在LRU缓存中查找已预热的worker
    - 若找到匹配worker，仅需验证body部分，复用已加载的上下文
    - 该机制将平均验证时间从0.099s降至0.051s（**1.94x加速**）

3. **数据提取（Infotree处理）**:

    - 处理Lean的infotree输出，将证明分解为不重叠的tactic序列
    - 每个tactic附带前后的tactic state（目标状态）
    - 支持所有Lean战术，包括 `have`、`let`、`calc` 模式、`conv` 模式
    - 处理流程：提取tactic位置 → 消除重叠 → 提取代码片段 → 处理空白/注释/特殊战术
    - 输出格式特别适合树搜索模型

4. **客户端API设计**:

    - 所有交互通过单一 `check` 函数完成
    - 输入：Lean脚本列表
    - 输出：每个脚本的消息（警告/错误）、REPL环境标识符、耗时、可选infotree
    - 支持 `run_benchmark` 高级函数自动处理数据加载和批处理

### 训练策略

- 与Lean 4 v4.15.0兼容
- 模块化架构：可通过修改进程派生逻辑适配其他Lean证明检查器
- 直接使用官方Lean REPL，兼容REPL支持的所有Lean版本

## 实验关键数据

### 主实验

在NuminaMath-LEAN数据集（9,419个有效、无sorry的证明）上的验证时间对比：

| 项目 | 8核 | 16核 | 32核 | 64核 |
|------|------|------|------|------|
| leanclient | 109:55 | 56:58 | 30:16 | 18:01 |
| LeanInteract | 87:35 | 45:51 | 24:11 | 12:56 |
| **Kimina Lean Server** | **42:40** | **21:48** | **11:33** | **7:56** |

在所有核心数配置下均实现**1.5-2x**的速度优势。

### 扩展性实验

| CPU核心数 | 总验证时间 | 平均每证明时间(s) |
|-----------|-----------|-------------------|
| 8 | 42:40 | 0.272 |
| 16 | 21:48 | 0.139 |
| 32 | 11:33 | 0.074 |
| 64 | 7:56 | 0.051 |

8→32核实现近4x加速，展示了出色的线性扩展能力。

### 消融实验（缓存效果）

| 模式 | 总验证时间 | 平均每证明时间(s) |
|------|-----------|-------------------|
| 有缓存 | 7:56 | 0.051 |
| 无缓存 | 15:28 | 0.099 |

缓存带来**1.94x**加速，对频繁复用相同import的工作流尤其有效。

### 关键发现

1. Kimina Lean Server在所有核心配置下**持续领先**最快的基线LeanInteract
2. 并行化设计使性能随核心数**近线性增长**
3. LRU缓存几乎将验证开销**减半**
4. 已在Kimina-Prover训练中实际应用（在miniF2F基准上两次达到SOTA），验证了系统的鲁棒性

## 亮点与洞察

- **工程导向但影响深远**：填补了大规模RL训练中Lean验证基础设施的关键空白
- LRU缓存设计精妙：利用RL训练中绝大多数证明共享相同import（Mathlib）的特点
- header/body拆分简单有效，将缓存粒度精确到import级别
- infotree后处理的tactic提取功能对树搜索模型（如MCTS）极有价值
- 基于官方Lean REPL构建，确保了长期兼容性和可维护性

## 局限与展望

1. 目前仅支持Lean 4（但架构可扩展到其他证明助手）
2. 客户端为同步API，在超大规模场景下可能需要异步支持
3. 缓存机制在import模式多样的场景下效果可能下降
4. 未与GPU加速方案结合（如将LLM推理和验证在同一机器上协调调度）
5. 内存占用随REPL池大小线性增长，需要充足的RAM

## 相关工作与启发

- **LeanDojo**: 提供gym-like环境但未针对验证速度优化
- **Pantograph**: 对LeanREPL的功能增强（支持have/let/calc模式），但单线程
- **leanclient/LeanInteract**: 支持并行但性能有限
- **ProofWala**: 综合gym环境，非面向RL训练速度设计
- 启发：在AI for Math领域，基础设施工具（高效验证器）的提升可直接转化为模型训练的加速

## 评分

- 新颖性: ⭐⭐⭐ 核心技术（并行+缓存）并非全新，但面向RL验证的系统设计有独到之处
- 实验充分度: ⭐⭐⭐⭐ 多核心配置对比，缓存消融，实际训练验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，代码示例丰富
- 价值: ⭐⭐⭐⭐⭐ 开源且已在SOTA模型训练中验证，社区价值极高
