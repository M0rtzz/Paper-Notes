---
title: >-
  [论文解读] Craftium: An Extensible Framework for Creating Reinforcement Learning Environments
description: >-
  [ICML 2025][RL环境] Craftium 基于开源 Minetest 游戏引擎构建了一个灵活高效的 3D RL 环境创建框架，通过 Lua API 实现完全自定义，同时提供标准 Gymnasium 接口和五个基准环境。
tags:
  - ICML 2025
  - RL环境
  - 3D环境
  - Minetest
  - Gymnasium
  - 环境创建框架
---

# Craftium: An Extensible Framework for Creating Reinforcement Learning Environments

**会议**: ICML 2025  
**arXiv**: [2407.03969](https://arxiv.org/abs/2407.03969)  
**代码**: https://github.com/mikelma/craftium/  
**领域**: Reinforcement Learning  
**关键词**: RL环境, 3D环境, Minetest, Gymnasium, 环境创建框架

## 一句话总结
Craftium 基于开源 Minetest 游戏引擎构建了一个灵活高效的 3D RL 环境创建框架，通过 Lua API 实现完全自定义，同时提供标准 Gymnasium 接口和五个基准环境。

## 研究背景与动机
**领域现状**: RL 研究高度依赖环境，但目前主流环境 (Atari ALE, DeepMind Lab, MineRL) 大多基于已有游戏/物理模拟器改编，定制能力有限。

**现有痛点**: (a) 游戏类环境只提供预定义任务，仅能调参无法新建环境；(b) 可创建环境的框架 (Griddly, MiniGrid) 仅支持 2D/网格世界，视觉复杂度不足；(c) MineRL/MineDojo 依赖闭源 Minecraft (Java 实现)，性能和定制受限。

**核心矛盾**: 丰富 3D 视觉环境 vs 灵活定制能力——两者难以兼得。

**本文切入**: 利用 Minetest (开源 C++ 体素引擎 + Lua API) 作为底层平台，兼具性能和可扩展性。

**核心 idea**: 对 Minetest 做最小化修改实现 RL 通信，通过 Lua 脚本定义奖励/终止逻辑，封装为 Gymnasium API。

## 方法详解

### 整体框架
输入：用户通过 Lua 脚本定义环境逻辑 + Minetest 世界地图 → Craftium Python 库通过 TCP 与修改版 Minetest 通信 → 输出 RGB 图像观测 + 接收动作 → 标准 Gymnasium API 供 RL 算法使用。

### 关键设计

1. **观测空间**:

    - 默认为可自定义尺寸的 RGB 图像 (如 64×64)
    - 支持 Gymnasium 的 observation wrapper (如 FrameStack 用于感知运动)
    - 设计动机：图像观测是 3D RL 最直观且通用的输入形式

2. **动作空间**:

    - 默认 21 个键盘动作 (二值) + 鼠标移动元组 $(\Delta x, \Delta y) \in [-1,1]^2$
    - 提供 action wrapper 简化为离散动作子集
    - 设计动机：完整控制保证灵活性，wrapper 降低特定任务的学习难度

3. **环境创建流程**:

    - 创建 Minetest 世界 (可手动/Lua 脚本/地图生成器)
    - 编写 Lua mod：`init.lua` 定义奖励函数和终止条件
    - 通过 `set_reward_once(1.0, 0.0)` 和 `set_termination()` 等 API 控制 RL 逻辑
    - 设计动机：利用 Minetest 成熟的 mod 生态，最小侵入式修改保证前向兼容

### 预定义环境
- **Chop Tree**: 砍树得分，稀疏奖励
- **Room / Small Room**: 导航到红色方块，稠密负奖励
- **Speleo**: 探洞下降，连续奖励 = -Y 坐标
- **Spiders Attack**: 击杀蜘蛛，多轮递增难度

## 实验关键数据

### 主实验
| 环境 | 动作数 | 最大步数 | 奖励类型 | 特点 |
|------|--------|---------|---------|------|
| Chop Tree | 3+鼠标 | 500 | 稀疏 (0/1) | 砍树任务 |
| Room | 1+鼠标 | 500 | 稠密 (-1/步) | 大房间导航 |
| Small Room | 1+鼠标 | 200 | 稠密 (-1/步) | 小房间导航 |
| Speleo | 2+鼠标 | 500 | 连续 (-Y) | 探洞下降 |
| Spiders Attack | 4+鼠标 | 2000 | 稀疏 (0/1) | 战斗 |

### 与其他框架对比
| 特性 | Craftium | MineRL | MineDojo | Griddly | MiniGrid |
|------|----------|--------|----------|---------|----------|
| 3D 视觉 | ✓ | ✓ | ✓ | ✗ | ✗ |
| 自定义环境 | ✓ | ✗ | 有限 | ✓ | ✓ |
| 开源引擎 | ✓ | ✗ (Minecraft) | ✗ | ✓ | ✓ |
| 高性能 (C++) | ✓ | ✗ (Java) | ✗ | ✓ | Python |
| Gymnasium API | ✓ | ✓ | ✗ | ✓ | ✓ |

### 关键发现
- 基于 Lua API 创建新环境只需编写简短脚本 (~10行)
- Minetest 社区已有大量现成 mod 可直接集成
- C++ 实现带来显著性能优势，对比 Java 的 Minecraft

## 亮点与洞察
- **工程导向的创新**：利用已有开源游戏引擎而非从头构建，大幅降低开发成本
- **最小修改原则**：对 Minetest 的改动尽量少，保持与上游的兼容性
- **生态复用**：Minetest 社区的 mod 和游戏可直接集成为 RL 环境

## 局限与展望
- 目前仅支持单智能体，未实现多智能体接口 (Petting Zoo API)
- 5 个预定义环境较少，缺乏标准化基准测试
- 论文缺少 RL 训练的性能实验数据 (如学习曲线)
- 体素风格可能限制视觉迁移到真实世界任务

## 相关工作与启发
- ALE (Atari), DeepMind Lab, MineRL 是主流环境但缺乏定制能力
- ProcGen 引入程序化生成但限于2D
- VizDoom 用 DSL 定制但灵活性不如通用编程语言
- 启发：RL 研究的瓶颈之一是环境构建成本，好的工具能加速研究

## 评分
- 新颖性: ⭐⭐⭐ 思路直觉但执行良好，利用已有引擎而非从头创建
- 实验充分度: ⭐⭐ 主要展示框架能力，缺少 RL 训练的定量实验
- 写作质量: ⭐⭐⭐⭐ 清晰实用，文档完善
- 价值: ⭐⭐⭐ 为 RL 社区提供有价值的工具，但需要更多采用才能验证

<!-- RELATED:START -->

## 相关论文

- [BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning](beaver_building_environments_with_assessable_variation_for_evaluating_multi-obje.md)
- [GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](../../ICLR2026/reinforcement_learning/graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL](pessimism_principle_can_be_effective_towards_a_framework_for_zero-shot_transfer_.md)

<!-- RELATED:END -->
