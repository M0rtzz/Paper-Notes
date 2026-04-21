---
title: >-
  [论文解读] Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling
description: >-
  [ACL 2025 (Outstanding Paper)][LLM/NLP][Token Recycling] 提出 Token Recycling——一种无需额外训练的投机解码方法，将解码过程中被拒绝的候选 token 存入轻量邻接矩阵，通过 BFS 算法构建 draft tree 并用 tree attention 验证，仅需 <2MB 存储即在所有规模 LLM 上实现约 2 倍加速。
tags:
  - ACL 2025 (Outstanding Paper)
  - LLM/NLP
  - Token Recycling
  - 投机解码
  - 邻接矩阵
  - BFS树构建
  - 无训练加速
---

# Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling

**会议**: ACL 2025 (Outstanding Paper)  
**arXiv**: [2504.15754](https://arxiv.org/abs/2504.15754)  
**代码**: 无  
**领域**: 模型压缩 / 高效推理 / 投机解码  
**关键词**: Token Recycling, 投机解码, 邻接矩阵, BFS树构建, 无训练加速  

## 一句话总结

提出 Token Recycling——一种无需额外训练的投机解码方法，将解码过程中被拒绝的候选 token 存入轻量邻接矩阵，通过 BFS 算法构建 draft tree 并用 tree attention 验证，仅需 <2MB 存储即在所有规模 LLM 上实现约 2 倍加速。

## 研究背景与动机

**领域现状**：LLM 参数规模快速增长使推理延迟成为核心瓶颈。投机解码（speculative decoding）通过"猜测-验证"范式提供无损加速，已成为 LLM 推理加速的主流方向。现有方法分为两大路线：基于额外 draft 模型的方法（如 Medusa、EAGLE）需要额外训练；基于检索的方法（如 REST）依赖大型 n-gram 检索库。

**现有痛点**：基于 draft 模型的方法需要额外训练小模型作为 drafter，训练成本高且需为每个目标模型单独训练；基于检索库的方法（REST 等）需要 GB 级存储构建领域特定的 n-gram 库，检索速度慢且跨领域适应性差。两条路线都引入了显著的额外资源开销，限制了投机解码在实际部署中的广泛应用。

**核心矛盾**：投机解码的核心价值在于"零成本"加速推理，但现有方法本身引入了显著的训练成本或存储成本——加速方法自身的开销抵消了其带来的收益，尤其在小规模部署场景下得不偿失。

**本文目标** 设计一种真正零训练、极低存储的投机解码方法：(1) 不需要额外的 draft 模型或训练流程；(2) 不需要预构建大型检索库；(3) 存储开销极小（<2MB）；(4) 能自适应当前任务和领域。

**切入角度**：解码过程中被采样但最终被拒绝的候选 token 虽然在当前步骤"不正确"，但包含了丰富的 next-token 概率信息——这些"废弃" token 高概率会在未来序列中出现。关键洞察是将这些被丢弃的 token 视为可回收利用的"宝藏"，通过邻接矩阵存储它们的后继关系，为后续步骤提供高质量的猜测候选。

**核心 idea**：回收解码过程中被拒绝的候选 token 存入轻量邻接矩阵，通过 BFS 构建多路径 draft tree 实现零训练、自适应的投机解码加速。

## 方法详解

### 整体框架

Token Recycling 在推理过程中维护一个轻量级邻接矩阵，记录 token 之间的后继关系。每步解码包含四个阶段：(1) **收集阶段**——将当前步骤中被采样但未被选中的候选 token（top-k 概率的 next-token）存入邻接矩阵，记录每个 token 最可能的 k 个后继 token 及其概率；(2) **构建阶段**——从当前已接受的 token 出发，在邻接矩阵上执行广度优先搜索（BFS），构建包含多条可能路径的 draft tree，每层展开概率最高的若干后继节点；(3) **验证阶段**——通过 tree attention 将整棵 draft tree 一次性送入目标 LLM 做前向传播，使用特殊的 tree attention mask 确保因果性，接受与目标 LLM 分布一致的 token 路径（保证无损）；(4) **更新阶段**——验证后新产生的候选 token 再次更新邻接矩阵，形成在线学习闭环。整个过程从零开始，邻接矩阵随解码逐步丰富。

### 关键设计

1. **邻接矩阵存储**:
    - 功能：以极低存储开销记录全局 token 后继关系
    - 核心思路：矩阵大小为词表大小 × k（k 为每个 token 保留的最大后继数），存储每个 token 最可能的后继 token 及其概率
    - 设计动机：仅需 <2MB 存储（对比检索库方法 GB 级别），且随解码过程动态更新，自然适应当前任务和领域，无需预构建领域特定库

2. **BFS 树构建**:
    - 功能：从当前 token 出发生成高质量的多路径猜测树
    - 核心思路：在邻接矩阵中做广度优先搜索，每层按概率排序展开后继节点，生成的 draft tree 覆盖多条并行猜测路径
    - 设计动机：树的深度和宽度可根据计算预算灵活调整——BFS 确保先扩展高概率路径，平衡猜测覆盖率与验证计算开销

3. **Tree Attention 并行验证**:
    - 功能：一次前向传播验证多条猜测路径
    - 核心思路：将整棵 draft tree 通过特殊 tree attention mask 送入目标 LLM，在单次前向中同时评估所有路径的 token 概率
    - 设计动机：与 tree attention 的天然匹配使 BFS 构建的多路径树可高效并行验证，无需逐路径串行检查

### 训练策略

完全无需训练——整个方法在推理时自组织。邻接矩阵从空矩阵开始，随解码过程在线更新。解码约 100+ token 后邻接矩阵即收敛到有效状态。

## 实验关键数据

### 主实验（加速比对比）

| 对比方法 | 类型 | Token Recycling 优势 |
|---------|------|---------------------|
| vs 无训练方法（REST, n-gram检索） | 无训练 | **加速比提升 +30%** |
| vs 有训练方法（Medusa, EAGLE） | 需训练 | **加速比提升 +25%** |
| 存储开销 | - | **<2MB**（REST 需 GB 级别） |
| 适用 LLM 规模 | - | 所有规模均实现 **~2x 加速** |

### 不同任务加速效果

| 任务类型 | 加速比 |
|----------|--------|
| 对话生成 | ~2.0x |
| 代码生成 | ~2.0x |
| 翻译 | ~1.8x |

### 消融实验

| 消融维度 | 发现 |
|----------|------|
| 邻接矩阵收敛速度 | 解码 100+ token 后收敛到有效状态 |
| BFS 树宽度/深度 | 存在最优平衡点——太宽浪费验证计算，太深匹配概率低 |
| 跨任务适应性 | 邻接矩阵在线更新使方法自动适应不同任务类型 |
| 冷启动阶段 | 前 ~100 token 加速比较低，之后迅速提升 |

### 关键发现

- 解码过程中被丢弃的候选 token 具有极高的复现率，验证了"废弃 token 回收"的核心假设
- 邻接矩阵的在线学习特性使方法天然具备领域适应能力，无需预构建领域库
- Token Recycling 在所有测试的 LLM 规模（7B～70B+）上均有效，加速比稳定在 **1.8x～2.2x**

## 亮点与洞察

- **"回收废弃 token"的洞察极为简洁优雅**：解码过程中自然产生的候选 token 包含丰富的 next-token 概率信息，但一直被白白丢弃。本文将这些"垃圾"变为加速推理的"宝藏"
- **零训练 + 极低存储**：<2MB 邻接矩阵即可实现 2x 加速，对比需要额外训练 draft model 或维护大型检索库的方法，实用性大幅提升
- **自适应性强**：邻接矩阵随解码过程在线更新，自然适应当前上下文、任务和领域
- **与 tree attention 的完美结合**：BFS 构建的多路径 draft tree 通过 tree attention 高效并行验证

## 局限与展望

- 对话开始时邻接矩阵为空，需要约 100 token 的"冷启动"期
- 邻接矩阵基于 token 级别的局部后继关系，无法捕捉更长距离的语义依赖
- 在高温度采样（高随机性生成）时，token 复现率降低，加速效果可能减弱
- 未探索与基于模型的投机解码方法（Medusa/EAGLE）的混合使用

## 相关工作与启发

- **vs Medusa/EAGLE 等 draft-model 方法**：后者需额外训练小模型作 drafter；Token Recycling 完全无训练，更通用
- **vs REST 等检索增强投机解码**：后者需 GB 级检索库；Token Recycling 仅需 <2MB 且自适应更新
- **vs N-gram 匹配方法**：后者依赖静态 n-gram 表；Token Recycling 的邻接矩阵动态更新适应性更强
- **启发**：邻接矩阵的在线学习思路可迁移到其他需要模式预测的场景；与 NSA 共同主题——充分利用"已有计算"的副产品

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](../../NeurIPS2025/llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)
- [The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](token_granularity_impact.md)
- [Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [Conformity in Large Language Models](conformity_in_large_language_models.md)
- [Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)

<!-- RELATED:END -->
