---
title: >-
  [论文解读] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
description: >-
  [CVPR 2026][机器人][视觉语言] MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2%、真机 SO101 上 90.0% 成功率。
tags:
  - CVPR 2026
  - 机器人
  - 视觉语言
  - 模型合并
  - 多任务机器人
  - task mask
  - 跨技能泛化
---

# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent

**会议**: CVPR 2026  
**arXiv**: [2511.18810](https://arxiv.org/abs/2511.18810)  
**代码**: [项目主页](https://mergevla.github.io/)  
**领域**: 机器人操控 / VLA模型 / 模型合并  
**关键词**: Vision-Language-Action, 模型合并, 多任务机器人, task mask, 跨技能泛化

## 一句话总结

MergeVLA 通过诊断 VLA 模型不可合并的两大根因（LoRA 参数冲突 + action expert 自注意力导致的架构不兼容），设计了稀疏激活的 task mask 和去除自注意力的 action expert 架构，实现了多个单任务 VLA 专家的免训练合并，在 LIBERO 上达到 90.2%、真机 SO101 上 90.0% 成功率。

## 研究背景与动机

**领域现状**：VLA（Vision-Language-Action）模型通过微调 VLM 做机器人操控，单任务效果优异但无法泛化多任务。模型合并（model merging）在 LLM/VLM 领域已证明有效。

**现有痛点**：

1. 直接合并 VLA 专家会导致**成功率降为零**——这在 LLM/VLM 合并中从未出现过
2. 合并 4 个任务时，75% 以上的 LoRA 参数是"自私"的（仅被一个任务保留），参数冲突极度严重
3. Action expert 中自注意力层在训练中积累强烈的任务依赖，使深层块参数距离爆炸式增长，破坏模块化可组合性

**核心矛盾**：VLA 模型的 LoRA 参数在不同任务间极度分化，action expert 的自注意力传播了任务专属信息到所有层——两者叠加导致现有合并方法完全失效。

**本文目标** 设计一种"为合并而生"的 VLA 架构，让多个单任务专家可以高效合并为一个通才模型。

**切入角度**：先精确诊断失败的两个根因，再针对性设计解决方案——task mask 解决参数冲突，去自注意力解决架构不兼容。

**核心 idea**：通过稀疏激活的 task mask 抑制 LoRA 冲突参数 + 删除 action expert 的自注意力层消除任务依赖传播，使 VLA 模型"天生可合并"。

## 方法详解

### 整体框架

MergeVLA 基于 VLA-Adapter 架构（Qwen2.5-0.5B 作为 VLM backbone），做了三个关键改造：(1) VLM 的 LoRA 加 task-specific binary mask 解决参数冲突；(2) action expert 去除所有自注意力层、只保留交叉注意力；(3) 推理时用免训练的 task router 自动判断当前任务。各任务独立微调后，合并阶段完全免训练。

### 关键设计

1. **Task Mask（解决 LoRA 参数冲突）**

    - 功能：为每个任务构建 binary mask，选择性激活与该任务一致的合并参数，抑制冲突参数
    - 核心思路：对每个参数位置，检查任务向量是否与合并向量方向一致且显著：S_m = I[|tau_m| > lambda * |tau_merge - tau_m|]，其中 lambda 控制容忍度
    - 实际效果：合并 4 个任务时自私参数占比超过 75%，mask 保留有益参数、抑制冲突，同时促使部分参数回退到预训练权重减轻视觉遗忘
    - 设计动机：直接合并会激活与当前任务无关甚至矛盾的参数，mask 实现了参数的选择性激活

2. **去自注意力的 Action Expert（解决架构不兼容）**

    - 功能：重新设计 action expert 架构使其天生可合并
    - 核心思路：(a) 删除所有自注意力层，只保留交叉注意力——迫使 expert 依赖 VLM 的鲁棒特征；(b) 把 tanh gate 换成 sigmoid gate，避免负激活抑制 VLM 信号
    - 浅层块直接用权重平均合并，最后一层（expert head）保持任务独立不合并
    - 设计动机：自注意力在从头训练中积累任务偏差且跨层传播，删除后强制依赖预训练 VLM 特征反而提升泛化性（OOD 上 +13.4%）

3. **Test-time Task Router（免训练任务推断）**

    - 功能：在任务身份未知时自动判断当前任务并选择对应的 mask 和 expert head
    - 核心思路：对每个候选任务 m，用对应 task mask 构建 VLM 变体 → 提取隐状态 → 投影到 action expert value 矩阵的 top-k_r 右奇异向量子空间 → 计算激活强度 → softmax 选最高分任务
    - 只需在 t=0 做一次路由，后续固定
    - 设计动机：value 子空间直接编码任务依赖信息，比 query/key 更稳定和区分性

### 损失函数 / 训练策略

- 各任务独立用标准模仿学习训练（30k-50k 步，batch size 8，LoRA rank 32）
- 合并阶段完全免训练：用 TIES/TA/WUDI 等方法合并 LoRA + 权重平均 action expert 浅层 + 保留各任务 expert head
- 设备：单卡 NVIDIA A6000 Ada 48GB

## 实验关键数据

### 主实验

| 方法 | 数据集 | 平均成功率(%) | 对比 |
|------|--------|------------|------|
| MergeVLA (TIES+Mask) | LIBERO (4 suites) | **90.2** | 单任务微调上限 96.7%（-6.5pp） |
| MergeVLA | LIBERO-Plus (OOD) | **62.5** | VLA-Adapter 59.0%（同为单任务微调） |
| MergeVLA | RoboTwin (跨具身) | **70.7** | 单任务微调上限 76.0%（-5.3pp） |
| MergeVLA | SO101 真机 (3 tasks) | **90.0** | 与单任务微调持平 |

| 方法 | Params(B) | Spatial | Object | Goal | Long | Avg |
|------|----------|---------|--------|------|------|-----|
| OpenVLA (TA 合并) | 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OpenVLA (TA+Mask) | 7 | 74.2 | 82.6 | 68.8 | 24.0 | 62.4 |
| VLA-Adapter (TA+Mask) | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MergeVLA (TIES+Mask) | 0.70 | 94.8 | 94.6 | 91.8 | 79.4 | **90.2** |

### 消融实验

| 配置 | LIBERO Avg | 说明 |
|------|-----------|------|
| 仅 Mask（不改 action expert） | 0.0% | mask 必要但不充分 |
| 仅去自注意力（无 mask） | 65.5% | 架构改造有效但需配合 mask |
| 去自注意力 + Mask | **90.2%** | 两者缺一不可 |
| lambda=0.6~0.9 | >70% | 最佳容忍度区间 |
| 路由用 Value | **89.7%** | 最稳定 |
| 路由用 Key | 下降严重 | 某些任务直接 0% |
| 去自注意力（LIBERO-Plus OOD） | +13.4% | 仅此一项修改就大幅提升泛化性 |

### 关键发现

- Task mask + 去自注意力**缺一不可**：前者解决 VLM 参数冲突，后者解决 action expert 不可组合
- 仅删除自注意力就在 OOD 上提升 13.4%，自注意力是泛化性的主要瓶颈
- Value 子空间做路由远优于 Query/Key
- 真机实验中合并模型与单任务微调持平（90.0%），证明实用可行

## 亮点与洞察

1. **诊断式研究范式**非常优雅：先用实验精确定位两个根因，再针对性设计解决方案
2. 架构修改极简但效果显著——去自注意力 + 换门控函数就大幅提升泛化性
3. Test-time task router 完全免训练，利用 Value 子空间的 SVD 做任务判别
4. 真机 SO101 上合并后性能等同单任务微调（90%），实用价值高

## 局限与展望

1. 每个任务仍需保留一个 expert head 和 task mask，任务数增多时存储线性增长
2. VLM backbone 只用了 0.5B 的 Qwen2.5，更大模型（7B+）的有效性未验证
3. 路由只在 t=0 做一次判断，需中途切换技能的长序列任务可能不够
4. 跨具身实验规模较小（3 种机器人），大规模异构合并的可扩展性待验证

## 相关工作与启发

- **vs OpenVLA**：直接合并完全失败（0%），因 LM body 的任务冲突无法通过简单方法解决。MergeVLA 用 task mask 绕过了问题
- **vs VLA-Adapter**：自注意力导致 action expert 不可合并，即使加 mask 也失败。MergeVLA 从架构上消除障碍
- **vs pi0/pi0.5**：大规模 VLA 依赖联合训练实现多任务，成本高昂。MergeVLA 允许独立训练再合并，更灵活
- **启发**："去自注意力提升泛化性"值得关注——其他从头训练的模块是否也存在类似现象？可探索扩展到连续技能学习场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 诊断+设计的范式清晰，但每个技术点本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个仿真 benchmark + 真机实验 + 丰富的消融和分析
- 写作质量: ⭐⭐⭐⭐⭐ 叙事逻辑清晰，从诊断到解决方案层层递进
- 价值: ⭐⭐⭐⭐ 解决了 VLA 合并的关键问题，对机器人多技能学习有实际意义

<!-- RELATED:START -->

## 相关论文

- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)

<!-- RELATED:END -->
