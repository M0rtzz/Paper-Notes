---
title: >-
  [论文解读] Exploration-Driven Generative Interactive Environments
description: >-
  [CVPR 2025][模型压缩][世界模型] 开源实现 Genie 世界模型（GenieRedux），增加真实动作条件、Token 距离交叉熵（TDCE）损失和 token 跳连得到 GenieRedux-G，并提出 AutoExplore 探索智能体用世界模型的 token 预测不确定性作为内在奖励驱动多样数据收集，将仿真质量提升高达 7.4 PSNR。
tags:
  - CVPR 2025
  - 模型压缩
  - 世界模型
  - 交互式环境
  - 探索智能体
  - 视频生成
  - 游戏模拟
---

# Exploration-Driven Generative Interactive Environments

**会议**: CVPR 2025  
**arXiv**: [2504.02515](https://arxiv.org/abs/2504.02515)  
**代码**: https://github.com/insait-institute/GenieRedux  
**领域**: 模型压缩  
**关键词**: 世界模型、交互式环境、探索智能体、视频生成、游戏模拟

## 一句话总结
开源实现 Genie 世界模型（GenieRedux），增加真实动作条件、Token 距离交叉熵（TDCE）损失和 token 跳连得到 GenieRedux-G，并提出 AutoExplore 探索智能体用世界模型的 token 预测不确定性作为内在奖励驱动多样数据收集，将仿真质量提升高达 7.4 PSNR。

## 研究背景与动机

**领域现状**：世界模型（如 Genie）学习模拟交互式环境——给定动作序列生成对应的视频帧。但 Genie 是闭源的，开源实现缺乏。训练数据通常由随机智能体收集，探索效率低。

**现有痛点**：(1) 随机智能体倾向于重复相同区域，收集的视频缺乏多样性。(2) Latent Action Model（LAM）从视频推断动作的精度有限，引入噪声。(3) 标准交叉熵损失不考虑 codebook 中 token 间的语义距离——预测相近的 token 和完全错误的 token 受到相同惩罚。

**核心矛盾**：世界模型需要多样化的训练视频，但多样化数据的收集需要智能体的有效探索。传统的探索奖励（如好奇心）需要手工设计且依赖环境。

**本文目标** (1) 提供 Genie 的开源实现并改进。(2) 设计环境无关的探索智能体为世界模型收集多样化训练数据。

**切入角度**：世界模型在不确定区域的预测会有高不确定性（高熵）。利用这种内在的"不确定性信号"作为探索奖励——世界模型不确定的区域就是需要更多数据的区域。

**核心 idea**：用世界模型自身的 token 预测不确定性作为内在探索奖励，驱动智能体收集世界模型最需要的数据，形成"探索-学习"的自改进循环。

## 方法详解

### 整体框架
GenieRedux（Video Tokenizer + LAM + Dynamics）→ GenieRedux-G 增加真实动作条件 + TDCE 损失 + token 跳连 → AutoExplore Agent 用 Dynamics 模块 top-25% 不确定 token 的平均熵作为奖励 → 收集多样化数据 → 微调世界模型。

### 关键设计

1. **Token Distance Cross-Entropy（TDCE）**:

    - 功能：根据 codebook token 间的语义距离加权预测损失
    - 核心思路：$TDCE(x,y) = (y^T K) \cdot \text{softmax}(x) + CE(x,y)$，其中 $K$ 是 codebook token 间的余弦距离矩阵。预测到语义相近的 token（视觉上相似）受到更轻的惩罚，完全错误的 token 受到更重的惩罚
    - 设计动机：消融显示 TDCE 贡献 +0.41 PSNR（从 26.65 到 27.06），因为它让模型分配更合理的概率给语义临近的备选 token

2. **AutoExplore Agent（不确定性驱动探索）**:

    - 功能：收集世界模型最需要的多样化训练数据
    - 核心思路：奖励 = 世界模型 Dynamics 模块对当前帧预测中 top-25% 不确定 token 的平均熵。高不确定性区域 = 世界模型不擅长预测的区域 = 需要更多训练数据的区域。Actor-Critic（CNN+LSTM）最大化累积探索奖励
    - 设计动机：随机探索 FID 42.34 / PSNR 27.04 → AutoExplore 后 FID 11.33 / PSNR 33.61（Adventure Island II），提升惊人

3. **RetroAct 数据集**:

    - 功能：标准化的复古游戏环境+动作标注数据集
    - 核心思路：974 个标注的复古游戏环境，包含行为和控制标签。支持标准化的世界模型训练和评估
    - 设计动机：之前没有统一的交互式环境世界模型基准

### 损失函数 / 训练策略
GenieRedux-G：TDCE + token 跳连。AutoExplore：Actor-Critic + 不确定性奖励。预训练在 Platformers-200（4.6M 图像）→ 微调在 Platformers-50（4.8M 图像）→ 每环境用 AutoExplore 收集数据微调。

## 实验关键数据

### 主实验

| 环境 | 策略 | FID↓ | PSNR↑ | ΔPSNR↑ |
|------|------|------|-------|--------|
| Adventure Island II | Random | 42.34 | 27.04 | 1.19 |
| Adventure Island II | **AutoExplore AR** | **11.33** | **33.61** | **2.09** |
| Super Mario Bros | Random | 29.83 | 34.24 | 0.56 |
| Super Mario Bros | **AutoExplore AR** | **9.33** | **37.77** | **0.76** |

### 消融实验

| 组件 | PSNR |
|------|------|
| GenieRedux-G 基础 | 26.36 |
| +Token 输入 | 26.65 |
| +TDCE 损失 | 27.06 |
| +自回归 | **28.07** |

### 关键发现
- **AutoExplore 提升高达 7.4 PSNR**（Adventure Island II）：显示智能探索数据收集的巨大价值
- **TDCE 利用 codebook 结构**：语义临近的 token 不应被同等惩罚
- **随机探索足以训练基础模型**：但智能探索使微调质量飞跃
- **用户研究验证**：探索训练模型以 0.75 偏好率被选为更好

## 亮点与洞察
- **"世界模型的不确定性作为探索奖励"**形成了优美的自改进循环——模型不擅长的地方就是需要更多数据的地方
- **TDCE 损失可推广到任何 VQ-VAE 基模型**——利用 codebook 的语义结构信息是一种通用改进
- **GenieRedux 开源**为世界模型研究提供了可复现的基础

## 局限与展望
- 仅在 2D 复古游戏上验证，3D 环境的适用性未知
- AutoExplore 的 Actor-Critic 需要在每个新环境上训练
- PSNR 改善主要来自探索覆盖新区域，对已见区域的质量提升有限

## 相关工作与启发
- **vs Genie (DeepMind)**：Genie 是闭源的。GenieRedux 提供了开源实现并通过 TDCE/GT 动作/探索智能体做了显著改进
- **vs GameNGen**：针对特定游戏的模型。GenieRedux 是跨环境的通用世界模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 不确定性驱动探索+TDCE损失是巧妙贡献
- 实验充分度: ⭐⭐⭐⭐ 多环境对比、组件消融、用户研究
- 写作质量: ⭐⭐⭐⭐ 探索-学习循环的叙述清楚
- 价值: ⭐⭐⭐⭐ 开源世界模型实现对社区有重要贡献
