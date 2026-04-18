---
title: >-
  [论文解读] Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy
description: >-
  [CVPR 2025][多模态Agent][Minecraft][行为克隆] 提出GOAP策略建模观察-动作因果关系，结合MLLM实现开放式指令理解，在原子/长程/开放任务上全面超越SOTA
tags:
  - CVPR 2025
  - 多模态大模型
  - 具身智能
  - Minecraft
  - 行为克隆
  - 开放世界Agent
  - MLLM
---

# Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy

**会议**: CVPR 2025  
**arXiv**: [2502.19902](https://arxiv.org/abs/2502.19902)  
**代码**: [https://cybertronagent.github.io/Optimus-2.github.io/](https://cybertronagent.github.io/Optimus-2.github.io/)  
**领域**: 多模态大模型 / 具身智能  
**关键词**: Minecraft Agent, 多模态大语言模型, 行为克隆, 目标条件策略, 观察-动作因果建模, GOAP, MGOA数据集

## 一句话总结

提出Optimus-2，通过MLLM进行高层规划，结合Goal-Observation-Action Conditioned Policy (GOAP)进行底层控制，其中GOAP使用Action-guided Behavior Encoder建模观察-动作因果关系，并用MLLM对齐行为token与语言指令，在Minecraft原子任务上平均提升27%、长程任务提升10%、开放指令任务提升18%。

## 研究背景与动机

**领域现状**: Minecraft作为开放世界环境的代表性测试平台，已催生了大量智能体研究。当前主流框架采用"规划器+策略"的两层架构——MLLM作为规划器将复杂任务分解为子目标序列，目标条件策略（如STEVE-1、GROOT）执行具体的低层控制动作。

**现有痛点**:
- 现有策略忽略了观察与动作之间的因果关系——当前观察是由上一步动作与环境交互产生的，但现有策略仅建模子目标与当前观察的关系（简单地将目标嵌入加到视觉特征上）
- 现有策略对开放式自然语言子目标的理解能力有限——STEVE-1使用MineCLIP作为目标编码器，GROOT使用视频编码器，产生的隐式目标嵌入表达能力不足
- 缺乏大规模高质量的目标-观察-动作对齐数据集——VPT数据缺少语言指令，STEVE-1数据仅有32K对齐样本

**核心矛盾**: 策略需要同时理解"去哪里"（子目标语义）和"怎么去"（观察-动作序列的时序依赖），但现有方法在两个维度上都存在瓶颈。

**本文目标** 设计一个能同时建模观察-动作因果关系和理解开放式语言指令的策略网络，并构建大规模训练数据。

**切入角度**: 引入MLLM作为策略的backbone来理解开放式指令，同时设计专门的编码器来捕获观察-动作的时序因果关系。

**核心 idea**: 用Action-guided Behavior Encoder将观察-动作序列压缩为固定长度的"行为token"，然后让MLLM在语言空间中对齐这些行为token与子目标指令来预测动作。

## 方法详解

### 整体框架

Optimus-2采用规划器-策略架构：MLLM规划器（GPT-4V）将复杂任务分解为子目标序列，GOAP策略依次执行每个子目标。GOAP由Action-guided Behavior Encoder和MLLM backbone两部分组成。

### 关键设计

1. **Action-guided Behavior Encoder**: 包含两个子模块。**Causal Perceiver**：在每个时间步，通过交叉注意力将动作嵌入（key/value）注入到视觉特征（query）中，显式建模"动作→观察"的因果关系，使视觉表示包含任务相关的动作信息。**History Aggregator**：引入固定长度的"行为token"，通过历史注意力层与历史行为token序列交互，结合记忆库（Memory Bank）动态聚合和压缩长期历史信息。这样既捕获了长程时序依赖，又不会因输入过长而超出模型限制。

2. **MLLM作为策略backbone**: 使用DeepSeek-VL-1.3B作为初始化，输入包含子目标文本、当前观察的视觉token和行为token。MLLM利用其语言理解能力将开放式子目标与行为序列对齐，自回归地预测下一步动作。使用VPT作为动作头（Action Head），将MLLM输出嵌入映射到Minecraft的控制动作空间。训练损失结合行为克隆损失和与教师模型VPT的KL散度损失。

3. **MGOA数据集构建**: 自动化流水线生成25,000个视频、约30M目标-观察-动作对齐数据，覆盖8个原子任务。使用STEVE-1执行GPT-4生成的指令，记录成功轨迹，过滤失败和超时的数据。整个过程无需人工标注，可并行化快速生成。两阶段训练：先行为预训练对齐行为编码器，再动作微调将语言空间转化为动作空间。

## 实验关键数据

- **原子任务**: GOAP在Logs/Seeds/Dirt/Stone四个任务上平均奖励19.0，超越GROOT(15.1)和STEVE-1(7.3)，提升约27%
- **长程任务**: Optimus-2在全部7个任务组上达到最高成功率，Diamond Group 13%、Redstone Group 28%，整体平均提升10%
- **开放指令任务**: GOAP在Golden Shovel(13%)、Diamond Pickaxe(16%)、Compass(17%)上取得成功，而现有策略全部失败(0%)
- **消融**: 去除Causal Perceiver性能下降47.4%，去除History Aggregator+Memory Bank下降44.2%
- **LLM backbone**: 将LLM替换为Transformer-XL后，开放指令任务性能显著下降，验证了MLLM语言理解能力的必要性
- **训练数据**: 仅用OpenAI Contractor数据集训练的Stone任务性能比混合数据集低89%

### 关键发现

- Causal Perceiver的动作引导使行为表示能清晰区分不同任务（t-SNE可视化中四个任务形成明确聚类），而ViT和MineCLIP的表示高度混淆
- VPT作为动作头显著优于2层MLP（因为VPT的大规模游戏数据预训练提供了领域知识）
- MGOA数据集的高质量对齐数据是性能提升的关键——MGOA与OpenAI Contractor Dataset混合训练效果最佳

## 亮点与洞察

- **观察-动作因果建模的洞察深刻**: 现有方法忽略了一个直觉上显而易见的事实——当前观察是前一动作的结果。将这种因果关系显式编码后，行为表示的区分度大幅提升
- **MLLM做策略的开创性**: 首次将MLLM作为Minecraft策略的核心架构（而非仅用于规划），释放了其开放式语言理解能力，使策略首次能处理"I need some iron ores, what should I do?"这样的开放指令
- **行为token的压缩设计**: 用固定长度的行为token加记忆库来表示任意长的历史序列，既保留了长期依赖又控制了计算开销，是对长视频建模的优雅解决方案
- **自动化数据流水线**: 用现有Agent生成训练数据的"自举"思路，低成本高效率，适合快速扩展

## 局限与展望

- 缺少开放式任务（如"建造房屋"、"击败末影龙"）的高质量训练数据，限制了复杂创造性任务的执行能力
- 仅在Minecraft平台验证，未扩展到其他模拟平台（如AI2-THOR、Habitat）或真实世界机器人
- 规划器依赖GPT-4V（闭源模型），增加了使用成本和可复现性限制
- GOAP的基础模型DeepSeek-VL-1.3B参数量较小，扩展到更大的MLLM是否能进一步提升尚未探索
- 数据生成依赖现有策略（STEVE-1）的能力，对于STEVE-1无法完成的任务无法生成训练数据
- 训练需要8张L40 GPU约2天，对计算资源有一定要求
