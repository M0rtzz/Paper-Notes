---
title: >-
  [论文解读] Sparse Imagination for Efficient Visual World Model Planning
description: >-
  [ICLR 2026][机器人][world model] 提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中通过随机丢弃 token 和随机分组注意力训练实现大幅推理加速（50% 丢弃率可减少约 50% 规划时间）…
tags:
  - "ICLR 2026"
  - "机器人"
  - "world model"
  - "sparse tokens"
  - "MPC"
  - "DINO"
  - "VLA"
  - "token dropout"
  - "planning efficiency"
---

# Sparse Imagination for Efficient Visual World Model Planning

**会议**: ICLR 2026  
**arXiv**: [2506.01392](https://arxiv.org/abs/2506.01392)  
**代码**: 无（基于 DINO-WM 框架）  
**领域**: 机器人  
**关键词**: world model, sparse tokens, MPC, DINO, VLA, token dropout, planning efficiency

## 一句话总结
提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中通过随机丢弃 token 和随机分组注意力训练实现大幅推理加速（50% 丢弃率可减少约 50% 规划时间），同时保持甚至在某些任务上超越全量 token 的规划性能。关键发现是简单随机丢弃优于复杂的 token 选择方法，原因是静态重要性排序在动态规划场景中存在"盲点问题"。

## 背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：1. 基于世界模型的规划（MPC）通过想象未来轨迹实现决策，但计算代价随 token 数量二次增长——每个规划步需要 $K \times M \times H$ 次世界模型前向传播
2. ViT patch token 作为视觉状态表示（如 DINO-WM）比单一 CLS token 保留更丰富的空间信息，在精细操作任务中优势明显
3. 但全量 patch token 在 MPC 中的计算开销使实时部署极其困难——尤其在机器人等计算受限的场景
4. ViT 表示存在已知的冗余性——Raghu et al., Pan et al., Kim et al. 等多项研究证明并非所有 patch 对下游任务同等重要
5. 机器人场景下计算资源尤其受限（嵌入式GPU），需要在保持精度的同时大幅降低推理开销
6. 现有 token 剪枝方法（注意力排序/学习选择/合并/训练时 dropout）在分类等静态任务上有效，但在规划这种迭代动态场景中未被验证

## 方法（框架/设计）

### 整体框架

世界模型以固定权重的预训练 DINO 编码器把每帧图像编码成 patch token $z_t \in \mathbb{R}^{H_p \times W_p \times D}$，再由一个因果 Transformer 解码器在 token 空间预测未来状态序列，训练目标是逐 token 的 MSE 预测损失 $\mathcal{L}_{wm} = \frac{1}{N}\sum_{i=1}^N \|\hat{z}_{t+1,i} - z_{t+1,i}\|^2$。规划时用 MPC（CEM 或 VLA 引导采样）在想象出的轨迹上择优动作，而本文的核心改造是让这套想象只在一小撮随机保留的 token 上进行，从而把每步 $K \times M \times H$ 次前向的开销直接砍掉一半。

### 关键设计

**1. Sparse Imagination：用随机丢弃直接削掉一半想象开销。** MPC 的计算瓶颈在于每个规划步都要对 $K$ 条候选轨迹、$M$ 个 CEM 样本、$H$ 步时域反复跑世界模型，而开销随 token 数二次增长。本文不做任何复杂的 token 选择，只是在推理阶段以比例 $p$ 随机丢弃 patch token，仅保留 $(1-p)N$ 个 token 喂进世界模型预测，$p=0.5$ 时规划时间几乎对半下降。关键发现是这种朴素随机采样反而优于注意力排序、学习排序等"聪明"方法：静态重要性度量在 MPC 的迭代优化里存在"盲点"——某个 patch 在当前状态看似无关紧要，但在评估某条候选动作序列时却可能变得关键；每步重新独立采样 mask 的无偏覆盖，恰好避免了静态排序的系统性遗漏，这也是为何 10–50% 是丢弃率的甜蜜区间、超过 70% 才明显退化。

**2. 随机分组注意力训练：让模型学会在任意 token 子集上预测。** 如果世界模型只见过全量 token，推理时突然抽走一半会严重失配（消融显示 PushT 直接从 70% 掉到 35%）。为此训练时把每帧 token 随机切成两组，用注意力掩码限制交互只发生在组内、时间维度上各组保持对齐，等于让模型反复在残缺视野下学习预测动力学。这样推理阶段无论丢掉哪些 token、丢多少，模型都能稳定外推，分组注意力因此是稀疏想象能成立的必要前提而非可选项。

**3. VLA 引导规划：把长时程的候选采样交给预训练策略。** 对长时程任务，CEM 在动作空间里盲目随机采样既慢又难命中有效轨迹。这里改为从预训练的 VLA 策略（SmolVLA）采样 $K$ 条候选动作序列来替代 CEM 的随机采样，再让稀疏世界模型快速评估打分。VLA 提供的动作先验把候选集中到合理区域，与稀疏想象的廉价评估叠加后，长时程任务上既提了约 4–7% 成功率又省了约 40% 计算。

## 实验关键数据

### 简单环境（MPC-CEM / CEM）


### 主实验

| 环境 | Full (p=0) | Drop 30% | Drop 50% | CLS-token | 说明 |
|------|-----------|---------|---------|-----------|------|
| Pointmaze | 98.3% | 98.3% | **100%** | 96.7% | 稀疏反超全量 |
| Wall | 91.7% | 93.3% | 95.0% | 85.0% | 稀疏优于全量 |
| PushT | 75.0% | 61.7% | 70.0% | 43.3% | 50% drop 接近全量 |
| Granular | 75.0% | **85.0%** | 60.0% | 20.0% | 30% drop 反超 |
| Rope | 63.3% | 70.0% | 73.3% | 36.7% | 稀疏显著优于 CLS |
| Block Push | 22.0% | 18.0% | 20.0% | 16.0% | 困难任务差距较小 |

### 复杂环境 + 真实世界（VLA 引导规划）


### 消融实验

| 任务 | Full | Drop 50% | VLA-only | 时间(Full→Drop) |
|------|------|---------|---------|----------------|
| PickPlace (真实) | - | **80%** | 60% | 19.1s→10.4s |
| Drawer (真实) | - | **70%** | 60% | 14.0s→10.6s |
| LIBERO-10 | 34% | 33% | 29% | 53.4s→29.7s |
| Meta-World | 48.8% | 47.7% | 42.7% | 3.63s→2.37s |

### 规划时间加速

| 环境 | Full 时间 | Drop 50% 时间 | 加速比 |
|------|----------|-------------|--------|
| PushT | 173s/iter | 82s/iter | **52.6%** |
| Pointmaze | 184s/iter | 102s/iter | **44.6%** |
| Block Push | 297s/iter | 161s/iter | **45.8%** |

## 亮点与洞察
- 极其简洁优雅：仅通过随机 dropout 即实现大幅加速，无需额外模型
- "盲点问题"分析深刻——解释了为何复杂 token 选择不如随机采样
- 通用性强：从简单轨迹优化到 VLA 引导规划到真实机器人均验证有效
- 训练阶段的分组注意力策略可无缝嵌入任何 Transformer 世界模型

## 消融实验与深入分析

| 消融/分析 | 结果 |
|-----------|------|
| 有 vs 无分组注意力训练 | 无分组注意力在 50% drop 时严重退化（PushT 从 70→35%），分组注意力是必要条件 |
| 随机 vs 注意力排序 vs 学习排序 | 随机采样在多数任务上竞争性或更优——"盲点问题"使静态排序失效 |
| Drop ratio 甜蜜点 | 10-50% 为最佳区间，>70% 明显退化 |
| VLA 引导 vs CEM 随机采样 | 长时程任务中 VLA 引导提升 ~4-7%，计算开销降低 ~40% |
| 仅训练阶段稀疏 vs 仅推理阶段稀疏 | 两者都需要：训练稀疏确保模型适应，推理稀疏提供加速 |

### "盲点问题"深入分析
- 静态重要性度量（如注意力权重、CLS token 相关性）在 MPC 的迭代优化过程中会产生系统性盲点
- 具体地：某些 patch 在当前状态下看似不重要，但在对候选动作序列评估时可能变得关键
- 随机采样通过无偏覆盖避免了系统性遗漏——每次迭代重新采样 mask 确保所有区域都有被覆盖的概率
- 这一发现与 token 剪枝文献中"学习选择优于随机"的常见结论相反，说明规划场景有其特殊性

## 局限与展望
- 最佳 drop ratio 需要根据任务手动选择，缺乏自适应机制——一个可能的改进是根据任务复杂度或当前状态动态调整
- 分组数固定为 2，未探索更多分组（如 3-4 组）的效果
- 依赖 DINO 特征的冗余性假设，对信息密集场景（如文本密集界面）可能不成立
- 真实世界验证仅限于两个较简单任务（PickPlace + Drawer），更复杂的操作任务未测试
- 未与 token 合并方法（如 ToMe）结合——稀疏选择+合并可能进一步提升效率

## 相关工作与启发
- **vs Dreamer 系列 (Hafner et al.)**：Dreamer 在低维向量潜在空间想象，本文在高维 patch token 空间想象——保留了更丰富的空间信息但计算更贵，稀疏想象正好弥补这一差距
- **vs DINO-WM (Zhou et al. 2024)**：本文直接构建在 DINO-WM 之上，用 sparse imagination 解决其计算瓶颈
- **vs ToMe (Bolya et al.)**：ToMe 通过 token 合并减少计算，本文通过 token 丢弃——设计更简单且不需要额外的合并逻辑
- **vs SmolVLA (Shukor et al.)**：SmolVLA 提供预训练策略用于引导规划，本文的稀疏想象加速了 VLA 引导的世界模型评估
- **启发**：稀疏想象的思路可推广到其他需要大量前向传播的场景——如 MCTS 搜索中的价值网络评估、多步推理中的 world simulation

## 评分
- 新颖性: ⭐⭐⭐⭐ 简单但有效的洞察，盲点问题分析有独特价值
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个仿真+2 个真实任务，多方法对比，消融充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表精美，方法图示直观
- 价值: ⭐⭐⭐⭐ 实用贡献，可直接集成到任何基于 Transformer 的世界模型流水线中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Visual Planning: Let's Think Only with Images](visual_planning_lets_think_only_with_images.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](../../CVPR2026/robotics/chain_of_world_world_model_thinking_in_latent_motion.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
