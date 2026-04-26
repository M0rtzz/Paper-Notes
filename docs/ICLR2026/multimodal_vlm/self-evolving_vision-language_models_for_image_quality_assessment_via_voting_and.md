---
title: >-
  [论文解读] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking
description: >-
  [ICLR 2026][多模态][VLM] 提出 EvoQuality 框架，通过成对多数投票生成伪排序标签、结合 GRPO 自迭代优化，使 VLM 在无人工标注下自主提升图像质量感知能力，零样本性能提升 31.8% PLCC，在 7 个 IQA 基准中 5 个超越有监督 SOTA。
tags:
  - ICLR 2026
  - 多模态
  - VLM
  - 图像质量评估
  - 自监督
  - GRPO
  - 投票排序
---

# Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking

**会议**: ICLR 2026  
**arXiv**: [2509.25787](https://arxiv.org/abs/2509.25787)  
**代码**: 无  
**领域**: 多模态视觉语言模型 / 图像质量评估  
**关键词**: VLM, 图像质量评估, 自监督, GRPO, 投票排序

## 一句话总结

提出 EvoQuality 框架，通过成对多数投票生成伪排序标签、结合 GRPO 自迭代优化，使 VLM 在无人工标注下自主提升图像质量感知能力，零样本性能提升 31.8% PLCC，在 7 个 IQA 基准中 5 个超越有监督 SOTA。

## 研究背景与动机

图像质量评估（IQA）是计算机视觉中的经典任务，目标是自动评估图像的感知质量。近年来，视觉语言模型（VLM）在多项视觉任务中展现了强大的能力，但将其应用于 IQA 仍然面临两大挑战：

**标注成本高昂**：传统的 VLM 后训练方法（如 SFT 或 RLHF）依赖大量人工标注的质量评分数据，采集成本极高，且主观性强、一致性差。

**自监督在感知领域的空白**：虽然自监督技术在推理能力增强方面（如数学推理）已被验证有效，但在感知类任务（如质量判断）中的应用几乎为空白。IQA 不同于推理任务——它没有唯一正确答案，而是依赖于相对比较。

**已有方法的局限**：现有的 VLM-based IQA 模型大多需要在有标注的 IQA 数据集上进行监督微调，泛化能力有限，且对未见数据集的迁移效果不佳。

本文的核心动机是：**能否让 VLM 自主地、无需任何人工标签地提升其图像质量感知能力？** 作者观察到，虽然 VLM 单次质量判断可能不准确，但通过多次成对比较的"群体智慧"可以涌现出可靠的质量排序。

## 方法详解

### 整体框架

EvoQuality 是一个自监督的迭代优化框架，核心流程为：
1. 利用 VLM 对图像对进行多次成对质量比较
2. 通过多数投票建立共识排序（伪标签）
3. 将排序结果转化为保真度奖励信号
4. 使用 GRPO（Group Relative Policy Optimization）优化 VLM
5. 迭代重复上述过程，VLM 逐步自我进化

### 关键设计

1. **成对多数投票（Pairwise Majority Voting）**:

    - 对于一组图像，VLM 对每一对图像执行多次（如 K 次）"哪张质量更好？"的比较
    - 对每一对的 K 次输出取多数投票，确定该对的质量偏好
    - 综合所有图像对的投票结果，通过排序算法（如 Bradley-Terry 模型或简单计数）得到全局质量排序
    - **设计动机**：利用 self-consistency 原则——单次采样可能不可靠，但多次采样的一致性方向反映了模型的真实倾向

2. **保真度奖励构建（Fidelity Reward）**:

    - 将投票产生的伪排序转化为奖励信号
    - 如果 VLM 的新输出与伪排序一致，则获得正奖励；不一致则获得负奖励
    - 奖励的强度与投票一致性成正比——投票越一致的对（即模型越确信的判断），奖励权重越大
    - **设计动机**：高一致性的投票结果更可靠，应赋予更高权重；低一致性的结果可能是困难样本或噪声，应降低权重

3. **GRPO 优化**:

    - 采用 Group Relative Policy Optimization 作为强化学习优化器
    - GRPO 无需额外的 critic/value 网络，直接使用组内相对奖励来更新策略
    - 每轮迭代中，从当前 VLM 采样多个响应，根据保真度奖励计算相对优势，更新模型参数
    - **设计动机**：GRPO 相比 PPO 更轻量，适合大模型的迭代优化场景

4. **迭代自进化机制**:

    - 每轮优化后，使用更新后的 VLM 重新执行成对投票，生成新的伪标签
    - 新一轮伪标签质量更高（因为 VLM 已经变好了），进一步推动模型改进
    - 形成正反馈循环：更好的模型 → 更准确的伪标签 → 更有效的训练 → 更好的模型
    - **设计动机**：借鉴自训练（self-training）的理念，但用排序而非绝对评分来避免误差累积

### 损失函数 / 训练策略

- 使用 GRPO 的策略梯度损失，核心为：
  $$L_{GRPO} = -\mathbb{E}\left[\sum_{i} A_i \log \pi_\theta(y_i | x_i)\right]$$
  其中 $A_i$ 为基于保真度奖励的相对优势
- 加入 KL 散度约束，防止模型偏离原始能力太远
- 多轮迭代训练，每轮使用新生成的伪标签

## 实验关键数据

### 主实验

EvoQuality 在 7 个主流 IQA 基准上进行了评估：

| 指标 | 本文 EvoQuality | 基础 VLM (零样本) | 提升幅度 |
|------|----------------|------------------|---------|
| PLCC (平均) | 显著提升 | 基线 | +31.8% |
| 击败有监督 SOTA | 5/7 基准 | - | - |

关键发现：
- 在 LIVE、CSIQ、TID2013、KADID-10K、SPAQ 等 5 个基准上超越了有监督的 SOTA VLM-based IQA 模型
- 在 KonIQ-10K 和 FLIVE 上接近有监督 SOTA
- 完全自监督训练，无需任何人工质量标注

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无投票直接优化 | 性能大幅下降 | 投票机制是核心 |
| 固定伪标签（单轮） | 低于迭代版本 | 迭代自进化带来持续提升 |
| 均匀奖励权重 | 低于保真度加权 | 一致性加权更有效 |
| 不同投票次数 K | K 增大初期提升，后饱和 | K=5~10 为较优选择 |

### 关键发现

1. **自监督可媲美有监督**：在 IQA 领域首次证明，自监督方法可以达到甚至超越有监督微调的性能
2. **灵活的可叠加性**：EvoQuality 可以与预训练的 IQA 模型叠加使用，进一步增强在未见数据集上的泛化能力
3. **跨数据集泛化**：无需目标数据集的任何标注，即可在多个分布外的 IQA 基准上取得 SOTA 性能

## 亮点与洞察

1. **将 self-consistency 从推理扩展到感知**：巧妙地将数学推理中"多次采样取一致性"的思想适配到了基于排序的 IQA 任务——用成对比较的多数投票替代了答案的一致性验证
2. **排序比评分更适合自监督**：绝对质量评分难以自评，但相对质量比较更容易通过投票达成共识
3. **正反馈循环设计优雅**：迭代自进化机制使得训练过程具有"自我加速"的特性
4. **实用价值高**：完全消除了 IQA 领域对人工标注的依赖，大幅降低了部署成本

## 局限与展望

1. **迭代效率**：多轮迭代+多次采样的计算开销较大，尤其是大规模 VLM
2. **投票质量依赖基础模型**：如果基础 VLM 的质量感知能力极弱，投票可能无法收敛到有意义的排序
3. **任务特异性**：框架针对排序/比较类任务设计，推广到其他感知任务（如美学评估、损伤检测）需要适配
4. **缺乏理论收敛保证**：迭代自进化的收敛性未有理论分析，实践中可能存在过拟合伪标签的风险
5. **可扩展方向**：将投票机制从成对扩展到列表级别（listwise）可能进一步提升效率

## 相关工作与启发

- **IQA 传统方法**：BRISQUE、NIQE 等手工特征方法 → 深度学习方法（DBCNN、HyperIQA）→ VLM-based 方法（Q-Align、Q-Instruct）
- **VLM 自改进**：Self-Play、Self-Rewarding LLMs 等自迭代范式的成功为本文提供了启发
- **GRPO**：DeepSeek 提出的 GRPO 优化方法，本文将其从推理任务迁移到感知任务
- **启发**：该框架的核心思想（投票→伪标签→自迭代优化）可能适用于其他"无唯一正确答案"的视觉任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 self-consistency 和 GRPO 创造性地应用于 IQA，但各组件本身不新
- **实验充分度**: ⭐⭐⭐⭐ — 7 个基准全面评估，消融充分，但缺乏与更多自监督方法的对比
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，框架直观
- **价值**: ⭐⭐⭐⭐⭐ — 消除 IQA 标注需求，实际应用价值大

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[CVPR 2026\] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](../../CVPR2026/multimodal_vlm/fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)

<!-- RELATED:END -->
