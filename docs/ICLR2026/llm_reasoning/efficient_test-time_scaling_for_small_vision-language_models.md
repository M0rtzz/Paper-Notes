---
title: >-
  [论文解读] Efficient Test-Time Scaling for Small Vision-Language Models
description: >-
  [ICLR 2026][LLM推理][test-time scaling] 为小型 VLM 提出两种高效的测试时缩放策略：TTAug（对输入做多种增强后在 token 级别聚合输出概率）和 TTAdapt（用 TTAug 生成的伪标签自适应调整模型参数），在 9 个基准上一致提升性能，同时计算效率远优于现有的基于重复采样的测试时方法。
tags:
  - ICLR 2026
  - LLM推理
  - test-time scaling
  - 视觉语言
  - test-time augmentation
  - test-time adaptation
  - token-level aggregation
---

# Efficient Test-Time Scaling for Small Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.03574](https://arxiv.org/abs/2510.03574)  
**代码**: [GitHub](https://github.com/monurcan/efficient_test_time_scaling)  
**领域**: LLM推理 / VLM效率  
**关键词**: test-time scaling, vision-language models, test-time augmentation, test-time adaptation, token-level aggregation

## 一句话总结

为小型 VLM 提出两种高效的测试时缩放策略：TTAug（对输入做多种增强后在 token 级别聚合输出概率）和 TTAdapt（用 TTAug 生成的伪标签自适应调整模型参数），在 9 个基准上一致提升性能，同时计算效率远优于现有的基于重复采样的测试时方法。

## 研究背景与动机

小型视觉语言模型（如 SmolVLM2-2.2B）提供了计算高效的替代方案，但泛化能力和下游任务性能弱于大型模型。测试时缩放（Test-Time Scaling, TTS）技术可以在推理时投入更多计算来弥补模型能力不足，但现有方法存在根本矛盾：

**Self-Consistency**: 通过温度采样生成多个候选答案，然后用多数投票聚合。但采样多次的计算开销大，且只在最终答案层面聚合，丢失了 token 级别的细粒度信息。

**Self-Selector / Sample-and-Rank**: 用模型自身或 log probability 选择最佳回答，但仍依赖多次独立采样。

**Self-Synthesizer**: 让模型综合多个回答生成最终答案，额外增加了综合步骤的开销。

核心问题：**现有 TTS 方法的计算开销与小模型的资源节约设计目标相矛盾。**

本文提出的关键洞察是两个设计选择可以同时提升效果和效率：
- **用输入增强替代温度采样来诱导多样性**：语义保持的增强比温度采样能产生更高质量的多样候选
- **在 token 级别而非答案级别聚合**：捕获更细粒度的置信度信号

## 方法详解

### 整体框架

两条互补的 pipeline：

**Pipeline 1 - TTAug（测试时增强）**: 
输入图像 + 文本 prompt → 生成多组增强版本 → VLM 分别处理每组 → 在每个解码步骤聚合 next-token 概率分布 → 贪婪解码生成最终回答

**Pipeline 2 - TTAdapt（测试时自适应）**: 
TTAug 生成伪标签 → 用伪标签微调 VLM 参数 → 重复此过程直至收敛或达到预算

### 关键设计

1. **双通道输入增强**: 

    - **图像增强**: 对输入图像施加多种语义保持的变换（如轻微旋转、裁剪、色彩抖动等），生成多个视觉变体
    - **文本增强**: 对文本 prompt 进行轻微扰动——注入拼写错误、分词噪声等（如 "Which country" → "Wh ich cou ntry"），同时附加原始完整问题作为参照（"In other words, ..."）
    - 设计动机：图像增强提供视角多样性，文本增强迫使模型关注核心语义而非表面形式

2. **Token 级别聚合（Key Insight #2）**: 

    - 在自回归解码的每一步，收集所有增强输入对应的 next-token 概率分布
    - 对这些分布做简单平均（或加权平均）后，贪婪选择概率最高的 token
    - 相比答案级别聚合（如多数投票），token 级别聚合能捕获局部置信度信号
    - 例如：10 个增强中有 6 个在某位置预测 "Germany"、3 个预测 "France"、1 个预测 "UK"，token 聚合能利用这个 60% 的一致性信号

3. **共识伪标签自适应（TTAdapt）**: 

    - 使用 TTAug 的输出作为高质量伪标签
    - 对模型参数进行轻量级适应（可能通过 LoRA 或直接微调少量层）
    - 适应后的模型可以再次运行 TTAug，形成迭代优化
    - 相比无参数更新的 TTAug，TTAdapt 能进一步内化增强带来的纠正信号

4. **增强多样性 vs 温度采样（Key Insight #1）**: 

    - 实验证明，输入扰动（augmentation）产生的答案多样性比温度采样更高质量
    - 温度采样只改变了解码的随机性，而输入增强改变了模型接收到的信息，产生更本质的视角差异
    - 在 Self-Consistency 和 Self-Selector 两种 TTS 策略下，增强均优于温度采样

### 损失函数 / 训练策略

TTAug 本身**无需训练**，是纯推理时方法。

TTAdapt 的适应使用：
- 伪标签来自 TTAug 的共识输出
- 标准的 next-token 交叉熵损失
- 轻量级参数更新，避免过拟合到单个测试样本

## 实验关键数据

### 主实验

在 9 个基准上评估（主要使用 SmolVLM2-2.2B）：
- VQA 类：GQA、TextVQA、OCRVQA
- 多选/判断类：AI2D、MME-RealWorld、AMBER
- 图表理解：ChartQA、OCRBench
- 图像描述：COCO Captions

蜘蛛图结果显示 TTAug 和 TTAdapt 在所有 9 个基准上均产生一致提升，TTAdapt > TTAug > Baseline。

### 与其他 TTS 方法对比

| 方法 | 多样性来源 | 聚合层级 | 效率 |
|------|----------|---------|------|
| Self-Consistency | 温度采样 | 答案级 | 低（多次完整解码）|
| Self-Selector | 温度采样 | 选择 | 低 |
| Sample-and-Rank | 温度采样 | 选择 | 低 |
| Self-Synthesizer | 温度采样 | LLM综合 | 最低 |
| **TTAug (本文)** | **输入增强** | **Token级** | **高** |

### 缩放行为

- 随增强数量增加，平均性能呈单调递增并趋于饱和
- 少量增强（如 4-8 组）即可获得大部分收益

### 跨模型泛化

虽然超参数针对 SmolVLM2-2.2B 优化，但在不同模型（不同规模和架构）上均有一致提升，证明了方法的通用性。

### 消融实验

| 配置 | 说明 |
|------|------|
| 聚合策略选择 | Token级 > 答案级 |
| 聚合层选择 | 探索最佳聚合位置 |
| 增强技术组合 | 图像+文本双增强最优 |
| 适应目标 | 不同 TTAdapt 损失的比较 |

### 定性结果

多个案例展示了增强+聚合的纠错能力：
- ChartQA：基线回答 "France" → TTAug/TTAdapt 正确回答 "Germany"
- OCRBench：基线回答 "100.00" → TTAug/TTAdapt 正确回答 "71.10"
- OCRVQA：基线回答 "Brushy" → TTAug/TTAdapt 正确回答 "Brush Dance"
- GQA：基线回答 "Blinds" → TTAug/TTAdapt 正确回答 "Desk"

## 亮点与洞察

1. **两个关键洞察构成核心贡献**: 增强 > 温度采样 + token级 > 答案级，这两个发现可以独立应用于其他 TTS 方法
2. **文本增强的巧妙设计**: 引入拼写噪声+原文参照的方式既简单又有效，迫使模型学会忽略表面噪声、关注语义
3. **TTAdapt 的迭代机制**: 用自己的共识输出作为伪标签来自适应，形成正反馈循环
4. **面向实际部署**: 明确以小模型为目标场景，效率约束贯穿始终
5. **系统性分析**: 提供了全面的消融、缩放行为、跨模型泛化等分析

## 局限与展望

1. **增强数量与延迟的权衡**: 虽然比采样类方法高效，但仍需处理多个增强输入，延迟会线性增长
2. **文本增强的通用性**: 拼写噪声注入对某些任务（如代码生成）可能不适用
3. **TTAdapt 的过拟合风险**: 在单个测试样本上适应参数可能导致过拟合，特别是在分布外数据上
4. **超参数转移**: 虽然展示了跨模型泛化，但承认"专用调参效果更好"
5. **评估局限**: 主要在判别式任务上评估，开放式生成任务的效果有待验证

## 相关工作与启发

- **Test-Time Training（TTT）**: 经典的测试时自适应方法，本文在 VLM 场景重新设计
- **Self-Consistency（Wang et al.）**: 多数投票 TTS 的开创性工作，本文在多样性来源和聚合粒度两方面改进
- **Test-Time Augmentation（传统CV）**: TTA 在传统 CV 中广泛使用，本文将其扩展到 VLM 的自回归生成
- **Token Merging / Pruning**: 效率优化的另一方向，可以与 TTAug 互补
- 对 VLM 推理效率的研究有启发：不仅可以从模型压缩角度省算力，也可以从更聪明的推理策略角度提升性能

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](../../ACL2025/llm_reasoning/revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)

<!-- RELATED:END -->
