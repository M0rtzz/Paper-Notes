---
title: >-
  [论文解读] When to Think and When to Look: Uncertainty-Guided Lookback
description: >-
  [CVPR 2026][多模态][视觉推理] 本文首次系统分析了 LVLM 中 test-time thinking 对视觉推理的影响，发现"多想不如多看"——长推理链常忽略图像导致"long-wrong"轨迹，并据此提出不确定性引导的 lookback 解码策略，通过在推理链漂移时注入视觉回看提示，在不修改模型的前提下将 MMMU 等 6 个基准提升 2-6 个点。
tags:
  - CVPR 2026
  - 多模态
  - 视觉推理
  - 链式思维
  - 大视觉语言模型
  - 自适应解码
  - 不确定性引导
---

# When to Think and When to Look: Uncertainty-Guided Lookback

**会议**: CVPR 2026  
**arXiv**: [2511.15613](https://arxiv.org/abs/2511.15613)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 视觉推理, 链式思维, 大视觉语言模型, 自适应解码, 不确定性引导

## 一句话总结
本文首次系统分析了 LVLM 中 test-time thinking 对视觉推理的影响，发现"多想不如多看"——长推理链常忽略图像导致"long-wrong"轨迹，并据此提出不确定性引导的 lookback 解码策略，通过在推理链漂移时注入视觉回看提示，在不修改模型的前提下将 MMMU 等 6 个基准提升 2-6 个点。

## 研究背景与动机

1. **领域现状**：Test-time thinking（推理时生成显式思维链）在 LLM 上已展现强大效果。InternVL3.5 和 Qwen3-VL 等最新 LVLM 家族也开始提供 thinking 模式（如 `<think>` token），在 MMMU 等基准上报告了 SOTA 结果。

2. **现有痛点**：虽然 thinking 模式总体上有帮助，但实际上没有人系统研究过它在视觉推理中到底何时有效、何时有害。实践中经常出现"long-wrong"现象：模型生成了很长的推理链但答案错误，因为链条中的推理逐渐偏离图像内容，堕入纯文本臆想。

3. **核心矛盾**：thinking 模式对推理密集的 STEM 类问题确实有效，但对需要视觉识别/检索的文学、历史、艺术等类别反而有害——因为冗长的推理链引入了噪声而非有用的推理步骤。更深层的矛盾是：现有 thinking 模式对所有问题一视同仁地"深度思考"，缺乏自适应控制。

4. **本文目标** (a) thinking 何时有益于视觉推理？(b) 如何权衡推理的广度（采样次数）与深度（thinking 模式）？(c) 能否自适应控制 thinking 以获得更好的视觉感知？

5. **切入角度**：通过 token 级别的 perplexity 对比实验（有图 vs 噪声图 vs 无图），发现正确答案的推理轨迹中存在频繁的"lookback"短语（显式回看图像），而错误轨迹则缺乏这种视觉锚定。据此挖掘两类短语：暂停/不确定短语（指示漂移）和 lookback 短语（重新锚定图像）。

6. **核心 idea**：在推理链出现不确定性信号时自动注入视觉回看提示，将"盲目深度思考"转化为"按需回看图像"。

## 方法详解

### 整体框架
分为两个阶段：**分析阶段**（离线）——通过 token 级视觉敏感性探针分析推理轨迹，挖掘暂停短语集 $\mathcal{P}$ 和 lookback 模板集 $\mathcal{L}$；**推理阶段**（在线）——在自回归解码过程中实时检测暂停短语并注入 lookback 提示，可选地配合并行采样选择最佳视觉锚定分支。

### 关键设计

1. **Token 级视觉敏感性探针**:

    - 功能：量化每个推理步骤对图像内容的依赖程度
    - 核心思路：对每个 token $s$，在三种视觉上下文下计算 perplexity：真实图像 $c=R$、噪声图像 $c=N$、无图像 $c=\varnothing$。定义两个差分指标：**内容对比** $\Delta_{content}(s) = PPL_R(s) - PPL_N(s)$——正确图像对预测的帮助程度；**存在对比** $\Delta_{presence}(s) = PPL_N(s) - PPL_\varnothing(s)$——有无图像对预测的影响。$|\Delta_{presence}|$ 大但 $|\Delta_{content}|$ 小的步骤表示模型知道要看图但没真正利用图像内容——这就是不确定/漂移的信号。$\Delta_{content}$ 高度负值的步骤则表示模型确实在利用图像信息推理——这些步骤中出现的短语就是 lookback 模板。
    - 设计动机：用噪声图而非不相关真实图作为控制条件，避免模型将无关图像的语义内容错误整合到推理中，保证探针的纯净性。

2. **Lookback-When-Uncertain 解码控制器**:

    - 功能：在推理时自适应注入视觉回看提示
    - 核心思路：在自回归流式解码中，检查最近生成的 $L$ 个 token 的后缀是否匹配暂停短语集 $\mathcal{P}$ 中的 n-gram。如果匹配，且模型仍在 thinking 阶段（未进入最终答案段），且最近 $L$ 个 token 内未触发过 lookback，则立即拼接一个 lookback 短语 $\ell \in \mathcal{L}$（如"Looking back at the image, …"）。所有重计算（perplexity 估计、短语挖掘）都在离线完成，推理时仅需高效的 n-gram 匹配，开销极小。
    - 设计动机：频繁提到"hmm""wait"等词的位置恰好是模型推理不确定的区域，此时注入视觉回看可以防止推理链进一步漂移。限制触发频率和禁止在答案阶段触发避免了退化。

3. **并行 Lookback 采样**:

    - 功能：在 lookback 触发点探索多个视觉锚定的推理分支，选出最佳
    - 核心思路：当 lookback 被触发时，在注入 $\ell$ 后并行采样 $M$ 个长度为 $H$ 的续写。对每个分支计算视觉有用性得分 $\mathcal{V}^{(m)} = -\frac{1}{H}\sum_{t=s}^{s+H-1}\Delta_{content}^{(m)}(t)$，选择 $\mathcal{V}$ 最大的分支继续解码。因为 lookback 事件稀疏且局部化，额外 token 开销很小。
    - 设计动机：仅靠 lookback 提示不能保证后续推理一定锚定于图像，通过并行采样+视觉有用性评分，确保至少有一个分支紧密依赖图像内容。小模型特别受益于此——通过探索多条视觉锚定路径增强鲁棒性。

### 损失函数 / 训练策略
完全 training-free 方法。离线阶段在 MMMUval 上用 10 次采样和三种视觉条件做 perplexity 估计来挖掘短语集，推理时无需额外训练。

## 实验关键数据

### 主实验（MMMU + 5 个额外基准）

| 模型 | 方法 | MMMU Pass@1 | Token使用% | MMBench | MMStar | MathVista | MathVision | MathVerse |
|------|------|-----------|-----------|---------|--------|-----------|------------|-----------|
| Qwen3-VL-4B | Original | 67.0 | 100 | 86.7 | 73.2 | 79.5 | 60.0 | 75.2 |
| | Ours (lookback) | **69.7**(+2.7) | 57.2 | **89.5**(+2.8) | **75.0**(+1.8) | **84.3**(+4.8) | **64.2**(+4.2) | **77.2**(+2.0) |
| | Ours (+sampling) | **73.0**(+6.0) | 59.5 | 88.2(+1.5) | **75.7**(+2.5) | **85.0**(+5.5) | **65.5**(+5.5) | **78.7**(+3.5) |
| Qwen3-VL-8B | Original | 70.3 | 100 | 87.5 | 75.3 | 77.2 | 62.7 | 77.7 |
| | Ours (lookback) | **73.0**(+2.7) | 62.1 | 88.7(+1.2) | **78.5**(+3.2) | 79.4(+2.2) | **67.9**(+5.2) | 78.9(+1.2) |
| | Ours (+sampling) | **74.2**(+3.9) | 63.0 | **89.8**(+2.3) | **79.6**(+4.3) | 79.7(+2.5) | **68.3**(+5.6) | 79.9(+2.2) |
| Qwen3-VL-32B | Original | 75.3 | 100 | 90.8 | 79.4 | 83.8 | 70.2 | 82.6 |
| | Ours (lookback) | **81.7**(+6.4) | 66.2 | **93.6**(+2.8) | **81.2**(+1.8) | **85.6**(+1.8) | **72.0**(+1.8) | **84.4**(+1.8) |
| | Ours (+sampling) | **79.2**(+3.9) | 70.3 | **93.9**(+3.1) | **82.5**(+3.1) | **85.9**(+2.1) | **73.3**(+3.1) | **84.7**(+2.1) |

### 基线对比（MMMU Qwen3-VL-4B）

| 方法 | MMMU Pass@1 | Token使用% |
|------|-----------|-----------|
| Original Thinking | 67.0 | 100 |
| DEER | 53.3 | 40.0 |
| DeepConf | 63.3 | 76.7 |
| REFRAIN | 63.3 | 73.3 |
| **Ours (lookback)** | **69.7** | **57.2** |
| **Ours (+sampling)** | **73.0** | **59.5** |

### 关键发现
- **Thinking 不总是有益**：识别类任务（文学、历史、艺术）中 thinking 反而引入噪声，不如简洁的 instruct 模式
- **广度 vs 深度权衡**：增加采样次数（pass@k）的收益在 k≥8 后迅速递减；thinking 模式提升每次采样质量但边际递减
- **容量决定推理效率**：32B 模型的正确推理轨迹比 4B 模型更短，说明更强的模型推理更高效
- **Lookback 短语自然富集于正确轨迹**：大规模统计验证了"回看图像"行为与视觉推理成功强相关
- **周期性注入无效**：定期插入 lookback（n=1...5）均不如不确定性引导触发，说明插入位置至关重要
- **方法跨家族迁移**：在 InternVL3.5-Think 上也有一致性提升（4B +1.5, 8B +3.3 on MMMU）

## 亮点与洞察
- **"Long-wrong" vs "Quiet-wrong" 的二分法非常有洞察力**：前者是推理链太长导致漂移，后者是模型容量不足无法启动有效推理。不同错误模式需要不同的干预策略
- **用 perplexity 对比作为视觉锚定探针**：三种视觉条件（真实图/噪声/无图）的 perplexity 差异提供了一个无需标注的自动化方法来量化推理链中每个 token 的视觉依赖程度。这个方法可直接迁移到其他多模态推理任务
- **Training-free 且兼容流式解码**：离线挖掘短语、在线做 n-gram 匹配，不需要在推理时计算 perplexity，实际部署开销极小。对闭源模型仅需其支持 log-prob 访问
- **在使用更少 token（减少 35-45%）的情况下取得更高准确率**，真正推动了 Pareto 前沿

## 局限与展望
- 探针构建和短语挖掘需要 token 级 log-probability，对不提供 log-prob 的闭源模型不适用
- 分析主要基于 MMMU，对其他格式的视觉推理任务（如 VQA、图像描述）的适用性待验证
- Lookback 短语是从特定模型家族挖掘的，不同模型的触发词可能不同
- 并行采样的视觉有用性评分仍需在线计算 perplexity（只是在 lookback 触发的稀疏位置），存在一定延迟
- 未探索将此策略与强化学习训练的 thinking 模型结合的可能性

## 相关工作与启发
- **vs DEER/DeepConf/REFRAIN**: 这些都是文本领域的自适应 CoT 方法，提供早退或信心评估。但它们忽略了视觉模态的特殊性——不确定性信号应该同时考虑视觉锚定程度。本文方法在 MMMU 上全面超越这些基线
- **vs VCoT/Visual Sketchpad**: 这些方法通过让模型画图来增强视觉推理，需要额外监督或工具。本文方法完全 training-free 且不需要外部工具
- **vs 自一致性 self-consistency**: 多次采样+投票只利用了广度。本文的关键洞察是：在正确的位置注入 lookback 比单纯增加采样更有效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统分析 LVLM thinking 的视觉影响，提出的 lookback 策略思路新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个模型变体、10 次采样、30 个类别细粒度分析、6 个基准测试、充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 分析→洞察→方法→验证的逻辑链非常完整，图表丰富且信息量大
- 价值: ⭐⭐⭐⭐⭐ Training-free 方法在多个基准上一致提升，对 LVLM 推理范式有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](../../NeurIPS2025/multimodal_vlm/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)
- [When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](../../NeurIPS2025/multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)

<!-- RELATED:END -->
