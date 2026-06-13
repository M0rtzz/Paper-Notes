---
title: >-
  [论文解读] How to Merge Your Multimodal Models Over Time?
description: >-
  [CVPR 2025][优化/理论][模型融合] 本文提出 TIME（Temporal Integration of Model Expertise）框架，系统研究了多模态专家模型随时间渐进融合的问题，通过初始化策略、部署策略和融合技术三个轴定义搜索空间，在 FoMo-in-Flux 基准上揭示了时序模型融合的关键设计原则。
tags:
  - "CVPR 2025"
  - "优化/理论"
  - "模型融合"
  - "时序模型融合"
  - "多模态学习"
  - "持续学习"
  - "基础模型"
---

# How to Merge Your Multimodal Models Over Time?

**会议**: CVPR 2025  
**arXiv**: [2412.06712](https://arxiv.org/abs/2412.06712)  
**代码**: [https://github.com/explainableml/time](https://github.com/explainableml/time)  
**领域**: 多模态VLM  
**关键词**: 模型融合, 时序模型融合, 多模态学习, 持续学习, 基础模型

## 一句话总结
本文提出 TIME（Temporal Integration of Model Expertise）框架，系统研究了多模态专家模型随时间渐进融合的问题，通过初始化策略、部署策略和融合技术三个轴定义搜索空间，在 FoMo-in-Flux 基准上揭示了时序模型融合的关键设计原则。

## 研究背景与动机

**领域现状**：模型融合（model merging）通过将多个从同一基础模型微调而来的专家模型合并为一个综合模型，在不需要额外训练的情况下获得多任务能力。现有融合方法（如 Task Arithmetic、TIES、DARE）已在静态场景下取得成功。

**现有痛点**：几乎所有现有模型融合工作都假设所有专家模型同时可用——即一次性拿到所有专家然后融合。但在现实中，新任务和新领域是随时间逐步出现的：今天有了图像分类专家，明天来了 OCR 专家，后天又来了医学影像专家。这种时序性带来的问题从未被系统研究过。

**核心矛盾**：在时序场景下，关键设计选择变得不确定——(1) 训练新专家时，应该从基础模型初始化还是从当前合并模型初始化？(2) 每个时间步都应该融合所有模型吗还是只融合新的？(3) 最终部署应该用当前合并版本还是从头融合所有专家？这些问题的答案不显然，且现有工作缺乏系统研究。

**本文目标**：建立时序模型融合的统一框架，系统性地回答上述设计问题，为实践者提供最佳实践。

**切入角度**：定义三个正交的设计轴——初始化、部署和融合技术——构成完整的搜索空间，通过大规模实验找到最优配置。

**核心 idea**：TIME 框架将时序模型融合分解为三个独立决策维度，通过在 FoMo-in-Flux 多模态基准上的系统实验，揭示了初始化策略（从基础模型 vs 从合并模型）和部署策略（持续融合 vs 重新融合）对最终性能的关键影响。

## 方法详解

### 整体框架
考虑一系列按时间顺序到达的专家模型训练任务 $\{T_1, T_2, \ldots, T_N\}$。在每个时间步 $t$，我们需要：(1) 选择如何初始化新专家的微调（I 轴），(2) 训练新专家，(3) 选择如何将新专家与已有知识融合（M 轴），(4) 决定部署哪个模型（D 轴）。TIME 框架清晰地定义了每个轴的可选方案。

### 关键设计

1. **初始化阶段（Initialization Phase）**:

    - 功能：决定训练新任务专家时的起始权重
    - 核心思路：两种主要策略——(a) **Base Init**：每个新专家都从原始基础模型 $\theta_0$ 开始微调，保证每个专家的 task vector $\tau_i = \theta_i - \theta_0$ 都相对于同一个锚点计算，便于后续融合；(b) **Merged Init**：从当前时间步的融合模型 $\theta_t^{merged}$ 开始微调新专家，利用已学到的知识作为初始化减少训练成本并可能促进正向迁移。但这会导致 task vector 的定义基点不断漂移，可能影响融合质量
    - 设计动机：初始化策略直接影响 task vector 空间的结构——Base Init 保证了各 task vector 的正交性假设更接近成立，而 Merged Init 牺牲了这一点但可能获得更好的少样本性能

2. **部署阶段（Deployment Phase）**:

    - 功能：决定每个时间步实际部署什么模型
    - 核心思路：(a) **Continual Merging**：每个新专家到来时渐进地融合到当前模型中，部署当前合并结果；(b) **Full Re-merge**：每当新专家到来时，从头融合所有专家模型（成本更高但可能更准确）；(c) **Deploy Latest Expert Only**：每次只用最新专家模型（作为 baseline）。部署策略还包括是否保留所有历史专家权重（存储成本）
    - 设计动机：Continual Merging 效率高但可能累积融合误差，Full Re-merge 理论上更好但计算和存储开销大

3. **融合技术（Merging Technique）**:

    - 功能：具体的模型参数融合算法
    - 核心思路：实验比较了多种融合方法在时序场景下的表现——(a) **Weight Averaging**：简单参数平均；(b) **Task Arithmetic**：$\theta_{merged} = \theta_0 + \lambda \sum_i \tau_i$；(c) **TIES-Merging**：通过符号一致性和剪枝减少 task vector 间的干扰；(d) **DARE**：随机丢弃 task vector 中的部分维度以减少冲突；(e) **Fisher Merging**：用 Fisher 信息矩阵加权。每种方法在时序场景下可能有不同于静态场景的行为
    - 设计动机：系统测试哪些融合技术在时序渐进场景下最鲁棒

### 损失函数 / 训练策略
框架本身不引入新的训练方法，各专家使用标准微调。核心贡献在于系统化框架和大规模实验设计。在 FoMo-in-Flux 基准上进行实验，涵盖不同模型规模（ViT-B/32, ViT-B/16, ViT-L/14）、计算预算和学习时长。

## 实验关键数据

### 主实验：不同 TIME 配置对比

| 初始化 | 部署 | 融合技术 | 平均准确率 | 遗忘率 | 说明 |
|--------|------|---------|-----------|--------|------|
| Base | Re-merge | Task Arithmetic | **最高** | 最低 | 总体最优配置 |
| Base | Continual | Task Arithmetic | 次优 | 低 | 效率最高 |
| Merged | Continual | Task Arithmetic | 中等 | 中等 | 漂移导致性能下降 |
| Merged | Re-merge | Task Arithmetic | 中等偏上 | 中等 | 部分缓解漂移 |
| Base | Re-merge | Simple Average | 较低 | 低 | 简单平均不够好 |
| Base | Re-merge | TIES | 较高 | 低 | 接近 Task Arithmetic |

### 模型规模影响

| 模型 | Base+Re-merge vs Merged+Continual 差距 | 说明 |
|------|--------------------------------------|------|
| ViT-B/32 | 较大 | 小模型更敏感于初始化策略 |
| ViT-B/16 | 中等 | 中等规模差距缩小 |
| ViT-L/14 | 较小 | 大模型对策略选择更鲁棒 |

### 关键发现
- **Base Init 显著优于 Merged Init**：从基础模型初始化每个专家是最重要的设计选择。Merged Init 导致 task vector 锚点漂移，累积到后期严重损害融合质量
- **Full Re-merge 优于 Continual Merging**：虽然成本更高，但从头融合所有专家避免了渐进融合的误差累积。差距在长时间线上更明显
- **Task Arithmetic 在时序场景下依然最优**：尽管 TIES 和 DARE 在静态场景下有优势，但在时序场景下 Task Arithmetic 的简单性反而成为优势，因为更少的超参数意味着更少的跨时间步的不一致
- **模型规模增大可以部分缓解策略选择的影响**：大模型对初始化和部署策略的选择更鲁棒，但最优策略的排序不变
- **初始化和部署的交互效应**：Base Init + Re-merge 的组合效果不是两者独立优势的简单叠加，存在正向交互

## 亮点与洞察
- **定义了一个重要的新问题**：时序模型融合是真实场景下的核心需求但此前无人系统研究。TIME 框架将问题空间清晰地参数化为三个正交轴，使得设计空间可探索、结论可迁移
- **Base Init 的发现违反直觉但很重要**：直觉上，从已融合的模型初始化新专家应该能利用先验知识加速训练。但实验表明这会破坏 task vector 空间的结构，在融合时带来更大的性能损失。这对所有使用 task arithmetic 的场景都有指导意义
- **框架的通用性**：TIME 不局限于 CLIP，适用于任何"基础模型+多专家微调+融合"的pipeline，包括 LLM 的 LoRA 合并

## 局限与展望
- 实验主要在 CLIP（多模态视觉-语言）模型上，未在生成模型（Stable Diffusion）或纯语言模型（LLM）上验证
- FoMo-in-Flux benchmark 虽然多样，但任务规模（~几十个任务）相比现实场景可能仍较小
- 未考虑专家之间任务相关性的影响——高度相关的任务和完全不相关的任务在融合时应否区别对待
- Full Re-merge 虽好但需要保存所有历史专家权重，存储开销随时间线性增长
- 没有探索自适应策略——根据当前时间步的特性动态选择初始化和融合方法

## 相关工作与启发
- **vs Task Arithmetic 原文**: Task Arithmetic 只考虑静态的一次性融合；TIME 证明了其结论在时序场景下依然适用，但最优实践（Base Init + Re-merge）是新发现
- **vs 持续学习（CL）**: CL 研究遗忘问题但通常假设单一模型序列训练；时序融合保留了所有专家，遗忘模式和解决方案不同
- **vs Model Soups/DARE/TIES**: 这些方法关注更好的融合技术，TIME 表明在时序场景下初始化和部署策略比融合技术本身更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统化定义和研究时序模型融合问题，框架设计清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型规模、多融合技术、多初始化/部署策略的全面组合实验
- 写作质量: ⭐⭐⭐⭐ 框架设计和实验设计都很清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 对模型融合实践者有直接的指导意义，特别是 Base Init + Re-merge 的最佳实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](../../NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](conformal_prediction_for_zero-shot_models.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/optimization/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)

</div>

<!-- RELATED:END -->
