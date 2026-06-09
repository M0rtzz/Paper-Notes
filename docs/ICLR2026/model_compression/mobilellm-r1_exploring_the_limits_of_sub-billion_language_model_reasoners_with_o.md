---
title: >-
  [论文解读] MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes
description: >-
  [ICLR 2026][模型压缩][小模型推理] 通过精心的数据筛选和自适应混合策略，仅用4.2T token（Qwen3的11.7%）预训练出亿级参数的推理模型 MobileLLM-R1-950M，在AIME等推理基准上匹配或超越 Qwen3-0.6B，同时完全开源数据源和训练配方。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "小模型推理"
  - "数据筛选"
  - "影响力分数"
  - "数据混合"
  - "端侧部署"
---

# MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes

**会议**: ICLR 2026  
**arXiv**: [2509.24945](https://arxiv.org/abs/2509.24945)  
**代码**: [GitHub](https://github.com/facebookresearch/MobileLLM-R1)  
**领域**: 模型压缩  
**关键词**: 小模型推理, 数据筛选, 影响力分数, 数据混合, 端侧部署

## 一句话总结
通过精心的数据筛选和自适应混合策略，仅用4.2T token（Qwen3的11.7%）预训练出亿级参数的推理模型 MobileLLM-R1-950M，在AIME等推理基准上匹配或超越 Qwen3-0.6B，同时完全开源数据源和训练配方。

## 研究背景与动机
大模型推理能力（o1范式）正在改变AI领域，但大模型在端侧设备上不可行——长CoT推理更加剧了KV缓存的内存压力。两个流行假设是：（1）推理能力仅在足够大的模型中涌现；（2）推理需要海量训练数据。假设1已被Qwen3-0.6B等亚十亿模型挑战，但假设2基本未被质疑。

核心问题：给定严格的容量约束，什么是赋予小模型强推理能力的最有效配方？

关键挑战：小模型对噪声极其敏感，数据中的噪声容易淹没有限容量；神经元需要编码更多重叠知识，增加干扰风险。因此数据质量和筛选远比大模型更重要。

核心idea：通过capability-aware的leave-one-out分析确定有益数据源，用影响力分数进行跨能力的自适应数据混合，并在mid-training阶段通过数据-模型协同进化迭代优化。

## 方法详解

### 整体框架
MobileLLM-R1 要回答的问题是：在亿级参数的容量约束下，怎样的数据配方能榨出最强的推理能力。它的答案不是堆更多数据，而是把"数据该选谁、按什么比例混、训练中途要不要换"这三件事都交给量化的影响力分数来决定。整条管线分三阶段串起来：预训练用 4.2T token、按影响力计算出的混合比例喂进去；mid-training 每阶段 100B token，让数据和模型在迭代中互相校准；后训练再做指令对齐和推理 SFT。贯穿始终的主线是同一套思想——**用模型自己对数据的反应来挑数据，而不是靠人工启发式或下游基准**。

### 关键设计

**1. 能力感知的数据筛选：用 leave-one-out 量化每个数据源的真实贡献。**

小模型容量有限，一旦混进噪声数据就会淹没有限的表示空间，所以"哪些数据源真正有用"必须先算清楚。作者用 leave-one-out 的方式逐个排除数据集：每次抽掉一个数据源重新训练，再观察模型在 Code / Math / Knowledge 三个能力探测集上的 NLL 变化，变化越大说明这个数据源越关键。影响力被形式化为 $\Delta\mathcal{L}(\mathcal{D}_j, \mathcal{D}^P) = \mathbb{E}[\ell(z;\hat{\theta}_{-j}) - \ell(z;\hat{\theta})]$，即去掉 $\mathcal{D}_j$ 后探测集损失相对完整训练的增量。这套分析直接推翻了几条数据筛选的常识：FineWeb-Edu 是跨域"胶水"，移除它三个能力一起退化；StarCoder 对数学的贡献竟然超过 OpenWebMath 对代码的贡献，说明代码数据对推理有强正迁移；而 Wikipedia 对代码和数学几乎没帮助。

**2. 跨能力影响力数据混合：把启发式均匀采样换成量化最优权重。**

知道了谁有用，下一步是定每个数据源的采样比例。作者不再用拍脑袋的均匀采样，而是借 AutoMixer 框架高效近似样本级影响力分数 $\mathcal{I}(x_i, x_{\text{test}}; \theta) \approx -\nabla\mathcal{L}(x_{\text{test}})^\top H^{-1} \nabla\mathcal{L}(x_i)$，再把跨能力、跨训练阶段的贡献聚合成"联合影响力"，最终折算成数据集级别的采样权重 $w_g = \frac{\rho_g}{\sum \rho_{g'}}$。这样得到的混合比例是数据本身告诉你的最优解，而非人工先验——实验中它在未见基准上一致优于均匀采样。

**3. Mid-training 数据-模型协同进化：把固定混合改成迭代去噪。**

预训练定好的混合比例并非一劳永逸——随着模型能力增长，原本有用的样本可能变得冗余甚至有害，固定混合不再最优。作者在 mid-training 阶段让数据和模型协同进化：每个阶段用当前模型重新计算每个样本的影响力，只保留仍有正影响力的样本 $\mathcal{D}_t = \{x_i : I(x_i; \theta_t) > 0\}$，同时更新数据集权重；下一阶段在筛过的数据上继续训练、再筛一次。这本质上是一个"迭代去噪"过程，把已被模型吃透、信息趋零的样本不断剔除，直到大部分样本的影响力都压到零附近为止（通常 2 个阶段即收敛）。

### 损失函数 / 训练策略
预训练用标准的 next-token 预测目标。后训练拆成两个阶段：先用 Tulu-3-SFT 做指令对齐，再用 OpenMathReasoning + OpenCodeReasoning + OpenScienceReasoning 做推理 SFT。一个值得注意的经验结论是，这两个阶段分开训练显著优于把它们的数据混在一起联合训练。

## 实验关键数据

### 主实验

| 模型 | 参数量 | 训练Token | MATH | GSM8K | AIME | LCBv6 |
|------|--------|----------|------|-------|------|-------|
| OLMo-2-1.48B | 1.48B | 4T+ | ~20 | ~50 | 0.6 | 11.4 |
| SmolLM-2-1.7B | 1.7B | 11T | ~15 | ~40 | 0.3 | 7.7 |
| Qwen3-0.6B | 0.6B | 36T | ~55 | ~65 | ~10 | ~12 |
| **MobileLLM-R1-950M** | 0.95B | 4.2T | **57.8** | **68.5** | **15.5** | **13.7** |
| MobileLLM-R1-360M | 0.36B | - | 19.2 | 23.8 | - | 4.0 |
| MobileLLM-R1-140M | 0.14B | - | 4.8 | 3.7 | - | 1.1 |

### 消融实验

| 配置 | MATH | GSM8K | LCBv6 | 说明 |
|------|------|-------|-------|------|
| 仅Math SFT (M) | 57.4 | 68.2 | 0.0 | 代码能力丧失 |
| 仅Code SFT (C) | 16.2 | 31.0 | 12.0 | 数学能力丧失 |
| M+C+S (分阶段) | 57.8 | 68.5 | 13.7 | 最佳平衡 |
| M+C+S (联合) | 56.2 | 53.1 | 14.9 | GSM8K急剧下降 |
| 无Tulu-3阶段 | 56.2 | 68.2 | 13.1 | 指令对齐有帮助 |
| 原始mid-training数据 | 偏低 | 偏低 | - | 有性能凹陷 |
| 筛选后mid-training数据 | 更高 | 更稳定 | - | 去除噪声样本有效 |

### 关键发现
- 仅用Qwen3的11.7% token即可匹配其推理性能，说明数据质量远比数量重要
- AIME上15.5分对比OLMo的0.6和SmolLM的0.3，说明预训练数据筛选对小模型至关重要
- StarCoder对数学的贡献大于OpenWebMath对代码的贡献——代码对推理有强正迁移
- mid-training的数据-模型协同进化在2阶段后收敛，影响力分布压缩到零附近

## 亮点与洞察
- "benchmark-free, self-evolving data optimization"的理念新颖——不看任何下游基准即可优化数据混合
- LOO分析揭示的跨域影响关系（如代码→数学的正迁移）对数据筛选有指导意义
- 数据-模型协同进化的收敛行为优美：影响力分布逐步压缩到零，说明数据信息已被充分利用
- 完全开源训练配方和数据源，高度可复现

## 局限与展望
- LOO分析需要为每个数据源单独训练模型，成本较高
- 影响力分数计算依赖AutoMixer的Hessian近似，可能引入误差
- 小于360M的模型推理能力仍然很弱，存在规模下限
- 后训练阶段复用现有SFT数据集，未对其进行同等的影响力筛选

## 相关工作与启发
- **vs Qwen3-0.6B**: 用11.7%的token达到相当性能，证明数据效率的巨大潜力
- **vs OLMo-2**: MATH准确率高5倍+，核心差距在预训练数据质量
- **vs SmolLM-2**: MATH准确率高2倍+，参数量更少

## 评分
- 新颖性: ⭐⭐⭐⭐ 影响力驱动的数据混合和协同进化思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 详尽的LOO分析、阶段消融、跨模型对比
- 写作质量: ⭐⭐⭐⭐⭐ 方法论清晰，图表精美，洞察深刻
- 价值: ⭐⭐⭐⭐⭐ 对小模型训练有极高参考价值，完全开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](../../ICML2026/model_compression/decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](../../ICML2026/model_compression/nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)

</div>

<!-- RELATED:END -->
