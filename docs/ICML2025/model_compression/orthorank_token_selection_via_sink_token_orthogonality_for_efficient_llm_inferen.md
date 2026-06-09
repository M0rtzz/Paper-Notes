---
title: >-
  [论文解读] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference
description: >-
  [ICML2025][模型压缩][LLM推理加速] 提出 OrthoRank，一种**无需额外训练**的动态 token 选择方法：利用 sink token 与其他 token 在隐藏状态空间中的正交性来衡量 token 重要性，在每层选出 Top-K 重要 token 进行完整计算…
tags:
  - "ICML2025"
  - "模型压缩"
  - "LLM推理加速"
  - "注意力汇聚"
  - "Token选择"
  - "正交性"
  - "动态剪枝"
---

# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference

**会议**: ICML2025  
**arXiv**: [2507.03865](https://arxiv.org/abs/2507.03865)  
**作者**: Seungjun Shin, Jaehoon Oh, Dokwan Oh
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: LLM推理加速, 注意力汇聚, Token选择, 正交性, 动态剪枝

## 一句话总结

提出 OrthoRank，一种**无需额外训练**的动态 token 选择方法：利用 sink token 与其他 token 在隐藏状态空间中的正交性来衡量 token 重要性，在每层选出 Top-K 重要 token 进行完整计算，其余 token 仅参与 KV 计算，在相同稀疏率下实现比层剪枝更低的困惑度和更高的零样本准确率。

## 研究背景与动机

### 问题背景
LLM 推理的计算成本极高，尤其在实时应用场景下。现有加速方法主要分为几类：
- **层剪枝（Layer Pruning）**：通过度量层的输入输出相似度或移除后对最终输出的影响来删减冗余层。简单高效，但对所有 token 施加**统一的**剪枝决策，无法适应不同 token 的计算需求差异
- **早退出（Early Exit）**：根据 token 特征动态决定是否提前退出后续层。灵活但**需要训练额外的路由器或分类器**
- **混合深度（Mixture of Depth）**：在每层使用路由器决定 token 是否需要计算。同样**需要额外训练**

### 核心动机
能否在**不引入任何额外训练**的前提下，逐层判断哪些 token 仍需更新、哪些可以跳过？

### 关键观察：Sink Token 的隐藏状态行为
论文从 attention sink 现象出发，深入分析了 sink token（通常是序列的第一个 token）与其他 token 在隐藏状态层面的关系，发现两个重要规律：

**余弦相似度递增**：随着层数加深，其他 token 归一化隐藏状态与 sink token 归一化隐藏状态的余弦相似度**持续增大**

**Sink token 隐藏状态近似不变**：sink token 的归一化隐藏状态在各层间几乎不发生变化

这两点意味着：**所有 token 在层间持续向 sink token 的方向靠拢**。已经接近 sink token 方向的 token 其信息已趋稳定，无需再更新；仍与 sink token 方向正交的 token 蕴含更多待提取的语义信息，需要继续计算。

## 方法详解

### 整体框架
OrthoRank 的流程如下：
1. **计算正交性**：在每个候选选择层，计算各 token 归一化隐藏状态 $\bar{h}_i^l$ 与 sink token 归一化隐藏状态 $\bar{h}_0^l$ 的正交度
2. **排序与选择**：按正交度降序排列，选取 Top-K 个 token 进入完整计算路径
3. **差异化执行**：
    - **被选中 token**：正常通过该层所有运算（Query、Key、Value 投影 + Attention + FFN），更新自身隐藏状态
    - **未选中 token**：仅参与 Key 和 Value 计算（为被选中 token 提供上下文），但**不更新自身隐藏状态**，直接旁路到下一层
4. **层选择策略**：借鉴层剪枝的评估方法，确定哪些层适合用 token 选择替代完整计算

### 关键设计 1：基于正交性的 Token 重要性度量

设第 $l$ 层 token $i$ 的隐藏状态为 $h_i^l$，归一化后为 $\bar{h}_i^l = h_i^l / \|h_i^l\|$。Token 重要性定义为其与 sink token 的**正交程度**：

$$\text{Importance}(i, l) = 1 - |\cos(\bar{h}_i^l, \bar{h}_0^l)| = 1 - \left|\frac{\bar{h}_i^l \cdot \bar{h}_0^l}{\|\bar{h}_i^l\| \cdot \|\bar{h}_0^l\|}\right|$$

直觉解释：
- **高正交性（重要）**：token 的隐藏状态方向与 sink token 差异大，说明该 token 仍在快速向 sink token 方向移动，包含尚未被充分处理的语义信息
- **低正交性（不重要）**：token 已基本与 sink token 对齐，信息已趋于稳定，跳过该层计算损失较小

### 关键设计 2：Token 选择层的确定

并非所有层都适合进行 token 选择。论文采用与层剪枝类似的**层评估方法**来确定最佳的 token 选择位置：

1. 在校准集上评估每一层的"可替代性"——将该层替换为 token 选择层后观察性能变化
2. 按性能损失从小到大排序，选择损失最小的若干层作为 token 选择层
3. 其余层保持完整计算

这使得 OrthoRank 能在不同稀疏率下自适应地调整 token 选择层的数量和位置。

### 关键设计 3：未选中 Token 的 KV 参与机制

未被选中的 token 虽然不更新自身隐藏状态，但仍参与当前层的 Key 和 Value 计算。这一设计至关重要：

- 保留了完整的上下文信息，被选中 token 的 attention 仍能覆盖所有 token 的 KV
- 类似于 early-exit 机制中对已退出 token 的处理，确保信息流不会断裂
- 计算开销主要省在：不为未选中 token 计算 Query 投影、不执行其 attention 计算、不通过 FFN 更新

### 与层剪枝的本质区别

| 维度 | 层剪枝 | OrthoRank |
|------|--------|-----------|
| 粒度 | 层级（整层跳过） | Token级（逐Token决定） |
| 适应性 | 静态（所有输入共用） | 动态（随输入变化） |
| 信息保留 | 该层所有信息丢失 | 保留KV信息流 |
| 额外训练 | 不需要 | 不需要 |
| 适用场景 | 存储/带宽受限 | 追求细粒度加速 |

## 实验关键数据

### 实验 1：语言建模困惑度对比

在相同稀疏率和可比吞吐量下，OrthoRank 与层剪枝方法在多个 LLM 上的困惑度（Perplexity，越低越好）对比：

| 方法 | 稀疏率 | LLaMA-2-7B PPL | LLaMA-2-13B PPL | LLaMA-3-8B PPL | 吞吐量比 |
|------|--------|----------------|-----------------|----------------|----------|
| 原始模型 | 0% | 5.47 | 4.88 | 6.14 | 1.0x |
| ShortGPT (层剪枝) | ~25% | 7.82 | 6.91 | 8.53 | ~1.3x |
| LaCo (层剪枝) | ~25% | 7.45 | 6.54 | 8.12 | ~1.3x |
| **OrthoRank** | **~25%** | **6.83** | **5.97** | **7.41** | **~1.3x** |
| ShortGPT (层剪枝) | ~40% | 12.6 | 9.87 | 15.2 | ~1.5x |
| **OrthoRank** | **~40%** | **9.14** | **7.65** | **10.8** | **~1.5x** |

OrthoRank 在所有模型和稀疏率设定下均实现更低的困惑度，且稀疏率越高优势越明显。

### 实验 2：零样本下游任务准确率

在多个零样本基准任务上的准确率（%，越高越好）：

| 方法 | 稀疏率 | ARC-C | HellaSwag | MMLU | WinoGrande | 平均 |
|------|--------|-------|-----------|------|------------|------|
| LLaMA-2-7B (原始) | 0% | 46.3 | 76.0 | 45.9 | 69.1 | 59.3 |
| ShortGPT | ~25% | 38.7 | 68.2 | 37.4 | 63.5 | 51.9 |
| LaCo | ~25% | 40.1 | 69.5 | 38.9 | 64.2 | 53.2 |
| **OrthoRank** | **~25%** | **42.8** | **72.1** | **41.5** | **66.4** | **55.7** |
| 层剪枝 (best) | ~40% | 31.2 | 57.8 | 29.6 | 56.1 | 43.7 |
| **OrthoRank** | **~40%** | **36.5** | **64.3** | **34.8** | **61.7** | **49.3** |

OrthoRank 在零样本任务上的平均准确率比最佳层剪枝方法高 2-5 个百分点。

### 实验 3：LongBench 长文本评估

OrthoRank 在 LongBench 上同样展示了优势，表明动态 token 选择在长序列场景下尤其有效——长序列中存在大量冗余 token，token 级别的选择比层级剪枝更能精准保留关键信息。

## 亮点与洞察

- **洞察深刻**：从 attention sink 现象出发，在隐藏状态空间发现了"所有 token 持续向 sink token 靠拢"的现象，将这一几何直觉转化为 token 重要性的定量度量，思路新颖且自然
- **无需训练**：与 early exit、MoD 等需要额外训练路由器的方法不同，OrthoRank 完全基于模型现有的隐藏状态计算正交性，可直接应用于任意预训练 LLM
- **细粒度动态性**：相比层剪枝对所有 token 一视同仁，OrthoRank 能根据每个 token 的当前状态动态决定是否需要更新，实现更精细的计算资源分配
- **KV 保留设计**：未选中 token 仍参与 KV 计算的设计巧妙地平衡了效率与质量——既节省了大部分计算（跳过 Q 投影、attention、FFN），又保留了完整的上下文信息
- **与层剪枝互补**：OrthoRank 借鉴层剪枝的层评估策略来确定选择层位置，形成了两者的有机结合

## 局限与展望

- **Sink token 定义简化**：论文统一以第一个 token 作为 sink token，但实际中分隔符 token（如"."、"\n"）也可能充当 sink 角色，未充分利用多 sink token 的信息
- **Top-K 选择的粒度**：固定 K 值对所有层和所有输入统一适用，未探索自适应 K 的方案
- **正交性计算开销**：虽然比 attention/FFN 计算轻量很多，但在每个选择层对所有 token 计算余弦相似度仍有额外开销
- **仅验证解码器架构**：实验主要在 LLaMA 系列等 decoder-only 模型上验证，对 encoder-decoder 或 encoder 架构的适用性未知
- **缺乏理论收敛分析**：未给出 token 选择对最终输出误差的理论上界
- **与其他加速技术的组合**：未探索与量化（如 GPTQ、AWQ）、KV Cache 压缩等正交加速技术的联合使用效果
- **Prefill 阶段适用性**：当前方法主要关注 decoding 阶段，对 prefill 阶段大量 token 并行计算的场景是否同样有效值得探索

## 相关工作与启发

### 注意力汇聚（Attention Sink）
- **Xiao et al. (2024b)**：首次发现并命名 attention sink 现象，即序列首 token 获得不成比例的高注意力权重
- **Sun et al. (2024), Cancedda (2024)**：进一步研究 sink 现象的成因和性质
- **本文贡献**：将对 sink token 的研究从 attention 分布层面扩展到**隐藏状态空间的几何结构**

### 层剪枝
- **ShortGPT (Men et al., 2024)**：基于层间隐藏状态相似度的剪枝
- **LaCo (Siddiqui et al., 2024)**：基于层重要性评分的剪枝
- **Kim et al. (2024)**：一次性剪枝 + LoRA 微调
- **对比**：OrthoRank 在层剪枝的框架上引入 token 级粒度，实现更优的效率-性能权衡

### 动态计算
- **Schuster et al. (2022), Bae et al. (2023)**：早退出方法，需要额外训练
- **Raposo et al. (2024) - MoD**：混合深度，需要路由器训练
- **对比**：OrthoRank 实现了无需训练的动态 token 级计算路径

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从 sink token 的隐藏状态几何性质推导出 token 重要性度量，切入角度新颖
- 实验充分度: ⭐⭐⭐⭐ — 涵盖困惑度、零样本任务、LongBench 多个维度，模型覆盖 LLaMA-2/3 系列
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，从观察到方法的推导自然流畅
- 价值: ⭐⭐⭐⭐ — 无需训练即可应用于现有 LLM 加速，实用价值高，对 token 冗余性提供了新的理解视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)

</div>

<!-- RELATED:END -->
