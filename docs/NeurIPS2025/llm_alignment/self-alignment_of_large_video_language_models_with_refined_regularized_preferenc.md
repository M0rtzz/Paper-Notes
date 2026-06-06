---
title: >-
  [论文解读] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization
description: >-
  [NeurIPS 2025][LLM对齐][video LLM] 提出 RRPO（Refined Regularized Preference Optimization），通过子序列级细粒度奖励和 token 级 KL 正则化替代 DPO 的响应级奖励，结合自对齐数据生成框架…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "video LLM"
  - "preference optimization"
  - "self-alignment"
  - "hallucination"
  - "temporal understanding"
---

# Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2504.12083](https://arxiv.org/abs/2504.12083)  
**代码**: [GitHub](https://pritamsarkar.com/RRPO)  
**领域**: LLM对齐  
**关键词**: video LLM, preference optimization, self-alignment, hallucination, temporal understanding

## 一句话总结
提出 RRPO（Refined Regularized Preference Optimization），通过子序列级细粒度奖励和 token 级 KL 正则化替代 DPO 的响应级奖励，结合自对齐数据生成框架，在视频理解任务上减少幻觉、提升时间推理能力。

## 研究背景与动机

**LVLM 的核心问题**：大型视频语言模型在细粒度时间理解、幻觉、简单 QA 任务上仍频繁犯错

**原因分析**：时空理解不足、视觉-语言表示不对齐、共现概念的虚假相关、过度依赖语言线索而忽略视觉信息

**DPO 的局限**：
   - 响应级奖励过于粗粒度，惩罚所有 token 而非关键差异 token
   - 长响应的梯度过大，导致模型偏离初始状态，丧失原始能力
   - 弱正则化无法有效控制偏离

## 方法详解

### 整体框架：自对齐 pipeline

1. 从开源基准采样视频-问题对
2. 对视频施加时空扰动（帧遮蔽 25%-50% + 时间乱序）
3. 用扰动视频做推理，错误响应作为 non-preferred，正确响应作为 preferred
4. 用 LLM 识别 preferred/non-preferred 之间的关键差异概念
5. 用 RRPO 优化模型偏好

### 关键设计：RRPO

**子序列级细粒度奖励**：仅对 preferred 和 non-preferred 响应中**差异的关键概念子序列**计算奖励，而非整个响应：

$$u = \sum_{i=1}^N (r_\theta(x, y_i^+) - r_\theta(x, y_i^-))$$

其中 $y_i^+$ 和 $y_i^-$ 是第 $i$ 个差异子序列。

**Token-wise KL 正则化**：在 preferred 响应上计算 token 级 KL 散度，防止模型偏离：

$$\mathbb{D}_{\text{TKL}}(x, y; \pi_{\text{ref}} \| \pi_\theta) = \sum_{t=1}^{|y|} \mathbb{D}_{\text{KL}}(\pi_{\text{ref}}(\cdot|[x,y_{<t}]) \| \pi_\theta(\cdot|[x,y_{<t}]))$$

**最终损失**：

$$\mathcal{L}_{\text{RRPO}} = -\mathbb{E}[\log\sigma(u) + \alpha \cdot \mathbb{D}_{\text{TKL}}(x, y^+)]$$

### 梯度分析

RRPO 梯度上界 $\|\nabla_\theta \mathcal{L}_{\text{RRPO}}^{(\text{rank})}\| \leq \beta M(2NL)$，DPO 梯度上界 $\|\nabla_\theta \mathcal{L}_{\text{DPO}}\| \leq \beta M(|y^+|+|y^-|)$。由于 $2NL \ll |y^+|+|y^-|$，RRPO 梯度更小，更新更稳定。加上 TKL 项的负梯度，进一步减小总梯度幅度。

### 损失函数 / 训练策略

- 使用 LoRA 仅训练 LLM 部分，冻结其他参数
- 训练帧数 16 帧，推理时可使用更多帧
- 4×A100 80GB，训练 1-10 小时
- 三个基础模型：VideoChat2、LLaVA-Video、LongVU

## 实验关键数据

### 主实验：RRPO vs 其他对齐方法

| 方法 | TVBench | VideoHallucer | VideoMME | MLVU | Δ/%Δ |
|------|---------|---------------|----------|------|------|
| LongVU (base) | 53.7 | 39.2 | 56.2 | 63.6 | - |
| + DPO | 54.3 | 40.9 | 56.6 | 63.6 | 0.7/1.5 |
| + DPA | 54.6 | 40.3 | 56.9 | 63.9 | 0.7/1.5 |
| + TDPO | 53.9 | 41.4 | 57.0 | 63.8 | 0.8/1.9 |
| + **RRPO** | **56.5** | **44.0** | **57.7** | **64.5** | **2.5/5.4** |

### 与现有对齐 LVLM 对比

| 模型 | TVBench | VideoHallucer | VideoMME | MLVU |
|------|---------|---------------|----------|------|
| LLaVA-Video-TPO | 51.1 | 50.6 | 65.6/71.5 | 68.7 |
| **LLaVA-Video-RRPO** | **52.2** | **55.8** | 65.5/**71.8** | **69.4** |

RRPO 在所有 setup 上超越 TPO，VideoHallucer 提升达 5.2%。

### 消融实验

| 变体 | TVBench | VideoHallucer | Δ |
|------|---------|---------------|---|
| RRPO w/o 细粒度奖励 | 54.3 | 43.0 | -1.5 |
| RRPO w/o TKL | 54.9 | 39.1 | -2.6 |
| 完整 RRPO | 56.5 | 44.0 | 基准 |

两个组件均有贡献，TKL 正则化影响更大。

### 模型偏离分析

- RRPO KL 散度 ≈ 1（使用 10× 更高学习率）
- DPO KL 散度 ≈ 20
- TDPO/DPA KL 散度 ≈ 1，但性能显著更差
- RRPO 实现最优的性能-偏离权衡

### 关键发现

1. 时间理解提升最高 2.8%（TVBench）
2. 幻觉减少 4.8%-8.8%（VideoHallucer）
3. 短视频和长视频理解均有一致提升
4. 扰动策略中 Mask + Local Shuffle 效果最佳

## 亮点与洞察

1. **自对齐数据生成**：通过时空扰动激发模型错误，无需人工标注
2. **理论支撑的梯度分析**：数学证明 RRPO 梯度更小更稳定
3. **概念级精准对齐**：只惩罚差异概念而非整个响应，避免过度惩罚
4. **TKL 作为信任区域约束**：防止模型大幅偏离，允许使用更大学习率

## 局限与展望

1. 扰动策略仍较简单（帧遮蔽+乱序），更复杂的视觉扰动可能更有效
2. 依赖 GPT-4o-mini 进行概念对比和正确性验证
3. 仅在 7B 模型上实验，更大模型待验证
4. 训练帧数（16帧）与推理帧数（64-100帧）不一致，可能存在分布偏移

## 相关工作与启发

- **DPO**：RRPO 的出发点，解决其响应级奖励和弱正则化问题
- **TDPO**：增强 DPO 正则化，但性能改善有限；RRPO 同时改进奖励和正则化
- **DDPO**：提供细粒度奖励，但缺乏强正则化；RRPO 结合两者优势
- **启发**：子序列级奖励思想可推广到任何偏好优化场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 子序列级奖励 + TKL 正则化的组合设计有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 3个基础模型 × 8个基准，全面的对比和消融
- 写作质量: ⭐⭐⭐⭐ 梯度分析清晰，实验丰富
- 价值: ⭐⭐⭐⭐ 对视频 LLM 对齐有实用参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](densedpo_finegrained_temporal_preference_optimization_for_vi.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/llm_alignment/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)

</div>

<!-- RELATED:END -->
