---
title: >-
  [论文解读] Segment-Level Attribution for Selective Learning of Long Reasoning Traces
description: >-
  [ICLR2026][LLM推理][reasoning trace] 用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。
tags:
  - ICLR2026
  - LLM推理
  - reasoning trace
  - integrated gradients
  - selective SFT
  - segment attribution
  - CoT compression
---

# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

**会议**: ICLR2026  
**arXiv**: [2602.00425](https://arxiv.org/abs/2602.00425)  
**代码**: [GitHub](https://github.com/SiyuanWangw/SegmentSelectiveSFT)  
**领域**: llm_reasoning  
**关键词**: reasoning trace, integrated gradients, selective SFT, segment attribution, CoT compression  

## 一句话总结
用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

## 背景与动机
1. 大推理模型(LRM)生成数千token的CoT，但仅少部分真正对答案预测有贡献，大量冗余重复/截断内容
2. 对冗余CoT做全量SFT会使模型学习冗长无信息模式，浪费学习能力甚至降低性能
3. 现有压缩方法token-level分析忽略语义完整性，segment-level的困惑度/熵指标与重要性不完全一致
4. 困惑度方法存在假阳性（高估过渡文本）和假阴性（低估验证/中间结论）问题
5. 需要直接度量segment对正确答案预测的因果贡献

## 方法详解

### 整体框架
Segment-Level Selective SFT = Segment分割 → IG归因 → 双指标筛选重要Segment → 选择性Loss训练

### Segment分割
用transition关键词（"\n\nWait", "\n\nAlternatively", "\n\nLet me"等）将长CoT分割为语义片段。每个segment对应推理链中的一个独立思考单元（如问题理解、中间探索、验证等）。

### Integrated Gradients归因
对每个token $o_n$ 计算IG值：沿padding baseline到实际embedding的直线路径积分梯度，衡量该token对正确答案概率的贡献方向和量级。用J个插值步近似积分：
$$\text{IG}_i(x) \approx (x_i - x_i') \times \frac{1}{J}\sum_{j=1}^{J}\frac{\partial F(x'+j/J \cdot (x-x'))}{\partial x_i}$$

### 两个Segment-level指标
- **Attribution Strength**: $\text{Strength}(S) = \sum_{o_n \in S}|IG(o_n)| / \sqrt{N}$，衡量影响量级。√N归一化避免长segment因token数多而占优。跨segment归一化后可比较同一CoT内各segment的相对重要性
- **Direction Consistency**: $\text{Consistency}(S) = |\sum IG(o_n)| / \sum|IG(o_n)|$，衡量正负贡献一致性。值接近1表示segment内token贡献方向一致（要么全正要么全负），反映浅层确认或严重错误探索；中等值表示混合了正负贡献——这是反思性推理的标志，segment内既有探索又有纠正

### 重要Segment选择（两步筛选）
1. **强度阈值**：按归因强度降序排列segment，取累计强度达τ=70%的top-k* segment（约30-40%的segment承载了80%+的归因）
2. **一致性过滤**：从top-k*中过滤掉方向一致性>β=0.8的segment，保留一致性≤0.8的segment作为重要segment。结果约33%的segment被标为重要（占45%的token——因为重要segment通常更长）

### Selective SFT（选择性训练）
完整CoT全部输入模型（保持自回归上下文），但仅在重要segment的token上计算cross-entropy loss，不重要segment的token loss被mask为0：
$$L_{\text{Selective-SFT}}(\theta) = -\frac{1}{\sum_t I(o_t)}\sum_{t=1}^{T}I(o_t)\log P(o_t|o_{<t}, q; \theta)$$
这起到隐式正则化作用——防止模型过拟合冗余/重复内容，同时保持完整上下文的连贯性。

## 实验
| 模型 | 方法 | Overall Acc | 输出长度 |
|------|------|:-----------:|:--------:|
| R1-Distill-Qwen-1.5B | Full SFT | 44.8 | 16520 |
| R1-Distill-Qwen-1.5B | **Segment Selective** | **46.9**(+4.7%) | 13506(-18%) |
| R1-Distill-Qwen-7B | Full SFT | 62.1 | 9693 |
| R1-Distill-Qwen-7B | **Segment Selective** | **64.5**(+3.9%) | 8499(-12%) |
| Qwen2.5-7B-Instruct | Full SFT | 44.2 | 10317 |
| Qwen2.5-7B-Instruct | **Segment Selective** | **45.6**(+3.2%) | 9852(-5%) |

### 消融实验与分析
| 设置 | Overall Acc | Overall Length |
|------|:-----------:|:--------------:|
| R1-Distill-Qwen-7B (base) | 57.7 | 12518 |
| + Full CoT SFT | 62.1 | 9693 |
| + Token-level pruning SFT | 60.5 | 8112 |
| + **Segment Selective SFT** | **64.5** | **8499** |
| Only Strength (无Consistency过滤) | 63.2 | 8856 |
| Only Consistency (无Strength排序) | 61.8 | 9234 |

**关键发现**:
1. 30-40%的segment贡献80%+的总归因(CDF曲线验证)，大量冗余
2. 重要segment具有更低困惑度/熵，不重要segment更多重复(高BLEU>0.8)和截断(49% vs 26%)
3. Selective SFT一致优于全量SFT和token-level剪枝方法——剪枝会破坏上下文完整性
4. 在OOD难题(AIME24)上提升最显著(+13.3 pp)，说明选择性学习帮助模型更好泛化
5. 方向一致性过滤(β=0.8)额外贡献约1.3%的准确率提升，验证了segment内正负混合推理的价值
6. 该方法思路可泛化到RL场景——在重要segment上加大policy gradient权重
7. 温度采样(pass@6)下Selective SFT优势更明显，说明学到了更好的推理模式而非仅拟合特定输出

## 亮点与洞察
- 用IG归因直接度量segment对答案的因果贡献，比PPL/熵等间接指标更可靠
- 方向一致性(consistency)指标设计巧妙：区分浅层确认vs反思性推理
- Selective SFT同时提升准确率和效率（缩短输出），双赢
- 分析透彻：验证了不重要segment确实对应重复/截断/废话

## 损失函数
标准SFT对所有token均等计算loss。本文的Selective SFT通过indicator函数$I(o_t)$对token进行mask：只有属于重要segment的token才贡献loss。这等价于在loss landscape中构建了一个关注重要推理步骤的隐式课程——模型的参数更新被引导向关键推理模式，而非冗余填充内容。

## 局限性
- IG计算需多步插值前向传播，计算开销较大（虽是一次性成本）
- 关键词分割方式较简单，可能不适应所有推理风格
- 仅在数学推理数据集上验证，对代码生成/自然语言推理的效果未知
- τ和β阈值需在验证集上搜索，增加调参成本

## 相关工作
- CoT压缩: Xia et al. 2025b token-level分析; Cui et al. 2025b segment-level PPL; Li et al. 2025b 基于熵
- Selective SFT: Lin et al. 2024 selective learning framework
- 归因方法: Sundararajan et al. 2017 Integrated Gradients; 本文首次应用于推理链segment
- 长推理冗余: Wang et al. 2025d 分析截断思维; Wu et al. 2025 冗长降低推理性能

## 评分
- 新颖性: ⭐⭐⭐⭐ (IG+segment归因+selective SFT组合新颖)
- 实验充分度: ⭐⭐⭐⭐ (多模型+ID/OOD+消融充分)
- 写作质量: ⭐⭐⭐⭐ (分析细致，可视化好)
- 价值: ⭐⭐⭐⭐ (对长推理链训练有直接工程价值)

<!-- RELATED:START -->

## 相关论文

- [Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](../../NeurIPS2025/llm_reasoning/segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [Is In-Context Learning Learning?](is_in-context_learning_learning.md)
- [Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)
- [Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)

<!-- RELATED:END -->
