---
title: >-
  [论文解读] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections
description: >-
  [NeurIPS 2025][AIGC检测][LLM创造力检测] 提出 CLAWS，通过分析 LLM 在生成数学解答时对不同 prompt 区段的注意力权重分布，无需人工评估即可将生成内容分类为"创造性"、"典型"或"幻觉"三类。 近年来，经过 RL 训练的推理语言模型（RLM）在数学问题求解中取得了显著进步…
tags:
  - "NeurIPS 2025"
  - "AIGC检测"
  - "LLM创造力检测"
  - "注意力分析"
  - "数学推理"
  - "幻觉检测"
  - "白盒方法"
---

# CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections

**会议**: NeurIPS 2025  
**arXiv**: [2510.17921](https://arxiv.org/abs/2510.17921)  
**代码**: [GitHub](https://github.com/kkt94/CLAWS)  
**领域**: AIGC检测  
**关键词**: LLM创造力检测, 注意力分析, 数学推理, 幻觉检测, 白盒方法

## 一句话总结

提出 CLAWS，通过分析 LLM 在生成数学解答时对不同 prompt 区段的注意力权重分布，无需人工评估即可将生成内容分类为"创造性"、"典型"或"幻觉"三类。

## 研究背景与动机

近年来，经过 RL 训练的推理语言模型（RLM）在数学问题求解中取得了显著进步，但其创造力评估被严重忽视。现有研究主要集中在写作任务的创造力评估上（如 TTCW），数学推理中的创造力鲜有关注。

这一研究空白源于两大挑战：

**创造力定义困难**：如何界定数学解答中"创造性"的范围缺乏共识

**依赖人工评估**：数学创造力评估需要高水平领域专家，大规模标注成本极高

此外，过度抑制 LLM 生成以避免幻觉，可能同时扼杀创造性输出。因此，区分"创造性解答"与"幻觉解答"对最大化 LLM 输出的有效性和多样性至关重要。

## 方法详解

### 整体框架

CLAWS 框架包含四个阶段：
1. **生成阶段**：RLM 根据结构化 prompt 生成数学解答
2. **特征提取**：在生成过程中从注意力权重提取特征
3. **LLM 评估器标注**：GPT-o4-mini 和 Gemini-1.5-Pro 双评估器判定标签
4. **检测方法评估**：基于提取特征的各方法分类性能对比

### 关键设计

**Prompt 结构化分段**：输入 prompt $X = G|P|S|I$ 被划分为五个语义区段：
- Guideline ($G$)：创造力评估标准
- Problem ($P$)：待解数学问题
- Reference Solutions ($S$)：1~n 个典型参考解
- Instruction ($I$)：生成创新解的指令
- Response ($R$)：模型生成的回答

**注意力权重矩阵构建**：对最后一层 $L$ 的每个注意力头 $h$，在解码时间步 $t$，提取注意力向量：

$$A_{t,h}^{(L)} = [a_{1,h}^{(L)} \; a_{2,h}^{(L)} \cdots a_{k+t,h}^{(L)}], \quad k = \text{len}(X)$$

将所有时间步堆叠并补齐为统一维度：$A_h^{(L)} \in \mathbb{R}^{T \times (\text{len}(X) + T)}$

**段落平均注意力计算**（AVGA）：

$$\text{AVGA}_{\mathcal{U}} = \frac{1}{H \cdot T \cdot |\mathcal{I}_{\mathcal{U}}|} \sum_{h=1}^{H} \sum_{t=1}^{T} \sum_{i \in \mathcal{I}_{\mathcal{U}}} A_h^{(L)}[t, i]$$

其中 $\mathcal{U} \in \{G, P, S, I, R\}$，$\mathcal{I}_{\mathcal{U}}$ 为各段落 token 索引集。

**归一化注意力比率**（CLAWS 特征）：

$$\text{CLAWS}_{\mathcal{U}} = \frac{\text{AVGA}_{\mathcal{U}}}{\sum_{\mathcal{U}' \in \{G,P,S,I,R\}} \text{AVGA}_{\mathcal{U}'}}$$

### 损失函数

CLAWS 本身是无训练的特征提取方法。在结合下游分类器（XGBoost/MLP/TabM）时，使用标准交叉熵损失进行三分类训练。

**三类标签定义**：
- **幻觉解答**：两个评估器均未判定为"正确"
- **创造性解答**：两个评估器均判"正确"，且至少一个判定有"创造性"
- **典型解答**：两个评估器均判"正确"，且均未判定有"创造性"

## 实验关键数据

### 主实验

在5个 RLM（7-8B 参数）上，使用 Prototype 策略评估 CLAWS 与5种基线方法：

| 模型 | 方法 | TEST F1w | AMC F1w | AIME F1w | A(J)HSME F1w |
|------|------|----------|---------|----------|-------------|
| DeepSeek | Perplexity | 48.09 | 44.56 | 55.93 | 42.34 |
| DeepSeek | **CLAWS** | **58.66** | **46.71** | **56.90** | **38.82** |
| Mathstral | Hidden Score | 49.86 | 37.37 | 65.96 | 33.42 |
| Mathstral | **CLAWS** | **63.20** | **51.47** | **65.25** | **49.13** |
| OpenMath2 | Window Entropy | 40.89 | 43.44 | 40.55 | 42.45 |
| OpenMath2 | **CLAWS** | 超越所有基线 | — | — | — |

### 消融实验

- CLAWS 五维特征的可视化显示三类解答在注意力分布上有明显差异模式
- 创造性解答对 Reference Solutions 段的注意力较高，幻觉解答对 Response 段的自注意力异常高
- 结合 TabM 分类器时，CLAWS 性能进一步提升

### 关键发现

1. CLAWS 在大多数模型和数据集上的 F1w、F1m、APm、AUROC 四项指标均优于5种基线方法
2. CLAWS 是唯一能稳定实现三分类（而非退化为二分类）的方法
3. 创造力高的模型（如 Qwen）对 Guideline 段和 Solutions 段的注意力分配与低创造力模型存在系统性差异
4. 结合 TabM 分类器时，CLAWS 性能进一步提升

## 亮点与洞察

- **核心洞察**：创造性生成依赖模型对 prompt 各部分的差异化关注——创造性解答更多关注参考解和题目，而幻觉解答对自身回复的注意力异常高
- **实用价值**：CLAWS 只需单次推理即可完成检测，无需额外模型调用或多次生成
- **三分类能力**：首次实现对"创造性/典型/幻觉"的同时检测，而非仅做幻觉二分类
- **通用性**：跨5个 RLM 和多个数学竞赛数据集保持一致优势

## 局限性

1. 仅在数学推理任务上验证，未拓展到代码生成或科学推理等其他推理任务
2. 创造力标签依赖 GPT-o4-mini 和 Gemini-1.5-Pro 的评估质量，非真正人工标注
3. 仅测试了 7-8B 参数量的模型，对更大参数模型的注意力模式差异未知
4. Creative 类样本数量远少于其他两类（类不平衡问题），影响检测性能
5. 仅利用最后一层注意力权重，多层聚合可能提取更丰富信息

## 相关工作与启发

- **与幻觉检测的关系**：传统方法（SelfCheckGPT、INSIDE）需多次生成或外部模型，CLAWS 实现单次白盒检测
- **与 RL 推理的联系**：RL 训练的推理模型展现的创造力差异，可通过注意力模式量化
- **启发方向**：可将 CLAWS 的分段注意力分析推广到代码生成的创造力评估，或用于 RL 训练过程中筛选高质量创造性解答作为训练数据

## 评分

- ⭐ 创新性：4/5 — 首次从注意力分段角度系统化地定义和检测数学推理中的创造力
- ⭐ 实用性：4/5 — 零额外计算开销的白盒检测方案，工程上易集成
- ⭐ 实验充分度：4/5 — 5个模型、4个数据集、5种评估策略组合全面
- ⭐ 写作质量：3/5 — 框架清晰但符号较多，部分实验细节需查阅附录

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [\[ICLR 2026\] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](../../ICLR2026/aigc_detection/death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](../../ACL2026/aigc_detection/detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)

</div>

<!-- RELATED:END -->
