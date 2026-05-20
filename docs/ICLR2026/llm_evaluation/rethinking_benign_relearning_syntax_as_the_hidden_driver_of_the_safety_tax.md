---
title: >-
  [论文解读] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures
description: >-
  [ICLR 2026][LLM评测][machine_unlearning] 本文揭示了 LLM 机器遗忘中"良性重学习"（benign relearning）的真正驱动因素不是主题相关性而是**句法相似性**…
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "machine_unlearning"
  - "LLM_safety"
  - "syntactic_similarity"
  - "benign_relearning"
---

# Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures

**会议**: ICLR 2026  
**arXiv**: [2602.03379](https://arxiv.org/abs/2602.03379)  
**代码**: 未公开  
**领域**: LLM评测  
**关键词**: machine_unlearning, LLM_safety, syntactic_similarity, benign_relearning  

## 一句话总结

本文揭示了 LLM 机器遗忘中"良性重学习"（benign relearning）的真正驱动因素不是主题相关性而是**句法相似性**，并提出**句法多样化（syntactic diversification）**策略来提升遗忘的鲁棒性。

## 研究背景与动机

机器遗忘旨在从已训练的模型中移除特定内容，同时保持整体性能。然而，"良性重学习"现象表明，即使在看似无关的良性数据上微调，已遗忘的信息也会重新浮现。

**现有认知的局限**：
- BLUR 基准将良性重学习归因于**主题相关性**（topical relevance），即重学习数据与遗忘数据在实体/主题上的重叠
- 例如：遗忘哈利·波特的段落后，在关于同一角色的 GPT 生成的描述上微调就能恢复已遗忘内容
- 这种直觉性的解释被广泛接受，但作者发现它并不完整

**关键发现**：
- BLUR 实验存在两个混淆因素：(1) 不同相关度数据集的大小不一致，导致梯度更新次数不同；(2) 恢复程度并非单调递增，仅在 epoch 末尾评估可能错过峰值
- 在公平评估下（标准化步数预算 + 报告最大恢复值），主题相关性的优势大幅消失

## 方法详解

### 整体框架

研究分为三个阶段：
1. **重新评估主题相关性**：在 WMDP、WHP、RWKU 基准上，统一评估步数后发现主题相关性不是主要驱动因素
2. **验证句法相似性**：在 TOFU 数据集上，构造受控实验比较主题相关集 vs 句法相似集
3. **提出句法多样化**：通过 GPT-4o 对遗忘集进行多样化改写，打破句法刚性

### 关键设计：句法相似性度量

使用归一化 Levenshtein 距离来量化句法相似性：

$$\text{Sim}(s_1, s_2) = 1 - \frac{d_{\text{Lev}}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

其中 $d_{\text{Lev}}$ 为最少的单字符编辑次数。该度量范围为 [0, 1]，纯粹衡量表面结构对齐，不涉及语义。

### 受控实验设计（TOFU 数据集）

在 TOFU 的 forget05 场景下（遗忘 10 位虚构作者的知识），定义：
- **目标集 $D_{\text{target}}$**：询问作者全名的 QA 对
- **主题相关重学集 $D_{\text{relearn}}^{\text{topic}}$**：关于同一作者的非姓名问题（如出生地、职业）
- **句法相似重学集 $D_{\text{relearn}}^{\text{syntactic}}$**：与目标集句式相同但关于不同作者的姓名问题

关键数据：$D_{\text{relearn}}^{\text{syntactic}}$ 与 $D_{\text{target}}$ 的句法相似度为 0.4513，而 $D_{\text{relearn}}^{\text{topic}}$ 仅为 0.2349。

### 为什么句法相似性驱动重学习

**表征和梯度对齐分析**：
- 句法相似集在遗忘模型中与目标集的隐藏表征余弦相似度和梯度余弦相似度都远高于主题相关集
- 这意味着句法重叠将模型的隐藏表征和优化方向都拉回被遗忘内容的方向

**模板 vs 关键词遗忘分析**：
- 将目标回答的 token 分为**模板 token**（通用短语）和**关键词 token**（需要遗忘的具体信息如作者名）
- 损失比率定义为：$\text{Loss Ratio} = \frac{\mathcal{L}_{\text{template}}}{\mathcal{L}_{\text{keyword}}}$
- 在遗忘过程中，损失比率持续上升，说明遗忘主要抑制了模板而非真正的关键词
- 当用句法相似数据微调时，被抑制的模板结构快速恢复，带动关键词重新浮现

### 句法多样化策略

具体流程：
1. 使用 GPT-4o 对 $D_{\text{forget}}$ 中的目标查询生成多种句法变体
2. 过滤保留低相似度的改写，确保多样性
3. 用多样化后的 $D_{\text{forget}}'$ 替代原始数据进行遗忘

效果：$D_{\text{relearn}}^{\text{syntactic}}$ 与 $D_{\text{forget}}'$ 的平均句法相似度从 0.4513 降至 0.2241。

### 损失函数

论文评估了三种标准遗忘方法：
- **Gradient Ascent (GA)**：在遗忘集上最大化损失
- **Negative Preference Optimization (NPO)**：通过偏好优化抑制遗忘内容
- **SCRUB**：结合遗忘和保留集的联合优化

句法多样化作为预处理步骤，可与上述任何方法组合。

## 实验关键数据

### 主实验：句法相似性 vs 主题相关性在 TOFU 上的重学习效果

| 遗忘方法 | 重学集类型 | 遗忘步50后重学效果 |
|---------|-----------|-----------------|
| GA | $D_{\text{relearn}}^{\text{topic}}$ | 无恢复 |
| GA | $D_{\text{relearn}}^{\text{syntactic}}$ | 少量更新即恢复关键词 |
| NPO | $D_{\text{relearn}}^{\text{topic}}$ | 微弱恢复 |
| NPO | $D_{\text{relearn}}^{\text{syntactic}}$ | 显著恢复 |
| SCRUB | $D_{\text{relearn}}^{\text{topic}}$ | 有限恢复 |
| SCRUB | $D_{\text{relearn}}^{\text{syntactic}}$ | 完全恢复遗忘内容 |

所有遗忘方法中，句法相似集的恢复效果都一致且显著优于主题相关集。SCRUB 虽然遗忘速度最快，但对重学习最脆弱。

### 消融实验：句法多样化的效果

**模型效用保持（GA 方法）**：

| 指标 | $D_{\text{forget}}$ | $D_{\text{forget}}'$ (Ours) |
|------|---------------------|---------------------------|
| Real Authors ROUGE↑ | 0.2608 | **0.4257** |
| Real Authors Prob↑ | 0.3665 | **0.4223** |
| Real Authors TR↑ | 0.5769 | **0.6075** |
| World Facts Avg↑ | 0.6056 | **0.6104** |
| Retain Set ROUGE↑ | 0.1036 | **0.4052** |
| Retain Set Avg↑ | 0.1607 | **0.3128** |

句法多样化不仅提升了遗忘鲁棒性，还显著改善了模型效用（特别是 Retain Set ROUGE 从 0.10 跃升至 0.41）。

### BLUR 基准重分析

| 基准 | $D_{\text{hi}}$ 句法相似度 | $D_{\text{mid}}$ | $D_{\text{low}}$ |
|------|--------------------------|-------------------|-------------------|
| WMDP | 0.2244 | 0.2059 | 0.1771 |
| WHP | 0.1894 | 0.1767 | 0.1818 |
| RWKU | 0.2250 | 0.2215 | 0.1883 |

WHP 中 $D_{\text{low}}$（Lorem Ipsum 填充文本）的句法相似度与 $D_{\text{hi}}$、$D_{\text{mid}}$ 接近，解释了其重学习效果为何也相当。

### 关键发现

1. **句法相似性 > 主题相关性**：在所有基准和遗忘方法中，句法相似性始终是良性重学习的主要驱动因素
2. **遗忘的偏斜性**：当前遗忘方法过度抑制模板 token 而非关键词 token，造成结构性脆弱
3. **句法多样化的三重收益**：(a) 抑制重学习，(b) 加速遗忘，(c) 缓解遗忘效果与模型效用的权衡
4. **安全训练 ≠ 遗忘**：DPO 等安全训练仅抑制输出而不移除知识，在句法重学习下更脆弱
5. **LoRA 的隐患**：LoRA 微调虽然参数少，但在重学习场景下恢复更快更有效

## 亮点与洞察

- **视角转换**：从语义层面（主题相关性）转向表面形式层面（句法相似性）来理解遗忘失败，是一个反直觉但实验充分的发现
- **实验设计精巧**：通过构造只共享句法模式但无主题重叠的数据集，干净地分离了两种因素
- **实用性强**：句法多样化策略简单易实现，只需一个 LLM 改写步骤，效果显著
- **安全意义重大**：揭示了实际部署中难以防御的攻击路径——句法相似但内容无关的微调数据即可恢复遗忘知识

## 局限性

1. 实验主要在 TOFU（合成数据集）上进行，真实场景中非结构化文本的句法多样性更高
2. 句法多样化依赖 GPT-4o 的改写质量，本身引入了额外成本
3. 仅评估了 Llama-2-7b-chat 和 Phi 两个模型家族，更大规模模型的行为待验证
4. 句法相似性的度量（Levenshtein 距离）较为简单，更复杂的语法结构对齐未被捕捉

## 相关工作与启发

- **BLUR (Hu et al., 2025b)**：提出了主题相关性三级划分，本文通过实验设计上的改进推翻了其核心结论
- **TOFU (Maini et al., 2024)**：标准的 LLM 遗忘基准，本文在此基础上设计了精细的受控实验
- **GA/NPO/SCRUB**：三种主流遗忘方法，本文揭示它们共享相同的句法脆弱性
- 对遗忘鲁棒性评估的启示：除了评估内容层面的恢复，还应关注结构层面的攻击面

## 评分

- **创新性**: ⭐⭐⭐⭐ — 识别句法相似性作为重学习驱动因素是新颖的洞察
- **实验设计**: ⭐⭐⭐⭐⭐ — 受控实验设计精巧，消融充分，分析深入
- **实用性**: ⭐⭐⭐⭐ — 句法多样化策略简单有效
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，图表直观
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_evaluation/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[ACL 2026\] Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs](../../ACL2026/llm_evaluation/gated_tree_cross-attention_for_checkpoint-compatible_syntax_injection_in_decoder.md)
- [\[NeurIPS 2025\] Reliably Detecting Model Failures in Deployment Without Labels](../../NeurIPS2025/llm_evaluation/reliably_detecting_model_failures_in_deployment_without_labels.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/llm_evaluation/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](../../NeurIPS2025/llm_evaluation/rethinking_losses_for_diffusion_bridge_samplers.md)

</div>

<!-- RELATED:END -->
