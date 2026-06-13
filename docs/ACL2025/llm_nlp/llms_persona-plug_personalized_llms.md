---
title: >-
  [论文解读] LLMs + Persona-Plug = Personalized LLMs
description: >-
  [ACL 2025][LLM 其他][个性化生成] 提出 PPlug 模型，通过轻量级插件式用户嵌入器将用户历史行为压缩为单一个性化嵌入，以 plug-and-play 方式引导 LLM 生成个性化输出，在 LaMP 基准上显著超越检索式和微调式基线，最高提升 35.8%。 当前 LLM 的主流使用模式是"一刀切"：对不同用…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "个性化生成"
  - "用户建模"
  - "插件式方法"
  - "用户嵌入"
  - "语言模型个性化"
---

# LLMs + Persona-Plug = Personalized LLMs

**会议**: ACL 2025  
**arXiv**: [2409.11901](https://arxiv.org/abs/2409.11901)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 个性化生成, 用户建模, 插件式方法, 用户嵌入, 语言模型个性化

## 一句话总结

提出 PPlug 模型，通过轻量级插件式用户嵌入器将用户历史行为压缩为单一个性化嵌入，以 plug-and-play 方式引导 LLM 生成个性化输出，在 LaMP 基准上显著超越检索式和微调式基线，最高提升 35.8%。

## 研究背景与动机

当前 LLM 的主流使用模式是"一刀切"：对不同用户的相同输入给出几乎相同的回复，忽略了用户个体差异。个性化 LLM 成为重要研究方向，但现有方案各有不足：

**微调式方法**（如 OPPU、LoRA per user）：为每个用户训练独立 LLM，训练和推理成本极高，无法大规模部署

**检索式方法**（如 RAG-based）：从用户历史中检索相关行为作为 in-context demonstration，但仅关注与当前输入最相关的几条历史，容易丢失用户的整体风格和偏好

核心问题：如何既捕获用户的整体风格偏好，又不修改 LLM 本身的参数？

## 方法详解

### 整体框架

PPlug 包含三个核心组件：
1. **用户行为编码器（User Behavior Encoder）**：将每条用户历史行为编码为稠密向量
2. **输入感知个性化聚合器（Input-aware Personal Aggregator）**：根据当前输入加权聚合所有历史行为向量为一个个性化嵌入
3. **LLM 个性化生成（PPlug for LLM Personalization）**：将个性化嵌入附加到 LLM 输入中引导生成

### 关键设计

**用户行为编码器**：使用 BGE-base 等编码器模型将每条历史行为 $h_i^u$ 编码为向量 $\mathbf{h}_i^u = \text{Enc}^{\text{his}}(h_i^u)$。使用小型编码器而非 LLM 的原因有二：（1）双向注意力能更好捕获行为信息；（2）轻量级编码器提升效率（仅 220M 参数，占 7B LLM 的 3.1%）。历史行为编码器的参数冻结不训练，仅微调输入编码器。

**输入感知个性化聚合器**：不同于简单平均所有历史向量，PPlug 采用注意力机制动态分配权重：

$$w_i = \frac{\exp(\mathbf{x}^{u\top} \mathbf{h}_i^u)}{\sum_k \exp(\mathbf{x}^{u\top} \mathbf{h}_k^u)}$$

$$\mathbf{P}^u = \sum_i w_i \cdot \text{Proj}(\mathbf{h}_i^u)$$

其中 Proj 是将编码器空间映射到 LLM 表示空间的 2 层 MLP。这种设计使得生成学术标题时，模型自动更关注与当前摘要话题一致的历史标题。

**LLM 个性化生成**：将个性化嵌入 $\mathbf{P}^u$、可训练指令嵌入 $\mathbf{I}$、和当前输入的 LLM 嵌入拼接后送入 LLM：

$$\mathbf{X}_i^u = [\mathbf{I}; \mathbf{P}^u; \text{Emb}_{\text{LLM}}(x^u); \text{Emb}_{\text{LLM}}(y_{<i}^u)]$$

关键特点：LLM 参数完全冻结，仅训练指令嵌入 $\mathbf{I}$、输入编码器和投影器。

### 损失函数 / 训练策略

使用标准的 next token prediction 损失在所有用户数据上端到端优化：

$$\mathcal{L} = -\sum_u \sum_i \log p_{\text{LLM}}(y_i^u | \mathbf{X}_i^u)$$

训练配置：AdamW 优化器，学习率 1e-4，warmup ratio 0.05，batch size 64，训练 2 epochs（LaMP-3 因数据量大只训练 1 epoch）。生成时使用 beam search（beam size=4）。

## 实验关键数据

### 主实验

在 LaMP 基准 6 个任务上的表现（表1）：

| 任务 | 最佳基线 | PPlug | 相对提升 |
|------|---------|-------|---------|
| LaMP-1 引用识别 (Acc) | 0.682 | 0.680 | -0.3% |
| LaMP-2 电影标签 (Acc) | 0.416 | 0.565 | +35.8% |
| LaMP-2 电影标签 (F1) | 0.337 | 0.501 | +48.7% |
| LaMP-3 产品评分 (MAE↓) | 0.246 | 0.231 | +6.1% |
| LaMP-4 新闻标题 (R-1) | 0.207 | 0.216 | +4.3% |
| LaMP-5 学术标题 (R-1) | 0.480 | 0.487 | +1.5% |
| LaMP-7 推文改写 (R-1) | 0.468 | 0.536 | +14.5% |

在 LaMP-2 和 LaMP-7 上的巨大提升（35.8% 和 14.5%）表明这两个任务更依赖用户整体风格而非单条历史记录。

### 消融实验

**输入感知注意力的影响**：移除后改为简单平均，性能下降但仍优于基线，说明即便不区分权重，整体行为压缩也是有效的。

**指令嵌入的影响**：移除后性能略降，说明指令嵌入帮助 LLM 分离全局任务知识和用户特定模式，但主要贡献来自个性化嵌入本身。

**与检索策略的整合**（表3）：PPlug + Retrieval 在多任务上进一步提升。PPlug 提供粗粒度用户风格，检索提供细粒度任务相关上下文，两者互补。

**LLM 和编码器分析**（表2）：
- 使用 FlanT5-XXL (11B) + BGE-base 获得最佳结果
- 性能与 LLM 大小正相关（FlanT5-XXL > Llama2 7B > FlanT5-XL 3B）
- BGE-base 和 Contriever 作为编码器性能相当，验证了方法的鲁棒性

### 关键发现

1. 将用户所有历史行为压缩为单一嵌入向量是比选择性检索更有效的个性化策略
2. 需要整体风格理解的任务（电影偏好、推文风格）受益最大
3. 粗粒度用户嵌入和细粒度检索 demonstration 是互补的
4. PPlug 可以端到端训练，比 ROPG/RSPG 的强化学习/知识蒸馏更简洁高效

## 亮点与洞察

- **设计的优雅性**：仅添加一个用户嵌入向量就能显著提升个性化效果，且 LLM 无需任何修改
- **plug-and-play 部署优势**：服务提供商只需部署一个 LLM，为不同用户提供不同的嵌入输入即可，极大简化基础设施
- **端到端优化**：相比基于 RL 反馈的检索模型优化，PPlug 的直接梯度训练更稳定高效
- **全局 vs 局部偏好的平衡**：输入感知注意力使模型能在保持全局偏好的同时关注与当前任务相关的历史

## 局限与展望

- 仅在 FlanT5 和 Llama2 等较早期模型上实验，更强的 LLM（GPT-4、Llama3）上的效果未验证
- 编码器使用 BGE-base（220M 参数），对于特别长或复杂的用户历史，表达能力可能不足
- 仅利用了用户历史行为数据，未扩展到用户属性（年龄、地区等）
- 个性化嵌入的压缩为单一向量可能丢失多面性的用户偏好信息
- 粗粒度嵌入 vs 细粒度检索的最优组合策略值得进一步探索
- 未考虑用户偏好随时间变化的动态性

## 相关工作与启发

- 与微调式方法（OPPU、per-user LoRA）相比：PPlug 无需为每个用户训练单独模型，是其高效替代
- 与检索式方法（ROPG、RSPG）相比：PPlug 整合全部历史而非仅检索最相关的几条，能更好捕获整体风格
- 未来可探索的方向：多粒度个性化嵌入（粗+细结合）、时序感知的用户建模、跨任务迁移的通用用户嵌入

## 评分

- **新颖性**：8/10 — 将用户全部历史压缩为单一嵌入的 plug-and-play 思路新颖实用
- **技术深度**：7/10 — 方法较为直接但设计合理
- **实验充分性**：8/10 — 全面的基线对比、消融、LLM/编码器分析
- **实用价值**：9/10 — 部署友好，单模型服务多用户
- **总体评分**：8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
