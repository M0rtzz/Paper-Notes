---
title: >-
  [论文解读] Cool-Fusion: Fuse Large Language Models without Training
description: >-
  [ACL 2025][LLM 其他][LLM融合] 提出 Cool-Fusion，一种无需任何训练即可融合异构 LLM 的方法，通过在文本段粒度上让多个模型互相评估和重排序生成内容，在 GSM8K 上相对最强源模型提升 17.4% 准确率。 现有痛点 现有痛点：领域现状：不同的大语言模型由于预训练数据、架构、优化器和训练方法…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "LLM融合"
  - "无训练集成"
  - "困惑度重排序"
  - "异构模型"
  - "文本段对齐"
---

# Cool-Fusion: Fuse Large Language Models without Training

**会议**: ACL 2025  
**arXiv**: [2407.19807](https://arxiv.org/abs/2407.19807)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: LLM融合, 无训练集成, 困惑度重排序, 异构模型, 文本段对齐

## 一句话总结

提出 Cool-Fusion，一种无需任何训练即可融合异构 LLM 的方法，通过在文本段粒度上让多个模型互相评估和重排序生成内容，在 GSM8K 上相对最强源模型提升 17.4% 准确率。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：不同的大语言模型由于预训练数据、架构、优化器和训练方法的差异，展现出不同的优缺点。现有的模型融合方法面临以下挑战：

**词表不兼容**：不同 LLM 的 token 词表差异巨大，例如 LLaMA-3 和 Phi-3 的共同 token 仅占总 token 的 6.4%，Phi-3 和 GLM-4 之间仅为 7.5%

**训练成本高昂**：现有融合方法通常需要微调、蒸馏或词表对齐的组合优化训练

**快速部署需求**：许多应用场景需要快速部署，无法承担训练开销

现有方法如权重合并（Model Soups）要求相同架构，传统 ensemble 要求相同词表，而 FuseLLM 和 EVA 等方法虽然能处理异构模型，但都需要不同程度的训练。Cool-Fusion 旨在提供一种完全免训练、适用于任意异构 LLM 集合的融合方案。

## 方法详解

### 整体框架

Cool-Fusion 采用迭代式文本生成循环，每次迭代包含三个步骤：

1. **生成（TextGen）**：每个源 LLM 独立生成一个文本段
2. **评估（Evaluate）**：将所有生成的文本段发送给所有源 LLM 计算困惑度，然后取平均困惑度
3. **选择（Select）**：选择平均困惑度最小的文本段作为联合预测结果，并广播更新所有模型的状态

在评估步骤中，采用平均困惑度作为选择标准有两个视角的合理性：
- **集成视角**：平均困惑度与 LLM 集成的交叉熵目标一致
- **评论家视角**：LLM 利用互补的批判能力，通过给出高困惑度来检测非事实性文本段

### 关键设计

**最短文本段（Shortest Text Segment）**：定义为从贪心解码生成的最短 token 序列中可解码的文本。不同 tokenizer 有不同的实现方式：
- LLaMA-3 类 tokenizer 提供 `word_ids` 函数，直接获取词边界
- LLaMA-2 类 tokenizer 提供 `offsets` 属性
- 其他 tokenizer 通过迭代追加 token 直到可逆解码

**对齐文本段（Aligned Text Segment）**：针对困惑度偏差问题提出的改进方案。不同 tokenizer 可能产生不同长度的最短文本段，而困惑度作为不确定性的度量，在每个词的首 token 处往往更大。这导致：
- 困惑度评估偏向更长的文本段
- 偏向产生更长平均文本段的 tokenizer

对齐文本段定义为：由 LLM 生成的、能被所有源 LLM 的 tokenizer 解码的最短文本段，从而减少因文本段长度不均导致的困惑度偏差。

**增量编解码（Incremental Encoding & Decoding）**：一些 tokenizer（如 LLaMA-2）的编解码函数是上下文依赖的，无法增量处理。解决方案是仅前置前 k=4 个已解码词的 token，确保编解码的常数时间复杂度。

**Rerank 组合**：在 Cool-Fusion 的细粒度文本段选择之外，同时让每个源 LLM 独立预测完整续文，最终通过平均困惑度对 k+1 个候选续文进行重排序，几乎无额外开销。

### 损失函数 / 训练策略

Cool-Fusion 完全无需训练。核心选择标准基于困惑度公式：

$$PPL_u(s) = \exp\left(\frac{1}{|s|}\sum_{s_i \in s} -\log p_u(s_i)\right)$$

其中 $\log p_u(s_i)$ 是 LLM $u$ 对每个 token $s_i$ 的 logit 输出。多个模型的困惑度取算术平均。

## 实验关键数据

### 主实验

实验使用 LLaMA-3 8B、Phi-3 mini (3.8B)、GLM-4 9B 三个异构模型，覆盖多个领域：

**GSM8K 数学推理**：
- LLaMA-3: 69.14% → Cool+R: 81.20%（+17.4%）
- Phi-3: 68.31% → Cool+R: 81.20%（+18.9%）
- GLM-4: 63.38% → Cool+R: 81.20%（+28.1%）

**与需训练方法比较**（GSM8K）：
- EVA（7个源LLM + 训练词表映射）: 42.91%
- PairRanker（7个源LLM + Ranker训练）: 39.58%
- Cool-Fusion（3个源LLM，无训练）: 33.5%（增量+6.6%）
- 注意：Cool-Fusion 仅用3个模型且源模型分数平均低4分

**跨领域表现**：
- Q&A 数据集（CoQA, DROP, TriviaQA）：Cool-Fusion 超越或持平最佳源模型
- 多语言 GSM：在大多数语言上超越最佳源模型
- 数学和 Unscramble：保持与最佳源模型相当的表现

### 消融实验

| 方法 | GSM8K 准确率 |
|------|-------------|
| LLaMA-3 | 69.14% |
| Phi-3 | 68.31% |
| GLM-4 | 63.38% |
| Cool2（LLaMA-3 + Phi-3）| 72.33% |
| Rerank3 | 77.79% |
| Cool-align（最短文本段）| 74.45% |
| Cool（对齐文本段）| 74.68% |
| Cool+R（Cool + Rerank）| **81.20%** |

关键发现：
- Cool2（两模型融合）即可获得显著提升（+4.6%）
- 对齐文本段略优于最短文本段（74.68% vs 74.45%），验证了困惑度偏差修正的有效性
- Rerank 本身非常有效（+12.5%），与 Cool 结合效果最佳
- Cool+R 比单独 Rerank 再提升 4.4%

### 关键发现

1. Cool-Fusion 在各领域均能超越或持平最佳源模型，即使某些源模型在特定领域表现较差也不受影响
2. 粗粒度 Rerank 和细粒度文本段选择互补，结合使用效果最佳
3. 该方法在异构模型（不同架构、不同词表大小从 32K 到 151K）间有效工作
4. FuseLLM（需蒸馏训练）在 GSM8K 上仅达 13.8%，而免训练的 Cool 达到 12.3%，几乎持平

## 亮点与洞察

1. **极简而有效**：核心思想是将 token 级集成推广到文本段级集成，巧妙绕开词表对齐问题
2. **理论合理性**：平均困惑度选择同时具有集成和评论家两个视角的支撑
3. **对齐文本段的偏差分析**：深入分析了困惑度在不同长度文本段上的偏差，提出了有理论依据的解决方案
4. **可扩展性**：给定 k 个 GPU，可扩展到 k 个源 LLM 且延迟恒定
5. **实用价值**：无需任何训练数据或微调，只要有模型推理能力即可使用

## 局限与展望

1. **推理速度**：当前实现速度约为标准 LLM 的六分之一，主要由模型间通信和频繁调用 tokenizer 导致
2. **实验规模**：受资源限制仅实验了2-3个源模型的融合
3. **评估方式**：仅使用自动指标，缺少人类或 GPT-4 评估
4. **可优化空间**：并行化 tokenizer、更长文本段减少通信开销、流水线化推理过程利用 GPU 空闲
5. 未探索是否适用于 Unicode 为基础的词表（如中文 glyph 编码）

## 相关工作与启发

- **EVA（ICLR 2024）**：通过训练词表投影矩阵实现 token 级集成，需要训练但更精细
- **LLM-Blender**：先用微调排名模型选择最优输出，再用微调 LLM 生成融合输出
- **对比解码（Contrastive Decoding）**：利用专家和业余 LLM 之间的对比，最大化 log-likelihood 差异
- **CALM**：通过模型间交叉注意力组合表示，支持新能力
- Cool-Fusion 的启发：文本级操作是跨越词表壁垒的优雅方案，可推广到其他需要跨模型协作的场景

## 评分

- **创新性**：⭐⭐⭐⭐ — 文本段级集成绕开词表对齐是巧妙的设计
- **实用性**：⭐⭐⭐⭐ — 真正的零训练方案，部署方便
- **实验充分性**：⭐⭐⭐⭐ — 覆盖数学、QA、多语言等多领域，消融详尽
- **写作质量**：⭐⭐⭐⭐ — 示例清晰，理论分析到位

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[ACL 2025\] Self-Training Elicits Concise Reasoning in Large Language Models](self-training_elicits_concise_reasoning_in_large_language_models.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)

</div>

<!-- RELATED:END -->
