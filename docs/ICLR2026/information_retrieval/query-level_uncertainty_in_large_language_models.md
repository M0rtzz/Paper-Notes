---
title: >-
  [论文解读] Query-Level Uncertainty in Large Language Models
description: >-
  [ICLR2026][uncertainty estimation] 提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。
tags:
  - ICLR2026
  - uncertainty estimation
  - knowledge boundary
  - adaptive inference
  - training-free
  - internal confidence
---

# Query-Level Uncertainty in Large Language Models

**会议**: ICLR2026  
**arXiv**: [2506.09669](https://arxiv.org/abs/2506.09669)  
**代码**: [GitHub](https://github.com/tigerchen52/query_level_uncertainty)  
**领域**: llm_reasoning  
**关键词**: uncertainty estimation, knowledge boundary, adaptive inference, training-free, internal confidence

## 一句话总结
提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

## 背景与动机
1. LLM存在知识边界，无法准确回答所有问题；知识边界感知对构建可信高效AI系统至关重要
2. 现有不确定性估计多为answer-level（生成后评估），计算开销大（需完整生成答案）
3. 自适应推理（如RAG、慢思考、模型级联）需要pre-generation信号来决定是否触发额外资源
4. 已有query-level方法需训练探针或微调模型（如IDK token、R-Tuning），泛化性受限
5. LLM内部隐藏状态蕴含丰富的知识可达性信息，跨层一致性可提升输出质量

## 方法详解

### 核心思路
用yes/no自评提示让LLM判断能否回答查询，提取P(Yes)作为置信度，无需生成答案。提示格式："Respond only with 'Yes' or 'No' to indicate whether you are capable of answering the {Query} accurately."

### 从P(Yes)到Internal Confidence
**基础P(Yes)**：仅用最后一层最后一个token的隐状态$\mathbf{h}_N^{(L)}$，通过unembedding矩阵投影到[Yes, No]词表上做softmax——类似无训练的linear probing。

**Internal Confidence扩展**：不仅用最后一个位置，而是跨所有层和token位置计算P(Yes)，用加权平均聚合：
$$\text{IC}(\mathbf{h}) = \sum_{n=1}^{N}\sum_{l=1}^{L}w_n^{(l)} P(\text{Yes}|\mathbf{h}_n^{(l)})$$

其中权重用**Attenuated Encoding**计算：$\delta_j^{(i)} = \exp(-\alpha|i-j|^2) / Z$，以"决策中心"(默认为最后层最后token)为圆心，随距离指数衰减。参数α=1.0控制locality：值越大权重越集中在中心附近。

### 设计激励
- AUROC热力图显示“决策中心”确实存在：最佳分辨位置并非最后层最后token，但近似效果良好
- 固定决策中心避免了需要验证集来确定最优位置，保持training-free特性
- 跨层聚合利用了LLM中间层编码的丰富知识，而非仅依赖最终表示

### 应用场景
1. **自适应RAG**：IC较低时触发检索增强，IC较高时直接回答，减屑50%+的RAG调用而性能几乎不降
2. **模型级联**：小模型IC较低时将查询转发给大模型，实现成本-质量最优平衡
3. **弃权策略**：对高不确定性查询拒绝回答，提升可信度

## 实验关键数据

### 主实验（跨模型跨任务）

| 方法 | Phi-3.8B AUROC | Llama-8B AUROC | Qwen-14B AUROC | Avg AUROC |
|------|:--:|:--:|:--:|:--:|
| Max(-log p) | 54.0 | 56.3 | 57.8 | 56.0 |
| Predictive Entropy | 57.9 | 60.1 | 62.4 | 60.1 |
| Semantic Entropy | 55.6 | 59.7 | 60.0 | 58.4 |
| P(Yes) top-right | 57.3 | 60.5 | 60.9 | 59.6 |
| **Internal Confidence** | **60.8** | **64.7** | **67.1** | **64.2** |

### 与Answer-level方法对比（速度）

| 方法 | GSM8K AUROC | 毛秒/样本 |
|------|:--:|:--:|
| IC(本文) | 66.8 | **0.3** |
| Predictive Entropy | 61.0 | 9.8 |
| Min-K Entropy | 60.4 | 3.8 |
| Semantic Entropy | 60.0 | 151.8 |

**关键发现**:
1. IC在3个数据集3个模型上一致优于所有baseline
2. 相比answer-level方法快32×-602×，并且精度更高
3. RAG场景下可在性能几乎不降的情况下减屑50%+的RAG调用
4. 模型越大IC效果越显著，因为更大的模型有更好的自我知识边界感知
5. 决策中心附近的层和token信息最有区分力，与AUROC热力图观察一致

## 亮点
- 首次形式化定义query-level uncertainty，将不确定性估计从"后验"推向"先验"
- 完全training-free，单次前向传播，实用性极强
- 跨层跨token的加权聚合策略（Attenuated Encoding）简洁有效
- 在RAG和模型级联中展示了显著的效率-质量权衡优势

## 局限
- 决策中心固定在最后层最后token，非最优（但为保持training-free的权衡）
- 仅在有明确答案的任务（事实QA/数学推理）上验证，开放式生成未涉及
- 贪心解码作为知识边界定义的代理较保守，可能低估模型能力
- 对reasoning-heavy任务（如多步数学推理）的区分能力相对弱于factual QA

## 相关工作
- Answer-level不确定性: Semantic Entropy (Kuhn et al. 2023), P(True) (Kadavath et al. 2022)
- 知识边界检测: IDK token (Cohen et al. 2024), R-Tuning (Zhang et al. 2024a) — 均需训练
- 内部状态探针: Gottesman & Geva 2024训练轻量探针; Semantic Entropy Probes (Kossen et al. 2024)

## 评分
- 新颖性: ⭐⭐⭐⭐ (query-level概念新颖，方法简洁)
- 实验充分度: ⭐⭐⭐⭐ (3模型3数据集，多应用场景)
- 写作质量: ⭐⭐⭐⭐⭐ (问题定义清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (对自适应推理有直接实用价值)

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](../../AAAI2026/information_retrieval/as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](../../ACL2025/information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](../../ACL2025/information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](../../ACL2025/information_retrieval/rare_retrieval_augmented_reasoning.md)

<!-- RELATED:END -->
