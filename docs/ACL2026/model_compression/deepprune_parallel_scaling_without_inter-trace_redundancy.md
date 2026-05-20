---
title: >-
  [论文解读] DeepPrune: Parallel Scaling without Inter-Trace Redundancy
description: >-
  [ACL 2026][模型压缩][并行推理] 本文提出 DeepPrune，通过训练专门的判断模型从部分推理轨迹预测答案等价性，结合在线贪心聚类算法动态剪枝冗余的并行 CoT 路径，在保持竞争准确率（3 个百分点以内）的同时减少 65.73%-88.50% 的 token 消耗。
tags:
  - "ACL 2026"
  - "模型压缩"
  - "并行推理"
  - "CoT剪枝"
  - "推理冗余"
  - "答案等价预测"
  - "推理效率"
---

# DeepPrune: Parallel Scaling without Inter-Trace Redundancy

**会议**: ACL 2026  
**arXiv**: [2510.08483](https://arxiv.org/abs/2510.08483)  
**代码**: [https://deepprune.github.io/](https://deepprune.github.io/)  
**领域**: 模型压缩  
**关键词**: 并行推理, CoT剪枝, 推理冗余, 答案等价预测, 推理效率

## 一句话总结

本文提出 DeepPrune，通过训练专门的判断模型从部分推理轨迹预测答案等价性，结合在线贪心聚类算法动态剪枝冗余的并行 CoT 路径，在保持竞争准确率（3 个百分点以内）的同时减少 65.73%-88.50% 的 token 消耗。

## 研究背景与动机

**领域现状**：并行扩展（如 best-of-n 采样）通过同时生成多条推理轨迹来增强 LLM 推理能力，总 token 消耗可达 100M+。现有高效推理方法主要关注序列扩展的过度思考问题，对并行扩展的效率研究较少。

**现有痛点**：(1) 超过 80% 的并行推理轨迹产生相同的最终答案，代表了大量浪费的计算；(2) 基于置信度的早停方法无法减少轨迹间冗余，且有过早终止正确推理的风险；(3) 浅层语义相似度（如 SentenceBERT）无法从早期推理阶段预测最终答案等价性。

**核心矛盾**：并行扩展的收益来自答案多样性（少数不同答案中可能包含正确答案），但绝大多数（80%+）并行轨迹产生相同答案，多样性极低。

**本文目标**：在保留答案多样性的前提下，主动剪枝冗余的并行推理轨迹。

**切入角度**：训练专门的判断模型来理解推理过程的深层语义，从部分推理轨迹预测两条轨迹是否最终会得到相同答案。

**核心 idea**：早期发现答案等价 → 保留多样轨迹 + 剪枝冗余轨迹 → 高效并行扩展。

## 方法详解

### 整体框架

DeepPrune 包含两个组件：(1) 判断模型——从部分推理轨迹预测两条轨迹的答案是否等价（AUROC 0.7072）；(2) 在线贪心聚类——在推理进行中，将轨迹动态聚类为答案等价组，剪枝每组中的冗余轨迹，仅保留一条代表轨迹。

### 关键设计

1. **答案等价判断模型**:

    - 功能：从部分推理轨迹预测最终答案是否相同
    - 核心思路：基于 Qwen3-4B 训练，使用 OOD 数据（AIME 2022/2023 和 MATH 500）+ 过采样技术平衡正负样本。输入为两条轨迹的前 N 个 token，输出为答案等价概率
    - 设计动机：浅层相似度方法（AUROC=0.58）和通用 LLM（AUROC=0.66）都不够准确，需要专门训练的模型理解推理过程

2. **在线贪心聚类与动态剪枝**:

    - 功能：在推理过程中实时剪枝冗余路径
    - 核心思路：维护答案等价组的集合，每产生新轨迹片段就用判断模型检查是否与已有组等价。如果等价则剪枝（停止生成），如果不等价则创建新组。保留每组的一条代表轨迹继续生成
    - 设计动机：在线处理比事后剪枝节省更多计算，贪心策略在实践中平衡了效率和多样性

3. **OOD 泛化训练策略**:

    - 功能：确保判断模型在未见过的推理模型上也有效
    - 核心思路：在 AIME 2022/2023 和 MATH 500 上训练（与评估数据 AIME 2024/2025 不重叠），泛化到不同推理模型生成的轨迹
    - 设计动机：实际部署中不可能为每个新推理模型重新训练判断模型

### 损失函数 / 训练策略

判断模型使用二分类交叉熵损失，过采样少数类（不等价对）平衡数据。训练数据来自多个推理模型的并行轨迹对。

## 实验关键数据

### 主实验

**与标准共识采样的对比（LLaDA 推理模型）**

| 方法 | Token 减少率 | 准确率差异 |
|------|------------|----------|
| 标准共识采样 | 0% | 基线 |
| 置信度早停 | ~30% | 可能损害 |
| **DeepPrune** | **65.73%-88.50%** | **≤3%** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 判断模型 AUROC | 0.7072（OOD 泛化） |
| SentenceBERT 基线 | 0.58（接近随机） |
| 通用 LLM 基线 | 0.66（次优） |

### 关键发现

- DeepPrune 在三个挑战性基准（AIME 2024、AIME 2025、GPQA）上减少 65-88% token
- 准确率损失控制在 3 个百分点以内
- 判断模型成功泛化到未见过的推理模型
- 剪枝保留了答案多样性——高多样性轨迹不会被误剪

## 亮点与洞察

- 定量揭示了并行推理的核心效率问题：80%+ 的轨迹产生相同答案
- 从"推理理解"而非"文本相似"出发训练判断模型，是对浅层方法的重要改进
- 在线剪枝设计使得加速在推理过程中即时生效

## 局限与展望

- 判断模型的 AUROC（0.7072）仍有提升空间，可能导致少量有价值轨迹被误剪
- 在线聚类的贪心策略可能次优
- 依赖特定的判断阈值，不同场景可能需要调整
- 仅在数学推理任务上验证，其他推理类型的效果待确认

## 相关工作与启发

- **vs 置信度早停**: 置信度方法不能减少轨迹间冗余，DeepPrune 直接解决冗余问题
- **vs 序列剪枝**: 序列方法减少单条轨迹的长度，DeepPrune 减少并行轨迹的数量

## 评分

- 新颖性: ⭐⭐⭐⭐ 并行推理冗余分析和答案等价判断模型是新颖贡献
- 实验充分度: ⭐⭐⭐⭐ 三个基准、多模型验证、OOD 泛化测试
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法直观
- 价值: ⭐⭐⭐⭐ 为推理时并行扩展的效率化提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Parallel Token Prediction for Language Models](../../ICLR2026/model_compression/parallel_token_prediction_for_language_models.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ACL 2026\] VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection](veccisc_improving_confidence-informed_self-consistency_with_reasoning_trace_clus.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)

</div>

<!-- RELATED:END -->
