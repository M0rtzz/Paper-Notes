---
title: >-
  [论文解读] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs
description: >-
  [ACL 2025 (Oral)][模型压缩][知识蒸馏] 证明了朴素的 Top-K 稀疏知识蒸馏会产生有偏估计，提出基于重要性采样的 Random Sampling Knowledge Distillation (RSKD) 方法，提供无偏梯度估计，仅需存储极度稀疏的 logits，训练开销仅比交叉熵增加不到 10%，性能与全量蒸馏持平。
tags:
  - ACL 2025 (Oral)
  - 模型压缩
  - 知识蒸馏
  - 稀疏logit
  - 重要性采样
  - LLM预训练
  - 无偏估计
---

# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs

**会议**: ACL 2025 (Oral)  
**arXiv**: [2503.16870](https://arxiv.org/abs/2503.16870)  
**代码**: [GitHub](https://github.com/SamsungLabs/RSKD)  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, 稀疏logit, 重要性采样, LLM预训练, 无偏估计  

## 一句话总结

证明了朴素的 Top-K 稀疏知识蒸馏会产生有偏估计，提出基于重要性采样的 Random Sampling Knowledge Distillation (RSKD) 方法，提供无偏梯度估计，仅需存储极度稀疏的 logits，训练开销仅比交叉熵增加不到 10%，性能与全量蒸馏持平。

## 研究背景与动机

**领域现状**：知识蒸馏（KD）是将大模型（teacher）的知识迁移到小模型（student）的经典技术。在 LLM 时代，KD 被视为降低部署成本的重要手段。一个自然的实现路径是：预先计算并缓存 teacher 的输出 logits，然后在训练 student 时加载使用，这样可以避免反复运行 teacher 前向传播。

**现有痛点**：LLM 的词表通常很大（32k-128k），完整缓存每个 token 的全部 logit 向量需要巨大的存储空间。一个直觉的解决方案是只缓存 Top-K 个 logit（如 K=10 或 100），然而这种做法在理论上是否正确此前没有被严格分析过。将稀疏蒸馏应用于 LLM 预训练阶段更是几乎未被探索。

**核心矛盾**：存储效率和蒸馏质量之间的 trade-off。缓存全部 logits 太耗存储，Top-K 缓存虽然节省空间但会引入偏差——被截断的概率质量无法正确归一化，导致 student 学到错误的概率分布。

**本文目标**：设计一种既稀疏又无偏的 logit 缓存策略，使得 LLM 知识蒸馏在存储效率和性能之间取得最优平衡。

**切入角度**：作者从统计学角度切入，首先严格证明 Top-K 方法产生有偏梯度估计，然后利用重要性采样理论设计无偏替代方案。

**核心 idea**：用随机采样（而非确定性 Top-K）来选择要缓存的 logit 位置，通过重要性权重修正使梯度估计无偏，同时只需存储极少量的 logits。

## 方法详解

### 整体框架

训练流程分为两个阶段：(1) 离线阶段——运行 teacher 模型对训练数据进行推理，对每个 token 位置随机采样 K 个 logit 索引及其值进行存储；(2) 在线阶段——训练 student 模型时，从缓存加载稀疏 logits，利用重要性采样权重构造蒸馏损失。整体训练流程与标准 KD 相同，仅在 logit 存储和损失计算方式上有区别。

### 关键设计

1. **Top-K 偏差的理论分析**:

    - 功能：从理论上证明朴素 Top-K 稀疏 KD 的问题
    - 核心思路：当只保留 teacher 概率分布的 Top-K 个 token 时，剩余的概率质量被丢弃或均匀分配到其余 token 上，这导致 student 看到的是一个被扭曲的分布。形式化地，$\hat{p}_{\text{Top-K}}$ 和真实 $p_{\text{teacher}}$ 之间的 KL 散度损失梯度不一致，student 无法正确学习 tail distribution
    - 设计动机：为后续提出无偏方案提供理论根基，也解释了为什么直接用 Top-K KD 会导致 calibration 变差

2. **Random Sampling Knowledge Distillation (RSKD)**:

    - 功能：提供无偏的稀疏知识蒸馏方案
    - 核心思路：不选择固定的 Top-K，而是根据 teacher 的概率分布按比例随机采样 K 个 token 位置进行缓存。对于采样到的 token $i$，记录其 logit 值和采样概率 $q(i)$，然后构造重要性采样估计：$\hat{\mathcal{L}}_{\text{KD}} = -\sum_{i \in S} \frac{p_T(i)}{q(i)} \log p_S(i)$，其中 $S$ 是采样集合。这保证了 $\mathbb{E}[\nabla \hat{\mathcal{L}}] = \nabla \mathcal{L}_{\text{full KD}}$，即梯度在期望意义下无偏
    - 设计动机：重要性采样是蒙特卡洛估计中的经典方法，将其引入知识蒸馏场景，可以在保持无偏性的同时大幅减少需要存储的 logit 数量

3. **自适应采样概率设计**:

    - 功能：优化采样效率，降低梯度估计的方差
    - 核心思路：采样概率 $q(i)$ 的选择直接影响估计方差。作者设计了一种与 teacher 概率值和 student 学习信号相关的采样分布，使高概率和高信息量的 token 更容易被采样，从而降低方差。实际实现中采用 teacher 的 softmax 概率作为采样分布
    - 设计动机：均匀随机采样虽然也无偏，但方差很大。自适应采样就像"聪明地选择样本"，在同样的稀疏度下获得更稳定的梯度

### 损失函数 / 训练策略

最终训练损失为标准交叉熵（CE）损失和 RSKD 蒸馏损失的加权组合：$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1-\alpha) \cdot \hat{\mathcal{L}}_{\text{RSKD}}$。RSKD 损失部分通过重要性权重确保无偏，整体训练开销比纯 CE 训练仅增加不到 10%。

## 实验关键数据

### 主实验

| 模型规模 | 蒸馏方法 | 困惑度 (PPL) ↓ | 下游平均准确率 ↑ | 存储开销 |
|----------|---------|---------------|----------------|---------|
| 300M | CE only (无蒸馏) | 基线 | 基线 | 0 |
| 300M | Full KD | 最优 | 最优 | 100% |
| 300M | Top-K KD (K=10) | 次优，但有偏 | 低于 Full KD | ~0.01% |
| 300M | RSKD (K=10) | 接近 Full KD | 接近 Full KD | ~0.01% |
| 1B | Full KD | 最优 | 最优 | 100% |
| 1B | RSKD (K=10) | 接近 Full KD | 接近 Full KD | ~0.01% |
| 3B | Full KD | 最优 | 最优 | 100% |
| 3B | RSKD (K=10) | 接近 Full KD | 接近 Full KD | ~0.01% |

### 消融实验

| 配置 | PPL | 说明 |
|------|-----|------|
| RSKD (K=10) | 接近最优 | 仅存储 10 个 logit 即可接近全量 KD |
| RSKD (K=5) | 略有下降 | 进一步稀疏化，仍优于 Top-K |
| Top-K (K=100) | 中等 | 即使 K 增大 10 倍仍有偏差 |
| Top-K (K=10) | 较差 | 偏差最明显 |
| 均匀采样 (K=10) | 高方差 | 无偏但不稳定 |
| RSKD + 自适应采样 | 最优 | 无偏且低方差 |

### 关键发现

- Top-K 方法即使 K=100 也无法消除偏差，而 RSKD 在 K=10 时就能接近全量蒸馏效果，存储比 Top-K 还少
- RSKD 在 calibration（模型概率估计的可靠性）方面显著优于 Top-K，验证了理论分析中关于偏差的推论
- 从 300M 到 3B 规模，RSKD 的优势一致保持，证明方法具有良好的可扩展性
- 训练时间开销仅增加不到 10%，主要来自读取缓存 logits 和计算重要性权重

## 亮点与洞察

- **理论驱动的方法设计**：先严格证明 Top-K 有偏，再基于重要性采样理论提出无偏方案，逻辑链非常完整。这种"先理论后实践"的研究范式值得学习
- **极致的存储效率**：只需缓存约 0.01% 的 logit 信息就能达到接近全量蒸馏的效果，这使得"预计算+缓存"的 KD 范式在实际中真正可行
- **RSKD 可直接应用于 LLM 预训练蒸馏**：这是少数将稀疏 KD 推广到预训练阶段的工作，突破了之前 KD 主要用于微调的局限

## 局限与展望

- 实验最大只到 3B 规模的 student，是否在 7B/13B 级别仍然有效需要验证
- Teacher 模型的预计算阶段本身也有成本，论文没有详细分析端到端的总成本
- 只在英文预训练上验证，多语言场景下不同语言的 logit 分布差异可能影响采样效率
- 未来可探索与量化、剪枝等其他压缩技术的结合，进一步降低部署成本
- 采样策略是否可以自适应地在训练过程中调整（如课程学习式的采样）是一个有趣方向

## 相关工作与启发

- **vs MiniLLM**: MiniLLM 使用 KL 散度的反向形式进行蒸馏，需要在线运行 teacher。RSKD 允许完全离线蒸馏，更实用
- **vs TinyLLaMA**: TinyLLaMA 从头训练小模型，不使用蒸馏。RSKD 证明即使简单的蒸馏也能带来显著提升
- **vs DistilBERT**: 经典的 BERT 蒸馏方法，但不涉及稀疏 logit 问题。RSKD 解决了大词表场景下的核心效率瓶颈
- ACL 2025 Oral 论文，理论+实验结合的典范，值得作为 KD 方向的 baseline 参考

## 评分

- 新颖性: ⭐⭐⭐⭐ 重要性采样用于 KD logit 缓存是新角度，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐ 多规模、多 K 值验证，消融全面，但最大规模可更大
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，Oral 论文写作水平高
- 价值: ⭐⭐⭐⭐⭐ 直接解决 LLM KD 的核心实用问题，产业价值高

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](../../ICCV2025/model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](../../NeurIPS2025/model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[ACL 2025\] Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition](compact_and_compressible_representations_for_llms_using_structured_sparse_decom.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)

<!-- RELATED:END -->
