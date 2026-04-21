---
title: >-
  [论文解读] Parallel Token Prediction for Language Models
description: >-
  [ICLR 2026][模型压缩][并行解码] 提出 Parallel Token Prediction (PTP)，通过将采样随机性从后处理移至模型输入（辅助变量），使未来 token 成为确定性函数，从而在单次前向传播中联合预测多个 token。
tags:
  - ICLR 2026
  - 模型压缩
  - 并行解码
  - 推测解码
  - 辅助变量
  - 自回归模型
  - 推理加速
---

# Parallel Token Prediction for Language Models

**会议**: ICLR 2026  
**arXiv**: [2512.21323](https://arxiv.org/abs/2512.21323)  
**代码**: [GitHub](https://github.com/mandt-lab/ptp)  
**领域**: 模型压缩  
**关键词**: 并行解码, 推测解码, 辅助变量, 自回归模型, 推理加速

## 一句话总结

提出 Parallel Token Prediction (PTP)，通过将采样随机性从后处理移至模型输入（辅助变量），使未来 token 成为确定性函数，从而在单次前向传播中联合预测多个 token。

## 研究背景与动机

自回归 Transformer 的顺序生成过程是推理延迟的主要瓶颈——每预测一个 token 需要一次前向传播。现有加速方法的局限：
- **推测解码**：使用小模型草拟再验证，但小模型本身仍是顺序生成
- **独立多 token 预测**：假设 token 条件独立，导致语义不一致（如生成 "def numpy"）
- **离散扩散**：需要多步迭代，仍有不可约的顺序成分

PTP 的核心洞察：如果把采样用的随机变量 $u_i \sim \mathcal{U}[0,1]$ 作为模型输入，那么每个 token $t_i$ 就成了 $u_i$ 和上文的确定性函数，模型可以并行预测所有未来 token。

## 方法详解

### 整体框架

PTP 有两种变体：O-PTP（预测 one-hot 分布）和 C-PTP（恢复完整条件分布），均支持单次前向传播预测多个 token。训练方式包括蒸馏和从头训练。

### 关键设计

1. **辅助变量采样机制**：

    - 标准采样：$t_i = \text{Pick}(u_i, P_i)$，其中 $u_i \sim \mathcal{U}[0,1]$ 通过逆 CDF 确定 token
    - 关键观察：给定 $u_i$，token $t_i$ 是确定性的，$u_i$ 携带与 $t_i$ 等价的信息
    - **Theorem 1**：$t_k = f_P(t_{<i}; u_i, \ldots, u_k)$，未来 token 可作为辅助变量的确定性函数

2. **O-PTP (One-Hot PTP)**：

    - 模型同时接收所有辅助变量 $u_i, \ldots, u_N$，预测 one-hot 分布
    - $t_k = \arg\max(P(t_k | t_{<i}; u_i, \ldots, u_k))$
    - 优点：高效并行预测；缺点：不暴露底层采样分布

3. **C-PTP (Categorical PTP)**：

    - **Theorem 2**：$P(t_k | t_{<i}, u_i, \ldots, u_{k-1}) = P(t_k | t_{<k})$
    - 通过隐藏 $u_k$ 来恢复完整条件概率分布
    - 可从头训练（逆自回归训练）或蒸馏

4. **Partial Quadratic Decoding**：

    - 草案和验证并行执行，为所有可能的接受数量准备分支
    - 利用模型置信度估计分支概率：$P(\#\text{correct}=m|t) \approx (1-c_{i+m})\prod_{k=i}^{i+m-1} c_k$
    - 贪心分配续写 token 到高概率分支，减少计算浪费

### 损失函数 / 训练策略

- **蒸馏训练**：反推教师模型下每个 token 的辅助变量 $u_k \in [F_{k,t_k-1}, F_{k,t_k})$
- O-PTP 损失：$\mathcal{L}(\theta; t, i) = -\sum_{k=i}^N \log P_\theta(t_k | t_{<i}, u_i, \ldots, u_k)$
- C-PTP 损失：$\mathcal{L}(\theta; t, i) = -\sum_{k=i}^N \log P_\theta(t_k | t_{<i}, u_i, \ldots, u_{k-1})$
- 辅助变量编码：$\text{embed}(u) = W \cdot \text{binary}(u) + b$，将 float32 映射为 32 位二进制向量

## 实验关键数据

### 主实验（SpecBench - Vicuna-7B 蒸馏）

| 方法 | MTC | TL | SUM | QA | Math | RAG | 平均 #accepted |
|------|-----|-----|-----|-----|------|-----|---------------|
| O-PTP | 2.77 | - | - | - | - | - | **4.2** |
| 自回归基线 | - | - | - | - | - | - | ~2.0 |
| 独立预测 | - | - | - | - | - | - | ~3.5 |

| 指标 | 本文 (O-PTP) | 说明 |
|------|------------|------|
| 墙钟加速比 | **2.4×** | 相比标准自回归解码 |
| 每步接受 token 数 | **4.2** | 投机解码步 |

### 消融实验

| 配置 | #accepted ↑ | 说明 |
|------|-----------|------|
| O-PTP (有辅助变量) | **7.0 ± 0.1** | token 间有协调 |
| 独立预测 (无辅助变量) | 6.2 ± 0.1 | token 间独立，不一致对 |
| C-PTP 从头训练 | PPL 19.88 | 接近自回归基线 (19.81) |

### 关键发现

- PTP 草稿模型每次调用预测多个 token，将最优模型大小推向更大模型（甚至直接微调教师）
- 辅助变量使 token 间产生协调，显著减少不兼容 token 对（"def numpy" 等降至 <1%）
- C-PTP 从头训练可与自回归模型达到相当的困惑度，验证了理论表达能力

## 亮点与洞察

- 理论贡献突出：Theorem 1/2 从概率论角度严格证明了并行采样的可行性
- 将 Normalizing Flow 中的逆自回归思想迁移到离散序列生成，跨领域创新
- 辅助变量机制自然解决了独立预测的不一致问题
- Partial Quadratic Decoding 利用置信度分配计算资源，实用性强

## 局限与展望

- 实际加速受限于模型容量——有限的 Transformer 容量限制了单次可准确预测的 token 数
- 需要教师模型来反推辅助变量，蒸馏成本较高
- 辅助变量的二进制编码可能不是最优的表示方式
- 未验证在更大规模模型（70B+）和更长上下文上的效果

## 相关工作与启发

- 与 Medusa/EAGLE 的区别：PTP 通过辅助变量实现 token 间的协调，而非独立多头预测
- 与 Normalizing Flow 的联系：PTP 本质上是 Inverse Autoregressive Flow 的离散版本
- 可与 GaLore、FlashAttention 等高效训练技术组合使用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 辅助变量并行采样框架是全新理论贡献
- 实验充分度: ⭐⭐⭐⭐ 多任务验证，但缺少大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 定理证明严谨，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 开辟了并行 token 生成的新设计空间

<!-- RELATED:START -->

## 相关论文

- [Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)
- [BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](../../NeurIPS2025/model_compression/vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)

<!-- RELATED:END -->
