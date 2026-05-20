---
title: >-
  [论文解读] QKV Projections Require a Fraction of Their Memory
description: >-
  [ICLR 2026][模型压缩][训练内存压缩] 提出 PAMM（Point-Approximate Matrix Multiplication），一种激活压缩技术，通过随机选取少量代表性 token 来近似 QKV 投影层激活，实现高达 512× 压缩率且不影响模型性能。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "训练内存压缩"
  - "注意力机制"
  - "矩阵乘法近似"
  - "激活压缩"
  - "LLM训练"
---

# QKV Projections Require a Fraction of Their Memory

**会议**: ICLR 2026  
**arXiv**: [2506.02939](https://arxiv.org/abs/2506.02939)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 训练内存压缩, 注意力机制, 矩阵乘法近似, 激活压缩, LLM训练

## 一句话总结

提出 PAMM（Point-Approximate Matrix Multiplication），一种激活压缩技术，通过随机选取少量代表性 token 来近似 QKV 投影层激活，实现高达 512× 压缩率且不影响模型性能。

## 研究背景与动机

LLM 训练中，注意力层的 QKV 投影占用大量内存：输入 $X$ 需要在前向过程中保存以用于反向传播（计算 $\nabla W = X^\top \cdot \nabla Z$）。这部分内存可占注意力块总峰值 GPU 内存的 20%。

现有内存优化方法的不足：
- **高效注意力**（FlashAttention 等）：优化缩放点积本身，未涉及线性投影
- **低秩方法**（CompAct 等）：沿隐藏维度压缩，但序列维度的冗余更大
- **优化器状态压缩**：不随 batch size 和序列长度扩展

核心洞察：**序列维度存在巨大冗余**。训练 batch 中的 token 数量 $b = BL$（如 16384）远大于隐藏维度 $n$（如 2048），$\text{rank}(X) \leq n$，理论上仅需 $n$ 个基向量即可表示 $X$，压缩比可达 8×。

## 方法详解

### 整体框架

PAMM 分两阶段工作：(1) 前向时将 $X$ 压缩为少量生成点和辅助信息；(2) 反向时用压缩表示近似计算梯度 $\nabla W$。

### 关键设计

1. **激活压缩 (Compression Stage)**：

    - 从 $X \in \mathbb{R}^{b \times n}$ 中随机采样 $k = r \cdot b$ 行作为生成点 $C \in \mathbb{R}^{k \times n}$
    - 对每个点 $A_i$，选择最佳生成点：$f(i) = \arg\max_j |\text{csim}(A_i, C_j)|$（Lemma 1）
    - 计算缩放系数：$\tilde{A}_i = \alpha(i, f(i)) \cdot C_{f(i)}$，其中 $\alpha = \frac{\langle A_i, C_j \rangle}{\|C_j\|_2^2}$
    - 邻域条件：$\|A_i - \tilde{A}_i\|_2 \leq \varepsilon \|A_i\|_2$，不满足则丢弃

2. **近似矩阵乘法 (Approximate Multiplication)**：

    - 不重建完整 $\tilde{A}$，而是先聚合 $\tilde{B}_j = \sum_{i:f(i)=j} \alpha_i B_i$
    - 计算 $\tilde{O} = C^\top \tilde{B}$，维度从 $b \times n$ 降为 $k \times n$
    - 引入归一化因子 $\beta = \frac{b}{b-\eta}$ 保证无偏估计 $\mathbb{E}[\tilde{O}] = O$

3. **理论保证**：

    - **Lemma 2**（$k$ 的充分条件）：$k > \frac{b}{n_{\min}} \ln(\frac{b}{\delta})$，仅需对数级别的生成点
    - 近似误差上界：$\|O - \tilde{O}\|_F^2 \leq \|B\|_2^2 (\varepsilon^2 \|A_\mathcal{I}\|_F^2 + \|A_{\bar{\mathcal{I}}}\|_F^2)$
    - 实践中 $\varepsilon \to \infty$（不使用邻域约束）效果最好

### 损失函数 / 训练策略

- PAMM 仅修改 QKV 投影的反向传播，前向和其他层梯度不受影响
- 与 FlashAttention、梯度检查点、LoRA 完全兼容
- 实验中压缩比 $r$ 低至 $1/512$
- 微调场景中甚至可以用 $k=1$（仅一个生成点）

## 实验关键数据

### 预训练实验（LLaMA on C4）

| 模型 | PAMM r | 验证 PPL | QKV 内存 (MB) | 内存减少 |
|------|--------|---------|-------------|---------|
| LLaMA-60M | 无 PAMM | 31.8 | 432 | - |
| LLaMA-60M | 1/512 | 31.6 | 0.85 | **>99%** |
| LLaMA-350M | 无 PAMM | 18.7 | 1,296 | - |
| LLaMA-350M | 1/512 | 18.5 | 2.53 | **>99%** |
| LLaMA-1B | 无 PAMM | 15.1 | 2,592 | - |
| LLaMA-1B | 1/512 | 15.0 | 5.06 | **>99%** |

### 微调实验（RoBERTa-base on GLUE）

| 方法 | QKV 内存 (MB) | GLUE 平均 | 内存减少 |
|------|-------------|----------|---------|
| Full Fine-Tuning | 288 | 86.28 | - |
| PAMM r=1/128 | 6.75 | 86.11 | **97.7%** |
| PAMM r=1/256 | 3.37 | 86.18 | **98.8%** |

### 吞吐量分析（LLaMA-1B）

| 阶段 | 基线 (tok/s) | PAMM (tok/s) | 吞吐量降低 |
|------|-----------|------------|----------|
| 前向 | 247.6K | 235.4K | 4.92% |
| 反向 | 141.9K | 138.3K | 2.53% |
| **总计** | 88.4K | 85.2K | **3.61%** |

### 关键发现

- 512× 压缩下 PPL 不降反升（大模型更明显），说明冗余 token 可能影响训练
- 随模型增大，吞吐量损失从 19.7%（60M）降至 2.1%（7B），大模型更实用
- PAMM 在所有 batch size 和序列长度配置下均表现稳定
- 对比 CompAct（沿隐藏维度压缩）：PAMM 在高压缩比下性能显著更好

## 亮点与洞察

- 洞察深刻：序列维度冗余远大于隐藏维度冗余，这是高压缩比的根本原因
- 极其简单有效：随机选取生成点就足够，无需复杂聚类
- 理论严谨：Lemma 1/2 提供了算法设计的理论指导
- 与 FlashAttention 等完全正交，可直接叠加使用
- 惊喜发现：高压缩比下 PPL 反而略有改善，暗示正则化效应

## 局限与展望

- 仅应用于 QKV 投影，未探索 FFN 层的激活压缩
- 邻域条件参数 $\varepsilon$ 的最优设置为 $\infty$（即不使用），理论解释不充分
- 额外计算（余弦相似度矩阵 + argmax）对小模型影响较大
- 未在分布式训练（多节点）场景下验证

## 相关工作与启发

- 与 CompAct 的关键区别：PAMM 沿序列维度压缩（冗余更大），CompAct 沿隐藏维度
- 与梯度检查点的关系：互补——梯度检查点减少存储的层数，PAMM 减少每层存储量
- 启示：训练内存优化不应只关注优化器状态和注意力机制，激活内存同样重要

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 发现序列维度冗余的新方向，方法极简高效
- 实验充分度: ⭐⭐⭐⭐⭐ 预训练/微调/吞吐量/消融全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论和实验结合好，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 实际可用于 LLM 训练的内存优化工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](llms_encode_their_failures_predicting_success_from_pre-generation_activations.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](../../ACL2026/model_compression/memp_exploring_agent_procedural_memory.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](../../ICML2026/model_compression/stochastic_sparse_attention_for_memory-bound_inference.md)

</div>

<!-- RELATED:END -->
