---
title: >-
  [论文解读] Task Singular Vectors: Reducing Task Interference in Model Merging
description: >-
  [CVPR 2025][模型压缩][模型合并] 提出 Task Singular Vectors (TSV) 框架，在逐层任务矩阵的 SVD 空间中分析和解决模型合并中的任务干扰问题：TSV-Compress 将任务向量压缩至 10% 保留 99% 精度，TSV-Merge 通过白化变换去相关化不同任务的奇异向量，在 8/14/20 任务合并上平均超过现有方法约 15 个百分点。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "模型合并"
  - "任务向量"
  - "奇异值分解"
  - "任务干扰"
  - "多任务学习"
---

# Task Singular Vectors: Reducing Task Interference in Model Merging

**会议**: CVPR 2025  
**arXiv**: [2412.00081](https://arxiv.org/abs/2412.00081)  
**代码**: [GitHub](https://github.com/AntoAndGar/task_singular_vectors)  
**领域**: 模型压缩/模型融合  
**关键词**: 模型合并, 任务向量, 奇异值分解, 任务干扰, 多任务学习

## 一句话总结

提出 Task Singular Vectors (TSV) 框架，在逐层任务矩阵的 SVD 空间中分析和解决模型合并中的任务干扰问题：TSV-Compress 将任务向量压缩至 10% 保留 99% 精度，TSV-Merge 通过白化变换去相关化不同任务的奇异向量，在 8/14/20 任务合并上平均超过现有方法约 15 个百分点。

## 研究背景与动机

- **模型合并的价值**：大量预训练模型公开可用，将多个任务特异微调模型合并为单一多任务模型（无需额外训练）具有重要实用价值。
- **Task Arithmetic 的局限**：TA 将网络视为扁平的高维向量进行加和，忽略了权重的矩阵结构信息，只能用余弦相似度等粗粒度度量评估任务间关系。
- **任务干扰的根源**：当不同任务的权重变化在相似方向上产生冲突时，简单平均会导致任务间干扰，合并后各任务性能下降。
- **低秩特性的直觉**：受 PEFT（如 LoRA）研究启发，微调产生的权重变化矩阵本身是低秩的——仅少量奇异向量即可忠实表示层的功能变化。
- **结构化分析的缺失**：已有方法（TIES、DARE、Consensus TA）虽尝试减少干扰，但仍在参数级别操作，未充分利用权重的矩阵结构。
- **本文思路**：在逐层保留矩阵结构的前提下，通过 SVD 分解提取任务奇异向量 (TSV)，在奇异向量空间中分析和减少任务干扰。

## 方法详解

### 整体框架

TSV 框架在逐层矩阵级别操作。对每个层 $l$、每个任务 $i$，计算任务矩阵 $\Delta_i^{(l)} = \theta_{\text{ft}_i}^{(l)} - \theta_{\text{pre}}^{(l)}$ 并做 SVD 分解 $\Delta_i = U_i \Sigma_i V_i^\top$。框架包含两个互补模块：TSV-C（压缩）利用低秩特性保留 top-$k$ 奇异分量；TSV-M（合并）在压缩基础上通过 Procrustes 正交化去除跨任务奇异向量的干扰。

### 关键设计

**设计一：TSV-Compress (TSV-C) — 任务向量低秩压缩**
- **功能**：将每个任务向量压缩至原始大小的 10%，保留 99% 精度
- **核心思路**：利用 Eckart-Young 定理，对每个层任务矩阵只保留 top-$k$ 奇异分量 $\hat{\Delta}_i = \sum_{j=1}^{k} \sigma_j^i u_j^i v_j^{i\top}$。当任务身份已知时（如通过路由器），设 $k = \frac{\text{rank}}{T}$ 即可将存储缩小 $T$ 倍。实验显示即使只保留 3% 的奇异分量，平均精度也仅下降 1.5%。
- **设计动机**：任务矩阵天然低秩，大量奇异分量携带的信息量极小；丢弃这些分量不仅节省存储，还能去除不同任务间的噪声干扰。

**设计二：Singular Task Interference (STI) — 任务干扰度量**
- **功能**：基于奇异向量的几何关系量化逐层任务干扰
- **核心思路**：定义 $\text{STI}(\{\Delta_i\}) = \|(U^\top U - I)\Sigma(V^\top V - I)\|_1$，其中 $U, V$ 为所有任务 TSV 的拼接。当不同任务的奇异向量正交时，$U^\top U$ 和 $V^\top V$ 接近单位阵，STI 趋近 0；当奇异向量高度重叠时，STI 值大，表明任务间存在强干扰。
- **设计动机**：相比余弦相似度等全局度量，STI 在逐层、逐奇异方向级别上刻画干扰，提供了远更精细的分析粒度。

**设计三：TSV-Merge (TSV-M) — 白化去干扰合并**
- **功能**：不需要验证数据/额外训练的模型合并
- **核心思路**：对压缩后拼接的 $\hat{U}$ 和 $\hat{V}$ 做正交 Procrustes 变换：$\hat{U}_\bot = P_U Q_U^\top$（其中 $\hat{U} = P_U D_U Q_U^\top$ 为 $\hat{U}$ 的 SVD），等价于白化变换 $X \mapsto X(X^\top X)^{-1/2}$。变换后不同任务的奇异向量被去相关化，然后重建合并矩阵 $\hat{M} = U_\bot \Sigma V_\bot^\top$，最终 $\theta_{\text{MT}} = \theta_{\text{pre}} + \alpha \hat{M}$。
- **设计动机**：白化/Procrustes 等价性保证了数值稳定性，且操作本身有闭式解，无需迭代优化；去相关化直接降低了 STI 度量。

### 损失函数

TSV-M 不涉及训练和损失函数——它是纯后处理的无训练模型合并方法，仅需各任务的微调模型和预训练模型权重。

## 实验关键数据

### 主实验：ViT-L-14 多任务合并平均精度

| 方法 | 8 tasks | 14 tasks | 20 tasks |
|------|---------|----------|----------|
| Zero-shot | 64.70 | 68.20 | 65.23 |
| Weight Averaging | 79.56 | 76.73 | 71.60 |
| Task Arithmetic | 84.93 | 79.41 | 74.01 |
| Consensus TA | 86.34 | 82.22 | 79.00 |
| **TSV-M (Ours)** | **~90+** | **~87+** | **~83+** |

### ViT-B-32 详细结果

| 方法 | 8 tasks | 14 tasks | 20 tasks |
|------|---------|----------|----------|
| Task Arithmetic | 70.79 | 65.32 | 60.52 |
| TIES | ~72 | ~66 | ~62 |
| Consensus TA | 75.03 | 70.39 | 65.43 |
| **TSV-M** | **85.86** | **80.06** | **~76** |

### 关键发现

1. TSV-M 相比 Consensus TA（此前 SOTA）在 ViT-B-32/8 tasks 上提升约 10.8 个百分点（75.03→85.86）
2. TSV-C 仅保留 10% 参数即保留 99% 精度，甚至 3% 参数仅下降 1.5%
3. 压缩和去干扰是互补的——两者单独使用均有提升，组合效果更优
4. TSV-M 不需要验证数据、额外训练或标签，需求最少（见 Table 1）
5. 随着任务数增加（8→14→20），TSV-M 的优势更加明显——在高任务数下干扰更严重，去干扰的价值更大

## 亮点与洞察

- **数学优雅**：白化与 Procrustes 的等价性证明简洁，为方法提供了坚实的理论基础
- **零额外开销**：TSV-M 不需要验证数据、标签、额外训练或路由器，是需求最少的模型合并方法
- **STI 度量**：提供了远比余弦相似度精细的任务干扰分析工具，可独立用于模型合并质量预测
- 低秩发现与 LoRA 等 PEFT 方法的内在一致性，为"微调产生低秩变化"提供了更多经验证据

## 局限与展望

- SVD 分解的计算成本随层数和参数规模增长，在超大模型（如 LLM）上的可行性待验证
- 当前仅在 ViT+CLIP 视觉分类任务上验证，对 NLP 等其他模态/任务的泛化性未知
- 白化变换可能过于激进地消除信息，在某些任务共享信息有价值的场景下可能不是最优的
- 缩放因子 $\alpha$ 仍需调优

## 相关工作与启发

- TSV 的低秩分析为 LoRA 等方法提供了新的理论视角——微调的本质可能就是在低秩子空间中寻找任务特异方向
- STI 度量可作为预筛选工具，帮助判断哪些任务适合合并、哪些不适合
- 白化去干扰的思路可推广到联邦学习中的模型聚合

## 评分

⭐⭐⭐⭐ — 理论扎实、实验充分，在模型合并领域取得了显著突破。STI 度量和 Procrustes 白化的设计都很有启发性，且方法简单优雅、无需额外训练。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[ACL 2025\] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](../../ACL2025/model_compression/osrm_lora_merging_orthogonal.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/model_compression/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)

</div>

<!-- RELATED:END -->
