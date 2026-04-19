---
title: >-
  [论文解读] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models
description: >-
  [NeurIPS 2025][多模态][N:M稀疏] 提出 PermLLM，首个可学习通道排列（LCP）框架，通过Sinkhorn归一化将离散排列矩阵松弛为可微分的软排列矩阵实现端到端优化，结合块级排列策略大幅降低计算开销，有效提升N:M稀疏LLM的性能。
tags:
  - NeurIPS 2025
  - 多模态
  - "N:M稀疏"
  - 通道排列
  - 模型剪枝
  - LLM压缩
  - Sinkhorn归一化
---

# PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.10136](https://arxiv.org/abs/2510.10136)  
**代码**: [GitHub](https://github.com/lanchengzou/PermLLM)  
**领域**: 多模态VLM  
**关键词**: N:M稀疏, 通道排列, 模型剪枝, LLM压缩, Sinkhorn归一化

## 一句话总结

提出 PermLLM，首个可学习通道排列（LCP）框架，通过Sinkhorn归一化将离散排列矩阵松弛为可微分的软排列矩阵实现端到端优化，结合块级排列策略大幅降低计算开销，有效提升N:M稀疏LLM的性能。

## 研究背景与动机

LLM的规模急剧增长，高效部署面临严峻挑战。半结构化剪枝（N:M稀疏）是一种极具潜力的压缩方案——在每M个连续权重中保留M-N个并将N个置零，可直接利用NVIDIA GPU的Sparse Tensor Core实现约2×加速，是当前最实用的稀疏范式。

通道排列（Channel Permutation）是提升N:M稀疏模型精度的关键技术——通过重排权重矩阵的输入通道顺序，使更重要的权重被保留。但现有排列方法存在根本性缺陷：

**依赖手工设计的质量度量**。如RIA使用保留权重重要性之和作为排列质量的代理指标，但这个指标与实际剪枝误差之间存在显著差距。论文通过一个直观的例子展示了这个问题——**最大化重要性分数的排列不一定能最小化输出误差，甚至可能增加误差**。这是因为手工度量忽略了权重之间的复杂交互关系和跨层依赖。

另一个挑战是排列的组合爆炸：$C_{in}$个通道有$C_{in}!$种排列方式，即使在N:M约束下简化后，$C_{in}=16, M=4$时仍有约260万个候选方案。对于LLM中数千维的隐藏层，穷举搜索完全不可行。

## 方法详解

### 整体框架

PermLLM是一个后训练剪枝框架，与现有one-shot剪枝方法（Wanda、RIA）无缝集成。核心流程：(1) 用Sinkhorn归一化将可学习矩阵转化为软排列矩阵；(2) 通过Hungarian算法硬化为严格排列矩阵；(3) 基于排列后的权重用one-shot方法确定剪枝mask；(4) 用余弦相似度损失端到端优化排列矩阵。

### 关键设计

1. **软排列矩阵松弛**：排列矩阵 $\mathbf{P}$ 是离散的二值矩阵，不可微分。PermLLM引入可学习矩阵 $\mathbf{W}_P$，通过Sinkhorn归一化将其转化为双随机矩阵（每行每列之和均为1）作为软排列矩阵 $\hat{\mathbf{P}}$：

   $$\hat{\mathbf{P}} = S^L(\mathbf{W}_P / \tau)$$

   其中 $S^L$ 是L次迭代的Sinkhorn归一化（交替行/列归一化），温度 $\tau$ 从1线性衰减到0.1以控制矩阵的"硬度"。前向传播时用Hungarian算法将 $\hat{\mathbf{P}}$ 硬化为严格排列矩阵 $\mathbf{P}$（求解线性分配问题），反向传播时用STE近似梯度 $\partial\mathbf{P}/\partial\hat{\mathbf{P}}=1$。

2. **块级通道排列**：全矩阵排列的参数量为 $C_{in}^2$，对LLM不现实。PermLLM将通道分为 $N_B$ 个大小为 $B$ 的块，每个块独立学习排列，参数量降为 $C_{in} \times B$（$B/C_{in}$倍压缩）。硬化的时间复杂度也从 $O(C_{in}^3)$ 降到 $O(C_{in} \cdot B^2)$。块排列矩阵为块对角矩阵 $\mathbf{P}_B = \text{diag}(\mathbf{P}_1, \mathbf{P}_2, \ldots, \mathbf{P}_{N_B})$。

3. **剪枝感知排列优化**：排列改变了权重顺序，因此剪枝mask也随之变化。mask的确定使用softmax做可微近似（反向），argmax做离散选择（前向）：

   $$\hat{\mathbf{M}}_{i,kM:(k+1)M} = \text{Softmax}(\hat{\mathbf{S}}_{i,kM:(k+1)M})$$

   优化目标是直接最小化密集模型与稀疏模型输出的差异：

   $$\mathcal{L}_{cosine}(\mathbf{y}, \widetilde{\mathbf{y}}) = 1 - \frac{\mathbf{y} \cdot \widetilde{\mathbf{y}}}{\|\mathbf{y}\| \cdot \|\widetilde{\mathbf{y}}\|}$$

   训练完成后，权重经排列和剪枝：$\hat{\mathbf{W}}' = \mathbf{M}^* \odot (\mathbf{W}\mathbf{P}_B^*)$，前一层的输出通道也需相应重排以保持计算一致性。

### 损失函数 / 训练策略

- 优化器：AdamW，学习率 {1e-3, 5e-3}
- Sinkhorn归一化迭代次数：5次
- 温度从1→0.1线性衰减
- 默认块大小：64
- 校准数据：C4数据集128个样本×1024 token
- 训练时间：7B模型约2.5小时（4×A100），13B模型约5.5小时（8×A100）
- 自定义CUDA核为通道排列操作实现84×加速

## 实验关键数据

### 主实验

**WikiText2困惑度（2:4稀疏，↓越低越好）**

| 方法 | OPT-6.7B | LLaMA-7B | LLaMA-2 7B | LLaMA-3.1 8B | Qwen-2.5 7B |
|------|----------|----------|------------|-------------|-------------|
| Dense | 10.86 | 5.68 | 5.47 | 6.24 | 7.74 |
| SparseGPT | 14.33 | 11.19 | 11.12 | 16.62 | 14.34 |
| Wanda | 16.29 | 11.59 | 12.16 | 23.42 | 24.44 |
| Wanda+CP | 15.28 | 11.07 | 11.00 | 21.09 | 18.76 |
| **PermLLM_Wanda** | **14.27** | **9.41** | **9.39** | **14.03** | **13.58** |
| RIA | 15.93 | 11.14 | 11.30 | 22.62 | 22.67 |
| RIA+CP | 15.13 | 10.99 | 10.26 | 19.80 | 17.58 |
| **PermLLM_RIA** | **14.23** | **9.95** | **9.60** | **15.79** | **15.93** |

**零样本任务平均精度（2:4稀疏）**

| 模型 | 方法 | HellaSwag | ARC_E | ARC_C | OBQA | RTE | 平均 |
|------|------|-----------|-------|-------|------|-----|------|
| LLaMA-2 7B | Dense | 57.13 | 76.30 | 43.26 | 31.60 | 62.45 | 54.15 |
| | Wanda | 41.59 | 61.74 | 30.20 | 24.00 | 53.07 | 42.12 |
| | Wanda+CP | 43.40 | 64.69 | 30.03 | 26.00 | 53.07 | 43.44 |
| | **PermLLM_Wanda** | **46.60** | **65.49** | **31.14** | **26.20** | **63.54** | **46.59** |
| Qwen-2.5 7B | Dense | 58.79 | 79.56 | 46.08 | 33.00 | 76.90 | 58.87 |
| | Wanda | 40.60 | 67.17 | 33.45 | 25.40 | 72.92 | 47.91 |
| | **PermLLM_Wanda** | **47.30** | **70.58** | **38.13** | **27.60** | **77.26** | **52.17** |

### 消融实验

| 配置 | WikiText2 PPL | 平均零样本精度 | 说明 |
|------|--------------|---------------|------|
| Sinkhorn迭代=0 (Qwen-2.5 7B) | 14.12 | 42.96 | 软排列偏离双随机矩阵 |
| Sinkhorn迭代=5 (Qwen-2.5 7B) | 14.03 | 43.33 | 双随机矩阵约束帮助学习 |
| Sinkhorn迭代=0 (LLaMA-3.1 8B) | 14.43 | 49.18 | - |
| Sinkhorn迭代=5 (LLaMA-3.1 8B) | 13.58 | 52.17 | 提升显著 |

**推理加速（LLaMA-2 7B，2048 token）**

| 组件 | 密集模型 | 2:4稀疏+CP | 加速比 |
|------|---------|-----------|--------|
| Q/K/V/O_proj | 1.513ms | 0.927ms | 1.63× |
| Up/Gate_proj | 2.607ms | 1.526ms | 1.71× |
| Down_proj | 2.614ms | 1.535ms | 1.70× |
| 通道排列开销 | - | 0.039ms | 可忽略 |

### 关键发现

- PermLLM在所有模型上都显著优于手工排列方法（Wanda+CP、RIA+CP）
- 对新模型（LLaMA-3.1、Qwen-2.5）提升尤为显著——Wanda+CP在LLaMA-3.1上PPL为21.09，PermLLM降到14.03
- 自定义CUDA核使排列开销仅0.039ms，整体仍能实现约1.67×加速
- 块大小64在精度和效率间取得良好平衡

## 亮点与洞察

- **核心贡献**是将离散排列优化问题转化为连续可微分优化问题——Sinkhorn归一化+Hungarian算法+STE的组合非常精巧
- 直接揭示了手工排列度量的根本缺陷——最大化重要性分数≠最小化输出误差
- 块级排列是巧妙的工程设计，将参数量和计算复杂度从 $O(C_{in}^2)$ 和 $O(C_{in}^3)$ 分别降到 $O(C_{in} \cdot B)$ 和 $O(C_{in} \cdot B^2)$
- 框架设计为插件式，可与任意one-shot剪枝方法结合

## 局限与展望

- 后训练过程仍需几小时GPU时间，对于资源极度受限的场景可能不够轻量
- 块级排列限制了跨块的通道重排，可能遗漏全局最优解
- 仅评估了2:4和4:8稀疏，更灵活的N:M配置（如1:4、3:8等）未探索
- 温度退火策略（线性衰减）是否最优未经验证，可能有更好的调度策略
- 与结合权重更新的方法（如SparseGPT）组合使用的效果值得探索

## 相关工作与启发

- RIA 提出了two-stage排列策略但依赖手工度量，是PermLLM直接改进的对象
- Wanda 和 SparseGPT 是主流one-shot剪枝基线
- SR-STE 将STE用于N:M稀疏mask学习，PermLLM将类似思路用于排列学习
- 启发：Sinkhorn归一化在其他组合优化问题（如最优传输、图匹配）中也广泛使用，本文展示了其在模型压缩中的新应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个可学习通道排列框架，Sinkhorn+Hungarian+STE组合解决离散优化问题非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 7个模型×多个基准×详细消融×推理速度分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐⭐ 通道排列方法具有广泛的实用价值，代码开源，84×CUDA加速很实用

<!-- RELATED:START -->

## 相关论文

- [Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)
- [MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](../../ICML2025/multimodal_vlm/mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe.md)
- [CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)

<!-- RELATED:END -->
