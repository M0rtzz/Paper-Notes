---
title: >-
  [论文解读] PT2-LLM: Post-Training Ternarization for Large Language Models
description: >-
  [ICLR 2026][LLM/NLP][三值化] 提出 PT2-LLM，首个针对 LLM 的后训练三值化框架，通过非对称三值量化器（含迭代三值拟合和激活感知网格对齐）与结构相似性重排序策略，在 1.58-bit 下实现优于 2-bit PTQ 方法的性能。
tags:
  - ICLR 2026
  - LLM/NLP
  - 三值化
  - 后训练量化
  - 极低比特
  - LLM压缩
  - 列重排序
---

# PT2-LLM: Post-Training Ternarization for Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.03267](https://arxiv.org/abs/2510.03267)  
**代码**: [GitHub](https://github.com/XIANGLONGYAN/PT2-LLM)  
**领域**: 模型压缩  
**关键词**: 三值化, 后训练量化, 极低比特, LLM压缩, 列重排序

## 一句话总结

提出 PT2-LLM，首个针对 LLM 的后训练三值化框架，通过非对称三值量化器（含迭代三值拟合和激活感知网格对齐）与结构相似性重排序策略，在 1.58-bit 下实现优于 2-bit PTQ 方法的性能。

## 研究背景与动机

三值化（权重约束为 $\{-1, 0, +1\}$）是极致压缩方案：
- 相比低比特量化（2-4 bit），三值化消除了大部分浮点乘法，仅需加法运算
- 相比二值化，三值化更好匹配 LLM 权重的单峰分布，表达能力更强

现有三值化方法（BitNet b1.58、TernaryLLM）均依赖 QAT，对 LLM 不切实际。PTQ-based 三值化面临两大挑战：
1. 无法通过梯度优化三值参数——需要训练无关的参数优化方案
2. 权重分布分散且存在异常值——极低比特量化误差更大

## 方法详解

### 整体框架

PT2-LLM 包含两个核心组件：非对称三值量化器（ATQ）和结构相似性重排序（SSR），在 GPTQ 框架下逐块应用。

### 关键设计

1. **非对称三值量化器 (ATQ)**：

    - 引入行级偏移 $\mu$：$\hat{\mathbf{W}} = \alpha \mathbf{T} + \mu$，适配非零均值权重分布
    - **迭代三值拟合 (ITF)**：交替优化三值网格和三值矩阵
        - 最优网格（闭式解）：$\alpha^* = \frac{m \cdot (\mathbf{W} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1}) \circ (\mathbf{W}\mathbf{1})}{m \cdot (\mathbf{T} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})^2}$
        - 灵活取整：$\mathbf{T}_{ij}^* = \arg\min_{t \in \{-1,0,1\}} |Z_{ij} - t|$，其中 $Z_{ij} = (W_{ij} - \mu_i^*) / \alpha_i^*$
        - 约 10 次迭代收敛
    - **激活感知网格对齐 (AGA)**：用校准数据优化输出误差 $\mathcal{E}_x = \|\mathbf{WX} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$

2. **结构相似性重排序 (SSR)**：

    - 动机：朴素分块三值化中，同一块内权重方差大且存在异常值列
    - 计算列间余弦相似度：$S_{ij} = \frac{\mathbf{W}_{:,i}^\top \mathbf{W}_{:,j}}{\|\mathbf{W}_{:,i}\|_2 \|\mathbf{W}_{:,j}\|_2}$
    - 将结构相似的列聚在同一块内，使块内分布更紧凑
    - 轻量化策略：每步选取与均值参考最相似的 top-k 列组成下一个量化块

### 损失函数 / 训练策略

- ITF 阶段最小化权重量化误差 $\mathcal{E}_w = \|\mathbf{W} - \hat{\mathbf{W}}\|_F^2$
- AGA 阶段最小化输出误差 $\mathcal{E}_x = \|\mathbf{WX} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$
- AGA 仅更新 $(\alpha, \mu)$ 一次（冻结 $\mathbf{T}$），避免在校准集上过拟合
- 量化块大小为 128，与 GPTQ 框架集成

## 实验关键数据

### 主实验（LLaMA-7B 零样本问答）

| 方法 | #W (bit) | Wiki2 PPL ↓ | C4 PPL ↓ | 7任务平均 Acc ↑ |
|------|---------|------------|---------|--------------|
| FP16 | 16 | 5.68 | 7.34 | 61.73% |
| AWQ 2-bit | 2 | 2.60e5 | 2.86e5 | 32.50% |
| GPTQ 2-bit | 2 | 129.19 | 79.06 | 34.35% |
| Slim-LLM 2-bit | 2 | 14.58 | 30.71 | 39.74% |
| PB-LLM 1.7-bit | 1.7 | 82.76 | 76.63 | 33.44% |
| **PT2-LLM 1.58-bit** | 1.58 | **11.39** | **24.55** | **45.07%** |

### LLaMA-13B 结果

| 方法 | #W (bit) | Wiki2 PPL ↓ | 7任务平均 Acc ↑ |
|------|---------|------------|--------------|
| FP16 | 16 | 5.09 | 63.81% |
| GPTQ 2-bit | 2 | 20.46 | 41.00% |
| **PT2-LLM 1.58-bit** | 1.58 | **8.93** | **49.14%** |

### 关键发现

- PT2-LLM 在 1.58-bit 下超越所有 2-bit PTQ 方法，内存占用更低
- ITF 和 AGA 两阶段优化分别降低权重误差和输出误差
- SSR 有效降低块内方差，离群列的聚集使其不再成为异常值
- 推理加速：prefill 和 decode 阶段均实现端到端加速

## 亮点与洞察

- 首次在 PTQ 设置下实现 LLM 三值化，填补重要空白
- ITF 的交替优化策略优雅——每步都有闭式最优解，无需梯度优化
- AGA 的关键设计决策：冻结 $\mathbf{T}$ 仅更新网格参数，有效避免过拟合
- SSR 的直觉精辟："异常值之间不再是异常值"

## 局限与展望

- 1.58-bit 精度仍与 FP16 有较大差距（如 LLaMA-7B 平均精度 45% vs 62%）
- SSR 每步重新计算相似度有一定开销
- 未与 QAT-based 三值化方法（BitNet b1.58）直接对比
- 仅验证了 LLaMA 系列，未覆盖 Qwen、Mistral 等模型

## 相关工作与启发

- 与 GPTQ 的关系：PT2-LLM 在 GPTQ 框架内进行三值化，继承其逐块误差补偿
- 与 BitNet b1.58 的区别：PT2-LLM 是 PTQ 方案，无需从头训练
- 启示：极低比特 PTQ 仍有很大空间，非对称量化和结构感知重排序是有效方向

## 评分

- 新颖性: ⭐⭐⭐⭐ PTQ 三值化是未探索的设定
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，可视化直观
- 价值: ⭐⭐⭐⭐ 为极低比特 LLM 部署提供新选择

<!-- RELATED:START -->

## 相关论文

- [Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [Self-Training Elicits Concise Reasoning in Large Language Models](../../ACL2025/llm_nlp/self-training_elicits_concise_reasoning_in_large_language_models.md)
- [Cool-Fusion: Fuse Large Language Models without Training](../../ACL2025/llm_nlp/cool-fusion_fuse_large_language_models_without_training.md)
- [The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)
- [DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)

<!-- RELATED:END -->
