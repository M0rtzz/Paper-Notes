---
title: >-
  [论文解读] PT2-LLM: Post-Training Ternarization for Large Language Models
description: >-
  [ICLR 2026][LLM/NLP][三值化] 提出 PT2-LLM，首个针对 LLM 的后训练三值化框架，通过非对称三值量化器（含迭代三值拟合和激活感知网格对齐）与结构相似性重排序策略，在 1.58-bit 下实现优于 2-bit PTQ 方法的性能。
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "三值化"
  - "后训练量化"
  - "极低比特"
  - "LLM压缩"
  - "列重排序"
---

# PT2-LLM: Post-Training Ternarization for Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.03267](https://arxiv.org/abs/2510.03267)  
**代码**: [GitHub](https://github.com/XIANGLONGYAN/PT2-LLM)  
**领域**: LLM/NLP  
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

PT2-LLM 把三值化嵌进 GPTQ 的逐块误差补偿流程，靠两个组件接力把 1.58-bit 的精度损失压下来：非对称三值量化器（ATQ）负责把单块权重拟合成最优的三值表示，结构相似性重排序（SSR）则在量化前重排列序，让每个块内的权重分布尽量紧凑。两者都不依赖梯度反传，全程只用闭式解和校准数据微调，因此能在纯 PTQ 设定下完成。

### 关键设计

**1. 非对称三值量化器 ATQ：用偏移和闭式迭代拟合非零均值权重。** LLM 权重并非零均值对称分布，若直接套对称三值网格 $\hat{\mathbf{W}}=\alpha\mathbf{T}$ 会系统性偏移整行。ATQ 引入行级偏移 $\mu$，把量化形式改成 $\hat{\mathbf{W}}=\alpha\mathbf{T}+\mu$，让网格中心对齐每行权重的真实均值。核心难点是在没有梯度的情况下同时定出尺度 $\alpha$ 和三值矩阵 $\mathbf{T}$，ATQ 用**迭代三值拟合（ITF）**交替求解：固定 $\mathbf{T}$ 时尺度有闭式最优解 $\alpha^* = \frac{m\cdot(\mathbf{W}\circ\mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})\circ(\mathbf{W}\mathbf{1})}{m\cdot(\mathbf{T}\circ\mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})^2}$；固定网格时三值矩阵按 $\mathbf{T}_{ij}^*=\arg\min_{t\in\{-1,0,1\}}|Z_{ij}-t|$ 灵活取整，其中归一化坐标 $Z_{ij}=(W_{ij}-\mu_i^*)/\alpha_i^*$。每步都有解析最优，约 10 次迭代即收敛。ITF 只盯权重本身的量化误差，但真正影响下游的是输出，因此再叠一层**激活感知网格对齐（AGA）**：用校准数据把目标从权重误差换成输出误差 $\mathcal{E}_x=\|\mathbf{WX}-\hat{\mathbf{W}}\mathbf{X}\|_F^2$，但只更新 $(\alpha,\mu)$ 一次、冻结 $\mathbf{T}$，既校正了激活敏感方向上的网格，又避免在小校准集上过拟合。

**2. 结构相似性重排序 SSR：把相似列聚到同一块以驯服异常值。** 朴素分块三值化按原始列序切块，同一块里常混进方差悬殊的列和孤立的异常值列，单块只有一组 $(\alpha,\mu)$ 难以同时照顾，量化误差被放大。SSR 在量化前先按列间余弦相似度 $S_{ij}=\frac{\mathbf{W}_{:,i}^\top\mathbf{W}_{:,j}}{\|\mathbf{W}_{:,i}\|_2\|\mathbf{W}_{:,j}\|_2}$ 衡量结构接近程度，把彼此相似的列聚拢到同一个量化块，使块内分布更紧凑、共享网格更贴合。为避免全量相似度矩阵的开销，它采用轻量贪心策略：每步以当前块均值为参考，选出与之最相似的 top-k 列组成下一个块。其效果可概括为"异常值之间不再是异常值"——原本分散的离群列被集中到一起后，它们相对块内参考反而不再突兀。

### 损失函数 / 训练策略

ITF 阶段最小化权重量化误差 $\mathcal{E}_w=\|\mathbf{W}-\hat{\mathbf{W}}\|_F^2$，AGA 阶段切换为输出误差 $\mathcal{E}_x=\|\mathbf{WX}-\hat{\mathbf{W}}\mathbf{X}\|_F^2$ 且仅对 $(\alpha,\mu)$ 更新一次（$\mathbf{T}$ 冻结）。整体量化块大小取 128，与 GPTQ 框架逐块集成、复用其误差补偿。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ACL 2025\] Self-Training Elicits Concise Reasoning in Large Language Models](../../ACL2025/llm_nlp/self-training_elicits_concise_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Cool-Fusion: Fuse Large Language Models without Training](../../ACL2025/llm_nlp/cool-fusion_fuse_large_language_models_without_training.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)

</div>

<!-- RELATED:END -->
