---
title: >-
  [论文解读] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces
description: >-
  [ICCV 2025][LLM/NLP][零阶优化] 提出 SubZero（random Subspace Zeroth-order），通过逐层低秩扰动在随机子空间中估计梯度，显著降低零阶优化的梯度方差和角度误差，以接近推理的内存开销实现 LLM 的高效微调。
tags:
  - ICCV 2025
  - LLM/NLP
  - 零阶优化
  - LLM微调
  - 随机子空间
  - 低秩扰动
  - 内存高效
---

# Zeroth-Order Fine-Tuning of LLMs in Random Subspaces

**会议**: ICCV 2025  
**arXiv**: [2410.08989](https://arxiv.org/abs/2410.08989)  
**代码**: [https://github.com/zimingyy/SubZero](https://github.com/zimingyy/SubZero)  
**领域**: 大语言模型微调  
**关键词**: 零阶优化, LLM微调, 随机子空间, 低秩扰动, 内存高效

## 一句话总结

提出 SubZero（random Subspace Zeroth-order），通过逐层低秩扰动在随机子空间中估计梯度，显著降低零阶优化的梯度方差和角度误差，以接近推理的内存开销实现 LLM 的高效微调。

## 研究背景与动机

大语言模型（LLMs）的微调通常依赖一阶优化器（SGD/Adam），但随着模型规模增长，反向传播所需的内存开销变得难以承受。MeZO 首次将零阶（ZO）优化引入 LLM 微调，仅需前向传播即可估计梯度，内存开销与推理相当。

**核心矛盾**：ZO 梯度估计的方差与扰动维度（即模型参数量）成线性关系——对于拥有数十亿参数的 LLM，这会导致极大的方差，严重拖慢收敛速度和最终性能。

现有缓解方案存在不足：
- **增大 batch size**：后期开销急剧增加
- **稀疏扰动**（如 S-MeZO 的剪枝掩码）：启发式选择，缺乏理论支撑
- **随机子空间方法**（如 S-RGF）：需要存储 $d \times q$ 的巨大投影矩阵（$q$ 倍于模型大小），对 LLM 完全不可行

**切入角度**：利用 LLM 的层式矩阵结构，在每层独立构建低秩子空间（$\mathbf{U}_i \in \mathbb{R}^{m_i \times r}$, $\mathbf{V}_i \in \mathbb{R}^{n_i \times r}$），只在 $r \times r$ 的极小空间中采样扰动。投影矩阵是列正交的且惰性更新，既避免了巨大投影矩阵的存储，又保证了低方差。

## 方法详解

### 整体框架

SubZero 保持 MeZO 的"两次前向传播估计梯度"范式，但扰动方式从全参数空间的高斯随机向量改为逐层低秩矩阵 $\tilde{\mathbf{Z}}_i = \mathbf{U}_i \mathbf{Z}_i \mathbf{V}_i^\top$，其中 $\mathbf{Z}_i \in \mathbb{R}^{r \times r}$ 是低维高斯随机矩阵，$r \ll \min(m_i, n_i)$。

### 关键设计

1. **逐层低秩扰动**：对第 $i$ 层参数矩阵 $\mathbf{W}_i \in \mathbb{R}^{m_i \times n_i}$，通过 QR 分解两个高斯随机矩阵获得列正交投影矩阵 $\mathbf{U}_i$ 和 $\mathbf{V}_i$。扰动的损失差为：
   $$\rho = \frac{\mathcal{L}(\mathcal{W} + \varepsilon\tilde{\mathcal{Z}}; \mathcal{B}) - \mathcal{L}(\mathcal{W} - \varepsilon\tilde{\mathcal{Z}}; \mathcal{B})}{2\varepsilon}$$
   第 $i$ 层的梯度估计为 $\hat{\nabla}\mathcal{L}(\mathbf{W}_i) = \rho \mathbf{U}_i \mathbf{Z}_i \mathbf{V}_i^\top$。

   **设计动机**：相比模型级投影（S-RGF 的 $\mathbf{P} \in \mathbb{R}^{d \times q}$），层级投影矩阵是块对角的，等价形式为 $\mathbf{P} = \text{bdiag}(\mathbf{V}_1 \otimes \mathbf{U}_1, \cdots, \mathbf{V}_l \otimes \mathbf{U}_l)$，满足 $\mathbf{P}^\top \mathbf{P} = \mathbf{I}_q$，且不需要存储完整的 $d \times q$ 矩阵。实验表明列正交矩阵显著优于高斯随机投影矩阵（Table 5，RTE 上 74.0% vs 67.5%）。

2. **惰性子空间更新（Lazy Update）**：投影矩阵 $\mathbf{U}_i, \mathbf{V}_i$ 每 $F$ 步重新生成一次（默认 $F=1000$），中间步骤复用。过于频繁的更新增加 QR 分解开销且限制子空间探索；过于稀疏则子空间过时。消融实验（Table 7）显示 $F=1000$ 是较优选择。

3. **非方阵 Reshape 策略**：LoRA 的低秩矩阵 $\mathbf{A}_i \in \mathbb{R}^{m_i \times k}$（$k \ll m_i$）过于瘦长，无法找到更小的 $r \ll k$ 来构造低秩扰动。解决方案：将 $\mathbf{A}_i$ reshape 为近似方阵 $\mathbf{A}'_i \in \mathbb{R}^{m'_i \times k'}$（保持元素总量不变），再在方阵上应用低秩扰动。消融（Table 8）证实该策略对 PEFT 方案至关重要：prompt tuning 从 74.2% 提升到 89.1%。

### 损失函数 / 训练策略

- 默认使用 SGD（无 momentum）作为基础优化器，保持与 MeZO 相同的内存效率
- 采用范数对齐技巧（Norm Alignment）：将低秩扰动按 $\mu = \sqrt{mn/r^2}$ 缩放，使其范数与全维扰动匹配，从而可直接复用 MeZO 的学习率和扰动尺度超参数
- 在 in-place 操作和逐层参数更新的实现下，内存开销与推理几乎相同

## 实验关键数据

### 主实验（OPT-13B, SuperGLUE 11 任务）

| 方法 | SST-2 | RTE | CB | BoolQ | WSC | WIC | MultiRC | COPA | ReCoRD | SQuAD | DROP | AVG偏差 |
|------|-------|-----|-----|-------|-----|-----|---------|------|--------|-------|------|--------|
| MeZO(FT) | 92.1 | 71.5 | 71.4 | 74.4 | 61.5 | 60.0 | 60.1 | 87.0 | 82.0 | 84.2 | 31.2 | 0% |
| SubZero(FT) | 92.1 | 74.0 | 73.2 | 75.3 | **65.4** | **60.8** | 61.0 | 88.0 | **82.3** | **84.5** | **32.0** | **+1.89%** |
| MeZO(LoRA) | 92.2 | 74.4 | 69.6 | 75.2 | 64.4 | 59.7 | 58.2 | 87.0 | 82.0 | 82.9 | 31.0 | 0% |
| SubZero(LoRA) | **93.8** | **75.5** | 71.4 | **76.1** | **65.4** | **60.3** | **60.3** | **89.0** | 81.9 | **83.7** | 31.3 | **+1.57%** |

### 消融实验

LLaMA2-7B 和 OPT-1.3B 在不同微调方案下的表现：

| 模型 | 方案 | MeZO | SubZero | SGD |
|------|------|------|---------|-----|
| LLaMA2-7B | FT | 64.3 | **71.4** | 69.6 |
| LLaMA2-7B | Prompt | 60.7 | **66.1** | 69.6 |
| OPT-1.3B | FT | 92.3 | **93.4** | 93.2 |
| OPT-1.3B | Prompt | 85.9 | **89.1** | 90.7 |

SubZero 在 LLaMA-7B 全参数微调上比 MeZO 提升 7.1%，甚至超过 SGD。

### 关键发现

- **梯度质量显著提升**（Fig. 1）：SubZero 的梯度与期望梯度的余弦相似度显著高于 MeZO，方差显著更低
- **内存几乎不增加**：OPT-13B 上 SubZero 仅比 MeZO 多 1.73% 内存（26.53 vs 26.08 GB），而 S-RGF 需要 23.8 GB（RoBERTa-large 实验）
- **时间开销可控**：QR 分解带来的额外时间开销在所有 OPT 模型上均 < 9%

## 亮点与洞察

- 核心洞察精准：LLM 微调的梯度快速收敛到低维子空间，这为低秩扰动提供了天然 justification
- Reshape 策略虽然简单，但对 PEFT 场景至关重要，解决了 LoRA 矩阵极端长宽比的问题
- 理论保证完善：证明了梯度估计与 BP 梯度在子空间中的接近性（Theorem 5b）和收敛率 $\mathcal{O}(d/\epsilon)$

## 局限与展望

- 未与二阶 ZO 优化器（如 HiZOO）和动量 ZO（如 ZO-AdaMU）进行系统组合评估
- 收敛率仍依赖参数维度 $d$，虽然常数项通过子空间减小了
- QR 分解的频率 $F$ 和秩 $r$ 的选择目前依赖手动调参
- 理论分析基于二次损失假设，与实际 LLM 损失景观的匹配度有待验证

## 相关工作与启发

- **MeZO** 是最直接的对比，SubZero 在其基础上引入结构化低秩扰动
- **GaLore**（梯度低秩投影）从一阶优化的角度利用了类似的低秩观察
- 与 LoRA 等 PEFT 方法正交且可组合——SubZero 可用于微调 LoRA 适配器的参数

## 评分

- 新颖性：⭐⭐⭐⭐ — 层级低秩扰动 + 惰性更新的设计简洁有效
- 理论深度：⭐⭐⭐⭐ — 梯度逼近和收敛性分析完整
- 实验充分度：⭐⭐⭐⭐⭐ — 多模型、多方案、多任务全面对比
- 实用性：⭐⭐⭐⭐⭐ — 即插即用，内存开销与推理持平

<!-- RELATED:START -->

## 相关论文

- [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)
- [GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](../../ACL2025/llm_nlp/gorp_continual_gradient_projection.md)
- [GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](../../ACL2025/llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)
- [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](../../ACL2025/llm_nlp/salience_sparse_fine_tuning.md)
- [Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)

<!-- RELATED:END -->
