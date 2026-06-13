---
title: >-
  [论文解读] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs
description: >-
  [NeurIPS 2025][模型压缩][双稀疏] 提出 DuoGPT，一种将激活稀疏（activation sparsity）重新解释为动态结构化权重稀疏、并与非结构化权重剪枝相结合的双稀疏（dual-sparse）框架，通过扩展 OBC 框架引入激活感知校准和稠密模型输出残差修正项，在不需要重训练的情况下实现 LLM 解码阶段的显著加速与内存节省。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "双稀疏"
  - "激活稀疏"
  - "非结构化剪枝"
  - "OBC框架"
  - "LLM加速"
---

# DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2506.20194](https://arxiv.org/abs/2506.20194)  
**代码**: [GitHub](https://github.com/)（论文中提及）  
**领域**: LLM推理效率 / 模型压缩  
**关键词**: 双稀疏, 激活稀疏, 非结构化剪枝, OBC框架, LLM加速  

## 一句话总结
提出 DuoGPT，一种将激活稀疏（activation sparsity）重新解释为动态结构化权重稀疏、并与非结构化权重剪枝相结合的双稀疏（dual-sparse）框架，通过扩展 OBC 框架引入激活感知校准和稠密模型输出残差修正项，在不需要重训练的情况下实现 LLM 解码阶段的显著加速与内存节省。

## 背景与动机

1. **部署成本高昂**：LLaMA-2-70B 需要约 150GB GPU/CPU 内存和 60 GFLOPs 来解码单个 token，实际部署面临严峻的存储和计算挑战。

2. **结构化 vs 非结构化剪枝的固有权衡**：结构化剪枝加速效果好但精度损失大（如 ShortGPT 在 31.25% 稀疏度下准确率仅 51.35%），非结构化剪枝精度保持好但加速有限——长期以来两者难以兼得。

3. **激活稀疏的内存瓶颈**：虽然 LLM 中普遍存在激活稀疏（由 ReLU/SwiGLU 等激活函数引起），可以跳过零激活对应的权重行从而加速计算，但运行时无法预知哪些行被激活，因此仍需在 GPU HBM 中存储完整的稠密模型，无法实际减少内存。

4. **朴素组合会降低性能**：简单地在已剪枝权重上应用激活稀疏，会因为权重稀疏分布不均匀导致有效计算量接近 $(1 - p^x)$ 的最坏情况，且剪枝校准未考虑运行时激活稀疏带来的误差。

5. **现有 OBC 框架的局限**：SparseGPT 等基于 OBC 的方法仅在稠密激活上做校准，忽略了推理时激活将被稀疏化的事实，导致校准目标与实际推理条件不匹配。

6. **朴素实现计算不可行**：将激活稀疏直接纳入 OBC 框架的推导会产生 $\mathcal{O}(nmk^2 + nk^3)$ 的计算复杂度（LLaMA-2-7B 中 $m = 262144$, $k = 4096$），远超现代 GPU 的实际处理能力。

## 核心问题
如何在 LLM 单次前向剪枝（one-shot pruning）中同时利用权重稀疏和激活稀疏，在保持精度的同时实现解码阶段的计算加速与内存压缩？

## 方法详解

### 核心洞察：激活稀疏 = 动态结构化权重稀疏
在解码阶段（GEMV 操作），激活向量中的零元素意味着对应的权重列不参与计算，等价于对权重矩阵的行做了动态结构化剪枝。结合静态非结构化权重剪枝，构成 spMspV（稀疏矩阵 × 稀疏向量）工作负载。

### 激活感知剪枝校准
传统 OBC 校准目标为 $\|\Delta\mathbf{w}\mathbf{X}\|_F^2$，仅使用稠密输入。DuoGPT 引入非对称校准：
- 使用稀疏激活 $\hat{\mathbf{X}}$（经过量级剪枝）作为校准输入
- 使用稠密模型输出 $\tilde{\mathbf{X}}$ 作为目标，计算残差 $\mathbf{r} = \mathbf{w}(\tilde{\mathbf{X}} - \hat{\mathbf{X}})$
- 校准目标变为 $\|\Delta\mathbf{w}\hat{\mathbf{X}} - \mathbf{r}\|_F^2$，同时适应激活稀疏并补偿信息损失

### 高效实现
通过三步优化将复杂度从 $\mathcal{O}(nmk^2 + nk^3)$ 降至 $\mathcal{O}(mk^2)$：
1. **Hessian 同步**：固定剪枝掩码后所有行共享同一 Hessian，通过 Cholesky 分解一次性预计算
2. **剪枝分数预计算**：将残差 $\mathbf{R}$ 分解为按列的外积之和，得到可向量化的分数公式 $\mathbf{S}_{:,p} = \mathbf{W}_{:,p}^2(1/\mathbf{H}_{pp}^{-1} + \mathbf{a}_p - \mathbf{b}_p + 2\mathbf{c}_p)$
3. **中间量复用**：共享矩阵 $\mathbf{Q} = \Delta\mathbf{X}\hat{\mathbf{X}}^\top\mathbf{L}$ 同时用于计算 $\mathbf{b}$、$\mathbf{c}$ 和补偿项 $\mathbf{D}$

### 理论保证
定理证明 DuoGPT 相比 SparseGPT 的损失改进下界与激活稀疏度 $p^x$ 成线性正比：$\Delta\mathcal{L} \geq \alpha p^x \sigma_r^2 C_\mathbf{w}^2 m / \lambda_{\max}(\mathbf{H})$。

## 实验关键数据

### 表1：与非结构化剪枝基线比较（50%双稀疏）

| 模型 | 方法 | Wiki2 PPL↓ | 平均准确率↑ |
|------|------|-----------|------------|
| LLaMA-3-8B | Dense | 6.14 | 72.71% |
| LLaMA-3-8B | SparseGPT | 14.05 | 59.51% |
| LLaMA-3-8B | Wanda | 15.98 | 57.05% |
| LLaMA-3-8B | **DuoGPT** | **13.41** | **60.04%** |
| LLaMA-3-70B | Dense | 2.86 | 80.09% |
| LLaMA-3-70B | SparseGPT | 7.54 | 71.65% |
| LLaMA-3-70B | **DuoGPT** | **7.38** | **72.56%** |

### 表2：与结构化剪枝基线比较（LLaMA-2-7B）

| 方法 | 模型大小 | 加速比 | 平均准确率↑ |
|------|---------|--------|------------|
| ShortGPT | 4.72B/6.74B | 1.44× | 51.35% |
| 2SSP | 4.72B/6.74B | 1.31× | 57.34% |
| SliceGPT | 5.29B/6.74B | 1.26× | 56.68% |
| **DuoGPT** | **3.50B/6.74B** | **1.39×** | **60.52%** |

DuoGPT 在相近加速比下比最优结构化剪枝方法 ShortGPT 高出 **9.17%** 准确率。

## 亮点

1. **视角独特**：将激活稀疏重解释为动态结构化权重稀疏，巧妙统一了两种稀疏性，无需实际进行结构化剪枝
2. **同时减内存和减计算**：双稀疏方案既压缩模型体积（50%权重剪枝减少存储），又在运行时跳过零激活对应的行（减少计算和 HBM→SRAM 带宽）
3. **高效实现**：70B 参数模型在单张 A100 80GB GPU 上仅需约 2.3 小时完成校准
4. **理论支撑**：提供了 DuoGPT 相对 SparseGPT 的损失改进下界，预测与实验趋势一致

## 局限与展望

1. **仅聚焦解码阶段**：主要针对单 batch 解码的 GEMV 操作，对 prefill 阶段（GEMM）和大 batch 推理场景未做优化
2. **统一稀疏度**：对所有 Transformer 层使用相同的激活稀疏度 $p^x$，未探索逐层自适应稀疏策略
3. **依赖激活稀疏的自然存在**：方法依赖 LLM 中由激活函数引起的自然激活稀疏，对激活稀疏不明显的模型架构可能效果有限
4. **硬件适配未深入**：虽然提到 spMspV 工作负载，但未提供专门的 GPU kernel 实现或实际端到端延迟测量
5. **精度损失在高稀疏度下仍然显著**：65% 双稀疏度下 PPL 从 5.47 升至 77.3，实用性有限

## 与相关工作的对比

- **vs SparseGPT/Wanda**：这些方法做权重剪枝但忽略运行时激活稀疏，校准目标与推理条件不匹配；DuoGPT 通过激活感知校准在所有模型和规模上一致取得更低 PPL 和更高下游准确率
- **vs TEAL/R-Sparse/CATS**：这些方法利用激活稀疏加速推理但保留稠密权重，无法减少模型存储开销；DuoGPT 同时减存储和计算，且 R-Sparse 的 SVD 分支反而增加 1% 内存
- **vs ShortGPT/SliceGPT/2SSP 等结构化剪枝**：在近似加速比（~1.4×）下 DuoGPT 准确率显著更高（60.52% vs 51.35%~57.34%），且模型体积更小
- **vs GPTQ-v2 非对称校准**：借鉴其非对称校准思想但应用于剪枝稀疏化而非量化场景，是首次将该技术引入双稀疏剪枝
- **vs STUN（结构化+非结构化联合剪枝）**：STUN 实际执行两种剪枝，DuoGPT 则将激活稀疏"重解释"为结构化稀疏，校准时只做非结构化剪枝，更加统一优雅

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将激活稀疏统一为动态权重稀疏的视角非常优雅
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 LLaMA-2/3 全系列，消融实验完整，但缺少实际延迟测量
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨清晰，从理论到高效实现的层层推进非常流畅
- 价值: ⭐⭐⭐⭐ — 对 LLM 部署优化有实际意义，双稀疏思路值得后续工作继续探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [\[ACL 2025\] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](../../ACL2025/model_compression/cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](../../ACL2026/model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)
- [\[NeurIPS 2025\] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)

</div>

<!-- RELATED:END -->
