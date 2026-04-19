---
title: >-
  [论文解读] The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM
description: >-
  [ICLR 2026][模型压缩][LLM 剪枝] 提出 Elsa 方法，通过无代理目标的 ADMM 约束优化直接求解稀疏性约束问题，突破 LLM 剪枝 50-60% 的"稀疏墙"瓶颈，在 90% 稀疏度下仍保持高模型保真度。
tags:
  - ICLR 2026
  - 模型压缩
  - LLM 剪枝
  - 极端稀疏性
  - ADMM
  - 无代理目标
  - 网络压缩
---

# The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM

**会议**: ICLR 2026  
**arXiv**: [2510.01650](https://arxiv.org/abs/2510.01650)  
**代码**: [https://github.com/log-postech/elsa](https://github.com/log-postech/elsa)  
**领域**: 模型压缩  
**关键词**: LLM 剪枝, 极端稀疏性, ADMM, 无代理目标, 网络压缩

## 一句话总结

提出 Elsa 方法，通过无代理目标的 ADMM 约束优化直接求解稀疏性约束问题，突破 LLM 剪枝 50-60% 的"稀疏墙"瓶颈，在 90% 稀疏度下仍保持高模型保真度。

## 研究背景与动机

- **LLM 部署挑战**: 大语言模型体积庞大，内存、计算和能耗需求巨大，严重制约广泛部署
- **剪枝瓶颈**: 现有 LLM 剪枝方法（SparseGPT、Wanda、ALPS 等）在 50-60% 稀疏度之后性能急剧下降，形成所谓的"稀疏墙"（sparsity wall）
- **根本原因分析**: 现有方法依赖逐层重建误差最小化的代理目标（surrogate objective），存在三个关键缺陷：
  1. **近似误差累积**: 逐层求解无法达到零重建误差，小误差逐层传播导致整体性能坍塌
  2. **全局次优**: 逐层独立优化限制了搜索空间，且前面层固定后无法根据后续层调整
  3. **代理目标偏差**: 最小化重建误差 $\tilde{f}$ 不等于最小化真正的语言建模目标 $f$

## 方法详解

### 整体框架

Elsa 直接求解稀疏性约束优化问题：

$$x^{\star} = \arg\min f(x) \quad \text{subject to} \quad \|x\|_0 \leq k$$

而非现有方法的逐层代理公式。通过 ADMM（交替方向乘子法）进行变量分裂，将不可处理的稀疏约束与训练目标解耦。

### 关键设计 1: 基于 ADMM 的无代理稀疏化

引入辅助变量 $z$，将问题转换为增广拉格朗日形式：

$$\mathcal{L}_{\lambda}(x,z,u) = f(x) + I_{\mathcal{S}}(z) + \frac{\lambda}{2}\|x - z + u\|_2^2 - \frac{\lambda}{2}\|u\|_2^2$$

交替优化三个子问题：
- **$x$-更新**: 在保持与稀疏 $z$ 接近的约束下最小化训练目标
- **$z$-更新**: 投影到稀疏集合 $\mathcal{S}$，保留最大的 $k$ 个参数
- **$u$-更新**: 对偶变量梯度上升

### 关键设计 2: 目标感知投影（Objective-Aware Projection）

标准投影基于欧几里得距离，与目标函数 $f$ 关联不大。Elsa 使用 Hessian 诱导范数进行投影：

$$z^{t+1} = \arg\min_{z \in \mathcal{S}} \sum_{i \leq d} \hat{\mathbf{F}}_{ii} (z_i - (x_i^{t+1} + u_i^t))^2$$

其中 $\hat{\mathbf{F}}$ 为经验 Fisher 信息矩阵的对角近似，可直接从 Adam 优化器的二阶矩估计中免费获取。

### 关键设计 3: 低精度状态扩展（Elsa-L）

为扩展到超大模型（27B），Elsa-L 对辅助变量使用量化存储：
- 使用 FP8 存储 $z$，BF16 存储 $u$
- 结合 Adam8bit 优化器
- 内存开销较 Elsa 降低 55%

### 收敛性保证

- **Elsa**: 在 $f$ 满足下界、$\beta$-光滑和 $\mu$-弱凸条件下，收敛到原问题的 $\lambda$-驻点
- **Elsa-L**: 在额外量化误差约束下同样收敛，有严格理论证明

## 实验

### 主实验：困惑度 vs 稀疏度

| 模型 | 方法 | 60% PPL | 70% PPL | 80% PPL | 90% PPL |
|------|------|---------|---------|---------|---------|
| OPT-125M | SparseGPT | 49.83 | - | >1000 | - |
| OPT-125M | Elsa | 42.99 | - | 47.45 | - |
| LLaMA-2-7B (90%) | 最优基线 | - | - | - | ~210 |
| LLaMA-2-7B (90%) | Elsa | - | - | - | 26.97 (Wiki) / 23.14 (C4) |
| LLaMA-2-13B (90%) | 其他方法 | - | - | - | >100 |
| LLaMA-2-13B (90%) | Elsa | - | - | - | 27.84 |

### 极端稀疏度实验（LLaMA-2-7B）

| 稀疏度 | 方法 | Wiki PPL | C4 PPL |
|--------|------|----------|--------|
| 90% | Wanda + Full | 42.40 | 34.87 |
| 90% | Elsa | **26.97** | **23.14** |
| 95% | Wanda + Full | 84.30 | 53.62 |
| 95% | Elsa | **38.91** | **28.39** |
| 99% | Wanda + Full | 146.37 | 71.64 |
| 99% | Elsa | **55.94** | **40.10** |

### 实际部署效益（LLaMA-2-7B）

| 稀疏度 | 延迟加速 | 吞吐提升 | 内存压缩 |
|--------|----------|----------|----------|
| 70% | 1.94× | 1.93× | 2.42× |
| 90% | 2.50× | 2.56× | 4.60× |
| 95% | 4.00× | 3.98× | 7.80× |

### 消融实验发现

- 目标感知投影比标准欧几里得投影显著提升性能
- Elsa 方法在所有测试架构（OPT、LLaMA-2/3、Gemma-2）和规模（125M-27B）上一致有效
- 零样本下游任务中，Elsa 在 90% 稀疏度下 7 个任务中 6 个保持最优精度

## 亮点

- **突破稀疏墙**: 首次证明 LLM 可在 90% 甚至 99% 稀疏度下保持有意义的性能
- **理论扎实**: 基于经典 ADMM 优化理论，有严格收敛保证
- **问题诊断深刻**: 系统分析现有方法失败的三大根因（误差累积、局部次优、代理偏差），并提出统一解决策略
- **实用性强**: 90% 稀疏度带来 2.5× 延迟降低和 4.6× 内存压缩

## 局限性

- 需要 4 GPU 训练约 1.78 小时（LLaMA-2-7B, 90%），计算成本高于一次性剪枝方法
- 需要存储完整模型参数和优化器状态，内存需求高于逐层方法
- 目前仅验证了非结构化稀疏和 N:M 半结构化稀疏，未广泛探索其他稀疏模式
- 对 MoE 架构和多模态模型的适用性尚未验证

## 相关工作

- **逐层剪枝**: SparseGPT、Wanda、ALPS、L-ADMM、SAFE、SparseLLM
- **ADMM 剪枝**: L-ADMM 使用逐层 ADMM，但仍受限于代理目标
- **全局优化视角**: 本文首次在 LLM 上成功应用全局无代理 ADMM 稀疏化
- **量化方法**: 与量化正交，Elsa-L 本身也使用低精度技术提升扩展性

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★★ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](mobilellm-r1_exploring_the_limits_of_sub-billion_language_model_reasoners_with_o.md)
- [Modality-free Graph In-context Alignment](modality-free_graph_in-context_alignment.md)
- [DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)

<!-- RELATED:END -->
