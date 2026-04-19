---
title: >-
  [论文解读] SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity
description: >-
  [ICML2025][LoRA] 提出 SparseLoRA，通过**上下文稀疏性 (contextual sparsity)** 动态选择权重子集进行前向/梯度计算，首次将推理时的稀疏加速思路迁移到 LLM 微调阶段，实现最高 2.2× FLOPs 降低和 1.6× 实测加速，同时保持精度。
tags:
  - ICML2025
  - LoRA
  - 上下文稀疏性
  - SVD稀疏估计器
  - 微调加速
  - 参数高效微调
---

# SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity

**会议**: ICML2025  
**arXiv**: [2506.16500](https://arxiv.org/abs/2506.16500)  
**代码**: [https://z-lab.ai/projects/sparselora](https://z-lab.ai/projects/sparselora)  
**领域**: LLM微调加速 / 模型压缩  
**关键词**: LoRA, 上下文稀疏性, SVD稀疏估计器, 微调加速, 参数高效微调

## 一句话总结

提出 SparseLoRA，通过**上下文稀疏性 (contextual sparsity)** 动态选择权重子集进行前向/梯度计算，首次将推理时的稀疏加速思路迁移到 LLM 微调阶段，实现最高 2.2× FLOPs 降低和 1.6× 实测加速，同时保持精度。

## 研究背景与动机

- **参数高效微调 (PEFT)** 方法如 LoRA、QLoRA、DoRA 虽然减少了可训练参数和显存占用，但**不减少计算量**，有时甚至更慢（DoRA 比 LoRA 慢 20%）。
- **上下文稀疏性**已在 LLM 推理加速中验证有效（Deja Vu 等），但从未被用于微调场景。
- 微调时的工作负载与推理不同：推理是单 token 自回归，微调是多序列 batch，需要新的稀疏选择策略。
- **核心观察**：微调时线性层占主导运行时间（尤其长序列），仅稀疏化主干分支即可获得显著加速，LoRA 分支本身计算量极小无需处理。

## 方法详解

### 整体框架

SparseLoRA 在 LoRA 微调基础上，仅对**主干权重分支**施加动态通道稀疏，保持 LoRA 适配器分支完整。步骤：

1. **离线 SVD 分解**：对预训练权重做 SVD 分解，构建轻量稀疏估计器
2. **在线稀疏预测**：根据输入 batch，用 SVD 估计器快速判断哪些通道需要激活
3. **稀疏计算**：动态切片权重，仅用激活通道进行前向和反向传播

### 稀疏神经元选择准则

将 LLM 线性层分为三类，提出两种选择准则：

**L2 Norm 准则（FFN + VO 投影）**：利用 SiLU 等激活函数导致的中间激活稀疏性，按 batch 内所有样本和 token 的 L2 范数累积选择最重要的通道。FFN 的 down projection 选中通道索引可直接复用到 gate/up projection。

**QK Norm 准则（QK 投影）**：QK 投影的输入激活稀疏性低，L2 Norm 不适用。提出基于注意力分数的准则：

$$\mathbf{q} = \|\mathbf{Q}\|_2, \quad \mathbf{k} = \|\mathbf{K}\|_2$$

$$\mathbf{s} = \mathbf{q} \odot \mathbf{k}$$

按 $\mathbf{s}$ 保留 top-n 通道，确保对注意力分数贡献最大的投影通道被保留。

### SVD 稀疏估计器

直接计算 oracle 准则需要部分全量前向计算，开销太大。解决方案：

- 对权重 $W$ 做 top-k SVD 分解：$W_A = U_{:,:k} \cdot \text{diag}(S_{:k})^{1/2}$，$W_B = \text{diag}(S_{:k})^{1/2} \cdot V_{:k,:}$
- 用低秩近似 $x \cdot W_A \cdot W_B$ 代替全量计算来预测稀疏掩码
- **免训练**（不同于 Deja Vu 的 look-ahead predictor），泛化性更好
- 额外开销仅 **0.05% FLOPs / 0.8% 运行时间**

### 三维敏感性分析

**层敏感性 → 非均匀稀疏**：深层冗余更多，可施加更激进的稀疏；浅层保持稠密。

**Token 敏感性 → 上下文-输出感知稀疏**：输出 token（用于计算 loss 的目标 token）对剪枝更敏感。策略：仅对 context token 施加稀疏，output token 保持稠密计算。

**步骤敏感性 → 渐进式稀疏微调**：前期迭代（最多 10%）保持稠密微调，建立好梯度基础后再切换到稀疏微调。

## 实验关键数据

### 常识推理 (CSR170K, 8个数据集平均)

| 方法 | FLOPs | 加速比 | 平均准确率 |
|------|-------|--------|-----------|
| LLaMA3-8B + LoRA | 100% | 1.0× | 87.1 |
| LLaMA3-8B + QLoRA | 100% | 0.9× | 87.1 |
| LLaMA3-8B + DoRA | 132% | 0.8× | 87.1 |
| **LLaMA3-8B + SparseLoRA** | **65%** | **1.3×** | **86.9** |
| LLaMA2-13B + LoRA | 100% | 1.0× | 84.7 |
| **LLaMA2-13B + SparseLoRA** | **61%** | **1.3×** | **85.0** |

### 数学推理 (Math10K)

| 方法 | FLOPs | 加速比 | 平均准确率 |
|------|-------|--------|-----------|
| LLaMA3-8B + LoRA | 100% | 1.0× | 81.0 |
| **LLaMA3-8B + SparseLoRA** | **46%** | **1.6×** | **81.1** |

### 指令跟随 (MT-Bench)

| 方法 | FLOPs | 加速比 | 平均分 |
|------|-------|--------|-------|
| LLaMA3.1-8B + LoRA | 100% | 1.0× | 6.03 |
| **LLaMA3.1-8B + SparseLoRA** | **53%** | **1.5×** | **6.06** |

### GLUE 序列分类

| 方法 | FLOPs | 加速比 | 平均分 |
|------|-------|--------|-------|
| LLaMA3-8B + LoRA | 100% | 1.0× | 87.3 |
| **LLaMA3-8B + SparseLoRA** | **61%** | **1.3×** | **87.7** |

## 亮点与洞察

1. **首次将上下文稀疏性从推理迁移到微调**，验证了微调阶段同样存在可利用的输入相关稀疏模式
2. **SVD 估计器免训练**，避免了学习式预测器在数据集间泛化不佳的问题，且开销极低（<1%）
3. **三维敏感性分析**（层/token/步骤）系统性地解决了稀疏微调的精度坍塌问题
4. **与现有 PEFT 正交**：SparseLoRA 减少计算量，LoRA/QLoRA 减少内存，可组合使用
5. 在 LLaMA2-13B 上 SparseLoRA 甚至**超过 LoRA 精度**（85.0 vs 84.7），说明适度稀疏有正则化效果

## 局限与展望

1. **敏感性分析需要离线搜索**：层稀疏配置依赖代理任务的敏感性分析，换任务可能需要重做
2. **实测加速 < 理论加速**：FLOPs 减少 2.2× 但实测仅 1.6×，受限于 GPU 稀疏计算效率和内存带宽
3. **仅验证了 LoRA 基础**：虽声称可扩展到其他 PEFT，但缺少与 DoRA/GaLore 组合实验
4. **注意力层稀疏粒度受限**：QK 通道级稀疏比 FFN 通道级稀疏效果差，头级稀疏又太粗糙
5. **缺少超大模型验证**：实验止于 13B，70B+ 模型效果未知

## 相关工作与启发

- **Deja Vu (Liu et al., 2023)**：推理时上下文稀疏性的先驱，用学习式预测器选择稀疏掩码
- **GaLore (Zhao et al., 2024)**：利用梯度低秩性减少优化器内存，与 SparseLoRA 互补
- **LongLoRA (Chen et al., 2024)**：通过 shifted sparse attention 加速长上下文微调
- 启发：稀疏性 + 低秩性是加速微调的两大正交维度，未来可探索联合优化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将上下文稀疏性系统性地引入微调，三维敏感性分析设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 覆盖推理/分类/代码/指令跟随多任务，多模型多基线对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机到方法论述流畅
- 价值: ⭐⭐⭐⭐ — 填补微调计算效率空白，与现有内存优化方法正交可组合

<!-- RELATED:START -->

## 相关论文

- [EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)
- [A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](../../NeurIPS2025/code_intelligence/a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)

<!-- RELATED:END -->
