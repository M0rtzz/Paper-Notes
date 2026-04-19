---
title: >-
  [论文解读] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning
description: >-
  [NeurIPS 2025][知识蒸馏] 提出Learning to Focus (LeaF)框架，通过梯度引导识别训练数据中的"混淆token"（confounding tokens），在知识蒸馏过程中剪枝这些token以构建反事实样本，使学生模型的注意力对齐到教师模型关注的关键上下文token上，从而提升数学推理和代码生成的准确性。
tags:
  - NeurIPS 2025
  - 知识蒸馏
  - 因果推理
  - 注意力对齐
  - Token剪枝
  - 混淆因子
---

# Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning

**会议**: NeurIPS 2025

**arXiv**: [2506.07851](https://arxiv.org/abs/2506.07851)

**代码**: 无

**领域**: 模型压缩 / 知识蒸馏

**关键词**: 知识蒸馏, 因果推理, 注意力对齐, Token剪枝, 混淆因子

## 一句话总结

提出Learning to Focus (LeaF)框架，通过梯度引导识别训练数据中的"混淆token"（confounding tokens），在知识蒸馏过程中剪枝这些token以构建反事实样本，使学生模型的注意力对齐到教师模型关注的关键上下文token上，从而提升数学推理和代码生成的准确性。

## 研究背景与动机

大语言模型在长上下文推理和复杂任务中常常被"干扰模式"（distracting patterns）误导注意力，导致推理错误。作者通过初步实验发现了一个关键现象：

- 在数学训练语料上，直接移除干扰模式后，小模型准确率提升超过 **20%**
- 在代码训练语料上，移除干扰模式后准确率提升超过 **10%**
- 复杂推理任务（如AMC_AIME）比简单任务（如GSM8K）受干扰模式的影响更大

作者将此现象归因于训练数据中的**虚假相关性**（spurious correlations），这些相关性阻碍了模型推断真正的因果指令-响应关系。传统知识蒸馏仅关注输出模仿，未能解决这一根本问题。

## 方法详解

### 整体框架

LeaF是一个两阶段框架，基于Pearl的结构因果模型（SCM），将推理过程建模为因果图：

- **输入token** $X = [x_1, x_2, \ldots, x_n]$
- **混淆token** $A \subset X$：引入虚假相关性的token子集
- **输出** $Y$：模型的推理结果

混淆token $A$ 同时影响 $X$ 和 $Y$，使观测分布偏离干预分布：

$$P(Y|X_i=x) = \sum_A P(Y|X_i=x, A) P(A|X_i=x)$$

### 关键设计

#### 阶段一：混淆Token检测（Confounding Token Detection）

采用梯度敏感性方法比较教师和学生模型：

1. 对每个token $x_i$，计算教师和学生模型的梯度敏感性：

$$g_i^{(T)} = \left|\frac{\partial \ell(x_i|X; \theta_T)}{\partial x_i}\right|, \quad g_i^{(S)} = \left|\frac{\partial \ell(x_i|X; \theta_S)}{\partial x_i}\right|$$

2. 对梯度进行min-max归一化后，计算差异：

$$\Delta\hat{g}_i = \hat{g}_i^{(T)} - \hat{g}_i^{(S)}$$

3. 当归一化后的差异低于阈值 $\tau_{\text{confounder}}$ 时（即学生模型高度关注但教师模型忽略的token），标记为混淆token

4. 额外验证：移除该token后两个模型都能正确预测

#### 剪枝策略

- **集体剪枝**（Collective Pruning）：一次性移除所有混淆token → 会破坏句子完整性
- **跨度剪枝**（Span Pruning）：每次只移除一个连续混淆跨度 $A_i$，生成多个反事实样本 → 效果更好

$$\mathcal{D}_{\text{pruned}} = \{(X \setminus A_i, y)\}_{i=1}^k$$

#### 阶段二：因果注意力蒸馏（Causal Attention Distillation）

两个互补的蒸馏目标：

- **标准蒸馏损失**：在原始指令上对齐

$$\mathcal{L}_{kd} = D_{\text{KL}}(p_T(y|X) \| p_S(y|X))$$

- **反事实蒸馏损失**：在剪枝后的指令上对齐

$$\mathcal{L}_{cd} = D_{\text{KL}}(p_T(y|X \setminus A) \| p_S(y|X \setminus A))$$

### 损失函数 / 训练策略

综合损失函数：

$$\mathcal{L} = \lambda \mathcal{L}_{kd} + (1-\lambda) \mathcal{L}_{cd}$$

其中 $\lambda \in [0,1]$ 控制标准蒸馏与反事实蒸馏的权衡。

**响应分割策略**：
- 指令级剪枝：仅在指令部分检测和剪枝混淆token
- 响应级剪枝：将之前生成的token也视为上下文，检测并剪枝误导后续生成的token（2段/3段分割）

**训练超参数**：采用Alpaca-LoRA框架，全参数logits蒸馏，余弦学习率调度，最大学习率 $10^{-5}$，训练3个epoch。

## 实验关键数据

### 主实验

| 模型 | GSM8K | MATH | OlympiadBench | 平均 | HumanEval+ | LeetCode | LivecodeBench | 平均 |
|------|-------|------|---------------|------|------------|----------|---------------|------|
| **教师: LLaMA3.3-70B** | 95.60 | 70.40 | 36.50 | 67.50 | 78.05 | 53.90 | 45.02 | 58.99 |
| **LLaMA3.2-1B (原始)** | 44.88 | 24.20 | 5.79 | 24.96 | 29.27 | 7.22 | 9.68 | 15.39 |
| KD w/o Mask | 56.79 | 33.40 | 8.90 | 33.03 | 32.32 | 6.11 | 13.74 | 17.39 |
| LeaF (Instr Mask) | 57.70 | 35.40 | 10.09 | 34.40 | - | - | - | - |
| **LLaMA3.2-3B (原始)** | 77.56 | 42.80 | 14.83 | 45.06 | 56.71 | 20.00 | 21.58 | 32.76 |
| KD w/o Mask | 80.59 | 50.00 | 18.99 | 49.86 | 59.76 | 24.44 | 23.87 | 36.02 |
| LeaF (Resp Mask) | **82.26** | **54.40** | **20.03** | **52.23** | - | - | - | - |

**关键发现**：LeaF在LLaMA-1B/3B上相比标准KD平均提升 **2.41%**（数学），**2.48%**（代码）。

### 消融实验

| 剪枝策略 | MATH-500 (1B) | MATH-500 (3B) |
|----------|---------------|---------------|
| 标准KD（无剪枝） | 34.00 | 50.00 |
| 集体剪枝 | 34.20 | 49.20 (↓) |
| **跨度剪枝** | **37.40** | **54.40** |

**掩码策略对比**：
- 随机掩码：在GSM8K和Olympiad上性能下降
- PPL掩码：在简单任务上有小幅提升，但在复杂任务上与随机掩码相当
- 梯度掩码（本文）：在所有任务上一致优于两个基线

**阈值敏感性分析**：
- 1B模型最优阈值较高（指令级0.10，响应级0.15）
- 3B模型最优阈值较低（指令级0.05，响应级0.10）
- 小模型更容易受混淆token影响，需要更高阈值来过滤

### 关键发现

1. 响应级剪枝（2段分割）显著优于指令级剪枝，表明响应中的干扰模式对后续生成也有重要影响
2. 3段分割与2段分割性能相当，进一步分割收益递减
3. 梯度方法在需要教师指导的复杂推理场景中不可替代
4. 注意力可视化证实LeaF使模型更关注关键信息如"实数""所有""均为实数"等约束条件

## 亮点与洞察

1. **因果视角的独特性**：将知识蒸馏中的注意力偏差问题建模为因果推理中的混淆因子问题，提供了理论解释
2. **实证发现有力**：仅移除混淆token（不做额外训练）就能提升20%+准确率，强有力地支持了核心假设
3. **可解释性**：通过注意力热图可视化，清楚展示了LeaF如何引导模型关注关键信息
4. **跨域有效性**：在数学推理和代码生成两个不同领域都展示了一致的改进

## 局限与展望

1. **依赖高能力教师模型**：混淆token检测需要教师-学生梯度对比，无法自我改进
2. **长文本泛化不足**：目前仅在数学和代码任务上验证，长文本理解等领域尚待探索
3. **计算开销**：需要同时计算教师和学生的梯度，增加了预处理成本
4. **潜在方向**：探索自改进机制，使模型无需教师即可识别自身的混淆注意力

## 相关工作与启发

- **推理一致性**：Self-Consistency等方法关注解码阶段的一致性，LeaF则从训练阶段入手
- **CoT知识蒸馏**：CD、SCORE等关注数据质量和多样性，LeaF关注数据中的因果结构
- **关键Token识别**：RHO-1、TokenSkip等工作从不同角度识别重要token，LeaF通过教师-学生梯度差异进行跨模型对比

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 因果视角+梯度引导的混淆token检测是新颖的组合
- **技术深度**: ⭐⭐⭐⭐ — 因果建模严谨，实验全面
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多模型、多任务、多消融，实验非常充分
- **实用性**: ⭐⭐⭐⭐ — 即插即用的蒸馏框架增强方法
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](../../ICML2025/interpretability/why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)
- [Dataset Distillation for Pre-Trained Self-Supervised Vision Models](dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)

<!-- RELATED:END -->
