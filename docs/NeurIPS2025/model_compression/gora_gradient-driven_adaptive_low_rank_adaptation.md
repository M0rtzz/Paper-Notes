---
title: >-
  [论文解读] GoRA: Gradient-Driven Adaptive Low Rank Adaptation
description: >-
  [NeurIPS 2025][模型压缩][LoRA] 提出 GoRA，利用**预计算梯度信息**在训练前同时完成自适应秩分配和权重初始化——基于参数敏感度分配各层 rank，用梯度伪逆初始化 $B$ 矩阵使初始输出近似一步梯度下降，统一解决 LoRA 的两大瓶颈。
tags:
  - NeurIPS 2025
  - 模型压缩
  - LoRA
  - 自适应秩分配
  - 梯度驱动初始化
  - 参数高效微调
  - LLM
---

# GoRA: Gradient-Driven Adaptive Low Rank Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2502.12171](https://arxiv.org/abs/2502.12171)  
**代码**: [GitHub](https://github.com/hhnqqq/MyTransformers)  
**领域**: 模型压缩 / LLM效率  
**关键词**: LoRA, 自适应秩分配, 梯度驱动初始化, 参数高效微调, LLM

## 一句话总结

提出 GoRA，利用**预计算梯度信息**在训练前同时完成自适应秩分配和权重初始化——基于参数敏感度分配各层 rank，用梯度伪逆初始化 $B$ 矩阵使初始输出近似一步梯度下降，统一解决 LoRA 的两大瓶颈。

## 研究背景与动机

LoRA 的性能受两个关键因素制约：**秩选择**和**权重初始化**。

**秩分配问题**：
- 更高的秩始终带来更好性能，但直接增加秩的内存开销大
- AdaLoRA 通过训练时 masking 动态调整秩，但需预先分配更大矩阵（1.5倍参数量），限制了秩上界
- MeLoRA 等方法修改 LoRA 结构，牺牲了通用性

**初始化问题**：
- PiSSA/MiLoRA 用 SVD 分解预训练权重初始化——任务无关，泛化受限
- LoRA-GA 用梯度奇异特征初始化——但需修改预训练权重 $W_0 \leftarrow W_0 - A_0 B_0$，造成训练-推理 gap
- 所有非零初始化方法都需要保存修改后的预训练权重，牺牲了 LoRA 的存储优势

**核心洞察**：将 LoRA 适配器重新理解为**梯度压缩器**。LoRA-FA 的实验表明冻结随机 $A$ 只训练 $B$ 即可接近 LoRA 性能，此时 $\Delta W = \frac{\alpha}{r} A_0 \Delta B_t = -\eta \frac{\alpha}{r} \sum_t A_0 A_0^T \frac{\partial \mathcal{L}}{\partial W_0}$，即 $A_0$ 压缩梯度、$A_0$ 再解压。

## 方法详解

### 整体框架

GoRA 分两步，均在正式训练开始前完成：

1. **梯度预计算**：在 $N$ 个 batch 上前向传播，累积每层权重的平均梯度 $G = \frac{1}{N} \sum_i \frac{\partial \mathcal{L}_i}{\partial W_0}$
2. **自适应秩分配**：基于权重-梯度交互的重要性分数确定各层 rank
3. **梯度驱动初始化**：用梯度伪逆初始化 $B_0$，使 $A_0 B_0 \approx -G$

### 关键设计

**自适应秩分配策略**：

1. 计算每层重要性（参数敏感度指标）：
$$I(W) = \text{avg}(|W \odot G|)$$
即权重与梯度 Hadamard 积的绝对值均值——直觉是权重大且梯度大的层更重要

2. 归一化为优势分数：$a^i = I(W_0^i) / \sum_j I(W_0^j)$

3. 基于参考秩 $r^{\text{ref}}$ 计算总参数预算 $b = \sum_i \sqrt{m_i + n_i} \times r^{\text{ref}}$

4. 分配各层秩：
$$r^i = \text{clip}\left(\text{round}\left(\frac{b \cdot a^i}{\sqrt{m+n}}\right), r^{\min}, r^{\max}\right)$$

设计目标：(1) 训练前一次性完成，无动态形状变化；(2) 总参数量与标准 LoRA 相当（±10%）；(3) 结构兼容 LoRA。

**梯度驱动初始化**：

$A_0$ 用 Kaiming 均匀分布初始化（与 PEFT 库一致），$B_0$ 用梯度的伪逆初始化：

$$B_0 = -(A_0^T A_0)^{-1} A_0^T G$$

这使得 $A_0 B_0 = -A_0(A_0^T A_0)^{-1} A_0^T G$ 成为 $G$ 在 $A_0$ 列空间上的最优低秩近似（最小化 $\|A_0 B_0 + G\|_F$）。

**缩放因子 $\xi$**：为匹配真实梯度幅度，引入缩放：
$$\frac{\alpha}{\sqrt{r}} A_0(\xi B_0) \approx -\gamma G$$
其中 $\xi = \gamma \sqrt{m} / \alpha$，$\gamma$ 为可调步长超参数（推荐 5e-2）。

**无需修改预训练权重**：与 PiSSA/LoRA-GA 不同，GoRA 不需要 $W_0 \leftarrow W_0 - A_0 B_0$，因为初始化的目标是让 $A_0 B_0 \approx -G$（近似一步梯度下降），而非分解 $W_0$。

### 损失函数 / 训练策略

- 前向计算：$W_t = W_0 + \frac{\alpha}{\sqrt{r}} A_t B_t$（借鉴 rsLoRA 的 $\sqrt{r}$ 缩放）
- 标准微调损失（next-token prediction 或分类损失）
- 正式训练过程与 LoRA 完全一致，GoRA 的创新全在初始化阶段

## 实验关键数据

### 主实验

**T5-Base 在 GLUE 上的结果**（$r^{\text{ref}}=8$）：

| 方法 | MNLI | SST-2 | CoLA | QNLI | MRPC | 平均 |
|------|------|-------|------|------|------|------|
| Full FT | 86.33 | 94.75 | 80.70 | 93.19 | 84.56 | 87.91 |
| LoRA | 85.30 | 94.04 | 69.35 | 92.96 | 68.38 | 82.08 |
| LoRA-GA | 85.70 | 94.11 | **80.57** | 93.18 | 85.29 | 87.77 |
| AdaLoRA | 85.45 | 93.69 | 69.16 | 91.66 | 68.14 | 81.62 |
| **GoRA** | **85.91** | **94.68** | 79.86 | **93.27** | **86.10** | **87.96** |

**Llama-3.1-8B-Base 生成任务**：

| 方法 | MTBench | GSM8k | HumanEval |
|------|---------|-------|-----------|
| Full FT | 5.88 | 73.69 | 51.63 |
| LoRA | 6.15 | 67.78 | 43.09 |
| LoRA-GA | 5.99 | 71.39 | 43.29 |
| **GoRA** | 6.34 | **72.91** | **48.98** |
| GoRA ($r^{\text{ref}}=128$) | 5.82 | **75.74** | **52.03** |

GoRA ($r^{\text{ref}}=128$) 在 GSM8k 和 HumanEval 上**超越全参数微调**。

### 消融实验

**秩分配范围的影响**（Llama-3.1-8B，$\gamma=5e{-2}$）：

| $r^{\min}$ | $r^{\max}$ | GSM8k | HumanEval |
|-----------|-----------|-------|-----------|
| 8 | 8（固定） | 72.10 | 44.75 |
| 6 | 15 | 72.25 | 45.85 |
| 4 | 32 | **72.88** | **48.98** |

**初始化缩放因子 $\gamma$ 的影响**：

| $\gamma$ | GSM8k | HumanEval |
|---------|-------|-----------|
| 0（无初始化） | 72.45 | 46.34 |
| 3e-2 | 72.71 | 45.93 |
| 5e-2 | 72.88 | **48.98** |
| 8e-2 | **72.91** | 46.54 |

**重要性指标对比**：

| 指标 | GSM8k | HumanEval |
|------|-------|-----------|
| $\text{avg}(\|W \odot G\|)$（GoRA） | **72.88** | **48.98** |
| $\|G\|_*$（核范数） | 72.70 | 43.09 |
| $\|W \odot G\|_*$ | 72.65 | 45.12 |

### 关键发现

1. **更宽的秩范围更好**：$(r^{\min}=4, r^{\max}=32)$ 显著优于固定 rank=8，大部分秩分配给了 $W_v$ 层
2. **初始化至关重要**：$\gamma=0$（无初始化）比最优 $\gamma$ 在 HumanEval 上低 2.64 分
3. **参数敏感度指标最佳**：优于梯度核范数和权重-梯度积的核范数
4. **跨模态一致**：在 NLU（T5）、NLG（Llama）和视觉分类（CLIP-ViT）上均优于基线
5. **高秩时超越全参数微调**：GoRA $r^{\text{ref}}=128$ 在数学推理上比全参数微调高 2.05 分

## 亮点与洞察

- **LoRA 即梯度压缩器**：这一重新解释统一了秩分配和初始化的设计逻辑
- **训练前一次性完成**：无需运行时动态调整，与分布式训练（FSDP/ZeRO）完全兼容
- **无训练-推理 gap**：不修改预训练权重，保留 LoRA 的存储优势
- **超参数自动调节**：提出自适应梯度累积停止和自适应 $\gamma$ 搜索策略，接近手动调参效果
- **秩分布模式**：$W_v$ 获得最多秩、$W_q$ 最少，与 LoRA 原始论文的观察一致

## 局限与展望

1. 梯度预计算需要额外的前向传播开销（$N$ 个 batch），对超大模型不可忽略
2. $\gamma$ 的最优值因任务而异（GSM8k 偏好 8e-2，HumanEval 偏好 5e-2）
3. $r^{\min}$ 和 $r^{\max}$ 仍需手动选择，虽然论文表明宽范围普遍较好
4. 伪逆计算假设 $A_0$ 满秩，极低秩时可能不稳定
5. 未探索与 QLoRA（量化）的组合效果

## 相关工作与启发

- **LoRA-GA**：同为梯度驱动初始化，但需修改预训练权重；GoRA 通过缩放因子避免了这个问题
- **AdaLoRA**：同为自适应秩方法，但通过训练时 masking，参数量膨胀 1.5 倍；GoRA 训练前分配无额外开销
- **rsLoRA**：$\alpha/\sqrt{r}$ 缩放规则被 GoRA 采用以更好利用高秩
- **启发**：梯度压缩器视角可能适用于其他 PEFT 方法（Adapter、Prefix-tuning），值得探索梯度驱动的统一框架

## 评分

- **新颖性**: ⭐⭐⭐⭐ — LoRA 即梯度压缩器的视角新颖，统一了两个独立问题
- **技术深度**: ⭐⭐⭐⭐ — 理论分析清晰，伪逆最优性有证明
- **实验充分度**: ⭐⭐⭐⭐⭐ — NLU/NLG/视觉三模态，多模型多基准，消融全面
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用，兼容 LoRA 生态，代码已开源
- **总体**: ⭐⭐⭐⭐

## 背景与动机

## 核心问题

## 方法详解

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)

<!-- RELATED:END -->
