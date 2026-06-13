---
title: >-
  [论文解读] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs
description: >-
  [ACL2025][LLM 其他][PEFT] 提出 GIFT-SW，一种新型参数高效微调方法：仅更新权重矩阵中的"显著列"(salient columns)，同时对非显著列注入高斯噪声，在相同计算预算下超越全参微调和 LoRA/DoRA 等现代 PEFT 方法。 问题定义 大语言模型（LLM）在经过预训练后…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "PEFT"
  - "Salient Weights"
  - "Gaussian Noise Injection"
  - "量化"
  - "LLM Fine-tuning"
---

# GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs

**会议**: ACL2025  
**arXiv**: [2408.15300](https://arxiv.org/abs/2408.15300)  
**代码**: 有（论文中提到）  
**领域**: LLM NLP / 参数高效微调  
**关键词**: PEFT, Salient Weights, Gaussian Noise Injection, Quantization, LLM Fine-tuning  

## 一句话总结

提出 GIFT-SW，一种新型参数高效微调方法：仅更新权重矩阵中的"显著列"(salient columns)，同时对非显著列注入高斯噪声，在相同计算预算下超越全参微调和 LoRA/DoRA 等现代 PEFT 方法。

## 研究背景与动机

### 问题定义
大语言模型（LLM）在经过预训练后，通常还需要进行微调以提升在特定任务上的表现，或在量化/剪枝等压缩操作后恢复性能。然而全参微调的计算和内存开销极大，现有的参数高效微调（PEFT）方法（如 LoRA）虽然降低了资源需求，但在准确率上仍难以匹配全参微调。

### 核心观察

**显著权重的重要性**：先前研究表明，LLM 中存在一小部分"显著权重"（weight outliers），它们对量化（PTQ）和剪枝的效果有显著影响。Dettmers 等人发现输入激活中的极少数 outlier 对模型性能影响巨大。

**噪声注入的正则化效果**：扰动梯度下降（PGD）通过在梯度步骤前/后注入噪声，能帮助模型逃离鞍点和局部最优，起到稳定收敛和防止过拟合的作用。

**量化噪声与高斯噪声的等价性**：量化感知训练（QAT）中的量化噪声注入（QNI）与高斯噪声注入（GNI）在数学形式上具有等价性，因此 GNI 也可以增强模型在后续量化中的鲁棒性。

### 关键问题
论文围绕三个核心问题展开：
- Q1: 仅更新显著权重子集是否足以有效调整模型？
- Q2: 噪声注入是否有助于收敛？
- Q3: 噪声注入是否有助于提升鲁棒性？

## 方法详解

### 整体框架

GIFT-SW 方法包含三个步骤：
1. **识别显著列**：使用敏感度指标，基于小的校准集，在每层中选出固定数量（默认128列）的显著列
2. **分割列**：将权重矩阵的列分为显著列子集和非显著列子集
3. **训练过程**：仅更新显著列的权重，对非显著列注入高斯噪声且保持冻结

### 关键设计1：通用化敏感度指标

论文提出了一个通用的列敏感度指标：

$$s_j = \|\mathbf{D}_j\|_\tau \|\mathbf{X}_j\|_\rho^\gamma$$

其中：
- $\mathbf{D}_j$ 是权重扰动度量（量化误差 $\mathbf{W}_{:,j} - Q(\mathbf{W}_{:,j})$）
- $\mathbf{X}_j$ 是输入特征（通过校准集计算）
- $\gamma \in \{1/2, 1, 2\}$，$\tau, \rho$ 为范数选择参数

这一指标统一了此前多种方法的指标：
- QUIK/SmoothQuant 使用 $\|\mathbf{X}\|_\infty$
- OWQ 使用 $\lambda_j \|\mathbf{D}_j\|_2^2$（对应 OBD 度量）
- Wanda 使用 $\|\mathbf{D}_j\|_1 \|\mathbf{X}_j\|_2$ 的逐元素变体

论文最终选择 $\gamma=1, \rho=\infty, \tau=\infty$ 的指标，理由是 $l_\infty$ 范数能更好地捕捉激活和权重误差中的极端异常值，不会被大量小值平均掉。

### 关键设计2：量化噪声注入

对于非显著列，注入噪声的操作定义为：

$$\mho(\mathbf{W}) = \begin{cases} \mathbf{W}_{[:,\text{salient}]} \\ \mathbf{W}_{[:,\text{non-salient}]} + \frac{1}{2}\text{diag}(\mathbf{\Delta})\mathbf{\Omega} \end{cases}$$

其中 $\mathbf{\Omega} \sim \mathcal{N}(0, 1)$，$\mathbf{\Delta}$ 为量化步长矩阵。

噪声参数设计的关键细节：
- **按行计算缩放因子**：$\Delta_i = \frac{\alpha_i}{2^{b-1}-1}$，其中 $b$ 为位宽
- **排除显著列**：计算缩放因子时排除显著列，因为它们可能导致异常大的量化误差并扭曲行级缩放因子
- **前向传播注入**：噪声在前向传播中注入（gradient step 之前），起正则化作用
- **仅对非显著列注入**：避免对敏感权重进行扰动产生模型退化

### 损失函数

采用标准的条件语言建模目标（即 next token prediction），关键区别在于：
- 仅显著列权重参与梯度更新
- 非显著列冻结但在每次前向传播时添加噪声
- 这等价于一种特殊的扰动梯度下降（PGD）

## 实验

### 实验设置
- **模型**：LLaMA2-7B, LLaMA2-13B, LLaMA3-8B
- **数据**：TÜLU-V2-mix（主数据集）, OpenOrca
- **训练**：500 iterations, 4 GPUs (40GB), batch size 128(7B)/64(13B)
- **显著列数**：固定128列（约3%参数）
- **评估**：HellaSwag, BoolQ, WinoGrande, PiQA, ARC-easy, ARC-challenge 的零样本平均准确率

### 主实验结果

| 方法 | LLaMA2-7B (TÜLU) | LLaMA2-13B (TÜLU) | LLaMA3-8B (TÜLU) |
|------|-------------------|---------------------|---------------------|
| Full FT | 71.97 | 75.09 | 76.13 |
| LoRA | 71.78 | 74.03 | 75.91 |
| DoRA | 72.03 | 73.97 | 75.89 |
| **GIFT-SW** | **73.33** | **75.93** | **76.37** |

**关键发现**：GIFT-SW 在全精度设置下超越全参微调和 LoRA/DoRA，在所有模型和大部分数据集上表现最优。

### 量化模型结果

| 位宽 | 方法 | LLaMA2-7B | LLaMA2-13B | LLaMA3-8B |
|------|------|-----------|------------|-----------|
| 4-bit | STE | 72.43 | **75.29** | 74.84 |
| 4-bit | QUIK+LoRA | 63.99 | 71.08 | 74.27 |
| 4-bit | **GIFT-SW** | **72.53** | 74.50 | **75.46** |
| 3-bit | **GIFT-SW** | **71.00** | 74.34 | **73.27** |
| 2-bit | **GIFT-SW** | **61.09** | **67.61** | **58.89** |

**关键发现**：在2-bit极低精度下，GIFT-SW 的优势最为突出，超过第二名5个百分点以上。

### 与 TÜLU2 的对比

| 方法 | LLaMA2-7B 性能 | 可训参数/迭代 |
|------|---------------|-------------|
| TÜLU2 | 73.49 | 6.7B / 5K |
| TÜLU2-DPO | 73.80 | 6.7B / 5K |
| **GIFT-SW** | 73.33 | **174M / 500** |

GIFT-SW 仅用 174M 可训参数和 500 次迭代，就达到了 TÜLU2（需要 6.7B 参数、5K 迭代）的可比水平。

### 消融实验

**1. 噪声注入的影响**

| 方法 | w/ Noise | w/o Noise |
|------|----------|-----------|
| Outlier FT (7B) | **73.33** | 73.16 |
| Outlier FT (13B) | **75.93** | 74.80 |
| Full FT (7B) | 71.64 | 71.97 |

噪声注入对显著列微调始终有益，但对全参微调可能反而降低性能。

**2. 敏感度指标对比**：$\|\mathbf{D}_j\|_\infty \|\mathbf{X}_j\|_\infty$ 在大多数设置下为最优指标，但没有单一指标在所有位宽和模型上占绝对优势。

**3. 量化前 vs 后训练**：
- 4-bit：先训练再量化（Pre-GIFT-SW）最优
- 2-bit：先量化再训练（Post-GIFT-SW）最优，因为极低位量化导致权重偏移过大，破坏了预训练时形成的权重关系

### 数据扩展稳定性
GIFT-SW 在不同数据预算下表现出与全参微调相当的稳定性，而 LoRA/DoRA 则表现出明显的不稳定性。

## 亮点与洞察

1. **简洁优雅的设计**：GIFT-SW 不需要引入额外参数（如 LoRA 的低秩矩阵），而是直接操作原始权重的一个子集，同时利用噪声注入作为正则化手段
2. **统一框架**：将来自量化（QUIK, OWQ, SmoothQuant）和剪枝（Wanda）两个领域的显著权重识别指标统一到同一个通用公式中
3. **噪声的双重角色**：噪声注入既起到训练时的正则化作用，又天然地为后续量化做了"预训练"
4. **极低位量化的优势**：在2-bit量化场景下优势尤为突出，说明方法在资源受限的极端场景下更有价值
5. **计算效率极高**：仅需约3%的参数和1/10的计算资源即可达到TÜLU2等重量级方法的效果

## 局限性

1. 仅在 LLaMA 系列模型上验证，未测试其他架构的 LLM
2. 量化实验仅使用 GPTQ 方法
3. 敏感度指标选择仍缺乏明确理论指导
4. 噪声参数仅使用 QNI 方式确定，其他噪声分布可能更优
5. 未提供 CUDA kernel 的高效实现
6. 仅使用少量微调指令集进行实验
7. 评估基准有限（仅6个零样本任务）

## 相关工作

- **PEFT方法**：LoRA, DoRA, prompt tuning 等
- **显著权重识别**：SparseGPT, Wanda, OWL（剪枝）; QUIK, OWQ, SmoothQuant（量化）
- **噪声注入**：Orvieto et al. 的单层GNI, Liu et al. 的可学习层级方差噪声
- **量化感知训练**：STE vs QNI, Défossez et al., Shin et al.

## 评分 ⭐⭐⭐⭐

方法设计简洁有效，实验完整且有说服力，特别是统一敏感度指标和噪声注入的双重角色设计颇有洞察力。在极低位量化场景下优势显著。不足之处在于模型覆盖面有限，且理论分析深度可以进一步加强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_nlp/space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)

</div>

<!-- RELATED:END -->
