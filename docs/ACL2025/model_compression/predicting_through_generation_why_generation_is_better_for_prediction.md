---
title: >-
  [论文解读] Predicting Through Generation: Why Generation Is Better for Prediction
description: >-
  [ACL2025][模型压缩][PredGen] 本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。
tags:
  - "ACL2025"
  - "模型压缩"
  - "PredGen"
  - "Generation for Prediction"
  - "Data Processing Inequality"
  - "Exposure Bias"
  - "Task Adapter"
---

# Predicting Through Generation: Why Generation Is Better for Prediction

**会议**: ACL2025  
**arXiv**: [2502.17817](https://arxiv.org/abs/2502.17817)  
**代码**: [GitHub](https://github.com/Kowsher/PredGen)  
**领域**: 其他  
**关键词**: PredGen, Generation for Prediction, Data Processing Inequality, Exposure Bias, Task Adapter  

## 一句话总结

本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。

## 研究背景与动机

### 问题背景
LLM在NLP任务中已展现强大能力，但在预测任务（分类、回归）中通常采用pooled表示（如[CLS] token或mean pooling）+ 分类头的方式。这种pooling操作会不可逆地丢弃位置信息和序列信息，限制模型捕捉细粒度依赖关系的能力。

### 核心动机
- LLM本身通过next-token prediction预训练，生成范式与其学习方式天然对齐
- Pooling操作是确定性压缩，根据数据处理不等式（DPI）必然导致信息损失
- 将预测任务重新表述为生成任务可以保留更多与目标相关的互信息
- 但直接用生成做预测面临两大挑战：exposure bias和格式不匹配

### 研究价值
提出了一个简洁而深刻的观点——"生成优于预测"，并用信息论严格证明，同时通过工程创新解决了实际问题。

## 方法详解

### 理论基础：为什么生成更好

**定理1（互信息在确定性压缩下不增加）**：设 $\mathbf{X}$ 为输入，$\mathbf{Z}$ 为隐藏表示，$\mathbf{Z_p} = g(\mathbf{Z})$ 为pooled表示（其中 $g$ 为确定性函数如first-token或mean pooling），$\mathbf{Y}$ 为目标，则：

$$I(\mathbf{Y}; \mathbf{Z}) \geq I(\mathbf{Y}; \mathbf{Z_p})$$

**证明核心**：基于数据处理不等式（DPI），确定性函数不增加互信息。由于 $\mathbf{Z_p}$ 由 $\mathbf{Z}$ 通过确定性过程导出，条件熵满足 $H(\mathbf{Y}|\mathbf{Z}) \leq H(\mathbf{Y}|\mathbf{Z_p})$，因而互信息 $I(\mathbf{Y}; \mathbf{Z_p}) \leq I(\mathbf{Y}; \mathbf{Z})$。

**经验验证**：使用MINE方法估计互信息，在多个数据集上验证PredGen一致保留更高互信息。token级互信息分析显示，预测token（如"positive"）与语义相关词（如"funny" 0.47, "pretty" 0.34）高度关联。

### PredGen框架

#### 1. 预测重述为生成
将预测任务的目标 $\mathbf{P}$（如13.4）表示为token序列 $\mathbf{Y} = ['1','3','.','4']$，模型自回归生成：
$$P(\mathbf{Y}|\mathbf{X};\theta) = \prod_{t=1}^m P(Y_t|\mathbf{X}, \mathbf{Y}_1, ..., \mathbf{Y}_{t-1}; \theta)$$

#### 2. Scheduled Sampling解决Exposure Bias
核心问题：训练时conditioned on真实token，推理时依赖自身生成，小错误会累积。

解决方案：以概率 $p$ 使用模型自身预测的token（而非ground-truth），$p$ 在训练过程中逐步增加：
$$\tilde{\mathbf{Y}} = \begin{cases} \mathbf{Y} & \text{with probability } (1-p) \\ \tilde{\mathbf{Y}} & \text{with probability } p \end{cases}$$

#### 3. Task Adapter解决格式不匹配
生成器产生离散token，但某些任务需要连续值或结构化输出。Task Adapter $\mathcal{T}$ 将生成token的隐藏表示转换为最终预测：
$$\hat{\mathbf{P}} = \mathcal{T}(\mathbf{Z}[n:n+m])$$
其中 $\mathbf{Z}[n:n+m]$ 是生成token对应的隐藏表示。

### 损失函数：Writer-Director Alignment Loss (WDAL)

采用"编剧-导演"类比：Writer（生成器）负责生成token，Director（task adapter）负责转换为任务格式。

$$L_{\text{WDAL}} = \max(L_W^2, L_D^2) \cdot \exp\left(-|\log L_W - \log L_D|\right)$$

- $L_W$：Writer Loss（交叉熵损失，衡量生成质量）
- $L_D$：Director Loss（任务特定损失，衡量预测精度）
- $\max(L_W^2, L_D^2)$：**权威项**，聚焦于误差更大的组件
- $\exp(-|\log L_W - \log L_D|)$：**对齐惩罚项**，确保两个损失保持平衡

当两个损失差异大时，惩罚项减小但max项增大，保持总损失高，驱使模型改善弱项。

## 实验

### 实验设置
- **模型**：Llama2-7B, Llama2-13B, Llama2-8B
- **PEFT方法**：LoRA, AdaLoRA, RoCoFT, DoRA
- **分类数据集**：BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA（指标：Accuracy）
- **回归数据集**：WASSA, SICK, STSB, LCP, CLEAR, Humicroedit（指标：MSE, MAE）

### 主要实验结果（分类任务，Llama2-7B + LoRA）

| 方法 | BoolQ | HellaSwag | WinoGrande | ARC-e | 平均 |
|------|-------|-----------|------------|-------|------|
| Predictor | 66.29 | 88.53 | 70.49 | 75.27 | 73.49 |
| Generator | 68.09 | 90.86 | 77.54 | 79.54 | 76.63 |
| **PredGen** | **73.82** | **93.14** | **83.21** | **84.79** | **79.67** |

关键发现：
1. **PredGen一致优于Predictor和Generator**：在所有模型×PEFT组合上，PredGen平均准确率提升约6个百分点（vs Predictor）
2. **Generator优于Predictor**：验证了生成优于分类的理论主张
3. **收益随模型增大而增加**：Llama2-13B上PredGen平均82.71%（vs Predictor 76.20%），提升更明显
4. **PEFT方法鲁棒**：不同PEFT方法下PredGen均保持优势

### 回归任务结果
PredGen在多个回归基准上也一致超越baseline，特别是在需要精确数值预测的任务上（如STS-B相似度评分），Task Adapter有效弥补了离散token与连续数值的鸿沟。

### 互信息验证
在SST-2、PIQA等数据集上的MINE估计显示：
- PredGen的隐藏表示与目标之间的互信息 > Generator > Predictor
- 验证了DPI理论预测

## 亮点与洞察

1. **理论优雅**：利用数据处理不等式严格证明了生成优于pooling的信息论基础，论证简洁有力
2. **WDAL损失设计巧妙**：Writer-Director类比直观，log-sum-exp稳定化技巧解决了数值不稳定问题
3. **端到端框架**：将scheduled sampling、task adapter、对齐损失无缝整合，不需要修改模型架构
4. **token级互信息可视化**：清晰展示了生成token如何捕获输入的语义依赖

## 局限性

1. 自回归生成增加了推理延迟（需要生成多个token而非单次前向传播）
2. Scheduled sampling的概率调度策略需要调参
3. 对于token数很多的数值（如长浮点数），生成准确性仍可能受限
4. 主要在Llama2系列验证，未扩展到更多架构（如encoder-only模型）

## 相关工作

- **LLM预测**：传统的分类头fine-tuning（BERT [CLS]）、in-context learning（GPT-3 few-shot）
- **生成式预测**：T5将所有任务统一为text-to-text、GPT系列的零样本推理
- **Exposure Bias**：Scheduled Sampling、Reward Augmented Maximum Likelihood
- **信息论与深度学习**：信息瓶颈理论、MINE互信息估计

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐⭐ |

> 这是一篇理论与实践结合得非常好的论文。核心insight——"生成保留更多信息，因此更适合预测"——通过DPI严格证明，并通过PredGen框架优雅地解决了实际挑战。WDAL损失函数设计新颖，Writer-Director类比让人印象深刻。在ACL2025论文中堪称上乘之作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ICML 2025\] Generation from Noisy Examples](../../ICML2025/others/generation_from_noisy_examples.md)

</div>

<!-- RELATED:END -->
