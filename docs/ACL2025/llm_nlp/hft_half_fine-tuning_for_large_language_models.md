---
title: >-
  [论文解读] HFT: Half Fine-Tuning for Large Language Models
description: >-
  [ACL 2025][LLM 其他][Half Fine-Tuning] 本文提出Half Fine-Tuning (HFT)，在每轮微调中随机冻结一半参数、只更新另一半，不改变模型架构的情况下显著缓解灾难性遗忘问题，同时在下游任务上取得与FFT相当甚至更好的性能，并减少约30%的训练时间。 LLM微调的遗忘困境：大模型经历…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "Half Fine-Tuning"
  - "灾难性遗忘"
  - "参数选择"
  - "持续学习"
  - "DPO"
---

# HFT: Half Fine-Tuning for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2404.18466](https://arxiv.org/abs/2404.18466)  
**领域**: LLM 微调 / 灾难性遗忘  
**关键词**: Half Fine-Tuning, 灾难性遗忘, 参数选择, 持续学习, DPO  

## 一句话总结

本文提出Half Fine-Tuning (HFT)，在每轮微调中随机冻结一半参数、只更新另一半，不改变模型架构的情况下显著缓解灾难性遗忘问题，同时在下游任务上取得与FFT相当甚至更好的性能，并减少约30%的训练时间。

## 研究背景与动机

**LLM微调的遗忘困境**：大模型经历预训练→SFT→DPO等多阶段训练后，虽然释放了强大的下游能力，但预训练阶段学到的参数化世界知识面临严重的灾难性遗忘风险。

**现有解决方案的不足**：
   - **额外模块方法**（如LoRA）：保持预训练参数不变，添加可学习模块，但修改了模型架构，给部署和后续微调带来障碍
   - **全参数微调（FFT）**：更新所有参数，新知识可能覆盖旧知识

**关键发现**（先导实验）：
   - 部分丢弃或裁剪task vector（微调后参数与预训练参数的差异）对目标任务影响较小（Yadav et al., 2023）
   - 这意味着：**部分新参数足以学习新能力**
   - 自然推论：**部分旧参数是否能维持预训练模型的能力？**

**Half-Reset实验**：将Llama 2-Chat-7b的50%参数重置回Llama 2-7b后，Half-Reset模型在基础知识上大幅恢复，同时保持了Chat版本出色的通用能力。这个发现直接启发了HFT。

## 方法详解

### 整体框架

HFT的核心思想极其简洁：每轮微调中，随机选择50%的参数进行更新，另外50%保持冻结。

$$\vartheta^t \leftarrow \vartheta^{t-1} - \eta \nabla_\vartheta \mathcal{L}(\theta^{t-1}), \quad \psi^t \leftarrow \psi^{t-1}$$

其中 $\vartheta^t$ 为待更新参数，$\psi^t$ 为冻结参数，$\theta^t = \{\vartheta^t, \psi^t\}$。

### 参数选择策略（Category-level）

以Llama 2架构为例，在每个Transformer层中：
- **自注意力**：从 $W_Q, W_K, W_V, W_O$ 四个矩阵中选两个更新
- **前馈网络**：从 $W_{up}, W_{down}, W_{gate}$ 三个矩阵中，一半层选两个、另一半层选一个，精确保证50%比例
- **LayerNorm**：同样选择一半
- **Embedding和LM_head**：默认更新

### 理论分析

HFT可等价于带正则化项的FFT优化问题：

$$\mathcal{O}_L = \min_\theta \max_\lambda \mathcal{L}(\theta) + \lambda \|(I-M)(\theta - \theta^0)\|^2$$

其中 $M$ 是参数掩码矩阵。通过Minimax不等式可推导出HFT优化的是FFT损失函数的上界加上一个正则化项 $\|(I-M)(\theta - \theta^0)\|^2$，这从理论上解释了为什么HFT有机会达到与FFT相当甚至更好的结果——正则化增加了稀疏微调模型的稳定性。

### 持续学习中的应用

在多轮微调场景下，每轮的冻结和更新参数集合不同（随机选择），这有利于在不同轮次间保持知识平衡，展现出显著的可扩展性。

## 实验关键数据

### SFT阶段：通用能力（Tülu V2）

| 模型 | MMLU | GSM8K | BBH | TyDiQA | TruthfulQA | HumanEval | 平均 |
|------|------|-------|-----|--------|------------|-----------|------|
| Llama2-7b (预训练) | 41.6 | 12.0 | 39.9 | 48.4 | 38.5 | 26.2 | 34.4 |
| Llama2-7b-SFT (FFT) | 48.5 | 25.0 | 42.2 | 51.2 | 41.7 | 36.9 | 41.0 |
| Llama2-7b-SFT (**HFT**) | **50.8** | **30.5** | **43.6** | **52.3** | **45.4** | 34.6 | **42.9 (+1.9)** |
| Llama2-13b-SFT (FFT) | 50.6 | 45.0 | 47.8 | 55.0 | 42.6 | 42.4 | 47.2 |
| Llama2-13b-SFT (**HFT**) | **54.5** | **46.5** | **53.7** | **56.7** | **45.7** | **43.5** | **50.1 (+2.9)** |

### SFT阶段：基础知识保留

| 模型 | NaturalQ | TriviaQA | HotpotQA | 平均 |
|------|----------|----------|----------|------|
| Llama2-7b (预训练) | 12.9 | 40.2 | 15.6 | 22.9 |
| Llama2-7b-SFT (FFT) | 3.2 | 26.4 | 14.5 | 14.7 |
| Llama2-7b-SFT (**HFT**) | **6.2** | **32.8** | **15.4** | **18.1 (+3.4)** |
| Llama2-13b-SFT (FFT) | 0.7 | 9.2 | 4.9 | 4.9 |
| Llama2-13b-SFT (**HFT**) | **2.7** | **12.4** | **8.2** | **7.8 (+2.9)** |

FFT导致基础知识断崖式下降，HFT显著缓解了遗忘。

### DPO阶段

| 模型 | 通用能力平均 | 基础知识平均 |
|------|------------|------------|
| Llama2-7b-DPO (FFT) | 41.9 | 10.7 |
| Llama2-7b-DPO (**HFT**) | 41.7 (-0.2) | **12.5 (+1.8)** |
| Llama2-13b-DPO (FFT) | 47.4 | 2.3 |
| Llama2-13b-DPO (**HFT**) | **48.2 (+0.8)** | **2.9 (+0.6)** |

DPO阶段HFT在通用能力上基本持平，基础知识上持续优于FFT。

### 持续学习（TRACE基准）

| 方法 | OP (FFT) | OP (HFT) | BWT (FFT) | BWT (HFT) |
|------|----------|----------|-----------|-----------|
| SeqFT (7b) | 45.7 | **51.3 (+5.6)** | -10.2% | **-5.6% (+4.6%)** |
| GEM (7b) | 48.2 | **50.2 (+2.0)** | -7.9% | **-5.9% (+2.0%)** |
| Replay (7b) | 54.3 | 54.1 (-0.2) | +1.4% | **+2.1% (+0.7%)** |
| SeqFT (13b) | 49.0 | **52.0 (+3.0)** | -9.4% | **-8.5% (+0.9%)** |
| GEM (13b) | 50.4 | **53.6 (+3.2)** | -8.9% | **-6.1% (+2.8%)** |
| Replay (13b) | 54.7 | **57.4 (+2.7)** | -0.6% | **+1.6% (+2.2%)** |

HFT作为即插即用方案，几乎所有持续学习方法都能从中受益。

### 参数选择策略对比（TRACE SeqFT）

| 策略 | OP | BWT |
|------|----|----|
| FFT | 45.7 | -10.2% |
| Model-level HFT | 46.9 (+1.2) | -9.2% |
| Layer-level HFT | 47.9 (+2.2) | -8.3% |
| **Category-level HFT** | **51.3 (+5.6)** | **-5.6%** |

Category-level选择（按参数类别细粒度选择）效果最佳，因为最大化了更新和冻结参数的交互。

### 训练效率

HFT在标准SFT中**减少约30%的训练时间**，因为冻结一半参数减少了梯度计算和优化器状态的开销。

## 亮点与洞察

1. **极简方法、显著效果**：仅需几行代码的mask操作，不改变架构，不增加参数，即可显著缓解遗忘
2. **从Half-Reset到HFT的洞见链**：先验证"重置一半参数可恢复旧知识"，再推导出"冻结一半参数可防止遗忘"，逻辑链完整
3. **理论解释清晰**：将参数冻结等价为正则化项，连接了稀疏微调理论
4. **鲁棒性强**：对参数选择策略和比例不敏感，约40-60%的更新比例都能获得良好效果
5. **通用适用性**：SFT、DPO、持续学习三个场景均有效，且可与GEM、Replay等方法叠加使用

## 局限性

1. **50%比例的启发性**：虽然实验验证了50%附近的效果最佳，但最优比例可能因任务和模型而异
2. **随机选择的方差**：每轮随机选择可能带来训练的不稳定性
3. **未考虑参数重要性**：对所有参数一视同仁，不考虑哪些参数更重要（如对知识存储更关键的参数）
4. **实验规模有限**：主要在Llama 2 7b/13b上实验，未验证在更大模型（70B+）上的效果
5. **与LoRA的比较不公平**：LoRA在TRACE上表现差，但其设计目标与HFT不同

## 相关工作

- **灾难性遗忘**：Lin et al. (2024) 分析微调对LLM知识的破坏；Neeman et al. (2023) 研究指令微调后的知识遗忘
- **参数高效微调**：LoRA (Hu et al., 2022) 通过低秩适配减少参数；Dou et al. (2023) 提出保持预训练参数不变加额外模块
- **Task Vector操作**：Ilharco et al. (2023) 定义task vector概念；TIES-Merging (Yadav et al., 2023) 部分裁剪task vector
- **持续学习**：GEM (Lopez-Paz and Ranzato, 2017)、Replay策略、TRACE基准 (Wang et al., 2023a)

## 评分

⭐⭐⭐⭐⭐ — 方法极度简洁却效果显著，理论解释清晰，实验覆盖全面（SFT/DPO/CL三场景），即插即用的特性使其具有极高的实用价值。"不改架构、不加参数、减少训练时间、缓解遗忘、提升性能"——罕见的多赢方案。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)

</div>

<!-- RELATED:END -->
