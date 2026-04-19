---
title: >-
  [论文解读] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning
description: >-
  [NeurIPS 2025][模型压缩][持续学习] 提出 DEAL 框架，通过小波核特征过滤保留 LoRA 低秩矩阵中的历史知识核心特征，结合受控知识更新模块和非对称正则化，实现 LLM 在小样本持续微调中学新不忘旧。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 持续学习
  - LoRA
  - 小波核
  - 知识保留
  - 参数高效微调
---

# Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2509.18942](https://arxiv.org/abs/2509.18942)  
**代码**: [GitHub](https://github.com/zzm-black/DEAL-Continuous-Low-Rank-Fine-Tuning)  
**领域**: model_compression  
**关键词**: 持续学习, LoRA, 小波核, 知识保留, 参数高效微调

## 一句话总结

提出 DEAL 框架，通过小波核特征过滤保留 LoRA 低秩矩阵中的历史知识核心特征，结合受控知识更新模块和非对称正则化，实现 LLM 在小样本持续微调中学新不忘旧。

## 研究背景与动机

LoRA 微调的 LLM 需要通过持续学习来集成新知识。然而传统持续学习方法面临两个核心挑战：

**灾难性遗忘**：在新任务上微调后丧失旧任务能力

**数据效率低下**：小规模领域数据难以充分适配

现有解决方案的问题：
- **直接编辑**（ROME/MEMIT）：需额外实验定位神经元，成本高
- **堆叠适配器**：增加推理开销
- **正交子空间约束**（O-LoRA）：限制了跨任务的有益迁移

核心问题：能否设计一种方法，在**不增加推理延迟**的前提下，用少量新数据持续微调 LoRA，同时保持所有历史任务的性能？

## 方法详解

### 整体框架

DEAL 由两个核心模块组成：
1. **小波核知识保留模块**：从 LoRA 低秩矩阵中提取和过滤奇异值，保留历史知识的核心表征
2. **受控知识更新模块**：通过高阶正则化约束参数更新，调节新知识集成

推理阶段：更新后的低秩矩阵直接替换原 LoRA 模块，**推理延迟不变**。

### 关键设计

**小波核特征过滤**：假设低秩矩阵 $\mathbf{Y}$（即 $\mathbf{A}$ 或 $\mathbf{B}$）可分解为任务相关核心成分 $\mathbf{X}$ 和冗余/噪声成分 $\mathbf{D}$：$\mathbf{Y} = \mathbf{X} + \mathbf{D}$。

在白噪声假设（$\mathbf{D}^\top \mathbf{D} = \sigma_D^2 I$, $\mathbf{X}^\top \mathbf{D} = 0$）下，核心特征的最小方差估计为：

$$\hat{\mathbf{X}} = \sum_{k=1}^{r_x} \frac{\sigma_k^2 - \sigma_D^2}{\sigma_k} \mathbf{u}_k \mathbf{v}_k^\top$$

由于 $\sigma_D^2$ 未知，使用热核作为低通滤波器定义多尺度小波网络：

$$\phi_{\sigma_j^2, c_j}(\mathbf{X}) = \exp\left(-\frac{1}{2\sigma_j^2}\|\mathbf{X} - c_j\|^2\right)$$

小波神经网络提取核心特征：

$$\mathbf{H}_{:,i}^{k+1} = \delta\left(\sum_j \phi_{\sigma_j^2, c_j} \mathbf{g}_j \phi_{-\sigma_j^2, c_j} \mathbf{H}_{:,j}^k\right)$$

其中 $\mathbf{g}_j$ 为可学习对角矩阵，$c_j$ 为可学习中心。

**受控知识更新**：通过 MLP 将新知识叠加到核心特征上，生成更新后的低秩矩阵 $\mathbf{A}'$ 或 $\mathbf{B}'$。

### 损失函数

$$\mathcal{L} = \text{MSE}(\mathcal{A}_{\mathbf{W}, \Delta\mathbf{W}'}(\mathbf{Q}), \mathbf{G}) + \lambda_1 \|\boldsymbol{\theta}_1\|_a^a + \lambda_2 \|\boldsymbol{\theta}_2\|_b^b$$

其中 $\boldsymbol{\theta}_1 = \{\mathbf{g}, \mathbf{C}\}$ 是保留模块参数，$\boldsymbol{\theta}_2 = \{\boldsymbol{\Omega}, \mathbf{B}\}$ 是更新模块参数。关键约束：$a \geq b$，确保保留模块的正则化强度不低于更新模块，**最小化对核心特征的扰动**。

## 实验关键数据

### 主实验

**持续学习性能对比**（Average Accuracy / ROUGE-1）：

| 方法 | 3-Task TC | 4-Task Standard | 15-Task Large |
|------|----------|----------------|--------------|
| | AA / R-1 | AA / R-1 | AA / R-1 |
| T5 + SeqLoRA | 52.4 / 52.8 | 44.6 / 44.6 | 42.1 / 44.0 |
| T5 + O-LoRA | 85.2 / 87.1 | 71.2 / 73.3 | 70.8 / 80.3 |
| T5 + PerTaskFT (oracle) | 90.3 / 91.7 | 70.0 / 73.0 | 76.5 / 78.2 |
| **T5 + DEAL** | **87.7 / 89.3** | **78.5 / 82.5** | **73.9 / 79.1** |
| LLaMA + SeqLoRA | 54.1 / 55.9 | 47.6 / 54.8 | 45.2 / 53.2 |
| LLaMA + O-LoRA | 86.4 / 88.1 | 75.3 / 80.8 | 73.2 / 77.4 |
| LLaMA + PerTaskFT | 88.2 / 90.0 | 77.5 / 79.4 | 77.1 / 82.5 |
| **LLaMA + DEAL** | **88.9 / 90.2** | **78.9 / 81.3** | **74.6 / 78.9** |

在 4-task 上 DEAL 比 O-LoRA 提升 7.3 pp AA（T5），比 SeqLoRA 提升 33.9 pp，接近 oracle PerTaskFT。

### 消融实验

**适配器更新策略**：联合更新 A+B 达到最高 AA（75.6%），优于单独更新 A（72.8%）或 B（70.2%）

**LoRA 秩敏感度**：

| 秩 | AA (%) |
|----|--------|
| 4 | 71.5 |
| 8 | 84.3 |
| 16 | 84.5 |
| 32 | 84.6 |

rank 8 已捕获大部分任务特定变化，继续增大收益递减。

**正则化权重**（$(a,b)$ 网格搜索）：

| $a$ | $b$ | AA (%) |
|-----|-----|--------|
| 1 | 1 | 74.8 |
| 5 | 1 | 83.9 |
| **10** | **2** | **85.5** |
| 10 | 5 | 84.1 |
| 20 | 2 | 82.7 |

**任务顺序鲁棒性**：3 种随机排列的 AA 范围仅 73.1%-75.6%，波动 < 3 pp。

### 关键发现

1. DEAL 在 15-task 大规模基准上超越 SeqLoRA 29 pp 以上，显示出强扩展性
2. 非对称正则化（$a > b$）是关键：保留模块需更强约束以保护历史知识
3. 推理延迟完全不变——更新后的低秩矩阵直接替换原模块

## 亮点与洞察

1. **小波核用于 LoRA 特征过滤**是新颖且合理的设计——基于 SVD 分频的信号处理思想
2. 推理时间不变是实际部署的关键优势，避免了堆叠适配器的推理开销问题
3. 非对称正则化设计有明确的直觉：保护历史知识应比允许新知识更新有更高优先级
4. 在 LLaMA-3.1-8B 上验证了对大模型的适用性

## 局限性

- 假设固定任务顺序和静态模型容量，未处理模糊任务边界
- 白噪声假设（$\mathbf{D}^\top \mathbf{D} = \sigma_D^2 I$）在实际低秩矩阵中可能不严格成立
- 小波核引入额外训练参数，虽然推理时消失但训练阶段有开销
- 未与 replay-based 方法对比（仅对比无记忆方法）

## 相关工作与启发

- 与 O-LoRA（正交子空间约束）互补：O-LoRA 限制干扰但也限制了跨任务迁移，DEAL 通过特征过滤保留核心特征同时允许非核心部分更新
- 与 CLoRA（角度正则化）相比，DEAL 的小波核特征过滤更精细
- 启发思考：小波核方法能否用于其他 PEFT 模块（如 Adapter、Prefix-tuning）的持续学习

## 评分

- ⭐ 新颖性: 4/5 — 小波核特征过滤用于 LoRA 持续学习是新颖的技术路线
- ⭐ 实验充分度: 4/5 — 15 个数据集 + 两个模型骨干 + 完整消融
- ⭐ 写作质量: 3/5 — 公式推导详尽但部分符号超载，可读性一般
- ⭐ 价值: 4/5 — 解决了 LoRA 持续学习中的实际痛点，推理零开销

<!-- RELATED:START -->

## 相关论文

- [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)

<!-- RELATED:END -->
