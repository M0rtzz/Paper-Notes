---
title: >-
  [论文解读] HAF-RM: A Hybrid Alignment Framework for Reward Model Training
description: >-
  [ACL 2025][LLM对齐][奖励模型训练] 提出混合对齐框架 HaF-RM，在奖励模型训练中保留策略层（policy layer），通过同时优化序列级奖励损失和 token 级策略损失来共同监督共享的内部偏好模型，在 5 个数据集上一致性超越标准 Baseline 和 DPO 方法。
tags:
  - ACL 2025
  - LLM对齐
  - 奖励模型训练
  - 混合对齐框架
  - 策略损失正则化
  - DPO
  - RLHF
  - Bradley-Terry模型
---

# HAF-RM: A Hybrid Alignment Framework for Reward Model Training

**会议**: ACL 2025  
**arXiv**: [2407.04185](https://arxiv.org/abs/2407.04185)  
**代码**: [https://haf-rm-anonymized.github.io](https://haf-rm-anonymized.github.io)  
**作者**: Shujun Liu, Xiaoyu Shen, Yuhang Lai, Siyuan Wang, Shengbin Yue, Zengfeng Huang, Xuanjing Huang, Zhongyu Wei
**机构**: Fudan University, Eastern Institute of Technology (Ningbo), University of Southern California
**领域**: LLM对齐 / 奖励模型  
**关键词**: 奖励模型训练, 混合对齐框架, 策略损失正则化, DPO, RLHF, Bradley-Terry模型

## 一句话总结

提出混合对齐框架 HaF-RM，在奖励模型训练中保留策略层（policy layer），通过同时优化序列级奖励损失和 token 级策略损失来共同监督共享的内部偏好模型，在 5 个数据集上一致性超越标准 Baseline 和 DPO 方法。

## 研究背景与动机

**领域现状**：奖励模型（Reward Model）是 LLM 对齐的核心组件，广泛用于 RLHF 训练、Best-of-N 采样、数据构建等场景。标准训练框架将奖励模型分为内部偏好模型（Transformer 主体）和奖励预测层（线性映射），使用 Bradley-Terry 偏好损失进行端到端训练。

**现有方法的不足**：
   - 多数奖励模型是闭源产业产品，限制了进一步训练和迁移
   - 训练数据中存在不正确和模糊的偏好标注
   - **标准训练范式可能导致内部偏好模型的监督不充分**——仅通过序列级奖励信号间接监督 Transformer 主体，token 级的偏好信息未被利用
   - DPO 虽然可以隐式产生奖励值，但对分布外数据的泛化能力极差

**核心动机**：奖励模型和策略模型在结构上高度相似——两者共享相同的 Transformer 主体（内部偏好模型），仅在输出层不同（奖励层 vs 策略层）。HaF 利用这一结构相似性，通过添加额外的策略损失在 token 级直接监督内部偏好模型，同时在序列级优化奖励映射层。

## 方法详解

### 整体框架

HaF 同时保留奖励层 F 和策略层 K（见 Figure 1），共享内部偏好模型 φ：
- 奖励模型输出：$\boldsymbol{r}(x, y) = F \circ \phi(x, y)$（序列级标量奖励）
- 策略模型输出：$\boldsymbol{\pi}(x, y) = K \circ \phi(x, y)$（token 级生成概率）

### 损失函数设计

**奖励损失**（序列级）：标准 Bradley-Terry 偏好损失

$$\mathcal{L}_s = \mathbb{E}_{(x,y,y') \sim \mathcal{D}} [-\log\sigma(\boldsymbol{r}(x,y) - \boldsymbol{r}(x,y'))]$$

**策略损失**（token 级）：使用 DPO 形式

$$\mathcal{L}_P = \mathbb{E}_{(x,y,y') \sim \mathcal{D}} [-\log\sigma(\tau(pd_{win} - pd_{lose}))]$$

其中 $pd_{win} = \log \frac{\boldsymbol{\pi}(x,y)}{\boldsymbol{\pi}_{ref}(x,y)}$ 为相对于参考策略的对数概率比，τ=0.1 为超参数。

**混合对齐损失**：

$$\mathcal{L}_H = \mathbb{E}_d [D_1(\boldsymbol{r}(d), \boldsymbol{r}^*(d)) + \alpha \cdot D_2(\boldsymbol{\pi}(d), \boldsymbol{\pi}^*(d))]$$

α 为平衡超参数，共享的内部偏好模型 φ 同时接收两个损失项的梯度。

### 为什么 HaF 更好？

论文提供了两个直觉性解释：

1. **Claim 1**：联合校准损失学到的模型优于仅使用标准奖励损失的模型——因为额外的策略约束限制了偏好空间的搜索范围，减少了过拟合
2. **Claim 2**：策略损失作为**正则化项**防止内部表示退化——标准奖励训练可能导致内部偏好模型的表示空间坍缩，而 token 级策略损失保持了表示的丰富性

**经验验证**（Figure 3）：共享参数的策略模型生成的内容被奖励模型评分更高，说明两者确实学到了相似的偏好。

### 模型结构

在标准 decoder-only LLM 基础上保留原始策略投影层（softmax 前的线性层），同时添加一个奖励投影层（输出标量），两者共享 Transformer 主体。

## 实验关键数据

### 数据集

5 个公开偏好数据集：
- HH-Harmless (12,915), HH-Helpful (13,543), Beaver Safe (47,625), Alpaca Human Pref (8,722), Chatbot Arena (19,466)

### 基础模型

- Phi-2-2.7B（全参数微调）
- Mistral-7B-base-v0.3（LoRA）
- Mistral-7B-Instruct-v0.2（LoRA）

### 主实验：内在偏好判断准确率

| 方法 | Helpful | Harmless | CA | BS | AHP | 平均 |
|------|---------|----------|-----|-----|-----|------|
| DPO (Mistral) | 74.29 | 70.30 | 81.90 | **92.70** | 60.30 | 75.90 |
| Baseline (Mistral) | 76.20 | 72.70 | 79.80 | 80.80 | 56.30 | 73.16 |
| **HaF (Mistral)** | 75.80 | **73.10** | 81.90 | 88.70 | **63.10** | **76.52** |
| DPO (Phi-2) | 69.70 | 66.30 | 66.80 | 87.80 | 52.60 | 68.64 |
| Baseline (Phi-2) | 64.30 | 69.50 | 79.30 | 76.00 | 58.40 | 69.50 |
| **HaF (Phi-2)** | **76.40** | **70.40** | 79.00 | 84.00 | **60.80** | **74.12** |

**关键发现**：
- HaF 在所有三个骨干模型上的平均准确率一致超越 Baseline 和 DPO
- DPO 在 BS 数据集上表现最好（数据分布集中），但其他数据集不稳定
- DPO 和 Baseline 学到的特征不同，HaF 有效整合了两者

### 混合数据训练

将 5 个数据集均匀混合后训练：
- HaF 在所有模型上取得最佳整体性能，说明更擅长学习混合偏好分布的多样性
- DPO 在混合数据上的 CA 和 Helpful 性能显著下降——倾向于拟合数据分布的主要特征

### 分布外（OOD）泛化

将数据集分为 Safety 类（BS, Harmless）和 Chat 类（AHP, CA, Helpful），跨类别评估：

**内部测试**（同类别不同数据集）：
- HaF (Mistral) 平均 70.30%，比 Baseline 高 9.07%，比 DPO 高 2.26%

**外部测试**（RewardBench）：
- HaF (Mistral) 平均 81.95%，比 Baseline 高 28.47%，比 DPO 高 7.80%
- DPO 的 OOD 测试结果几乎都在 50% 附近——完全失去建模能力

### 下游任务：Best-of-N 采样

使用各奖励模型从生成模型的多个候选中选择最佳响应：
- HaF 的 GPT-4 评判胜率最高（Phi-2: 52.0% vs Baseline 27.4%; Mistral: 51.1%）
- HaF 选择的 Top-1 响应与 GPT-4 排名的一致性也最高（Phi-2: 33.77%, Mistral: 18.19%）

### 下游任务：RLHF

使用各奖励模型进行 PPO 训练：
- HaF 奖励模型训练出的策略在 GPT-4 评判中多数场景下胜率最高
- DPO 奖励模型训练的 RLHF 策略表现不稳定

## 亮点与洞察

1. **结构相似性的巧妙利用**：奖励模型和策略模型本质上共享内部偏好模型的洞察是整篇论文的核心，混合监督的想法简洁优雅
2. **DPO 的OOD脆弱性**：实验清晰揭示了 DPO 作为奖励模型在分布外场景下几乎完全失效的问题（准确率接近 50%），这是由其对语言风格的强偏好导致的
3. **正则化视角**：策略损失作为正则化防止内部表示退化的解释具有说服力，且得到了实验支持
4. **实验全面**：从内在评估到 OOD 泛化、再到 Best-of-N 和 RLHF 的外在评估，覆盖了奖励模型的主要应用场景

## 局限性

1. **骨干模型规模有限**：最大只到 Mistral-7B，未在更大模型（如 70B 级别）上验证
2. **策略损失选择**：仅使用 DPO 作为策略损失的实现，未充分探索其他策略优化方法（如 KTO、IPO 等）
3. **超参数 α 的敏感性**：混合损失中的平衡系数可能需要针对不同数据集/模型调整
4. **理论分析较弱**：Claim 1 和 Claim 2 更多是直觉性解释，缺乏严格的理论证明
5. **BS 数据集上 DPO 仍占优**：HaF 在分布集中的数据上未能完全超越 DPO，说明混合框架在某些场景下可能引入噪声

## 相关工作

- **RLHF 与奖励模型**：Christiano et al. (2017), Ouyang et al. (2022) 建立了标准训练框架
- **DPO**：Rafailov et al. (2023) 的直接偏好优化将策略模型隐式转化为奖励模型；本文揭示了其 OOD 局限性
- **数据增强方向**：Li et al. (2023a), Wu et al. (2023) 等从数据角度改进奖励模型，与本文的训练框架改进互补
- **细粒度奖励信号**：Cao et al. (2024), Lai et al. (2024) 利用细粒度信号改进奖励模型；HaF 的 token 级策略损失可视为一种互补方法
- **Bradley-Terry 模型**：标准偏好建模框架，将奖励差转化为概率进行优化

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：混合监督的想法虽然简单但有效，利用结构相似性的视角新颖
- **实验充分性** ⭐⭐⭐⭐⭐：5 个数据集 × 3 个骨干模型 × 混合训练 + OOD + Best-of-N + RLHF 的全面评估
- **理论深度** ⭐⭐⭐：直觉性解释有说服力但缺乏严格证明
- **实用价值** ⭐⭐⭐⭐：改进简单、即插即用，对奖励模型训练实践有直接参考价值

<!-- RELATED:START -->

## 相关论文

- [On the Robustness of Reward Models for Language Model Alignment](../../ICML2025/llm_alignment/on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [MPO: Multilingual Safety Alignment via Reward Gap Optimization](mpo_multilingual_safety_alignment.md)
- [Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)

<!-- RELATED:END -->
