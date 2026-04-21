---
title: >-
  [论文解读] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks
description: >-
  [ICML 2025][AI安全][backdoor attack] 首次提出"双重学习假说"揭示 ICL 后门攻击的理论机制，并设计 ICLShield 防御方法，通过动态添加高置信度和高相似度的干净示例来调节概念偏好比，平均降低攻击成功率 26.02%。
tags:
  - ICML 2025
  - AI安全
  - backdoor attack
  - 上下文学习
  - 大语言模型安全
  - 潜在概念
  - 防御机制
---

# ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks

**会议**: ICML 2025  
**arXiv**: [2507.01321](https://arxiv.org/abs/2507.01321)  
**代码**: 无  
**领域**: AI安全  
**关键词**: backdoor attack, 上下文学习, 大语言模型安全, 潜在概念, 防御机制

## 一句话总结

首次提出"双重学习假说"揭示 ICL 后门攻击的理论机制，并设计 ICLShield 防御方法，通过动态添加高置信度和高相似度的干净示例来调节概念偏好比，平均降低攻击成功率 26.02%。

## 研究背景与动机

上下文学习（In-Context Learning, ICL）允许 LLM 通过少量示例完成新任务而无需修改参数，已被广泛应用于文本分类、推理和生成等任务。然而，ICL 引入了一个关键的安全漏洞：攻击者只需在 ICL 示例中注入少量含有触发器的中毒样本，即可操纵模型在推理阶段输出恶意结果。这种攻击（ICL 后门攻击）不需要修改任何训练数据或模型参数，因此适用于任何模型，甚至包括 GPT-3.5 和 GPT-4 等 API 服务。

当前 ICL 后门攻击研究面临两个核心问题：

**机制理解缺乏**：现有研究（ICLAttack、BadChain 等）主要验证攻击有效性，但未揭示攻击如何影响模型输出的内在机理

**防御方法空白**：传统后门防御（Anti-Backdoor Learning、ONION、Back-Translation 等）主要针对数据投毒或模型投毒，对不修改模型参数的 ICL 攻击效果有限

## 方法详解

### 整体框架

ICLShield 建立在对 ICL 后门攻击机制的理论分析之上，整体框架分为两个部分：

1. **理论分析**：提出双重学习假说（Dual-Learning Hypothesis），推导攻击成功概率的上界，揭示概念偏好比（Concept Preference Ratio）是决定攻击效果的核心因素
2. **防御设计**：基于理论分析，通过动态添加精选的干净示例来提高概念偏好比，从而降低攻击成功率

### 关键设计

#### 1. 双重学习假说

基于 Xie et al. (2021) 和 Wang et al. (2024) 的潜在概念理论，作者提出：当 LLM 接收到被中毒的 ICL 示例时，会同时学习两个离散的潜在概念：

- **任务潜在概念 $\theta_1$**：编码正常任务信息（如情感分类目标）
- **攻击潜在概念 $\theta_2$**：编码后门攻击信息（如"含触发器时输出负面"）

模型输出概率被建模为：

$$P_M(\mathbf{y} | \mathcal{S}_t, \mathbf{x}) = P_M(\mathbf{y} | \mathbf{x}, \theta_1) P_M(\theta_1 | \mathcal{S}_t, \mathbf{x}) + P_M(\mathbf{y} | \mathbf{x}, \theta_2) P_M(\theta_2 | \mathcal{S}_t, \mathbf{x})$$

其中 $P_M(\theta_i | \mathcal{S}_t, \mathbf{x})$ 为后验分布，反映模型从输入中激活各概念的程度。

#### 2. 攻击成功概率上界

通过 Jensen 不等式和独立性假设，作者推导出攻击成功概率的上界：

$$\tilde{P}_M(\mathbf{y}_t | \mathcal{S}_t, \hat{\mathbf{x}}) \leq \frac{1}{\frac{P_M(\theta_1 | \mathcal{S}_t)}{P_M(\theta_2 | \mathcal{S}_t)} + 1}$$

这表明攻击成功率由**概念偏好比** $\frac{P_M(\theta_1|\mathcal{S}_t)}{P_M(\theta_2|\mathcal{S}_t)}$ 主导——该比值越高，攻击成功率上界越低。

#### 3. 概念偏好比的分解

通过贝叶斯定理进一步分析，概念偏好比与三个因子呈正相关：

$$\frac{P_M(\theta_1|\mathcal{S}_t)}{P_M(\theta_2|\mathcal{S}_t)} \propto \underbrace{\frac{P_M(\theta_1)}{P_M(\theta_2)}}_{\text{任务先验权重}} \cdot \underbrace{\left(\frac{P_M(\mathbf{y}_t|\hat{\mathbf{x}},\theta_1)}{P_M(\mathbf{y}_t|\hat{\mathbf{x}},\theta_2)}\right)^m}_{\text{中毒影响因子}} \cdot \underbrace{\left(\frac{P_M(\mathbf{y}_{gt}|\mathbf{x},\theta_1)}{P_M(\mathbf{y}_{gt}|\mathbf{x},\theta_2)}\right)^n}_{\text{干净影响因子}}$$

由于任务先验权重和中毒影响因子由任务和攻击场景决定不可控，防御只能通过**调节干净影响因子**来实现。

#### 4. ICLShield 防御策略

基于理论分析的三个关键观察，设计防御示例集 $\mathcal{S}_d = \mathcal{S}_d^s + \mathcal{S}_d^c$，包含两种选择策略：

**相似度选择（Similarity Selection）**：选取与中毒示例语义相似度最高的 $k/2$ 个干净样本。直觉是当干净样本含有与触发器相似的内容时，攻击潜在概念被激活但标签指向正确答案，从而降低攻击概率。使用 LLM 嵌入的余弦相似度计算：

$$\mathcal{S}_d^s = \arg\text{top}_{(\mathbf{x}_i, \mathbf{y}_i) \subseteq \mathcal{D}}^{k/2} \cos(\mathbf{e}(\mathbf{x}_i), \mathbf{e}(\mathcal{S}_t))$$

**置信度选择（Confidence Selection）**：选取模型在中毒示例条件下预测置信度最高的 $k/2$ 个干净样本。高置信度意味着任务潜在概念被更强烈地激活：

$$\mathcal{S}_d^c = \arg\text{top}_{(\mathbf{x}_i, \mathbf{y}_i) \subseteq \mathcal{D}}^{k/2} P_M(\mathbf{y}_i | \mathbf{x}_i, \mathcal{S}_t)$$

最终将防御示例集与中毒示例拼接，送入 LLM 进行推理。

### 损失函数 / 训练策略

ICLShield 是一个**推理阶段的防御方法**，无需额外训练或微调。核心流程：
1. 检测到可能被中毒的 ICL 示例集 $\mathcal{S}_t$
2. 从干净数据集 $\mathcal{D}$ 中通过相似度选择和置信度选择各挑选 $k/2$ 个样本
3. 将选中的干净样本附加到 ICL 示例中，形成增强后的示例集
4. 使用增强后的示例集进行推理，概念偏好比被提升，攻击成功率降低

## 实验关键数据

### 主实验

实验覆盖 11 个开源 LLM、2 种攻击方法（ICLAttack 和 BadChain）、3 类任务（分类、生成、推理）。

| 模型 | 方法 | SST-2 CA↑ | SST-2 ASR↓ | AG's News CA↑ | AG's News ASR↓ |
|------|------|-----------|------------|---------------|----------------|
| GPT-NEO-1.3B | No Defense | 78.25 | 92.19 | 69.30 | 94.80 |
| GPT-NEO-1.3B | ONION | 71.66 | 98.13 | 70.00 | 46.53 |
| GPT-NEO-1.3B | Back-Translation | 78.47 | 82.30 | 68.80 | 43.50 |
| GPT-NEO-1.3B | **ICLShield** | **77.32** | **35.97** | **58.60** | **15.24** |
| GPT-J-6B | No Defense | 89.84 | 71.73 | 75.00 | 26.28 |
| GPT-J-6B | **ICLShield** | **83.14** | **19.58** | **69.50** | **6.83** |
| GPT-NEOX-20B | No Defense | 90.01 | 99.45 | 69.10 | 20.37 |
| GPT-NEOX-20B | **ICLShield** | **85.78** | **38.39** | **51.90** | **9.29** |

不同模型架构上的 ICLAttack 防御结果（SST-2 分类任务）：

| 方法 | OPT-6.7B ASR↓ | MPT-7B ASR↓ | LLaMA2-7B ASR↓ | LLaMA3-8B ASR↓ |
|------|---------------|-------------|-----------------|-----------------|
| No Defense | 99.78 | 99.45 | 93.26 | 47.63 |
| ONION | 100.00 | 99.01 | 84.52 | 60.07 |
| Back-Translation | 85.26 | 94.72 | 66.56 | 41.80 |
| **ICLShield** | **30.36** | **46.53** | **33.11** | **17.16** |

闭源模型上的防御效果：

| 方法 | GPT-3.5 SST-2 ASR↓ | GPT-4o SST-2 ASR↓ | GPT-3.5 AG's News ASR↓ | GPT-4o AG's News ASR↓ |
|------|---------------------|---------------------|------------------------|------------------------|
| ICLAttack | 6.86 | 7.16 | 5.58 | 11.78 |
| **ICLShield** | **3.67** | **3.88** | **0.29** | **2.34** |

### 消融实验

在 GPT-NEO-1.3B + SST-2 上的选择策略消融（ICLAttack 攻击下）：

| 配置 | ASR↓ | 说明 |
|------|------|------|
| No Defense | 92.19 | 无防御基线 |
| 随机选择 | ~83.0 | 随机添加干净示例，ASR 降低约 9.20% |
| 仅相似度选择 | ~48.3 | 验证观察❷有效，ASR 降约 43.89% |
| 仅置信度选择 | ~53.7 | 验证观察❸有效，ASR 降约 38.50% |
| **ICLShield（两者结合）** | **~35.97** | 组合效果最佳，额外降低 7.92%~2.53% |

防御示例数量 $k$ 的影响：添加 6 个干净示例时达到最佳平衡（ASR 降低 51.05%），继续增加示例数量（7 个）时 ASR 降低放缓。

### 关键发现

1. **ONION 和 Back-Translation 对 ICL 后门几乎无效**：平均仅降低 ASR 3.47% 和 3.06%，而 ICLShield 降低 29.14%，约为前者的 10 倍
2. **跨模型泛化性极强**：从 1.3B 到 66B 参数量、从 GPT-NEO 到 LLaMA 到 OPT，ICLShield 始终保持最佳防御效果
3. **可迁移至闭源模型**：在开源模型上选择的防御示例可直接迁移到 GPT-3.5/GPT-4o，AG's News 上 ASR 降低达 84.85%
4. **对 BadChain 推理攻击同样有效**：GSM8K 上 ASR 从 92.31% 降至 39.58%，CSQA 上从 26.01% 降至 5.89%

## 亮点与洞察

1. **理论贡献突出**：首次从潜在概念角度建立 ICL 后门攻击的理论框架，双重学习假说将复杂的攻击-防御问题转化为一个可量化的概念偏好比调节问题
2. **防御设计优雅**：不需要识别或消除触发器（这对 ICL 攻击极其困难），而是通过"以量制毒"——用精选的干净示例稀释中毒示例的影响
3. **实用性强**：纯推理阶段防御，不修改模型参数，适用于任何 LLM（包括 API 服务），可跨模型迁移
4. **理论与实验高度一致**：理论预测的攻击成功概率分布偏移在实验中得到了直接验证

## 局限与展望

1. **更复杂的提示工程**：尚未在 Tree-of-Thought、Graph-of-Thought 等复杂 ICL 范式下验证
2. **更具挑战性的应用领域**：医疗、金融等高风险场景下的效果有待探索
3. **大模型上的 CA 下降**：OPT-66B 上 CA 从 87.64% 降至 68.59%，表明大模型上干净准确率的损失可能更明显
4. **防御示例的来源假设**：需要获取与目标任务相关的干净数据集，这一假设在实际场景中可能不够鲁棒
5. **自适应攻击**：未讨论攻击者知道防御机制时的自适应攻击场景

## 相关工作与启发

- **ICLAttack (Zhao et al., 2024)**：通过在 ICL 示例中嵌入触发器实施后门攻击，是本文主要的攻击基线
- **BadChain (Xiang et al., 2024)**：通过在 CoT 推理步骤中插入后门推理实现攻击，是另一重要攻击方法
- **ONION (Qi et al., 2020)**：基于困惑度过滤的词级后门防御，对 ICL 后门效果差
- **Wang et al. (2024a)**：LLM 潜在变量模型理论，是双重学习假说的理论基础
- **启发**：该工作提示我们 LLM 安全不仅要关注训练/微调阶段，推理阶段的 ICL 同样是重要的攻击面；防御不一定需要检测攻击，稀释攻击效果也是可行的策略

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 5 | 首次系统分析 ICL 后门机制并提出防御 |
| 理论深度 | 5 | 严谨的数学推导，假说-定理-证明体系完整 |
| 实验充分性 | 5 | 11 个开源 + 2 个闭源模型，3 类任务，消融全面 |
| 实用性 | 4 | 推理阶段即插即用，但需干净数据集 |
| 写作质量 | 4 | 结构清晰，理论-动机-方法-实验逻辑连贯 |
| **总分** | **4.6** | 理论和实验均为该方向的开创性工作 |

<!-- RELATED:START -->

## 相关论文

- [Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)
- [The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](../../CVPR2025/ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/ai_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)

<!-- RELATED:END -->
