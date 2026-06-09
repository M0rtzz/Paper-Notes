---
title: >-
  [论文解读] On the Robustness of Reward Models for Language Model Alignment
description: >-
  [ICML 2025][LLM对齐][奖励模型鲁棒性] 提出 Batch-wise Sum-to-Zero Regularization (BSR)，通过约束每个 batch 内奖励分数之和为零来抑制隐状态范数的过度弥散，从根源上解决奖励模型的过优化问题…
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "奖励模型鲁棒性"
  - "过优化"
  - "Bradley-Terry模型"
  - "正则化"
  - "RLHF"
---

# On the Robustness of Reward Models for Language Model Alignment

**会议**: ICML 2025  
**arXiv**: [2505.07271](https://arxiv.org/abs/2505.07271)  
**代码**: [LinkedIn-XFACT/RM-Robustness](https://github.com/LinkedIn-XFACT/RM-Robustness)  
**领域**: LLM对齐/RLHF  
**关键词**: 奖励模型鲁棒性, 过优化, Bradley-Terry模型, 正则化, RLHF

## 一句话总结

提出 Batch-wise Sum-to-Zero Regularization (BSR)，通过约束每个 batch 内奖励分数之和为零来抑制隐状态范数的过度弥散，从根源上解决奖励模型的过优化问题，使 8B 规模 RM 在复杂偏好预测任务上超越 SOTA 5%+，并在 RLHF 下游训练中将生成长度降低 40% 同时提升 7% 胜率。

## 研究背景与动机

### 奖励模型过优化的普遍问题

奖励模型 (RM) 是 RLHF 管线中的关键组件，用于作为人类偏好的代理来对齐大语言模型。目前主流方法采用 Bradley-Terry (BT) 模型损失训练 RM，即最大化 chosen 和 rejected 回复之间的奖励差值。然而，已有大量工作指出 BT 模型训练的 RM 存在**过优化 (over-optimization)** 问题：

- RM 在训练集和同分布验证集上准确率持续提升
- 但在分布外 (OOD) 数据上的性能停滞甚至退化
- 这种退化会传播到 RLHF 下游训练，导致 policy 无法真正对齐真实人类偏好

### 现有方法的局限

以往的工作主要从数据质量角度缓解过优化（如高质量偏好数据集、更好的标注策略），或在 BT 模型上添加辅助组件（如 margin loss、label smoothing）。但这些方法：

1. 没有从根本上解决 BT 模型训练目标的固有缺陷
2. 使用 LLM 作为数据生成器/标注器会引入冗长偏差、自我增强偏差等固有问题
3. 缺乏对过优化在不同泛化场景下表现的系统性分析

### 本文动机

作者观察到一个关键现象：RM 的打分可以分解为 $r_\theta(x,y) = \|W_p\| \cdot \|h(x,y)\| \cdot \cos\psi$，其中 $W_p$ 是投影头，$h(x,y)$ 是最后一层隐状态。训练过程中 $\|W_p\|$ 基本维持在初始值 1 附近不变，但隐状态范数 $\|h(x,y)\|$ 的**方差会急剧膨胀**，这才是过优化的根本原因。

## 方法详解

### 整体框架

本文的方法分为三个层次：

1. **问题诊断**：提出四种泛化场景，系统分析 RM 过优化的表现
2. **原因分析**：从梯度动态角度证明隐状态范数弥散是过优化的主因
3. **解决方案**：提出 BSR 正则化，通过约束 batch 内奖励和为零来控制弥散

#### 四种泛化场景

作者将 RM 的泛化能力按照 prompt 和 response 两个维度的分布偏移进行分类：

| 场景 | Prompt 分布 | Response 分布 | 说明 |
|------|:---------:|:-----------:|------|
| In-domain ($\mathcal{D}_{ID}$) | 训练集内 | 训练集内 | 标准评估，无分布偏移 |
| Prompt-disjoint ($\mathcal{D}_{\sim\text{Prompt}}$) | **未见** | 训练集内 | 新 prompt + 同源 response 模型 |
| Response-disjoint ($\mathcal{D}_{\sim\text{Response}}$) | 训练集内 | **未见** | 同 prompt + 新 response 模型 |
| Mutual-disjoint ($\mathcal{D}_{\sim\text{Mutual}}$) | **未见** | **未见** | 最难场景，双重分布偏移 |

过优化的定义：在 $\mathcal{D}_{ID}$ 上准确率提升的同时，在后三种场景上性能停滞或退化。

### 关键设计

#### 1. 过优化的根因分析

RM 的奖励分数由三个因素的乘积决定：

$$r_\theta(x,y) = \|W_p\| \cdot \|h(x,y)\| \cdot \cos\psi$$

**投影头 $W_p$ 不是过优化的原因：**

由于 chosen 和 rejected 共享同一个投影头，$W_p$ 的梯度为：

$$\frac{\partial \mathcal{L}_{BT}}{\partial W_p} = -\sigma(-\Delta r) \cdot (h(x,y_w) - h(x,y_l))$$

其梯度范数为 $\sigma(-\Delta r) \cdot \|h(x,y_w) - h(x,y_l)\|$。训练初期 $\Delta r \approx 0$ 而隐状态差异小（因 LLM 隐状态有效秩低），故 $W_p$ 的更新幅度有限。实验验证训练后 $\|W_p\| \approx 1$，与初始值几乎无变化。

**隐状态范数弥散才是真正原因：**

BT 损失通过最大化 $\Delta r$ 来最小化，这驱动模型增大 $\|h(x,y_w) - h(x,y_l)\|$。具体而言：

$$\Delta r = \|W_p\| \cdot \|h(x,y_w) - h(x,y_l)\| \cdot \cos\psi$$

由于 $\|W_p\|$ 基本恒定，模型只能通过增大隐状态差异范数来增大 $\Delta r$。这导致：

- 隐状态范数的方差 $\text{Var}(\|h(x,y)\|)$ 在训练过程中持续膨胀
- 分布呈现右偏（right-skewed），出现极端大的范数异常值
- 这些异常值在 OOD 数据上会产生不可控的极端奖励分数

#### 2. Batch-wise Sum-to-Zero Regularization (BSR)

BSR 的核心思想：强制每个 batch 内所有样本的奖励分数之和趋近于零，从而惩罚极端奖励值、控制隐状态范数的弥散。

正则化项定义为：

$$\mathcal{L}_{BSR} = \left(\frac{1}{2B} \sum_{i=1}^{B} \left[r(x_i, y_{w,i}) + r(x_i, y_{l,i})\right]\right)^2$$

其中 $B$ 是 batch 中的样本对数，因子 $2B$ 对应 batch 中的总样本数（每对包含 chosen 和 rejected 各一个）。

**BSR 的作用机制：**

- 当 batch 内奖励分数的均值偏离零时，$\mathcal{L}_{BSR}$ 会施加二次惩罚
- 这迫使模型不能无限制地放大 chosen 的奖励或压低 rejected 的奖励
- 间接约束了隐状态范数不能过度弥散
- 零中心约束确保奖励分数在合理范围内波动

**BSR 的优势：**

- 不改变 BT 损失对偏好排序的学习能力（因为它只约束均值，不约束差值）
- 计算开销几乎可忽略（每个 batch 只需额外算一次均值和平方）
- 无需引入额外的超参数搜索维度（仅一个 $\lambda$）

### 损失函数 / 训练策略

最终的训练目标是标准 BT 损失加上 BSR 正则化：

$$\mathcal{L}_{BT\text{-}BSR} = \mathcal{L}_{BT} + \lambda \cdot \mathcal{L}_{BSR}$$

$$= -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma(\Delta r)\right] + \lambda \left(\frac{1}{2B} \sum_{i=1}^{B} \left[r(x_i, y_{w,i}) + r(x_i, y_{l,i})\right]\right)^2$$

**训练配置：**

- 基座模型：Llama-3 系列 (1B/3B/8B) 和 Qwen2.5 系列 (1.5B/3B/7B)
- 所有模型先在 UltraChat 上做 SFT，然后进行 RM 训练
- 训练数据：UltraFeedback（51,200 个样本），使用 17 个不同模型生成的 response
- 验证模型集 $\mathcal{M}_{valid}$：Gemma-2-2B-It、Olmo2-7B-Instruct、SmolLM2-1.7B-Instruct、Mistral-Instruct-v0.2（排除了 Llama 和 Qwen 家族以避免污染）
- Gold RM：ArmoRM（作为真实偏好模型 $r^*$）

**RLHF 实验配置：**

- 初始策略：Qwen2.5-1.5B + UltraChat SFT
- RM：Qwen2.5-3B (Part 2) / Llama-3.1-8B (Part 3)
- RL 算法：RLOO
- 高质量数据扩展：Skywork-Reward-Preference-80K-v0.2 + TULU3 SFT mixture

## 实验关键数据

### 主实验

**四种泛化场景下的 RM 准确率对比：**

| 模型 | 方法 | In-domain | Prompt-disjoint | Response-disjoint | Mutual-disjoint |
|------|------|:---------:|:---------------:|:-----------------:|:---------------:|
| Qwen2.5-3B | BT | 高 (基线) | 下降明显 | 下降明显 | 最差 |
| Qwen2.5-3B | BT-BSR | 略低 | **显著提升** | **显著提升** | **显著提升** |
| Llama-3.2-3B | BT | 高 (基线) | 下降明显 | 下降明显 | 最差 |
| Llama-3.2-3B | BT-BSR | 略低 | **显著提升** | **显著提升** | **显著提升** |

BSR 在所有四种泛化场景下均表现出更好的鲁棒性，尤其在 Mutual-disjoint（最严苛场景）下提升最大。

**8B 规模 RM-Bench 评估（复杂偏好预测）：**

| 配置 | RM-Bench 准确率 | 对比 BT 提升 | 说明 |
|------|:--------------:|:----------:|------|
| Llama-3.1-8B + BT | 基线 | — | 标准 BT 损失训练 |
| Llama-3.1-8B + BT-BSR | **+5%+** | +5%+ | BSR 正则化 |
| 此前 8B SOTA | 低于 BT-BSR | — | 被超越 |

### 消融实验

| 配置 | 隐状态范数方差 | OOD 准确率 | 说明 |
|------|:----------:|:--------:|------|
| 标准 BT | 持续膨胀 | 退化 | 基线，过优化严重 |
| BT + BSR | **稳定可控** | **提升** | BSR 有效抑制弥散 |
| $\|W_p\|$ 分析 | ≈1（不变） | — | 证明投影头不是过优化原因 |
| 隐状态差异范数 | 持续增长+右偏 | — | 证明隐状态范数弥散是根因 |

**RLHF 下游传播实验（AlpacaEval 2.0）：**

| RM 类型 | 生成长度变化 | 胜率变化 | 说明 |
|---------|:--------:|:------:|------|
| BT 训练的 RM | 基线 | 基线 | 存在冗长偏差 |
| BSR 训练的 RM | **↓40%** | **↑7%** | 鲁棒性传播到 RLHF |

### 关键发现

1. **隐状态范数弥散是过优化的根因**：BT 损失驱动模型增大 $\|h(x,y_w) - h(x,y_l)\|$，导致隐状态范数方差持续膨胀、分布右偏，产生极端奖励分数
2. **投影头 $W_p$ 不是过优化原因**：训练前后 $\|W_p\| \approx 1$，梯度受 sigmoid 衰减，更新幅度有限
3. **BSR 一致性**：在 Llama-3 和 Qwen2.5 两个模型家族、三种不同规模上均表现一致
4. **鲁棒性可传播**：RM 的鲁棒性会传播到 RLHF 训练，BSR-RM 训练出的 policy 生成更简洁、更高质量的回复
5. **可扩展性**：BSR 在 8B 规模模型 + 高质量数据上依然有效，超越 SOTA

## 亮点与洞察

- **诊断→治疗**范式：先用梯度分析定位过优化的根因（隐状态范数弥散），再针对性地设计正则化，思路清晰且有说服力
- **四种泛化场景**的分类为 RM 评估提供了一个更精细的框架，区分了 prompt 分布偏移、response 分布偏移及其组合
- BSR 的设计极其简洁：仅一个额外正则项，计算开销可忽略，但效果显著。这种"最小干预"的方法论值得借鉴
- RLHF 端到端实验令人印象深刻：生成长度降低 40% 说明 BSR-RM 有效抑制了冗长偏差，这是一个长期困扰 RLHF 的问题
- 隐状态范数弥散与 classification 场景中的 logit 范数过大问题有相似的机制（参考 Wei et al., 2022），跨领域的 insight 迁移

## 局限与展望

1. **BSR 仅约束 batch 内均值**：batch 内部的奖励分布可能仍然存在问题，更精细的约束（如方差约束）可能进一步提升效果
2. **$\lambda$ 的选择**：虽然只有一个超参数，但不同任务/模型规模的最优 $\lambda$ 可能不同，缺乏自适应调整机制
3. **Gold RM 的假设**：使用 ArmoRM 作为 $r^*$ 是一种近似，真实人类偏好的评估仍然缺失
4. **仅验证了 decoder-only 模型**：未在 encoder-only 或 encoder-decoder backbone 上验证
5. **与 DPO 等方法的结合**：BSR 目前仅与显式 RM 训练结合，是否能扩展到隐式 RM（如 DPO）值得探索
6. **动态正则化**：训练不同阶段的弥散程度不同，$\lambda$ 是否应该随训练动态调整

## 相关工作与启发

- **Gao et al. (2023)**：首次系统研究 RM 过优化，使用 gold RM 评估。本文在此基础上提出了更精细的四种泛化场景分类
- **Wei et al. (2022)**：发现 LLM 中范数增大导致 softmax 过度自信，本文将此 insight 迁移到 RM 场景
- **RLOO (Ahmadian et al., 2024)**：作为 RLHF 的 RL 算法，BSR 与其的结合展示了鲁棒 RM 对下游训练的正向传播
- **Skywork/TULU3 数据**：证明 BSR 在高质量数据上的互补效果，不与数据工程方法冲突
- **启发**：对于 DPO/ORPO 等方法，也可以分析其隐式奖励的范数弥散行为，或许能发现类似的过优化机制并设计对应的正则化

## 评分

- 新颖性: ★★★★☆ — 四种泛化场景分类和隐状态范数弥散分析有新意，BSR 设计简洁但不算革命性
- 理论深度: ★★★★☆ — 梯度分析严谨，从 $W_p$ 和 $h$ 两个方向排除法定位根因
- 实验充分度: ★★★★★ — 两个模型家族、三种规模、四种场景、RLHF 端到端验证、RM-Bench 评估
- 实用价值: ★★★★★ — BSR 实现极简、即插即用，对实际 RLHF pipeline 有直接帮助
- 写作质量: ★★★★☆ — 结构清晰，但数学符号较多，部分推导可以更简洁

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HAF-RM: A Hybrid Alignment Framework for Reward Model Training](../../ACL2025/llm_alignment/haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)
- [\[ICML 2025\] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](layer-wise_alignment_examining_safety_alignment_across_image_encoder_layers_in_v.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](../../ACL2025/llm_alignment/rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)
- [\[ICML 2025\] PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](poisonbench_assessing_large_language_model_vulnerability_to_data_poisoning.md)
- [\[ICML 2025\] Improving Model Alignment through Collective Intelligence of Open-Source LLMs](improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)

</div>

<!-- RELATED:END -->
