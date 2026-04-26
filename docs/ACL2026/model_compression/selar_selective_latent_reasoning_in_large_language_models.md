---
title: >-
  [论文解读] SeLaR: Selective Latent Reasoning in Large Language Models
description: >-
  [ACL 2026][模型压缩][潜在推理] 本文提出 SeLaR，一种轻量级无训练框架，通过熵门控机制仅在模型不确定的"探索步"激活软嵌入潜在推理、在高置信的"确定步"保持离散解码，并引入熵感知对比正则化防止软嵌入向主导 token 坍缩，在五个推理基准上一致超越标准 CoT 和 SOTA 无训练方法。
tags:
  - ACL 2026
  - 模型压缩
  - 潜在推理
  - 熵门控
  - 软嵌入
  - 对比正则化
  - 无训练推理增强
---

# SeLaR: Selective Latent Reasoning in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.08299](https://arxiv.org/abs/2604.08299)  
**代码**: [GitHub](https://github.com/Parker-rfu/SeLaReasoning)  
**领域**: LLM 推理 / 模型效率  
**关键词**: 潜在推理, 熵门控, 软嵌入, 对比正则化, 无训练推理增强

## 一句话总结

本文提出 SeLaR，一种轻量级无训练框架，通过熵门控机制仅在模型不确定的"探索步"激活软嵌入潜在推理、在高置信的"确定步"保持离散解码，并引入熵感知对比正则化防止软嵌入向主导 token 坍缩，在五个推理基准上一致超越标准 CoT 和 SOTA 无训练方法。

## 研究背景与动机

**领域现状**：思维链（CoT）已成为 LLM 多步推理的主流范式，通过显式生成中间推理步骤提升复杂任务表现。近期的潜在推理方法尝试用软嵌入或隐状态替代离散 token 采样，以在单次前向传播中隐式探索多条推理路径。

**现有痛点**：(1) 标准 CoT 在每步必须承诺单个离散 token，丢弃了关于替代推理路径的分布信息；(2) 训练型潜在推理方法（如 Coconut）因隐状态与嵌入空间的域差异导致灾难性遗忘；(3) 无训练型方法（如 Soft Thinking）全局激活软嵌入，在模型已经高置信的步骤引入不必要的扰动，破坏推理稳定性。

**核心矛盾**：CoT 解码过程中模型的熵分布呈现清晰的长尾结构——大多数步骤是低熵的确定步，只有少量步骤是高熵的探索步。全局激活忽略了这种长尾结构，在确定步引入扰动而在探索步又因软嵌入向主导 token 坍缩而失去多路径探索能力。

**本文目标**：解决两个问题——何时激活潜在推理（选择性激活）以及如何维持有效探索（防止坍缩）。

**切入角度**：利用 token 级预测分布的熵作为置信度信号，将解码步骤分为确定步和探索步，只在关键探索步启用潜在推理。

**核心 idea**：熵门控选择性激活 + 熵感知对比正则化——前者决定"何时"用潜在推理，后者解决"如何"在激活后维持多路径探索。

## 方法详解

### 整体框架

SeLaR 在解码的每一步：(1) 计算 top-k token 的归一化熵 $\bar{H}_t$；(2) 若 $\bar{H}_t \leq \tau$（确定步），使用标准离散解码；(3) 若 $\bar{H}_t > \tau$（探索步），构建 top-k 候选的概率加权软嵌入并应用对比正则化，将正则化后的软嵌入作为下一步输入。整个过程无需训练，即插即用。

### 关键设计

1. **熵门控选择性激活机制**:

    - 功能：仅在模型不确定时启用潜在推理，确定步保持标准解码
    - 核心思路：计算 top-k token 的截断熵 $H_t = -\sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \log \hat{p}_t(v)$，归一化为 $\bar{H}_t = H_t / \log k$。若 $\bar{H}_t \leq \tau$ 则使用采样/贪心的离散 token 嵌入，否则使用概率加权的软嵌入 $e_t = \sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \cdot e_v$。阈值 $\tau$ 位于熵分布的低密度过渡带，在 $[0.3, 0.7]$ 范围内稳定
    - 设计动机：实验表明只有少量步骤是高熵的探索步，在确定步引入软嵌入是净负面影响——去掉选择性激活导致平均准确率下降 5.19%

2. **熵感知对比正则化**:

    - 功能：防止软嵌入在潜在推理过程中向主导 token 方向坍缩
    - 核心思路：计算软嵌入与主导 token 嵌入的差向量 $\Delta_t = e_t - e_{v_t^*}$，归一化后用熵加权推离主导方向：$\tilde{e}_t = e_t + \bar{H}_t \cdot \hat{\Delta}_t \cdot \|\Delta_t\|$。熵越高推离力度越大，模型变得自信时效果自然减弱
    - 设计动机：先前工作发现软嵌入会快速被最高概率 token 主导，退化为贪心解码。对比正则化通过 logit lens 分析验证：未施加时 top-1 overlap 在深层主导，施加后 top-1 和 top-2 overlap 保持可比，说明多条推理轨迹共存

3. **Top-k 截断熵估计**:

    - 功能：高效精确地估计模型的决策不确定性
    - 核心思路：仅在 top-k token（而非全词表）上计算熵，先重新归一化概率再计算。这捕获了模型在最可能候选之间的不确定性，同时避免低概率 token 的干扰
    - 设计动机：全词表熵计算开销大且受长尾噪声影响，top-k 截断既高效又聚焦于决策相关的不确定性

### 损失函数 / 训练策略

SeLaR 完全无训练。使用 Qwen3-1.7B/8B/32B 和 DeepSeek-R1-Distill-Llama-8B 评估。解码设置：temperature=0.6, top-p=0.95, top-k=20, min-p=0.0。

## 实验关键数据

### 主实验

**五个推理基准上的准确率对比（Qwen3-8B）**

| 方法 | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|------|-------|---------|------|--------|--------|-----|
| CoT (Sampling) | 95.45 | 98.00 | 61.62 | 76.67 | 66.67 | 79.68 |
| Soft Thinking | 94.92 | 95.80 | 57.58 | 70.00 | 66.67 | 76.99 |
| SwiR | 95.68 | 97.00 | 62.63 | 60.00 | 66.67 | 76.40 |
| **SeLaR** | **95.83** | **97.00** | **61.62** | **83.33** | **80.00** | **83.56** |

### 消融实验

**组件消融（Qwen3-8B）**

| 配置 | Avg | 说明 |
|------|-----|------|
| Full SeLaR | 83.56 | 完整模型 |
| w/o 选择性激活 | 78.37 | 全局激活掉 5.19% |
| w/o 对比正则化 | 75.74 | 无防坍缩掉 7.82% |

### 关键发现

- SeLaR 在所有模型规模上一致超越 CoT，Qwen3-8B 上平均提升 +3.88%，且是唯一在所有模型上一致超越的方法
- 在最难的 AIME 基准上提升最显著：AIME24 +6.66%、AIME25 +13.33%（Qwen3-8B）
- 对比正则化贡献最大（去掉后掉 7.82%），尤其在 AIME24/25 上从 83.33/80.00 降至 70.00/60.00
- 计算效率：SeLaR 在 AIME24 上 TPCA 比 CoT 减少 19.2%，而 SwiR 反而增加 33.2%
- Logit lens 分析证实：对比正则化使 top-1 和 top-2 的 overlap 保持可比，维持了真正的多路径探索

## 亮点与洞察

- 长尾熵分布的观察是全文的基石——大多数步骤模型已经很确定，潜在推理只在少数关键步骤有价值
- 对比正则化的设计优雅：用熵本身作为推离强度的权重，在探索步强力推离、在接近确定步时自然消退
- Logit lens 分析提供了机制性证据，而非仅依赖消融实验——直接可视化了多轨迹共存与否

## 局限与展望

- 阈值 $\tau$ 虽然在 $[0.3, 0.7]$ 范围内稳定，但仍是数据集特定的超参数，未实现完全自适应
- 在知识密集型任务（GPQA）上效果有限，因为领域知识召回比多步推理更关键
- 仅在推理型 LLM 上评估，未验证在通用指令遵循或代码生成任务上的效果
- 对比正则化的方向选择（仅推离 top-1）可能不够——top-2、top-3 也可能是需要推离的坍缩方向

## 相关工作与启发

- **vs Soft Thinking (Zhang et al., 2025)**: 后者全局激活软嵌入，SeLaR 选择性激活——去掉选择性后掉 5.19% 验证了全局激活的危害
- **vs SwiR (Shi et al., 2025)**: 后者基于相邻步熵变化触发切换，易受伪触发影响需窗口平滑；SeLaR 直接用绝对熵阈值，更简洁稳定
- **vs Coconut (Hao et al., 2025)**: 后者需微调传播隐状态，存在灾难性遗忘；SeLaR 完全无训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 选择性激活 + 对比正则化的组合新颖且动机清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 5 基准 × 4 模型 × 详细消融 + logit lens 机制分析
- 写作质量: ⭐⭐⭐⭐ 从观察到方法到分析的逻辑链条完整
- 价值: ⭐⭐⭐⭐ 无训练即插即用，实用价值高

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

<!-- RELATED:END -->
