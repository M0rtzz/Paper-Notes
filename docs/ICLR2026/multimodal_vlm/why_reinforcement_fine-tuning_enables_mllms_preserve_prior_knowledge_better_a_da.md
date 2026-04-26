---
title: >-
  [论文解读] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective
description: >-
  [多模态] 通过拼图任务系统研究 SFT 与 RFT 对先验知识的影响，揭示 RFT 避免灾难性遗忘的核心在于**数据分布**而非算法差异——RFT 采样的数据天然与基模型概率景观对齐，干扰更小。
tags:
  - 多模态
---

# Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2506.23508](https://arxiv.org/abs/2506.23508)
- **代码**: 未公开
- **领域**: 多模态大模型 / 强化微调 / 灾难性遗忘
- **关键词**: RFT, SFT, 灾难性遗忘, 学习动力学, GRPO, 拼图任务

## 一句话总结

通过拼图任务系统研究 SFT 与 RFT 对先验知识的影响，揭示 RFT 避免灾难性遗忘的核心在于**数据分布**而非算法差异——RFT 采样的数据天然与基模型概率景观对齐，干扰更小。

## 研究背景与动机

SFT 和 RFT 是大模型后训练的两种主要方法，但它们对先验知识保留的影响尚不明确。现有研究主要关注下游任务性能提升，忽视了微调对预训练知识的影响。

关键观察：
- SFT 能快速学习新任务但导致**灾难性遗忘**
- RFT 学习更慢但能**更好地保持先验知识**
- 原因尚不清楚：是算法差异还是数据差异？

本文引入**拼图任务**作为全新任务（现有 MLLM 包括 GPT-4o 准确率均为 0%），系统研究这一问题。

## 方法详解

### 实验设计

**拼图任务**：将原始图像切分为 $3 \times 3$ 网格，随机打乱顺序，模型需输出正确的位置索引序列以重建图像。选择拼图任务的原因：所有 SOTA MLLM（包括 GPT-4o、Qwen-2.5-VL-72B）准确率均为 0%，确保任务确实是"全新的"。

### RFT 奖励设计

采用 GRPO 算法，定义三种规则奖励：

$$R = R_{\text{hit}} + R_{\text{acc}} + R_{\text{fmt}}$$

- **命中奖励**：$R_{\text{hit}} = \frac{\#\text{correct indices}}{m \times n}$，衡量部分正确性
- **准确度奖励**：$R_{\text{acc}} \in \{0, 1\}$，全部正确才得 1
- **格式奖励**：$R_{\text{fmt}} \in \{0, 1\}$，输出格式需满足 `<think>...</think><answer>...</answer>` 结构

### SFT 数据格式

两种数据格式对比：
- **Non-Reasoning (Non-Rea)**：直接提供答案，无推理过程
- **Reasoning (Rea-4o-Rollout)**：GPT-4o 生成的推理轨迹 + 答案
- **GRPO-Rollout**：RFT 训练后模型自身生成的正确 rollout

### 学习动力学分析框架

核心问题：训练样本 $x_u$ 如何影响先验知识样本 $x_v$ 的概率？

基于 empirical Neural Tangent Kernel (eNTK) 分析：

$$\Delta \log \pi^t(x_v)|_{x_u} \approx \eta \cdot \underbrace{\nabla_\theta \log \pi_{\theta^t}(x_v)^\top \nabla_\theta \log \pi_{\theta^t}(x_u)}_{\text{eNTK}(x_u, x_v)}$$

分解为两个因素：
- **幅度 (Magnitude)**：eNTK 的范数大小，反映干扰强度
- **方向 (Direction)**：梯度对齐程度，反映干扰方向（正向增强 or 负向遗忘）

## 实验

### 新任务学习能力（Qwen2.5-VL-3B / 7B）

| 方法 | 训练步数 | 3×3 拼图准确率 |
|------|---------|---------------|
| Base | - | 0% |
| RFT (GRPO) | 27,360 | 66% / 75% |
| SFT-Non-Rea | 200 / 400 | 53% / 80% |
| SFT-Rea-4o-Rollout | 4,100 | 70% / 78% |
| SFT-Rea-GRPO-Rollout | 2,670 / 3,000 | 70% / 81% |

### 先验知识保留（Qwen2.5-VL-7B，性能变化）

| 基准 | RFT | SFT-Non-Rea | SFT-Rea-4o | SFT-GRPO-Roll |
|------|-----|-------------|------------|----------------|
| RefCOCO_val | ↓0.6 | **↓57.2** | **↓37.5** | ↓8.6 |
| DocVQA | ±0.0 | **↓27.4** | ↓2.3 | ↓0.9 |
| MME | ↓8 | **↓1854** | ↓249 | ↓126 |
| MMStar | ↑1.7 | **↓62.8** | ↓3.7 | ↓2.4 |
| POPE | ↓0.2 | **↓69.9** | **↓12.1** | ↓3.1 |

### 关键发现

1. **RFT 可以从零学会全新任务**：经过充分探索（27k 步），准确率从 0% 提升到 66-75%
2. **SFT 学得快但遗忘严重**：仅需 200-400 步即达到 RFT 水平，但 Grounding 能力暴跌 57.2%
3. **数据是关键，非算法**：用 RFT 模型生成的 rollout 训练 SFT（SFT-Rea-GRPO-Rollout），同样能保留先验知识，遗忘远小于标准 SFT
4. **推理轨迹有助于减轻遗忘**：带推理的数据（Rea-4o 和 GRPO-Rollout）遗忘明显小于 Non-Rea
5. **eNTK 幅度分析**：RFT rollout 数据的 eNTK 范数更小，对先验知识干扰更弱
6. **eNTK 方向分析**：RFT rollout 位于基模型已有中等概率的区域，梯度方向与先验知识更对齐

### 数学/科学 QA 验证

在 Qwen2.5 上的数学和科学 QA 任务中观察到一致的遗忘和学习动力学趋势，验证了结论的泛化性。

## 亮点

- 巧妙选择拼图任务作为"全新知识"的试金石，所有 MLLM 零基线
- 首次从数据视角解释 RFT vs SFT 的遗忘差异，提出可验证的因果链
- 学习动力学分析框架（幅度 + 方向分解）提供了理论深度
- SFT 在 GRPO-Rollout 上训练的实验是关键证据，干净地分离了算法与数据的作用
- 对"RFT 不能习得新能力"的观点提出反证

## 局限性

- 拼图任务虽然新颖，但相对简单且输出格式固定，泛化到更复杂任务的程度未验证
- 仅在 Qwen2.5-VL 3B/7B 上实验，未验证更大规模模型
- eNTK 分析基于近似，在大规模模型中的精确性可能受限
- 未提出新的抗遗忘算法，主要是分析和解释性工作
- GRPO-Rollout 的生成依赖于先完成 RFT 训练，增加了总体成本

## 相关工作

- **拼图任务**：Noroozi & Favaro (自监督学习)；Lyu et al. (MLLM 弱点探测)；Jigsaw-R1 (RFT 解拼图)
- **MLLM 强化微调**：DeepSeek-R1、Meng et al.（OOD 泛化）、感知任务 RFT
- **灾难性遗忘缓解**：EWC（正则化）、经验回放（数据混合）、架构方法 — 均不适用于大规模 MLLM
- **RL's Razor**：Shenfeld et al. 认为 RL 隐式偏向 KL 最小解，本文从数据视角给出互补解释

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从数据分布角度解释遗忘差异的新视角
- **技术深度**: ⭐⭐⭐⭐⭐ — 学习动力学分析严谨，理论与实验紧密结合
- **实验充分度**: ⭐⭐⭐⭐ — 多模型多任务验证，消融设计精巧
- **实用价值**: ⭐⭐⭐⭐ — 对后训练策略选择有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)

<!-- RELATED:END -->
