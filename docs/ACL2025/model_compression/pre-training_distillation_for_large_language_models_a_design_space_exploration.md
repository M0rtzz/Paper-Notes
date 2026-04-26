---
title: >-
  [论文解读] Pre-training Distillation for Large Language Models: A Design Space Exploration
description: >-
  [ACL 2025][模型压缩][pre-training distillation] 系统性地探索大语言模型预训练蒸馏（Pre-training Distillation）的设计空间，从 logits 处理、损失函数选择、scaling law 和 offline/online logits 四个维度进行广泛实验，找到更优配置并得出有价值的结论。
tags:
  - ACL 2025
  - 模型压缩
  - pre-training distillation
  - 知识蒸馏
  - logits processing
  - scaling law
  - LLM
---

# Pre-training Distillation for Large Language Models: A Design Space Exploration

**会议**: ACL 2025  
**arXiv**: [2410.16215](https://arxiv.org/abs/2410.16215)  
**代码**: —  
**领域**: 模型压缩 / 知识蒸馏 / 大语言模型预训练  
**关键词**: pre-training distillation, knowledge distillation, logits processing, scaling law, LLM  

## 一句话总结

系统性地探索大语言模型预训练蒸馏（Pre-training Distillation）的设计空间，从 logits 处理、损失函数选择、scaling law 和 offline/online logits 四个维度进行广泛实验，找到更优配置并得出有价值的结论。

## 研究背景与动机

### 问题背景
知识蒸馏（KD）是将大模型知识迁移到小模型的标准方法。在 LLM 时代，KD 主要应用于**后训练阶段**（post-training），学生模型直接从教师生成的指令-响应对上学习。但将 KD 扩展到**预训练阶段**的探索非常有限。

### 核心动机
- 教师模型的 logits 包含比 one-hot 标签更丰富的信息，可作为标签平滑，加速训练并提升性能
- 预训练蒸馏（PD）虽然直觉上有优势，但缺乏系统性研究来指导如何更好地实施
- 本文旨在填补这一空白，通过大量受控实验探索预训练蒸馏的设计空间

### 基本公式
$$\theta_S^* = \arg\min_{\theta_S} [(1-\alpha)\mathcal{L}_{\text{lm}} + \alpha\mathcal{L}_{\text{kd}}]$$

其中 $\mathcal{L}_{\text{lm}}$ 是传统语言建模损失，$\mathcal{L}_{\text{kd}}$ 是蒸馏损失，$\alpha$ 控制二者比例。

## 方法详解

### 设计空间四个维度

**维度一：Logits 处理**
- **截断方法**：提出 **top-p-k 两阶段截断**——先 top-p 截断（分布尖锐时有效），再 top-k 截断（分布均匀时二次截断）
    - GLM-4-9B 词表约 150K，全量存储需 58.6 PB，top-0.95-100 截断后减为约 **15 TB**（4000× 压缩）
- **温度归一化**：$F(\mathbf{z}) = \text{softmax}(\text{Truncate}(\mathbf{z}) / \tau)$

**维度二：损失函数选择**
- 蒸馏损失 $L$ 的选择：NLL（负对数似然）、KLD（KL 散度）、MSE
- $\alpha$ 的调度策略：静态、线性递增/递减、周期性、WSD 调度

**维度三：Scaling Law**
- 学生模型大小：330M、670M、1.9B、3.8B、6.8B
- 教师模型大小：9B、32B
- 预训练语料大小：100B、500B tokens

**维度四：Offline vs. Online Logits**
- Offline：使用预训练好的教师模型生成 logits
- Online：在教师模型预训练过程中同步存储 logits

### 初步实验（Preliminary）
- 教师模型：GLM-4-9B
- 学生模型：1.9B
- 训练数据：100B tokens
- 初步配置下 PD 带来平均 **1.6%** 的性能提升

## 实验

### 维度一：Logits 处理实验结果

**top-p-k 截断**：
- 不同 p 值（top-p-100）性能差异不大，**更小的 p 可进一步减少存储**
- 不同 k 值（top-0.95-k），k=50 效果最佳；k=1 也有提升（相当于用教师标签做 LM 训练，隐式噪声过滤）

**温度**：

| τ | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|------|-----|-----|-----|-----|-----|-----|------|
| 提升(%) | 1.6 | 2.1 | 2.5 | **2.7** | 1.6 | 2.5 | -0.1 | 1.0 |

- τ≤2.0 效果相当，τ≥5.0 时改善有限（过于均匀的分布不利于学生学习）
- 自适应温度（AdaKDH）最佳（+2.8%），但相比静态最优无显著额外增益

### 维度二：损失函数实验结果

**蒸馏损失对比**：

| 方法 | 平均提升 |
|------|---------|
| NLL | +1.6% |
| KLD | **+2.6%** |
| MSE | **-7.6%** |

- KLD 和 NLL 均有效，MSE 显著下降——与 CV 中 MSE 最优的结论相反
- **最佳 α 调度**：WSD-α + WSD-LR，达到 **+8.0%** 提升
    - 核心洞察：在保持最大学习率阶段使用更多 KD 损失可显著提升性能
    - 线性递减 α（先 KD 后 LM）优于线性递增（先 LM 后 KD）

### 维度三：Scaling Law 实验结果

**模型大小**：
- **更大的学生模型受益更多**：6.8B > 3.8B > 1.9B > 670M > 330M
- **更大的教师模型不一定更好**：9B 教师在某些学生上优于 32B 教师，可能因为容量差距（capacity gap）
- PD 在学生模型大小 ≥ 教师模型的 ~10% 时有效

**语料大小**（500B tokens 实验）：
- PD 在整个预训练过程中持续带来改善
- 增益先增后趋于稳定，最终仍显著——PD 不仅提升训练效率，还提升性能上界

### 维度四：Offline vs. Online 实验结果

| 方法 | 平均提升 |
|------|---------|
| LLM-Online-100B-L（教师早期 logits） | -20.9% |
| LLM-Online-100B（教师后期 logits） | -3.9% |
| LLM-Online-100B*（α=0.1，调优） | +0.5% |

- 即使是未收敛教师的 logits 也能略微帮助学生训练
- Online logits 效果不如 offline，但优势是**无需额外推理开销**
- 建议：若需训练一系列大小不同的 LLM，可在最大模型预训练时存储 online logits

## 亮点与洞察

1. **系统性设计空间探索**：首次对 LLM 预训练蒸馏的四个关键维度进行全面受控实验
2. **WSD-α + WSD-LR 组合** 带来最大收益（+8.0%），这是一个关键的实践指导
3. **MSE 损失在 LLM 蒸馏中失效**，与 CV 中的经验相反，暗示 LLM 预训练蒸馏有独特的训练动态
4. **"容量差距"效应**：更大教师不一定更好，为实际部署中的教师选择提供依据
5. **Online logits 的可行性**：首次验证了在教师预训练期间同步存储 logits 用于后续蒸馏的有效性
6. **存储效率**：top-p-k 截断实现 4000× 压缩，使大规模 logits 存储变得可行

## 局限性

1. 未探索不同因素之间的交互作用（组合实验代价过高）
2. 未达到万亿级 token 的预训练规模（当前最先进 LLM 使用的规模）
3. 未探索学生模型超过教师的 weak-to-strong 场景
4. 实验计算成本极高，碳排放是潜在的伦理问题

## 相关工作

- **Pre-ChatGPT 时代蒸馏**：DistilBERT、TinyBERT 等基于百万级参数模型，配置不直接适用于十亿级 LLM
- **LLM 蒸馏**：Gemma 2、AFM、LokiLM、Minitron 等采用了预训练蒸馏但细节有限
- **后训练蒸馏**：Alpaca、Vicuna 等从 GPT-4 响应中学习
- **MiniLLM**：在预训练 LLM 基础上蒸馏，非从头训练

## 评分

⭐⭐⭐⭐ — 大量实验极具参考价值，结论清晰实用（特别是 WSD 调度、MSE 失效、容量差距等发现）。作为设计空间探索系统性强，但缺乏跨维度组合与更大规模验证。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)

<!-- RELATED:END -->
