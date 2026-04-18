---
title: >-
  [论文解读] CTPD: Cross Tokenizer Preference Distillation
description: >-
  [AAAI 2026][目标检测][知识蒸馏] 提出 Cross-Tokenizer Preference Distillation (CTPD)，首个支持不同分词器间偏好蒸馏的统一框架，通过 Aligned Span Projection、跨分词器重要性加权和 Teacher-Anchored Reference 三项创新，在多个 benchmark 上显著超越现有方法。
tags:
  - AAAI 2026
  - 目标检测
  - 知识蒸馏
  - 偏好对齐
  - 跨分词器
  - DPO
  - 语言模型
---

# CTPD: Cross Tokenizer Preference Distillation

**会议**: AAAI 2026  
**arXiv**: [2601.11865](https://arxiv.org/abs/2601.11865)  
**代码**: [有](https://github.com/dinhtruongng/CTPD)  
**领域**: 目标检测  
**关键词**: 知识蒸馏, 偏好对齐, 跨分词器, DPO, 语言模型

## 一句话总结

提出 Cross-Tokenizer Preference Distillation (CTPD)，首个支持不同分词器间偏好蒸馏的统一框架，通过 Aligned Span Projection、跨分词器重要性加权和 Teacher-Anchored Reference 三项创新，在多个 benchmark 上显著超越现有方法。

## 研究背景与动机

将 LLM 与人类偏好对齐已成为核心研究课题。DPO 等方法在大模型上效果显著，但小模型受限于表示能力难以直接对齐。知识蒸馏（KD）是一条有前景的路线——先用大模型完成昂贵的对齐过程，再将对齐行为迁移到小模型。

然而，白盒蒸馏面临**跨分词器问题**：teacher 和 student 模型通常使用不同的 tokenizer，导致 logit 分布不兼容，无法直接进行 token 级知识迁移。已有的跨分词器蒸馏工作（ULD、DSKD、Multi-Level OT）仅针对预训练/微调场景设计，**不适用于偏好对齐任务**。目前仅有一篇工作研究了偏好蒸馏，但限于 teacher-student 共享同一 tokenizer 的简化场景。

CTPD 的核心洞察：尽管 teacher 和 student 的 token 在语法上不同，但两者最终编码的是**同一段自然语言子串**。通过字符级对齐，可以在异构分词器间建立精确的对应关系。

## 方法详解

### 整体框架

CTPD 框架分三个阶段：

1. **SFT 阶段**：teacher 和 student 分别在指令微调数据上进行 SFT
2. **Teacher 对比模型训练**：用 DPO 训练 teacher 的正向模型 $\pi^+$ 和反向模型 $\pi^-$，用于后续重要性权重估计
3. **CTPD 偏好蒸馏**：student 使用偏好数据和预计算的对齐 span 权重，以 SFT teacher 作为 reference 进行偏好训练

### 关键设计

**1. Aligned Span Projection（对齐 Span 投影）**

核心机制：将 teacher 和 student 的 token 序列分割为一系列**对齐 span**——每对 span 对应原始字符串中完全相同的字符区间。

定义：teacher token 子序列 $\{t_i, ..., t_j\}$ 和 student token 子序列 $\{s_k, ..., s_l\}$ 构成 aligned span，当且仅当它们解码的字符覆盖原始字符串 $S$ 中完全相同的起止位置。

这一机制允许将 teacher 的任何信号（log-概率、重要性权重等）在 span 内聚合后投影到 student 的对应 token 上，**不引入任何可学习参数**。

**2. Cross-Tokenizer Importance Weighting（跨分词器重要性加权）**

扩展 TIS-DPO 到跨分词器场景。将 token 级奖励分解为 span 级奖励：

$$r(p^t | x, p^{<t}) = \sum_i r(y_{t_i} | x, y_{t_{<i}})$$

基于 Theorem 1（span 级标签噪声）证明：span 级奖励的显著波动是偏好数据中标签噪声的指标。通过重要性采样将理想无噪声分布 $\mathcal{D}^*$ 下的期望转化为真实分布 $\mathcal{D}$ 下的加权期望：

$$w_t = k \cdot \exp\left(\mu \cdot \text{clamp}\left(\log\frac{\pi^+(p^t | x, p^{<t})}{\pi^-(p^t | x, p^{<t})}, L, U\right)\right)$$

利用 teacher 模型的 DPO 正/反模型来估计权重，使 teacher 的精细奖励判断蒸馏到 student。

**3. Teacher-Anchored Reference（教师锚定参考）**

传统 DPO 使用 student 自身（SFT checkpoint）作为 reference model $\pi_\text{ref}$。CTPD 创新性地使用**teacher 模型**作为 reference。

理论依据：DPO 梯度中 reference model 通过权重 $\lambda$ 来调控训练过程。更强的 reference model 能提供更好的样本加权，引导 policy 模型朝正确方向优化。通过 aligned span projection 机制，student 可以在自己的 token 空间中近似计算 teacher 的 log-概率，从而实现异构分词器下的 teacher-anchored DPO 目标。

### 损失函数 / 训练策略

最终 CTPD 损失函数：

$$\mathcal{L}_\text{CTPD} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta(r(x, y_w) - r(x, y_l))\right)\right]$$

其中 $r(x, y) = \sum_{i=1}^T w_i \log \frac{\pi_\theta(p_i | x, p_{<i})}{\pi_\text{ref}(p_i | x, p_{<i})}$，$\pi_\text{ref}$ 为 teacher 模型。

训练超参数：
- 数据：UltraFeedback Binarized（63k+ 偏好对）
- SFT 阶段：lr = $4 \times 10^{-6}$
- Teacher DPO：lr = $2 \times 10^{-6}$，$\beta = 0.3$
- CTPD：lr = $1 \times 10^{-6}$，$\beta = 0.1$
- 权重裁剪范围 $[L, U] = [-0.5, 1.5]$
- 8× NVIDIA H100-80GB，全局 batch size 16

## 实验关键数据

### 主实验

**表 1：Qwen-2.5-14B → Llama-3.1-8B 蒸馏结果**

| 方法 | HellaSwag | ARC | MMLU | TruthfulQA | Winogrande | GSM8k | Average |
|------|-----------|-----|------|------------|------------|-------|---------|
| Teacher | 84.34 | 67.06 | 79.74 | 58.51 | 80.58 | 84.23 | 75.74 |
| Student | 81.99 | 57.59 | 65.48 | 45.19 | 77.43 | 50.27 | 62.99 |
| DPO | 82.42 | 60.84 | 65.26 | 52.16 | 78.31 | 54.87 | 65.64 |
| TIS-DPO | 81.08 | 61.92 | 66.73 | 53.86 | 79.05 | 54.31 | 66.16 |
| DSKD | 79.24 | 58.19 | 64.82 | 51.77 | 74.82 | 50.11 | 63.16 |
| **CTPD** | 82.25 | **63.92** | 66.65 | **55.22** | **79.29** | **57.47** | **67.42** |

**表 2：Qwen-2.5-7B → Llama-3.2-1B 蒸馏结果**

| 方法 | HellaSwag | ARC | TruthfulQA | Winogrande | GSM8k | Average |
|------|-----------|-----|------------|------------|-------|---------|
| Student | 65.59 | 39.33 | 37.66 | 62.75 | 6.82 | 40.67 |
| TIS-DPO | 66.23 | 40.92 | 43.49 | 64.34 | 9.13 | 42.60 |
| **CTPD** | **67.30** | 40.61 | **46.34** | **64.50** | **9.72** | **43.26** |

### 消融实验

**权重估计策略对比（Llama-3.1-8B）**：

| 策略 | Average |
|------|---------|
| CTPD (Origin) | **67.42** |
| Average weight | 65.47 |
| Student estimate | 65.88 |
| Teacher-student estimate | 64.51 |
| Random weight | 54.80 |

**Reference 模型选择**：使用 student 作为 reference 时平均分降至 65.27，远低于使用 teacher 的 67.42，验证了 teacher-anchored reference 的有效性。

### 关键发现

- CTPD 平均分比 TIS-DPO 高 +1.26（14B→8B）和 +0.66（7B→1B）
- 在需要推理和事实精度的 GSM8k（+3.16）和 TruthfulQA（+2.85）上提升最显著
- 传统 KD 方法（DSKD、ULD、Multi-Level OT）在偏好蒸馏场景下反而不如直接 DPO
- 随机权重导致严重性能退化（67.42→54.80），证明权重估计的关键性

## 亮点与洞察

1. **首个解决跨分词器偏好蒸馏的工作**：填补了一个明确的研究空白
2. **字符级对齐的优雅设计**：Aligned Span 以原始字符串为锚点，天然解决了分词器不兼容问题，零额外参数
3. **Reference model 作为重加权机制的理论分析**：从 DPO 梯度推导出 reference model 的本质作用是样本权重调控器，这一洞察为使用 teacher 作为 reference 提供了理论支撑
4. **实验设计严谨**：消融实验系统性地验证了每个组件的贡献

## 局限与展望

- 仅在 Qwen → Llama 这一模型系列组合上实验，未验证更多架构组合
- aligned span 的计算可能在长序列上引入额外开销，未报告时间复杂度
- 当前 importance weight 估计需要另外训练 teacher 的正/反向 DPO 模型，增加了 pipeline 复杂度
- 仅在英文 benchmark 上评估，跨语言场景（分词器差异更大）未涉及
- 未探索从更大模型（70B+）蒸馏的场景

## 相关工作与启发

- 与 DPO/TIS-DPO 的偏好对齐方法对比，CTPD 在端到端框架中引入了 teacher 信号
- 与 ULD/DSKD/Multi-Level OT 的 KD 方法对比，CTPD 专门为偏好蒸馏设计
- aligned span 机制可潜在地应用于其他需要跨 tokenizer 信号传递的场景（如跨模型评估、联邦学习等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个解决跨分词器偏好蒸馏问题，问题定义清晰
- **技术深度**: ⭐⭐⭐⭐ — 基于重要性采样的理论推导完整，span 级噪声分析严谨
- **实验充分性**: ⭐⭐⭐⭐ — 6 个 benchmark，多种 baseline，详细消融
- **实用价值**: ⭐⭐⭐⭐ — 解锁了异构模型间的偏好迁移，对模型压缩和部署有实际意义
