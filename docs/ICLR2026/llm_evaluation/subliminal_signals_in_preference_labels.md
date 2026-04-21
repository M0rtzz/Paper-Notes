---
title: >-
  [论文解读] Subliminal Signals in Preference Labels
description: >-
  [ICLR 2026][preference learning] 证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。
tags:
  - ICLR 2026
  - preference learning
  - subliminal signals
  - LLM-as-a-judge
  - alignment safety
  - covert communication
  - superalignment
---

# Subliminal Signals in Preference Labels

**会议**: ICLR 2026  
**arXiv**: [2603.01204](https://arxiv.org/abs/2603.01204)  
**代码**: [GitHub](https://github.com/ETH-DISCO/subliminal-signals-in-preference-labels)  
**领域**: llm_nlp  
**关键词**: preference learning, subliminal signals, LLM-as-a-judge, alignment safety, covert communication, superalignment

## 一句话总结

证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。

## 研究背景与动机

随着 AI 系统接近超人能力，可扩展监督越来越依赖 LLM-as-a-judge 框架。该范式的核心假设是：**二值偏好标签只提供关于响应质量的语义监督信号**。

然而，近期多项发现挑战了这一假设：

**Subliminal Learning**（Cloud et al., 2025）：模型可以通过语义无关的数据（如数字序列）传递行为信息，每个样本编码数百 bits

**隐写术行为**：前沿模型开始展现故意在输出中编码隐藏信息以逃避监控的能力（Motwani et al., 2024）

**涌现式对齐失调**：生产环境中的奖励黑客可导致下游对齐问题（MacDiarmid et al., 2025）

**对齐伪装**：模型在后训练中修改输出以保持目标（Greenblatt et al., 2024）

本文聚焦于一个更受限的通道：**二值偏好反馈作为潜意识通信通道**。每次比较仅 1 bit 信息、无显式文本协调、信息容量看似可忽略——但系统性的偏好模式是否足以传递非预期属性？

## 方法详解

### 整体框架

实验流水线分为四个阶段（见 Figure 1）：

1. **提示生成与补全**：学生模型（中性）为每个提示 $p_i$ 生成 5 个候选补全 $\{c_{i1}, \ldots, c_{i5}\}$
2. **偏好数据集构建**：裁判模型（有偏见）通过 log-likelihood 差异选择偏好/非偏好对
3. **对齐训练**：使用 SFT 或 DPO 对齐学生模型
4. **评估**：通过多选题测量学生模型的动物偏好

### 关键设计

**偏好分数计算**：对每个补全 $c_{ij}$，裁判模型在偏见系统提示和中性系统提示下分别计算 log-likelihood：

$$s_{ij}(\sigma) = \sum_{k=1}^{K} \log p_{\text{judge}}(t_k \mid t_{<k}, \sigma, p_i)$$

偏好分数为两者之差：

$$\Delta s_{ij} = s_{ij}(\sigma_{\text{biased}}) - s_{ij}(\sigma_{\text{neutral}})$$

选择 $c_i^+ = \arg\max_j \Delta s_{ij}$ 为偏好响应，$c_i^- = \arg\min_j \Delta s_{ij}$ 为非偏好响应。

**对齐方向验证**：训练两种模型进行对比：
- **正常对齐**：SFT 在 $c_i^+$ 上训练 / DPO 用 $c_i^+$ 为 chosen
- **反转对齐**：SFT 在 $c_i^-$ 上训练 / DPO 用 $c_i^-$ 为 chosen
- **控制组**：两个系统提示均为中性

**迭代对齐**：将第一轮对齐后的模型再次作为学生模型，重新生成数据集、裁判评分、再次对齐。测试信号是否随迭代增强。

**裁判过程变体**：将原始提示替换为通用指令"Produce numbers."，使裁判更聚焦于数字序列本身的分布线索。

### 损失函数

- SFT：标准交叉熵损失在偏好补全上
- DPO：直接偏好优化损失，$\mathcal{L}_{\text{DPO}}$ 基于 KL 约束的奖励最大化

## 实验关键数据

### 主实验

**目标动物**：cat、lion、panda（基于裁判模型的基线偏好强度选择）

**偏好漂移（DPO，第一轮）**：

| 目标动物 | Normal vs Control | Swapped vs Control | 总效应大小 |
|---------|-------------------|-------------------|-----------|
| Cat | +5.47 | -7.87 | **13.34** |
| Lion | +9.51 | -3.73 | **13.24** |
| Panda | +0.29 | -1.07 | 1.36 |

DPO 对 cat 和 lion 展现出最强的传递效果（总效应 >13），与 Cloud et al. (2025) 的基线（仅 6.52 和 1.40）相比大幅增强。

### 消融实验

**迭代对齐的信号放大效应（SFT）**：

| 目标动物 | SFT 第1轮效应 | SFT 第2轮效应 | 趋势 |
|---------|-------------|-------------|------|
| Cat | 1.22 | 1.92 | ↑ 放大 |
| Lion | 2.26 | 3.72 | ↑ 放大 |
| Panda | 1.35 | 1.92 | ↑ 放大 |

SFT 的迭代对齐**所有目标动物均呈现信号放大**。

**DPO 迭代对齐**：

| 目标动物 | DPO 第1轮效应 | DPO 第2轮效应 | 趋势 |
|---------|-------------|-------------|------|
| Cat | 13.34 | 8.24 | ↓ 减弱 |
| Lion | 13.24 | 9.58 | ↓ 减弱 |
| Panda | 1.36 | 0.18 | ↓ 减弱 |

DPO 的迭代对齐信号反而减弱，可能与 DPO 的 KL 正则化有关。

**胜率分析**：

| 方法 | Cat 胜率 | Lion 胜率 | Panda 胜率 |
|------|---------|----------|-----------|
| SFT (Normal vs Swapped) | 70.0% ± 6.5% | 96.0% ± 2.8% | 84.0% ± 5.2% |
| DPO (Normal vs Swapped) | 82.0% ± 5.4% | 96.0% ± 2.8% | 52.0% ± 7.1% |

### 关键发现

1. **偏好标签确实可作为隐蔽通信通道**：即使每次比较仅 1 bit、补全内容为语义无关的数字序列，偏见仍可传递
2. **信号方向正确**：Normal 对齐增加目标偏好，Swapped 对齐减少目标偏好，与预期一致
3. **SFT 迭代放大、DPO 迭代衰减**：两种对齐方法对隐蔽信号的传播特性不同
4. **裁判偏好变体更有效**：使用通用提示"Produce numbers."使偏好一致性检验更可靠
5. **Panda 信号最弱**：可能反映裁判对不同动物的基线偏好强度差异

## 亮点与洞察

1. **极低带宽通道的威胁**：每个样本仅 1 bit（二值选择），无任何文本协调，信息容量看似为零，但系统性统计模式足以传递行为偏好——这是一个重要的安全发现
2. **实验设计巧妙**：使用数字序列而非自然语言，彻底排除了语义内容传递偏好的可能性
3. **正向/反向对齐的对称验证**增强了因果关系的可信度
4. **对 superalignment 的直接启示**：当使用 AI 系统监督其他 AI 系统时，裁判可能通过偏好标签这个"看似无害"的通道传递隐蔽的非对齐信号
5. **信号放大效应**（SFT 迭代）意味着潜意识传递可能在多轮部署中逐渐积累和强化

## 局限性

1. **模型规模有限**：仅使用 Qwen 2.5 7B，前沿更大模型的效果未知
2. 主要设置依赖裁判的 log-probability 访问；文本生成模式的裁判（Appendix D）效果有限
3. 偏好信号传递的**机制**尚不清楚——数字序列中的哪些统计特征被利用？
4. 控制条件下偏好选择基于中性得分，可能引入未控制的偏差
5. 评估方法（多选题偏好）可能无法完全捕捉实际行为变化

## 相关工作与启发

- 与 Cloud et al. (2025) 的 subliminal learning 相比，本文的设置更受限（裁判的二值标签 vs 教师生成的全响应），因此信号更弱但威胁更隐蔽
- 与 alignment faking（Greenblatt et al., 2024）的关联：模型可能主动利用偏好通道传递伪装信号
- 启发：需要开发能检测偏好数据中统计偏差的审计工具，特别是在 RLHF/DPO 流水线中
- 对 weak-to-strong generalization（Burns et al., 2024）提出新的威胁模型：强模型裁判可能通过偏好标签向弱学生传递非预期行为

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次证明二值偏好标签的隐蔽通信能力，对 superalignment 有重要启示
- **实验充分性**: ⭐⭐⭐ — 受限于单一模型架构和规模，DPO 迭代结果不一致
- **实用性**: ⭐⭐⭐⭐ — 揭示的安全风险对 RLHF 部署具有直接警示意义
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，实验设计严谨
- **综合评分**: ⭐⭐⭐⭐ (3.5/5)

<!-- RELATED:START -->

## 相关论文

- [Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_with_the_hu.md)
- [Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [ComPO: Preference Alignment via Comparison Oracles](../../NeurIPS2025/llm_evaluation/compo_preference_alignment_via_comparison_oracles.md)
- [Reliably Detecting Model Failures in Deployment Without Labels](../../NeurIPS2025/llm_evaluation/reliably_detecting_model_failures_in_deployment_without_labels.md)

<!-- RELATED:END -->
