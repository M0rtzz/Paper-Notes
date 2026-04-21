---
title: >-
  [论文解读] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence
description: >-
  [ICML 2025][模型压缩][知识蒸馏] 本文深入分析了知识蒸馏中 FKLD 和 RKLD 的概率质量分配缺陷，发现它们在 Hardness-Concentration 和 Confidence-Concentration 两种效应上分别处于极端，提出基于 α-β-divergence 的 ABKD 框架，通过调节 α 和 β 灵活平衡两种效应，在 17 个语言/视觉数据集、12 种师生配置上取得了 SOTA 性能。
tags:
  - ICML 2025
  - 模型压缩
  - 知识蒸馏
  - α-β-Divergence
  - 概率质量分配
  - 分布匹配
  - 散度函数
---

# ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence

**会议**: ICML 2025  
**arXiv**: [2505.04560](https://arxiv.org/abs/2505.04560)  
**代码**: [ghwang-s/abkd](https://github.com/ghwang-s/abkd)  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, α-β-Divergence, 概率质量分配, 分布匹配, 散度函数

## 一句话总结

本文深入分析了知识蒸馏中 FKLD 和 RKLD 的概率质量分配缺陷，发现它们在 Hardness-Concentration 和 Confidence-Concentration 两种效应上分别处于极端，提出基于 α-β-divergence 的 ABKD 框架，通过调节 α 和 β 灵活平衡两种效应，在 17 个语言/视觉数据集、12 种师生配置上取得了 SOTA 性能。

## 研究背景与动机

### 现状
知识蒸馏（KD）是将大型教师模型的知识转移到小型学生模型的经典技术。核心操作是最小化教师分布 $p$ 和学生分布 $q_\theta$ 之间的散度 $\mathbb{D}(p \| q_\theta)$。当前主流选择是前向 KL 散度（FKLD）和反向 KL 散度（RKLD）。

### 痛点
- **FKLD** 产生过于平滑的学生分布，均匀对待所有类别的匹配误差，无法引导学生聚焦目标类，导致错误预测。
- **RKLD** 使学生过度聚焦目标类，忽略教师分布中的软标签信息，退化为类似 one-hot 标签的监督。
- 两者的缺陷源于对概率质量分配方式的极端选择，但此前缺乏系统性理论分析。

### 核心矛盾
如何在关注困难类别（Hardness-Concentration）和关注高置信类别（Confidence-Concentration）之间取得平衡？FKLD 两者都太弱，RKLD 两者都太强，没有中间地带。

### 本文切入角度
通过跟踪梯度更新过程中的 log mass ratio $\mathsf{LogR}$，分析不同散度函数如何影响概率质量的重新分配，揭示两种浓度效应的共同作用机制，进而引入 α-β-divergence 作为统一框架。

## 方法详解

### 整体框架

ABKD 的训练目标为：

$$\ell = \ell_{\text{CE}} + \lambda \cdot \mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q_\theta)$$

其中 $\mathbb{D}_{\text{AB}}^{(\alpha,\beta)}$ 是 α-β-divergence，通过调节超参数 α 和 β 来控制概率质量分配。

### 关键设计

#### 1. Log Mass Ratio 分析框架

定义监控量 $\mathsf{LogR}_t^{\mathcal{A}}(y) = \log\frac{q_{t+1}^{\mathcal{A}}(y)}{q_t(y)}$，该量与 logit 梯度成正比。通过分析 $|\mathsf{LogR}|$ 的上界，揭示两种浓度效应：

- **Hardness-Concentration**（项 b）：$|s(p(k)) - s(q_t(k))|$，衡量匹配误差。函数 $s$ 越尖锐，效应越强。
- **Confidence-Concentration**（项 a）：$q_t(y)^\beta$，用学生置信度加权。$\beta$ 越大，越聚焦高置信类别。

**FKLD**：$s(x) = x$（线性，弱 Hardness）、$\beta = 0$（无 Confidence 加权）→ 两者都太弱。
**RKLD**：$s(x) = \log(x)$（对数，强 Hardness）、$\beta = 1$（强 Confidence 加权）→ 两者都太强。

#### 2. α-β-Divergence 定义

$$\mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q) = -\frac{1}{\alpha\beta} \sum_k \left[ p(k)^\alpha q(k)^\beta - \frac{\alpha}{\alpha+\beta} p(k)^{\alpha+\beta} - \frac{\beta}{\alpha+\beta} q(k)^{\alpha+\beta} \right]$$

这是一个通用的散度族，统一了多种已知散度：

| 散度 | α | β |
|------|---|---|
| FKLD | 1 | 0 |
| RKLD | 0 | 1 |
| Hellinger 距离 | 0.5 | 0.5 |
| 欧氏距离 | 1 | 1 |
| α-divergence | α+β=1 |
| β-divergence | α=1 |

#### 3. 平衡机制

α-β-divergence 的 LogR 上界为：
$$|\mathsf{LogR}_t^{(\alpha,\beta)}(y)| \leq \eta \cdot q_t(y)^{\beta} \cdot \left|\frac{p(y)^\alpha - q_t(y)^\alpha}{\alpha}\right| + \ldots$$

- **β 控制 Confidence-Concentration**：$\beta \to 0$ 接近 FKLD 效应，$\beta \to 1$ 接近 RKLD 效应。
- **α 控制 Hardness-Concentration**：$\alpha \to 1$ 接近 FKLD 效应，$\alpha \to 0$ 接近 RKLD 效应。

通过选择适当的 α 和 β，可以实现两种效应的连续插值，避免极端情况。

### 损失函数 / 训练策略

- 最终损失：$\ell = \ell_{\text{CE}} + \lambda \cdot \mathbb{D}_{\text{AB}}^{(\alpha,\beta)}(p \| q_\theta)$
- 仅需修改损失函数，不引入额外可训练参数
- α 和 β 作为超参数，论文提供了详细的调参指南

## 实验关键数据

### 主实验：LLM 指令跟随（GPT-2 XL → GPT-2）

| 方法 | Dolly Eval | Self-Instruct | Vicuna Eval | Super-Natural | Unnatural |
|------|-----------|---------------|-------------|---------------|-----------|
| SFT | 23.14 | 10.22 | 15.15 | 17.41 | 19.76 |
| KD (FKLD) | 23.80 | 10.01 | 15.25 | 17.69 | 18.99 |
| MiniLLM (RKLD) | 24.62 | 12.49 | 17.30 | 23.76 | 24.30 |
| DISTILLM | 25.32 | 11.65 | 16.76 | 23.52 | 25.79 |
| **ABKD** | **25.65** | **13.47** | 16.06 | **26.47** | **29.32** |

ABKD 在 1.5B→0.1B 的蒸馏配置上，ROUGE-L 相比 FKLD/RKLD 提升 0.81~3.31 分。

### 消融实验：α-β 超参数空间可视化

论文 Figure 1(a) 展示了在 ABKD 的 (α, β) 二维搜索空间中性能的热力图。FKLD 和 RKLD 仅是该空间中的特殊点，α-divergence 只能沿 α+β=1 的子流形搜索，而 ABKD 提供了完整的二维搜索空间。

### 关键发现

1. ABKD 在所有 12 种师生配置上均优于或匹配 SOTA 方法
2. 在视觉分类任务上同样有效，证明框架的通用性
3. ABKD 可以进一步增强现有 KD 方法——修改它们的损失函数即可带来额外增益

## 亮点与洞察

1. **理论深度出色**：通过 LogR 分析框架，将 FKLD/RKLD 的经验缺陷归因为两种浓度效应的极端组合，提供了统一的理论解释
2. **极简实现**：仅需修改损失函数，无需额外参数、无需数据增强，即可显著改善蒸馏效果
3. **通用框架**：α-β-divergence 统一了大量已知散度，ABKD 天然泛化了整个散度选择空间
4. **实践指导性强**：论文提供了详细的超参数调优指南，降低使用门槛
5. **对比 WSD 和 JSD 的缺陷分析**：FKLD+RKLD 加权和（WSD）在极端概率比处不稳定，JSD 在分布远离时梯度消失，而 α-β-divergence 自然避免这些问题

## 局限与展望

1. **超参数搜索成本**：α 和 β 两个超参数需要额外调优，虽然论文提供了指南但仍增加调参负担
2. **缺乏自适应机制**：α 和 β 在训练全程固定，缺乏动态调节策略（如随训练阶段自动调整）
3. **Token-level 蒸馏限制**：在 LLM 中仅考虑 token-level 分布匹配，未涉及 sequence-level 的蒸馏
4. **理论分析依赖简化假设**：LogR 分析中使用的 softmax 近似在实际深层网络中可能存在偏差

## 相关工作与启发

- **FKLD 系列**：传统 KD (Hinton 2015)、SeqKD (Kim & Rush 2016) → 过于平滑
- **RKLD 系列**：MiniLLM (Gu et al. 2024)、Kim et al. 2024 → 过度聚焦
- **混合方法**：GKD (Agarwal et al. 2024) 使用 JSD、DISTILLM (Ko et al. 2024) 使用 Skewed KLD → 启发式缓解但缺乏统一框架
- **本文启发**：选择散度函数时应关注其对概率质量分配的影响，而非仅从信息论角度考虑

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从概率质量分配角度统一分析散度函数，理论视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 17 数据集、12 配置、语言+视觉全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图示直观，逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 即插即用的通用 KD 框架，理论与实践价值均很高

<!-- RELATED:START -->

## 相关论文

- [Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [Local Dense Logit Relations for Enhanced Knowledge Distillation](../../ICCV2025/model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution](when_data-free_knowledge_distillation_meets_non-transferable_teacher_escaping_ou.md)

<!-- RELATED:END -->
