---
title: >-
  [论文解读] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning
description: >-
  [ICLR 2026][预训练][学习率调度] 提出 Warmup-Stable-Only (WSO) 学习率调度策略——在预训练中完全去掉学习率衰减阶段，虽然预训练指标较差，但在 SFT 后一致性地超越所有衰减策略，通过损失景观分析揭示 WSO 保持更平坦的极小值区域是其优势根源。
tags:
  - "ICLR 2026"
  - "预训练"
  - "学习率调度"
  - "监督微调"
  - "损失景观"
  - "Warmup-Stable-Only"
---

# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning

**会议**: ICLR 2026  
**arXiv**: [2603.16127](https://arxiv.org/abs/2603.16127)  
**代码**: 未开源  
**领域**: LLM预训练  
**关键词**: 学习率调度, 预训练, 监督微调, 损失景观, Warmup-Stable-Only

## 一句话总结

提出 Warmup-Stable-Only (WSO) 学习率调度策略——在预训练中完全去掉学习率衰减阶段，虽然预训练指标较差，但在 SFT 后一致性地超越所有衰减策略，通过损失景观分析揭示 WSO 保持更平坦的极小值区域是其优势根源。

## 研究背景与动机

### 预训练中的学习率调度现状

大语言模型预训练中，学习率 (LR) 调度是最关键但操作最复杂的超参之一。主流做法如下：

- **Cosine decay**: 自 GPT-3 以来最常用，LR 随训练按余弦曲线衰减到接近 0
- **Linear decay**: 近期研究表明线性衰减到 0 可取得更低预训练 loss
- **WSD (Warmup-Stable-Decay)**: 在训练末期才短暂衰减，更灵活，被 MiniCPM 等模型采用

这些策略的共同点：都在训练末期衰减 LR 以优化预训练指标。

### 核心矛盾

现有 LR 策略的优化目标是预训练阶段的性能 $\mathtt{Task}_{\rm pre}(M_{\rm pre})$，但实际应用中真正重要的是 SFT 后的最终性能 $\mathtt{Task}_{\rm post}(M_{\rm post})$。

近期研究（Sun & Dredze 2025; Springer et al. 2025）明确指出：**预训练性能更好的模型不一定 SFT 后更好**。这提出了一个根本性问题：为优化预训练指标而选择的 LR 衰减，在模型要经历 SFT 时是否仍是最优选择？

### 形式化定义

传统流程分阶段贪心选最优：

$$\widehat{M}_{\rm pre} = \arg\max_{M_{\rm pre} \in \mathcal{M}_{\rm pre}} \{\mathtt{Task}_{\rm pre}(M_{\rm pre}[M_{\rm rand}])\}$$

但理想目标应是全局联合优化：

$$\widehat{M}_{\rm post} = \arg\max_{(M_{\rm pre}, M_{\rm post}) \in (\mathcal{M}_{\rm pre}, \mathcal{M}_{\rm post})} \{\mathtt{Task}_{\rm post}(M_{\rm post}[M_{\rm pre}[M_{\rm rand}]])\}$$

## 方法详解

### 整体框架

论文的方法主体不是提出复杂模块，而是把一个被普遍接受的训练习惯拿来反问：预训练末期到底要不要衰减学习率。为此它构造了一个统一的调度器族，用一个标量参数把 WSO、WSD、Cosine、Linear 串到同一坐标系下，再在两阶段（预训练 + SFT）和三阶段（预训练 + 中间训练 + SFT）两套流程里逐格对比，最后用损失景观的曲率给出"为什么不衰减反而更好"的解释。

### 关键设计

**1. Warmup-Stable-Only (WSO)：把衰减阶段整段删掉。** 主流调度都默认训练末期要把学习率压低以收紧预训练 loss，而 WSO 直接质疑这一步是否值得。它是 WSD 的极简变体——只保留 warmup 与 stable 两段，彻底去掉 decay：

$$\eta^{\text{WSO}}(t, \alpha_{\text{pre}}) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \eta_{\max} & T_{\text{warmup}} < t \leq T_{\text{pre}} \end{cases}$$

学习率热身到 $\eta_{\max}$ 后就一直保持恒定直到预训练结束。这样做的代价是预训练 loss 会更差（恒定大 LR 没法把参数收敛进窄谷），但好处是模型停在一个更"松"的位置，为后续 SFT 留出调整空间——这正是后文损失景观分析要量化的东西。

**2. min LR factor 统一参数化：让四种调度器可比。** 要公平比较"衰减多少才好"，就得让不同调度器落在同一根轴上。论文用最小学习率因子 $\alpha_{\text{pre}}$（衰减终点占 $\eta_{\max}$ 的比例）把所有调度器参数化：$\alpha_{\text{pre}}=0.0$ 是衰减到 0 的最激进档（Linear/Cosine 常用），$\alpha_{\text{pre}}=0.1$ 是衰减到 10% 的温和档（Llama 3、OLMo 2 采用），$\alpha_{\text{pre}}=1.0$ 则等价于完全不衰减、即 WSO。这样"衰减强度"从一个离散的策略选择变成一条连续的扫描轴，WSO 不再是孤立方案，而是这条轴上的极端端点，使得"衰减越狠 SFT 越差"的趋势可以被直接读出来。

**3. 中间训练的衰减开关：把质疑推广到三阶段流程。** 现代预训练常在 SFT 前插入一段中间训练（mid-training），它末期通常也要衰减。论文再引入一个 $\alpha_{\text{mid}}$ 控制这一段：$\alpha_{\text{mid}}=0.0$ 让中间训练 Linear 衰减到 0，$\alpha_{\text{mid}}=1.0$ 则保持恒定 LR。借这个开关，"任何阶段的衰减是否都有害"成了可验证的命题，而结论是即便只在中间训练衰减，也会拖低最终 SFT 表现。

**4. 用 Hessian trace 量化损失景观的平坦度。** 前三点是"现象层"的设计，这一点是要回答"为什么"。论文用 sharpness——损失对参数的二阶曲率之和，即 Hessian 的迹——刻画极小值的平坦程度：

$$\text{Sharpness}(\theta_t) = \text{Tr}(\mathbf{H}_{\mathcal{L}}(\theta_t)) = \sum_{i=1}^{d} \frac{\partial^2 \mathcal{L}(\theta_t; \mathcal{D})}{\partial \theta_i^2}$$

直接算 Hessian 对 LLM 不现实，论文借 Hutchinson 无偏估计器只用若干次 Hessian-向量积就近似出迹。量出的结果给衰减损害适应性提供了机制解释：衰减策略把模型收进尖锐极小值，sharpness 比 WSO 高出 2–3 倍，而更尖的谷意味着 SFT 稍一扰动 loss 就剧烈变化、难以迁移；WSO 停在平坦区域，反而经得起微调的推动。

## 实验关键数据

### 主实验：两阶段设置（预训练 + SFT）

模型架构：1B 和 8B（Llama 3 系列架构）；预训练数据：FineWeb-Edu；SFT 数据：Tulu-3 SFT mixture。

| 模型 | 调度器 | $\alpha_{\text{pre}}$ | PT Valid Loss ↓ Δ | PT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|-------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | +0.071 | -1.7 | **+0.3** |
| 1B | WSD | 0.1 | +0.004 | -1.5 | +0.0 |
| 1B | WSD | 0.0 | +0.000 | -1.2 | -1.0 |
| 1B | Linear | 0.0 | +0.016 | +0.0 | -0.9 |
| 1B | Cosine | 0.1 | +0.019 | -0.1 | -0.7 |
| 8B | **WSO** | 1.0 | +0.127 | -0.8 | **+1.1** |
| 8B | WSD | 0.1 | +0.019 | -0.2 | -0.8 |
| 8B | WSD | 0.0 | +0.014 | +0.0 | -0.3 |
| 8B | Linear | 0.0 | +0.000 | -1.8 | +0.0 |

**关键发现**：WSO 预训练 loss 最差（8B 高出 0.127），但 SFT 后最优（8B 高出 1.1 分）。

### 三阶段设置（预训练 + 中间训练 + SFT）

| 模型 | 调度器 | $\alpha_{\text{pre}}$ | $\alpha_{\text{mid}}$ | MT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|-----------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | 1.0 | -0.1 | **+0.8** |
| 1B | WSD | 1.0 | 0.0 | +0.0 | +0.0 |
| 1B | Cosine | 0.1 | 0.0 | -3.1 | -3.7 |
| 8B | **WSO** | 1.0 | 1.0 | -2.1 | **+1.1** |
| 8B | WSD | 1.0 | 0.0 | +0.0 | -1.4 |
| 8B | Linear | 0.1 | 0.0 | -9.0 | -3.7 |

### 消融实验：Over-training 设置（2T tokens）

| 模型 | 调度器 | $\alpha_{\text{pre}}$ | PT Task Avg Δ | SFT Task Avg Δ |
|------|--------|----------------------|---------------|----------------|
| 1B | **WSO** | 1.0 | -1.5 | **+0.7** |
| 1B | WSD | 0.1 | +0.0 | +0.0 |
| 1B | WSD | 0.0 | +0.0 | -0.3 |

Over-training + 中间训练（2T + 500B tokens）时 WSO 优势更大：SFT Task Avg Δ 达 **+1.4**。

### 关键发现

1. **性能反转现象**：预训练表现最好的调度器（衰减到 0）在 SFT 后表现最差
2. **WSO 全面胜出**：在 1B/8B、两阶段/三阶段、标准/过训练所有设置下一致最优
3. **任何阶段的衰减都有害**：在三阶段设置中，即使只在中间训练衰减也会降低 SFT 性能
4. **Sharpness 负相关**：sharpness 与 SFT 性能的 Pearson 相关系数 $r=-0.709$

## 亮点与洞察

1. **反直觉的核心发现**：预训练 loss 更好 ≠ 下游任务更好，LR 衰减实际上损害模型可适应性
2. **理论解释清晰**：通过损失景观分析给出了 flat minima → 更好 SFT 性能的完整因果链
3. **极简实现**：WSO 比任何衰减策略都更简单——不需要调衰减比例和衰减阶段长度
4. **实践价值巨大**：建议开源模型应以 WSO 方式训练后再发布，给下游用户最大适应性
5. **规模一致性**：1B 到 8B、100B 到 2T tokens 训练规模下结论一致

## 局限性

1. 仅考察了 SFT 这一种后训练方式，未实验 DPO、RLHF 等对齐阶段
2. 实验规模最大 8B，更大模型（70B+）是否成立有待验证
3. WSO 预训练 loss 显著更差，若某些场景确实需要低预训练 loss（如蒸馏）可能不适用
4. Sharpness 与 SFT 性能的相关性分析样本量较小

## 相关工作与启发

- **Bergsma et al. 2025**：主张 Linear decay 到 0 最优——但这仅对预训练 loss 成立
- **WSD (Hu et al. 2024)**：WSO 可视为 WSD 的极限简化，呼应 WSD 的灵活性优势
- **Wen et al. 2025**：理论分析 WSD 时发现 decay 阶段导致 sharpness 增加，WSO 避免了这一问题
- **启发**：未来应基于最终部署目标（SFT/RLHF 后性能）而非预训练指标来选择训练策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 挑战了 "衰减越低越好" 的广泛共识，观点鲜明且有力
- **理论深度**: ⭐⭐⭐⭐ — 损失景观分析和形式化框架完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 2 种规模 × 4 种调度器 × 3 种训练设置 × 过训练，极其全面
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可用的实践建议，对 LLM 训练和模型发布策略有指导意义
- **总评**: ⭐⭐⭐⭐☆ — 实验扎实、结论反直觉且实用，是预训练策略研究的重要工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICML 2026\] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](../../ICML2026/llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](../../ICML2025/llm_pretraining/dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](../../ACL2026/llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)

</div>

<!-- RELATED:END -->
