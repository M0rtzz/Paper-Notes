---
title: >-
  [论文解读] ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection
description: >-
  [ICLR 2026][仇恨言论检测] 提出 ExPO-HM，受人类审核员培训流程启发，结合策略手册 SFT 预热、GRPO 课程学习和条件决策熵（CDE）奖励，首次实现 Explain-then-Detect 仇恨 Meme 检测在二分类、细粒度分类和推理质量上全面超越直接检测基线，F1 提升最高达 15-17%。
tags:
  - ICLR 2026
  - 仇恨言论检测
  - 多模态
  - GRPO
  - 课程学习
  - 条件决策熵
  - 可解释性
---

# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection

**会议**: ICLR 2026  
**arXiv**: [2510.08630](https://arxiv.org/abs/2510.08630)  
**代码**: [GitHub](https://github.com/JingbiaoMei/ExPO-HM)  
**领域**: llm_reasoning  
**关键词**: 仇恨言论检测, 多模态, GRPO, 课程学习, 条件决策熵, 可解释性

## 一句话总结

提出 ExPO-HM，受人类审核员培训流程启发，结合策略手册 SFT 预热、GRPO 课程学习和条件决策熵（CDE）奖励，首次实现 Explain-then-Detect 仇恨 Meme 检测在二分类、细粒度分类和推理质量上全面超越直接检测基线，F1 提升最高达 15-17%。

## 研究背景与动机

仇恨 Meme 检测是极具挑战性的在线内容审核任务。现有方法主要存在两个范式：

**直接检测（Direct Detection）**：仅输出二分类结果（hateful/benign），代表工作如 RA-HMD 等基于 CLIP 的方法，性能较好但无法提供解释，不满足真实审核需求。

**Explain-then-Detect**：先生成自然语言解释再做分类，但现有此类系统（如 LOREHM、U-CoT+）使用 CoT 提示或 agent 框架，**性能反而低于简单的 SFT 基线**。即使使用 GRPO 等后训练方法也无法缩小差距。

作者分析发现两个关键问题：

**模型解释遗漏关键线索**：如攻击目标和攻击类型等策略相关信息未被模型作为可能的解释假设

**二值奖励信号不足以引导推理**：正如人类标注员无法仅从 yes/no 标签学习，模型也需要更细粒度的反馈

核心类比：**人类审核员的培训流程**——先学习详细的审核策略手册，然后从细粒度类别练习到二分类判断——激发了 ExPO-HM 的设计。

## 方法详解

### 整体框架

ExPO-HM 包含三个阶段，模拟人类审核员培训过程：

1. **SFT-PM 预热**：策略手册增强的监督微调，教模型理解审核策略
2. **GRPO-CL**：课程学习式 GRPO，先细粒度后二分类
3. **CDE 奖励**：条件决策熵作为推理质量的代理奖励

### 关键设计

**SFT 策略手册预热（SFT-PM）**：将数据集的细粒度标签转化为结构化策略手册作为输入提示，每个策略项附有来自标注指南的描述。用策略手册增强输入训练 LMM，目标响应为细粒度标签。不使用人工撰写的金标准解释（off-policy，性能更差）。

**GRPO 课程学习（GRPO-CL）**：采用简单的 50/50/50 策略：
- 前 50% 训练步骤：仅使用细粒度数据（激励推理探索）
- 后 50% 训练步骤：50/50 混合细粒度和二分类数据

关键效果：标准 GRPO 二分类平均响应长度仅 28 tokens，GRPO-CL 几乎翻倍至 52 tokens，表明产生了更详细的推理。

**条件决策熵（CDE）**：

定义：给定解释 $\mathbf{e}$ 和输入 $\mathbf{x}$，CDE 是决策在解释条件下的熵：

$$H(d \mid \mathbf{e}, \mathbf{x}) = -\mathbb{E}_{d \sim \pi_\theta(\cdot|\mathbf{e},\mathbf{x})}[\log \pi_\theta(d \mid \mathbf{e}, \mathbf{x})]$$

原理：好的推理应导致清晰正确的决策（低熵），差的推理产生混淆（高熵）。

**CDE 奖励设计**：

$$r_{\text{CDE}}(h, \delta) = \delta \cdot f_{\text{correct}}(h) + (1-\delta) \cdot f_{\text{wrong}}(h)$$

其中 $\delta = \mathbf{1}[d = d^*]$ 表示预测正确性。正确且自信（低 CDE）→ 奖励；错误但自信 → 惩罚（系数 $\rho$）；错误但不确定 → 容忍。

### 损失函数 / 训练策略

总奖励：$r = r_{\text{format}} + r_{\text{acc}} + w \cdot r_{\text{CDE}}$

其中 $r_{\text{format}} \in \{0,1\}$ 检查输出格式，$r_{\text{acc}} \in [0,1]$ 衡量预测正确性。使用标准 GRPO 的 clipped surrogate loss 和 KL 正则化。默认超参数：$a=0.1$，$b=0.5$，$w=0.2$，$\rho=0.25$。

## 实验关键数据

### 主实验

在 HatefulMemes / MAMI / PrideMM 三个数据集上评估，基础模型为 Qwen2.5-VL-3B 和 7B。

**Qwen2.5-VL-7B 在 HatefulMemes 上的结果：**

| 方法 | Binary F1 | Attack F1 | Target F1 | LLM Judge | CDE ↓ |
|------|-----------|-----------|-----------|-----------|-------|
| Zero-shot | 65.9 | 44.7 | 64.5 | 5.0 | 0.33 |
| SFT | 74.5 | 58.4 | 69.4 | 5.0 | 0.33 |
| DPO | 73.6 | 63.2 | 66.6 | 4.9 | 0.32 |
| GRPO | 74.5 | 61.2 | 64.5 | 5.2 | 0.26 |
| RA-HMD（SOTA直接检测） | 80.2 | — | — | 5.5 | — |
| **ExPO-HM** | **81.1** | **75.6** | **77.2** | **6.2** | **0.03** |

ExPO-HM 首次让 Explain-then-Detect 系统全面超越直接检测的 SOTA（RA-HMD），同时在推理质量上大幅领先。

**跨数据集一致性**（7B 模型）：

| 数据集 | GRPO Binary F1 | ExPO-HM Binary F1 | 提升 |
|--------|---------------|-------------------|------|
| HatefulMemes | 74.5 | 81.1 | +6.6 |
| MAMI | 76.8 | 82.3 | +5.5 |
| PrideMM | 73.2 | 78.7 | +5.5 |

### 消融实验

| # | SFT-PM | GRPO-CL | CDE | Binary F1 | Attack F1 | Target F1 | LLM ↑ | CDE ↓ |
|---|--------|---------|-----|-----------|-----------|-----------|-------|-------|
| 1 | - | - | - | 74.5 | 61.2 | 64.5 | 5.2 | 0.263 |
| 2 | ✓ | - | - | 75.8 | 70.8 | 70.2 | 5.6 | 0.092 |
| 3 | ✓ | ✓ | - | 78.4 | 74.3 | 76.1 | 5.8 | 0.056 |
| 4 | ✓ | ✓ | ✓ | **81.1** | **75.6** | **77.2** | **6.2** | **0.026** |

三个组件均有贡献：SFT-PM 大幅提升细粒度指标，GRPO-CL 进一步全面提升，CDE 显著改善推理质量（LLM Judge 5.8→6.2）。

### 关键发现

1. **Explain-then-Detect 首次超越 Direct Detection**：之前所有此类系统均不如 SFT 基线
2. **CDE 与 LLM-Judge 强相关**：Pearson $r=-0.78$，Spearman $\rho=-0.81$（$p<0.001$）
3. **SFT 预热策略至关重要**：Binary-only SFT 在 RL 阶段反而劣于无预热基线
4. **CDE 不导致策略熵坍塌**：整体策略熵与不使用 CDE 的基线相当
5. **人工评估验证**：ExPO-HM 100% 逻辑一致性 vs GRPO 96%，帮助性评分 2.2 vs 1.6

## 亮点与洞察

1. **人类标注员培训的类比非常精准**：策略手册→细粒度练习→二分类判断的渐进式流程
2. **CDE 是推理质量的优秀代理指标**：定义简洁（条件熵），与人工评估高度相关，且可作为可微奖励信号
3. **关键发现：好的 SFT 不一定导致好的 RL**：Binary SFT 在 SFT 阶段最好但 RL 后最差
4. **三数据集一致性**：方法在不同仇恨内容类型上泛化良好
5. **实验极其全面**：消融、预热策略比较、CDE 分析、校准分析、人工评估一应俱全

## 局限与展望

1. **数据集规模有限**：仇恨 Meme 标注数据（尤其是带解释的）非常稀缺
2. **单轮交互**：仅评估单轮推理，未考虑多轮审核对话场景
3. **文化依赖性**：审核策略高度依赖文化背景，跨文化适用性未验证
4. **基础模型限制**：仅在 Qwen2.5-VL 3B/7B 上验证
5. 可扩展到其他内容审核任务（如虚假信息检测、网络暴力识别）

## 相关工作与启发

- **RA-HMD** (Mei et al., 2025)：之前的 SOTA 直接检测方法，ExPO-HM 首次超越
- **LOREHM** (Huang et al., 2024)：基于 LLaVA-Next-34B 的推理 agent框架
- **GRPO** (Shao et al., 2024)：ExPO-HM 在其基础上加入 CDE 奖励和课程学习
- 启发：CDE 的"推理质量代理"思路可推广到其他需要可解释推理的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — CDE 概念新颖，课程学习策略设计精巧
- **技术深度**: ⭐⭐⭐⭐ — 从人类培训流程到具体算法设计的映射完整
- **实验充分性**: ⭐⭐⭐⭐⭐ — 三数据集、多基线、消融、人工评估极其全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，实验组织有序
- **实用价值**: ⭐⭐⭐⭐ — 对内容审核有直接应用价值
- **综合推荐**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](../../ACL2026/interpretability/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)

<!-- RELATED:END -->
