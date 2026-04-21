---
title: >-
  [论文解读] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
description: >-
  [CVPR2026][多模态][自演化学习] 提出 EvoLMM，一个完全无监督的自演化框架：从单一骨干 LMM 中分出 Proposer（生成视觉问题）和 Solver（多次回答），通过连续自一致性奖励取代离散多数投票，让模型仅用原始图片即可自我提升多模态数学推理能力（ChartQA +2.7%, MathVista +2.1%）。
tags:
  - CVPR2026
  - 多模态
  - 自演化学习
  - 大型多模态模型
  - 连续奖励
  - Proposer-Solver
  - 无监督强化学习
  - 自一致性
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**会议**: CVPR2026  
**arXiv**: [2511.16672](https://arxiv.org/abs/2511.16672)  
**代码**: [mbzuai-oryx/EvoLMM](https://github.com/mbzuai-oryx/EvoLMM)  
**领域**: 多模态VLM  
**关键词**: 自演化学习, 大型多模态模型, 连续奖励, Proposer-Solver, 无监督强化学习, 自一致性

## 一句话总结

提出 EvoLMM，一个完全无监督的自演化框架：从单一骨干 LMM 中分出 Proposer（生成视觉问题）和 Solver（多次回答），通过连续自一致性奖励取代离散多数投票，让模型仅用原始图片即可自我提升多模态数学推理能力（ChartQA +2.7%, MathVista +2.1%）。

## 背景与动机

1. **标注依赖瓶颈**：现有 LMM 训练高度依赖人工标注数据（QA 对、元数据），限制了可扩展性和领域泛化能力。
2. **外部奖励模型依赖**：许多方法需要额外的 reward model 或人类评估来提供学习信号，增加系统复杂度。
3. **离散奖励不稳定**：先前自进化 LLM 工作（如 SQLM）使用多数投票离散奖励，在多模态场景中由于 Solver 输出早期高度不一致，经常产生零奖励，导致优化不稳定。
4. **语言域方法难迁移**：现有自演化研究（Multi-Agent Evolve 等）局限于纯语言领域，多模态场景需要视觉接地，增加了自评估难度。
5. **半监督残留**：已有的多模态自改进方法（Vision-SR1、ViPER、Vision-Zero）仍依赖重建目标、SFT 冷启动或外部生成模型（GPT-4o/Gemini）产生的隐式监督。
6. **核心问题**：能否让 LMM 在完全无人工标注和外部奖励的条件下，自主提升推理能力？

## 方法详解

### 整体框架

EvoLMM 从同一预训练骨干模型中实例化两个协作角色：

- **Proposer $\pi_\phi(q|x)$**：给定原始图像 $x$，生成视觉接地的数学问题 $q$
- **Solver $\pi_\theta(y|x,q)$**：对问题 $q$ 采样 $N=5$ 个独立回答 $y_{1:N}$

两者通过内部一致性信号形成闭环训练，不使用任何 ground-truth 标签。

### 连续自一致性 Solver 奖励

$$r_i^{\text{sol}} = \big(p(y_i|x,q)\big)^\gamma \cdot \Big(1 - \lambda_{\text{len}} \cdot \max\{0, (w_i - \tau)/\tau\}\Big)$$

- $p(y_i|x,q)$：经验答案分布中 $y_i$ 的一致性得分
- $\gamma=0.7$：奖励柔化指数，放大中等置信度差异
- 长度惩罚项：鼓励简洁回答（目标长度 $\tau=6$ 词）
- **关键优势**：相比离散多数投票，即使部分一致（如 5 个中 2 个一致）也能提供非零梯度

### 熵引导连续 Proposer 奖励

$$r^{\text{prop}} = \exp\!\left(-\frac{(H(x,q) - \mu_H)^2}{2\sigma_H^2}\right)$$

- $H(x,q)$：Solver 答案分布的熵，衡量问题难度
- 带通滤波器设计：$\mu_H=0.90$，$\sigma_H=0.35$
- 当 $H \approx 0$ 时（问题过简单）奖励低；$H$ 过大时（问题不可解）奖励也低
- 自适应课程学习：随着 Solver 变强，Proposer 需生成稍难但仍可解的问题

### 训练优化

- **策略梯度**：REINFORCE + EMA 基线 $b_A$ 方差缩减
- **KL 正则化**：token 级 KL 散度约束，防止偏离预训练分布过远
- **动态 KL 控制器**：$\beta_A \leftarrow \text{clip}(\beta_A \cdot \exp(\eta \cdot (\overline{KL}_A - \tau_A)/\tau_A), \beta_{\min}, \beta_{\max})$
- **更新频率**：Solver 每步更新，Proposer 每 5 步更新一次
- **参数高效**：双 LoRA 适配器，骨干冻结
- **训练配置**：6000 步，batch size 1，AdamW，lr=1e-6，8× AMD MI250X GPU

## 实验关键数据

### 主实验：连续 vs 离散奖励（Qwen2.5-VL-7B）

| 方法 | ChartQA | MathVista | MathVision | MathVerse | ScienceQA | AI2D | MMMU |
|------|---------|-----------|------------|-----------|-----------|------|------|
| Baseline | 84.00 | 68.46 | 23.91 | 43.78 | 88.30 | 82.61 | 51.11 |
| + 离散奖励 | 84.62 | 68.88 | 22.52 | 42.10 | 87.98 | 82.18 | 50.84 |
| + 连续奖励 (Ours) | **86.70** | **70.52** | **24.81** | **44.88** | **89.50** | **83.41** | **52.01** |
| Δ | +2.7% | +2.06% | +0.9% | +1.1% | +1.2% | +0.8% | +0.9% |

离散奖励在 MathVision/MathVerse 上甚至出现退化，说明多数投票在多模态推理中过于脆弱。

### 跨骨干泛化（Table 3）

| 模型 | ChartQA Δ | MathVista Δ | ScienceQA Δ |
|------|-----------|-------------|-------------|
| Qwen2.5-VL-7B | +2.70 | +2.06 | +1.20 |
| InternVL3-8B | +2.57 | +2.00 | +0.36 |
| Gemma3-12B-It | +2.97 | +2.00 | +1.08 |
| Llama-3.2-11B | +3.00 | +2.00 | +1.20 |

所有骨干均获 +1~3% 提升，证明方法的架构无关性。

### 消融：微调策略（Table 2）

| 策略 | ChartQA | MathVista | MathVision |
|------|---------|-----------|------------|
| LoRA (默认) | **86.70** | **70.52** | **24.81** |
| QLoRA | 85.32 | 68.92 | 23.97 |
| Full Fine-tune | 84.20 | 68.41 | 23.37 |

- Full Fine-tune 反而退化，因为无监督设定下全参更新易过拟合且与 KL 正则冲突
- QLoRA 因量化噪声降低 Solver 一致性

### 模型规模扩展（Table 4）

Qwen2.5-VL-72B 上 ChartQA 88.20→91.04（+2.84），MathVista 73.93→76.44（+2.51），更大模型获更大绝对增益。

## 亮点

1. **完全无监督**：仅用约 6k 张原始图片（无任何 QA 对或元数据），即可稳定提升推理能力
2. **连续奖励设计精巧**：Solver 侧基于一致性概率的软奖励 + Proposer 侧基于熵的带通奖励，避免了离散奖励的梯度消失
3. **涌现式课程学习**：Proposer 自动从简单问题过渡到中等难度问题，无需人工设计课程
4. **方法轻量可复现**：双 LoRA + 冻结骨干，6000 步即可完成训练
5. **泛化性强**：在 4 种不同架构（Qwen、InternVL、Gemma、Llama）和 2 种规模（7B/72B）上均有效

## 局限与展望

1. **提升幅度有限**：最大约 +3%，对于已接近饱和的 benchmark（如 ScienceQA 88→89%）边际效益较小
2. **仅验证数学推理**：未扩展到更广泛任务（OCR、VQA、视觉对话等），通用性存疑
3. **训练数据量小**：仅用 ~6k 图片，未探索更大规模数据是否能带来更大收益
4. **单轮问答**：Proposer 生成单个问题，未考虑多轮交互或链式提问
5. **Solver 采样数固定**：$N=5$ 作为超参，未充分探索不同 $N$ 的影响
6. **无与 RLHF/RLVR 的直接对比**：缺少与有监督强化学习方法的公平比较

## 与相关工作的对比

| 方法 | 监督类型 | 奖励形式 | 外部模型 | 多模态 |
|------|---------|---------|---------|--------|
| SQLM [5] | 无标注 | 离散多数投票 | 无 | ❌ 仅文本 |
| Multi-Agent Evolve [6] | 无标注 | Judge 角色 | 隐式 Judge | ❌ 仅文本 |
| Vision-SR1 [18] | SFT 冷启动 | 感知+推理分解 | 无 | ✅ |
| ViPER [47] | 重建目标 | 图像/实例重建 | OmniGen2/Qwen-Image | ✅ |
| Vision-Zero [36] | 合成图对 | 社交推理游戏 | GPT-4o/Gemini | ✅ |
| **EvoLMM (本文)** | **完全无监督** | **连续自一致性** | **无** | ✅ |

EvoLMM 是对比中唯一不依赖任何形式外部监督或模型的多模态自演化方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ （连续奖励替代离散奖励、熵引导带通 Proposer 奖励设计巧妙）
- 实验充分度: ⭐⭐⭐⭐ （跨骨干、跨规模、多消融，但缺少与有监督 RL 的比较）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富，公式推导完整）
- 价值: ⭐⭐⭐⭐ （为无监督多模态自演化提供了简洁有效的 baseline，但实际提升幅度不大）

<!-- RELATED:START -->

## 相关论文

- [Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)
- [Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)

<!-- RELATED:END -->
