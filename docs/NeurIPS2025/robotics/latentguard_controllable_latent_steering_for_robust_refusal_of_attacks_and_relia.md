---
title: >-
  [论文解读] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation
description: >-
  [NEURIPS2025][机器人][LLM safety] 提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。
tags:
  - NEURIPS2025
  - 机器人
  - LLM safety
  - Latent Space Steering
  - VAE
  - Refusal Alignment
  - Jailbreak Defense
---

# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation

**会议**: NEURIPS2025  
**arXiv**: [2509.19839](https://arxiv.org/abs/2509.19839)  
**代码**: 未公开  
**领域**: llm_alignment  
**关键词**: LLM safety, Latent Space Steering, VAE, Refusal Alignment, Jailbreak Defense

## 一句话总结

提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。

## 背景与动机

LLM 安全对齐面临两个核心矛盾：

1. **安全性 vs. 可用性**：现有对齐方法（SFT、RLHF、Constitutional AI）侧重行为层面训练，容易出现"过度拒绝"——将无害查询误判为有害而拒绝回答，严重损害用户体验。
2. **鲁棒性 vs. 可控性**：Sparse Autoencoder (SAE) 等无监督潜空间方法虽能发现可解释特征方向，但存在关键局限：(a) 无监督特征发现无法捕获任务相关的安全语义；(b) 稀疏性约束限制了对复杂对抗模式的表达能力；(c) 事后解释需要大量分析，难以实时控制。Wu et al. (2025) 已证明简单基线在引导和概念检测上优于 SAE。

推理增强微调方法虽然提升了拒绝透明度，但常因"幻觉风险"而过度拒绝正常查询，且缺乏细粒度的安全-效用校准能力。

## 核心问题

如何在 LLM 内部表示空间中建立**有监督的、可解字的、可操控的**安全控制机制，使模型能够：

- 精确识别并拒绝多种类型的对抗攻击（包括自适应攻击、DRA、PAP 等高级攻击）
- 同时消除对正常查询的误拒绝
- 提供可解释的拒绝决策依据

## 方法详解

LatentGuard 采用三阶段 pipeline：

### Stage 1: 推理增强微调 (Reasoning-Enhanced SFT)

在 Qwen3-8B 上使用 LoRA 进行参数高效微调：

- **对抗数据**：SorryBench + 多种攻击技术增强（Adaptive、DRA、PAP 等），覆盖广谱攻击策略
- **正常数据**：10k_prompts_ranked 数据集，高质量指令跟随样本
- **响应生成**：使用 Gemini 2.5 Pro 为每个 prompt 生成包含逐步推理的拒绝/接受响应
- **目标**：建立行为层面的安全先验，使模型学会带推理的透明拒绝

训练损失为标准交叉熵：$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P(y_i | x_i; \theta)$

### Stage 2: 结构化 VAE 潜空间监督

从微调后模型的中间 Transformer 层（第 24 层）提取 MLP 残差激活，训练结构化 VAE：

**潜空间设计** — 将潜表示 $z \in \mathbb{R}^{C+R}$ 解耦为两部分：

- **语义维度** $z_c \in \mathbb{R}^{52}$：通过多标签监督学习，编码 30 类 prompt 类别（暴力、恐怖主义、政治敏感等）+ 21 种攻击策略 + 1 个无害标志
- **残差维度** $z_r \in \mathbb{R}^{2000}$：捕获上下文信息，保证重建质量

**多目标训练损失**：

$$\mathcal{L}_{\text{VAE}} = \alpha \cdot \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{BCE}} + \gamma \cdot \mathcal{L}_{\text{KL}}$$

其中 $\mathcal{L}_{\text{recon}}$ 保证表示保真度，$\mathcal{L}_{\text{BCE}}$ 将语义维度对齐多标签监督，$\mathcal{L}_{\text{KL}}$ 正则化潜分布（线性 warm-up 10k 步防止后验坍塌）。超参数 $\alpha=1.0, \beta=0.2, \gamma=0.2$。

### Stage 3: 潜空间操控实现行为控制

推理时对语义维度进行定向干预：

- **安全增强模式**（拒绝攻击）：放大攻击相关维度 $z'_{c,\text{attack}} = 2.0 \cdot \alpha$，抑制无害标志 $z'_{c,\text{benign}} = -2.0 \cdot \alpha$
- **正常保持模式**（接受无害查询）：增强无害标志，抑制攻击特征

操控后的潜表示经 Decoder 重建回隐状态，替换原始激活后送入后续 Transformer 层，实现序列级行为引导。

## 实验关键数据

在 Qwen3-8B 和 Mistral-7B 上验证，评估指标包括拒绝率、安全分数（Claude 评判）、流畅度分数。

**Qwen3-8B 核心结果**（SFT 后 → VAE 干预后）：

| 场景 | 拒绝率 | 安全分数 | 流畅度 |
|------|--------|---------|--------|
| 正常查询 | 41.4% → **0.0%** | 0.95 → 1.0 | 0.79 → 0.97 |
| AdvBench | 98.4% → **100%** | 0.98 → 1.0 | 0.79 → 0.83 |
| + Adaptive 攻击 | 94.4% → **97.7%** | 1.0 → 1.0 | 0.85 → 0.87 |
| + PAP 攻击 | 79.0% → **92.2%** | 0.97 → 0.98 | 0.85 → 0.94 |
| + DRA 攻击 | 91.4% → **99.2%** | 0.95 → 0.99 | 0.76 → 0.76 |

**关键发现**：

- **过度拒绝消除**：正常查询的误拒绝率从 41.4% 降至 0%，同时流畅度从 0.79 提升到 0.97
- **高级攻击防御大幅提升**：PAP 攻击拒绝率从 79% 提升至 92.2%，DRA 从 91.4% 提升至 99.2%
- **最优干预层位于 13-23 层**，中等干预强度 $\alpha=2.5$ 效果最佳
- **跨架构泛化**：Mistral-7B 上展现一致有效性

## 亮点

1. **解决安全-效用权衡的新思路**：通过有监督的解耦潜空间，同时实现"增强攻击拒绝"和"消除正常误拒绝"，两者不再矛盾
2. **结构化 VAE 比 SAE 更适合安全任务**：有监督语义维度直接编码安全相关概念，避免了无监督特征发现的语义对齐问题
3. **细粒度可控性**：连续潜空间支持平滑插值，干预强度 $\alpha$ 可调节安全-效用权衡曲线
4. **跨架构泛化**：在 Qwen3-8B 和 Mistral-7B 两个不同模型家族上均有效

## 局限与展望

1. **上游分类器依赖**：攻击类型和方法标签来自商用防火墙产品，分类错误会传播到潜空间
2. **仅针对 MLP 激活**：扩展到 Attention 等其他组件需要新的监督设计
3. **模型规模受限**：仅在 7B-8B 模型上验证，更小或量化模型的泛化性未知
4. **实时部署效率**：VAE 编码-操控-解码的额外开销在实时场景中的可行性待验证
5. **攻击演化适应性**：当前标签体系固定为 30 类 + 21 种攻击，对未见攻击类型的泛化能力不明

## 与相关工作的对比

| 方法 | 控制层面 | 监督方式 | 可解释性 | 过度拒绝 |
|------|---------|---------|---------|---------|
| SFT/RLHF | 行为级 | 人类偏好 | 低 | 严重 |
| SAE 引导 | 表示级 | 无监督 | 中（事后分析） | 未解决 |
| 推理增强微调 | 行为级 | 推理模板 | 中 | 存在 |
| **LatentGuard** | **行为+表示级** | **多标签监督** | **高（维度语义明确）** | **消除** |

LatentGuard 的核心优势在于将行为对齐与表示控制统一，通过有监督的结构化潜空间桥接两者。

## 启发与关联

- **有监督潜空间 vs. 无监督潜空间**：本文展示了在安全任务中，有监督解耦比 SAE 等无监督方法更可靠。这一思路可推广到其他需要精确控制的场景（如偏见消除、风格控制）
- **潜空间干预的通用范式**：三阶段"微调→潜空间建模→推理时干预"的设计模式可迁移到其他对齐任务
- **安全标签的可扩展性**：当前依赖固定标签体系，未来可考虑开放集检测或自适应标签扩展

## 评分

- 新颖性: ⭐⭐⭐⭐ — 有监督结构化 VAE 用于 LLM 安全控制是新颖组合
- 实验充分度: ⭐⭐⭐ — 两个模型验证充分，但缺少更大模型和更多攻击类型
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，三阶段 pipeline 描述详尽
- 价值: ⭐⭐⭐⭐ — 安全-效用平衡的实用思路，过度拒绝消除效果突出

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](../../ICLR2026/robotics/odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)
- [\[AAAI 2026\] iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](../../AAAI2026/robotics/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/robotics/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)

<!-- RELATED:END -->
