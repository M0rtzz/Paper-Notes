---
title: >-
  [论文解读] Curriculum Direct Preference Optimization for Diffusion and Consistency Models
description: >-
  [CVPR 2025][LLM对齐][DPO] 首次将课程学习引入 DPO 并首次将 DPO 适配到一致性模型，通过从"容易区分的偏好对"到"难以区分的偏好对"渐进训练，在文本对齐、美学和人类偏好上全面超越标准 DPO 和 DDPO，且仅需 1/10 训练数据量。
tags:
  - CVPR 2025
  - LLM对齐
  - DPO
  - 课程学习
  - 扩散模型
  - consistency model
  - 偏好优化
---

# Curriculum Direct Preference Optimization for Diffusion and Consistency Models

**会议**: CVPR 2025  
**arXiv**: [2405.13637](https://arxiv.org/abs/2405.13637)  
**代码**: https://github.com/CroitoruAlin/Curriculum-DPO  
**领域**: LLM对齐/RLHF  
**关键词**: DPO, 课程学习, 扩散模型, consistency model, 偏好优化

## 一句话总结
首次将课程学习引入 DPO 并首次将 DPO 适配到一致性模型，通过从"容易区分的偏好对"到"难以区分的偏好对"渐进训练，在文本对齐、美学和人类偏好上全面超越标准 DPO 和 DDPO，且仅需 1/10 训练数据量。

## 研究背景与动机

**领域现状**：DPO 已成功应用于 LLM 对齐，Diffusion-DPO 将其扩展到扩散模型的图像生成对齐。但当前 DPO 训练将所有偏好对等同对待，忽略了偏好对的"难度"差异。

**现有痛点**：标准 DPO 一次性喂入所有偏好对训练，但有些偏好对差异明显（很好 vs 很差），有些差异微小（两张都中等但略有差距）。混合训练导致效率低下——模型可能先被简单样本饱和，无法有效学习困难样本的微妙偏好。

**核心矛盾**：偏好学习需从丰富训练信号中提取渐进的偏好排序信息，但现有方法将信号压平为二元对比，丢失了排序中的层次结构。此外 DPO 尚未扩展到一致性模型。

**本文目标** (1) 如何利用偏好对间的难度梯度改善 DPO 训练？ (2) 如何将 DPO 推广到一致性模型？

**切入角度**：借鉴人类学习"先易后难"的课程学习思想——按排名差距分组偏好对，先训练差距大的"容易"对，逐步引入差距小的"困难"对。

**核心 idea**：用 reward model 排序生成图像，按排名差距分层构建课程从易到难渐进训练 DPO，同时提出 Consistency-DPO 损失函数首次适配一致性模型。

## 方法详解

### 整体框架
两阶段流程：(1) 排序阶段——对每个 prompt 生成 M 张图像，用 reward model 排序；(2) 课程训练阶段——将偏好对按难度分为 B 个批次，从易到难累积训练。适用于 Stable Diffusion 和 Latent Consistency Model 两种生成模型。

### 关键设计

1. **排序与课程划分**

    - 功能：将 M 张图像按 reward 排序，按排名差距构建不同难度的偏好对
    - 核心思路：对 prompt $c$ 生成 M 张图像，用 reward model $r_\phi$ 降序排列。创建偏好对 $(x_0^w, x_0^l)$，按排名差距分为 B 个批次：$L_k = (M-1)(B-k)/B$, $R_k = (M-1)(B-(k-1))/B$
    - 设计动机：批次 1 包含排名差距最大的"容易"对，批次 B 包含差距最小的"困难"对

2. **累积训练策略**

    - 功能：每加入新批次时保留所有之前的简单批次
    - 核心思路：第 k 阶段使用 $P = \bigcup_{i=1}^k S_i$ 训练
    - 设计动机：防止遗忘简单模式，困难样本在已有简单知识基础上学习

3. **Consistency-DPO 损失函数（首创）**

    - 功能：将 DPO 损失从扩散模型适配到一致性模型
    - 核心思路：$\mathcal{L}_{\text{Con-DPO}}(\phi) = -\mathbb{E}[\log \sigma(-\beta(d^w - d^l))]$，其中 $d^*$ 基于一致性函数的距离度量
    - 设计动机：一致性模型不使用噪声预测，无法直接用 Diffusion-DPO 的 $\epsilon$ 损失

4. **Diffusion-DPO 损失（改进版）**

    - 标准噪声预测损失，使用 LoRA 高效微调
    - $\beta=5000$（Diff-DPO）vs $\beta=200$（Con-DPO）

### 损失函数 / 训练策略
- AdamW 优化器，学习率 $3\times10^{-4}$
- 课程 B=5 批次，每批 $H_i=400$ 迭代；总 10000 迭代
- LCM: LoRA rank=64, ~2天 A100(64GB)；SD: LoRA rank=8, ~1天 A100(36GB)
- Reward models: Sentence-BERT（文本对齐）、LAION Aesthetics（美学）、HPSv2（人类偏好）

## 实验关键数据

### 主实验

**D1 数据集 — Latent Consistency Model：**

| 任务 | Baseline | DDPO | DPO | Curriculum DPO |
|------|---------|------|-----|----------------|
| 文本对齐 | 0.7243 | 0.7490 | 0.7502 | **0.7548** |
| 美学评分 | 6.0490 | 6.3730 | 6.4741 | **6.6417** |
| 人类偏好 | 0.2912 | 0.2952 | 0.2990 | **0.3237** |

**人类评估（1-5 分，11520 条标注）：**

| 设置 | Baseline | DDPO | DPO | Curriculum DPO |
|------|---------|------|-----|----------------|
| LCM 文本 | 2.778 | 2.810 | 2.846 | **3.440** (p<0.005) |
| LCM 美学 | 2.718 | 2.765 | 2.782 | **3.006** |
| SD 文本 | 2.276 | 2.983 | 2.821 | **3.175** |

### 消融实验

| 超参数 | 最优值 | 说明 |
|--------|--------|------|
| $\beta$ (Con-DPO) | 200 | 范围 {50,100,200,300,500} |
| $\beta$ (Diff-DPO) | 5000 | — |
| K (每批迭代) | 300-400 | {100,200,300,400,500} |
| B (课程批次) | 5 | {3,5,7}，所有值均优于无课程 |
| M (图片数) | 50 | Curriculum DPO 用 M=50 达 DPO M=500 效果 |

### 关键发现
- **数据效率提升 10 倍**：Curriculum DPO 用 M=50 达到标准 DPO 用 M=500 的性能
- **人类评估统计显著**：LCM 文本对齐 3.440 vs DPO 2.846（p<0.005）
- 课程学习在所有 B 值（3/5/7）下都优于无课程 baseline
- LoRA alone 降低性能，必须结合 DPO 才有效

## 亮点与洞察
- **课程学习在偏好优化中极其自然**：偏好对本身有内在难度梯度，利用这个结构是好洞察
- **10 倍数据效率**：实际应用中大幅减少 reward model 评估的计算开销
- **Consistency-DPO 首创**：将 DPO 扩展到一致性模型打开新的对齐方向
- 课程划分策略可迁移到 LLM 的 DPO 训练中

## 局限与展望
- Reward model 质量直接影响排序可靠性
- 仅在 SD v1.5 和 LCM 上验证，未测试 SDXL/SD3
- B 和 K 仍需手动调整
- 未探索在线课程——动态根据模型能力调整难度

## 相关工作与启发
- **vs Diffusion-DPO**: 标准 DPO 等同训练所有偏好对，本文通过课程分层显著提升
- **vs DDPO**: 使用 RL 方式，计算开销更大
- **vs SPO**: 分步优化策略与课程学习正交，可能互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 课程学习+DPO 和 Consistency-DPO 都是有价值的首创
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多模型、自动+人工评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验严谨
- 价值: ⭐⭐⭐⭐ 10x 数据效率有很强应用价值

<!-- RELATED:START -->

## 相关论文

- [Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](../../ICML2025/llm_alignment/adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [DiffPO: Diffusion Alignment with Direct Preference Optimization](../../ACL2025/llm_alignment/diffpo_diffusion_alignment.md)
- [Self-Consistency Preference Optimization](../../ICML2025/llm_alignment/self-consistency_preference_optimization.md)
- [D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](../../ICML2025/llm_alignment/d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)

<!-- RELATED:END -->
