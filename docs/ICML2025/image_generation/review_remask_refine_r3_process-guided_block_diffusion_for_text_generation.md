---
title: >-
  [论文解读] Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation
description: >-
  [ICML2025 (MOSS Workshop)][图像生成][扩散模型] 提出 R3 (Review, Remask, Refine) 框架，在推理阶段利用过程奖励模型 (PRM) 评估掩码扩散模型的中间生成块，对低质量块进行比例性重掩码并重新生成，实现无需额外训练的定向纠错，在数学推理任务上以极低的 PRM 调用次数取得显著提升。
tags:
  - "ICML2025 (MOSS Workshop)"
  - "图像生成"
  - "扩散模型"
  - "过程奖励模型"
  - "迭代优化"
  - "remasking"
  - "文本生成"
  - "推理时缩放"
---

# Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation

**会议**: ICML2025 (MOSS Workshop)  
**arXiv**: [2507.08018](https://arxiv.org/abs/2507.08018)  
**作者**: Nikita Mounier, Parsa Ideahpour (University of Pennsylvania)
**领域**: 图像生成  
**关键词**: masked diffusion model, 过程奖励模型, 迭代优化, remasking, 文本生成, 推理时缩放

## 一句话总结

提出 R3 (Review, Remask, Refine) 框架，在推理阶段利用过程奖励模型 (PRM) 评估掩码扩散模型的中间生成块，对低质量块进行比例性重掩码并重新生成，实现无需额外训练的定向纠错，在数学推理任务上以极低的 PRM 调用次数取得显著提升。

## 研究背景与动机

掩码扩散模型 (MDMs) 近年来成为文本生成的有力范式，通过迭代去掩码/去噪逐步生成连贯文本，代表模型包括 LLaDA 和 BD3-LM。这类模型的核心优势在于其迭代特性允许回溯和修正已生成的内容，但关键问题在于**应该重新掩码哪些部分**——随机重掩码效率低下，缺乏有目标的纠错能力。

现有方法的不足：
- **ReMDM**：允许在推理时重掩码已生成的 token，但缺乏智能的重掩码策略，无法精准定位错误
- **d1 框架**：通过 SFT + RL 对预训练掩码扩散模型进行后训练以增强推理能力，但需要额外训练成本
- **Best-of-N 采样**：对每个块生成 N 个候选并选最优，虽然效果好但 PRM 调用次数极高（每生成一个块都需 N 次评估），计算开销大
- **结果奖励模型 (ORM)**：仅评估最终输出，无法对中间生成步骤提供细粒度反馈

核心洞察：过程奖励模型 (PRM) 能评估中间步骤的质量，天然适合指导迭代式生成——PRM 分数低的块需要更多修正，PRM 分数高的块应保留。这一信号可以转化为智能的重掩码策略。

## 方法详解

### 整体框架

R3 由三个阶段构成循环的推理时纠错流程：
1. **Review（审查）**：用 PRM 对已生成的文本块打分
2. **Remask（重掩码）**：根据 PRM 分数将低质量块中更多的 token 重新掩码
3. **Refine（精炼）**：用扩散模型对掩码部分重新生成，选择最优候选替换原文

两个核心组件均为现成的预训练模型，无需任何微调：
- 基础掩码扩散模型 $M_{\text{diff}}$：给定含掩码 token 的序列，预测被掩码位置的内容
- 过程奖励模型 $M_{\text{PRM}}$：评估文本块 $x_b$ 在上下文 $C_b$ 条件下的质量分数 $S_b = M_{\text{PRM}}(x_b | C_b)$，$S_b \in [0,1]$

### 窗口化评估与批量精炼机制

R3 逐块构建文本，采用窗口化的评估和精炼策略：

**步骤 1 — 生成当前块**：用 $M_{\text{diff}}$ 生成块 $x_j$，拼接到已有序列 $X^{(j+1)} = X^{(j)} \oplus x_j$。

**步骤 2 — 窗口审查**：每隔 $K$ 个块，对最近 $K$ 个块组成的窗口 $W_j = \{x_{j-K+1}, \ldots, x_j\}$ 逐块用 PRM 评分，得到分数集 $\mathcal{S}_{W_j}$。

**步骤 3 — 精炼触发**：若窗口内最低分 $\min(\mathcal{S}_{W_j}) < \tau_{\text{thresh}}$（默认 0.8），则触发精炼。

**步骤 4 — 比例性重掩码与候选生成**：
- 对窗口内每个块 $x_b$，根据其 PRM 分数 $S_b$ 计算重掩码概率
- 重掩码比例 $\rho_b = \beta_I \cdot \tilde{P}_R(S_b)$，其中 $\beta_I$ 为强度因子
- 生成 $N_S$ 个候选精炼版本

**步骤 5 — 候选评分与选择**：用 PRM 对所有候选评分，选择最优候选替换原窗口。

### PRM 分数到重掩码概率的映射

这是 R3 的核心设计，将 PRM 分数 $S_b$ 转换为重掩码概率 $P_R(S_b)$：

1. 计算中间质量值：$q_b = \exp(-\alpha_B \cdot S_b)$，其中 $\alpha_B$（默认 10.0）控制指数衰减的陡峭程度
2. 窗口内归一化到 $[p_{\min}, 1]$：

$$P_R(S_b) = p_{\min} + (1 - p_{\min}) \cdot \frac{q_b - \min_{b' \in W} q_{b'}}{\max_{b' \in W} q_{b'} - \min_{b' \in W} q_{b'} + \epsilon}$$

其中 $p_{\min}$（如 0.01）确保即使高分块也有微小概率被重新评估。这一映射使得低 PRM 分数（高 $q_b$）对应高重掩码概率，实现**分数越低、修正越多**的直觉。

### 计算效率分析

设总块数为 $N_{\text{total}}$，窗口大小为 $K$：
- **Best-of-N**：每个块都需 $N_S$ 次 PRM 调用 → 共 $N_{\text{total}} \times N_S$ 次（如 $16 \times 5 = 80$ 次）
- **R3 最佳情况**：仅需 $\lceil N_{\text{total}} / K \rceil$ 次窗口评估（如 $K=8$ 时仅 2 次）
- **R3 最差情况**：每个窗口都触发精炼 → $2 \times \lceil N_{\text{total}} / K \rceil$ 次（如 4 次）

## 实验关键数据

### 实验设置
- **基础扩散模型**：LLaDA-8B-Instruct
- **过程奖励模型**：Qwen2.5-Math-PRM-7B
- **评测集**：MATH 500 数据集中 127 道需逐步推导的题目
- **超参数**：采样温度 0.8，PRM 阈值 $\tau_{\text{thresh}} = 0.8$，重掩码强度 $\beta_I = 0.8$，候选数 $N_S = 5$，$\alpha_B = 10.0$，16 个块 × 32 token = 512 总 token，128 步去掩码

### Table 1: MATH 500 子集 (127题) 准确率

| 方法 | 正确数 | 准确率 |
|---|---|---|
| Simple Diffusion (pass@1) | 37 / 127 | 29.13% |
| R3 (K=4) | 42 / 127 | 33.07% |
| R3 (K=6) | 44 / 127 | 34.65% |
| R3 (K=8) | 54 / 127 | **42.52%** |
| Block-wise Best-of-N (BoN) | 61 / 127 | 48.03% |

### Table 2: 不同方法的 PRM 调用次数对比 (16 块序列)

| 方法 | PRM 调用次数 | 准确率 |
|---|---|---|
| Simple Diffusion | 0 | 29.13% |
| R3 (K=8) | 2-4 | 42.52% |
| R3 (K=6) | 3-6 | 34.65% |
| R3 (K=4) | 4-8 | 33.07% |
| Block-wise BoN (N=5) | 80 | 48.03% |

## 关键发现

1. **PRM 引导的定向纠错有效**：R3 (K=8) 将简单扩散基线的准确率从 29.13% 提升到 42.52%，绝对提升 **13.4%**
2. **效率远优于暴力搜索**：R3 (K=8) 仅需 2-4 次 PRM 调用，而 BoN 需 80 次调用，R3 以 **~5% 的 PRM 计算开销达到 BoN ~88% 的准确率**
3. **更大窗口更优**：K=8 显著优于 K=4/K=6，说明更大的上下文窗口有助于 PRM 做出更准确的质量判断，精炼效果更好
4. **定性案例验证**：在三角函数题目中，R3 成功识别并修正了计算错误（如 $b = 2\pi/\pi = 3$ 被纠正为 $b = 2$）

## 亮点与洞察

- **极致简洁的设计**：整个框架仅依赖两个现成预训练模型的组合使用，无需任何训练或微调，即插即用
- **计算-质量的优雅平衡**：通过窗口化评估和条件触发精炼，避免了对每个块的冗余评估，将 PRM 调用集中在真正需要修正的区域
- **分数驱动的软重掩码**：不是简单地全部重掩码或不重掩码，而是根据 PRM 分数 proportionally 决定掩码比例，保留了高质量部分的信息
- **通用性强**：理论上可应用于任何掩码扩散模型 + 任何 PRM 的组合，不限于特定领域

## 局限性

- **实验规模有限**：仅在 MATH 500 的 127 题子集上评测，缺乏更大规模和更多任务（如 GSM8K、代码生成、通用文本生成）的验证
- **Workshop 论文深度**：作为 MOSS Workshop 论文，方法和实验都偏 preliminary，缺乏系统性的消融实验（如 $\alpha_B$、$\beta_I$、$\tau_{\text{thresh}}$ 的影响）
- **与 BoN 差距仍存在**：R3 最佳 42.52% vs BoN 的 48.03%，效率优势是否在更大计算预算下缩小尚未探讨
- **PRM 依赖**：效果高度依赖 PRM 的质量，而 PRM 目前主要面向数学推理等可验证任务，在开放域文本生成中的适用性存疑
- **固定窗口大小**：K 为固定超参数，自适应窗口大小、层次化评估等更灵活的策略有待探索
- **未与其他推理时扩展方法对比**：如 d1、SVDD 等利用扩散模型特性的推理时方法

## 相关工作与启发

- **LLaDA** (Nie et al., 2025)：大语言扩散模型，通过掩码扩散进行迭代去噪生成文本，是 R3 的基础模型之一
- **BD3-LM** (Arriola et al., 2025)：块扩散语言模型，将自回归块级生成与块内离散扩散结合，R3 可直接应用
- **ReMDM** (Wang et al., 2025)：允许在推理时重掩码已生成 token，R3 在此基础上引入 PRM 引导的智能重掩码
- **d1** (Zhao et al., 2025)：通过 SFT + RL 后训练掩码扩散模型以增强推理能力，与 R3 的免训练方法形成互补
- **Qwen2.5-Math-PRM** (Zhang et al., 2025)：本文使用的过程奖励模型，提供步骤级质量评估

**启发**：R3 的思路可推广——在任何迭代式生成模型中引入外部质量评估器进行定向精炼，不限于扩散模型。例如可以将类似策略应用于 draft-then-revise 范式的自回归模型。

## 评分

- 新颖性: ⭐⭐⭐⭐ — PRM 引导重掩码的思路直觉清晰且合理，但方法本身较为简单，缺乏深层技术创新
- 实验充分度: ⭐⭐⭐ — 仅 127 题单一数据集的评测，缺乏消融和更多基线对比，实验规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，方法描述严谨，算法伪代码和定性案例辅助理解
- 价值: ⭐⭐⭐⭐ — 提出了一个实用的免训练推理时增强框架，虽然实验初步但方向有价值，为后续研究提供了清晰的基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ICML 2025\] Origin Identification for Text-Guided Image-to-Image Diffusion Models](origin_identification_for_text-guided_image-to-image_diffusion_models.md)
- [\[ICML 2026\] DFlash: Block Diffusion for Flash Speculative Decoding](../../ICML2026/image_generation/dflash_block_diffusion_for_flash_speculative_decoding.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](../../ECCV2024/image_generation/local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->
