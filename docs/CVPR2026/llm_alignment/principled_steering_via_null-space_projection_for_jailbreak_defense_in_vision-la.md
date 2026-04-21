---
title: >-
  [论文解读] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models
description: >-
  [CVPR 2026][LLM对齐][越狱防御] 提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。
tags:
  - CVPR 2026
  - LLM对齐
  - 越狱防御
  - 激活转向
  - 零空间投影
  - VLM安全
  - 推理时防御
---

# Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.22094](https://arxiv.org/abs/2603.22094)  
**代码**: 无  
**领域**: LLM Alignment / VLM Safety  
**关键词**: 越狱防御, 激活转向, 零空间投影, VLM安全, 推理时防御

## 一句话总结

提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。

## 研究背景与动机

视觉语言模型（VLM）在开放场景部署时极易受到视觉越狱攻击的威胁——攻击者通过对图像添加对抗扰动或嵌入恶意指令来绑架模型生成有害内容。现有防御方法主要包括三个方向：

**训练时防御**（如对抗训练、安全微调）：计算成本高，需要额外标注数据

**推理时防御**（如提示重写、多轮检测）：延迟大、效率低

**激活转向**（Activation Steering）：轻量且无需训练，但存在关键缺陷

激活转向方法通过向模型隐状态注入"拒绝方向向量"来引导安全输出，是一种高效的推理时防御。然而，**它不区分良性和恶意输入**——拒绝向量同时影响所有输入，导致良性请求也被错误拒绝（即过度拒绝问题），严重损害模型的通用能力。作者通过观察发现，转向后良性激活也发生了偏移，这解释了过度拒绝现象的根源。

核心动机在于：能否设计一个"**选择性转向**"机制，让转向操作仅作用于恶意输入，而对良性输入"透明"？

## 方法详解

### 整体框架

NullSteer 的设计思路清晰：构建一个线性变换矩阵 Δ，使其在良性激活子空间内的投影为零（不产生扰动），而在恶意方向上则动态引导模型走向拒绝语义。整个过程无需训练，仅需少量良性和恶意样本来估计投影矩阵。

推理时，对每一层的隐状态进行修改：
$$\mathbf{h}^{(l)'} = \mathbf{h}^{(l)} + \lambda \tilde{\Delta}^{*(l)} \mathbf{P}^{(l)} \mathbf{h}^{(l)}$$

### 关键设计

1. **零空间投影矩阵 P 的构建**：核心保障良性不受干扰

    - 收集 $N_b$ 个良性输入的隐状态，形成矩阵 $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$
    - 要求变换矩阵满足 $\Delta \mathbf{H}_b = \mathbf{0}$，即对良性激活零扰动
    - 利用等价关系 $\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b \mathbf{H}_b^\top)$，将计算从 $d \times N_b$ 降至 $d \times d$
    - 对协方差矩阵做 SVD，选取近零特征值对应的特征向量构成投影矩阵：$\mathbf{P} = \hat{\mathbf{U}} \hat{\mathbf{U}}^\top$
    - **设计动机**：提供数学上的保证——良性输入的激活在转向前后完全不变

2. **恶意方向的转向学习**：让恶意输入被引导至拒绝语义

    - 收集 $N_m$ 个恶意输入的隐状态 $\mathbf{H}_m$
    - 目标：$\tilde{\Delta} \mathbf{P} \mathbf{H}_m = \mathbf{R}$（拒绝方向的目标激活）
    - 存在闭合解，无需迭代训练

3. **有害方向抑制项**：进一步消除越狱残余语义

    - 通过遮蔽视觉显著性 token 并测量激活变化来提取有害方向 $\mathbf{V}$
    - 在优化目标中加入 $\|\tilde{\Delta} \mathbf{P} \mathbf{H}_m - \mathbf{V}\|_F^2$ 来显式抑制越狱相关特征

### 损失函数 / 训练策略

NullSteer 的优化目标由三项组成，存在闭合解：

$$\tilde{\Delta}^* = \arg\min_{\tilde{\Delta}} \left( \|\tilde{\Delta}\mathbf{P}\mathbf{H}_m - \mathbf{R}\|_F^2 + \alpha\|\tilde{\Delta}\mathbf{P}\|_F^2 + \beta\|\tilde{\Delta}\mathbf{P}\mathbf{H}_m - \mathbf{V}\|_F^2 \right)$$

- 第一项：将恶意激活对齐到拒绝语义
- 第二项：正则化，确保变换的平滑性
- 第三项：抑制残余的越狱特征方向

闭合解通过 Moore-Penrose 伪逆直接求得，**完全无需梯度下降训练**。

## 实验关键数据

### 主实验

在三个 VLM（MiniGPT-4、Qwen2-VL、LLaVA-v1.5）上评估，对抗 PGD 扰动攻击：

| 模型 | 指标 | NullSteer | ASTRA (前SOTA) | 无防御 |
|------|------|-----------|---------------|--------|
| MiniGPT-4 (unconstrained) | Toxicity ↓ | **2.89%** | 4.48% | 52.12% |
| MiniGPT-4 (unconstrained) | ASR ↓ | **7.32%** | 9.09% | 53.64% |
| Qwen2-VL (ε=32/255) | Toxicity ↓ | **3.51%** | 5.45% | 51.62% |
| Qwen2-VL (ε=32/255) | ASR ↓ | **4.55%** | 5.00% | 70.46% |
| LLaVA-v1.5 (ε=32/255) | Toxicity ↓ | **31.82%** | 34.76% | 84.40% |
| LLaVA-v1.5 (ε=32/255) | ASR ↓ | **8.75%** | 10.91% | 56.36% |

通用能力（Utility）保持：

| 模型 | MM-Vet | MMBench | XSTest |
|------|--------|---------|--------|
| MiniGPT-4 原始 | 19.40 | 35.90 | 87.60 |
| MiniGPT-4 NullSteer | **21.05** | **36.25** | **87.80** |
| Qwen2-VL 原始 | 49.13 | 78.00 | 73.60 |
| Qwen2-VL NullSteer | 49.02 | **78.82** | **74.50** |

### 消融实验

| 配置 | Toxicity ↓ | ASR ↓ | Utility ↑ | 说明 |
|------|-----------|-------|-----------|------|
| 无防御 | 30.65% | 34.55% | 35.90 | 基线 |
| 正则化+拒绝对齐 | 3.58% | 8.36% | 36.00 | 缺少有害抑制 |
| 正则化+有害抑制 | 4.02% | 8.57% | 36.00 | 缺少拒绝对齐 |
| 全部三项 | **2.89%** | **7.32%** | **36.25** | 三者互补 |

### 关键发现

- 仅需约 8 个良性样本即可构建稳定的零空间投影
- 约 100 个恶意样本即可充分估计有害方向
- 转向强度 λ≈5 时安全性与实用性达到最佳平衡
- 在自适应攻击（攻击者已知防御）下仍保持有效——Jailbreak ASR 从49.1%降至19.3%

## 亮点与洞察

1. **理论优雅**：将安全对齐问题转化为零空间约束优化，提供了良性表征不变性的数学保证，这在VLM防御中是首次
2. **无需训练**：整个方法存在闭合解，无需微调或梯度下降，推理时几乎不增加延迟
3. **选择性机制**：完美解决了传统激活转向的过度拒绝问题——良性输入的激活完全不受影响
4. **跨模型泛化**：在三种不同架构的VLM上均表现一致，说明零空间约束的普适性

## 局限与展望

1. 依赖于线性假设——假设良性和恶意激活分布可以通过线性子空间分离，对高度非线性的攻击可能失效
2. 零空间维度 r 的选择需要预定义，不同模型/层可能需要不同设置
3. 目前仅评估了基于扰动的攻击，对基于嵌入文本的视觉攻击（typography-based）的评估较有限
4. 转向层 l 的选择依赖人工经验（13B取20层、7B取14层），缺乏自适应选择机制

## 相关工作与启发

- **AlphaEdit**：在LLM知识编辑中首次将零空间投影用于保护已有知识，是本文的核心灵感来源
- **ASTRA**：本文的主要比较基线，使用自适应激活转向但缺乏零空间约束
- 零空间约束在持续学习中已被广泛验证（GNSP、NS-Net等），本文将其首次引入VLM安全对齐领域
- 启示：零空间投影提供了一种通用的"选择性控制"范式，可能扩展到更多需要"改变某些行为但保留其他行为"的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 零空间投影用于VLM安全是新的组合，但零空间本身在CL/知识编辑中已广泛使用
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个模型、多种攻击强度、自适应攻击、完整消融
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰、动机阐述到位
- 价值: ⭐⭐⭐⭐ — 提供了VLM安全防御的实用且理论可解释的方案

<!-- RELATED:START -->

## 相关论文

- [AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](../../ICLR2026/llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)
- [Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](../../ICLR2026/llm_alignment/mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

<!-- RELATED:END -->
