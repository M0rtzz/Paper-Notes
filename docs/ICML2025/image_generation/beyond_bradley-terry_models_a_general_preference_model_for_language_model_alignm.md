---
title: >-
  [论文解读] Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment
description: >-
  [ICML 2025][图像生成][偏好建模] 提出偏好嵌入（Preference Embedding）——将响应嵌入到多维潜空间中捕捉复杂偏好结构（包括不可传递偏好），实现 $O(K)$ 的查询复杂度（与 BT 模型相同但表达力更强）…
tags:
  - "ICML 2025"
  - "图像生成"
  - "偏好建模"
  - "Bradley-Terry"
  - "偏好嵌入"
  - "不可传递偏好"
  - "奖励模型"
---

# Beyond Bradley-Terry Models: A General Preference Model for Language Model Alignment

**会议**: ICML 2025  
**arXiv**: [2410.02197](https://arxiv.org/abs/2410.02197)  
**代码**: [https://github.com/general-preference/general-preference-model](https://github.com/general-preference/general-preference-model)  
**领域**: 图像生成/LLM对齐  
**关键词**: 偏好建模, Bradley-Terry, 偏好嵌入, 不可传递偏好, 奖励模型

## 一句话总结
提出偏好嵌入（Preference Embedding）——将响应嵌入到多维潜空间中捕捉复杂偏好结构（包括不可传递偏好），实现 $O(K)$ 的查询复杂度（与 BT 模型相同但表达力更强），配合 General Preference Optimization (GPO) 在 RewardBench 和 AlpacaEval2.0 上超越 BT 奖励模型。

## 研究背景与动机

**领域现状**：建模人类偏好是基础模型对齐的核心。Bradley-Terry（BT）模型用标量奖励表示偏好，是 RLHF 的标准选择。监督配对偏好模型（PairRM/PairPM）直接拼接两个响应预测偏好。

**现有痛点**：
   - BT 模型假设偏好可由标量奖励决定——无法表达不可传递偏好（$A>B, B>C$ 但 $C>A$，如"石头剪刀布"式偏好循环）
   - PairRM/PairPM 虽表达力强但查询复杂度为 $O(K^2)$——需要评估所有 $K(K-1)/2$ 个配对
   - 人类偏好本质上是多维的、上下文依赖的——标量奖励过度简化
   - 在 LLM 推理时扩展（test-time scaling）中，需要对大量候选响应排序——$O(K^2)$ 成本不可承受

**核心矛盾**：表达力（BT:低, PairPM:高）与效率（BT: $O(K)$, PairPM: $O(K^2)$）的权衡。

**本文目标**：设计兼具高表达力和低查询复杂度的偏好模型。

**切入角度**：将每个响应嵌入到 $d$ 维潜空间（而非 1 维标量奖励），用嵌入向量间的关系表示偏好——$d$ 维嵌入比标量丰富得多，但查询仍为 $O(K)$（每个响应独立嵌入一次）。

**核心 idea**：偏好嵌入 = 高维奖励表示，偏好关系由嵌入间的非对称函数决定（如 $P(A>B) = \sigma(\phi(e_A, e_B))$），保持 $O(K)$ 效率但突破标量奖励的表达力瓶颈。

## 方法详解

### 整体框架
1. **General Preference Embedding Model (GPM)**：将每个响应独立嵌入到 $d$ 维空间
2. **偏好预测**：给定两个嵌入 $e_A, e_B \in \mathbb{R}^d$，用非对称函数计算 $P(A \succ B)$
3. **GPO (General Preference Optimization)**：基于偏好分数（而非标量奖励）的策略优化

### 关键设计

1. **偏好嵌入 (Preference Embedding)**:

    - 功能：将每个 (prompt, response) 对映射到 $d$ 维向量
    - 核心思路：给定 prompt $x$ 和响应 $y$，嵌入器 $E(x, y) = e \in \mathbb{R}^d$
    - 偏好计算：$P(y_1 \succ y_2 | x) = \sigma(e_1^T W e_2)$，其中 $W$ 是可学习的反对称矩阵（确保 $P(A>B) + P(B>A) = 1$）
    - 与 BT 的关系：当 $d=1$ 时退化为 BT 模型（$P(y_1 \succ y_2) = \sigma(r_1 - r_2)$）
    - 查询复杂度：$O(K)$——每个响应嵌入一次，然后 $O(K^2)$ 的偏好计算仅涉及嵌入间的简单矩阵运算（不需要重新推理模型）
    - 设计动机：嵌入是"一次计算、多次复用"——真正的计算瓶颈是模型推理（$O(K)$ 次），而非嵌入间的偏好计算

2. **不可传递偏好建模**:

    - 功能：证明 GPM 可以精确建模循环偏好
    - 核心思路：$d \geq 2$ 时，嵌入空间中可以存在 $e_A, e_B, e_C$ 使得 $A>B, B>C, C>A$——这在 $d=1$（BT）时不可能
    - 理论结论：BT 模型在循环偏好上的准确率等同于随机猜测——从数学上证明了 BT 的表达力限制
    - 设计动机：人类偏好中循环现象真实存在（如在不同维度上 A 优于 B 但在另一维度上 B 优于 A）

3. **GPO (General Preference Optimization)**:

    - 功能：用偏好分数（而非标量奖励）优化 LLM 策略
    - 核心思路：将 DPO/RLHF 中的标量奖励替换为偏好嵌入的分数——$\mathcal{L}_{\text{GPO}} = -\log \sigma(s(y_w) - s(y_l))$，其中 $s(y) = e_y^T W \bar{e}$（$\bar{e}$ 是参考方向）
    - 设计动机：GPM 的偏好信号比 BT 奖励更丰富→用于训练 LLM 时提供更精确的梯度

### 损失函数 / 训练策略
- GPM 训练：二元交叉熵损失 on 偏好对 $(y_w, y_l)$
- GPO 训练：类 DPO 的损失，但用偏好嵌入分数替代标量奖励
- 嵌入维度 $d$ 通常取 32-128（远多于 BT 的 1 维）
- 基架构：基于 LLaMA/Mistral 加嵌入投影头

## 实验关键数据

### 主实验
RewardBench 基准（偏好模型评估）：

| 模型 | 总分 ↑ | Chat | 安全 | 推理 | 循环偏好 |
|------|-------|------|------|------|---------|
| BT-RM (Mistral-7B) | 81.2 | 96.3 | 84.2 | 63.1 | ~50% (随机) |
| PairRM | 82.5 | 95.8 | 86.1 | 65.5 | 78.6% |
| **GPM (d=64)** | **84.3** | **96.8** | **87.5** | **68.5** | **96.8%** |

### AlpacaEval 2.0 下游评估（GPO 训练后）

| 训练方法 | LC Win Rate ↑ |
|---------|-------------|
| DPO + BT-RM | 29.5% |
| DPO + PairRM | 31.2% |
| **GPO + GPM** | **33.8%** |

### 消融实验

| 配置 | RewardBench 总分 | 循环偏好准确率 |
|------|----------------|-------------|
| d=1 (退化为 BT) | 81.2% | ~50% |
| d=4 | 82.5% | 82.3% |
| d=16 | 83.5% | 91.5% |
| d=64 | **84.3%** | **96.8%** |
| d=128 | 84.1% | 97.1% |

### 关键发现
- 嵌入维度 $d>1$ 是建模不可传递偏好的必要条件——BT ($d=1$) 在循环偏好上等同随机猜测（理论证实+实验验证）
- GPM 在标准偏好（可传递）和循环偏好上都优于 BT——高维嵌入不仅处理循环偏好，在一般偏好上也更精确
- $d=64$ 是实践中的好选择——继续增大收益递减
- GPO 在下游任务上优于 DPO+BT——更丰富的偏好信号→更精准的策略优化
- 查询效率与 BT 相同（$O(K)$ 推理）但效果显著更好

## 亮点与洞察
- **"偏好是多维的"**这个直觉终于得到了数学化和实证——人类在评判两个响应时确实在多个维度上权衡
- BT 模型在循环偏好上等同于随机猜测的理论结果是令人印象深刻的负面结果——揭示了一个被广泛使用的模型的根本局限
- $O(K)$ 的效率使 GPM 可以直接替代 BT 奖励模型——无需任何效率妥协
- 对 LLM 的 test-time scaling（如 Best-of-N 采样）有直接价值——更好的排序 = 更好的最终输出
- 偏好嵌入的思想可推广到其他需要偏好建模的领域（如推荐系统、多目标优化）

## 局限与展望
- 反对称矩阵 $W$ 的选择和初始化对性能有影响
- 循环偏好的真实数据稀缺——主要在合成数据上验证
- GPO 的训练复杂度与 DPO 相当，但需要先训练 GPM
- 多维嵌入的可解释性不如标量奖励——难以直接告诉用户"为什么 A 优于 B"
- 未探索嵌入维度的自适应选择

## 相关工作与启发
- **vs BT 奖励模型**: 标量表示→不可传递偏好上失败；GPM 多维表示→成功
- **vs PairRM/PairPM**: $O(K^2)$ 效率→不可扩展；GPM $O(K)$→可扩展
- **vs DPO**: 用标量奖励优化策略；GPO 用多维偏好分数→更丰富的信号
- **启发**：偏好建模中的维度选择类似于嵌入维度选择——1维太少，太多过拟合，需要找到甜点

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 偏好嵌入突破BT模型的根本局限
- 实验充分度: ⭐⭐⭐⭐⭐ RewardBench+循环偏好+下游AlpacaEval+消融
- 写作质量: ⭐⭐⭐⭐⭐ BT vs PairPM vs GPM的对比图极其清晰
- 价值: ⭐⭐⭐⭐⭐ 对LLM对齐的偏好建模有基础性推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond One-Hot Labels: Semantic Mixing for Model Calibration](beyond_one-hot_labels_semantic_mixing_for_model_calibration.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)
- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
