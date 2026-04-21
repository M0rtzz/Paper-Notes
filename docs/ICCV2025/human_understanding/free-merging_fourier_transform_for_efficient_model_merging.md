---
title: >-
  [论文解读] FREE-Merging: Fourier Transform for Efficient Model Merging
description: >-
  [ICCV 2025][人体理解][模型合并] 首次发现模型合并中任务干扰在频域上的表现，提出 FR-Merging 通过高通滤波去除低频干扰构建高质量合并骨干网络，并结合轻量级任务专家模块（FREE-Merging），在视觉、语言和多模态任务上实现性能-成本的最优平衡。
tags:
  - ICCV 2025
  - 人体理解
  - 模型合并
  - 傅里叶变换
  - 任务干扰
  - 频域分析
  - 轻量级专家
---

# FREE-Merging: Fourier Transform for Efficient Model Merging

**会议**: ICCV 2025  
**arXiv**: [2411.16815](https://arxiv.org/abs/2411.16815)  
**代码**: [GitHub](https://github.com/Zhenghe123/FREE-Merging)  
**领域**: 人体理解  
**关键词**: 模型合并, 傅里叶变换, 任务干扰, 频域分析, 轻量级专家

## 一句话总结

首次发现模型合并中任务干扰在频域上的表现，提出 FR-Merging 通过高通滤波去除低频干扰构建高质量合并骨干网络，并结合轻量级任务专家模块（FREE-Merging），在视觉、语言和多模态任务上实现性能-成本的最优平衡。

## 研究背景与动机

随着大量开源微调模型的涌现，模型合并（Model Merging）成为将多个任务特定模型整合为单一多任务模型的高效方法，可避免多任务联合训练的高成本和数据隐私问题。然而，现有方法面临两个核心挑战：

**任务干扰导致性能下降**：不同任务的微调权重之间存在冲突。现有方法（如 Task Arithmetic、Ties-Merging、DARE 等）仅在空间域操作（剪枝、符号解冲突等），忽略了频域干扰。本文首次揭示：任务干扰在频域上表现显著且集中在低频区域，而空间域方法几乎无法缓解频域干扰（仅降低 1-5% 的频域振幅方差，而 FR-Merging 可降低 20-24%）。

**性能与部署成本的矛盾**：引入任务专家可提升性能，但现有方法（EMR-Merging、Twin-Merging）需要存储大量任务特定知识（2-3% 参数），且忽视骨干网络优化。

本文的核心洞察是：低频信号捕捉全局结构信息，更可能包含任务特定信息从而导致任务间干扰；高频信号表示细粒度变化，具有更强的泛化能力。因此，直接滤除低频部分可以在保持性能的同时显著减少任务干扰。

## 方法详解

### 整体框架

FREE-Merging 是两阶段方法：
- **第一阶段 FR-Merging**（免训练）：对每个任务向量 $v_k = \theta_k - \theta_{\text{pre}}$ 应用高通滤波，去除低频干扰信号，然后合并得到高质量骨干网络
- **第二阶段 Expert Extraction**（免训练）：从任务向量中提取仅占 ~1% 参数量的轻量级任务专家，推理时通过路由器动态分配

### 关键设计

1. **FR-Merging（频域高通滤波合并）**:

    - 功能：对每个任务向量进行傅里叶变换，滤除低频干扰区域后逆变换回来
    - 核心思路：对任务向量 $v(x,y)$ 应用理想高通滤波器：
    $G(x,y) = \mathcal{F}^{-1}\{H(\eta, \gamma) \cdot \mathcal{F}\{v(x,y)\}\}$
      其中 $H(\eta, \gamma) = \begin{cases} 1, & \sqrt{\eta^2 + \gamma^2} \geq D_0 \\ 0, & \sqrt{\eta^2 + \gamma^2} < D_0 \end{cases}$，$D_0$ 是截断频率。
      合并系数基于任务向量的均值归一化：
    $\lambda_i = \mathbb{E}(v_i) \left(\sum_{j=1}^{K} \mathbb{E}(v_j)\right)^{-1}$
    - 设计动机：微调后的权重在损失景观中占据不同位置，线性插值容易落入高损失区域。去除低频信号后模型差异减小，合并结果更可能落入损失盆地。实验验证去除低频仅带来微小的任务性能下降（对角线），但泛化能力（非对角线）显著提升。

2. **轻量级任务专家提取**:

    - 功能：从任务向量中选取变化最大的 top-d% 参数作为任务专家，仅需约 1% 参数量
    - 核心思路：选择幅度最大的参数并进行缩放：
    $e(v_i) = \mu_i M(v_i, d), \quad \mu_i = -\frac{\mathbb{E}(M(v_i, d)) \cdot \log(d)}{\lambda_i \cdot \mathbb{E}(v_i)}$
      其中 $M(v_i, d)$ 是 top-d% 的参数，$\mu_i$ 是确保输出一致性的缩放因子
    - 设计动机：有理论保证（Theorem 5.1）合并模型无法在不引入额外信息的情况下同时保留所有原始模型能力（No Free Lunch）。低频信号包含任务特定信息，直接保存低频区域需要每次推理做逆 FFT，不切实际，因此用参数幅度近似替代。

3. **MoE 路由器动态分配**:

    - 功能：推理时根据输入动态选择激活的任务专家
    - 核心思路：$\theta_* = \theta_m + \sum_{i=1}^{K} w_i e_i$，其中权重 $[w_1, \ldots, w_K] \leftarrow \arg\max(R(x))$，$R$ 是轻量级 MLP 路由器
    - 设计动机：受 MoE 启发，动态路由避免了为每个输入加载所有专家的开销

### 损失函数 / 训练策略

FR-Merging 和专家提取都是**完全免训练**的，只需一次性计算。路由器可使用简单的 MLP 或其他分类器实现。整体流程仅在合并时执行一次 FFT 操作，推理时仅增加一个轻量级路由器。

## 实验关键数据

### 主实验

8 个视觉任务合并（ViT-B/32 / ViT-L/14 平均准确率）：

| 方法 | 额外开销 | ViT-B/32 Avg | ViT-L/14 Avg | 备注 |
|------|---------|-------------|-------------|------|
| Individual | - | 90.5 | 94.2 | 上界 |
| Task Arithmetic | 无 | 70.1 | 84.5 | 基线 |
| Ties-Merging | 无 | 73.6 | 86.0 | |
| PCB-Merging | 无 | 75.8 | 86.9 | 前SOTA(免训练) |
| **FR-Merging** | **无** | **78.1** | **88.3** | **+2.3/+1.4** |
| EMR-Merging | 3%存储 | 87.7 | 92.8 | |
| Twin-Merging | 2%存储 | 87.8 | 92.7 | |
| **FREE-Merging** | **1%存储** | **89.7** | **93.7** | **最低存储，最高性能** |

### 消融实验

频域干扰量化（ViT-B/32 振幅方差）：

| 方法 | 频域振幅方差 | 方差降低 | 说明 |
|------|-----------|---------|------|
| Task Arithmetic | 0.059 | - | 基线 |
| DARE | 0.057 | ↓3% | 空间域方法 |
| Ties-Merging | 0.058 | ↓2% | 空间域方法 |
| PCB-Merging | 0.056 | ↓5% | 空间域方法 |
| **FR-Merging** | **0.045** | **↓24%** | 频域方法效果显著 |

跨领域验证（语言模型，RoBERTa / T0-3B / Qwen-14B 平均）：

| 方法 | RoBERTa | T0-3B | Qwen-14B | 平均 |
|------|---------|-------|----------|------|
| Task Arithmetic | 66.65 | 63.91 | 66.40 | 65.65 |
| FR-Merging | 70.02 | 66.88 | 68.00 | 68.30 |
| EMR-Merging | 74.20 | 67.11 | 70.98 | 70.76 |
| FREE-Merging | **80.16** | **68.68** | **72.78** | **73.87** |

### 关键发现

- FR-Merging 在 30 个视觉任务合并中，免训练方法从 48.88%（Task Arithmetic）提升至 53.90%，FREE-Merging 达到 79.67%
- 高通滤波在提升泛化性（off-diagonal）的同时仅造成微小的任务性能损失（diagonal），这验证了低频信号主要编码任务特定信息
- FREE-Merging 仅需 1% 额外参数存储，但性能超越需要 2-3% 的 EMR-Merging 和 Twin-Merging
- 方法对 PEFT（LoRA、IA³）同样有效，展现了良好的泛化性

## 亮点与洞察

1. **频域视角的突破性发现**：首次将频域分析引入模型合并领域，揭示了任务干扰在低频区域集中的现象，为理解模型合并提供了新的理论视角
2. **极简且有效**：FR-Merging 仅需一次 FFT 操作，理论上时间复杂度 $O(nm\log m)$，无需任何训练数据或梯度计算
3. **理论保证**：Theorem 5.1 严格证明了"合并无免费午餐"定理，从理论上论证了引入任务专家的必要性
4. **跨模态泛化**：在 CV、NLP、多模态三类任务上都验证了有效性，包括从 ViT 到 LLaMa 的不同规模模型

## 局限与展望

- 理想高通滤波器存在振铃效应，可尝试 Butterworth 或 Gaussian 高通滤波器以获得更平滑的过渡
- 截断频率 $D_0$ 是固定超参数，不同层/不同任务可能需要不同的截断策略
- 路由器的准确性直接影响 FREE-Merging 的性能，在任务边界模糊时可能失效
- 在超大规模合并（如 30+ 模型）中，专家数量线性增长仍是存储负担

## 相关工作与启发

- 频域分析思路可推广到其他参数空间操作，如知识蒸馏、模型压缩中的频域剪枝
- 轻量级专家提取方法可与 LoRA、Adapter 等 PEFT 方法结合，实现更高效的多任务部署
- "低频=任务特定、高频=泛化能力" 的洞察对理解微调过程中参数变化的本质有启发

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将频域分析引入模型合并，发现低频干扰现象极具开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖视觉/语言/多模态，full/PEFT，8/30个任务合并等多种设定
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验对比公平系统，图示直观
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，免训练方法降低了模型合并的门槛，对大模型部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/human_understanding/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/human_understanding/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [IDFace: Face Template Protection for Efficient and Secure Identification](idface_face_template_protection_for_efficient_and_secure_identification.md)

<!-- RELATED:END -->
