---
title: >-
  [论文解读] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation
description: >-
  [ICLR 2026][test-time adaptation] 提出 IMSE——将预训练 ViT 线性层通过 SVD 分解为"谱专家"，仅微调奇异值实现极端参数高效的测试时适应，并通过多样性最大化损失和域感知谱码检索机制，在 TTA/CTTA/渐进 CTTA 三种场景下达到 SOTA。
tags:
  - ICLR 2026
  - test-time adaptation
  - singular value decomposition
  - mixture of experts
  - continual adaptation
  - distribution shift
---

# IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation

**会议**: ICLR 2026  
**arXiv**: [2603.07926](https://arxiv.org/abs/2603.07926)  
**代码**: [github](https://github.com/baek85/IMSE)  
**领域**: llm_efficiency  
**关键词**: test-time adaptation, singular value decomposition, mixture of experts, continual adaptation, distribution shift

## 一句话总结

提出 IMSE——将预训练 ViT 线性层通过 SVD 分解为"谱专家"，仅微调奇异值实现极端参数高效的测试时适应，并通过多样性最大化损失和域感知谱码检索机制，在 TTA/CTTA/渐进 CTTA 三种场景下达到 SOTA。

## 研究背景与动机

测试时适应（TTA）旨在让源域预训练模型在线适应未知目标域，无需访问源数据。现有方法面临三个关键问题：

**预训练特征利用不充分**：大型预训练模型蕴含丰富的表征能力，如何在最少参数更新下充分利用这些表征仍未被充分探索。现有方法要么只调 BN 参数（适应能力有限），要么引入额外模块（增加推理开销）。

**熵最小化导致特征坍缩**：在无标签 TTA 场景中，熵最小化常常驱使模型利用域特定特征而非类别判别特征，反而加剧性能退化。

**连续 TTA 中域知识遗忘**：在 CTTA 设定下，模型不仅需要保持预训练知识，还需保留并复用先前遇到的域知识。现有方法缺乏高效的域知识保存与复用机制。

## 方法详解

### 整体框架

IMSE 包含三个核心组件，围绕"将线性层视为内在谱专家混合"这一思想展开：

- **内在谱专家混合**：对每个线性层做 SVD 分解，将秩-1 分量视为独立谱专家。
- **多样性最大化损失**：鼓励谱专家的多样化使用，抵抗熵最小化导致的特征坍缩。
- **域感知谱码检索（IMSE-Retrieval）**：在 CTTA 中存储并复用已适应的奇异值，实现快速域切换适应。

### 关键设计

#### 谱专家与谱码

对第 l 层线性变换做 SVD 分解：W = U Sigma V^T = sum_i sigma_i u_i v_i^T。每个秩-1 分量被视为第 i 个**谱专家**。由于奇异向量相互正交，不同专家输出也正交。**谱码**定义为所有层奇异值的集合。

适应时**仅调整奇异值**（冻结正交基），既保留预训练特征提取器的子空间，又通过调整各专家贡献权重来适应新域。

#### 专家-输入对齐统计量

为量化特征坍缩，定义第 i 个专家与输入的归一化对齐 a = v_i^T x / ||x||。计算均值和标准差 Std_i。低标准差表明该专家捕获域特定模式而非类别判别特征。

#### 域感知谱码检索（CTTA 专用）

维护**域银行**，存储 [域描述符, 谱码] 对。域描述符通过 patch token 的通道均值/方差的 EMA 累积。域切换检测采用对称 KL 散度，超过阈值时存储当前谱码并检索最相似历史域的谱码初始化。

### 损失函数 / 训练策略

**多样性最大化损失**：L_dm = -sum_l (1/r) sum_i Std_i^(l)，最大化对齐标准差。

总损失：L_IMSE = L_entmin + lambda_dm * L_dm

同时采用锐度感知最小化（SAM）增强稳定性。选择靠近分类头的后部层应用多样性约束。

## 实验关键数据

### 主实验

**ImageNet-C (50k) 单域 TTA**（ViT-Base，severity 5）：

| 预训练策略 | 方法 | 平均准确率 (%) |
|:---:|:---:|:---:|
| Supervised | DPAL | 67.0 |
| Supervised | **IMSE** | **69.0** |
| MAE | DPAL | 65.9 |
| MAE | **IMSE** | **68.3** |
| CLIP | DPAL | 62.3 |
| CLIP | **IMSE** | **65.5** |

三种预训练策略下均超越前 SOTA DPAL，MAE/CLIP 上提升 2.4/3.2 百分点。

**ImageNet-R / ImageNet-A**：

| 方法 | ImageNet-R | ImageNet-A |
|:---:|:---:|:---:|
| DPAL | 64.8 | 49.9 |
| **IMSE** | **69.8** | **54.8** |

分别提升 5.0pp 和 4.9pp。

### 消融实验

**CTTA 设定**（ImageNet-C, 15 域连续适应）：IMSE-Retrieval 比 ViDA 提升 3.4pp，可训练参数量仅为 ViDA 的 **1/385**。渐进 CTTA（135 域）提升 2.4pp。

### 关键发现

1. **参数效率极致**：仅微调奇异值即可多种预训练策略下达到 SOTA
2. **多样性损失有效抗坍缩**：加入 L_dm 后各专家使用更均衡
3. **域银行机制实用**：谱码紧凑，存储开销极小
4. **跨预训练策略通用**：Supervised/MAE/CLIP 均有效

## 亮点与洞察

- 🔍 **谱专家视角新颖**：SVD 秩-1 分量重释为"专家"，无需额外架构，利用预训练权重内在结构
- 💡 **特征坍缩的量化与解决**：首次从谱专家对齐统计量角度量化 TTA 中的特征坍缩
- 🔄 **谱码紧凑性天然适配检索**：存储与检索成本极低
- ⚡ **参数量极低**：比 ViDA 少 385 倍可训练参数

## 局限与展望

1. **SVD 分解一次性计算开销**：初始 SVD 分解有一定成本
2. **域描述符鲁棒性**：极端域漂移或小批量时可能不够鲁棒
3. **分类任务限定**：拓展到检测/分割需额外设计
4. **骨干兼容性**：主要验证 ViT，CNN 适配性待验证

## 相关工作与启发

| 方法 | 策略 | 与 IMSE 的区别 |
|:---:|:---:|:---:|
| TENT | BN affine + 熵最小化 | 仅调 BN，适应能力有限 |
| SAR | 锐度感知 + 样本过滤 | IMSE 增加谱专家和多样性损失 |
| DPAL | 域提示 + 适配器 | 引入额外模块；IMSE 无额外结构 |
| ViDA | 视觉域适配器 | IMSE 参数量为其 1/385 |
| SVFT/SVDiff | 仅调奇异值 | 聚焦 LLM/Diffusion；IMSE 首次用于 TTA |

核心启发：预训练大模型权重矩阵蕴含功能分化的"内在专家"结构，通过 SVD 即可揭示和利用。

## 评分

| 维度 | 分数 |
|:---:|:---:|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | ⭐⭐⭐⭐ |

> 扎实的 TTA 工作，谱专家视角新颖且具启发性，实验覆盖 TTA/CTTA/渐进 CTTA 三种场景和多种预训练策略，参数效率惊人。主要遗憾是任务类型限于分类。

<!-- RELATED:START -->

## 相关论文

- [Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](../../NeurIPS2025/code_intelligence/flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)
- [SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](../../ICML2025/code_intelligence/sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)

<!-- RELATED:END -->
