---
title: >-
  [论文解读] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs
description: >-
  [AAAI 2026][医学图像][策略违规检测] 将 LLM 的策略违规检测重构为激活空间中的分布外（OOD）检测问题，提出无需训练的白化方法：对合规激活拟合白化变换，用欧几里得范数作为合规分数，仅需策略文本和少量示例即可部署，在 DynaBench 上达到 86.0% F1，超越微调基线 9.1 个点、LLM-as-Judge 16 个点。
tags:
  - AAAI 2026
  - 医学图像
  - 策略违规检测
  - 激活空间白化
  - 分布外检测
  - LLM 内部表示
  - 无需训练
---

# Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs

**会议**: AAAI 2026  
**arXiv**: [2512.03994](https://arxiv.org/abs/2512.03994)  
**代码**: [GitHub](https://github.com/orgs/FujitsuResearch/repositories)  
**领域**: 医学图像  
**关键词**: 策略违规检测, 激活空间白化, 分布外检测, LLM 内部表示, 无需训练

## 一句话总结

将 LLM 的策略违规检测重构为激活空间中的分布外（OOD）检测问题，提出无需训练的白化方法：对合规激活拟合白化变换，用欧几里得范数作为合规分数，仅需策略文本和少量示例即可部署，在 DynaBench 上达到 86.0% F1，超越微调基线 9.1 个点、LLM-as-Judge 16 个点。

## 研究背景与动机

- **企业 LLM 部署的合规挑战**：组织在法律、金融、医疗等敏感领域部署 LLM 时，必须确保符合内部组织策略和外部监管要求
    - 企业策略通常包含数十条策略，每条策略可能有数百条规则
    - 即使高性能 LLM 也可能无意违反组织策略，造成法律和财务风险
- **现有方法的局限**：
    - **Guardrail 系统**（LlamaGuard 等）：局限于安全分类法，无法泛化到复杂组织策略
    - **LLM-as-a-Judge**：灵活但引入显著延迟（1.47 秒/样本）
    - **微调检测器**（DynaGuard）：需要大量标注数据和训练成本，适应性差
    - 以上方法均在**生成文本层面**评估合规性
- **核心洞察**：LLM 的内部激活状态编码了关于输出正确性的信息，这些信息**未完全反映在生成的 token 中**
- **假设**：策略违规状态在 LLM 嵌入空间中占据不同区域，可通过 OOD 检测方法识别

## 方法详解

### 核心思路

将策略合规检测建模为激活空间中的 OOD 问题：
- **合规行为** → 分布内（in-distribution）
- **违规行为** → 分布外（out-of-distribution）

### 离线阶段：参考统计预处理

#### 白化变换

从合规（in-policy）交互中提取每层激活向量 $\{x_i^{(\ell)}\}_{i=1}^N$，计算经验均值和协方差：

$$\mu^{(\ell)} = \frac{1}{N} \sum_{i=1}^N x_i^{(\ell)}, \quad \Sigma^{(\ell)} = \frac{1}{N-1} \sum_{i=1}^N (x_i^{(\ell)} - \mu^{(\ell)})(x_i^{(\ell)} - \mu^{(\ell)})^\top$$

白化矩阵 $W^{(\ell)}$ 满足 ${W^{(\ell)}}^\top W^{(\ell)} = (\Sigma^{(\ell)})^{-1}$，通过 PCA 白化计算。

白化后的表示：

$$y^{(\ell)} = W^{(\ell)}(x^{(\ell)} - \mu^{(\ell)})$$

#### 合规分数

在白化空间中，偏离合规行为通过**欧几里得范数**量化：

$$s^{(\ell)} = \|y^{(\ell)}\|_2$$

此分数等价于原始空间中的 Mahalanobis 距离，但聚焦于合规变异的主要方向。

#### 层选择

每层独立计算白化参数，使用小规模混合（合规+违规）样本评估各层分离度，选择最佳操作层 $\ell^\star$。

#### 阈值校准

在操作层 $\ell^\star$ 上，通过最大化 Youden 统计量（$J = TPR - FPR$）校准决策阈值 $\tau$。

#### 策略条件白化

将策略分组为具有共享行为模式的类别，为每个类别估计独立的白化参数，实现类别特异性检测。

### 在线阶段：实时检测

每个响应在返回前验证：

$$\hat{y} = \mathbb{I}[s^{(\ell^\star)} > \tau]$$

$\hat{y}=1$ 表示违规，$\hat{y}=0$ 表示合规。使用策略分组时，通过余弦相似度选择最近的策略类别。

### 对比数据构建

- 为 DynaBench 每条策略规则，使用 GPT-5.1 生成自然语言 prompt
- 为每个 prompt 生成对比样本对（合规 good + 违规 bad）
- 使用 GPT-5.1 验证器确保数据质量

## 实验

### 基准与设置

| 基准 | 描述 |
|------|------|
| DynaBench | 多轮用户-Agent 对话的策略合规评估，12 个商业影响类别 |
| τ-bench | AI Agent 的工具调用正确性评估（航空领域） |

评估模型：Mistral-7B、Gemma-2-9B、Llama-3.1-8B、Qwen3-8B、Qwen2.5-7B

### 主实验结果（DynaBench）

| 方法类别 | 模型 | F1 (%) |
|----------|------|--------|
| LLM-as-Judge | GPT-4o-mini | 70.1 |
| LLM-as-Judge | Qwen3-8B | 60.7 |
| 微调 | LlamaGuard-3 | 20.9 |
| 微调 | DynaGuard-8B | 73.1 |
| **白化（Ours）** | Mistral-7B | 66.8 |
| **白化（Ours）** | Gemma-2-9B | 75.2 |
| **白化（Ours）** | Llama-3.1-8B | 75.6 |
| **白化（Ours）** | Qwen3-8B | 78.4 |
| **白化（Ours）** | **Qwen2.5-7B** | **86.0** |

- 在 5 个骨干中的 4 个上达到 SOTA
- Qwen2.5-7B 达到 86.0% F1，超越最强微调基线 DynaGuard-8B 12.9 个点
- 无需任何微调

### 表示级 vs 生成级分析

| 模型 | 生成分类器 F1 | 白化 F1 |
|------|-------------|---------|
| DynaGuard-1.7B | 65.2 | **77.6** |
| DynaGuard-4B | 72.0 | **78.5** |
| DynaGuard-8B | 73.1 | **80.6** |

关键发现：对同一微调模型，白化方法比其原生生成分类器提升 5-12 个点。说明**模型内部表示编码了比输出 token 更丰富的策略相关信息**。

### τ-bench 泛化

- **合成数据**：白化方法大幅超越 DynaGuard-8B 和 GPT-4o-mini
- **真实轨迹**：白化方法 AUC=0.87，激活空间分离泛化到不同交互格式

### 与其他 OOD 方法对比

| 方法 | Qwen2.5-7B F1 | Llama-3.1-8B F1 |
|------|-------------|----------------|
| Mahalanobis | 67.2 | 65.8 |
| KNN | 78.5 | 66.2 |
| Energy Score | 66.4 | 72.1 |
| **白化（Ours）** | **82.2** | **74.3** |

白化方法超越最强 OOD 基线 3.7%/2.2%。Mahalanobis 距离在高维设置下受全协方差矩阵限制。

### 运行时效率

| 类别 | 模型 | 时间（秒/样本）|
|------|------|---------------|
| LLM-as-Judge | GPT-4o-mini | 1.47 |
| 微调检测器 | DynaGuard-8B | 2.71 |
| **同模型表示** | **Qwen2.5-7B** | **0.03** |
| 代理模型表示 | Llama-3.1-8B | 0.98 |

使用内部表示时仅增加 0.03-0.05 秒开销，适合实时监控。

### 消融实验

- **Top-K 成分**：K=10-50 范围内 F1 波动仅 72.4%-76.7%，鲁棒性强
- **每类样本数**：100 样本达 75.6%，增至 750 样本仅提升至 79.1%（边际收益递减）
- **逐层分析**：不同策略类别的最佳层不同（如信息泄露偏早期层，交易类偏中后期层）
- **类别特异白化** vs 统一白化：类别特异始终更优

## 亮点与洞察

1. **OOD 框架重构策略违规检测**：从生成文本分析转向激活空间分析，是范式级创新
2. **核心发现**：策略合规信息已编码在模型内部表示中，解码过程是有损瓶颈——白化只是暴露了预存的结构
3. **0.03 秒/样本**的推理开销使实时部署成为可能，比 LLM-as-Judge 快 50 倍
4. **无需训练，极少校准数据**（约每条规则 1 个样本）：支持策略快速更新
5. 模型安全基准得分（SORRY-Bench、HarmBench）与白化分离度正相关，建立了安全性与内部表示质量的联系

## 局限性

- 性能依赖模型内部表示的质量——安全意识弱的模型（如 Mistral-7B）分离度较低
- 需要校准阈值，分布漂移时可能需要重新校准
- **仅检测不预防**：不直接干预生成过程
- 需要访问模型内部激活（或使用代理模型），对纯 API 模型需额外推理
- 对比数据由 GPT-5.1 生成，可能引入生成偏差
- DynaBench 基准的策略复杂度和真实世界策略的差距未充分讨论

## 相关工作

- **Guardrail 系统**：LlamaGuard（Inan et al. 2023）、NeMo Guardrails
- **策略合规**：DynaBench/DynaGuard（Hoover et al. 2025）
- **OOD 检测**：Mahalanobis 距离、Energy Score、KNN、白化变换（Betser et al. 2025）
- **LLM 内部表示分析**：Zou et al. 2023（真实性探针）、Gekhman et al. 2025（错误检测）
- **激活空间控制**：SCANS（激活引导缓解过度安全）

## 评分 ⭐⭐⭐⭐⭐

思路极其优雅——将复杂的策略违规检测简化为白化空间中的范数计算。无需训练、极快推理、少量数据校准的特性使其高度实用。实验全面且令人信服（5 个模型、2 个基准、多种 OOD 对比、运行时分析）。核心发现（内部表示 > 生成输出）具有深远影响。是 LLM 治理和企业安全部署方向的标杆工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)
- [\[AAAI 2026\] From Policy to Logic for Efficient and Interpretable Coverage Assessment](from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](../../CVPR2025/medical_imaging/univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)

</div>

<!-- RELATED:END -->
