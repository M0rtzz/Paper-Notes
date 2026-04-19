---
title: >-
  [论文解读] Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection
description: >-
  [CVPR 2026][多模态][VLM安全] 提出 CARE 框架，先用因果中介分析精确定位 VLM 中与不安全行为因果相关的神经元和层（诊断），再通过广义特征分解构建双模态安全子空间并在推理时投影激活值（修复），将攻击成功率降至 10% 以下且几乎不损失通用能力。
tags:
  - CVPR 2026
  - 多模态
  - VLM安全
  - 因果中介分析
  - 安全子空间投影
  - 对抗攻击防御
  - 双模态修复
---

# Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection

**会议**: CVPR 2026  
**arXiv**: [2603.27240](https://arxiv.org/abs/2603.27240)  
**代码**: 无  
**领域**: Multimodal / VLM  
**关键词**: VLM安全, 因果中介分析, 安全子空间投影, 对抗攻击防御, 双模态修复

## 一句话总结
提出 CARE 框架，先用因果中介分析精确定位 VLM 中与不安全行为因果相关的神经元和层（诊断），再通过广义特征分解构建双模态安全子空间并在推理时投影激活值（修复），将攻击成功率降至 10% 以下且几乎不损失通用能力。

## 研究背景与动机
**领域现状**: 大型视觉语言模型 (LVLM) 在多模态理解上表现出色，但面临越狱攻击（jailbreak）——精心构造的多模态提示可绕过安全对齐机制。

**现有痛点**: (1) 输入预处理和对抗训练计算昂贵且可能降低通用性能；(2) 现有激活层面防御（ASTRA, SPO-VLM）缺乏对不安全组件的精确定位，仅使用单一模态，且启发式的线性转向会扭曲通用表征。

**核心矛盾**: 如何精确定位 VLM 内部与不安全行为相关的组件，并在不损害通用能力的前提下修复它们？

**本文目标**: 建立一个因果驱动的、非线性的、双模态的 VLM 安全诊断与修复框架。

**切入角度**: 先因果定位（哪些层/神经元导致不安全输出），再子空间投影（将激活投影到安全方向）。

**核心 idea**: 诊断——因果中介分析定位 FFN 中层 → 修复——广义特征分解找到恶意子空间 → 投影到其正交补空间。

## 方法详解

### 整体框架
三步：(1) 层级因果追踪（ablation）→ (2) 双模态 token 归因（RBF 核）→ (3) 安全子空间投影（广义特征分解 + 正交投影）。整个过程无需重训练，仅在推理时介入。

### 关键设计
1. **层级因果追踪与组件分析**: 通过系统性地 block 不同层，观察攻击成功率 (ASR) 变化来粗粒度定位安全关键层。进而分别 ablate FFN 和 MHSA，发现 **FFN 对安全性的影响远大于 MHSA**。设计动机：FFN 激活展示低样本间相关性（每个样本独立投影），使得安全信号更容易分离；而 MHSA 整合全局上下文，安全信号被扩散难以隔离。量化验证了 Silhouette 系数、类分离度、Mahalanobis 距离三个指标在第 16-17 层达到峰值。

2. **双模态 Token 归因**: 

    - **视觉归因**: 使用 RBF 核计算视觉-文本 token 间的跨模态相关性 $MI_i^v = \frac{s_i - s_{min}}{s_{max} - s_{min}}$，其中 $s_i = \|\tilde{K}_{i,:}\|_2^2$ 是居中跨模态核矩阵行的 L2 范数。选出与攻击最相关的 top-k 视觉 token。
    - **文本归因**: 使用自模态 RBF 核矩阵计算文本 token 的语义独立性分数，选出最有影响力的文本 token。
   设计动机：不是所有 token 都与越狱行为等价相关，聚焦于高归因 token 使后续投影更精确。

3. **安全子空间投影**: 收集良性和恶意样本在目标层的激活 $A_b, A_m$，中心化后计算协方差矩阵 $C_b, C_m$，通过广义特征分解 $C_m u = \lambda C_b u$ 找到恶意偏离最大的方向。取 top-k 特征向量 $U_k$ 构建恶意子空间，构造安全投影算子：
    $P_{\text{safe}} = I - U_k U_k^T$
   推理时投影：$h' = P_{\text{safe}} h + \beta (1 - P_{\text{safe}}) h_{\text{benign}}$
   
   对视觉和文本模态分别构建安全子空间，通过自适应融合权重 $w_{vis} = \frac{\|h'_{vis} - h_{txt}\|}{\|h'_{vis} - h\| + \|h'_{txt} - h\|}$ 组合两个模态的投影结果。设计动机：广义特征分解能找到恶意激活偏离良性激活最大的方向，投影到其正交补可以精确抑制不安全成分。双模态分别投影再融合，因为视觉和文本攻击的机制不同。

### 损失函数 / 训练策略
**无需训练**——完全在推理时介入。仅需离线收集少量良性/恶意样本提取激活用于构建投影矩阵。

## 实验关键数据

### 主实验（攻击成功率 ASR % ↓）
| 方法 | JailBreakV | MMSafety | PGD-Toxic κ=64 | PGD-Jailbreak κ=64 |
|------|-----------|---------|---------------|-------------------|
| LLaVA 原始 | 45.71 | 36.48 | 60.38 | 65.15 |
| SPO-VLM | 10.37 | 16.26 | 17.90 | 17.38 |
| ASTRA | 11.98 | 15.37 | 16.37 | 14.85 |
| **CARE (Ours)** | **7.03** | **9.13** | **12.78** | **8.46** |

**Qwen2.5-VL 上类似趋势**：JailBreakV 6.55%, MMSafety 8.72%

### 消融实验
| 配置 | JailBreakV ASR↓ | PGD-Toxic-64 ASR↓ | 说明 |
|------|----------------|-------------------|------|
| CARE (full) | 7.03 / 6.55 | 12.78 / 4.60 | 双模态完整版本 |
| CARE w/o text | 15.26 / 14.3 | — | 文本子空间对语言越狱关键 |
| CARE w/o visual | — | 45.71 / 46.13 | 视觉子空间对图像攻击关键 |

### 关键发现
- **FFN > MHSA**: block FFN 对 ASR 影响远大于 block MHSA，证实 FFN 是安全机制的主要载体
- **中间层最关键**: 安全相关表征在第 16-17 层（LLaVA）或 12-14 层（Qwen）达到峰值聚类分离度
- **双模态缺一不可**: 去掉文本子空间→语言越狱 ASR 翻倍；去掉视觉子空间→PGD 攻击 ASR 翻 10 倍
- **通用能力保持**: MMBench、MM-Vet、SQA 上仅 2-8% 的性能下降
- **迁移防御**: 对未见过的 PGD 攻击也有效

## 亮点与洞察
- **因果驱动的精确定位**：不是盲目地在所有层做干预，而是先定位安全关键层和组件（FFN），减少对无关表征的干扰。
- **广义特征分解**的理论优雅性：直接在"良性 vs 恶意"的协方差空间中找到最大偏离方向，比启发式的线性转向更有原理性。
- **无需训练**：仅需离线提取少量激活，推理时做矩阵乘法，开销极低。
- **FFN 的"判别投影器"角色**：FFN 激活低样本间相关性意味着安全信号在其中更"纯净可分"，这一发现对理解 VLM 内部安全机制具有理论价值。

## 局限与展望
- 安全子空间的构建依赖于离线收集的恶意样本，可能对全新类型的攻击覆盖不足
- 投影操作虽然轻量但在每次推理时增加了计算开销
- 良性正则化项 $\beta$ 需要调节，不同模型可能需要不同超参数
- 仅在 7-8B 规模的模型上验证，更大规模模型是否有相同安全机制有待验证
- 安全子空间的维度 $k$ 选择需要经验调参
- 对纯文本越狱（无图像输入）的防御效果未单独评估
- 安全机制在更深层被"特征纠缠"稀释的现象值得进一步研究

## 相关工作与启发
- 与 ASTRA、SPO-VLM 的区别：CARE 通过因果分析精确定位而非粗粒度干预，使用非线性 RBF 核和广义特征分解而非线性转向。
- 与 Refusal Pairs（微调方法）相比：CARE 无需重训练，且效果更好。
- 因果中介分析在 NLP 可解释性中已有应用，本文首次将其用于 VLM 安全定位。
- 广义特征分解也用于 LDA 等经典判别分析，本文将其创新性地用于安全/恶意方向的分离。
- FFN 的"判别投影器"角色与 Neural Collapse 现象的联系值得深入探索

## 技术细节补充
- **RBF 核带宽**: $\sigma = \sqrt{0.5 \cdot \text{median}(D_{ij})}$，自适应于数据分布
- **Kernel 居中**: 视觉单侧 $\tilde{K} = K_{cross}H_t$，文本双侧 $\tilde{K} = HKH$
- **安全投影**: $h' = P_{safe}h + \beta(1-P_{safe})h_{benign}$
- **融合权重**: $w_{vis} = \frac{\|h'_{vis}-h_{txt}\|}{\|h'_{vis}-h\|+\|h'_{txt}-h\|}$
- **定位验证**: Silhouette/Class Sep./Mahalanobis 三指标在层 16-17 峰值
- **攻击数据**: JailbreakVBench + AdvBench + FigStep
- **通用性能**: MMBench 降 2-3%, MM-Vet 降 5-8%, SQA 降 2-4%

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 因果诊断+双模态安全子空间投影的组合是首创
- 实验充分度: ⭐⭐⭐⭐⭐ 2 个 VLM × 多 benchmark × PGD 攻击 × 消融全面
- 写作质量: ⭐⭐⭐⭐ 框架清晰，因果分析部分深入，但数学符号较密
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 安全防御具有重要实践意义，无需重训练即可部署

<!-- RELATED:START -->

## 相关论文

- [HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](ssr2gcd_rate_reduction_category_discovery.md)
- [Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)

<!-- RELATED:END -->
