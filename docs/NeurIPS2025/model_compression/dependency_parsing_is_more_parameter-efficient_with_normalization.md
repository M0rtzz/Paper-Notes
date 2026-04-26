---
title: >-
  [论文解读] Dependency Parsing is More Parameter-Efficient with Normalization
description: >-
  [NeurIPS 2025][模型压缩][dependency parsing] 揭示依存句法/语义分析中 biaffine scoring 缺乏归一化导致模型过参数化，通过简单的 $1/\sqrt{d}$ 缩放即可在减少高达 85% BiLSTM 参数的同时匹配甚至超越原始性能。
tags:
  - NeurIPS 2025
  - 模型压缩
  - dependency parsing
  - biaffine scoring
  - normalization
  - parameter efficiency
  - overparameterization
---

# Dependency Parsing is More Parameter-Efficient with Normalization

**会议**: NeurIPS 2025  
**arXiv**: [2505.20215](https://arxiv.org/abs/2505.20215)  
**代码**: https://github.com/paolo-gajo/EfficientSDP  
**领域**: model_compression  
**关键词**: dependency parsing, biaffine scoring, normalization, parameter efficiency, overparameterization

## 一句话总结
揭示依存句法/语义分析中 biaffine scoring 缺乏归一化导致模型过参数化，通过简单的 $1/\sqrt{d}$ 缩放即可在减少高达 85% BiLSTM 参数的同时匹配甚至超越原始性能。

## 研究背景与动机

**领域现状**：依存分析（Dependency Parsing）的主流方法基于 Dozat & Manning 的 biaffine 分类器——用 BiLSTM 编码后对每对词计算 biaffine 分数来预测边和关系。当前 SOTA 通常堆叠 3 层 BiLSTM + hidden dim 400 + MLP dim 500。

**现有痛点**：biaffine scoring $QK^\top$ 与 Transformer 自注意力结构相同，但绝大多数依存分析工作都没有对 score 做归一化。Transformer 注意力用 $1/\sqrt{d_k}$ 缩放是为了控制方差避免 softmax 饱和——为什么依存分析不用？

**核心矛盾**：不做归一化 → 高方差输入导致 softmax 输出极化 → 梯度消失/爆炸 → 需要更多 BiLSTM 层的隐式正则化效果来补偿。这意味着额外的参数实际上是在弥补归一化的缺失，而非捕捉更丰富的语言特征。

**本文目标**：证明缺乏归一化导致过参数化，并用理论和实验说明加入归一化后可以大幅削减参数。

**切入角度**：从隐式正则化理论出发——深层线性网络在梯度下降中会降低权重矩阵的有效秩，而低秩权重的输出方差更小。因此更深的 BiLSTM 实际上通过降秩起到了类似归一化的作用。

**核心 idea**：直接加 $a = 1/\sqrt{d}$ 的 score 缩放，就能用 1 层 BiLSTM + 更小参数匹配 3 层无缩放的性能，参数减少高达 85%。

## 方法详解

### 整体框架
沿用 Bhatt et al. 的架构：BERT encoder（冻结）→ Tagger (单层 BiLSTM 预测词性) → Parser ($N$ 层 BiLSTM + 4 个 MLP 头 → biaffine scoring) → Decoder (贪心/MST 解码)。唯一修改：在 biaffine scoring 后加入 $1/\sqrt{d}$ 缩放。

### 关键设计

1. **Biaffine Score Normalization**:

    - 功能：将 biaffine score $s = Q K^\top$ 缩放为 $s / \sqrt{d}$，其中 $d$ 是 MLP 投影维度。
    - 理论依据：假设 $q, k$ 零均值单位方差，则 $\text{Var}(s) = d_k$，$\text{Std} = \sqrt{d_k}$。缩放后每个 score 的标准差约为 1，避免 softmax 输入极端化。
    - 设计动机：这正是 Transformer 注意力使用 $1/\sqrt{d_k}$ 的原因，但依存分析社区一直忽视了这一点。

2. **层深度隐式补偿归一化的理论分析**:

    - **Result 1 (隐式正则化)**：$N$ 层线性网络在梯度下降中，奇异值按 $\sigma_r(t+1) \leftarrow \sigma_r(t) - \eta \cdot \langle \nabla\mathcal{L}, \mathbf{u}_r \mathbf{v}_r^\top \rangle \cdot N \cdot \sigma_r(t)^{2-2/N}$ 演化。$N$ 越大，小奇异值衰减越快 → 有效秩越低。
    - **Claim 1 (秩与方差单调关系)**：截断 SVD 近似 $\mathbf{A}_r$ 的输出方差 $\text{tr}(\text{Cov}(Y_r))$ 随秩 $r$ 单调递增。
    - 推论：更深的 BiLSTM → 更低的有效秩 → 更低的输出方差 → softmax 更稳定。所以 3 层 BiLSTM 之所以比 1 层好，部分原因不是特征更丰富，而是变相做了归一化。有了显式归一化就不再需要这些"冗余"层。
    - 实验验证：Figure 1 展示有效秩 $\rho(W)$ 确实随 BiLSTM 层数增加而下降。

3. **参数高效配置搜索**:

    - 功能：在不同数据集上搜索最优 $(N, h_\psi, d_{\text{MLP}})$ 组合。
    - 发现：baseline 用 (3, 400, 500)，归一化后最优配置通常是 (1, 200-400, 100-300)，参数量减少 ~85%。

### 损失函数
标准多任务损失：$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{tag}} + \lambda_2 (\mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{rel}})$，$\lambda_1 = 0.1$，$\lambda_2 = 1$。

## 实验关键数据

### 主实验（有标签边预测，Micro-F1 / LAS）

| 模型 | 缩放 $a$ | 层数 $N$ | ADE | CoNLL04 | SciERC | ERFGC | enEWT | SciDTB |
|------|---------|---------|-----|---------|--------|-------|-------|--------|
| Baseline | 1 | 3 | 0.653 | 0.566 | 0.257 | 0.701 | 0.804 | 0.915 |
| Ours | $1/\sqrt{d}$ | 1 | 0.668 | 0.597 | 0.299 | 0.692 | 0.789 | 0.904 |
| Ours | $1/\sqrt{d}$ | 2 | 0.676 | 0.596 | 0.312 | 0.699 | 0.805 | 0.916 |
| **Ours** | $1/\sqrt{d}$ | **3** | **0.686** | **0.602** | **0.320** | **0.708** | **0.807** | **0.919** |

归一化 + 3 层在所有 6 个数据集上全面超越 baseline。归一化 + 1 层已在多数数据集上追平或超过无归一化 3 层。

### 消融实验（归一化效果 vs 层深度）

| 层数 $N$ | 无归一化 ($a=1$) | 有归一化 ($a=1/\sqrt{d}$) | 提升 |
|---------|----------------|------------------------|------|
| 0 | 0.147 (SciERC) | 0.181 | +23% |
| 1 | 0.282 | 0.299 | +6% |
| 2 | 0.273 | 0.312 | +14% |
| 3 | 0.299 | 0.320 | +7% |

### 关键发现
- **$N=0$（无 BiLSTM）时差异最大**：SciERC 上无归一化 0.147 vs 归一化 0.181（+23%），因为完全没有隐式正则化的补偿
- **层数增加时差异收敛**：符合理论预测——更多层数提供更强的隐式正则化，显式归一化的边际收益递减
- 归一化在难任务上收益更大：SciERC 训练/测试实体重叠少、依存图复杂，是性能提升最显著的数据集
- 效果跨语言通用：在 6 种语言的 Universal Dependencies 数据集上同样有效
- 效果跨领域通用：在非语言的分子图推理（QM9）和图像超像素（CIFAR10 Superpixel）上也观察到类似现象
- 用归一化 + 1 层 BiLSTM + (200, 100) 参数就能匹配无归一化 3 层 (400, 500) 的性能，参数减少约 85%

## 亮点与洞察
- **被忽视的简单改进**：整个 NLP 依存分析社区数年来一直使用无归一化的 biaffine scorer，而 Transformer 一开始就指出了归一化的必要性。本文用理论和实验证明这不只是"可选的好习惯"，而是导致了系统性过参数化。
- **理论分析优雅**：从隐式正则化→秩降低→方差减小的链条完整解释了"为什么多层 BiLSTM 能部分补偿缺乏归一化"的内在机理。
- **实际价值大**：任何使用 biaffine scorer 的模型（关系抽取、共指消解等）都可以立即添加 $1/\sqrt{d}$ 缩放，几乎零成本换取更好的参数效率。

## 局限与展望
- 只在 BERTbase（冻结）上做主实验，fine-tuning 和更大预训练模型下的效果待验证
- 理论分析基于深层线性网络假设，BiLSTM 是非线性的——理论和实际之间有 gap
- 只比较了 $1/\sqrt{d}$ 一种归一化，未探索 LayerNorm、RMSNorm 等其他方案
- 未在非 biaffine 的最新方法（如 LLM-based IE）上验证

## 相关工作与启发
- **vs Dozat & Manning (2017)**: 原始 biaffine parser 奠基之作，未使用归一化，后续大量工作沿用了这一设计而从未质疑
- **vs Vaswani et al. (2017)**: Transformer 注意力从一开始就使用 $1/\sqrt{d_k}$ 缩放，本文将这一洞察迁移到依存分析
- **vs SENet/Attention normalization**: 通道/注意力归一化在 CV 中广泛研究，本文在 NLP 结构预测中发现了类似的被忽视需求
- 对任何使用点积得分的模型都有启发：检查是否做了适当的方差归一化

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现被整个子领域忽视的问题并给出理论解释，虽然"修复"本身极简
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个 NLP 数据集 + 6 种语言 + 3 个非语言数据集、大量消融、5 种随机种子 + 统计检验
- 写作质量: ⭐⭐⭐⭐ 理论和实验论证逻辑严密，从 Result 1 → Claim 1 → 实验验证的推导链清晰
- 价值: ⭐⭐⭐⭐ 实际改进简单易用但影响广泛，对所有使用 biaffine scoring 的工作都有直接价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](../../ICML2025/model_compression/moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[CVPR 2025\] HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](../../CVPR2025/model_compression/hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

<!-- RELATED:END -->
