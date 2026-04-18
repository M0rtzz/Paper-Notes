---
title: >-
  [论文解读] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios
description: >-
  [AAAI 2026][时间序列][推测解码] 提出 SpecFormer，一种融合单向和双向注意力的非自回归草稿模型架构，在大批次推理场景下通过降低对复杂前缀树的依赖、减少位置相关参数，实现了对 LLM 推理的一致性加速。
tags:
  - AAAI 2026
  - 时间序列
  - 推测解码
  - 非自回归生成
  - 大批次推理
  - LLM加速
  - SpecFormer
---

# Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios

**会议**: AAAI 2026  
**arXiv**: [2511.20340](https://arxiv.org/abs/2511.20340)  
**代码**: [https://github.com/ShiLuohe/SpecFormer](https://github.com/ShiLuohe/SpecFormer)  
**领域**: 时间序列 / LLM推理加速  
**关键词**: 推测解码, 非自回归生成, 大批次推理, LLM加速, SpecFormer

## 一句话总结

提出 SpecFormer，一种融合单向和双向注意力的非自回归草稿模型架构，在大批次推理场景下通过降低对复杂前缀树的依赖、减少位置相关参数，实现了对 LLM 推理的一致性加速。

## 研究背景与动机

### 问题起源

大语言模型（LLM）采用自回归解码，每次只生成一个 token，导致算术强度（Arithmetic Intensity, AI）低，芯片计算能力严重浪费。**推测解码（Speculative Decoding, SD）** 是提升 AI 的核心方法：先用小模型快速生成多个草稿 token，再由大模型并行验证，从而在一次前向传播中接受多个 token。

### 现有方法的痛点

**大批次场景下失效**：连续批处理（continuous batching）已被主流推理框架广泛采用，它本身就会压缩可用的空闲计算资源。当 batch size 增大时，每个参数的计算强度提高，留给草稿 token 的计算预算急剧缩减。现有 SD 方法依赖大规模前缀树（prefix tree），在大批次下无法正常工作。

**位置相关参数过多**：无论是自回归方法（如 EAGLE、HASS）还是非自回归方法（如 Medusa、MTP），它们的参数量都与草稿序列长度线性相关。自回归方法需要为每个位置重复访问参数，非自回归方法为每个位置分配独立参数。这使得这些方法在资源受限时难以扩展。

**草稿质量 vs 草稿数量的矛盾**：在低计算预算下，不能再依赖"广撒网"式的大前缀树，必须提升每个草稿 token 本身的准确率。

### 核心洞察

SD 的草稿生成本质上只需要有限长度的未来 token（而非开放式生成），因此可以用**双向注意力**进行并行生成。同时，通过从 LLM 的多层隐状态中提取丰富的上下文信息，可以在不微调原始 LLM 的前提下显著提升草稿质量。

## 方法详解

### 整体框架

SpecFormer 由两个核心模块组成：
- **Context Causal Attention（上下文因果注意力）**：从 LLM 的多层隐状态中提取信息，生成位置特定的初始表示
- **Draft Bi-directional Attention（草稿双向注意力）**：在草稿 token 维度上进行双向自注意力，实现并行的精细化生成

### 关键设计

#### 1. **Context Causal Attention**

该模块的目标是从 LLM 的隐状态中充分提取上下文信息，为后续草稿生成提供高质量输入。

**Hook 与 Downsampler**：从 LLM 的 4 个关键层提取隐状态：
- $\mathrm{HS}[0]$：embedding层，包含未经上下文处理的原始 token 信息
- $\mathrm{HS}[L/2]$：中间层，作为信息补充
- $\mathrm{HS}[L-1]$：倒数第二层，编码了最抽象的当前 token 信息
- $\mathrm{HS}[L]$：最后一层，直接用于预测下一个 token

将这 4 层的隐状态拼接后，经过 Grouped RMS Norm 归一化（每个层切片有独立的缩放参数），再通过线性下采样器将 $4d_h$ 维压缩到 $d_h$ 维：

$$I_D = (\mathrm{MSA} \cdot \mathrm{RMS} + \mathbb{I})(W_D \cdot I_{\mathrm{Cat}})$$

这里的 MSA 可视为 LLM 的第 $(L+1)$ 层，能与现有 KV cache 管理框架无缝集成。

**Positional FFN**：通过线性投影将 $d_h$ 维映射到 $l_d \cdot d_h$ 维，为每个草稿位置注入位置特定信息：

$$D = W_P \cdot \mathrm{RMS}(I_D) + b_P$$

位置相关参数量为 $l_d \cdot d_h^2$，比 Medusa 的 $8 \cdot l_d \cdot d_h^2$ 更高效。

#### 2. **Draft Bi-directional Attention**

核心创新：在草稿 token 维度上使用**标准双向自注意力**（无因果掩码），让所有草稿位置能够相互交流信息。

$$E = (\mathrm{SwiGLU} \cdot \mathrm{RMS} + \mathbb{I})((\mathrm{SA} \cdot \mathrm{RMS} + \mathbb{I})(D))$$

关键点：
- 注意力操作在草稿 token 维度进行，有效 batch size 变为 $bs \cdot |c|$
- 由于 FlashAttention 2 的 batch size 限制（4095），采用分组计算策略（每组 3072）
- 大部分参数是**位置无关**的（SA 和 SwiGLU 的参数在所有位置共享），仅 Positional FFN 包含位置相关参数

#### 3. **效率分析框架**

论文提出了系统的 SD 效率评估体系：

**冗余比** $\rho = AI_c / AI_m$：衡量硬件的理论加速上限。对 A100，$\rho \approx 152.86$。

**优化系数** $\kappa$：在固定草稿 token 预算下评估方法效率：

$$\kappa = \frac{a \cdot l_d}{k}$$

其中 $a$ 是平均接受长度，$l_d$ 是草稿序列长度，$k$ 是总草稿 token 数。

**额外计算开销** $p$：

$$p = 1 + \frac{m_s + l_d \cdot m_p}{M}$$

SpecFormer 的 $m_p$（位置相关参数）远小于 Medusa 等方法，从而在相同预算下能获得更高的 $\kappa$。

### 训练策略

- **自蒸馏（Self-Distillation）**：不直接使用 UltraChat-200K 的原始回复，而是只保留问题部分、用基座 LLM 重新生成回复，确保草稿模型学习的分布与基座模型严格对齐
- **训练目标**：标准的多位置下一 token 预测损失（公式 6b）
- **无需微调 LLM**：所有训练仅限于 SpecFormer 的参数

## 实验关键数据

### 主实验

在 Qwen2.5-7B 上的对比（多 benchmark 平均结果）：

| Batch Size | Draft Tokens $k$ | 方法 | $\kappa$ | TPS | 加速比 |
|:---:|:---:|:---|:---:|:---:|:---:|
| 1 | 4 | W/o SD | 1 | 41 | 1.00× |
| 1 | 4 | HASS | 2.14 | 69 | 1.70× |
| 1 | 4 | EAGLE-3 | 2.16 | 70 | 1.73× |
| 1 | 4 | **SpecFormer** | **2.20** | **73** | **1.78×** |
| 64 | 4 | W/o SD | 1 | 2590 | 1.00× |
| 64 | 4 | HASS | 2.13 | 4454 | 1.72× |
| 64 | 4 | EAGLE-3 | 2.15 | 4429 | 1.71× |
| 64 | 4 | **SpecFormer** | **2.19** | **4610** | **1.78×** |
| 128 | 4 | W/o SD | 1 | 5143 | 1.00× |
| 128 | 4 | HASS | 2.14 | 8800 | 1.71× |
| 128 | 4 | EAGLE-3 | 2.16 | 8846 | 1.72× |
| 128 | 4 | **SpecFormer** | **2.18** | **9154** | **1.78×** |

跨模型规模（基于 Qwen3 系列）：

| Batch Size | 模型 | 无SD TPS | SpecFormer TPS | 加速比 |
|:---:|:---|:---:|:---:|:---:|
| 1 | Qwen3-4B | 30 | 46 | 1.54× |
| 1 | Qwen3-8B | 31 | 46 | 1.49× |
| 1 | Qwen3-14B | 26 | 39 | 1.46× |
| 64 | Qwen3-4B | 2346 | 3621 | 1.53× |
| 64 | Qwen3-8B | 1904 | 2834 | 1.48× |
| 64 | Qwen3-14B | 1713 | 2524 | 1.47× |

### 消融实验

| 配置 | $\kappa$ | TPS | 说明 |
|:---|:---:|:---:|:---|
| SpecFormer（自蒸馏） | 1.90 | 56 (1.76×) | 完整方法 |
| SpecFormer（无自蒸馏） | 1.19 | 30 (0.94×) | 性能严重退化 |

自蒸馏的效果极为显著：不使用自蒸馏时 $\kappa$ 仅 1.19，几乎无法加速甚至略有减速。

### 关键发现

1. **一致性加速**：SpecFormer 在 bs=1 到 bs=128 的所有场景下都保持约 1.78× 的加速比，而基线方法在大批次下加速比下降明显
2. **自蒸馏至关重要**：训练数据与基座模型分布的对齐是获得高质量草稿的关键
3. **跨模型泛化**：方法在 4B 到 14B 不同规模的模型上都保持有效
4. **低训练成本**：相比修改LLM本身的方法，SpecFormer 仅训练小型草稿头，训练资源需求低

## 亮点与洞察

1. **问题定义精准**：将 SD 的效率评估从粗粒度的"平均接受长度"细化为固定预算下的优化系数 $\kappa$，为不同部署场景提供了统一的评估标准
2. **双向注意力的合理性**：巧妙利用了"SD 只需有限长度草稿"这一特点，突破了开放式生成必须自回归的限制
3. **工程优化到位**：包括 Triton 实现的 Grouped RMS Norm、LM Head 的梯度累积策略等，确保方法能在实际部署中获得真实加速

## 局限与展望

1. **仅验证了无损SD**：论文聚焦于无损推测解码，未探索有损（近似）场景下的潜力
2. **训练数据依赖**：自蒸馏需要用基座模型重新生成数据，在某些场景下可能增加准备成本
3. **FlashAttention 的 batch size 限制**：当 $bs \cdot |c|$ 很大时需要分组处理，引入额外的调度开销
4. **未讨论与 PD 分离等系统级优化的协同**：实际部署中 SD 需要与 prefill-decode 分离、连续批处理等技术协同

## 相关工作与启发

- **EAGLE 系列**（EAGLE, EAGLE-3）：LHS 级自回归方法，SpecFormer 超越了 EAGLE-3
- **Medusa**：非自回归方法代表，位置相关参数是 SpecFormer 的 8 倍
- **MTP（Deepseek-V3）**：多 token 预测，位置共享参数少、位置特定参数多
- 启发：未来可以将双向注意力思想推广到其他"有限长度生成"任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将双向注意力引入 SD 的草稿生成是一个新颖且有充分动机的设计
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖多种模型规模、batch size，自蒸馏消融很有说服力
- **写作质量**: ⭐⭐⭐⭐ — 效率分析框架清晰，但部分公式符号较多
- **价值**: ⭐⭐⭐⭐⭐ — 解决了 SD 在实际大批次部署中的核心痛点，实用价值很高
