---
title: >-
  [论文解读] Teaching LLMs to Speak Spectroscopy
description: >-
  [ICML 2025 (ML4Astro Workshop)][LLM 微调] 仅使用 16 GPU 小时和 0.04% 的参数适配，通过 LoRA 将 **LLaMA-3.1-8B** 改造为可从光谱数据预测星系红移的模型，同时保留 85%+ 的语言能力，证明通用 LLM 可高效适配非文本科学模态。
tags:
  - ICML 2025 (ML4Astro Workshop)
  - LLM 微调
  - LoRA
  - 光谱分析
  - 红移预测
  - 星系
  - LLaMA
  - 跨模态学习
  - 参数高效微调
---

# Teaching LLMs to Speak Spectroscopy

**会议**: ICML 2025 (ML4Astro Workshop)

**arXiv**: [2508.10075](https://arxiv.org/abs/2508.10075)

**作者**: Nesar Ramachandra, Yuan-Sen Ting, Zechang Sun, Azton Wells, Salman Habib

**领域**: 物理 / 天体物理 / LLM 跨模态适配

**关键词**: LLM 微调, LoRA, 光谱分析, 红移预测, 星系, LLaMA, 跨模态学习, 参数高效微调

## 一句话总结

仅使用 16 GPU 小时和 0.04% 的参数适配，通过 LoRA 将 **LLaMA-3.1-8B** 改造为可从光谱数据预测星系红移的模型，同时保留 85%+ 的语言能力，证明通用 LLM 可高效适配非文本科学模态。

## 研究背景与动机

将 Transformer 模型应用于天文学光谱分析通常需要从头训练专用模型，这面临三个实际问题：

**计算资源高**：需要大量 GPU 和专业知识设计定制 tokenizer、位置编码等

**生态割裂**：专用模型无法利用 LLM 生态的优化推理框架（vLLM、TensorRT-LLM 等）

**集成困难**：在 Agent 工作流中需要在 LLM 和专用模型之间构建复杂接口

核心问题：**能否通过高效微调将预训练 LLM 直接用于处理非文本科学数据？**

该思路在化学（Jablonka et al., 2024）、材料设计（Gruver et al., 2024）、蛋白质设计（Lv et al., 2024）等领域已有探索，但在天文学中尚属首次。

## 方法详解

### 整体框架

```
光谱数据 → 数字序列化（Tokenization）→ LLaMA-3.1-8B + LoRA → 红移预测值
```

模型同时保留原始语言能力，实现"一个模型处理光谱分析 + 自然语言推理"。

### 数据准备

- **数据源**：SDSS DR16，选择 $0 < z < 0.5$、去红化 $i < 18$ 的星系
- **数据量**：10,000 个星系光谱，其中 3,000 个训练、1,000 个验证
- **采样策略**：等频分 bin，确保均匀覆盖红移范围
- **预处理**：对数波长转线性波长，归一化光通量

### 关键设计：Tokenization

放弃训练专用 tokenizer，直接利用 LLM 的文本 tokenizer：

将光通量值 $4.56$ 序列化为 `"4|5|6"`（base=10, prec=2），完整光谱连接为：

```
"4|5|6 , 7|5|4 , 1|1|2|0 ,"
```

输入提示前缀：`"Galaxy spectrum is rescaled and encoded to an input series:"`

目标格式：`"Redshift: [value]"`

这种方法不需要任何架构修改，测试了"最低努力"下的性能下界。

### LoRA 微调

权重更新分解为低秩矩阵：

$$W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

**核心配置**：
- 模型：LLaMA-3.1-8B-Instruct
- LoRA rank: 8
- 可训练参数：3.4M（占总参数 0.04%）
- 训练：2 个 epoch
- 计算资源：16 A100 GPU 小时
- 每条训练样本占 8K 上下文窗口的不到 7%

## 实验关键数据

### 超参数消融实验

| 学习率 | LoRA Rank | 训练数据量 | Epochs | 红移 MAE↓ | 科学 QA 保留↑ | 通用 QA 保留↑ |
|:---|:---|:---|:---|:---|:---|:---|
| $10^{-5}$ | 8 | 3,000 | 2 | 0.104 | 96.5% | 95.1% |
| **$10^{-4}$** | **8** | **3,000** | **2** | **0.043** | **85.2%** | **89.4%** |
| $10^{-3}$ | 8 | 3,000 | 2 | 0.065 | 76.2% | 79.8% |

### LoRA Rank 消融

| LoRA Rank | 红移 MAE↓ | 科学 QA 保留↑ | 通用 QA 保留↑ |
|:---|:---|:---|:---|
| 4 | 0.078 | 87.8% | 91.2% |
| **8** | **0.043** | **85.2%** | **89.4%** |
| 16 | 0.057 | 82.1% | 86.7% |

### 训练轮数消融

| Epochs | 红移 MAE↓ | 科学 QA 保留↑ | 通用 QA 保留↑ |
|:---|:---|:---|:---|
| 1 | 0.099 | 87.9% | 91.5% |
| **2** | **0.043** | **85.2%** | **89.4%** |
| 3 | 0.074 | 83.7% | 88.1% |

### 关键发现

1. **学习率 $10^{-4}$ 为最优平衡点**：MAE=0.043，语言能力退化 <15%
2. **Rank 8 最优**：更低的 rank 容量不足，更高的 rank 收益递减且语言退化加剧
3. **2 个 epoch 最佳**：1 个 epoch 适配不充分，3 个 epoch 过拟合且知识遗忘加重
4. **模型保留多模态推理能力**：微调后仍可回答关于红移 $z=0.315$ 处星系类型的领域问题

### 与专用方法对比

- 专用光谱红移方法可达 MAE < 0.01（Bolton et al., 2012）
- 本方法 MAE=0.043，竞争性但非 SOTA
- **核心价值不在于精度**，而在于单一模型同时处理数据分析和自然语言推理

## 亮点与洞察

- **极低的适配成本**：0.04% 参数 + 16 GPU 小时 → 竞争性光谱分析能力
- **"增强而非替代"的理念**：保留语言能力的同时增加科学模态，支持端到端 Agent 工作流
- **揭示了 Transformer 表示的通用性**：在文本上预训练的模型捕获了适用于序列信号处理的一般计算策略
- **降低门槛**：领域科学家无需设计专用架构，只需使用标准微调 API
- **实际的 Agent 集成演示**：同一模型先预测红移，再用自然语言讨论星系类型和观测策略

## 局限性

1. **仅验证了单一任务**（红移预测），未测试恒星参数、元素丰度等其他光谱任务
2. **Tokenization 次优**：文本 tokenizer 处理数值的效率远低于学习得到的 tokenizer
3. **短论文（6 页）**：消融实验虽有系统性，但深度有限
4. **MAE=0.04 与 SOTA 差距较大**：在高精度需求场景下可能不够用
5. **未测试更大模型**：70B 等更大模型是否能在保留更多语言能力的同时达到更好精度？
6. **Workshop 论文**：而非主会议论文，peer review 深度有限

## 相关工作与启发

- **TransformerPayne (Różański et al., 2025)**：专用天文 Transformer，从头训练
- **AstroConformer (Pan et al., 2024)**：天文时间序列专用模型
- **Jablonka et al. (2024)**：LLM 用于化学预测（类似思路在化学领域的先驱）
- **Gruver et al. (2024)**：微调 LLM 生成无机材料（材料领域的类似工作）
- **ProLLaMA (Lv et al., 2024)**：蛋白质序列的 LLM 适配

**启发**：该工作提出了一个重要的实用范式——与其从头训练专用模型，不如在通用 LLM 上做轻量适配。随着 LLM 基础能力持续提升，这种"搭便车"策略的性价比会越来越高。对于计算资源有限的领域科学家尤其友好。

## 评分

- **新颖性**: ⭐⭐⭐ — 思路在其他领域已有先例，天文领域首次
- **技术深度**: ⭐⭐⭐ — 方法简单直接，消融实验系统但论文较短
- **实用性**: ⭐⭐⭐⭐⭐ — 极低成本 + 标准 API = 高度可复现和可推广
- **写作质量**: ⭐⭐⭐⭐ — 清晰简洁，适合 workshop 篇幅
- **综合评分**: 7/10 — 实用价值高，但技术贡献有限

<!-- RELATED:START -->

## 相关论文

- [Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)
- [Rethink the Role of Deep Learning towards Large-scale Quantum Systems](rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)
- [Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents](gravity-bench-v1_a_benchmark_on_gravitational_physics_discovery_for_agents.md)
- [Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)
- [Compact Matrix Quantum Group Equivariant Neural Networks](compact_matrix_quantum_group_equivariant_neural_networks.md)

<!-- RELATED:END -->
