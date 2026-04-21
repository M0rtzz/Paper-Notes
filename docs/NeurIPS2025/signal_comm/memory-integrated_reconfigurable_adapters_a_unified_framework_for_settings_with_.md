---
title: >-
  [论文解读] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks
description: >-
  [NeurIPS 2025][联想记忆] MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。
tags:
  - NeurIPS 2025
  - 联想记忆
  - Hopfield网络
  - 适配器
  - 持续学习
  - 领域泛化
---

# Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2512.00940](https://arxiv.org/abs/2512.00940)  
**代码**: https://snimm.github.io/mira_web/ (有)  
**领域**: 持续学习 / 领域泛化  
**关键词**: 联想记忆, Hopfield网络, 适配器, 持续学习, 领域泛化

## 一句话总结

MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。

## 研究背景与动机

**领域现状**：深度学习中的领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）是三个重要但相互独立发展的研究方向。DG 要求模型在未见域上泛化，CL 要求模型在新任务到来时不遗忘旧知识。现有方法通常针对单一场景设计专门的架构和策略。

**现有痛点**：生物体能在毫秒级切换多种行为模式（如蝙蝠回声定位时从20Hz调至200Hz、爵士钢琴家即兴演奏），同时保持已学习的知识不被遗忘。这依赖于大脑中同一神经回路被神经调节信号（多巴胺、乙酰胆碱等）动态复用的机制。然而现有深度学习方法缺乏这种统一的"多任务快速切换+持久记忆"机制。

**核心矛盾**：DG、CIL、DIL 看似不同，但本质上都需要模型在多任务/多域间高效适配并保持知识。现有工作将它们完全割裂处理，且没有借鉴生物联想记忆（AM）来统一解决这些问题。

**本文目标** (1) 如何构建一个统一架构同时处理 DG、CIL、DIL；(2) 如何利用联想记忆机制实现每样本级别的适配器动态组合检索；(3) 如何学习有效的检索键来索引存储的适配器权重。

**切入角度**：受神经科学启发，联想记忆可以存储和检索特定任务的权重调制信号。如果将 LoRA 适配器作为"值"存入联想记忆，并通过可学习的"键"按样本检索，就能实现类似大脑神经调制的快速任务切换。

**核心 idea**：将 Hopfield 联想记忆嵌入 ViT 每层，存储任务特定的 LoRA 适配器为值，通过后置学习的检索键实现每样本的适配器仿射组合检索，统一解决 DG/CIL/DIL。

## 方法详解

### 整体框架

MIRA 基于一个冻结的 ViT-B/16 骨干网络（CLIP 初始化），在每一层附加 Universal Hopfield Network (UHN) 记忆单元。输入图像经过骨干网络时，每层的记忆单元根据前一层的输出生成查询向量，检索存储在记忆中的适配器权重的加权组合并加载到该层。整体训练分两阶段：**Adaptation**（为各任务训练独立 LoRA 适配器并存入记忆）和 **Consolidation**（仅优化检索键和查询模块使检索的组合最优）。推理时只需前向传播。

### 关键设计

1. **联想记忆适配器存储与检索**:

    - 功能：以键值对形式存储每个任务/域训练的 LoRA 适配器，推理时按需检索组合
    - 核心思路：在 ViT 的每层 $\ell$ 附加记忆单元 $\mathcal{M}_\ell$，写操作将训练好的适配器 $\theta_\ell^{(t)}$ 存入，读操作通过查询向量 $q$ 与所有键 $\mathbf{K}$ 计算相似度后加权组合：$\hat{\theta}_\ell = \Theta_\ell \cdot \text{sep}(\text{sim}(K_\ell^\top, q))$。关键在于使用**仿射函数**作为分离函数（而非 Softmax），允许负权重以主动"去除"干扰信息，而非仅"遮蔽"
    - 设计动机：存储权重适配器而非数据使模型可在推理时动态组合多任务参数化知识，无需梯度计算。仿射函数在 CIL 和 DG 中优于 Softmax（消融证实）

2. **两阶段训练流程（Adaptation + Consolidation）**:

    - 功能：解耦适配器训练和检索优化
    - 核心思路：Adaptation 阶段用交叉熵损失训练各任务的 LoRA 适配器（rank=4），然后以随机高斯初始化键存入记忆。Consolidation 阶段冻结适配器值，仅训练每层的查询模块 $g_\ell$ 和键 $\mathbf{K}_\ell$，使检索到的组合最小化交叉熵。DG 中所有域数据联合巩固，CL 中按任务顺序依次巩固
    - 设计动机：分离两阶段使检索空间可独立优化，Consolidation 本质是通过 AM 内积近似求解最优适配器组合问题（论文 Lemma 1 形式化证明了这一点）

3. **可学习查询模块**:

    - 功能：将前层输出从表示空间对齐到键空间
    - 核心思路：每层配备轻量模块 $g_\ell: \mathbb{R}^{d_h} \to \mathbb{R}^{d_k}$，可以是恒等映射、线性变换或小型网络。查询 $q_\ell = g_\ell(h_{\ell-1})$ 与键的内积经分离函数得到适配器组合权重。键和查询模块通过反向传播联合优化
    - 设计动机：层输出和键可能处于不同表示空间，查询模块负责对齐。后置学习（而非固定键）使检索适应性地优化

### 损失函数 / 训练策略

两阶段均使用交叉熵损失。Adaptation 只更新 LoRA 参数，Consolidation 只更新键和查询模块。CL 场景中可在 Consolidation 内嵌 DualGPM 等遗忘缓解技术。Hopfield 键引入的额外参数不到总参数的 0.4%（~276K / 86M），推理延迟开销仅 ~0.4%。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | MIRA | 前SOTA | 提升 |
|--------|------|------|------|--------|------|
| iDigits | CIL | Avg Acc↑ | **83.00%** | 71.53% (ICON) | +11.47% |
| CORe50 | CIL | Avg Acc↑ | **83.39%** | 80.85% (ICON) | +2.54% |
| DomainNet | CIL | Avg Acc↑ | **67.29%** | 65.43% (ICON) | +1.86% |
| CORe50 | DIL | Avg Acc↑ | **93.89%** | 89.01% (ICON) | +4.88% |
| DomainNet | DIL | Avg Acc↑ | **69.18%** | 54.44% (ICON) | +14.74% |
| PACS | DG | Acc | **97.01%** | 96.50% (PEGO) | +0.51% |
| OfficeHome | DG | Acc | **87.36%** | 84.20% (PEGO) | +3.16% |
| DomainNet | DG | Acc | **61.19%** | 59.80% (CoOp) | +1.39% |
| DN4IL | DIL | Last Acc | **78.40%** | 44.45% (DUCA) | +33.95% |
| ImageNet-R | CIL-5 | ACC5 | **78.06%** | 75.85% (C-LoRA) | +2.21% |

### 消融实验

| 分离函数 | CIL Acc | DIL Acc | DG Acc | 平均 |
|----------|---------|---------|--------|------|
| Affine (默认) | **67.29** | 69.18 | **61.19** | **65.89** |
| Softmax | 66.87 | **69.21** | 60.82 | 65.63 |
| ReLU | 66.60 | 69.20 | 60.90 | 65.57 |
| Tanh | 66.73 | 68.96 | 60.94 | 65.54 |

| 适配器数量/任务 | CIL Acc | DIL Acc | DG Acc |
|----------------|---------|---------|--------|
| 1 | 63.75 | 69.08 | 61.21 |
| 5 | 67.21 | 69.10 | 61.01 |
| 10 | **67.29** | **69.18** | 61.19 |

### 关键发现

- 仿射分离函数在 CIL 和 DG 中表现最佳，因为允许负权重主动去除干扰信息；Softmax/ReLU 只能遮蔽但不能移除
- 适配器数量从 1 增到 5 提升显著（CIL: +3.46%），5→10 边际提升。说明 5 个适配器已足够捕获大部分任务特异性
- 在 DN4IL 上以 78.40% 大幅超过使用 200 样本回放缓冲的 DARE++（44.11%），表明联想记忆可以替代回放缓冲
- 推理延迟仅增加 ~0.4%（0.0241s vs 0.0240s），额外参数 <0.4%，实际部署开销忽略不计

## 亮点与洞察

- **"存储权重而非数据"的联想记忆新用法**：传统 AM 存储数据/特征用于回放，MIRA 存储适配器权重用于直接调制网络。规模与任务数而非数据量成正比，非常高效。这是对 AM 在深度学习中应用方式的范式性转变
- **后置键学习的巧妙之处**：先训练适配器再学检索键，相当于先"写完百科全书"再"编制索引系统"，两步解耦使各自优化更简单
- **极低开销的统一性**：同一架构仅通过改变损失函数/数据提供方式即可处理三种学习范式，推理开销几乎为零。"换损失不换架构"的设计哲学值得迁移到其他多场景学习中
- **理论保证（Lemma 1）**：严格证明了 AM 检索可以表达最优适配器组合问题，从理论上验证了方案的合理性

## 局限与展望

- 当前仅使用仿射（线性）组合，非线性组合（如 MoE 门控）可能进一步提升，特别是对 OOD 外推的场景
- 实验仅限于分类任务和 ViT 架构，尚未验证在检测/分割/生成/NLP 等任务上的表现
- 每任务需独立训练一组适配器，任务数极多时存储线性增长
- 适配器维度很高时 Hopfield 网络的检索保真度分析不足
- 所有消融均在 DomainNet 上进行，其他数据集是否有相同规律未验证

## 相关工作与启发

- **vs ICON**: ICON 统一 CIL 和 DIL 但不支持 DG，依赖专门的提示池机制。MIRA 通过联想记忆提供更自然的统一，且 DG 能力是 ICON 不具备的
- **vs L2P/DualPrompt/CODA-P**: 基于提示学习的 CL 方法在每层附加可学习提示，但提示间缺乏显式的存储-检索语义。MIRA 的 Hopfield 记忆提供了明确的索引查找机制
- **vs PEGO**: DG 专用方法，在 VLCS 上略优于 MIRA，但无法处理 CL 场景。MIRA 在 OfficeHome 和 DomainNet 上显著超越
- **vs LoRA/VeRA 等 PEFT**: 标准 PEFT 只解决单任务适配，未考虑跨任务知识巩固。MIRA 在 PEFT 基础上增加了记忆索引层

## 评分

- 新颖性: ⭐⭐⭐⭐ 联想记忆存储权重适配器+后置键学习的想法新颖，理论支撑扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 CIL/DIL/DG 三种场景、7+ 数据集，消融分析详尽
- 写作质量: ⭐⭐⭐⭐ 生物动机叙述生动，从理论到实验链路完整
- 价值: ⭐⭐⭐⭐ 为统一多任务学习提供了优雅框架，AM+adapter 范式可能启发后续工作

<!-- RELATED:START -->

## 相关论文

- [Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)
- [ConTextTab: 语义感知的表格上下文学习器](contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)

<!-- RELATED:END -->
