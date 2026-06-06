---
title: >-
  [论文解读] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning
description: >-
  [ICML 2026][模型压缩][RTL质量估计] 提出 StructRTL，在 RTL 设计的控制数据流图（CDFG）上做结构感知的图自监督预训练（掩码节点建模 + 边预测），再配合从后映射网表到 CDFG 的知识蒸馏，大幅超越 LLM 和手工特征方法在面积/延迟预测任务上的 SOTA。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "RTL质量估计"
  - "图自监督学习"
  - "控制数据流图"
  - "知识蒸馏"
  - "硬件设计自动化"
---

# Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning

**会议**: ICML 2026  
**arXiv**: [2508.18730](https://arxiv.org/abs/2508.18730)  
**代码**: https://github.com/cure-lab/StructRTL  
**领域**: 图学习 / 自监督学习 / EDA  
**关键词**: RTL质量估计, 图自监督学习, 控制数据流图, 知识蒸馏, 硬件设计自动化  

## 一句话总结
提出 StructRTL，在 RTL 设计的控制数据流图（CDFG）上做结构感知的图自监督预训练（掩码节点建模 + 边预测），再配合从后映射网表到 CDFG 的知识蒸馏，大幅超越 LLM 和手工特征方法在面积/延迟预测任务上的 SOTA。

## 研究背景与动机

**领域现状**：现代硬件设计流程中，RTL（寄存器传输级）代码经过逻辑综合后才能获得面积、延迟等质量指标，但这一过程耗时且计算开销巨大。因此，直接从 RTL 阶段快速估计设计质量成为 EDA 研究的热点。

**现有痛点**：早期方法从数据流图（DFG）或抽象语法树（AST）中提取手工特征（如节点类型频率、最长组合路径长度等），表达能力有限。近期工作（VeriDistill）利用大语言模型从 RTL 代码中提取 token 级嵌入，取得了不错效果，但存在两个根本问题：其一，LLM 的预训练目标是代码生成而非质量估计，学到的表示可迁移性受限；其二，token 视角只能隐式编码设计的结构语义，不如图视角显式。

**核心矛盾**：DFG 只建模数据依赖、忽略控制流，AST 侧重语法层次、缺乏结构语义，LLM token 序列丢失拓扑信息——RTL 设计质量（特别是延迟）与结构紧密耦合，现有表示无法充分捕获。

**本文目标**：(1) 设计一个直接建模 CDFG 结构语义的自监督预训练框架；(2) 通过知识蒸馏将后映射网表的底层信息迁移到 RTL 阶段预测器。

**切入角度**：CDFG 同时整合了控制流和数据流依赖，提供了比 DFG/AST/token 更完整的设计结构视图。作者观察到直接对 CDFG 做节点/边掩码会引入歧义（例如掩掉加法节点后，减法、乘法都可能是合法替代），因此在 GNN 产生的上下文感知嵌入层面做掩码，既保留了计算图的完整性，又让模型利用邻域上下文来恢复被掩信息。

**核心 idea**：用 GNN + Transformer 在 CDFG 上做结构感知自监督预训练，学习显式结构表示，再用后映射网表做知识蒸馏增强 RTL 阶段预测。

## 方法详解

### 整体框架
输入是 RTL Verilog 代码，首先通过 Yosys 编译为 RTLIL 中间表示，再解析为 CDFG（节点表示操作/存储元素，有向边表示数据/控制依赖）。节点嵌入初始化为节点类型的 one-hot 编码与位宽的拼接：$\mathbf{h}_i^0 = \text{concat}(\text{one-hot}(\text{type}(v_i)), \text{width}(v_i))$。嵌入经过 8 层 GIN 得到上下文感知嵌入，加上基于 Laplacian 特征向量的全局位置编码后，送入 8 层 Transformer 编码器。预训练阶段执行两个自监督任务；微调阶段通过均值+最大值联合池化聚合为图级表示，经 3 层 MLP 回归面积/延迟。

### 关键设计

1. **结构感知掩码节点建模（Structure-Aware Masked Node Modeling）**:

    - 功能：通过预测被掩码节点的类型来学习结构感知表示
    - 核心思路：随机掩码 20% 的 post-GNN 嵌入（替换为可学习 [MASK] token），用 Transformer 预测其原始节点类型（32 类分类）。关键在于掩码发生在 GNN 输出之后而非原始图上，避免对计算图语义的破坏。针对类别不平衡（如运算符节点远少于线网/寄存器），采用分层掩码策略保证每类至少 $m$ 个节点被选中，并用 class-balanced focal loss 替代交叉熵，权重公式为 $w_c = (1-\beta) / (1-\beta^{S_c})$，其中 $\beta=0.9999$，$\gamma=2.0$
    - 设计动机：直接掩码原始节点会引入歧义（加法被掩后减法/乘法皆合理），post-GNN 嵌入已编码邻域上下文，掩码后可利用丰富上下文做无歧义重建

2. **边预测（Edge Prediction）**:

    - 功能：学习图的拓扑连接模式
    - 核心思路：当 post-GNN 嵌入被展平为序列输入 Transformer 时，图的连接信息丢失（等价于所有边被掩码）。每次迭代随机采样 20% 真实边为正样本，等量不存在的边为负样本，拼接源/目标节点的 Transformer 输出嵌入，经 3 层 MLP 做二分类预测边是否存在。预训练总损失为 $\mathcal{L}_{pre} = \alpha \mathcal{L}_{mnm} + (1-\alpha) \mathcal{L}_{ep}$，$\alpha=0.5$
    - 设计动机：Transformer 天然丢失图拓扑信息，边预测迫使模型从嵌入中恢复连接结构，与掩码节点建模互补（一个学节点语义、一个学边拓扑）

3. **后映射网表知识蒸馏（Post-Mapping Knowledge Distillation）**:

    - 功能：将后映射（PM）网表的底层信息迁移到 CDFG 预测器
    - 核心思路：先用 Yosys + ABC 将 RTL 综合为 PM 网表，训练一个 20 层 GIN 的教师模型直接从网表预测面积/延迟（因为指标直接源自网表，天然更准确）。冻结教师后，通过对齐 CDFG 学生与 PM 教师的最后一层激活来做蒸馏：$\mathcal{L}_{kd} = \tau \cdot \mathcal{L}_{cos}(z_{CDFG}^{-1}, z_{PM}^{-1}) + (1-\tau) \cdot \mathcal{L}_{mse}(z_{CDFG}^{-1}, z_{PM}^{-1})$，$\tau=0.7$。总训练损失 $\mathcal{L}_{total} = \mu \mathcal{L}_{qe} + (1-\mu) \mathcal{L}_{kd}$，$\mu=0.5$。推理时只保留 CDFG 预测器
    - 设计动机：RTL 与实际度量之间存在抽象层级差距，PM 网表离物理实现更近，蒸馏弥合了这一 gap

### 训练策略
- 质量估计用 log-cosh 损失（对异常值鲁棒），目标值取对数变换
- 全局位置编码采用有向图的对称归一化 Laplacian 矩阵最小 $k=16$ 个特征向量，训练时随机翻转特征向量符号以增强泛化

## 实验关键数据

### 主实验（无知识蒸馏）

| 方法 | Area $R^2$↑ | Area MAPE↓ | Delay $R^2$↑ | Delay MAPE↓ |
|------|------------|-----------|-------------|------------|
| Graph-XGBoost | 0.3987 | 0.19 | 0.3362 | 0.12 |
| Graph-GNN | 0.5857 | 0.09 | 0.6639 | 0.13 |
| CodeV-DS-6.7B | 0.4862 | 0.17 | 0.3905 | 0.12 |
| CodeV-CL-7B | 0.5755 | 0.15 | 0.5174 | 0.10 |
| CodeV-QW-7B | 0.6353 | 0.13 | 0.5277 | 0.09 |
| **StructRTL** | **0.7463** | **0.06** | **0.7630** | **0.10** |

### 消融实验 + 知识蒸馏效果

| 配置 | Area $R^2$↑ | Delay $R^2$↑ | 说明 |
|------|-----------|-------------|------|
| StructRTL (full, w/o KD) | 0.7463 | 0.7630 | 完整模型无蒸馏 |
| w/o $\mathcal{L}_{mnm}$ (w/o KD) | 0.7249 | 0.7473 | 去掉掩码节点建模，$R^2$ 下降 |
| w/o $\mathcal{L}_{ep}$ (w/o KD) | 0.7018 | 0.7368 | 去掉边预测，$R^2$ 下降更多 |
| **StructRTL + KD** | **0.8676** | **0.8872** | 加知识蒸馏后大幅提升 |
| w/o $\mathcal{L}_{mnm}$ + KD | 0.8557 | 0.8796 | 蒸馏下消融仍一致 |
| w/o $\mathcal{L}_{ep}$ + KD | 0.8480 | 0.8654 | 边预测贡献略大于掩码建模 |
| CodeV-QW-7B + KD | 0.8174 | 0.7687 | 最强 LLM baseline + KD |
| PM 教师（上界） | 0.9334 | 0.9484 | 直接从网表预测 |

### 关键发现
- 无 KD 时 StructRTL 在 $R^2$ 上比最强 LLM baseline（CodeV-QW-7B）高出 +0.11（面积）和 +0.24（延迟），延迟预测优势更大，印证了结构信息对延迟估计至关重要
- 知识蒸馏为 StructRTL 带来 +0.12（面积）和 +0.12（延迟）的 $R^2$ 提升
- 仅用 20% 训练数据时，StructRTL 已达到 LLM baseline 全量训练的竞争水平（面积 0.56 / 延迟 0.60）
- 工业设计评估：在 51 对真实工业设计上，排序准确率达面积 82.35%、延迟 88.23%
- 推理速度：平均 0.096 秒/设计 vs 综合 13.97 秒/设计，加速 145 倍

## 亮点与洞察
- **Post-GNN 掩码而非原始图掩码**——这是最精妙的设计。计算图不同于自然语言/社交网络图，节点语义严格受限，直接掩码原始节点可能有多个合法替代，post-GNN 掩码既保留原始图完整性又提供充分上下文，可迁移到其他领域的计算图（如编译器 IR、电路网表）
- **CDFG vs Token 的"显式 vs 隐式结构"之争**——用 7B LLM 看代码 token 不如用轻量 GNN+Transformer 看图结构，说明任务对齐的表示比通用大模型更重要。这一洞察可迁移到代码分析的其他下游任务
- **跨抽象层级蒸馏**——教师在 PM 网表层、学生在 RTL 层，利用同一设计在不同抽象级的天然对应关系做蒸馏，思路可推广到任何多级抽象的工程设计问题

## 局限与展望
- 数据集以中小规模开源设计为主（13,200 个），虽然工业评估了 51 对设计，但未在大规模工业 SoC 级设计上验证可扩展性
- CDFG 构建依赖 Yosys 编译成功，无法处理含 proprietary IP 或不可综合的 RTL 代码
- 知识蒸馏需要先做综合以获取 PM 网表标签，部分抵消了"跳过综合"的初衷（虽然只在训练阶段需要）
- 作者指出跨工艺节点迁移（technology-node migration）仍是开放问题，当前模型需要每个工艺节点的标注数据重新校准
- 未来方向：将方法扩展到更多 EDA 质量指标（功耗、时序裕量）、探索无需综合标签的纯自监督质量排序

## 相关工作与启发
- **VeriDistill (Moravej et al., 2025)**：LLM-based RTL 嵌入 + 知识蒸馏的先驱，本文在蒸馏框架基础上将表示从 token 切换到 CDFG 图
- **GraphMAE (Hou et al., 2022)** / **MaskGAE (Li et al., 2023)**：通用图自监督学习方法，分别做节点特征掩码重建和边掩码恢复，本文在此基础上针对计算图语义做了关键适配
- **MasterRTL (Fang et al., 2023)**：基于 SOG 手工特征的质量估计，但 SOG 构建本身需要部分综合步骤

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)
- [\[ACL 2025\] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](../../ACL2025/model_compression/llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)
- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)
- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](../../ICML2025/model_compression/toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [\[ICML 2026\] Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation](global_convergence_of_adaptive_sensing_for_principal_eigenvector_estimation.md)

</div>

<!-- RELATED:END -->
