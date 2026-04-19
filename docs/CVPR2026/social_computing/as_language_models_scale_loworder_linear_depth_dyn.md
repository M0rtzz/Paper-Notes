---
title: >-
  [论文解读] As Language Models Scale, Low-order Linear Depth Dynamics Emerge
description: >-
  [CVPR 2026][Transformer] 将 Transformer 的逐层前向传播视为离散时间动力系统，构建32维低阶线性层变体（LLV）代理模型来近似最后token隐状态的深度传播动力学——发现该代理在GPT-2-large上预测逐层干预增益的Spearman相关可达0.995，且这种线性可辨识性随模型规模单调增强（GPT-2→medium→large），进而利用代理模型的闭式最优解实现比启发式干预策略能量低2-5倍的多层激活引导方案。
tags:
  - CVPR 2026
  - Transformer
  - 线性代理模型
  - 模型缩放规律
  - 激活干预
  - 系统辨识
---

# As Language Models Scale, Low-order Linear Depth Dynamics Emerge

**会议**: CVPR 2026  
**arXiv**: [2603.12541](https://arxiv.org/abs/2603.12541)  
**代码**: 无（承诺开源配置和脚本）  
**领域**: LLM 可解释性 / 控制理论  
**关键词**: Transformer深度动力学, 线性代理模型, 模型缩放规律, 激活干预, 系统辨识

## 一句话总结

将 Transformer 的逐层前向传播视为离散时间动力系统，构建32维低阶线性层变体（LLV）代理模型来近似最后token隐状态的深度传播动力学——发现该代理在GPT-2-large上预测逐层干预增益的Spearman相关可达0.995，且这种线性可辨识性随模型规模单调增强（GPT-2→medium→large），进而利用代理模型的闭式最优解实现比启发式干预策略能量低2-5倍的多层激活引导方案。

## 研究背景与动机

**领域现状**：激活引导（activation steering）已成为控制LLM行为的主流方法——通过在特定层注入对比激活向量来修改情感、毒性等属性。但现有方法存在两大局限：(1) 选择干预层依赖启发式逐层扫描或固定规则（如"最后一层注入"）；(2) 多层干预时如何分配注入能量缺乏理论指导。

**现有痛点**：
- 最优干预层是任务相关的（有些任务中间层最有效，有些任务后面层最有效），固定规则普遍次优
- 逐层暴力扫描在大模型上成本高、无法指导多层协同干预
- 缺乏对 Transformer 深度方向表示传播的系统级数学描述

**核心矛盾**：Transformer 是高维非线性系统，通常被视为黑盒。但如果每层变换在给定prompt上下文下可用低维线性模型局部近似，就可以将干预设计从启发式搜索变为有解析解的最优控制问题。

**切入角度**：借鉴控制论中的系统辨识方法——将深度视为离散时间，最后token隐状态视为系统状态，在冻结上下文下对逐层变换做Jacobian线性化+Krylov子空间降维→得到紧凑的低阶线性代理。

## 方法详解

### 整体框架

对于给定prompt $p$，定义最后token的深度索引隐状态为 $x_\ell(p) = h_\ell(p)[t(p),:] \in \mathbb{R}^H$。冻结所有非最后token的表示，定义prompt条件下的冻结上下文映射 $x_{\ell+1} = f_\ell(x_\ell; p)$。对该映射在操作轨迹 $\bar{x}_\ell(p)$ 处做Jacobian线性化，得到 $A_\ell(p)$。沿概念方向 $v_\ell$ 注入干预 $u_\ell$，经Krylov基 $P_\ell \in \mathbb{R}^{H \times d}$ 降维后得到低阶LLV代理。该代理可预测逐层增益曲线并推导最优多层干预方案。

### 关键设计

1. **冻结上下文局部线性化**
    - 功能：将高维非线性Transformer块变换转化为可分析的线性状态转移
    - 核心思路：固定所有非最后token的表示，仅变化最后token状态，在操作点处计算Jacobian $A_\ell(p) = \frac{\partial f_\ell}{\partial x_\ell}\big|_{\bar{x}_\ell}$，得到线性化动力学 $\delta x_{\ell+1} \approx A_\ell(p) \delta x_\ell + A_\ell(p) v_\ell u_\ell$
    - 设计动机：冻结上下文确保只研究最后token的深度传播——这正是激活引导直接影响的状态。Jacobian通过JVP或中心差分高效计算，无需显式构建 $H \times H$ 的完整矩阵

2. **概念锚定Krylov基降维**
    - 功能：将 $H$ 维（如GPT-2-large的1280维）的状态空间压缩到 $d=32$ 维
    - 核心思路：降维基 $P_\ell$ 的第一列为概念方向 $v_\ell$（正负类均值差方向），其余31列由可达性启发的Krylov构造填充——从注入后的种子方向 $A_\ell v_\ell$ 出发，通过均值Jacobian逐步传播、正交化得到。降维后的LLV模型：$r_{\ell+1} \approx \bar{A}_\ell r_\ell + \bar{B}_\ell u_\ell$，其中 $\bar{A}_\ell = P_{\ell+1}^\top A_\ell P_\ell$，$\bar{B}_\ell = P_{\ell+1}^\top A_\ell v_\ell$
    - 设计动机：Krylov基优先覆盖干预实际激发的可达子空间，系统性优于随机正交基（消融验证）。概念方向作为第一列确保降维后保留对目标属性最敏感的方向

3. **最小能量多层最优控制**
    - 功能：在给定目标概念偏移 $\Delta y_{\text{tar}}$ 下，求解最小注入能量的多层干预方案
    - 核心思路：降维模型中最终概念偏移对控制向量线性：$\delta y \approx h^\top u$，其中 $h$ 的各分量即各层预测增益。最小范数解为闭式：$u^{\star} = \frac{\Delta y_{\text{tar}}}{\|h\|_2^2} h$。即按各层增益大小成比例分配能量
    - 设计动机：高增益层多分配、低增益层少分配→自然平衡干预效率。在完整模型中通过一维二分搜索验证所需最小振幅

### 损失函数 / 训练策略

无需训练——纯分析方法。概念方向由标注prompt的类条件均值差估计（concept split, $n_{\text{concept}}=400$/类）。Jacobian通过JVP或中心差分近似（步长 $2 \times 10^{-3}$）。模型在operating split ($n_{\text{operating}}=200$/类) 上辨识，增益在独立held-out split ($n_{\text{eval}}=200$/类) 评估，确保无数据泄露。

## 实验关键数据

### 主实验

**缩放规律：LLV代理 ($d=32$) 的增益预测一致性**

| 模型 | 参数量 | 隐状态维度H | Spearman↑ | Pearson↑ |
|---|:---:|:---:|:---:|:---:|
| GPT-2 | 124M | 768 | 0.77 | 0.68 |
| GPT-2-medium | 355M | 1024 | 0.81 | 0.74 |
| GPT-2-large | 774M | 1280 | **0.995** | **0.997** |

*单调递增的一致性→模型越大，局部深度动力学越可由低阶线性模型精确描述*

**GPT-2-large 逐任务增益预测（$d=32$, $\epsilon=0.1$）**

| 任务 | Spearman | Pearson |
|---|:---:|:---:|
| Amazon Polarity | 1.00 | 1.00 |
| Yelp Polarity | 1.00 | 1.00 |
| SST-2 | 0.99 | 0.99 |
| IMDB | 1.00 | 1.00 |
| Civil Comments Toxicity | 0.99 | 0.99 |
| TweetEval-Irony | 0.99 | 0.99 |
| TweetEval-Hate | 0.99 | 0.99 |

**多层控制能量对比（GPT-2-large, 归一化到LLV最优=1.0x）**

| 干预策略 | 能量倍数 (中位数) |
|---|:---:|
| LLV最优 | **1.0x** |
| Uniform-all | 2-5x |
| Last-layer-only | 10-100x |
| Random single-layer | 10-1000x |

### 消融实验

| 消融维度 | 配置 | 一致性变化 |
|---|---|---|
| 降维基 | Krylov (默认) | **最优** |
| | 随机正交基 | 困难任务下降显著 |
| 降维维度d | 极小 → 32 → 更大 | 快速提升后饱和 |
| 扰动幅度ε | 0.01-0.5 | 宽范围内稳定 |
| 概念方向 | 类条件均值差 | 标准方法，有效 |

### 关键发现

- **「模型越大，局部深度动力学越线性」**：这是一个反直觉但实证强劲的缩放规律。更大的模型全局更复杂，但局部深度响应可被更紧凑的线性代理精确捕获
- **增益预测不仅识别最优层，还捕获完整的深度响应形状**：包括平坦高原、非单调景观等丰富结构
- **最优干预深度是任务相关的**：某些任务呈单调后期放大，某些呈宽中后段高原→"总是在最后层注入"的启发式普遍次优
- **LLV最优控制在所有任务上达到最低能量或并列最低**：Uniform-all是最强启发式基线但仍需2-5倍能量

## 亮点与洞察

- **控制论视角分析Transformer**：将深度传播建模为状态空间系统→把干预设计从暴力搜索提升为有解析解的最优控制问题
- **缩放规律的科学意义**：提出"可辨识性"作为比较模型架构、训练策略的新系统级指标。规模不仅增强能力，还增强局部动力学的可压缩性和可预测性
- **分析-设计-验证闭环**：代理模型上做分析和设计→完整模型上做验证→实际控制效果反向确认代理模型的保真度

## 局限与展望

- 仅在GPT-2家族（最大774M）验证→需扩展到LLaMA/Mistral等数十亿参数级模型验证缩放规律是否持续
- 代理模型是prompt条件的局部描述→不同prompt需重新辨识，计算开销与prompt数量线性增长
- 概念方向仅用类条件均值差估计→复杂概念（如"诚实"）可能需要更精细的方向估计方法
- 未讨论多概念同时干预场景→概念间可能存在耦合/冲突
- 所有10个任务都是二分类NLP任务→复杂任务（生成、推理）的适用性未验证

## 相关工作与启发

- **vs Activation Addition (Turner et al., 2023)**：提供干预方向但不知道哪层最有效→本文预测完整深度增益曲线
- **vs 线性表示假说 (Park et al., 2024)**：解释概念为何可线性编码（静态表示）→本文研究扰动如何跨层传播（动力学视角）
- **vs Moon (2024) 可控性分析**：一般性地讨论NN可控性/可观性→本文具体辨识降维代理并验证实际控制效果
- **启发**：深度动力学的低秩性暗示可能存在可安全跳过的冗余层→对层级剪枝有理论指导意义；最优多层干预方案对安全对齐有直接应用价值

## 评分

⭐⭐⭐⭐⭐ (5/5)

控制论视角分析Transformer深度动力学非常新颖，「模型越大越线性」的缩放规律发现令人惊讶且实证扎实（10任务×3模型×多维度消融），理论与实验紧密衔接，从诊断（增益预测）到设计（最优控制）形成完整闭环——是将系统辨识理论引入LLM分析的开创性工作。GPT-2家族的规模限制是唯一遗憾。

<!-- RELATED:START -->

## 相关论文

- [Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](../../ICLR2026/social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)
- [Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [Scalable Multi-Task Low-Rank Model Adaptation](../../ICLR2026/social_computing/scalable_multi-task_low-rank_model_adaptation.md)

<!-- RELATED:END -->
