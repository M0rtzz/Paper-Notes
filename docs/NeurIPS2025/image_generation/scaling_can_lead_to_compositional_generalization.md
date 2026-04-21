---
title: >-
  [论文解读] Scaling Can Lead to Compositional Generalization
description: >-
  [NeurIPS 2025][图像生成][compositional generalization] 通过理论证明和大规模实验表明，标准MLP仅需扩大数据量和模型规模即可实现组合泛化，无需显式模块化架构设计，且组合泛化成功时任务成分可从隐层激活线性解码——该指标与扩散模型的图像组合成功率正相关。
tags:
  - NeurIPS 2025
  - 图像生成
  - compositional generalization
  - scaling
  - linear decodability
  - hyperteacher
  - MLP
---

# Scaling Can Lead to Compositional Generalization

**会议**: NeurIPS 2025  
**arXiv**: [2507.07207](https://arxiv.org/abs/2507.07207)  
**代码**: [GitHub](https://github.com/smonsays/scale-compositionality)  
**领域**: 理论机器学习 / 组合泛化  
**关键词**: compositional generalization, scaling, linear decodability, hyperteacher, MLP

## 一句话总结

通过理论证明和大规模实验表明，标准MLP仅需扩大数据量和模型规模即可实现组合泛化，无需显式模块化架构设计，且组合泛化成功时任务成分可从隐层激活线性解码——该指标与扩散模型的图像组合成功率正相关。

## 研究背景与动机

**领域现状**：组合泛化——理解和生成已知成分的全新组合——被认为是智能的核心能力。自Fodor & Pylyshyn (1988)以来，神经网络能否真正实现组合泛化一直是争论焦点。大规模模型展现了令人印象深刻的组合能力，但即使最强模型仍有频繁的组合失败案例。

**现有痛点**：许多工作主张必须赋予网络显式的组合结构（模块化架构、图网络等）才能实现组合性。但这一立场与大模型的经验性成功之间存在矛盾——大模型并未使用显式模块化设计却似乎具备了某种程度的组合能力。

**核心矛盾**：组合任务族的任务数量按 $\mathcal{O}(M^K)$ 指数增长，穷举覆盖不可能。但记忆所有任务需要指数级网络容量，而如果存在一个利用组合结构的泛化解且其复杂度远低于记忆解，那么扩大规模是否就能"自动"发现这个泛化解？

**本文切入**：在严格的数学框架下研究这个问题——定义组合任务族、设计Hyperteacher作为通用测试平台、证明泛化解的存在性和效率性、實验验证scaling确实能导致组合泛化。核心idea是"泛化解的算法复杂度线性于模块数M，而记忆解指数于M，因此规模增大时泛化解越来越占优势"。

## 方法详解

### 整体框架

研究范式：(1) 形式化定义组合任务族（Definition 2.1）和组合泛化（Definition 2.2）；(2) 用Hyperteacher实例化通用组合任务族——M个模块中选K个组合，产生 $\mathcal{O}(M^K)$ 个任务；(3) 让标准MLP在子集任务上训练，评估其在未见组合任务上的泛化性能；(4) 变化数据规模和模型规模，观察组合泛化的涌现；(5) 用线性解码器探测隐层是否形成了任务成分的线性表示。

### 关键设计

1. **组合任务族与Hyperteacher（实验平台）**:
    - 功能：构建一个可精确控制的合成任务族，作为研究组合泛化的实验平台
    - 核心思路：Hyperteacher定义为线性超网络：$C(\mathbf{z}) = \mathbf{x} \mapsto \mathbf{\Omega} \text{ReLU}(\sum_{k:z_k \neq 0} z_k \mathbf{\Theta}_k \mathbf{x})$，其中 $\{\mathbf{\Theta}_m\}_{m=1}^M$ 是M组权重矩阵，$\mathbf{\Omega}$ 是共享读出投影。每个任务由选择K个模块的组合向量 $\mathbf{z}$ 指定，任务函数 $f_{\mathbf{z}}$ 是单隐层ReLU网络。组合算子C满足单射性（每种组合产生不同任务）和低复杂度（实现C的程序长度亚指数于K）
    - 设计动机：神经网络的灵活性使Hyperteacher覆盖广泛的函数行为，同时保持组合结构的可控性。补充实验在Compositional Preference任务族（网格世界+组合奖励）上复现了所有发现

2. **泛化解的理论保证（Theorem 3.1）**:
    - 功能：证明ReLU MLP可以用线性于模块数M的神经元数量逼近Hyperteacher，而记忆所有任务需要指数级容量
    - 核心思路：对任意 $M, \varepsilon > 0$，在紧输入集上存在ReLU MLP用 $\mathcal{O}(\frac{1}{\sqrt{\varepsilon}} + M)$ 个神经元逼近Hyperteacher到 $\varepsilon$ 精度（$\|\cdot\|_\infty$ 范数）。关键在于泛化解的复杂度**线性于M**，而记忆 $\mathcal{O}(M^K)$ 个任务需要指数级容量，因此M增大时泛化解的效率优势呈指数级放大
    - 设计动机：为"规模增大→泛化解被发现"提供理论支撑：SGD倾向于找到低复杂度解（simplicity bias），而泛化解恰好比记忆解简单得多

3. **线性可解码性指标（Theorem 4.1 + 实验验证）**:
    - 功能：证明成功组合泛化的模型中，任务成分必然可从模型表示中解码，实验发现更强的结论——可从隐层激活**线性**解码
    - 核心思路：Theorem 4.1证明若模型在 $1-\delta$ 概率上正确预测，则存在解码器以 $1-\sqrt{\delta}$ 概率从任务编码恢复任务成分。实验训练线性探针发现：(a) 组合泛化 $R^2$ 与线性可解码 $R^2$ 之间高度正相关（$R^2 > 0.95$）；(b) 即使给模型直接提供任务成分（identity编码），不能组合泛化的网络也会在深层丢失这些信息——拥有解纠缠表示**不等于**能组合泛化
    - 设计动机：提供一个可操作的诊断工具——通过检查模型隐层是否能线性解码任务成分来预测其组合泛化能力。应用于扩散模型验证：线性可解码性与图像生成的组合成功率正相关

### 损失函数 / 训练策略

标准MLP（ReLU激活），输入为数据点 $\mathbf{x}$ 与任务编码 $\varphi(\mathbf{z}, r)$ 的拼接。6种任务编码：Identity、Orthogonal、Language、Invertible NN、Interval Shuffle、Few-shot。训练分布必须满足"组合支持"（每个模块至少出现在某些训练任务中）和"连通支持"（没有子集模块仅单独出现）两个条件。

## 实验关键数据

### 主实验

| 任务编码 | Task Decoder R² | Input Decoder R² | 组合泛化 R² |
|---------|-----------------|------------------|------------|
| Identity | 0.95 | 1.00 | 1.00 |
| Orthogonal | 0.96 | 1.00 | 1.00 |
| Language | 0.99 | 1.00 | 1.00 |
| Invertible NN | 0.94 | 0.56 | 0.95 |
| Interval Shuffle | 0.96 | 0.73 | 0.98 |
| Few-shot | 0.90 | -0.23 | 0.97 |

### 消融实验

| 配置 | 组合泛化效果 | 说明 |
|------|------------|------|
| 增加训练任务数 | 单调提升 | 所需训练任务数亚指数增长于总任务数 |
| 增加模型宽度 | 单调提升 | 所有任务编码均受益 |
| 增加模型深度 | 单调提升 | 更深模型更好 |
| 违反组合支持 | 显著下降 | 某模块完全缺席→无法学习该模块 |
| 违反连通支持 | 显著下降 | 模块子集孤立出现→无法泛化到新组合 |
| 少数模块低频出现 | 明显下降 | 不平衡的根因是欠表示而非不对称性本身 |
| Transformer架构 | 更好的数据效率 | 达到同等泛化所需训练任务更少 |

### 关键发现

- 实现组合泛化所需的训练任务数随总任务数亚指数增长，符合Definition 2.2的理论要求
- 所有6种任务编码（包括高度非线性的few-shot编码）都能导致组合泛化，只要模型足够大
- 线性可解码性与组合泛化之间存在极强正相关，在图像生成模型（Stable Diffusion等）上也得到验证
- Identity编码下不能组合泛化的网络会在深层丢失成分信息——"看到了成分"并不意味着"能组合它们"
- 训练分布中少数模块低频出现导致泛化下降，但少数模块高频出现不影响——关键是"不要欠表示"

## 亮点与洞察

- 颠覆性结论：标准MLP无需特殊模块化设计即可组合泛化，挑战了"组合性需要符号机制"的长期观点。理论和实验完美对应——Theorem 3.1给出线性复杂度泛化解的存在性，实验验证规模增大确实找到了这个解。
- 线性可解码性作为组合泛化的诊断工具具有实际应用价值：可以直接用于检测扩散模型在哪些概念组合上会失败（可解码性低→组合成功率低），无需实际生成大量图片。
- 与生物学的联系：任何保信息映射（不只是语言）都足以实现组合泛化，可能解释了缺乏复杂语言的动物如何实现组合泛化——语言并非组合泛化的必要条件。

## 局限与展望

- SGD保证找到泛化解（而非记忆解）的理论条件尚未明确，仅依赖"简洁性偏好"的经验假设
- 仅考虑任务被完全指定的情况，未处理模糊/不完整的任务描述
- 合成数据上的scaling实验需要已知的生成过程，真实数据上如何设计覆盖充分的训练分布仍是开放问题
- 缺乏NLP领域的验证实验（仅有合成任务和图像生成验证）

## 相关工作与启发

- **vs Fodor & Pylyshyn (1988)**：经典批评认为连接主义模型无法系统性组合——本文证明在足够规模下可以
- **vs Boopathy et al. (2025)**：模块化架构可以打破scaling law——本文补充说明即使不模块化，scaling本身也能达到组合泛化，但架构先验有助于数据效率
- **vs 图像生成文献**：提供了一个新的角度理解扩散模型的组合失败——不是"模型不够大"而是"隐层未形成线性可解码的成分表示"

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 结合理论证明和大规模实验回答了一个根本性问题
- 实验充分度: ⭐⭐⭐⭐ 合成任务全面，图像生成验证有说服力，缺NLP验证
- 写作质量: ⭐⭐⭐⭐⭐ 定义严谨、论证清晰、图表精美
- 价值: ⭐⭐⭐⭐ 线性可解码性指标可直接用于诊断生成模型的组合能力

<!-- RELATED:START -->

## 相关论文

- [Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [Scaling Offline RL via Efficient and Expressive Shortcut Models](scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)
- [Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)

<!-- RELATED:END -->
