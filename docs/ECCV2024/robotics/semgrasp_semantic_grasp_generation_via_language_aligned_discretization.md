---
title: >-
  [论文解读] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization
description: >-
  [ECCV 2024][机器人][语义抓取生成] 提出SemGrasp方法，设计层次化VQ-VAE将抓取姿态离散为"方向-方式-精修"三个语义token，然后微调多模态大语言模型(MLLM)在统一语义空间中融合物体、抓取与语言，实现根据自然语言指令生成物理合理且语义一致的人类抓取姿态。
tags:
  - ECCV 2024
  - 机器人
  - 语义抓取生成
  - VQ-VAE离散化
  - 多模态大语言模型
  - 手物交互
  - 抓取表示学习
---

# SemGrasp: Semantic Grasp Generation via Language Aligned Discretization

**会议**: ECCV 2024  
**arXiv**: [2404.03590](https://arxiv.org/abs/2404.03590)  
**代码**: https://kailinli.github.io/SemGrasp  
**领域**: 机器人 / 语义抓取生成  
**关键词**: 语义抓取生成, VQ-VAE离散化, 多模态大语言模型, 手物交互, 抓取表示学习

## 一句话总结

提出SemGrasp方法，设计层次化VQ-VAE将抓取姿态离散为"方向-方式-精修"三个语义token，然后微调多模态大语言模型(MLLM)在统一语义空间中融合物体、抓取与语言，实现根据自然语言指令生成物理合理且语义一致的人类抓取姿态。

## 研究背景与动机

**领域现状**：在AR/VR和机器人操作等应用中，生成类人抓取姿态具有重要价值。现有抓取生成方法主要依赖物体的几何信息（点云/网格），采用MANO模型参数、接触区域或隐式表示来描述抓取姿态。

**核心问题**：仅依赖几何信息的抓取生成存在明显不足——抓取不仅需要考虑物体形状，还需理解操作意图。例如，抓取一个杯子时，"避开热水"和"准备拧开瓶盖"需要完全不同的抓取方式。然而现有方法的抓取表示难以嵌入语义信息，无法将详细的语言描述与抓取生成过程融合。

**人类抓取规划的启发**：作者观察到，人类在规划抓取时遵循三步策略：

**确定抓取方向**（orientation）：由物体类别和指令语义引导

**决定抓取方式**（manner）：受操作意图和物体形状影响

**精修抓取姿态**（refinement）：根据物体几何细节和接触状态确保物理合理性

这一观察启发了作者设计一种模拟人类抓取规划、显式包含这三步并隐式嵌入语义信息的抓取表示。

**数据集缺乏**：现有的抓取-语言对齐数据集非常稀缺，已有标注仅覆盖简单意图，不足以训练语义抓取生成模型。

## 方法详解

### 整体框架

SemGrasp由两个核心组件构成：(1) **抓取离散化模块**——使用层次化VQ-VAE将连续的抓取姿态离散为三个语义token；(2) **抓取感知语言模型**——基于LLaVA架构的MLLM，在统一语义空间中融合物体特征、抓取token和语言描述，实现从语言指令到抓取姿态的生成。训练数据来自作者构建的CapGrasp数据集，包含约26万条详细描述和5万种多样化抓取。

### 关键设计

#### 1. **层次化VQ-VAE抓取离散化**

**核心思路**：将抓取姿态 $\boldsymbol{G} = (\boldsymbol{T}, \boldsymbol{\theta}, \boldsymbol{\beta})$ 离散为三个相互关联的token $\langle \texttt{o}, \texttt{m}, \texttt{r} \rangle$，分别表示方向（orientation）、方式（manner）和精修（refinement）。

**设计动机**：语言本质上是离散的，将抓取也离散化可以自然地与语义空间对齐。此外，根据Grasp Taxonomy，人类抓取可归为33种离散类型。离散化带来两个优势：(a) 增强可控性和可解释性；(b) 大幅降低抓取空间维度，简化学习过程。

**具体实现**：采用层次化VQ-VAE结构，包含三级编码器 $\mathcal{E}_i$、解码器 $\mathcal{D}_i$ 和码本 $\mathcal{B}_i$（$i \in \{1,2,3\}$），逐步从低层到高层捕获抓取信息：

- **方向token** $\langle \texttt{o} \rangle$：捕获手的全局变换 $\boldsymbol{T}$，即 $\hat{\boldsymbol{T}} = \mathcal{D}_1(\texttt{o}, \boldsymbol{O})$
- **方式token** $\langle \texttt{m} \rangle$：在方向条件下捕获手部姿态参数 $\boldsymbol{\theta}, \boldsymbol{\beta}$，即 $\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\beta}} = \mathcal{D}_2(\texttt{o}, \texttt{m}, \boldsymbol{O})$
- **精修token** $\langle \texttt{r} \rangle$：在方向+方式条件下预测增量参数 $\Delta\boldsymbol{T}, \Delta\boldsymbol{\theta}, \Delta\boldsymbol{\beta}$

最终抓取重建为：$\hat{\boldsymbol{G}} = (\Delta\hat{\boldsymbol{T}} \cdot \hat{\boldsymbol{T}}, \Delta\hat{\boldsymbol{\theta}} + \hat{\boldsymbol{\theta}}, \Delta\hat{\boldsymbol{\beta}} + \hat{\boldsymbol{\beta}})$

编码器通过最近邻查找将输入映射到码本：$\texttt{z} = \mathcal{E}(\boldsymbol{z}) = \text{argmin}_k \|\mathcal{N}_{\mathcal{E}}(\boldsymbol{z}) - \boldsymbol{b}_k\|_2$，其中码本 $\mathcal{B}$ 含 $K=512$ 个条目，维度 $d_{\mathcal{B}}=256$。

#### 2. **抓取感知MLLM**

**核心思路**：基于Vicuna-7B微调一个MLLM，将抓取离散token、物体特征和语言描述融合在统一语义空间中。

**三模态输入**：
- **抓取模态**：VQ-VAE编码器冻结后作为tokenizer，输出三个grasp token，前后添加特殊标记 `<SG>` 和 `<EG>`
- **物体模态**：使用PointBERT提取点云特征 $f_{\boldsymbol{O}} \in \mathbb{R}^{513 \times 384}$，通过线性投影层 $\mathcal{P}_{\boldsymbol{O}}$ 映射到4096维语言空间，同时添加物体尺寸token `<OS>`
- **语言模态**：使用SentencePiece将文本tokenize为32K词片

**训练过程**：采用LoRA微调（rank=64，约微调6%参数），分两阶段：
1. **多模态对齐**：训练模型从物体特征和语言描述预测抓取token，更新投影层 $\mathcal{P}_{\boldsymbol{O}}$ 和embedding层
2. **指令微调**：在抓取生成任务+语言输出上进一步微调，冻结投影层保证稳定性

#### 3. **CapGrasp数据集**

**核心思路**：基于GPT-4自动标注方法扩展现有手物交互数据集OakInk，构建大规模语言-抓取对齐数据集。

**三级标注层次**：
- **低层标注**：通过手部和物体位置计算接触状态（手指与物体部件的接触关系），阈值设为3mm
- **高层标注**：利用低层信息和GPT-4/GPT-4v推断操作意图、抓取力度等
- **会话标注**：利用GPT-4构建多轮抓取-语言混合对话

**统计**：约1.8k个物体模型，5万个手物抓取对，每对平均5条详细描述和会话标注。

### 损失函数 / 训练策略

**VQ-VAE训练损失**：

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{emb}} + \mathcal{L}_{\text{com}}$$

- **重建损失**：$\mathcal{L}_{\text{rec}} = \|\boldsymbol{H} - \mathcal{M}(\hat{\boldsymbol{G}})\|_2^2$，在手部顶点空间计算
- **嵌入损失+承诺损失**：$\mathcal{L}_{\text{emb}} + \mathcal{L}_{\text{com}} = \|\text{sg}[\mathcal{N}_{\mathcal{E}}(\boldsymbol{z})] - \boldsymbol{b}_{\texttt{z}}\|_2^2 + \|\mathcal{N}_{\mathcal{E}}(\boldsymbol{z}) - \text{sg}[\boldsymbol{b}_{\texttt{z}}]\|_2^2$

**MLLM训练损失**：标准自回归NLL损失

$$\mathcal{L}_{\text{NLL}} = -\sum_i \log p(\hat{x}^i | \hat{x}^{<i}, x)$$

训练配置：batch size 128，学习率5e-4(对齐阶段)/3e-5(微调阶段)，cosine退火，4×A100 GPU，20 epochs。

## 实验关键数据

### 主实验：VQ-VAE离散表示重建质量

| 方法 | MPVPE↓ | PD↓ | SIV↓ | SD mean↓ | SD std↓ |
|------|--------|-----|------|----------|---------|
| CapGrasp (GT) | - | 0.11 | 0.62 | 0.94 | 1.62 |
| GrabNet w/o refine | 18.14 | 0.76 | 5.42 | 1.75 | 2.61 |
| GrabNet w/ refine | 27.49 | 0.54 | 3.45 | 1.77 | 2.36 |
| Jiang et al. w/ TTA | 33.84 | 0.58 | 2.78 | 1.36 | 1.55 |
| **SemGrasp** | **14.97** | **0.46** | **2.72** | 2.14 | 2.37 |
| SemGrasp w/ TTA | 23.61 | **0.37** | **1.27** | 1.90 | 2.12 |

SemGrasp在MPVPE上较GrabNet提升18%，加TTA后在PD和SIV上达到SOTA。

### MLLM语义抓取生成结果

| 方法 | P-FID↓ | PD↓ | SIV↓ | GPT-4↑ | PS↑ |
|------|--------|-----|------|--------|-----|
| CapGrasp (GT) | - | 0.11 | 0.62 | 82.3 | 4.7 |
| BERT基线 | 3.32 | 0.49 | 4.60 | 47.3 | 3.7 |
| **SemGrasp** | **2.28** | **0.48** | **4.24** | **74.5** | **4.6** |

SemGrasp在语义一致性指标GPT-4评分上达到74.5分（满分100），显著优于BERT基线（47.3分），接近CapGrasp真值（82.3分）。

### 消融实验

| 配置 | MPVPE↓ | PD↓ | SIV↓ | 说明 |
|------|--------|-----|------|------|
| 单token | 29.95 | 0.66 | 5.14 | 压缩为单码本严重降低重建精度 |
| 二token <o,m> | 25.73 | 0.58 | 4.32 | 无精修token，性能次优 |
| 三token <o,m,r> (ours) | **14.97** | **0.46** | **2.72** | 最优配置 |
| 三token + r×2 | 15.37 | 0.50 | 2.98 | 多个精修token反而增加MLLM训练复杂度 |
| 单VQ-VAE | 28.02 | 0.68 | 5.31 | 共享码本难以捕获复杂表示 |
| 无语义分配 | 21.94 | 0.60 | 4.59 | 不给token赋予语义含义导致性能下降 |

### 关键发现

1. **三token层次化表示最优**：方向-方式-精修的语义分解设计比单token提升MPVPE约50%，精修token带来26%的额外提升
2. **离散表示具有可控性**：固定 $\langle \texttt{o}, \texttt{m} \rangle$ 可在不同形状物体上生成方向和方式一致的抓取，而cVAE方法不具备此可解释性
3. **码本大小敏感**：K=256会导致不收敛，K=1024会欠拟合，K=512是最优选择
4. **Vicuna优于Llama**：指令微调过的Vicuna在抓取任务中表现更好

## 亮点与洞察

1. **优雅的设计理念**：从人类抓取规划的认知过程出发，将连续高维的抓取空间离散为"方向→方式→精修"的层次化token，既符合直觉又便于与语言空间对齐
2. **极低维度的抓取表示**：仅用3个离散token（每个从512个码本条目中选择）即可表达复杂的手部姿态，大幅降低了学习难度
3. **可控且可解释**：与cVAE等连续潜空间方法不同，离散token具有明确的语义含义，使抓取生成过程透明可控
4. **端到端应用验证**：在D-grasp（AR/VR）和UniDexGrasp（机器人）两个下游任务中验证了生成抓取的实用性

## 局限与展望

1. **仅支持单手静态抓取**：未涉及双手协作操作，需要大量双手运动捕捉数据
2. **动态抓取需RL辅助**：需结合D-grasp等额外RL策略才能生成动态抓取序列，非端到端
3. **数据集依赖GPT-4标注**：CapGrasp的高层标注依赖GPT-4可能存在幻觉问题，虽有人工审核但质量仍有提升空间
4. **评估指标局限**：GPT-4辅助评估的语义一致性评分可能不够客观

## 相关工作与启发

- **GrabNet / Jiang et al.**：基于cVAE的抓取生成方法，仅利用几何信息，缺乏语义可控性
- **LLaVA**：MLLM架构的灵感来源，作者将其扩展到3D抓取领域
- **MotionGPT**：将运动序列tokenize后与语言统一的思路与本文类似
- **启发**：离散化+MLLM的范式可推广到其他需要语义控制的3D交互任务（如全身运动、物体操作）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 层次化VQ-VAE抓取离散化的设计新颖且优雅
- **实验充分度**: ⭐⭐⭐⭐ — 包含重建、生成、消融和下游应用，评估指标全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述自然，图表丰富
- **价值**: ⭐⭐⭐⭐ — 首次将语言指令引入精细的人类抓取生成，开辟了语义抓取的新方向

<!-- RELATED:START -->

## 相关论文

- [An Economic Framework for 6-DoF Grasp Detection](an_economic_framework_for_6-dof_grasp_detection.md)
- [Prioritized Semantic Learning for Zero-shot Instance Navigation](prioritized_semantic_learning_for_zero-shot_instance_navigation.md)
- [LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control](disco_embodied_navigation_and_interaction.md)
- [AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation](aff-ttention_affordances_and_attention_models_for_short-term_object_interaction_.md)

<!-- RELATED:END -->
