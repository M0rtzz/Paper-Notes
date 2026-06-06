---
title: >-
  [论文解读] Neural Synchrony Between Socially Interacting Language Models
description: >-
  [ICLR 2026][医学图像][神经同步] 首次研究社会交互中 LLM 间的神经同步现象：通过训练仿射变换预测交互伙伴的未来表征，定义 $SyncR^2$ 指标量化同步强度，发现该同步依赖于社会参与和时间邻近性，且与 LLM 的社会行为表现高度相关（Pearson $r$ = 0.88-0.99）…
tags:
  - "ICLR 2026"
  - "医学图像"
  - "神经同步"
  - "社会交互"
  - "LLM表征分析"
  - "多Agent系统"
  - "脑间同步类比"
  - "可预测性"
---

# Neural Synchrony Between Socially Interacting Language Models

**会议**: ICLR 2026  
**arXiv**: [2602.17815](https://arxiv.org/abs/2602.17815)  
**代码**: [zzn-nzz/LM_neural_synchrony](https://github.com/zzn-nzz/LM_neural_synchrony)  
**领域**: LLM/NLP  
**关键词**: 神经同步, 社会交互, LLM表征分析, 多Agent系统, 脑间同步类比, 可预测性

## 一句话总结

首次研究社会交互中 LLM 间的神经同步现象：通过训练仿射变换预测交互伙伴的未来表征，定义 $SyncR^2$ 指标量化同步强度，发现该同步依赖于社会参与和时间邻近性，且与 LLM 的社会行为表现高度相关（Pearson $r$ = 0.88-0.99），呼应了人类脑间同步（IBS）的神经科学发现。

## 研究背景与动机

### 人类脑间同步（IBS）

神经科学发现：当人类社交互动时（对话、合作、共同注意），脑活动会产生同步。这种**脑间同步**（Inter-Brain Synchrony）不仅是共享感觉输入的副产品，更是预测和促进社会协调、合作和相互理解的功能性机制。更强的 IBS 关联着更高的合作率、更好的学习效果和更佳的团队表现。

### 研究动机

LLM 在行为层面展现出惊人的社会交互能力，但在**表征层面**是否存在类似人类社会脑的内部机制，尚属未知。已有工作主要从行为评估（如 Theory of Mind 测试）或单模型内部分析（如特定注意力头）入手，但缺乏对**多模型交互过程中表征动态**的研究。

### 核心假设

如果 LLM 在社会交互中不仅基于自身角色行事，还推理伙伴的情感、意图和交互走向，那么一个 LLM 的内部表征应该包含可预测另一个 LLM 表征的信息。

## 方法详解

### 整体框架

1. 两个 LLM agent 在 Sotopia 环境中进行多轮社会交互（各有独立背景、人格和目标）
2. 每轮提取 prompt 最后一个 token 的隐藏状态作为表征
3. 训练仿射变换预测交互伙伴的未来表征，量化预测性能

### 表征提取

对于 $T$ 轮对话，每个 LLM backbone $M \in \{A, B\}$ 在第 $t$ 轮产生：

$$\boldsymbol{h}_t^{(M)} \in \mathbb{R}^{L_M \times D_M}, \quad t = 1, \dots, T$$

提取 prompt 输入最后一个 token 位置（整合了所有之前 token 信息）的各层隐藏状态。

### 数据集构建

实验组——来自真实交互的时间对齐表征对：

$$\mathcal{D}^{A \to B}_{l_A \to l_B} = \{(\boldsymbol{h}^{(A)}_{t, l_A}, \boldsymbol{h}^{(B)}_{t, l_B}) \mid t = 1, \dots, T\}$$

### 学习仿射变换

使用 Ridge 回归（含截距项）：

$$\hat{\boldsymbol{W}}, \hat{\boldsymbol{b}} = \arg\min_{\boldsymbol{W}, \boldsymbol{b}} \|\boldsymbol{Y} - \boldsymbol{X}\boldsymbol{W} - \mathbf{1}\boldsymbol{b}\|_F^2 + \lambda \|\boldsymbol{W}\|_F^2$$

正则化系数 $\lambda = 0.1$，截距不正则化。

### $SyncR^2$ 指标定义

对 source 模型 $A$ 的每一层 $l_A$，找到 target 模型 $B$ 中最佳匹配层：

$$r_A^{\star}(l_A) = \max_{l_B} R^2_{\text{test}}(l_A \to l_B), \quad \tilde{r}_A(l_A) = \max\{0, r_A^{\star}(l_A)\}$$

然后在所有层上平均，并双向对称化：

$$SyncR^2(A, B) = \frac{1}{2}(SyncR^2(A \to B) + SyncR^2(B \to A))$$

负 $R^2$ 截断为 0（比预测均值更差意味着无同步）。

### 关键对照实验

**对照组 1（无社会参与）**：引入"被动" agent——只读对话历史，不生成回复，不进行角色扮演。预期同步减弱。

**对照组 2（无时间邻近性）**：将 source 表征与 $k$ 轮后的 target 表征配对（$k \geq 1$）。如果同步仅反映静态表征相似性，则不应随时间延迟衰减。

$$\mathcal{D}^{\text{lag-}k, A \to B}_{l_A \to l_B} = \{(\boldsymbol{h}^{(A)}_{t, l_A}, \boldsymbol{h}^{(B)}_{t+k, l_B}) \mid t = 1, \dots, T-k\}$$

## 实验关键数据

### 实验设置

- **6 个开源模型**：Mistral-7B-v0.1/v0.2/v0.3, Llama-2-7B-Chat, Llama-3-8B, Llama-3.2-3B
- **21 个模型对**：涵盖 intra-family 和 cross-family 配对
- **450 个交互场景**，每个场景最多 8 轮，3 个随机种子
- 每个模型对固定 6,500 个样本

### 主实验：对照组验证

| 条件 | $SyncR^2$ 水平 |
|------|---------------|
| 实验组（真实交互） | 显著高（0.1-0.3+） |
| 对照组1（无社会参与） | 大幅下降 |
| 对照组2（时间延迟 $k \geq 1$） | 迅速崩塌至 ≈0 |

结果确认：神经同步确实依赖于**真实社会参与**和**时间邻近性**。

### 核心发现：同步与社会表现的相关性

| 模型家族类型 | Pearson $r$ | $p$ 值 |
|------------|------------|--------|
| Mistral 家族（3 对） | 0.88 | $< 0.05$ |
| Cross-family（Mistral×Llama） | 0.89 | $< 0.001$ |
| Llama 家族（3 对） | 0.99 | $< 0.001$ |

更同步的模型对系统性地获得更好的社会表现。

### 控制混淆因素

使用 IFEval（指令跟随）和 MuSR（长上下文推理）作为控制变量计算偏相关：

| 模型家族 | 控制 IFEval 后 | 控制 MuSR 后 |
|---------|--------------|-------------|
| Mistral | 0.81 | 0.92 |
| Cross | 0.71 | 0.89 |
| Llama | 0.27 | 0.99 |

控制后相关性仍为正且大多显著，证明同步反映的是**社会特有能力**而非通用能力的副产品。

### 关系亲密度的影响

$SyncR^2$ 分布随 agent 间关系亲密度增加而上移——更亲密的社会关系伴随更强的神经同步，呼应人类神经科学中的发现（如情侣比陌生人有更强 IBS）。

### 关键发现

1. 仿射变换（极简假设）即可有效捕捉同步，非线性变换并未显著提升泛化性能
2. 同步在中间层最为显著
3. LLM 的表征确实编码了交互伙伴的情绪状态，且能预测伙伴未来的情绪和行动分布

## 亮点与洞察

1. **开创性的视角**：首次将神经科学的 IBS 概念转移到 LLM 领域，架起了人类社会认知与 AI 系统之间的桥梁
2. **严谨的对照设计**：社会参与和时间邻近性两个控制组排除了多种替代解释
3. **简约方法的力量**：仅用线性（仿射）变换就能揭示深层的表征同步，支持 LLM 表征的线性结构假说
4. **Theory of Mind 的隐性证据**：agent 的表征编码了伙伴不可见的内部状态（如情绪），暗示了隐式 ToM 能力
5. **与社会预测编码理论的呼应**：仿射变换直接将"对他人未来状态的预测"操作化

## 局限性

1. **模型规模有限**：最大仅 8B 参数，未包含当前主流的大规模模型（如 70B+）
2. **仅用 Sotopia 环境**：社会交互场景的多样性可能不足以代表现实世界的社会互动
3. **因果方向不明**：同步-表现相关是否存在因果关系，还是两者都是模型能力的反映？
4. **仿射变换的限制**：虽然有效，但可能遗漏表征中的非线性同步模式
5. **评估器依赖**：社会表现评分依赖 GPT 模型，可能引入系统偏差

## 相关工作与启发

- **IBS 神经科学**（Dumas et al., 2010; Hasson et al., 2012）：本文的直接灵感来源
- **Brain-LLM 对齐**（Mischler et al., 2024）：证明 LLM 与大脑表征的相似性，但限于单模型
- **Sotopia**（Zhou et al., 2023）：提供社会模拟环境的基础设施
- **启发**：为多 agent 系统的设计提供了新的表征分析视角——可通过优化 agent 间同步来提升协作表现

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 极具原创性，开辟了 LLM 社会神经科学的全新研究方向
- **技术深度**: ⭐⭐⭐⭐ — 方法简洁但对照实验设计精妙，统计分析严谨
- **实验充分度**: ⭐⭐⭐⭐ — 21 个模型对、450 个场景、多种控制条件，但模型规模偏小
- **实用价值**: ⭐⭐⭐ — 目前更偏分析性和启发性，直接应用路径待明确
- **总体推荐**: ⭐⭐⭐⭐ — 非常有趣且有深度的跨学科工作，打开了理解 LLM "社会心智" 的新窗口

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_imaging/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)
- [\[ICLR 2026\] Neuro-Symbolic Decoding of Neural Activity](neuro-symbolic_decoding_of_neural_activity.md)
- [\[ICLR 2026\] SONIC: Spectral Oriented Neural Invariant Convolutions](sonic_spectral_oriented_neural_invariant_convolutions.md)

</div>

<!-- RELATED:END -->
