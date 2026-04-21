---
title: >-
  [论文解读] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss
description: >-
  [ICLR 2026][视频理解][Mixture-of-Experts] 提出 Expert-Router Coupling (ERC) Loss，一种轻量级辅助损失函数，通过将路由器参数视为聚类中心的代理 token 并约束专家对其激活范数，实现路由器决策与专家能力的紧密耦合，仅需 $n^2$ 次激活计算即可显著提升 MoE-LLM 性能。
tags:
  - ICLR 2026
  - 视频理解
  - Mixture-of-Experts
  - 路由-专家耦合
  - 辅助损失
  - 专家特化
  - 大语言模型
---

# Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss

**会议**: ICLR 2026  
**arXiv**: [2512.23447](https://arxiv.org/abs/2512.23447)  
**代码**: 无  
**领域**: 模型架构 / MoE  
**关键词**: Mixture-of-Experts, 路由-专家耦合, 辅助损失, 专家特化, 大语言模型

## 一句话总结

提出 Expert-Router Coupling (ERC) Loss，一种轻量级辅助损失函数，通过将路由器参数视为聚类中心的代理 token 并约束专家对其激活范数，实现路由器决策与专家能力的紧密耦合，仅需 $n^2$ 次激活计算即可显著提升 MoE-LLM 性能。

## 研究背景与动机

MoE（Mixture-of-Experts）是现代大语言模型的核心架构，通过路由器为每个 token 选择 top-K 专家处理，以稀疏激活实现高效的参数扩展。然而，传统 MoE 存在一个根本性问题：**路由器和专家之间缺乏显式约束来确保路由决策与专家实际能力对齐**。

具体而言：
- 路由器是一个线性分类器 $\mathbf{R} \in \mathbb{R}^{n \times d}$，通过内积 $\text{softmax}(\mathbf{x}\mathbf{R}^\top)$ 决定 token 分配
- 专家是独立的 FFN 模块，有自己的参数 $\mathbf{W}_g, \mathbf{W}_p, \mathbf{W}_o$
- 路由器无法直接获取专家参数（因此不知道专家真实能力），只能通过试错学习路由策略
- 这常导致"误路由"——token 被发送到不擅长处理它的专家，产生的梯度反而干扰专家的特化

先前的解决方案 Autonomy-of-Experts (AoE) 通过让所有专家部分处理每个 token 来获取路由信号，但这导致了远超标准 MoE 的计算和内存开销（训练时间增加 1.6×，内存增加 1.3×），且开销随 token 数量线性增长。

## 方法详解

### 整体框架

ERC Loss 的核心思想来自一个优雅的观察：路由器参数矩阵 $\mathbf{R}$ 的每行 $\mathbf{R}[i]$ 可以被解释为分配给专家 $i$ 的 token 集合 $\mathcal{X}_i$ 的**聚类中心**。因此，$\mathbf{R}[i]$ 可以作为 $\mathcal{X}_i$ 中 token 的代理（proxy），用于探测专家 $i$ 的响应性，而无需将所有 token 送入所有专家。

### 关键设计

1. **代理 Token 生成（Step 1: 噪声注入）**:

    - 对每个聚类中心 $\mathbf{R}[i]$ 施加有界乘性随机噪声得到 $\tilde{\mathbf{R}}[i] = \mathbf{R}[i] \odot \boldsymbol{\delta}_i$
    - 噪声 $\boldsymbol{\delta}_i \sim \mathcal{U}(1-\epsilon_i, 1+\epsilon_i)^d$ 模拟 $\mathcal{X}_i$ 内的 token 变化
    - **噪声边界推导**: $\epsilon_i \leq \frac{\|\mathbf{R}[i] - \mathbf{R}[j]\|}{2\|\mathbf{R}[i]\|}$（$j$ 为最近邻聚类中心），确保扰动后的代理不越过聚类边界
    - $\epsilon_i$ 在每层每步动态计算，反映训练过程中聚类的变化
    - **关键**: 被扰动的 $\tilde{\mathbf{R}}$ 仅用于损失计算，实际路由仍使用原始 $\mathbf{R}$

2. **激活矩阵计算（Step 2: 探测专家响应）**:

    - 将每个代理 token $\tilde{\mathbf{R}}[i]$ 送入所有 $n$ 个专家的 $\mathbf{W}_g$ 参数
    - 构造 $n \times n$ 的激活矩阵: $\mathbf{M}[i,j] = \|\tilde{\mathbf{R}}[i] \cdot \mathbf{W}_g^j\|$
    - $\mathbf{M}[i,j]$ 反映专家 $j$ 对分配给专家 $i$ 的代理 token 的响应强度
    - 选择 $\mathbf{W}_g$ 的激活范数而非最终输出，因为实验表明 $\mathbf{W}_g$ 的中间激活最有效

3. **ERC 损失函数（Step 3: 双向约束）**:
    $\mathcal{L}_{\text{ERC}} = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j \neq i}^{n} \left(\max(\mathbf{M}[i,j] - \alpha \mathbf{M}[i,i], 0) + \max(\mathbf{M}[j,i] - \alpha \mathbf{M}[i,i], 0)\right)$
   
   两个约束项的含义：
    - **约束 1** ($\mathbf{M}[i,j] < \alpha \mathbf{M}[i,i]$): **专家特化** — 代理 token $\tilde{\mathbf{R}}[i]$ 在专家 $i$ 上的激活必须显著强于在其他专家上的激活，确保专家 $i$ 针对其分配的 token 族进行了特化
    - **约束 2** ($\mathbf{M}[j,i] < \alpha \mathbf{M}[i,i]$): **精准路由** — 专家 $i$ 对自身代理 $\tilde{\mathbf{R}}[i]$ 的响应必须高于对其他代理 $\tilde{\mathbf{R}}[j]$ 的响应，确保 $\mathbf{R}[i]$ 准确代表了专家 $i$ 的能力

4. **超参数 $\alpha$ 的控制作用**:

    - $\alpha \in [0, 1]$ 控制耦合强度
    - $\alpha \to 0$: 鼓励 $\mathbf{R}[i]$ 与其他专家参数正交，最大化特化
    - $\alpha \to 1$: 放松约束，允许专家之间更多的重叠
    - $\alpha$ 同时是探索专家特化程度的工具——通过对比不同 $\alpha$ 下的性能，可以找到特化与协作的最优平衡

### 效率分析

- **计算开销**: 仅需 $2n^2 D d$ 额外 FLOPs，与 token 数量 $T$ 无关
- **实际影响**: 3B 模型 ($n=64$) 增加 0.18%，15B 模型 ($n=256$) 增加 0.82%
- **对比 AoE**: AoE 增加 $2T(n-K)dr$ FLOPs，随 token 数线性增长
- **推理零开销**: ERC Loss 仅在训练时使用

## 实验关键数据

### 主实验（3B 参数模型）

- 64 experts, $K=8$, 500B tokens 训练
- ERC Loss 显著优于 vanilla MoE，缩小与 AoE 的性能差距
- AoE 需要 ~1.6× 训练时间和 ~1.3× 内存

### 15B 参数模型扩展

| 基准 | MoE | MoE + ERC | 提升 |
|------|-----|-----------|------|
| ARC-C | 63.2 | 64.6 | +1.4 |
| HellaSwag | 67.5 | 69.0 | +1.5 |
| MMLU | 31.0 | 31.9 | +0.9 |
| MMLU-Pro | 42.0 | 44.2 | +2.2 |
| BBH | 44.3 | 45.6 | +1.3 |
| MATH | 25.7 | 26.1 | +0.4 |
| GSM8K | 45.2 | 45.8 | +0.6 |
| 平均 | 47.2 | 49.1 | **+1.9** |

AoE 在 15B 规模下因成本过高无法训练。

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| 不同 $\alpha$ 值 | $\alpha=1$ 在 3B ($n=64$) 最优；$\alpha=0.5$ 在 15B ($n=256$) 最优 |
| 去除噪声 $\boldsymbol{\delta}$ | 性能大幅下降，耦合过拟合到 $\mathbf{R}$ 本身 |
| 仅路由器正交化 | 有限增益，因基线路由器本已近乎正交（余弦相似度 0.15） |
| $\alpha > 1$ | $\alpha=2$ 有限提升，$\alpha=3$ 几乎无效果 |
| 不同激活选择 | $\tilde{\mathbf{R}} \mathbf{W}_g$ 效果最好 |

### 关键发现

- **特化-协作权衡**: 追求极端特化并不可取，存在最优特化度。较小的 $n$ 偏好通才专家，较大的 $n$ 支持更高特化。3B 模型 ($n=64$) 的最优 $\alpha=1$，15B 模型 ($n=256$) 的最优 $\alpha=0.5$
- **噪声边界 $\epsilon$ 作为特化度量**: $\epsilon$ 与 $\alpha$ 强相关，可在训练过程中量化追踪专家特化程度的变化
- **t-SNE 可视化**: vanilla MoE 的专家参数没有形成有意义的聚类，而加入 ERC Loss 后聚类显著清晰
- **参数范数分析**: 模型通过学习有意义的耦合来降低 ERC Loss，而非简单操纵参数范数

## 亮点与洞察

1. **聚类视角的优雅设计**: 将路由器参数视为聚类中心、用聚类中心做代理来探测专家能力，这一观察既简洁又有力，避免了将所有 token 送入所有专家的高开销
2. **固定成本 vs. 可变成本**: $O(n^2)$ 的计算量与 batch size 无关，对于每批百万 token 的预训练场景，这个固定成本可以忽略不计
3. **特化度的可控探索**: $\alpha$ 既是训练参数又是探索工具，$\epsilon$ 提供量化度量，这为理解 MoE 行为提供了新视角
4. **挑战传统观点**: 实验证明"更多特化并不总是更好"，在小规模 MoE 中过度特化反而有害

## 局限与展望

1. **$\alpha$ 需要手动调节**: 不同模型配置（$n$, $K$, 深度）的最优 $\alpha$ 不同，目前缺乏自动确定方法
2. **线性路由器假设**: 聚类中心的解释依赖于 softmax 线性路由器，对非线性路由机制的适用性未探讨
3. **未与共享专家机制结合测试**: DeepSeek 等使用的共享专家可能改变最优特化度
4. **仅在预训练中验证**: 对微调和持续学习场景的效果未知
5. **缺乏与更多 MoE 变体的对比**: 如 Megablocks、Switch Transformer 等

## 相关工作与启发

- **Autonomy-of-Experts (AoE)** (Lv et al., 2025): 将路由编码进专家参数，通过激活范数选路，效果好但开销大，本文方法可视为 AoE 的轻量替代
- **Switch Transformer** (Fedus et al., 2022): 引入负载均衡损失，ERC Loss 与之兼容（负载均衡差异在 $10^{-5}$ 量级）
- **OLMoE**: 本文实现基于此开源 MoE 框架
- **DeepSeek-MoE**: 引入共享专家促进特化，与 ERC Loss 的方向互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 聚类视角 + 代理 token 探测是巧妙设计，固定成本的辅助损失实用性强
- 实验充分度: ⭐⭐⭐⭐⭐ — 从 3B 到 15B，丰富的消融和分析，特化度探索出色
- 写作质量: ⭐⭐⭐⭐⭐ — 思路清晰，三步框架直观，附录详尽
- 价值: ⭐⭐⭐⭐⭐ — 实用、高效、可推广，直接改善 MoE 预训练，代码实现简洁

<!-- RELATED:START -->

## 相关论文

- [Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](../../ECCV2024/video_understanding/occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)
- [QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](../../CVPR2025/video_understanding/question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)
- [FuncBenchGen: 面向可靠基准测试的无污染可控评估框架](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)

<!-- RELATED:END -->
