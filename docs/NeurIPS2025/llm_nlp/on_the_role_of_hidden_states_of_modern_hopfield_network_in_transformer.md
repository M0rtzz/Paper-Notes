---
title: >-
  [论文解读] On the Role of Hidden States of Modern Hopfield Network in Transformer
description: >-
  [NeurIPS 2025][LLM 其他][Modern Hopfield Network] 本文突破现代 Hopfield 网络（MHN）与 Transformer 对应关系的绝热近似限制，发现保留 MHN 的隐状态动力学会在自注意力层中引入跨层注意力分数传播机制（Modern Hopfield Attention, MHA），不增加训练参数即可系统性改善 ViT 和 GPT-2 的性能，并从理论和实验上证明 MHA 有效缓解了深层 Transformer 的 rank collapse 问题。
tags:
  - "NeurIPS 2025"
  - "LLM 其他"
  - "Modern Hopfield Network"
  - "注意力机制"
  - "rank collapse"
  - "token uniformity"
  - "hidden state"
  - "Transformer"
  - "GPT-2"
---

# On the Role of Hidden States of Modern Hopfield Network in Transformer

**会议**: NeurIPS 2025  
**arXiv**: [2511.20698](https://arxiv.org/abs/2511.20698)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: Modern Hopfield Network, self-attention, rank collapse, token uniformity, hidden state, Vision Transformer, GPT-2

## 一句话总结

本文突破现代 Hopfield 网络（MHN）与 Transformer 对应关系的绝热近似限制，发现保留 MHN 的隐状态动力学会在自注意力层中引入跨层注意力分数传播机制（Modern Hopfield Attention, MHA），不增加训练参数即可系统性改善 ViT 和 GPT-2 的性能，并从理论和实验上证明 MHA 有效缓解了深层 Transformer 的 rank collapse 问题。

## 研究背景与动机

Hopfield 网络与 Transformer 的关系是近年来的热门研究方向：

**已有对应关系**：Ramsauer et al. 和 Krotov & Hopfield 证明了现代连续 Hopfield 网络（MCHN）在**绝热极限**（$\tau_h \approx 0$）下的状态更新规则恰好等价于 Transformer 的自注意力机制

**绝热近似的局限**：绝热极限消除了隐状态 $\bm{h}$ 的动力学（直接令 $\bm{h}_n = \bm{x}_n \bm{W}_2$），但一般 MCHN 同时包含可见状态 $\bm{x}$ 和隐状态 $\bm{h}$ 两个动态变量，后者的物理意义和对 Transformer 的启示尚未被探索

**Transformer 的 rank collapse 困境**：随着 Transformer 深度增加，token 表征趋同（余弦相似度趋近 1），导致信息多样性丧失。Dong et al. 证明纯注意力网络的 rank 以双指数速度衰减

**核心问题**：如果不做绝热近似，MHN 的隐状态对应 Transformer 中的什么结构？这个结构能否改善 Transformer 的性能？

## 方法详解

### 整体框架

从 MCHN 的连续时间动力学出发，通过精确离散化（不忽略离散化步长的影响），推导出包含隐状态的新型注意力机制 MHA，并将其作为 Transformer 自注意力层的替代品。

**MCHN 的动力学方程**：

$$\tau_v \frac{d\bm{x}}{dt} = \bm{f}(\bm{h})\bm{W}_1^\top - \bm{x}, \quad \tau_h \frac{d\bm{h}}{dt} = \bm{g}(\bm{x})\bm{W}_2 - \bm{h}$$

其中 $\bm{x}$ 为可见状态（feature neuron），$\bm{h}$ 为隐状态（memory neuron），$\tau_{v,h}$ 为时间常数。

**精确离散化**：引入参数 $\alpha = 1 - \Delta t / \tau_v$ 和 $\alpha' = 1 - \Delta t / \tau_h$：

$$\bm{x}_{n+1} = \alpha \bm{x}_n + (1-\alpha) \bm{f}(\bm{h}_n) \bm{W}_1^\top$$
$$\bm{h}_{n+1} = \alpha' \bm{h}_n + (1-\alpha') \bm{g}(\bm{x}_n) \bm{W}_2$$

### 关键设计

**从绝热极限到完整动力学**：

- **绝热极限**（$\alpha' = 0$）：$\bm{h}_n = \bm{x}_n \bm{W}_2$，代入后恢复标准自注意力：$\bm{x}_{n+1} = \text{softmax}(\bm{q}_n \bm{K}_n^\top) \bm{V}_n$
- **保留隐状态**（$\alpha' \neq 0$）：得到 **Modern Hopfield Attention (MHA)**：

$$\bm{x}_{n+1} = \alpha \bm{x}_n + (1-\alpha) \text{softmax}(\bm{h}_n) \bm{V}_n$$
$$\bm{h}_n = \alpha' \bm{h}_{n-1} + (1-\alpha') \bm{q}_n \bm{K}_n^\top$$

**MHA 的物理意义**：

- 隐状态 $\bm{h}_n$ 以**指数滑动平均**的形式跨层累积注意力分数 $\bm{Q}_\ell \bm{K}_\ell^\top$
- softmax 不再作用于当前层的注意力分数，而是作用于**累积的隐状态**，实现了注意力信息从浅层到深层的传递
- $\alpha$ 控制残差连接强度，$\alpha'$ 控制历史注意力分数的记忆程度
- **不增加任何训练参数**，计算复杂度增量仅为 $O(T^2)$（对比自注意力的 $O(dT^2)$，$d \gg 1$）

### 损失函数 / 理论分析

**Rank collapse 的缓解**：对于纯注意力网络（无 skip connection），标准自注意力的 rank 衰减上界为（Dong et al.）：

$$\|\text{Res}(\text{AttnNet}(\bm{X}))\|_{1,\infty} \leq (rC)^{\frac{3^L-1}{2}} \|\text{Res}(\bm{X})\|_{1,\infty}^{3^L}$$

即 rank 以 $3^L$ 的双指数衰减。而 MHA 将衰减改善为：

$$\|\text{Res}(\text{AttnNet}(\bm{X}))\|_{1,\infty} \leq \max_{m=0}^{L} (r(1-\alpha')C_1)^{\frac{3^m-1}{2}} (r\alpha' C_2)^{3^m(L-m)} \|\text{Res}(\bm{X})\|_{1,\infty}^{3^m}$$

当 $m=0$ 项主导时，衰减退化为线性衰减 $(r\alpha' C_2)^L$，**从双指数降为线性**，本质原因是隐状态的线性传播项打断了 softmax 导致的非线性压缩链。

## 实验关键数据

### 主实验：GPT-2 文本生成（WikiText-103 困惑度）

| 模型 | 标准注意力 | MHA (α=0.5) |
|------|-----------|-------------|
| GPT-2 Small (124M) | 22.87 | **20.70** |
| GPT-2 Medium (350M) | 20.85 | **19.61** |

### 主实验：LLaMA 架构文本生成（困惑度）

| 数据集 | 标准注意力 | MHA (α=0.5) |
|--------|-----------|-------------|
| WikiText-103 | 14.49 | **14.29** |
| CNN DailyMail | 19.36 | **18.97** |
| BookCorpus | 23.76 | **23.50** |

### 主实验：ViT 图像分类（CIFAR-100 准确率）

| 模型 | 标准注意力 | MHA (α=0.5) | MHA (α=0.7) |
|------|-----------|-------------|-------------|
| ViT-Tiny (5.5M) | **73.08** | 72.03 | 72.57 |
| ViT-Small (22M) | 74.49 | 75.42 | **75.59** |
| ViT-Base (86M) | 75.36 | **76.22** | 75.59 |
| ViT-Large (303M) | 72.91 | **75.78** | 75.37 |

### 主实验：ViT-Base ImageNet-1k

| 方法 | Top-1 准确率 |
|------|-------------|
| 标准注意力 | 76.07 |
| MHA (α=0.5) | 76.43 |
| MHA (α=0.7) | **77.06** |

### 消融实验：α 与 α' 的独立效果（ViT-Tiny CIFAR-100）

| 实验设置 | 关键观察 |
|---------|---------|
| 固定 α=0.5，α'=0→1 | α'=0.2 达到峰值 72.29，α'=1.0 崩溃至 66.10 |
| 固定 α'=0.5，α=0→1 | α=0.6 达到峰值 72.66，α=1.0 崩溃至 1.00（全为 skip） |
| α=0（无残差连接）| 性能降至 69.89，说明两个参数都是必要的 |
| α'=0（无隐状态）| 性能降至 71.16，回退到接近标准注意力 |

### 消融实验：无 skip connection 网络随深度变化

| 深度 | 标注注意力 (CIFAR-10 / CIFAR-100) | MHA (CIFAR-10 / CIFAR-100) |
|------|----------------------------------|---------------------------|
| 1 | 55.08 / 30.90 | 65.41 / 40.08 |
| 2 | 63.72↑ / 40.06↑ | 79.75↑ / 56.94↑ |
| 4 | 57.38↓ / 32.25↓ | **85.74↑** / **64.39↑** |
| 8 | 48.59↓ / 17.19↓ | 80.34↓ / 49.90↓ |
| 12 | 10.00↓ / 1.00↓ | 10.00↓ / 1.00↓ |

标准注意力在深度 4 即崩溃，MHA 在深度 4 仍在上升，展示了对 rank collapse 的显著抑制。

### 关键发现

1. **MHA 带来系统性改进**：在 GPT-2、LLaMA、ViT 三种架构、五个数据集上均观察到性能提升，且不增加参数
2. **模型越大效果越明显**：ViT-Large 在 CIFAR-100 上提升 2.87%（72.91→75.78），而 ViT-Tiny 几乎无变化
3. **α 和 α' 协同必要**：两个参数分别控制残差连接和注意力记忆，缺一不可
4. **理论预测与实验吻合**：MHA 将 rank 衰减从双指数改善为线性，无 skip connection 实验直接验证了这一点
5. **迁移学习效果显著**：ImageNet 预训练的 MHA-ViT 在 4 个下游数据集中 3 个大幅超越标准 ViT

## 亮点与洞察

- **物理直觉优美**：从联想记忆的隐状态动力学自然推导出注意力分数的跨层传播，而非人为设计
- **零参数增加的改进**：MHA 仅增加 $O(T^2)$ 计算（相对 $O(dT^2)$ 可忽略），却带来一致性能提升
- **理论与实验的深度结合**：rank collapse 的理论分析（从双指数到线性衰减）在消融实验中得到定量验证
- **Hopfield 网络的新价值**：证明了 Hopfield 网络不仅是 Transformer 的理论解释工具，还能直接指导架构改进
- **简洁的实现**：只需用指数滑动平均替换注意力分数计算 + 修改 skip connection 权重，工程改动极小

## 局限性

1. **超参数 α 需要调整**：虽然 α=0.5 在多数情况下有效，但不同任务的最优 α 不同（GPT-2 用 0.5，ViT 的某些任务用 0.7 更好）
2. **大规模验证不足**：仅在 GPT-2 (124M-350M) 和 ViT (5.5M-303M) 上验证，未扩展到现代大模型（如 7B+ LLM）
3. **编码器 vs 解码器差异未深入**：GPT-2 使用因果注意力，ViT 使用双向注意力，MHA 在两者上的行为可能有本质不同
4. **能量函数未定义**：打破了 $\bm{W}_1$ 和 $\bm{W}_2$ 的对称性假设，导致 MHA 没有单调递减的能量函数，理论收敛性存疑
5. **与其他注意力改进的组合未探索**：如 FlashAttention、Group Query Attention 等工程优化的兼容性未验证
6. **任务过于接近饱和**：CIFAR-10 的准确率已接近上限，难以展示 MHA 的效果

## 相关工作与启发

- **Ramsauer et al. (2021)**：建立 MHN 与 Transformer 的绝热极限对应，本文的直接理论基础
- **Krotov & Hopfield (2021)**：提出 Dense Associative Memory 的通用理论框架，推导了绝热极限的公式化
- **RealFormer (He et al., 2021)**：从工程角度提出跨层注意力分数复用，但仅限于编码器且缺乏理论基础；MHA 从 Hopfield 网络自然推导出更一般的机制
- **Dong et al. (2021)**：rank collapse 的理论分析，本文直接在其框架上推导 MHA 的改善效果
- **启发**：物理模型（联想记忆）可以为深度学习架构设计提供系统性指导，而非仅作为事后解释；保留而非简化动力系统的完整动力学可能蕴含未被挖掘的架构改进

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从 Hopfield 隐状态推导出实用注意力改进，理论视角独到且优美
- 实验充分度: ⭐⭐⭐⭐ 覆盖 NLP（GPT-2、LLaMA）和 CV（ViT），消融详尽，但缺乏大规模验证
- 写作质量: ⭐⭐⭐⭐ 推导严谨，理论与实验结合紧密，但部分符号密集
- 价值: ⭐⭐⭐⭐ 提供了 Hopfield-Transformer 关联的新深度，rank collapse 缓解具有实际意义，但需大模型验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](../../ACL2025/llm_nlp/mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
