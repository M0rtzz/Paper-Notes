---
title: >-
  [论文解读] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training
description: >-
  [NeurIPS 2025][scaling laws] 提出了一套针对 LLM 预训练中权重衰减 $\lambda$ 和批大小 $B$ 的幂律缩放定律（power laws），通过 AdamW 时间尺度 $\tau$ 的概念统一了超参数缩放关系，使得在大规模训练前即可准确预测最优超参数。
tags:
  - NeurIPS 2025
  - scaling laws
  - weight decay
  - batch size
  - LLM pre-training
  - AdamW
---

# Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training

**会议**: NeurIPS 2025  
**arXiv**: [2505.13738](https://arxiv.org/abs/2505.13738)  
**代码**: 待确认  
**领域**: llm_nlp  
**关键词**: scaling laws, weight decay, batch size, LLM pre-training, AdamW

## 一句话总结

提出了一套针对 LLM 预训练中权重衰减 $\lambda$ 和批大小 $B$ 的幂律缩放定律（power laws），通过 AdamW 时间尺度 $\tau$ 的概念统一了超参数缩放关系，使得在大规模训练前即可准确预测最优超参数。

## 研究背景与动机

大语言模型（LLM）的预训练效果高度依赖超参数调优，包括学习率 $\eta$、权重衰减 $\lambda$ 和批大小 $B$。然而在最前沿的训练规模下，根本没有做超参数搜索的余裕。现有工作大多关注学习率和批大小的缩放关系：

- **DeepSeek LLM** 拟合了 $B_{\text{opt}}$ 和 $\eta_{\text{opt}}$ 关于总计算量 $C$ 的幂律
- **μP (Maximal Update Parameterization)** 允许最优基础学习率在不同模型宽度间迁移
- **Wang et al.** 提出 AdamW 时间尺度 $\tau_{\text{epoch}}$ 在图像任务上保持稳定

但存在几个核心缺口：

**权重衰减 $\lambda$ 的缩放几乎无人研究** — 实践中通常直接设 $\lambda=0.1$
2. 依赖唯一最优 $B_{\text{opt}}$ 缺乏灵活性 — 无法适应硬件约束和时间/计算的 trade-off
3. 尚不清楚 $B_{\text{opt}}$ 和 $B_{\text{crit}}$ 究竟应该用 $C$、$L$、$N$ 还是 $D$ 来解释

本文旨在用统一的幂律框架回答上述问题，覆盖 compute-optimal 和 overtrained 两种训练范式。

## 方法详解

### 整体框架

核心思想是将 AdamW 的权重衰减和学习率统一为一个 **「AdamW 时间尺度」$\tau$**，然后研究 $\tau$ 如何随模型规模和数据量缩放。

**AdamW 的 EMA 视角**：AdamW 的更新可以写成指数移动平均的形式：

$$\theta_t = (1 - \eta\lambda)\theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中参数 $\theta_t$ 可视为权重更新的 EMA，平滑系数 $\alpha = \eta\lambda$，对应迭代时间尺度 $\tau_{\text{iter}} = 1/(\eta\lambda)$。

**归一化时间尺度 $\tau$**：由于 LLM 预训练只用一遍数据，本文将时间尺度归一化为：

$$\tau = \frac{B}{\eta\lambda D}$$

其中 $S = D/B$ 为总优化步数。$\tau$ 反映了最终权重中过去迭代信息被纳入的比例。

### 关键设计

**发现 1：$\tau_{\text{opt}}$ 关于 tokens-per-parameter (TPP) 满足幂律**

$$\tau_{\text{opt}}(\text{TPP}) = c_\tau \cdot \text{TPP}^{m_\tau}, \quad \text{TPP} = D/N$$

拟合结果 $R^2 = 0.975$，bootstrap 得到 $m_\tau$ 的 10th/90th 百分位为 (-0.529, -0.507)，极为稳定。这意味着从 compute-optimal（~20 TPP）到 overtrained（1000+ TPP）模型，$\tau_{\text{opt}}$ 从 ~1.0 下降到 ~0.01。

**由此推导 $\lambda_{\text{opt}}$**：

$$\lambda_{\text{opt}} = \frac{B \cdot \text{TPP}^{-m_\tau}}{c_\tau \cdot \eta \cdot D}$$

给定任意 $N$、$D$、$B$，即可提前准确预测最优权重衰减。

**发现 2：$\lambda_{\text{opt}}$ 随 $B$ 线性缩放（当 $B \leq B_{\text{crit}}$）**

在 610M 模型、20 TPP 设置下验证：当 $B \in [63, 2016]$ 时，$\tau_{\text{opt}}$ 稳定在 $\sim$0.21 附近，即 $\lambda_{\text{opt}}$ 与 $B$ 成线性关系。当 $B > B_{\text{crit}}$ 后，这一关系开始漂移。

**发现 3：$B_{\text{opt}}$ 和 $B_{\text{crit}}$ 都关于 $D$ 满足幂律**

$$B_{\text{crit}}(D_{\min}) = c_{B_{\text{crit}}} \cdot D_{\min}^{m_{B_{\text{crit}}}}$$

这与 prior work 中关于 $C$ 或 $L$ 的拟合形成对比 — 作者说明那些关系只在固定 TPP 下成立，根本关系在 $D$。

### 估算 $B_{\text{crit}}$ 的新方法

核心创新是一个不依赖特定 LR schedule 或优化器的通用方法：

1. 对每个 $B$，在不同 $D$ 下训练，拟合 batch-specific 幂律 $L_B(D) = E_N + D_{\text{const}} \cdot D^{-\beta}$
2. 用拟合曲线插值得到达到目标 loss $\hat{L}$ 所需的 $D_B$
3. 在 $\langle D_B, S = D/B \rangle$ 对上拟合 McCandlish 公式，从而得到 $B_{\text{crit}} = D_{\min}/S_{\min}$

### 损失函数 / 训练策略

- 使用 GPT2-like 架构，含 ALiBi 位置编码和 SwiGLU 激活
- 训练数据：SlimPajama，验证集 1.1B tokens
- 优化器：AdamW + μP，线性 LR schedule（10% warmup 后线性衰减至零）
- $\lambda$ 在每个 $(N, D, B)$ 下以 $2\times$ 间隔进行 sweep

## 实验关键数据

### 主实验

| 实验设置 | 关键发现 |
|---------|---------|
| $\tau_{\text{opt}}$ vs TPP 幂律 | $R^2 = 0.975$，跨 3 个数量级计算量泛化良好 |
| $\lambda$ vs $B$ 线性关系 (610M, 20TPP) | $B \in [63, 2016]$ 时 $\tau_{\text{opt}} \approx 0.21$ |
| $B_{\text{opt}}$ vs $D$ | 幂律拟合显著优于 vs $C$ 的拟合（$C$ 的拟合有虚假 $N$ 依赖） |
| $B_{\text{crit}}$ vs $D$ | 幂律拟合质量高，与 Zhang et al. 在 302M 上的结果一致 |

**调 $\lambda$ vs 调 $\eta$ 对比（610M, 20TPP）**：

| $B$ | 固定 $\eta=0.016$ 调 $\lambda$ | 固定 $\lambda=0.1$ 调 $\eta$ |
|-----|------|------|
| 63 | 2.583 | 2.579 |
| 126 | 2.570 | 2.565 |
| 252 | **2.563** | **2.563** |
| 504 | **2.571** | 2.570 |
| 2016 | **2.625** | 2.637 |
| 4032 | **2.754** | 2.733 |

在 8 个 $B$ 值中，调 $\lambda$ 在 6 个上严格优于调 $\eta$。关键原因：$\eta$ 存在导致训练不稳定的上限，灵活性不如 $\lambda$。

### 消融实验

- **111M, 200TPP 模型**：默认 HP loss = 2.810，调 $\eta$ = 2.808，调 $\lambda$ = 2.805 — 差异虽小但调 $\lambda$ 始终更优
- **泛化验证**：4 个「盲测」点（未参与拟合，但规模相差 1000×）完美落在幂律曲线上
- **$\tau$ 对 $D$ 不恒定**：反驳了 Wang et al. 在多 epoch 训练中 $\tau$ 恒定的结论 — 在 LLM 单 epoch 预训练中 $\tau_{\text{opt}}$ 是随 TPP 变化的

### 关键发现

1. 权重衰减比学习率更适合作为缩放超参数 — 线性关系清晰且不导致训练不稳定
2. $B_{\text{opt}}$ 和 $B_{\text{crit}}$ 与 $D$ 的关系是基本关系，之前报道的与 $C$ 或 $L$ 的关系是从属的
3. Pareto 前沿分析表明：**小模型 + 大量过训练**可能同时提供更快的步速和更大的并行度

## 亮点与洞察

1. **统一视角**：AdamW 时间尺度 $\tau$ 将 $\eta$、$\lambda$、$B$、$D$ 统一在一个框架中，极大简化了超参数选择
2. **实用价值极高**：给出可直接使用的 $\lambda_{\text{opt}}$ 预测公式，无需在大规模下做超参数搜索
3. **反直觉发现**：$\lambda$ 而非 $\eta$ 才是应该随 $B$ 和 $D$ 缩放的超参数
4. **$B_{\text{crit}}$ 估算新范式**：不依赖特定 LR schedule，可用于任意优化器和 schedule
5. **所有实验在 Cerebras CS-3 上完成**，数百模型的大规模验证确保了结论的可靠性

## 局限与展望

1. 仅使用 GPT2-like 架构和 SlimPajama 数据，未验证在其他架构（如 MoE）或数据上的适用性
2. 结论依赖 μP 参数化 — 对不使用 μP 的训练设置，幂律系数可能不同
3. 仅研究了线性 LR schedule，未覆盖余弦衰减等常用 schedule
4. $B > B_{\text{crit}}$ 时的 $\lambda$ 漂移现象缺乏深入分析和建模
5. 未涉及训练的后期阶段（如 cooldown phase）对 $\tau$ 的影响

## 相关工作与启发

- **Chinchilla (Hoffmann et al.)**: 建立了 $N$-$D$ compute-optimal 关系，本文在此基础上进一步建立了超参数的缩放关系
- **DeepSeek Scaling**: 将 $B_{\text{opt}}$ 和 $\eta_{\text{opt}}$ 拟合为 $C$ 的幂律，本文指出 $D$ 才是更根本的变量
- **μP (Yang et al.)**: 为跨宽度的 LR 迁移提供理论基础，本文扩展到跨 $D$ 和 $B$ 的 $\lambda$ 迁移
- **Wang et al. (EMA 视角)**: 首先提出 AdamW 的 EMA 解释和 $\tau_{\text{epoch}}$，本文发现 $\tau$ 在 LLM 中并非常数

**对未来工作的启发**：这套幂律框架可期待扩展到 MoE 模型、多模态模型，以及与其他 schedule（WSD、trapezoidal）的结合。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究权重衰减缩放，$\tau$ 的 EMA 统一视角新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 数百个模型训练，跨 3 个数量级 FLOPs 验证，bootstrap 统计分析严谨
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 直接给出可使用的 $\lambda_{\text{opt}}$ 预测公式，对工业界大规模训练有即时实用价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](../../ACL2025/llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](scaling_embedding_layers_in_language_models.md)

<!-- RELATED:END -->
