---
title: >-
  [论文解读] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model
description: >-
  [ICLR 2026][文本生成][离散扩散模型] 提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。
tags:
  - ICLR 2026
  - 文本生成
  - 离散扩散模型
  - 少步采样
  - flow matching
  - 累积标量
---

# FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model

**会议**: ICLR 2026  
**arXiv**: [2509.20624](https://arxiv.org/abs/2509.20624)  
**代码**: [GitHub](https://github.com/apple/ml-fs-dfm)  
**领域**: 模型压缩 / LLM效率  
**关键词**: 离散扩散模型, 少步采样, flow matching, 累积标量, 文本生成

## 一句话总结

提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。

## 研究背景与动机

自回归语言模型（ARM）生成质量优秀但串行性强——每生成一个 token 需要一次前向传播，限制了吞吐量。扩散语言模型（DLM）可以跨位置并行生成，但标准离散扩散通常需要数百到数千步模型评估才能达到高质量，本质上是用迭代深度换取了并行宽度。

具体来说：
- **ARM 的瓶颈**：严格串行，每个 token 一次前向传播，长序列延迟高
- **DLM 的瓶颈**：虽然并行但步数多。例如 LLaDA 需要大约每个输出 token 一步推理才能匹配 ARM 质量；DFM（Discrete Flow-Matching）在 1024-token 生成时需要约 1024 步

本文的目标是让 DLM 在少步（如 8 步）下就能达到多步（1024 步）的生成质量。先前的少步方法（如 SDTT）仍需 16-256 步且仅针对短文本。FS-DFM 是首个针对长文本（1024 tokens）的少步离散 flow-matching 方法。

## 方法详解

### 整体框架

FS-DFM 建立在 Discrete Flow-Matching（DFM）之上。DFM 将文本建模为连续时间离散马尔可夫链（CTMC），学习一个概率速度场将源分布（如 uniform 或 mask）沿概率路径传输到目标数据分布。生成时，从噪声序列出发，沿学习到的速度场迭代去噪。

FS-DFM 的两大支柱：

1. **步数感知 DFM 训练**：将步数预算 h 作为显式条件信号注入模型，并通过 shortcut teacher 蒸馏实现大步一致性
2. **累积标量更新**：替代瞬时标量，在有限步间隔上积分 scheduler 的速率函数，确保大步更新也能提供正确的概率流量

### 关键设计

**1. 步数感知离散 Flow-Matching**

将步数预算 h 作为显式条件信号：模型输出 logits = theta(x_t, t; h)。训练时从对数网格 h = {2^k, k = -10,...,0} 中采样 h，覆盖从精细路径跟踪到大步生成的全部范围。

为了让模型学会大步跳跃，需要一个 teacher 来提供大步间隔上的"正确答案"。然而直接从生成器 u_t 求解大间隔的转移概率是不可解的。由于转移概率满足 Kolmogorov 方程（本质是 ODE），可以用数值方法近似。

本文使用 **RK-4（经典四阶 Runge-Kutta）** 作为 shortcut teacher：
- 在时刻 t、t+h/2、t+h 处分别评估模型得到 logits
- 按 RK-4 公式加权平均：l_bar = 1/6 * (l_1 + 2*l_2 + 2*l_3 + l_4)
- 每步需 4 次模型评估，但提供比 RK-2 显著更好的近似精度

**EMA Teacher 稳定训练**：由于训练非平稳，直接用当前模型做 teacher 会导致 target 随参数更新漂移。因此使用 EMA copy 作为 teacher（beta 接近 1），停止梯度，提供稳定、低方差的目标。

**2. 累积标量（Cumulative Scalar）**

这是 FS-DFM 的核心创新。在 DFM 中，边际速度分解为标量（scale）和方向：

u_t^i(x^i, z) = g(t) * [p_{1|t}(x^i|z) - delta_z(x^i)]

其中 g(t) = kappa'(t) / (1-kappa(t)) 是瞬时速率。问题在于：少步采样时，第一步通常发生在 t 很小处，此时瞬时标量 g(t) 太弱，无法触发有效的 token 跳转，导致采样停滞。

累积标量通过在整个步间隔 [t, t+h] 上积分 g 并归一化来解决这个问题：

g_bar_{t,h} = (1/h) * ln((1-kappa(t)) / (1-kappa(t+h)))

对于线性 scheduler kappa(t)=t，有 g_bar = (1/h) * ln((1-t)/(1-t-h))。

替换后的速度为 u_bar_t^i = g_bar_{t,h} * [p_{1|t} - delta_z]。这样即使 t 很小，只要 h 足够大，累积标量就能提供足够的概率流量驱动 token 跳转。

**3. 预算感知混合损失**

训练时根据步数大小切换损失：

- 小步（h < tau = 2^{-9}）：使用 DFM 路径损失 L_dfm（Bregman 散度），确保局部路径忠实
- 大步（h >= tau）：使用 KL 散度损失 L_dist 对齐 student 和 shortcut teacher 的输出

最终损失对每个样本根据 h 的大小选择性混合：L = (1/B) * sum(m_b * L_dfm^(b) + (1-m_b) * L_dist^(b))

### 损失函数 / 训练策略

采用**预训练后微调**的两阶段策略：

1. **阶段一**：先训练一个标准 DFM 骨干（无步数感知），使用 FineWeb-Edu 数据集
2. **阶段二**：用步数感知目标和累积标量微调 DFM checkpoint

训练细节：
- 数据集：FineWeb-Edu 训练，WikiText-103 评估
- Tokenizer：GPT-2 tokenizer
- 序列长度：1024 tokens（文档拼接填充）
- 模型规模：0.169B、1.3B、1.7B
- 源分布：uniform 和 mask 两种
- Scheduler：线性 kappa(t) = t

## 实验关键数据

### 主实验

**表1：累积标量消融（0.169B 模型）**

| 采样步数 (NFE) | 普通标量 PPL | 累积标量 PPL | 降低 |
|:---:|:---:|:---:|:---:|
| 1 | 1312.65 | 514.40 | -60.8% |
| 2 | 462.31 | 333.07 | -28.0% |
| 4 | 194.29 | 176.19 | -9.3% |
| 8 | 97.51 | 90.49 | -7.2% |
| 1024 | 85.61 | 87.36 | ~持平 |

累积标量在少步时带来巨大提升（1步降低60.8%），多步时收敛到相同水平。

**表2：FS-DFM vs 扩散语言模型（512 token 续写）**

| 方法 | 规模 | 1步 PPL | 2步 PPL | 4步 PPL | 8步 PPL | 16步 PPL |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Dream | 7B | 1163.08 | 785.87 | 752.11 | 739.40 | 630.30 |
| LLaDA | 8B | 256.07 | 290.35 | 495.17 | 441.26 | 432.65 |
| **FS-DFM** | **0.17B** | **173.39** | **143.77** | **97.07** | **75.78** | **67.42** |
| **FS-DFM** | **1.3B** | **231.89** | **169.99** | **99.79** | **70.97** | **59.84** |
| **FS-DFM** | **1.7B** | **191.20** | **155.01** | **101.20** | **72.84** | **61.67** |

FS-DFM 0.17B 在 8 步时的 PPL（75.78）已远优于 LLaDA 8B 在 16 步的结果（432.65），尽管参数量小 40 倍。Dream 和 LLaDA 在少步时生成内容多为重复 token。

### 消融实验

- **RK-4 vs RK-2**：RK-4 在各 NFE 下 PPL 比 RK-2 低约 12%（中位数比值 0.88x），代价是训练时额外的模型评估
- **EMA Teacher**：使用 EMA copy 作为 teacher 比直接用当前模型显著提升训练稳定性
- **源分布**：uniform 和 mask 两种源分布下 FS-DFM 都有效
- **步数采样策略**：对数网格覆盖全范围优于集中在某个区间

### 关键发现

1. **128 倍加速**：FS-DFM 用 8 步匹配 DFM 的 1024 步质量（1024 token 生成）
2. **小模型胜大模型**：0.17B FS-DFM 在少步生成上远优于 7-8B 的 Dream/LLaDA
3. **大步收敛性**：随着 NFE 增大（h 趋近 0），FS-DFM 平滑收敛到标准 DFM，因为累积标量 g_bar 此时趋近于瞬时 g(t)
4. **累积标量是关键**：在大步时（1-2 NFE），累积标量带来最大提升
5. **MAUVE 对比**：FS-DFM 在 16 步时 MAUVE 达 0.39-0.58，而 LLaDA/Dream 16 步仅约 0.005

## 亮点与洞察

- 问题切入精准：识别出 DFM 少步采样的核心障碍是瞬时标量在小 t 时过弱，提出累积标量是优雅的数学解决方案
- Shortcut teacher 的 RK-4 实现将 ODE 求解器与离散扩散训练无缝结合
- 预训练+微调策略避免了从零开始的昂贵步数感知训练
- 首个开源的 DFM + FS-DFM 模型和代码（Apple 出品）
- 少步下 0.17B 小模型碾压 8B 大模型的结果非常 surprising

## 局限与展望

1. 评估主要在 1024 token 长度，更长序列（4K+）的扩展性未验证
2. 仅用 GPT-2 tokenizer 和中小规模模型，与现代 LLM（7B+ ARM）的直接质量对比缺失
3. RK-4 teacher 训练时需要 4 次前向传播/步，训练成本较高
4. 未探索 FS-DFM 在条件生成（摘要、翻译等）上的表现
5. 生成的文本多样性（MAUVE）虽优于 LLaDA/Dream 但绝对值仍较低

## 相关工作与启发

FS-DFM 是连续空间 flow-matching 中少步/单步蒸馏思想（如 consistency models、shortcut models）在离散文本空间的首次全面实践。累积标量的思路对其他需要离散大步更新的 CTMC 应用也有参考价值。该工作也表明离散扩散在语言建模中仍有巨大潜力，关键在于解决步数效率问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（累积标量 + RK-4 shortcut teacher 在离散文本 FM 中的首次应用）
- 技术深度: ⭐⭐⭐⭐⭐（CTMC 理论 + Kolmogorov ODE + Bregman divergence，数学严谨）
- 实验充分度: ⭐⭐⭐⭐（多模型规模 + 多基线 + 消融充分，但缺少条件生成评估）
- 实用性: ⭐⭐⭐⭐（128x 加速有实际意义，但 DLM 整体生态仍不成熟）
- 写作质量: ⭐⭐⭐⭐（数学表述清晰，但符号较多需要反复查阅）

<!-- RELATED:START -->

## 相关论文

- [Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/nlp_generation/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes](../../ACL2025/nlp_generation/tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)
- [Context-Aware Hierarchical Merging for Long Document Summarization](../../ACL2025/nlp_generation/context-aware_hierarchical_merging_for_long_document_summarization.md)
- [LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](../../CVPR2025/nlp_generation/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)
- [C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)

<!-- RELATED:END -->
