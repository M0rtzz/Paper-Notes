---
title: >-
  [论文解读] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators
description: >-
  [ECCV 2024][图像恢复][Transformer] 发现 Diffusion Transformer 中 query-key 交互存在显著冗余（尤其在去噪早期），提出 Attention Mediator 机制将注意力复杂度降至线性，并设计逐步动态调整策略，在 SiT-XL/2 上实现 SOTA FID 2.01，同时减少计算量。
tags:
  - ECCV 2024
  - 图像恢复
  - Transformer
  - 注意力中介者
  - 动态网络
  - 线性注意力
  - 去噪冗余
---

# Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators

**会议**: ECCV 2024  
**arXiv**: [2408.05710](https://arxiv.org/abs/2408.05710)  
**代码**: 有 (https://github.com/LeapLabTHU/Attention-Mediators)  
**领域**: 图像生成 / 扩散模型加速  
**关键词**: 扩散Transformer, 注意力中介者, 动态网络, 线性注意力, 去噪冗余

## 一句话总结

发现 Diffusion Transformer 中 query-key 交互存在显著冗余（尤其在去噪早期），提出 Attention Mediator 机制将注意力复杂度降至线性，并设计逐步动态调整策略，在 SiT-XL/2 上实现 SOTA FID 2.01，同时减少计算量。

## 研究背景与动机

### Diffusion Transformer 的效率困境

Diffusion Transformer（DiT）因其简洁性、有效性和可扩展性，正在取代 U-Net 成为扩散模型的主流骨架，驱动了 Stable Diffusion V3、Pixart-α/Σ/δ、华为 DiT、Sora 等应用。然而，DiT 的广泛批评在于**全局注意力机制的高计算消耗**——self-attention 的 $O(N^2C)$ 复杂度成为推理瓶颈，严重阻碍了高分辨率图像和长视频的实际部署。

虽然视觉识别领域已有多种注意力加速方法（窗口注意力、线性注意力等），但**扩散生成领域的注意力效率优化几乎是空白**。

### 关键观察：去噪过程中的注意力冗余

本文通过定量分析发现了 DiT 中的两个关键现象：

**观察1：大量 query-key 冗余普遍存在**。在所有 self-attention 层中，不同 query 对 key 的注意力分布高度相似。例如 DiT-S/2 的第10层，在初始几步中所有 query 的内部距离几乎为零——它们完全同质化。

**观察2：冗余随去噪进程递减**。在去噪初期（纯噪声阶段），注意力冗余最为严重；随着去噪推进，query 变得越来越多样化。这意味着早期的**全量一对一注意力交互是不必要的**。

### 量化冗余的度量方法

本文使用 Jensen-Shannon Divergence (JSD) 设计冗余度量。将注意力矩阵 $\mathbf{A}^{(m)}$ 的每一行视为一个概率分布（query 对所有 key 的权重分布），计算第 $l$ 层的冗余分数：

$$S_l = \frac{2}{MN(N-1)} \sum_{m=1}^{M} \sum_{i_1=1}^{N-1} \sum_{i_2=i_1+1}^{N} \mathcal{D}_{\text{JS}}(\mathbf{A}_{i_1}^{(m)}, \mathbf{A}_{i_2}^{(m)})$$

$S_l$ 低 → 注意力分布高度相似 → 冗余严重。实验在 DiT-S/2 和 SiT-S/2 上测量了所有层和所有时间步的 $S_l$，验证了上述两个观察。

## 方法详解

### 整体框架

在标准 self-attention 层中引入一组额外的**中介者 tokens（Attention Mediators）**，数量远少于原始 tokens（如 <10%），分别与 query 和 key 交互。同时，根据去噪时间步的冗余程度动态调整中介者数量——早期少、后期多。

### 关键设计

#### 1. **Attention Mediators 机制**

**功能**：用一组少量中介者 tokens $\mathbf{t}^{(m)} \in \mathbb{R}^{n \times d}$（$n \ll N$）压缩 query-key 之间的冗余交互。

**核心思路**：将标准注意力的一步 Q-K-V 交互拆分为两步：

**Step 1**：中介者聚合 key 信息（ $n \times N$ 交互）：
$$\mathbf{v}_{\text{med}}^{(m)} = \text{Softmax}\left(\frac{\mathbf{t}^{(m)} \mathbf{k}^{(m)\top}}{\sqrt{d}}\right) \mathbf{v}^{(m)}$$

**Step 2**：query 从中介者提取信息（$N \times n$ 交互）：
$$\mathbf{h}^{(m)} = \text{Softmax}\left(\frac{\mathbf{q}^{(m)} \mathbf{t}^{(m)\top}}{\sqrt{d}}\right) \mathbf{v}_{\text{med}}^{(m)}$$

中介者 tokens 的生成方式：对 query tokens 进行自适应池化——先 reshape 为 latent 图像形状 $\mathbb{R}^{H \times W \times d}$，在空间维度池化到 $\mathbb{R}^{h \times w \times d}$，再 flatten 得到 $n = h \times w$ 个中介者。

**设计动机**：(1) 中介者作为信息"瓶颈"，压缩了冗余的一对一 Q-K 交互；(2) 由于 Q 和 K 被中介者解耦，可以交换计算顺序——先计算 $\mathbf{A}_{\text{tk}}^{(m)} \cdot \mathbf{v}^{(m)}$（$n \times N \cdot N \times d$），再与 Q 交互，避免了 $N \times N$ 的矩阵；(3) 补充 DWConv 弥补线性注意力的特征多样性损失。

#### 2. **复杂度分析**

标准 self-attention 的复杂度为 $O(N^2 C)$。中介者注意力的每一步均为 $O(Nnd)$，总复杂度为 $O(nNC)$。由于 $n \ll N$，计算量**从二次降至线性**。

| 操作 | 标准注意力 | 中介者注意力 |
|------|-----------|-------------|
| 复杂度 | $O(N^2C)$ | $O(nNC)$ |
| 256×256图像（$N=256$） | $\propto 65536$ | $\propto 256n$（$n=64$时约1/4） |
| 分辨率增长 | 二次增长 | 线性增长 |

**高分辨率优势**：图像分辨率越高，线性复杂度的优势越突出。

#### 3. **时间步动态中介者调整**

**功能**：根据去噪过程中冗余程度的变化，动态增加中介者 tokens 数量。

**核心思路**：利用相邻去噪步之间的 latent 距离 $\Delta_t = \|x_t - x_{t+1}\|$ 来量化冗余变化程度。当距离降至初始距离的某个阈值以下时，切换到更多中介者：

$$n_t = \begin{cases} n_1, & \Delta_t > \rho_0 \cdot \Delta_0 \quad \text{(早期，高冗余，少中介者)} \\ n_2, & \Delta_t \leq \rho_1 \cdot \Delta_0 \quad \text{(中期)} \\ \vdots \\ n_k, & \Delta_t \leq \rho_{k-1} \cdot \Delta_0 \quad \text{(后期，低冗余，多中介者)} \end{cases}$$

**每个样本独立调度**：阈值切换是样本自适应的，因为不同图像的去噪过程不同，latent 变化速度也不同。

**设计动机**：(1) 早期冗余高，少量中介者即可充分表达——大幅节省计算；(2) 后期细节丰富，需更多中介者保留多样性；(3) L1 距离比 L2 效果更好（消融验证）。

### 损失函数 / 训练策略

- 训练使用 ImageNet-1k，类条件扩散模型
- AdamW 优化器，无 weight decay，学习率 $1 \times 10^{-4}$
- 全局 batch size 256，训练 400K 迭代
- EMA decay 0.9999
- 仅替换前 4 层 self-attention 为中介者注意力（XL 模型）
- 高分辨率（512/1024）通过从 256 模型 finetune 获得

## 实验关键数据

### 主实验：ImageNet 256×256 类条件生成

| 模型 | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|-------|-----|-----------|---------|
| ADM | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| StyleGAN-XL | 2.30 | 4.02 | 265.12 | 0.78 | 0.53 |
| VDM++ | 2.12 | - | 267.7 | - | - |
| DiT-XL (cfg=1.5) | 2.27 | 4.60 | 278.24 | 0.83 | 0.57 |
| SiT-XL (cfg=1.5) | 2.06 | 4.50 | 270.27 | 0.82 | 0.59 |
| **Ours (cfg=1.5)** | **2.01** | **4.49** | **271.04** | **0.82** | **0.60** |

在 SiT-XL/2 基础上，方法取得 **FID 2.01 的 SOTA 结果**，同时减少了计算量。

### 消融实验：静态中介者数量对比（SiT-S/2, 256×256）

| 配置 | FLOPs(G) | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|----------|------|-------|-----|-----------|---------|
| SiT-S/2 baseline | 6.06 | 58.61 | 9.25 | 24.31 | 0.41 | 0.59 |
| + Ours (n=4) | 5.49 (-9.4%) | 57.67 | 10.01 | 26.66 | 0.42 | 0.56 |
| + Ours (n=16) | 5.55 (-8.4%) | 54.55 | 9.28 | 26.55 | 0.43 | 0.59 |
| + Ours (n=64) | 5.78 (-4.6%) | **53.57** | **9.01** | **27.26** | **0.43** | **0.61** |

即使使用最少的 4 个中介者，FID 也优于 baseline；n=64 时 FID 降低 5.04，FLOPs 仍减少 4.6%。

### 消融实验：对比简单 Q-K 维度压缩

| 方法 | FLOPs(G) | FID↓ | Precision↑ | Recall↑ |
|------|----------|------|-----------|---------|
| SiT-S/2 baseline | 6.06 | 58.61 | 0.41 | 0.59 |
| Q-K 维度压缩 r=0.875 | 5.91 | 58.98 (+) | 0.40 | 0.60 |
| Q-K 维度压缩 r=0.750 | 5.76 | 59.18 (+) | 0.39 | 0.59 |
| Q-K 维度压缩 r=0.500 | 5.46 | 60.02 (+) | 0.40 | 0.57 |
| **Ours (n=64)** | **5.78** | **53.57** | **0.43** | **0.61** |

直接降低 Q-K 隐藏维度虽然节省计算，但 FID **持续恶化**；而本文方法在节省计算的同时**显著提升质量**。

### 关键发现

1. **中介者不仅降低计算量，还提升生成质量**：这是因为压缩冗余交互等价于一种隐式正则化，减少了 attention 输出的同质化
2. **高分辨率加速更显著**：SiT-B/2 在 512² 分辨率加速 15.7%，在 1024² 分辨率加速 45.4%，线性复杂度的优势随分辨率增长而放大
3. **动态策略优于静态**：通过时间步自适应调整中介者数量，在相同 FLOPs 预算下始终取得更好的 FID
4. **L1 距离优于 L2**：在 latent 变化量度量中，L1 距离是更好的阈值判据

## 亮点与洞察

1. **从冗余分析到方法设计的self-contained逻辑**：先用JSD量化冗余→发现时间步变化规律→中介者机制解决冗余→动态调整适应变化规律，整条技术路线一气呵成
2. **"质量提升+效率提升"的罕见双赢**：通常加速方法会牺牲质量，但这里压缩冗余反而改善了特征多样性
3. **中介者的语义理解**：中介者tokens不仅是计算优化手段，还可以理解为对 latent 信息的语义压缩——用少量代表性表示引导生成过程
4. **样本自适应的无训练调度**：动态阈值基于 latent 变化量，无需额外训练网络来决定何时切换

## 局限与展望

1. **仅替换部分层**：XL 模型仅替换前4层为中介者注意力，未全面探索最优替换策略
2. **阈值搜索**：动态调整的阈值 $\rho_i$ 通过 grid search 获取，搜索空间有限
3. **仅验证了类条件生成**：未在 text-to-image（如 Stable Diffusion）等更实际的场景中验证
4. **训练成本未大幅降低**：方法主要在推理阶段加速，训练阶段的效率提升有限
5. **未与其他加速方法组合**：如 distillation、step reduction 等方法可能互补

## 相关工作与启发

- **Agent Attention [ECCV 2024]**：在视觉识别任务中也使用额外tokens作为Q-K桥梁，本文将此思路延伸到扩散生成
- **SiT [ICML 2024]**：本文的主要基座模型，引入插值框架从离散到连续时间
- **DiT [ICCV 2023]**：证明了ViT在扩散模型中的可扩展性，本文在其架构上优化注意力效率

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 冗余分析驱动的中介者设计+时间步动态策略，思路清晰独特
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多尺度模型、多分辨率、详细消融、与多种方法对比、可视化验证
- **写作质量**: ⭐⭐⭐⭐ — 从观察到方法的叙事逻辑流畅，复杂度分析清晰
- **价值**: ⭐⭐⭐⭐ — SOTA FID + 计算减少，对DiT推理优化有直接实践价值

<!-- RELATED:START -->

## 相关论文

- [Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)
- [Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](../../ICLR2026/image_restoration/diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)

<!-- RELATED:END -->
