---
title: >-
  [论文解读] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin
description: >-
  [ICML 2026][模型压缩][码本坍塌] 作者诊断出 VQ-VAE 的码本坍塌根源是"码本向量 ℓ2 范数失衡 + 几何聚集"，于是提出 SAMP：Ball-Bounded Norm Regularization 把所有码本向量约束在时变 Euclidean 球内、ArcCosine Additive…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "码本坍塌"
  - "角度 margin"
  - "球面学习"
  - "范数正则"
  - "码本利用率"
---

# ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin

**会议**: ICML 2026  
**arXiv**: [2605.13517](https://arxiv.org/abs/2605.13517)  
**代码**: https://github.com/goals4292/ArcVQ-VAE  
**领域**: VQ-VAE / 图像生成 / 离散表示  
**关键词**: 码本坍塌, 角度 margin, 球面学习, 范数正则, 码本利用率

## 一句话总结
作者诊断出 VQ-VAE 的码本坍塌根源是"码本向量 ℓ2 范数失衡 + 几何聚集"，于是提出 SAMP：Ball-Bounded Norm Regularization 把所有码本向量约束在时变 Euclidean 球内、ArcCosine Additive Margin Loss 借鉴 ArcFace 在球面上推开 latent 向量，从而让码本均匀分散、利用率大幅上升，在 ImageNet 重建和生成 FID 上都击败主流 VQ-VAE 变体。

## 研究背景与动机
**领域现状**：VQ-VAE 把连续 latent 离散化成有限码本，是 autoregressive 图像生成（VQGAN / RQ-VAE）、扩散先验（LDM）以及 multimodal token 化的基础组件。提升 VQ-VAE 的方法很多：SQ-VAE（随机量化）、CVQ-VAE（在线 K-means 把不常用码本拉近 latent）、VQGAN-LC（预训练 encoder 抽码本）、Wasserstein VQ 等。

**现有痛点**：（1）固定大小的码本无法表达数据集的全部丰富性。（2）码本坍塌——只有一小部分码本被频繁使用、其余几乎不动，码本利用率经常低于 50%。（3）现有方法主要从"如何更新/选择码本"这种机制层面修补，没动到根本——码本向量在 latent 空间的**几何不均衡**。

**核心矛盾**：作者通过 Figure 2/3 实证发现：训练初期所有码本初始化在原点附近、被选中的码本沿 encoder 输出方向加速增长 ℓ2 范数，未被选中的留在原点附近。高范数码本同时离 encoder feature 更近，更易被选中，形成正反馈循环——这是个**几何动力学**问题，而非简单的"sample 不到就饿死"问题。

**本文目标**：（1）从根上抑制码本向量的范数失衡；（2）让 latent 向量在 latent 空间均匀分散，使每个 latent 都有机会绑定到不同码本；（3）做到不引入新网络组件、几乎零额外计算成本。

**切入角度**：作者借鉴 face recognition 中 ArcFace 的"角度 margin + 球面学习"思想——如果把所有 latent 和码本都 ℓ2 归一化到单位球面，码本选择就从 Euclidean 最近邻变成最大角余弦匹配，再加 angular margin 推开类间距离，就能强制 latent 均匀分散。但 face recognition 是 supervised classification，VQ-VAE 没有显式 class label——需要把"top-k 最近码本"当作隐式类别。

**核心 idea**：用 Spherical Angular-Margin Prior (SAMP) = Ball-Bounded Norm Regularization (限制码本在时变 Euclidean 球内) + ArcCosine Additive Margin Loss (在球面上加角度 margin 推开 latent)，让码本几何上均匀分散、利用率大幅上升。

## 方法详解
SAMP 由两个互补组件构成。Ball-Bounded Norm Regularization 是个"硬约束"，每个 batch 后强行把超界码本投影回球内；ArcCosine Additive Margin Loss 是个"软约束"，通过 loss 鼓励 latent 角度分散。两者共同把 VQ-VAE 重塑为球面学习问题。

### 整体框架
- **架构**：与 VQGAN 完全相同的 encoder-decoder + codebook，**不引入任何新网络组件**。改动只在两处：(1) 每个 batch 后对码本做一次 norm clipping；(2) 在原 VQ loss 基础上加一项 ArcLoss。
- **训练循环**：标准 VQ-VAE forward → 算 $\mathcal{L}_\text{VQ}$（重建 + commit + codebook loss） → 算 $\mathcal{L}_\text{A}$（ArcLoss with stop-grad on codebook） → 总 loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$ → backprop → batch 后对每个码本向量做 ball projection。
- **推理 / 量化**：encoder 输出和码本都 ℓ2 归一化后做角余弦匹配，等价于在单位球上找最近码本。
- **后续生成**：在 ArcVQ-VAE 训出的 $32^2$ token 上训练 LDM 作为 prior，sampling 250 步。

### 关键设计

1. **Ball-Bounded Norm Regularization（时变球约束）**:

    - 功能：限制每个码本向量的 ℓ2 范数不超过一个随训练步指数增长的上界 $M(t)$，从根上切断"码本范数失衡 → 高范数码本被偏好"的正反馈。
    - 核心思路：初始化所有码本到单位球面 $\mathbf{e}_k^{(0)} \sim \ell_2(\text{Unif}(-1,1)^d)$。每个训练步设上界 $M(t) = \exp(\alpha t)$，$\alpha$ 取很小（如 $10^{-5}$）使 $M$ 在训练早期接近 1。每 batch 结束后对范数超界的码本做投影：$\mathbf{e}_k^{(t)} \leftarrow \frac{\mathbf{e}_k^{(t)}}{\|\mathbf{e}_k^{(t)}\|_2} M(t)$，未超界的保留。整个码本集合落在球 $\mathcal{C}^{(t)} \subset \mathbb{B}_{M(t)}^d$ 内。这把训练分两阶段：早期所有码本几乎都在单位球面附近，公平竞争 latent feature；后期球放大，让码本可以有更丰富的范数表达。
    - 设计动机：作者通过 Figure 2 实证发现传统 VQ-VAE 训练后期，少数高频码本范数远超低频码本——而高范数 + 接近 latent 中心区 = 几乎垄断所有 latent 分配。把"早期严格、晚期放松"的球约束加进来直接打破这种 winner-takes-all 动力学。

2. **ArcCosine Additive Margin Loss（球面角度 margin）**:

    - 功能：把 latent 向量 ℓ2 归一化到单位球面后，用借鉴 ArcFace 的 angular margin 推开它们，让每个 latent 占据不同球面区域、关联到不同码本。
    - 核心思路：先把 encoder 输出 $z_{e,i}(x)$ 和码本 $e_j$ 都 ℓ2 归一化：$\hat{z}_i = z_{e,i}/\|z_{e,i}\|$, $\hat{e}_j = e_j/\|e_j\|$。量化规则等价改为最大角余弦：$k = \arg\max_j \hat{z}_i^\top \hat{e}_j$。然后对每对 $(\hat{z}_i, \hat{e}_j)$ 计算角度 $\theta_{i,j} = \arccos(\hat{z}_i^\top \hat{e}_j)$。ArcLoss 形如 ArcFace 的 additive margin softmax：$\mathcal{L}_\text{A} = -\frac{1}{K}\sum_j \log \frac{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)}}{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)} + \sum_{i \notin \mathcal{N}_j^{(k)}} e^{s\cos\theta_{i,j}}}$，其中 $\mathcal{N}_j^{(k)}$ 是离码本 $e_j$ 最近的 top-k 个 latent token 集合（作者取 k=3、$s=10$、$m=0.1$）。loss 把"latent 应该 angularly align 到它最近的码本"（正对）与"远离其他码本"（负对）显式拉开，margin $m$ 强制更紧的对齐。**关键 trick**：对码本应用 stop-gradient $\text{sg}(\hat{e}_j)$，让 ArcLoss 只回传到 encoder 不直接改码本——否则码本会被推向当前 batch 的 latent 分布而失去全局可分性。
    - 设计动机：传统 VQ loss 只让 encoder feature "靠近最近码本"，但不约束 latent 之间是否分散；多个 latent 可能 collapse 到同一区域、共享一个码本。ArcLoss 把"latent 分散"显式写进 loss，且通过球面归一化让"近 vs 远"变成纯角度问题，避开了 Euclidean 空间中"近距离码本互相抢"的几何病。

3. **Decay-Weighted 联合损失 + Stop-Gradient on Codebook**:

    - 功能：在训练早期强约束角度结构、后期让模型专注重建精度，并通过 stop-grad 隔离 ArcLoss 对码本的直接干扰。
    - 核心思路：总 loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$ 其中 $\gamma(t) = \gamma_0 \exp(-\lambda t)$。$\gamma_0=1.0$，$\lambda$ 在 MNIST/CIFAR 取 $5\times 10^{-4}$、ImageNet 取 $10^{-4}$。早期 ArcLoss 权重大、强制 latent 球面分散；后期权重衰减、让 VQ loss 主导以保重建 fidelity。Stop-gradient 让 ArcLoss 只优化 encoder（推 latent 散开），码本由标准 VQ loss 间接更新——encoder 散开后，standard codebook update 自然把码本拉到更分散的 latent 上。
    - 设计动机：直接让 ArcLoss 推码本会导致 batch-driven 局部坍塌——码本被推向当前 batch 的 latent 而失去全局分散性。把"latent 散开"和"码本跟随"职责分开，二者通过标准 VQ commit loss 间接耦合，训练稳定且不互相干扰。

### 损失函数 / 训练策略
$\mathcal{L}_\text{VQ}$ 是标准 VQ loss：reconstruction + codebook + commit（系数 $\beta$）。$\mathcal{L}_\text{A}$ 是 ArcLoss（$s=10$, $m=0.1$, top-k=3）。$\gamma(t)$ 指数衰减。其余超参（learning rate / discriminator weight 等）与 VQGAN 默认一致，方便公平对比。

## 实验关键数据

### 主实验
ImageNet-1K 重建（$256\times 256$，downsample $16\times$ 或 $8\times$）：

| Method | S | K | 码本利用率 | rFID ↓ |
|--------|---|---|-----------|--------|
| VQGAN | $16^2$ | 1024 | 44% | 7.94 |
| VQGAN-FC | $16^2$ | 16384 | 11.2% | 4.29 |
| VQGAN-EMA | $16^2$ | 16384 | 83.2% | 3.41 |
| ViT-VQGAN | $32^2$ | – | – | 较低 |
| **ArcVQ-VAE** | – | – | **接近 100%** | **更低** |

MNIST / CIFAR10 重建对比：

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | rFID ↓ |
|---------|--------|--------|--------|---------|--------|
| MNIST | VQ-VAE | 26.48 | 0.9777 | 0.0282 | 3.43 |
| MNIST | CVQ-VAE | 27.87 | 0.9833 | 0.0222 | 1.80 |
| MNIST | **ArcVQ-VAE** | **28.01** | **0.9840** | **0.0217** | **1.68** |
| CIFAR10 | VQ-VAE | 23.32 | 0.8595 | 0.2504 | 39.67 |
| CIFAR10 | CVQ-VAE | 24.72 | 0.8978 | 0.1883 | 24.73 |
| CIFAR10 | **ArcVQ-VAE** | **24.78** | **0.8989** | **0.1857** | 26.91 |

### 消融实验

| 配置 | 关键观察 |
|------|----------|
| Full SAMP | 码本利用率高、rFID 最低 |
| 仅 Ball-Bounded Norm（无 ArcLoss） | 范数均衡但 latent 仍可能聚集 |
| 仅 ArcLoss（无 Norm Reg） | 高 utilization 码本仍可能 norm 爆炸 |
| 去掉 stop-gradient on codebook | 码本被 batch 拉走、丧失全局分散 |
| 改变 m (0 / 0.1 / 0.3) | $m=0.1$ 最优；过大伤重建 |
| 改变 top-k (1/3/5) | k=3 最优；k=1 过严、k=5 过松 |
| 改变 $\alpha$（球膨胀速率） | $\alpha$ 太大导致早期约束失效、太小后期表达受限 |

### 关键发现
- 码本利用率从 VQGAN 的 44% 提升到接近 100%（Figure 1 t-SNE 可视化中几乎所有码本都是绿色 active），这是几何重新设计带来的根本改善。
- 量化 latent map 的 PCA 可视化（Figure 5）显示 ArcVQ-VAE 激活强度更高、轮廓更清晰，说明码本不仅"用起来了"，还编码了更精细的空间结构。
- Ball-Bounded 和 ArcLoss 互补不可分：单独用任一都不能完全消除坍塌；两者结合才同时解决"范数失衡"和"几何聚集"两个病根。
- Stop-gradient 是稳定 ArcLoss 的必要 trick：直接让 ArcLoss 改码本会导致 batch-driven 局部坍塌——这个发现对其它"在 VQ 上加 metric learning loss"的工作有警示意义。

## 亮点与洞察
- **"码本坍塌是几何问题"的诊断**：作者通过 Figure 2/3 实证地把坍塌归因于"码本范数失衡 + 空间聚集"的动力学循环，而不是简单的"sample 概率低"。这种从几何视角重新诊断老问题非常 refreshing，且直接导出方法设计。
- **把 ArcFace 跨界搬到 VQ-VAE**：face recognition 的角度 margin 思想在监督分类领域非常成熟，但 VQ-VAE 没有 class label——作者用"top-k 最近码本"当作隐式类别，让 angular margin 在无监督场景也能用。这种跨域迁移很巧妙。
- **接近零额外成本**：不引入新网络层、不增 forward FLOPs，只是 batch 后一次 norm clip + 一项 loss term。所以可以 plug 到任何现有 VQ-VAE / VQGAN 上立刻用。

## 局限与展望
- 球约束的 $\alpha$ 和 ArcLoss 的 $m, s, k$ 都是手工 sweep 出来的，不同数据集/分辨率可能需要重调；自适应 schedule 是开放方向。
- 主要在 ImageNet 重建 + LDM 生成上验证，对 video / 3D / multimodal token 化的迁移效果未测。
- top-k 选 latent 集合 $\mathcal{N}_j^{(k)}$ 会引入一个额外的 $O(K \cdot Bhw)$ 排序成本，大码本（>16k）时可能不可忽略。
- 没有跟最新的 FSQ（Finite Scalar Quantization）/ LFQ（Lookup-Free Quantization）做对比——那两种方法直接跳过码本学习，挑战 SAMP 的核心 motivation。

## 相关工作与启发
- **vs CVQ-VAE (Zheng & Vedaldi 2023)**：CVQ 用在线 K-means 把不常用码本拉近 latent；本文从几何空间正则化角度入手，互补但本文 rFID 更低。
- **vs SQ-VAE (Takida et al. 2022)**：SQ 改后验为随机分布；本文走球面 + margin 路线，机制完全不同。
- **vs VQGAN-LC (Zhu et al. 2024)**：那个用预训练 encoder scale up 码本；本文不依赖外部模型，纯几何约束。
- **vs ArcFace (Deng et al. 2019)**：本文是首次把 angular margin 用到无监督 VQ 场景，扩展了 ArcFace 的应用范围。
- **vs FSQ / LFQ**：那两种方法绕过码本学习直接用 scalar quantization；本文坚持学码本但解决了它的固有问题——两种思路对应不同的工程权衡。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 ArcFace 引入 VQ-VAE 是聪明的跨域迁移，球约束 + stop-grad 组合也有原创性
- 实验充分度: ⭐⭐⭐⭐ 多数据集 + 多基线 + 消融完整；缺与 FSQ/LFQ 的直接对比
- 写作质量: ⭐⭐⭐⭐ 诊断部分（Fig 2/3）非常 illuminating，推导清晰；ArcLoss 公式排版略密
- 价值: ⭐⭐⭐⭐ 几乎零额外成本可以 plug 到任何 VQGAN，工程落地价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] Embedding Compression via Spherical Coordinates](../../ICLR2026/model_compression/embedding_compression_via_spherical_coordinates.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](../../ICCV2025/model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
