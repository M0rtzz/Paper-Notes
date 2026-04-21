---
title: >-
  [论文解读] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference
description: >-
  [AAAI 2026][AI安全][分割推理] 提出 InfoDecom，通过两级信息分解（频域视觉信息去除 + 互信息抑制）减少 smashed data 中的冗余信息，再添加闭式计算的高斯噪声提供理论隐私保证，在浅层客户端模型下实现远优于现有方法的 utility-privacy trade-off。
tags:
  - AAAI 2026
  - AI安全
  - 分割推理
  - 数据重构攻击
  - 隐私保护
  - 信息分解
  - 频域变换
---

# InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference

**会议**: AAAI 2026  
**arXiv**: [2511.13365](https://arxiv.org/abs/2511.13365)  
**代码**: [github.com/SASA-cloud/InfoDecom](https://github.com/SASA-cloud/InfoDecom)  
**领域**: AI安全  
**关键词**: 分割推理, 数据重构攻击, 隐私保护, 信息分解, 频域变换

## 一句话总结

提出 InfoDecom，通过两级信息分解（频域视觉信息去除 + 互信息抑制）减少 smashed data 中的冗余信息，再添加闭式计算的高斯噪声提供理论隐私保证，在浅层客户端模型下实现远优于现有方法的 utility-privacy trade-off。

## 研究背景与动机

**分割推理（Split Inference, SI）** 将 DNN 分为浅层的客户端模型（bottom model）和服务器端模型（top model），客户端仅发送中间表示（smashed data）给服务器。然而数据重构攻击（DRA）能从 smashed data 恢复原始输入，造成严重隐私泄露。

**现有防御的两条路线及其局限**：

**正则化方法**（Shredder、Nopeek、InfoScissors 等）：通过启发式优化目标（如互信息上界、距离相关性）指导 smashed data 产生扰动
   - 缺点：无严格的可证明隐私保证

**闭式噪声计算**（dFIL、FSInfo 等）：基于 Fisher Information 或条件熵计算满足特定隐私预算的噪声尺度
   - 缺点：当 bottom model 浅层时（资源受限设备的常见场景），smashed data 保留大量输入信息 → 需要大量噪声才能满足隐私要求 → 严重损害任务性能

**核心洞察**：现有防御之所以 UPT（utility-privacy trade-off）差，根源在于它们将扰动浪费在 smashed data 中大量的**任务无关冗余信息**上。

**核心思路**：先分解并去除冗余信息，减少需要保护的敏感内容量 → 同等隐私保证下所需噪声更少 → 性能退化更小 → 更好的 UPT。

## 方法详解

### 整体框架

InfoDecom 包含三个阶段：

1. **视觉信息去除（Visual Information Removal）**：频域变换 → 丢弃人眼感知必要的低频分量
2. **互信息抑制（Mutual Information Suppression）**：基于 IB 原理正则化 bottom model → 保留任务相关信息、抑制任务无关信息
3. **噪声扰动（Noise Perturbation）**：闭式计算 FSInfo 引导的高斯噪声 → 理论隐私保证

### 关键设计

1. **视觉信息去除（频域分解）**

   **通信三元组的启發**：语法通信（传输所有比特）→ 语义通信（传输含义）→ 语用通信（传输任务贡献）。直接输入原始图像是语法通信，包含大量冗余。

   **操作流程**：
   - RGB → YUV 颜色空间
   - 每个分量分成 8×8 块
   - Forward DCT（离散余弦变换）→ 64 个频率系数
   - 删除振幅最高的 K 个 DCT 系数（低频分量 $X_l$）
   - 仅保留高频系数 $X_h$ 给 DNN

   **设计动机**：JPEG 压缩理论告诉我们低频分量对人眼视觉感知至关重要（包含主要视觉信息），而 DuetFace 的实验表明高频分量仍包含足够的语义信息供 DNN 完成分类。因此丢弃低频分量 = 隐藏大部分人类可感知的隐私信息，同时保留 DNN 需要的语义信息。

2. **互信息抑制（基于 IB 原理）**

   虽然去除了部分视觉信息，剩余的高频分量 $X_h$ 可能仍包含可被 DRA 利用的隐私敏感信息。

   **优化目标**：$\min_Z \lambda I(X_h; Z) - I(Y; Z)$

   **(a) 最小化 $I(X_h; Z)$** — 聚类损失：

   借鉴 CLUB（MI 上界）的思想，设计聚类损失使不同输入的 smashed data 相互纠缠，降低可区分性：
   $$\mathcal{L}_{cl} = \frac{1}{N} \sum_{i=1}^{N} \|z_i - z_j\|_2^2$$
   其中 $j$ 从 $\{1, ..., N\}$ 均匀采样。

   **设计动机**：推不同 smashed data 彼此靠近 → 条件分布 $p(Z|X)$ 变得更模糊 → 攻击者更难从 smashed data 反推原始输入。

   **(b) 最小化 $-I(Y; Z)$** — 交叉熵损失：

   用 Barber-Agakov 下界替代，最终简化为标准交叉熵：
   $$\mathcal{L}_{ce} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_i^{(k)} \log(f_{\theta_2}(z_i))^{(k)}$$

3. **闭式噪声扰动（理论隐私保证）**

   采用 FSInfo 隐私度量计算高斯噪声尺度：
   $$\tilde{Z} = Z + \delta, \quad \delta \sim \mathcal{N}\left(0, \frac{\det(J^T J)^{\frac{1}{2d}}}{e^{FSInfo}(2\pi e)^{\frac{1}{2}}}\right)$$

   其中 $J$ 是 $Z$ 关于原始输入 $X$ 的 Jacobian。FSInfo 值越低（如 -1）意味着更少的隐私泄露。

   **关键创新**：由于前两级已经去除了冗余信息，smashed data 中需要保护的内容量大幅减少 → 达到相同 FSInfo 级别所需的噪声尺度也更小 → 性能退化更少。

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$$

- Top model 由 $\mathcal{L}_{ce}$ 优化
- Bottom model 由 $\lambda \mathcal{L}_{cl} + \mathcal{L}_{ce}$ 优化
- 推理时：高频输入 → 更新后的 bottom model → 正则化的 smashed data → 添加噪声 → 发送给服务器

**超参数**：
- Adam optimizer, lr = 3e-4, weight decay = 0.01
- 全局训练 150 epochs, batch size = 128
- 默认 $|X_h| = 54$（保留 54/64 个频率系数）, $\lambda = 10$, FSInfo = -1
- CIFAR-10：2×RTX 4090；CelebA：4×A100

## 实验关键数据

### 主实验：Utility-Privacy Trade-off 对比

在 CIFAR-10 和 CelebA 上，使用 ResNet-18，split point 在 C64 层（浅层模型）：

| 方法 | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|------|:---:|:---:|:---:|:---:|
| Raw（无防御） | 高 | 低（隐私泄露） | 高 | 低 |
| Nopeek | 中 | 中 | 中 | 中 |
| Shredder | 中 | 中 | 中 | 中 |
| inv_dFIL_def | 低 | 高 | 中 | 中 |
| FSInfoGuard | 中 | 中 | 中 | 中 |
| **InfoDecom** | **0.7329** | **0.0843** | **0.9693** | **0.1942** |

InfoDecom 在 utility-privacy 平面上实现最佳 trade-off（曲线位于其他方法的右上方）。

### 消融实验

| 配置 | CIFAR-10 Acc. | CIFAR-10 MSE | 说明 |
|------|:---:|:---:|------|
| **InfoDecom (完整)** | **0.7329** | **0.0843** | 默认设置 |
| w/o 视觉信息去除 | 0.6273 | 0.0849 | Acc 下降因需更多噪声满足 FSInfo=-1 |
| w/o $\mathcal{L}_{cl}$ | 0.7453 | 0.0835 | Acc 略升但 MSE 降（防御变弱） |
| w/o FSInfo 噪声 | 0.7274 | 0.0826 | 失去理论隐私保证 |

### 信息控制器参数影响

**保留系数数量 $|X_h|$**（FSInfo=-1, λ=10）：

| $|X_h|$ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---------|:---:|:---:|:---:|:---:|
| 54 | 0.7329 | 0.0843 | 0.9693 | 0.1984 |
| 41 | 0.6905 | 0.1497 | 0.8036 | 0.3273 |
| 32 | 0.3645 | 0.2337 | 0.6135 | 1.1024 |
| 18 | 0.1004 | 0.2492 | 0.6135 | 1.1022 |

**权重因子 λ**（$|X_h|$=54, FSInfo=-1）：

| λ | CIFAR-10 Acc. | CIFAR-10 MSE | CelebA Acc. | CelebA MSE |
|---|:---:|:---:|:---:|:---:|
| 1 | 0.7570 | 0.0822 | 0.9515 | 0.1925 |
| 10 | 0.7329 | 0.0843 | 0.9693 | 0.1942 |
| 20 | 0.7250 | 0.0854 | 0.8997 | 0.1950 |

### 关键发现

1. **信息分解是 UPT 改善的关键**：相同隐私级别下，InfoDecom 的 Acc 远高于直接加噪声的方法
2. **视觉信息去除不可或缺**：去掉后 Acc 从 0.7329 降到 0.6273（因需更大噪声来满足 FSInfo=-1）
3. **互信息抑制的微妙作用**：去掉后 Acc 略升但隐私防御变弱 → 正则化确实压缩了有用信息以外的内容
4. **CelebA 上效果更佳**：二分类任务（吸引力分类）中 InfoDecom 达到 96.93% Acc 同时 MSE=0.1942
5. **计算开销可接受**：InfoDecom 推理每个样本 6.64ms（vs 基本前向传播 0.26ms），与其他 Jacobian 方法相当

## 亮点与洞察

1. **"分解再加噪"范‎式的创新**：不直接加噪声保护所有信息，而是先去除冗余再保护精华 → 概念简单但极其有效
2. **频域处理的巧妙应用**：借鉴 JPEG 压缩理论，丢弃低频视觉信息 → DNN 能用高频信息完成任务但人眼无法识别
3. **三级控制器设计**：$|X_h|$（视觉冗余度）、λ（语义冗余度）、FSInfo（隐私保证级别）提供灵活的 trade-off 调节
4. **理论保证 + 实用性兼具**：FSInfo 提供可证明的隐私下界，而两级信息分解确保噪声不会过大

## 局限与展望

- 目前仅适用于视觉任务（视觉信息去除依赖频域变换）→ 作者提到 MIS 和 NP 组件是模态无关的
- 计算开销相比非 Jacobian 方法偏高（6.64ms vs <1ms）→ 更高效的 Jacobian 近似策略值得探索
- 仅在两个数据集（CIFAR-10、CelebA）上验证，未涵盖高分辨率或更复杂的视觉任务
- split point 固定在第一层（最浅）→ 不同 split point 的效果未充分探讨
- DRA 攻击只用了 invNet 一种 → 对更高级的攻击方法（如 GAN-based DRA）的鲁棒性待验证

## 相关工作与启发

- **与正则化方法的区别**：Shredder/Nopeek/InfoScissors 等仅靠优化目标，无理论保证；InfoDecom 先降低需保护的信息量再加有保证的噪声
- **与闭式噪声方法的区别**：dFIL/FSInfoGuard 直接计算噪声尺度，但在浅层模型下噪声过大；InfoDecom 通过分解冗余信息降低了所需噪声尺度
- **信息瓶颈（IB）原理的实用化**：将 IB 的理论框架转化为实际可训练的聚类损失和交叉熵损失
- **对隐私保护 ML 的启示**："减少需要保护的信息量"比"加更多噪声"更有效

## 评分

- 新颖性: ⭐⭐⭐⭐（"分解再加噪"思路直观有效，频域+IB+闭式噪声的组合新颖）
- 实验充分度: ⭐⭐⭐（两个数据集 + 参数敏感性分析完整，但场景偏少）
- 写作质量: ⭐⭐⭐⭐⭐（动机清晰、公式推导规范、三级分解逻辑性强）
- 价值: ⭐⭐⭐⭐（解决了浅层客户端模型下隐私-效用 trade-off 的核心痛点）

<!-- RELATED:START -->

## 相关论文

- [Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](../../ICML2025/ai_safety/privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)
- [Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)
- [An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlear.md)
- [HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)

<!-- RELATED:END -->
