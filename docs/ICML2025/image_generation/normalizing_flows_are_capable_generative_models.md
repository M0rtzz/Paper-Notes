---
title: >-
  [论文解读] Normalizing Flows are Capable Generative Models
description: >-
  [ICML2025][图像生成][Normalizing Flows] 提出 TarFlow（Transformer AutoRegressive Flow），用堆叠因果 ViT 实现分块自回归 Normalizing Flow，首次在 ImageNet 64×64 上突破 3 BPD…
tags:
  - "ICML2025"
  - "图像生成"
  - "Normalizing Flows"
  - "Transformer"
  - "自回归流"
  - "密度估计"
---

# Normalizing Flows are Capable Generative Models

**会议**: ICML2025  
**arXiv**: [2412.06329](https://arxiv.org/abs/2412.06329)  
**代码**: [apple/ml-tarflow](https://github.com/apple/ml-tarflow)  
**领域**: 图像生成  
**关键词**: Normalizing Flows, Transformer, 自回归流, 图像生成, 密度估计  
**作者**: Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu 等 (Apple)

## 一句话总结

提出 TarFlow（Transformer AutoRegressive Flow），用堆叠因果 ViT 实现分块自回归 Normalizing Flow，首次在 ImageNet 64×64 上突破 3 BPD，并通过高斯噪声增强、score-based 去噪和 guidance 三项技术使 NF 模型的生成质量首次媲美扩散模型。

## 研究背景与动机

Normalizing Flows（NF）是一类基于精确似然的生成模型，通过可逆变换将数据分布映射到简单先验（如高斯分布），具有精确似然计算、确定性目标函数和高效双向计算的优势。然而近年来 NF 在实际应用中的影响力远不如 Diffusion Models 和 LLM，SOTA 也长期停滞不前。

核心问题是：**NF 是否在建模范式上存在根本性局限？还是我们尚未找到合适的架构和训练方式来释放其潜力？**

作者认为是后者。过去 NF 的瓶颈在于：

**架构设计过于复杂受限**：Glow、RealNVP 等使用精心设计的耦合层，难以自由扩展模型容量

**训练不稳定**：连续 NF（如 FFJORD）存在数值不稳定问题

**生成质量差**：传统均匀噪声反量化不足以支撑高质量采样

## 方法详解

### 2.1 Normalizing Flow 基础

NF 通过变量替换公式建模数据密度：

$$p_{\text{model}}(x) = p_0(f(x)) \left|\det\left(\frac{df(x)}{dx}\right)\right|$$

其中 $f: \mathbb{R}^D \to \mathbb{R}^D$ 是可逆变换，$p_0$ 为标准高斯先验。MLE 训练目标为：

$$\min_f \; 0.5\|f(x)\|_2^2 - \log\left|\det\left(\frac{df(x)}{dx}\right)\right|$$

第一项驱动模型将数据映射到小范数的隐变量，第二项防止模型坍缩（collapse）。

### 2.2 分块自回归流（Block Autoregressive Flows）

TarFlow 是 MAF（Masked Autoregressive Flow）的分块推广。输入表示为序列 $x \in \mathbb{R}^{N \times D}$，流变换堆叠 $T$ 层，每层包含：

1. **序列置换** $\pi^t$：交替反转序列方向（奇偶层方向相反）
2. **仿射变换**：
$$z_i^{t+1} = (\tilde{z}_i^t - \mu_i^t(\tilde{z}_{<i}^t)) \odot \exp(-\alpha_i^t(\tilde{z}_{<i}^t)), \quad i > 0$$

其中 $\mu^t, \alpha^t$ 是因果函数（第 $i$ 个位置的输出只依赖前 $i-1$ 个位置）。当 $D=1$ 时退化为标准 MAF。

**Jacobian 行列式**的对数可高效计算：
$$\log|\det(df^t/dz^t)| = -\sum_{i=1}^{N-1}\sum_{j=0}^{D-1} \alpha_i^t(\tilde{z}_{<i}^t)_j$$

最终训练损失简洁地表示为：
$$\min_f \; 0.5\|z^T\|_2^2 + \sum_{t=0}^{T-1}\sum_{i=1}^{N-1}\sum_{j=0}^{D-1} \alpha_i^t(\tilde{z}_{<i}^t)_j$$

### 2.3 Transformer 自回归流架构

核心创新是用 **因果 Vision Transformer（causal ViT）** 替换 MAF 中简单的 masked MLP。对于 $C \times H \times W$ 的图像，先切分为 patch 序列（$N = HW/S^2$，$D = CS^2$），然后用标准 causal attention 实现每层自回归变换。

关键优势：
- **简洁模块化**：每个 flow block 内部就是标准 Transformer，深度和宽度与输入维度完全解耦
- **训练稳定**：双重残差连接（Transformer 内部 + 隐变量 $z_i^t$ 之间），训练难度等同标准 Transformer
- **高可扩展性**：可自由增加 block 数 $T$ 和每 block 层数 $K$

### 2.4 高斯噪声增强训练

传统做法是加小量均匀噪声做反量化，作者发现这**远不够**。关键发现：

- 最优高斯噪声 $\sigma \approx 0.05$（像素值在 $[-1,1]$），而传统均匀噪声标准差仅 0.002
- 噪声增强的本质：丰富逆模型 $f^{-1}$ 的训练分布支撑，避免采样时的 OOD 问题
- 高斯噪声（vs 均匀噪声）将训练分布支撑扩展到整个环境空间

### 2.5 Score-Based 去噪

噪声增强训练后直接采样会产生带噪样本。利用 Tweedie 公式进行无需额外训练的去噪：

$$\hat{x} = y + \sigma^2 \nabla_y \log p_{\text{model}}(y)$$

其中 $y = f^{-1}(z)$ 是带噪样本。去噪仅需 TarFlow 模型自身计算 score，无需额外模块。

### 2.6 Guidance

**条件 guidance**：与 CFG 完全一致，训练时以 0.1 概率随机 drop 类别标签：
$$\tilde{\mu}_i^t = (1+w)\mu_i^t(\cdot; c) - w \cdot \mu_i^t(\cdot; \varnothing)$$

**无条件 guidance**（本文首创）：用注意力温度 $\tau$ 构造劣质预测充当"无条件预测"：
$$\tilde{\mu}_i^t = (1+w)\mu_i^t(\cdot; 1) - w \cdot \mu_i^t(\cdot; \tau)$$

## 实验关键数据

### 密度估计：ImageNet 64×64 (BPD ↓)

| 模型 | 类型 | BPD |
|------|------|-----|
| Flow Matching | Diff/FM | 3.31 |
| NFDM | Diff/FM | 3.20 |
| VDM | Diff/FM | 3.40 |
| Sparse Transformer | AR | 3.44 |
| Flow++ | Flow | 3.69 |
| Glow | Flow | 3.81 |
| **TarFlow [2-768-8-8]** | **NF** | **2.99** |

首次突破 3 BPD！比之前最强的 NFDM 低 0.21。

### 条件生成：ImageNet 64×64 (FID ↓)

| 模型 | 类型 | FID |
|------|------|-----|
| EDM | Diff/FM | 1.55 |
| ADM (dropout) | Diff/FM | 2.09 |
| BigGAN | GAN | 4.06 |
| **TarFlow (w=2)** | **NF** | **5.7** |

### 条件生成：ImageNet 128×128 (FID ↓)

| 模型 | 类型 | FID |
|------|------|-----|
| Simple Diffusion | Diff/FM | 1.94 |
| ADM-G | Diff/FM | 2.97 |
| BigGAN-deep | GAN | 5.70 |
| **TarFlow** | **NF** | **5.03** |

### 无条件生成：ImageNet 64×64 (FID ↓)

| 模型 | 类型 | FID |
|------|------|-----|
| AGM | Diff/FM | 10.07 |
| IC-GAN | GAN | 10.40 |
| **TarFlow** | **NF** | **18.42** |

### 消融实验关键发现

- **VP vs NVP**：去掉 scale 项 $\alpha$（VP 模式）FID 从 5.7 恶化至 51.0
- **通道耦合 vs 自回归**：替换为 channel coupling，FID 恶化至 20.4
- **深度配置**：$T=K$（block 数=每 block 层数）时最优；$T=1$（单方向自回归）完全失败（FID=267）
- **噪声消融**：去噪步骤在 $\sigma=0.05$ 附近达到最佳 FID，且去噪后 FID 的 $\sigma$ 鲁棒性大幅提升

### 训练配置

- 优化器：AdamW，动量 (0.9, 0.95)，余弦学习率调度，峰值 $10^{-4}$
- 硬件：A100 GPU，所有实验在 14 天内完成
- 精度：生成任务用 bfloat16，似然估计用 float32
- 采样速度：32 张图约 2 分钟（单卡 A100，ImageNet 64×64）

## 亮点与洞察

1. **架构极简主义的胜利**：不需要 1×1 卷积、多尺度耦合层等复杂模块，仅靠堆叠 causal ViT + 交替方向即可大幅超越历史最佳
2. **NF 与 Diffusion 的桥梁**：采样轨迹可视化显示 TarFlow 的 $z^t$ 序列从噪声到图像的变化过程与扩散模型非常相似，尽管训练目标完全不同
3. **损失与 FID 的正相关**：训练损失（似然）下降直接带来 FID 改善，这是 NF 相对于其他生成模型的独特优势
4. **Guidance 与 NF 的兼容性**：首次证明 CFG 和无条件 guidance 可以直接应用于 NF 模型

## 局限与展望

1. **采样速度慢**：逆变换必须对序列维度逐步自回归，虽然使用了 KV-cache 但仍远慢于扩散模型的并行去噪
2. **FID 仍有差距**：条件 ImageNet 64×64 上 FID 5.7 vs EDM 的 1.55，差距约 3-4 倍
3. **无条件生成较弱**：无条件 FID 18.42 远不如 AGM 的 10.07
4. **分辨率受限**：最高仅展示 256×256（AFHQ），未在高分辨率（512+）上验证
5. **去噪步骤内存开销大**：score-based 去噪需要缓存全部中间激活做反向传播
6. **Guidance schedule 未充分探索**：论文初步发现线性递增 $w_i$ 更优但未深入研究

## 相关工作与启发

- **与 MAF/IAF 的关系**：TarFlow 是 MAF 的分块推广 + Transformer 骨干替换
- **与 Flow Matching 的区别**：Flow Matching 训练 velocity 预测的 ODE，需要大量高斯噪声；TarFlow 直接用 MLE 训练，噪声量小一个数量级
- **与 JetFormer 的区别**：JetFormer 用 NF 做 tokenizer + AR Transformer 两阶段；TarFlow 是单模型端到端
- **启发**：NF 可能被长期低估，核心是缺乏可扩展的架构。Transformer 的引入可能为其他"被遗忘"的经典方法提供类似的复兴机会

## 评分
- 新颖性: ⭐⭐⭐⭐ — 架构思路简单但洞察深刻，三项采样技术（尤其是无条件 guidance）颇有创意
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融充分（噪声、去噪、guidance、VP/NVP、深度配置），多数据集多设置
- 写作质量: ⭐⭐⭐⭐⭐ — 清晰流畅，公式推导完整，动机阐述到位
- 价值: ⭐⭐⭐⭐ — 为 NF 领域注入新活力，但 FID 与扩散模型仍有差距，实际应用前景待观察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/image_generation/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](../../NeurIPS2025/image_generation/multimodal_generative_flows_for_lhc_jets.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](../../ICML2026/image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
