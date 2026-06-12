---
title: >-
  [论文解读] From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness
description: >-
  [CVPR2026][模型压缩][数据集蒸馏] 提出 QuADD 框架，将可微量化模块嵌入数据集蒸馏循环中，联合优化合成数据与量化参数，实现在固定比特预算下"更少样本 + 更低精度"的帕累托最优压缩。
tags:
  - "CVPR2026"
  - "模型压缩"
  - "数据集蒸馏"
  - "量化感知训练"
  - "率失真优化"
  - "低比特数据压缩"
  - "非均匀量化"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness

**会议**: CVPR2026  
**arXiv**: [2603.02411](https://arxiv.org/abs/2603.02411)  
**代码**: 未开源  
**领域**: 模型压缩 / 数据蒸馏  
**关键词**: 数据集蒸馏, 量化感知训练, 率失真优化, 低比特数据压缩, 非均匀量化

## 一句话总结

提出 QuADD 框架，将可微量化模块嵌入数据集蒸馏循环中，联合优化合成数据与量化参数，实现在固定比特预算下"更少样本 + 更低精度"的帕累托最优压缩。

## 背景与动机

1. **数据集蒸馏的局限**：现有 DD 方法主要通过减少合成样本数量 $M$ 或数据维度 $D$ 来提升紧凑性，但每个数据元素仍以 32-bit 全精度存储，忽视了比特精度 $b$ 这一可控自由度
2. **总比特预算视角**：数据集的真实存储开销为 $\text{Budget} = M \times D \times b$，仅优化 $M$ 或 $D$ 而不考虑 $b$ 无法充分挖掘存储效率
3. **后量化的缺陷**：先蒸馏再量化（post-quantization）会导致显著精度下降，因为合成样本未针对低精度进行优化
4. **分布式/边缘场景需求**：IoT 设备和带宽受限的分布式学习场景要求数据在传输和存储时尽可能紧凑，需要从"更少样本"转向"更少比特"
5. **已有方法的局限**：AutoPalette 等颜色量化方法仅适用于图像领域，增加训练复杂度且难以泛化到其他模态
6. **量化与蒸馏的协同难题**：量化涉及 clipping 和 rounding 两个不可微操作，直接嵌入梯度优化循环存在技术障碍

## 方法详解

### 整体框架

QuADD 想回答一个被忽略的问题：数据集蒸馏一直在"减少样本数 $M$ 或维度 $D$"上做文章，可每个数据元素仍按 32-bit 全精度存，比特精度 $b$ 这个自由度被白白浪费了——而真实存储开销其实是 $M \times D \times b$。它的做法是在标准 DD 循环里插一个可微量化层 $Q(\cdot)$，把优化目标从"全精度合成数据 $\mathcal{S}$ 匹配真实数据 $\mathcal{T}$"换成"量化后的 $\mathcal{S}^q$ 匹配 $\mathcal{T}$"：

$$\mathcal{S}^* = \arg\min_{\mathcal{S}} \mathbb{E}_{\theta \sim \Theta} [\mathcal{L}(\phi(\mathcal{T};\theta), \phi(\mathcal{S}^q;\theta))]$$

每轮迭代里，合成数据先过量化层得 $\mathcal{S}^q = Q(\mathcal{S})$ 再算蒸馏损失，梯度顺着链式法则同时回传到合成数据和量化器参数。这样合成样本从一开始就是"为低精度存储而生"的，而不是先蒸馏好再硬量化。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    T["真实数据 𝒯"] --> INIT["量化引导初始化<br/>均匀量化 + 图割贪心选样"]
    INIT --> S["合成数据 𝒮<br/>(可学习, 含量化阈值 α)"]
    S --> Q["可微量化层 Q<br/>hard rounding + STE / 非均匀 APoT"]
    Q --> SQ["量化合成数据 𝒮q"]
    T --> PHI["特征提取器 φ(·;θ)"]
    SQ --> PHI
    PHI --> L["蒸馏损失 ℒ(φ(𝒯), φ(𝒮q))"]
    L -->|"梯度同时回传至 𝒮 与 α"| S
    S --> OUT["低比特紧凑数据集<br/>存储预算 M×D×b"]
```

### 关键设计

**1. 可微量化层：让 clipping 和 rounding 进得了梯度循环**

量化天然带两个不可导操作——clipping 和 rounding，直接塞进端到端优化会断梯度。QuADD 在前向用 hard rounding 把连续值映射到最近码本级别（均匀/非均匀都适用），反向用 STE（Straight-Through Estimator）在 clipping 范围内把量化器当恒等映射，$\partial x^q / \partial x \approx \mathbf{1}(|x| \le \alpha)$；均匀量化基线另提供 soft rounding 配解析代理梯度。均匀码本是等间距级别 $Q^u(\alpha, b) = \alpha \times \{-1, \pm\frac{1}{2^{b-1}-1}, \ldots, 1\}$。

更进一步，真实数据分布并不均匀，于是默认采用自适应非均匀量化 APoT（Additive Powers-of-Two）：把每个量化值写成若干个缩放后 2 次幂之和，在数据密集区域分配更细的量化粒度。整个方案只引入一个可学习参数 $\alpha$（clipping 阈值），并通过 RCF（Reparameterized Clipping Function）让 $\alpha$ 从所有样本接收梯度。这一层轻量且模态无关，不引入额外计算开销。

**2. 量化引导初始化：让起点就贴合量化后的目标**

随机初始化合成数据会让低比特优化更难收敛。QuADD 改用基于广义图割的贪心选择来挑初始合成数据：先把真实数据均匀量化得 $\mathcal{T}^q$，再迭代地选那个使条件增益 $G^*(A|C)$ 最大的样本，样本相似度由最后一层梯度的余弦相似度衡量。这样初始集合在量化视角下就已经有代表性，给后续联合优化一个好起点。

### 训练流程

每轮迭代依次为：采样真实数据 mini-batch → 量化合成数据 → 计算蒸馏损失 → 梯度同时回传至 $\mathcal{S}$ 和 $Q$ → 更新两者。量化层轻量且模态无关，使得 $M$（样本数）和 $b$（比特数）能在同一个率失真框架下被联合优化。

## 实验关键数据

### 主实验：固定存储预算下的精度对比

| 方法 | CIFAR-10 IPC10 | CIFAR-100 IPC10 | ImageNette IPC10 | 压缩比 |
|------|---------------|----------------|-----------------|--------|
| DATM (32-bit) | 65.7% | 47.3% | 67.8% | 1× |
| FreD | 57.3% | 34.9% | 66.2% | 9.6-12× |
| AutoPalette (7-bit) | 63.5% | 45.6% | 63.0% | 9.6× |
| **QuADD (9-bit)** | **65.4%** | **46.2%** | **67.0%** | **10.6×** |

QuADD 在 10.6× 压缩比下精度仅比全精度基线低约 1%，大幅优于 AutoPalette 和 FreD。

### 跨域实验：3GPP 波束管理

- 全量 32-bit 数据训练：89% 精度
- 全量 8-bit 量化：87% 精度，4× 压缩
- 无量化 DD：77% 精度，46× 压缩
- **QuADD**：81.9% 精度，36× 压缩；77.5% 精度，**183× 压缩**

### 消融分析

- **率失真分析**：固定预算下，低比特 + 更多样本通常优于高比特 + 少样本，最优精度集中在 2-3 bits/sub-pixel
- **跨架构泛化**（Table 2）：在 AlexNet/VGG/ResNet 上，QuADD 在 IPC10 的 ResNet 上达 51.8%，优于 DATM (49.0%) 和 AutoPalette (50.1%)
- **跨蒸馏方法**（Table 3）：QuADD 兼容 TM 和 DM 两种蒸馏框架，在 TM 上 IPC10 达 63.2%，接近全精度的 65.2%
- **训练效率**：QuADD 在中高 IPC 下训练时间比 DATM 减少 25-30%，远快于 AutoPalette 和 FreD

## 亮点

- **视角新颖**：将 DD 问题从"减少样本"重新定义为"减少比特"，引入率失真理论框架分析 $(M, b)$ 的最优分配
- **框架通用**：量化模块模态无关、蒸馏方法无关，可即插即用到 TM/DM/DATM 等多种 DD 方法中
- **实用价值高**：在边缘计算和分布式学习场景下，10× 以上的存储压缩且精度损失极小
- **实现轻量**：自适应量化仅引入一个可学习参数 $\alpha$，不增加训练开销

## 局限与展望

- 实验主要在 CIFAR 和小规模数据集上验证，缺乏 ImageNet 等大规模数据集的结果
- APoT 非均匀量化的 bit-width 分解方式 $n = b/k$ 要求 $b$ 为 $k$ 的整数倍，灵活性受限
- 未与近期 latent-space DD 方法（如 SRe2L、RDED）结合验证
- 跨域实验仅涉及一个表格数据任务，NLP/音频等模态的泛化性有待验证
- 量化引导初始化的贪心选择策略计算梯度相似度，在大规模数据集上可能成本较高

## 与相关工作的对比

| 方法 | 压缩维度 | 适用模态 | 量化方式 | 是否端到端 |
|------|---------|---------|---------|-----------|
| FreD | 频域维度 $D$ | 图像 | 无 | 是 |
| AutoPalette | 颜色比特 $b$ | 图像 | 调色板量化 | 是 |
| IDC/HaBa/SPEED | 参数化维度 | 图像 | 无 | 是 |
| **QuADD** | **比特精度 $b$** | **图像+表格** | **自适应非均匀** | **是** |

QuADD 与 FreD/AutoPalette 的核心区别：(1) 量化模块模态无关，不局限于颜色空间；(2) 自适应非均匀量化根据数据分布分配量化粒度；(3) 从率失真角度联合优化 $M$ 和 $b$。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将量化与蒸馏联合优化的视角新颖，率失真分析有洞察力
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、跨架构、跨蒸馏方法、跨域验证较全面，但缺大规模数据
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 对资源受限场景有直接应用价值，框架通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](../../NeurIPS2025/model_compression/beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)

</div>

<!-- RELATED:END -->
