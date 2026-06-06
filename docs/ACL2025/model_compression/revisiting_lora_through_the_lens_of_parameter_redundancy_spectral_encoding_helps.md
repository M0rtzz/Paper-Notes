---
title: >-
  [论文解读] Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps
description: >-
  [ACL 2025][模型压缩][LoRA] 本文系统研究了 LoRA 微调中的参数冗余问题，发现降低密度冗余不会损害表达能力（稀疏性质），并提出 SeLoRA——利用频谱变换（Fourier/Wavelet）从稀疏频谱子空间重参数化 LoRA 矩阵，以更少参数实现更优性能…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "spectral encoding"
  - "sparse learning"
  - "low-rank adaptation"
---

# Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps

**会议**: ACL 2025  
**arXiv**: [2506.16787](https://arxiv.org/abs/2506.16787)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: LoRA, parameter-efficient fine-tuning, spectral encoding, sparse learning, low-rank adaptation

## 一句话总结

本文系统研究了 LoRA 微调中的参数冗余问题，发现降低密度冗余不会损害表达能力（稀疏性质），并提出 SeLoRA——利用频谱变换（Fourier/Wavelet）从稀疏频谱子空间重参数化 LoRA 矩阵，以更少参数实现更优性能，且可即插即用地集成到多种 LoRA 变体中。

## 研究背景与动机

### 问题背景
Low-Rank Adaptation (LoRA) 是当前大模型微调的主流方法，通过两个低秩矩阵的乘积近似权重更新。然而近期研究发现 LoRA 的参数存在显著冗余，限制了其容量和效率。

### 核心发现：LoRA 的稀疏性质
作者从两个角度系统研究了 LoRA 中的冗余影响：

**秩冗余 (Rank Redundancy)**：直接降低 LoRA 秩 → 性能明显下降

**密度冗余 (Density Redundancy)**：在固定秩下将部分参数 mask 为零 → 即使 mask 掉 60% 参数，性能仍与完整 LoRA 相当

这一发现说明 LoRA 的参数并未被充分利用，存在提升空间。核心问题：**如何利用 LoRA 的稀疏性质释放其潜力？**

### 动机
传统剪枝方法在 LoRA 微调中需要复杂策略，而权重矩阵的频谱编码方法可以用稀疏的频谱分量实现高表达力的表示学习，是更简洁有效的替代方案。

## 方法详解

### SeLoRA 框架总览
SeLoRA (Spectral-encoding Low-Rank Adaptation) 的核心思想是将 LoRA 的低秩矩阵 A 和 B 重参数化为稀疏频谱分量的空间域等价物：

$$\mathbf{W}' = \mathbf{W}_0 + \tilde{\mathbf{B}} \tilde{\mathbf{A}}, \quad \tilde{\mathbf{A}} = \mathcal{T}(\mathbf{F}_A), \quad \tilde{\mathbf{B}} = \mathcal{T}(\mathbf{F}_B)$$

其中 $\mathcal{T}(\cdot)$ 为逆频谱变换，$\mathbf{F}_A, \mathbf{F}_B$ 为稀疏频谱矩阵。

### 关键设计

1. **稀疏比率 η**：控制 mask 掉的参数比例，可学习的频谱分量数为 $\lfloor(1-\eta) \cdot rd\rfloor$
2. **全局共享索引集 Ω**：随机初始化，指定所有低秩矩阵共享的可学习频谱位置
3. **两种频谱变换**：

    - **Fourier 编码 (SeLoRA_F)**：使用离散逆 2D 傅里叶变换，取实部简化计算；擅长从稀疏分量捕获高保真信息
    - **Wavelet 编码 (SeLoRA_W)**：使用离散逆 2D 小波变换（默认 Haar 小波）；提供局部化和层次化的信息重建，在平滑性与细节保留间取得更好平衡

4. **初始化策略**：对频谱矩阵使用 Kaiming 初始化后通过辅助矩阵校正方差，确保空间域方差一致
5. **即插即用**：可无缝集成到 DoRA、X-LoRA、HiRA 等 LoRA 变体中

### 效率优势
- 训练时利用快速频谱变换 (FFT/FWT)，额外计算开销极小
- 推理时无额外开销（频谱分量可预先转换为空间域权重）
- 参数量显著减少（通常降至 LoRA 的 40%-60%）

## 实验关键数据

### 表1：常识推理任务（8个基准平均准确率）

| 方法 | 模型 | 参数量(%) | 训练时间 | BoolQ | PIQA | HellaS. | ARC-c | OBQA | **平均** |
|------|------|-----------|----------|-------|------|---------|-------|------|----------|
| LoRA | LLaMA2-7B | 0.83 | 7.4h | 71.4 | 81.4 | 87.8 | 67.5 | 81.5 | 79.4 |
| SeLoRA_F | LLaMA2-7B | 0.50 | 7.6h | 72.8 | 83.4 | 90.9 | 70.5 | 83.4 | **81.3** (+1.9) |
| SeLoRA_W | LLaMA2-7B | 0.50 | 7.5h | 72.9 | 83.3 | 92.1 | 71.9 | 83.2 | **81.6** (+2.2) |
| DoRA | LLaMA2-7B | 0.84 | 12.2h | 71.8 | 83.1 | 90.1 | 69.5 | 82.4 | 80.1 |
| SeDoRA_W | LLaMA2-7B | 0.51 | 12.4h | 73.7 | 83.8 | 92.0 | 71.6 | 83.0 | **81.9** (+1.8) |
| LoRA | LLaMA3-8B | 0.70 | 7.8h | 74.0 | 88.2 | 94.0 | 78.1 | 84.0 | 84.0 |
| SeLoRA_W | LLaMA3-8B | 0.28 | 8.0h | 76.0 | 89.3 | 95.9 | 81.4 | 86.6 | **85.9** (+1.9) |
| SeDoRA_W | LLaMA3-8B | 0.28 | 13.0h | 76.2 | 89.7 | 96.0 | 82.0 | 87.8 | **86.5** (+1.3) |

### 表2：数学推理与代码生成

| 方法 | 模型 | 参数量(%) | GSM8k | MATH | 数学平均 | HumanEval | MBPP | 代码平均 |
|------|------|-----------|-------|------|----------|-----------|------|----------|
| LoRA | LLaMA2-7B | 0.83 | 60.5 | 11.7 | 36.1 | 32.1 | 35.8 | 31.8 |
| SeLoRA_W | LLaMA2-7B | 0.66 | 62.4 | 13.7 | **38.1** (+2.0) | 35.2 | 40.1 | **34.8** (+3.0) |
| SeDoRA_W | LLaMA2-7B | 0.67 | 63.0 | 14.1 | **38.6** (+1.9) | 33.5 | 41.0 | **34.8** (+1.1) |
| LoRA | LLaMA3-8B | 0.70 | 77.2 | 28.2 | 52.7 | 57.9 | 64.8 | 57.7 |
| SeLoRA_W | LLaMA3-8B | 0.42 | 80.3 | 29.8 | **55.1** (+2.4) | 59.3 | 66.1 | **59.4** (+1.7) |
| SeDoRA_W | LLaMA3-8B | 0.42 | 80.4 | 30.3 | **55.4** (+2.0) | 63.4 | 63.5 | **59.6** (+1.6) |

### 表3：不同小波基的鲁棒性（LLaMA3-8B）

| 小波基 | 常识推理 | 数学 | 代码 | 平均 |
|--------|----------|------|------|------|
| LoRA (基线) | 83.9 | 52.7 | 57.6 | 64.7 |
| Haar | 85.9 | 55.1 | 59.4 | 66.8 |
| Daubechies-4 | 85.9 | 55.4 | 59.1 | 66.8 |
| Biorthogonal | 85.9 | 54.8 | 59.5 | 66.7 |
| Coiflets | 86.2 | 55.2 | 59.8 | **67.0** |

## 亮点

1. **深刻的冗余分析**：系统区分秩冗余与密度冗余的不同影响，揭示 LoRA 的"稀疏性质"——mask 掉 60% 参数性能不降，为后续方法设计提供了清晰的理论动机
2. **频谱重参数化的精巧设计**：利用 Fourier/Wavelet 变换的天然稀疏表达能力，无需复杂剪枝策略即可高效利用冗余参数空间
3. **出色的即插即用特性**：可无缝集成到 LoRA、DoRA、HiRA 等多种变体，且 Wavelet 编码在几乎所有场景下均稳定优于 Fourier 编码
4. **参数效率极高**：以约 40% 的参数量（如 LLaMA3-8B 上 0.28% vs 0.70%）实现 +1.9 的常识推理平均提升
5. **数据效率**：仅用 25% 训练数据即可超过 LoRA 使用全部数据的性能
6. **子空间分析**：通过放大因子 (AF) 和反向放大因子 (RAF) 证明 SeLoRA 更高效地放大任务相关特征、抑制已强调特征的冗余放大

## 局限性

1. **高秩场景收益递减**：随着秩增大，SeLoRA 相对 LoRA 的提升逐渐减小，最终收敛到相似的上界，受限于 LoRA 本身的容量极限
2. **仅适用于同更新范式的变体**：当前仅兼容与 LoRA 相同更新模式的变体（BA 分解），对 SVD 等替代更新策略的集成尚未探索
3. **大模型验证缺失**：受限于计算资源，未在 70B 级别模型上验证可扩展性
4. **频谱位置选择**：索引集 Ω 采用随机初始化而非自适应选择，可能不是最优策略

## 相关工作

- **LoRA 系列**：LoRA (Hu et al., 2021)、DoRA (Liu et al., 2024)、HiRA (Huang et al., 2025)、X-LoRA (Buehler & Buehler, 2024)
- **稀疏重参数化**：LS-LoRA (He et al., 2022)、LoRETTA (Yang et al., 2024)、FourierFT (Gao et al., 2024)、LoRA-XS (Balazy et al., 2024)
- **冗余分析**：参数剪枝 (Han et al., 2015)、Lottery Ticket Hypothesis (Frankle & Carlin, 2018)
- **频谱编码**：权重矩阵的频谱编码 (Koutnik et al., 2010; Van Steenkiste et al., 2016; Wolter et al., 2020)

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 将频谱编码与 LoRA 稀疏性质巧妙结合，思路新颖；但频谱编码本身并非全新概念 |
| 实验充分度 | 5 | 三大任务类型、两个模型规模、六种 LoRA 变体、多维度消融分析，极为全面 |
| 写作质量 | 4 | 逻辑清晰，从冗余分析到方法设计的推导自然流畅 |
| 实用性 | 5 | 即插即用、无推理开销、参数更少性能更好，实际部署价值高 |
| **总分** | **4.5** | 一篇扎实的 PEFT 改进工作，兼具理论洞见与工程实用性 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](../../CVPR2025/model_compression/faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

</div>

<!-- RELATED:END -->
