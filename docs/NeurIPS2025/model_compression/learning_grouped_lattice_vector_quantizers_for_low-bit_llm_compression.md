---
title: >-
  [论文解读] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression
description: >-
  [NeurIPS 2025][模型压缩][格向量量化] GLVQ 提出为 LLM 权重的每个分组学习专属的格（lattice）码本（由可学习生成矩阵定义），配合分组特异的 μ-law companding 变换适应重尾分布，在 2-bit 量化下 Llama-2-70B 的 Wikitext-2 困惑度达到 3.36，大幅领先 QuIP#（3.91）和 QTIP（3.78）。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 格向量量化
  - 低比特压缩
  - 后训练量化
  - 可学习码本
  - companding变换
---

# Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression

**会议**: NeurIPS 2025  
**arXiv**: [2510.20984](https://arxiv.org/abs/2510.20984)  
**代码**: [GitHub](https://github.com/xzhang9308/GLVQ)  
**领域**: model_compression  
**关键词**: 格向量量化, 低比特压缩, 后训练量化, 可学习码本, companding变换

## 一句话总结
GLVQ 提出为 LLM 权重的每个分组学习专属的格（lattice）码本（由可学习生成矩阵定义），配合分组特异的 μ-law companding 变换适应重尾分布，在 2-bit 量化下 Llama-2-70B 的 Wikitext-2 困惑度达到 3.36，大幅领先 QuIP#（3.91）和 QTIP（3.78）。

## 研究背景与动机

**领域现状**：后训练量化（PTQ）是 LLM 部署压缩的主流方案。标量量化（如 GPTQ）在 4-bit 以上效果尚可，但低于 3-bit 时性能严重退化。向量量化（VQ）方法（如 QuIP#、AQLM）通过利用高维空间的结构化码本提升量化保真度。

**现有痛点**：QuIP# 使用固定的 $E_8$ 格来量化所有组/层的权重，忽略了不同权重组统计特性的差异，导致某些组量化失配。AQLM 学习自由形式的 VQ 码本，虽灵活但解码需要查表操作，速度慢。

**核心矛盾**：固定格（如 $E_8$）结构化程度高但适应性差；自由形式 VQ 适应性强但计算复杂度高——需要在码本灵活性和解码效率之间取得平衡。

**本文目标** 设计一种既保持格量化的高效解码（简单矩阵乘法），又能自适应不同权重组分布的量化方案。

**切入角度**：每个权重组学习一个独立的生成矩阵 $\mathbf{G}_g$ 定义格码本，配合可学习的 companding 变换 $F_g$ 处理非均匀分布。

**核心 idea**：分组学习格生成矩阵 + 分组 μ-law companding = 保持格结构化解码效率的同时适配局部权重分布。

## 方法详解

### 整体框架
输入：LLM 权重矩阵 $\mathbf{W}$。首先通过 Salience-Determined Bit Allocation 确定每组比特宽度 → 各组权重经分组 companding 变换 → Babai rounding 量化到学习的格码本 → 反向 companding 重建。解码时仅需矩阵-向量乘法 $\hat{\mathbf{w}} = F_g^{-1}(\mathbf{G}_g \mathbf{z})$。

### 关键设计

1. **Salience-Determined Bit Allocation (SDBA)**:

    - 功能：在全局比特预算约束下为每个权重组分配最优比特宽度
    - 核心思路：最小化量化后输出的 KL 散度 $D_{KL}(\mathbf{WX} \| \hat{\mathbf{W}}\mathbf{X})$，约束 $\frac{1}{G}\sum_g b_g = N$ 且高 1-bit 和低 1-bit 的组数相等
    - 搜索算法：双指针法，仅需 $\mathcal{O}(\log m)$ 次迭代
    - 例如 2-bit 目标：高 salience 组用 3-bit，低 salience 组用 1-bit，其余用 2-bit

2. **可学习格码本（Lattice Codebook Learning）**:

    - 功能：为每个权重组学习专属的格结构
    - 核心公式：把权重组 $\mathbf{W}_g \in \mathbb{R}^{m_g \times n_g}$ reshape 为 $d \times \ell_g$，通过生成矩阵量化：$\hat{\mathbf{W}}_g = \mathbf{G}_g \mathbf{Z}_g$
    - 优化目标：$\mathcal{L}_g = \|\mathbf{W}_g \mathbf{X} - \mathbf{G}_g \mathbf{Z}_g \mathbf{X}\|_2^2 + \lambda \|\mathbf{G}_g - \mathbf{G}_g^{(0)}\|_2^2$
    - 交替优化：(i) 固定 $\mathbf{G}_g$，Babai rounding 更新整数索引 $\mathbf{z}_i = \lfloor \mathbf{G}_g^{-1} \mathbf{w}_i \rceil$（复杂度 $\mathcal{O}(d^3)$）；(ii) 固定 $\mathbf{Z}_g$，梯度下降更新 $\mathbf{G}_g$，梯度 $\nabla_{\mathbf{G}_g} \mathcal{L}_g = -2(\mathbf{W}_g \mathbf{X} - \mathbf{G}_g \mathbf{Z}_g \mathbf{X})(\mathbf{Z}_g \mathbf{X})^\top$
    - 初始化：$\mathbf{G}_g^{(0)}$ 由组协方差矩阵的 Cholesky 分解得到，使初始格方向对齐权重的主分布
    - 稳定化：谱归一化限制 $\mathbf{G}_g$ 的奇异值在 $[\sigma_{\min}, \sigma_{\max}]$ 范围内
    - **关键区别**：QuIP# 全局用固定 $E_8$ 格，AQLM 学习自由码本需查表解码；GLVQ 学习组特异的格但保持格结构，解码仅需矩阵乘法

3. **分组 μ-law Companding**:

    - 功能：在量化前将重尾权重分布压缩为更均匀的分布，减少低幅值区域的量化误差
    - 变换公式：$F_g(x) = \text{sgn}(x) \frac{\ln(1 + \mu_g |x|)}{\ln(1 + \mu_g)}$，反变换 $F_g^{-1}(y) = \text{sgn}(y) \frac{(1+\mu_g)^{|y|}-1}{\mu_g}$
    - 可学习参数：$\mu_g > 0$ 控制压缩强度，与 $\mathbf{G}_g$ 联合梯度优化
    - 初始化：$\mu_g^{(0)} = 100 \tanh(\kappa_g / 10)$，$\kappa_g$ 为组的样本峰度——重尾组初始压缩更强
    - 约束：$\mu_g \in [10, 255]$，保证数值稳定性
    - 完整编解码链：$\tilde{\mathbf{W}}_g = F_g(\mathbf{W}_g) \to \mathbf{Z}_g = \lfloor \mathbf{G}_g^{-1} \tilde{\mathbf{W}}_g \rceil \to \hat{\mathbf{W}}_g = F_g^{-1}(\mathbf{G}_g \mathbf{Z}_g)$

### 运行时特性
- **存储开销极小**：每组只需存储 $d \times d$ FP16 生成矩阵 + 1个 FP16 标量 $\mu_g$，对 Llama 2-7B 仅增加约 2MB（总量 1.1GB 的 0.2%）
- **解码高效**：每个子块仅需 $d^2 + d$ 次乘法，端到端延迟比 4-bit uniform PTQ 仅增加 2-3%
- **流式解码**：推理时仅物化少量子块，即用即释，峰值内存比预先解压整层降低 >10×

## 实验关键数据

### Perplexity 主表（2-bit 量化）

| 方法 | Llama1-7B | Llama1-13B | Llama1-65B | Llama2-7B | Llama2-13B | Llama2-70B |
|------|-----------|------------|------------|-----------|------------|------------|
| FP16 | 5.68 | 5.09 | 3.53 | 5.12 | 4.57 | 3.12 |
| OmniQuant | 15.5 | 13.2 | 7.58 | — | — | — |
| QuIP# | 6.86 | 5.97 | 4.36 | 6.19 | 5.35 | 3.91 |
| QTIP | 6.52 | 5.80 | 4.21 | 5.91 | 5.26 | 3.78 |
| GLVQ-8D | 6.28 | 5.64 | 4.01 | 5.69 | 5.02 | 3.62 |
| **GLVQ-32D** | **6.00** | **5.38** | **3.81** | **5.41** | **4.80** | **3.36** |

GLVQ-32D 在所有模型规模上均取得最低困惑度，Llama2-70B 2-bit 下 PPL=3.36 vs QuIP# 3.91（降低0.55）。

### Zero-Shot 准确率（Llama-2-70B，4-bit）

| 方法 | ARC-C | ARC-E | PIQA | WINO |
|------|-------|-------|------|------|
| FP16 | 51.1 | 77.7 | 81.1 | 77.0 |
| QuIP# | 50.6 | 78.1 | 81.4 | 77.1 |
| QTIP | 50.0 | 77.6 | 81.5 | 77.0 |
| **GLVQ-8D** | **51.2** | 78.0 | **81.6** | **77.3** |

4-bit 下 GLVQ-8D 在多数任务超越 QuIP# 和 QTIP；2-bit 下优势更加明显。

### GLVQ-8D vs GLVQ-32D 对比

| 维度 | GLVQ-32D | GLVQ-8D |
|------|----------|---------|
| PPL | 更低，更好保真度 | 略高于 32D |
| 编码速度 | Babai rounding $\mathcal{O}(d^3)$ 更慢 | 更快 |
| 适用场景 | 追求极致压缩质量 | 平衡效率与质量 |

### 关键发现
- 比特宽度越低，GLVQ 的优势越明显（2-bit > 3-bit > 4-bit），说明组自适应格在极端压缩下价值最大
- 更大的格维度 $d$ 带来更好的量化保真度，但编码复杂度为 $\mathcal{O}(d^3)$
- Companding 的贡献在重尾分布权重组上最显著
- Cholesky 初始化比随机初始化收敛更快、最终 PPL 更低
- 存储开销仅 0.2%，延迟增加仅 2-3%，实用性极强

## 亮点与洞察
- **结构化与灵活性的巧妙平衡**：GLVQ 保持了格量化的高效解码（矩阵乘法），同时通过学习生成矩阵获得了自由码本的灵活性。这种"在约束空间内最优化"的思路值得借鉴
- **Companding 的优雅应用**：将通信工程中经典的 μ-law 变换引入权重量化，且每组学习不同的 $\mu_g$，配合峰度初始化策略，用简单手段解决了重尾分布问题
- **极低存储开销**：每组仅需一个小矩阵+一个标量的额外存储，几乎可忽略，工程部署友好
- **Babai rounding 代替精确最近格点搜索**：虽然是近似，但有形式化的误差界，且可微训练流程自然补偿误差

## 局限与展望
- 仅在 Llama 1/2 系列验证，缺少 Llama 3、Qwen、Mistral 等新模型的实验
- SDBA 的比特分配策略来自 Slim-LLM，不是本文的原创贡献
- 格维度 $d$ 的选择缺乏自适应机制，目前为手动设定
- 未与 PV-Tuning（分阶段优化连续参数和离散赋值）进行对比
- 推理速度实验仅在 RTX 4090 上测试，不同硬件上的实际加速比可能有差异
- 未验证权重+激活联合量化的场景

## 相关工作与启发
- **vs QuIP#**: 使用固定 $E_8$ 格 + Hadamard 旋转预处理，所有组用相同码本；GLVQ 为每组学习独立生成矩阵，2-bit 下 PPL 持续低 0.4-0.5 点
- **vs AQLM**: 学习自由形式向量码本，灵活但需查表解码；GLVQ 保持格结构，解码仅需矩阵乘法，速度更快
- **vs QTIP**: 通过有状态解码实现超高维 VQ，解耦了码本大小与比特率；GLVQ 思路不同，在固定格维度下优化格几何结构
- **vs GPTQ/AWQ**: 标量量化方法，4-bit 以上竞争力强，但 2-3 bit 下大幅落后于 VQ 方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 分组可学习格码本 + 分组 companding 的组合是有意义的创新，但各单项技术（格量化、μ-law、SDBA）已有先例
- 实验充分度: ⭐⭐⭐⭐ Llama 1/2 全系列、PPL+zero-shot 双评估、8D/32D两个配置对比，但缺少更多模型族
- 写作质量: ⭐⭐⭐⭐⭐ 方法推导严谨，pipeline 图清晰，算法伪代码完整，存储/延迟分析透彻
- 价值: ⭐⭐⭐⭐⭐ 在极低比特量化这个工程价值极高的方向上取得实质性突破，代码开源、延迟增加微小，部署友好
