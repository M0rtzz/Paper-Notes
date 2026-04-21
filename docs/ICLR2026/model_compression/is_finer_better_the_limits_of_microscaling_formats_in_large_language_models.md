---
title: >-
  [论文解读] Is Finer Better? The Limits of Microscaling Formats in Large Language Models
description: >-
  [ICLR 2026][模型压缩][微缩放量化] 发现并解释了微缩放（microscaling）量化中"更细粒度反而更差"的反直觉异常——当block size减小到阈值以下时，FP8 UE4M3 scale的有限动态范围导致窄分布张量的量化误差反而增大，并提出 FP8 UE5M3 scale格式作为硬件友好的解决方案。
tags:
  - ICLR 2026
  - 模型压缩
  - 微缩放量化
  - FP4
  - 量化异常
  - 动态范围
  - LLM量化
---

# Is Finer Better? The Limits of Microscaling Formats in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2601.19026](https://arxiv.org/abs/2601.19026)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 微缩放量化, FP4, 量化异常, 动态范围, LLM量化

## 一句话总结
发现并解释了微缩放（microscaling）量化中"更细粒度反而更差"的反直觉异常——当block size减小到阈值以下时，FP8 UE4M3 scale的有限动态范围导致窄分布张量的量化误差反而增大，并提出 FP8 UE5M3 scale格式作为硬件友好的解决方案。

## 研究背景与动机
LLM对计算和内存的需求不断增长，降低数值精度成为关键优化路径。微缩放格式（microscaling）通过共享block-wise scale来实现FP4级别的激进压缩，已被NVIDIA和AMD硬件原生支持（如NVFP4使用16个FP4元素共享一个FP8 UE4M3 scale）。

一般直觉认为：block size越小→每个block的scale越精准→量化误差越低。用BF16（16位）scale确实如此。但当scale被量化为FP8 UE4M3时，出现了**困惑度反转(perplexity inversion)**——在某些模型上，block size从16减小到8反而使困惑度增大。

这个反直觉现象的发现具有实际重要性：业界正积极追求更小的block size以提高量化精度，如果存在根本性限制，设计方向需要调整。核心问题：为什么更细的粒度有时更差？如何修复？

## 方法详解

### 整体框架
三步分析路径：(1) 实验发现异常现象并定位根因；(2) 建立理论框架从第一性原理解释；(3) 基于分析提出硬件友好的解决方案 UE5M3 scale格式。

### 关键设计

1. **异常现象的实验定位**:

    - 功能：系统分析不同模型、不同block size下的量化行为
    - 核心思路：
        - BF16 scale下（scale未量化）：所有模型在block size减小时困惑度单调下降——符合预期
        - FP8 UE4M3 scale下：granite-3.3-8b在block size 16处出现反转，llama-3.1-8b在block size 8处反转，而llama-2-7b无反转
        - 逐张量MSE分析发现：约25%的block在更细粒度下误差更大。MSE与权重标准差 $\sigma$ 的关系呈交叉：$\sigma < 2\times10^{-2}$ 时，block size 8比16的MSE更高
    - 设计动机：模型间差异源于权重分布宽窄——窄分布模型（granite）受影响更大

2. **理论框架（正态分布假设）**:

    - 功能：从第一性原理推导MSE与 $\sigma$ 的解析关系
    - 核心思路：假设权重 $X \sim \mathcal{N}(0, \sigma)$，将MSE分解为三个独立贡献：
        - $\text{MSE}_{Z, x_i \neq x_{\max}}$：普通元素的量化误差，由scale $s_k$ 离散化和FP4元素量化共同决定
        - $\text{MSE}_{Z, x_i = x_{\max}}$：最大值元素的误差（scale未量化时此项为0，量化后不再为0）
        - $\text{MSE}_{Z, s=0}$：所有block元素被四舍五入为0的误差（当 $x_{\max} < s_{\min}/2$ 时触发）
    - 理论预测与实验数据吻合度极高（$\chi^2 \approx 4 \times 10^{-8}$）
    - 设计动机：理论揭示根因——窄分布时scale量化的有限动态范围（UE4M3最小非零值 $2^{-9}$）无法精准表示小块的scale

3. **FP8 UE5M3 Scale格式**:

    - 功能：扩展scale的动态范围以消除反转异常
    - 核心思路：将8位无符号FP8 scale中未使用的1个bit重新分配为指数位：UE4M3（4位指数+3位尾数，最小值 $2^{-9}$）→ UE5M3（5位指数+3位尾数，最小值 $2^{-17}$）。保持尾数处理逻辑不变，仅扩展指数处理1位
    - 硬件影响：尾数处理决定主要硬件复杂度，增加1位指数仅带来可忽略的硬件开销。scale生成可复用现有FP8 E5M2量化逻辑
    - 设计动机：扩展256倍的最小scale值范围（$2^{-9} \to 2^{-17}$），有效解决窄分布张量的表示问题

### 对比方案：Per-Tensor Scaling
NVIDIA当前的实现使用per-tensor scale来预先放大窄分布，但存在：(1)对outlier敏感——单个大值影响整个张量；(2)推理时需额外absmax计算或预校准。UE5M3无需per-tensor scaling即可达到更好或相当的效果。

## 实验关键数据

### 主实验（FP4量化，block size 8）

| 模型 | 格式 | Wiki PPL↓ | PIQA↑ | HellaSwag↑ | GSM8K↑ | MMLU↑ |
|------|------|----------|-------|-----------|--------|------|
| granite-3.3-8b | BF16 | 4.72 | 80.41 | 61.49 | 62.47 | 60.55 |
| granite-3.3-8b | UE4M3 | 7.43 | 76.50 | 55.98 | 32.37 | 48.82 |
| granite-3.3-8b | UE4M3-S | 5.39 | 78.84 | 58.86 | 44.88 | 55.23 |
| granite-3.3-8b | **UE5M3** | **5.04** | **79.98** | **60.26** | **56.17** | **57.51** |
| llama-3.1-8b | BF16 | 6.24 | 79.87 | 60.05 | 50.49 | 63.28 |
| llama-3.1-8b | UE4M3 | 7.23 | 78.29 | 57.72 | 32.30 | 56.18 |
| llama-3.1-8b | **UE5M3** | **6.79** | **78.84** | **58.94** | **42.15** | **60.97** |

### 消融：MSE三项分解

| $\sigma$范围 | 主导误差项 | 解释 |
|-------------|----------|------|
| 大（$>0.02$） | $\text{MSE}_{x_i \neq x_{\max}}$ | 普通元素量化误差主导 |
| 中（~$0.005$） | $\text{MSE}_{x_i = x_{\max}}$ | 最大值元素的scale量化误差重要 |
| 小（$<0.001$） | $\text{MSE}_{s=0}$ | 全block归零误差主导 |

### 关键发现
- UE5M3在granite-3.3-8b上比UE4M3+per-tensor scaling更优：GSM8K提升56.17 vs 44.88（+11.3%）
- 理论框架适用于多种分布（正态、均匀、拉普拉斯等）和多种格式（FP4/INT4/FP6 scale）
- 异常在窄权重分布模型（如SSM模型mamba-codestral-7b）上更加严重
- Block size越小，$\text{MSE}_{x_i=x_{\max}}$ 项的相对权重越大——这解释了为什么更细粒度更差

## 亮点与洞察
- 发现"更细=更差"的反直觉现象本身就很有价值——对业界盲目追求更小block size的设计方向是重要警示
- 理论框架的精确度令人印象深刻（$\chi^2$ 在 $10^{-8}$ 级别），且容易扩展到新格式
- UE5M3方案优雅简单——仅重新分配1个bit就获得256倍的动态范围扩展，硬件改动极小

## 局限与展望
- 仅分析了权重量化，激活量化的异常行为值得进一步研究
- UE5M3的最大值范围从 $2^{15}$（UE4M3）扩展到 $2^{31}$，对于大outlier可能有trade-off
- 理论框架假设正态分布，虽然实验验证吻合度好但严格证明仍缺

## 相关工作与启发
- **vs NVFP4**: NVFP4使用UE4M3+per-tensor scaling，UE5M3无需per-tensor scaling即可更优
- **vs BlockDialect**: BlockDialect通过codebook扩展元素表示，UE5M3从scale端解决问题，二者正交可组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 发现了新的量化异常并给出严密的理论解释
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型（LLM/SSM/混合）、多格式、理论-实验完美对应
- 写作质量: ⭐⭐⭐⭐⭐ 从现象到理论到方案的叙事逻辑清晰流畅
- 价值: ⭐⭐⭐⭐⭐ 直接影响硬件设计和量化实践

<!-- RELATED:START -->

## 相关论文

- [InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)

<!-- RELATED:END -->
