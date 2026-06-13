---
title: >-
  [论文解读] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models
description: >-
  [ACL 2025][模型压缩][三值量化] 本文系统研究三值语言模型（TriLM）的缩放规律，发现 TriLM 从增加训练数据中获益远大于增加参数量，基于此训练了在 1.2T token 上预训练的 Spectra-1.1 模型族（1B/2B/3B），并提出 1.6-bit 和 2-bit 权重打包方案及 TriRun GPU 内核，实现最高 8 倍的推理加速。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "三值量化"
  - "Scaling Law"
  - "推理加速"
  - "权重打包"
  - "GPU内核"
---

# Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models

**会议**: ACL 2025  
**arXiv**: [2506.23025](https://arxiv.org/abs/2506.23025)  
**代码**: 即将开源（Spectra-1.1 模型 + TriRun 推理内核，MIT 许可证）  
**领域**: 模型压缩 / LLM 效率  
**关键词**: 三值量化, Scaling Law, 推理加速, 权重打包, GPU内核

## 一句话总结

本文系统研究三值语言模型（TriLM）的缩放规律，发现 TriLM 从增加训练数据中获益远大于增加参数量，基于此训练了在 1.2T token 上预训练的 Spectra-1.1 模型族（1B/2B/3B），并提出 1.6-bit 和 2-bit 权重打包方案及 TriRun GPU 内核，实现最高 8 倍的推理加速。

## 研究背景与动机

**领域现状**：LLM 的推理效率受制于内存带宽瓶颈——GPU 的算力增长速度远快于内存容量和带宽的提升。后训练量化（PTQ）是当前主流的推理加速方案，但通常只能支持到 4-bit 精度，更低位宽会导致严重性能退化。近期的量化感知训练（QAT）方法表明，三值权重（-1, 0, 1）模型可以在更大参数规模下接近全精度模型的性能，展现出更高的比特效率。

**现有痛点**：三值语言模型（TriLM）领域存在三个关键空白。（1）缺乏 Scaling Law 分析——前期工作仅验证了参数扩展的效果，未系统研究训练 token 数量对 TriLM 性能的影响，无法指导最优的计算资源分配。（2）推理加速空白——现有的高效推理研究几乎全部集中在 4-bit 量化，sub-4-bit（尤其是三值）的专用推理内核和权重存储方案几乎不存在。（3）缺乏强开源模型——没有大规模训练的强开源 TriLM 模型族供社区研究，制约了后训练方法和应用探索。

**核心矛盾**：三值量化提供了极端的内存压缩比（16 倍于 FP16），但目前既没有理论指导告诉我们该如何分配训练计算（参数量 vs 训练数据），也没有工程基础设施让三值模型在实际推理中真正实现加速。

**本文目标** 三个子问题：（1）建立 TriLM 的 Scaling Law 并训练强开源模型族；（2）设计三值权重的高效存储方案（逼近信息论最优的 1.585 bit/weight）；（3）开发 GPU 推理内核实现端到端加速。

**切入角度**：作者首先通过系统的 Scaling Law 实验发现 TriLM 的关键特性：$\hat{L}(N,D) \approx 2.19 + 4.73/N^{0.32} + 5.18/D^{0.81}$，其中数据项指数（0.81）远大于参数项指数（0.32），说明增加训练数据比增加参数更有效。基于此洞察，他们固定参数量并将训练数据从 300B 扩展到 1.2T token。在推理端，利用三值权重仅有 3 个取值的特殊结构，设计了不需要浮点乘法的高效打包/解包方案。

**核心 idea**：TriLM 的 Scaling Law 指数显示"多喂数据"远比"加大模型"更有效，基于此训练 1.2T token 的 Spectra-1.1 模型族，配合专用 TriRun GPU 内核实现端到端 5 倍推理加速。

## 方法详解

### 整体框架

工作分为两个相对独立的部分。训练侧：在 decoder-only Transformer 架构上使用量化感知训练（QAT），线性层权重限制为 {-1, 0, 1} 加共享浮点缩放因子，前向传播时在线三值化、反向传播更新潜在的浮点权重。在 5 个参数规模 × 4 个数据量上训练 20 个模型拟合 Scaling Law，然后按最优策略在 1.2T token 上训练 1B/2B/3B 三个最终模型。推理侧：提出 2-bit（TQ2）和 1.6-bit（TQ1）两种权重打包方案，在 CPU 上集成到 llama.cpp，在 GPU 上开发 TriRun 混合精度内核（FP16 × INT2），利用异步内存拷贝和 Tensor Core 加速矩阵乘法。

### 关键设计

1. **TriLM Scaling Law 分析**:

    - 功能：建立三值模型的验证损失与参数量 $N$、训练 token 数 $D$ 的定量关系，指导计算资源分配
    - 核心思路：在 99M-1.1B 参数和 20B-150B token 的网格上训练 20 个 TriLM，拟合 Chinchilla 形式的参数化缩放定律 $\hat{L}(N,D) = E + A/N^\alpha + B/D^\beta$。拟合结果 $\alpha=0.32, \beta=0.81$ 表明数据项的衰减指数是参数项的 2.5 倍。对比全精度模型的缩放定律 $\alpha=0.56, \beta=0.53$（两项几乎对称），TriLM 明显偏向"数据优先"策略
    - 设计动机：这一发现直接指导了 Spectra-1.1 的训练策略——不追求更大参数量，而是在 1B-3B 规模下用 1.2T token 充分训练。相比 Spectra 1.0（300B token），MMLU 等基准显著提升

2. **高效权重打包方案（TQ2 和 TQ1）**:

    - 功能：将三值权重从朴素的 2-bit 表示压缩到更接近信息论极限的 1.6-bit 表示
    - 核心思路：TQ2（2-bit）方案将每个三值 $d_i \in \{-1,0,1\}$ 映射为 $d_i' = d_i + 1 \in \{0,1,2\}$，每 4 个 trit 用 8 bit 存储，每 256 个元素一个块加 FP16 缩放因子，有效位宽 2.0625 bit/weight。TQ1（1.6-bit）方案利用 $3^5 = 243 < 256 = 2^8$ 的近似关系，将 5 个 trit 编码为一个 8-bit 整数，有效位宽 1.6 bit/weight，接近理论最优 $\log_2(3) \approx 1.585$。解码时用乘法近似代替除法和取模运算以适配 SIMD
    - 设计动机：TQ2 更快（解码仅需位移和掩码），TQ1 更省内存（节省约 20%）。在内存受限环境下优先选 TQ1，在计算受限环境下选 TQ2

3. **TriRun GPU 推理内核**:

    - 功能：实现 FP16 × INT2 的混合精度矩阵乘法，在 GPU 上高效推理三值模型
    - 核心思路：基于 2-bit 打包方案，使用 CUDA 异步内存拷贝（cp.async）将 FP16 输入加载到共享内存并重叠计算，INT2 权重通过带缓存提示的异步拷贝加载以减少 L2 缓存污染。解包后的 FP16 权重片段与输入片段通过 Tensor Core mma 指令做块矩阵乘法，中间结果在 FP32 寄存器中累加以保持精度，最终转回 FP16 写入全局内存。使用双缓冲流水线和分层归约
    - 设计动机：三值权重的 2-bit 表示使得单个权重仅需 0.25 bytes，而 L40 GPU 的 FLOPs/Bytes 比约为 105，意味着在 batch size 约 13 以上时计算成为瓶颈而非内存，TriRun 正是为这种高 batch 场景优化

### 损失函数 / 训练策略

训练使用标准的因果语言建模交叉熵损失。量化感知训练策略保持潜在浮点权重，前向传播时通过取符号+缩放因子进行三值化。使用 AdamW 优化器，在 AMD MI250X 集群上实现了近线性的多 GPU 扩展（高达 2048 GPU）。

## 实验关键数据

### 主实验

Spectra-1.1（TriLM，1.6-bit 有效权重）vs LLaMA-1 7B（FP16）基准对比：

| 基准 | Spectra-1.1 1B | Spectra-1.1 2B | Spectra-1.1 3B | LLaMA-1 7B |
|------|---------------|---------------|---------------|------------|
| ARC Challenge (acc_norm) | 36.43 | 39.69 | 42.58 | 44.80 |
| ARC Easy (acc_norm) | 62.54 | 67.42 | 71.93 | 72.81 |
| HellaSwag (acc_norm) | 56.61 | 61.37 | 66.28 | 76.21 |
| BoolQ (acc) | 62.57 | 56.70 | 66.15 | 75.11 |
| LAMBADA (acc) | 47.31 | 48.85 | 54.22 | 73.53 |

TriRun GPU 推理加速（对比 PyTorch FP16 基线）：

| 场景 | 70B 模型 | 405B 模型 |
|------|---------|----------|
| Time to First Token (64 输入) | 4.7× | — |
| Time per Output Token | 4.9× | — |
| 层级加速（batch 16-32） | ~5× | **7.98×** |
| 端到端生成加速 | **4.9×** | — |

### 消融实验

| 配置 | 说明 |
|------|------|
| Spectra 1.0 (300B token) vs 1.1 (1.2T) | MMLU 持续提升，证实数据扩展有效 |
| $\alpha=0.32$ (TriLM) vs $\alpha=0.56$ (Float) | TriLM 参数扩展收益仅为 Float 的 57% |
| $\beta=0.81$ (TriLM) vs $\beta=0.53$ (Float) | TriLM 数据扩展收益为 Float 的 153% |
| TQ2 (2-bit) vs TQ1 (1.6-bit) | TQ2 速度更快，TQ1 内存更省 |

### 关键发现

- **TriLM 的 "数据优先" 特性**：缩放定律中数据项指数 $\beta=0.81$ 远大于参数项 $\alpha=0.32$，这与全精度模型的对称特性（$\alpha \approx \beta \approx 0.5$）形成鲜明对比。直觉上理解：三值权重的表达能力有限，增加参数只能有限扩展模型容量，但更多训练数据能让有限的参数学到更高效的表示
- **3B TriLM vs 7B Float**：Spectra-1.1 3B 在 ARC 等任务上接近 LLaMA-1 7B（42.58 vs 44.80），但内存占用仅约 1/15（三值 3B ≈ 0.6GB vs FP16 7B ≈ 14GB）
- **推理加速主要在大模型+高 batch**：TriRun 的加速倍率随模型增大和 batch 增大而提升（405B, batch 32 时达到 8×），因为此时内存带宽瓶颈更突出
- **70B TriLM 单 GPU 运行**：TriRun 使得 70B 三值模型可以在单张 L40S GPU 上运行，而 FP16 版本需要 4 张 GPU

## 亮点与洞察

- **TriLM 的 Scaling Law 反直觉但有理论支撑**：三值权重的极低表达能力使得参数扩展的边际收益快速递减，而数据扩展通过更充分的训练让每个三值参数承载更多信息。这个发现对所有极低位宽模型的训练策略都有指导意义
- **1.6-bit 打包的精巧设计**：利用 $3^5 \approx 2^8$ 的数论近似，将 5 个 trit 无损压缩到 1 byte，并用乘法迭代代替除法/取模解码以适配 SIMD，兼顾了理论最优性和工程实用性
- **TriRun 的实际部署价值**：在单 GPU 上运行 70B 模型并实现 5× 加速，对实际部署极低位宽模型有直接意义。这是首个面向 sub-4-bit 模型的 GPU 推理内核系统

## 局限与展望

- **Scaling Law 未考虑 bit 数 $b$ 的影响**：当前的缩放定律将 TriLM 和 FloatLM 分开拟合，未建立统一的 $(N, D, b)$ 三变量缩放定律
- **模型规模有限**：Spectra-1.1 仅训练了 3B 以下的模型，与 7B/13B 的全精度模型仍有较大差距，需要更多计算资源验证更大规模
- **TriRun 仅支持 2-bit**：更省内存的 1.6-bit 打包方案（TQ1）的 GPU 内核尚未开发，因解包操作更复杂
- **下游任务评估不含 NLP 生成任务**：所有基准均为选择题/分类任务，缺少对文本生成质量的评估

## 相关工作与启发

- **vs Spectra 1.0 (Kaushal et al. 2024)**：Spectra 1.0 使用 300B token 训练，未分析 Scaling Law；本文扩展到 1.2T token 并首次建立 TriLM 的 Scaling Law
- **vs GPTQ/AWQ (PTQ 方法)**：后训练量化方法在 4-bit 以下严重退化，而 QAT 训练的 TriLM 在 1.58-bit 有效精度下保持合理性能
- **vs BitNet (Wang et al. 2023)**：BitNet 验证了 1-bit/ternary 模型的可行性但未提供推理加速方案；本文的 TriRun 填补了推理工程空白

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次建立 TriLM 的 Scaling Law 并发现"数据优先"特性，1.6-bit 打包方案和 TriRun GPU 内核均为新贡献
- 实验充分度: ⭐⭐⭐⭐ 涵盖 Scaling Law 拟合、多基准评测、CPU/GPU 推理加速、多硬件平台验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Scaling Law 推导严谨，打包方案的理论分析完整
- 价值: ⭐⭐⭐⭐⭐ 全栈贡献（缩放定律+模型+推理内核+开源），对极低位宽模型的研究和部署有系统性推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](../../ICML2025/model_compression/an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)

</div>

<!-- RELATED:END -->
