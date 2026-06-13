---
title: >-
  [论文解读] CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs
description: >-
  [NeurIPS 2025][模型压缩][模型量化] 提出 CodeGEMM，一种以 codebook 为中心的 GEMM kernel，通过预计算 centroid 与 activation 的内积并缓存为 Psumbook，替代传统反量化流程，在 2-bit 量化 LLM 上实现 1.83×（8B）到 8.93×（70B）的端到端加速。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "模型量化"
  - "codebook 量化"
  - "GEMM 加速"
  - "LLM 推理"
  - "低比特量化"
  - "CUDA kernel"
  - "lookup table"
---

# CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2512.17970](https://arxiv.org/abs/2512.17970)  
**代码**: [GitHub](https://github.com/naver-aics/codegemm)  
**领域**: 模型压缩  
**关键词**: 模型量化, codebook 量化, GEMM 加速, LLM 推理, 低比特量化, CUDA kernel, lookup table

## 一句话总结

提出 CodeGEMM，一种以 codebook 为中心的 GEMM kernel，通过预计算 centroid 与 activation 的内积并缓存为 Psumbook，替代传统反量化流程，在 2-bit 量化 LLM 上实现 1.83×（8B）到 8.93×（70B）的端到端加速。

## 研究背景与动机

- **领域现状**：weight-only quantization 是 LLM 推理中缓解内存瓶颈的主流方案；codebook-based 量化（如 AQLM、GPTVQ、QuIP#）在极低比特（如 2-bit）下能保持较好精度，已成为前沿方向
- **现有痛点**：codebook 量化的推理 kernel 依赖 dequantization 流程——需将完整 codebook 加载到片上可编程缓存（shared memory），再逐元素查找 centroid 并重建权重，导致：(1) codebook 可能超出 shared memory 容量（如 AQLM 1×16 配置需 1MB，远超 A100 的 164KB）; (2) 每次 GEMM 都重复计算相同 centroid 与 input 的乘积，存在大量冗余
- **核心矛盾**：codebook 量化在精度上优于 uniform 量化，但其推理效率反而可能更差——AQLM 1×16 的 kernel 延迟甚至高于 FP16 baseline，完全抵消了压缩带来的内存优势
- **本文目标**：设计一个高效的 codebook-centric GEMM kernel，同时降低计算复杂度和片上存储需求，使 codebook 量化在极低比特下既保精度又提速
- **切入角度**：观察到 codebook 中 centroid 数量有限（$2^b$ 个），而权重矩阵行数 $M$ 远大于 $2^b$，因此大量 code 指向相同 centroid，导致与 input 的内积被重复计算。将这些内积预计算并缓存即可消除冗余
- **核心 idea**：用 Psumbook（预计算的 partial sum 查找表）替代 codebook 本身缓存到 shared memory，推理时直接按 code 索引取出预计算结果做累加，跳过 dequantization 步骤

## 方法详解

### 整体框架

CodeGEMM 将传统 codebook GEMM 的"加载 codebook → 反量化 → 矩阵乘法"三步流程简化为"构建 Psumbook → 按 code 索引取值 → 累加"。具体地：(1) 将 input tile 按 vector length $v$ 分段；(2) 对每段 input 与所有 $2^b$ 个 centroid 做内积，结果存入 Psumbook（缓存在 shared memory 中）；(3) 利用 code 矩阵中的索引直接从 Psumbook 读取对应的 partial sum 并累加得到输出。整个过程无需加载完整 codebook，也无需逐元素 dequantize。

### 关键设计一：Psumbook 预计算与缓存

- **功能**：对每段 input $\mathbf{x}^j$，预计算其与所有 centroid 的内积 $p_i^j = \sum_{k=0}^{v-1} c_k^i \times x_k^j$，将这些标量结果存入 Psumbook
- **核心思路**：codebook 存储的是 $v$ 维向量（centroid），而 Psumbook 存储的是标量（内积结果），空间复杂度从 $\mathcal{O}(m \cdot 2^b \cdot v)$ 降至 $\mathcal{O}(m \cdot 2^b \cdot t_w/v)$，与向量长度 $v$ 成反比
- **设计动机**：传统方法需要将整个 codebook 放入 shared memory，大 codebook（如 $2^{16}$ 个 centroid）直接超出容量限制；Psumbook 仅存标量结果，显著降低片上存储需求，使得各类 codebook 配置都能在 shared memory 中运行

### 关键设计二：计算复杂度缩减

- **功能**：CodeGEMM 的计算复杂度为 $\mathcal{O}(MNK \cdot m/v)$，相比标准 GEMM 的 $\mathcal{O}(MNK)$ 降低了 $v/m$ 倍
- **核心思路**：Psumbook 构建阶段 $C_{build} = \mathcal{O}(m \cdot 2^b \cdot K \cdot N)$ 在 $M \gg 2^b$ 时可忽略；读取阶段每个 code 仅需一次查表（而非 $v$ 次乘加），故 $C_{read} = \mathcal{O}(m \cdot M \cdot K/v \cdot N)$
- **设计动机**：dequantization-based kernel 只优化了数据搬运效率，计算量与 FP16 GEMM 完全相同；CodeGEMM 同时优化数据搬运和计算量，是真正的计算效率提升

### 关键设计三：统一 kernel 支持灵活超参数探索

- **功能**：单一 kernel 实现支持 codebook 数量 $m$、向量长度 $v$、bits per code $b$、group size $g$ 等超参数的任意组合
- **核心思路**：不同超参数组合可在相同的平均 bit 数下产生截然不同的 latency-accuracy 权衡（如 $(v=4,m=1,b=8,g=128)$ 与 $(v=16,m=3,b=8,g=32)$ 均约 2-bit，但性能差异显著）
- **设计动机**：现有 codebook kernel 通常仅针对固定配置优化；统一 kernel 允许用户系统性地探索 latency–memory–accuracy 三角权衡，找到特定场景下的最优配置

### 关键设计四：细粒度 group normalization

- **功能**：在量化前对权重按 group size $g$ 做归一化，$g$ 越小归一化越精细（$g=v$ 为 per-vector，$g=-1$ 为 per-row）
- **核心思路**：细粒度归一化降低量化误差，以少量额外 memory（存储 scale factor）换取精度提升
- **设计动机**：在 70B 模型上，细粒度归一化使 CodeGEMM 的精度追平 AQLM 1×16（后者使用 $2^{16}$ codebook），但 throughput 高出 8.93×

## 损失函数与训练策略

- **量化优化**：采用 AQLM 的 block-wise codebook 优化策略，在校准数据集上通过 K-means 聚类确定 centroid
- **PV-Tuning**：可选的后量化校准方法，进一步优化 codebook 以提升精度（+PV-Tuning 后 CodeGEMM-m1v4g128 在 Llama-3.1-8B 上平均精度从 53.93 提升至 63.96）

## 实验

### 实验一：Kernel 级延迟与端到端吞吐（Llama-3.1-8B，2-bit）

| 方法 | 配置 | Kernel 延迟 (μs) | 端到端吞吐 (tok/s) | 平均精度 |
|------|------|:-:|:-:|:-:|
| cuBLAS (FP16) | — | 332.45 | 103.8 | 71.26 |
| AQLM | 1×16 | 645.51 | 49.0 | 63.57 |
| AQLM | 2×8 | 250.12 | 124.5 | 47.82 |
| QuIP# | e8p | 162.63 | — | — |
| LUTGEMM | q2-g128 | 160.1 | — | — |
| **CodeGEMM** | m1v4g128 | **152.69** | **228.3** | 53.93 |
| **CodeGEMM+PV** | m1v4g128 | 152.69 | 228.3 | **63.96** |

CodeGEMM-m1v4g128 kernel 延迟最低（152.69μs），端到端吞吐 228.3 tok/s 是 AQLM 1×16 的 4.66×；加上 PV-Tuning 后精度（63.96）超过 AQLM 1×16（63.57），实现 1.83× 的端到端加速。

### 实验二：70B 模型可扩展性

| 方法 | $\bar{q}$ | tok/s | MMLU | WG | HS | ARC-E | ARC-C | Avg. |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| AQLM 1×16 | 2.055 | 5.5 | 73.07 | 76.16 | 80.83 | 82.20 | 57.17 | 73.89 |
| AQLM 2×8 | 2.002 | 19.0 | 61.45 | 59.59 | 52.83 | 48.82 | 28.67 | 50.27 |
| **CodeGEMM** m1v4g32 | 2.500 | 49.1 | 71.21 | 76.64 | 79.43 | 82.41 | 56.06 | **73.15** |
| **CodeGEMM** m1v4g128 | 2.125 | **51.2** | 68.15 | 74.90 | 75.37 | 79.42 | 52.73 | 70.11 |

在 70B 模型上，CodeGEMM-m1v4g128 吞吐 51.2 tok/s 是 AQLM 1×16（5.5 tok/s）的 **8.93×**；CodeGEMM-m1v4g32 精度 73.15 接近 AQLM 1×16 的 73.89，但吞吐快近 9 倍。

### 实验三：能效与硬件利用率

| 方法 | TFLOPS | 功耗 (W) | GFLOPS/W | Mem Util (%) |
|------|:-:|:-:|:-:|:-:|
| cuBLAS (FP16) | 1.58 | 318.55 | 4.95 | 96.94 |
| AQLM 1×16 | 0.75 | 126.54 | 5.93 | 6.00 |
| AQLM 2×8 | 2.59 | 254.20 | 10.18 | 19.96 |
| **CodeGEMM** m1v4g128 | **6.12** | 316.38 | **19.36** | **49.80** |

CodeGEMM 的能效（19.36 GFLOPS/W）是 AQLM 2×8 的 1.9×，内存子系统利用率（49.80%）远高于 AQLM（6%–20%），说明 DRAM 访问更结构化、更高效。

## 亮点

- **核心创新**：用 Psumbook（预计算内积）替代 codebook 缓存，同时降低计算复杂度（$v/m$ 倍）和空间复杂度（$v^2/t_w$ 倍），是对 codebook 量化推理范式的根本性改进
- **大模型扩展性极佳**：模型越大优势越明显（8B 1.83× → 70B 8.93×），因为大模型的 codebook 更容易超出 shared memory 限制
- **统一 kernel 支持超参数探索**：单一 kernel 覆盖多种 $(m, v, b, g)$ 配置，首次系统性揭示 codebook 量化的 latency–memory–accuracy 三维权衡
- **工程完成度高**：提供能效、硬件利用率等系统级评估，不仅是算法创新也是工程贡献

## 局限性

- 实验仅在 NVIDIA A100 上测评，未验证对其他 GPU 架构（如 H100、consumer GPU）的适配性
- 精度评估主要依赖 perplexity 和 zero/few-shot 任务，缺少长文本生成等实际应用场景的评测
- Psumbook 构建开销在 batch size 较大时可能增加（$C_{build}$ 与 $N$ 线性相关），论文对 large-batch 场景分析不足
- 量化方法本身沿用 AQLM，未在量化算法层面创新；PV-Tuning 为外部方法
- 未评估与 KV cache 量化、speculative decoding 等推理优化技术的兼容性

## 相关工作

- **Codebook 量化**：AQLM（加性多 codebook）、GPTVQ（GPTQ+codebook）、QuIP#/QTIP（旋转平滑+格 codebook/trellis 编码）、VPTQ（向量量化）
- **Uniform 量化 kernel**：GPTQ/AWQ 的 INT3/INT4 kernel、LUT-GEMM（lookup table 加速 BCQ 格式）
- **Look-Up Table 计算**：LUT-GEMM、FigLUT 等在硬件层面利用 LUT 加速；CodeGEMM 可视为 codebook 域的 LUT-based 计算
- **关键区别**：现有 codebook kernel 均基于 dequantization 范式（load codebook → reconstruct → multiply），CodeGEMM 首次提出 codebook-centric 范式（precompute → lookup → accumulate）

## 评分

- 新颖性: ⭐⭐⭐⭐ — Psumbook 的思路简洁优雅，对 codebook 量化推理范式是本质性改进
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 8B/70B 模型、多种配置对比、kernel/端到端/能效多维评测，但缺 large-batch 和多 GPU 架构实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、图示直观、复杂度分析完整，整体写作质量高
- 价值: ⭐⭐⭐⭐ — 对 codebook 量化部署有直接实用价值，70B 模型 8.93× 加速令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](../../ICML2025/model_compression/toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](../../ACL2025/model_compression/one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)

</div>

<!-- RELATED:END -->
