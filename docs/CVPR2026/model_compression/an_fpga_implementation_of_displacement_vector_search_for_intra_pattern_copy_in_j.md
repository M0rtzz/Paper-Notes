---
title: >-
  [论文解读] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
description: >-
  [CVPR 2026][模型压缩][FPGA] 针对 JPEG XS 屏幕内容编码中 Intra Pattern Copy（IPC）模块的位移向量（DV）搜索计算瓶颈，首次提出四级流水线 FPGA 架构并设计基于 IPC Group 对齐的内存组织方式，在 Xilinx Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277 mW 功耗，为 IPC 的实际硬件部署提供了可行方案。
tags:
  - CVPR 2026
  - 模型压缩
  - FPGA
  - JPEG XS
  - Intra Pattern Copy
  - 位移向量搜索
  - 硬件加速
---

# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**会议**: CVPR 2026  
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: FPGA, JPEG XS, Intra Pattern Copy, 位移向量搜索, 硬件加速

## 一句话总结

针对 JPEG XS 屏幕内容编码中 Intra Pattern Copy（IPC）模块的位移向量（DV）搜索计算瓶颈，首次提出四级流水线 FPGA 架构并设计基于 IPC Group 对齐的内存组织方式，在 Xilinx Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277 mW 功耗，为 IPC 的实际硬件部署提供了可行方案。

## 研究背景与动机

JPEG XS 是 JPEG 委员会为远程桌面、KVM 和沉浸式视频等场景设计的低延迟、低复杂度图像压缩标准。为了提升其对屏幕内容的编码效率，研究者提出了 Intra Pattern Copy（IPC）技术，在小波域内进行帧内预测以消除空间冗余，取得了显著的 BD-PSNR 提升。

然而，IPC 流程中的 **位移向量搜索（DV Search）** 是计算最密集的模块：它需要遍历所有候选预测偏移量，计算残差并选择使编码代价最小的最优 DV。这一过程的高计算复杂度和不规则内存访问模式成为实时硬件部署的关键瓶颈。

已有的 H.264/HEVC 运动估计 FPGA 实现虽然成熟，但它们针对的是像素域的固定块分区，而非 JPEG XS 小波域中按 IPC Group 和 IPC Unit 组织的频域预测流。因此，**需要一种专门针对 JPEG XS IPC 框架的 FPGA 架构设计**。

核心 idea：设计四级流水线架构将残差计算与 DV 比较解耦并行化，同时通过按 IPC Group 对齐的内存组织方式消除散乱的小波系数访问开销。

## 方法详解

### 整体框架

系统由两个主引擎组成：**残差计算引擎**（Residual Calculation Engine）和 **DV 比较引擎**（DV Comparison Engine）。输入为经过 RCT 和 DWT 后的原始/重建小波系数，存储在 DRAM 的两个 IPC Unit 存储库中。残差计算引擎从内存中读取 IPC Unit，计算块级残差；DV 比较引擎评估每个残差的编码比特代价，搜索最优 DV。

### 关键设计

1. **残差计算引擎（Residual Calculation Engine）**:

    - 功能：从 DRAM 读取原始/重建系数块，计算有符号残差
    - 核心思路：通过 CMD 模块将 precinct 编号映射为内存地址，用 FIFO 阵列（Q0-Q3 存原始数据，C0-C3 存重建数据）缓存系数，CTRL 模块协调读写同步。SIG_MAG_SUB 模块将 32-bit 符号-幅度值拆分并通过 4 路并行减法路径计算残差
    - 设计动机：IPC 的系数来自不同子带且需按 Group 组织，FIFO 阵列设计使得同一 IPC Group 的数据能被顺序喂入，避免乱序内存访问

2. **四级流水线 DV 比较引擎（4-Stage Pipeline DV Comparison）**:

    - 功能：评估每个候选 DV 的编码代价，选择最优 DV
    - 核心思路：Stage 0 加载残差系数和 Group 参数（BandIdx, GrpSize, UnitWidth）；Stage 1 GetOrMask 模块对 Group 内残差做 bitwise OR，生成 OrIdx 和 OrAll；Stage 2 CalGCLI 模块根据 OR 结果计算 GCLI 编码代价 BitsTest；Stage 3 Compare 模块比较当前 BitsTest 与历史最小代价 BitsBest，通过 MUX 选择最优 DV
    - 设计动机：将比较过程分为 4 级流水线实现了残差计算和 DV 比较的并行化，在延迟和吞吐量之间取得平衡

3. **基于 IPC Group 对齐的内存组织（Method 1）**:

    - 功能：重新组织 DRAM 中小波系数的存储方式
    - 核心思路：不同于 Method 0 按 precinct 线性存储（导致同一 IPC Unit 的系数分散在不同位置），Method 1 按 IPC Group 和 IPC Unit 组织存储，同一 Group 中的 Unit 顺序排列，每个 Unit 包含所有子带块。这样只需一个基地址加固定偏移即可加载整个 IPC Unit，支持 burst 读取
    - 设计动机：IPC 的访问模式是以 Group 为单位遍历所有 Unit，Method 0 需要按 group/unit/band 三级索引定位，控制复杂且吞吐量低；Method 1 天然适配这一访问模式

4. **片上 TLB RAM**:

    - 功能：存储不同 IPC Group 中系数块的变长长度信息
    - 核心思路：由于不同 Group 的块大小不同（取决于小波分解级数），CMD 模块利用片上 TLB 查表来生成正确的 entry 地址，在 DV 搜索切换到下一个 precinct 时更新 TLB
    - 设计动机：避免运行时动态计算变长块地址的开销

### 损失函数 / 训练策略

本文为硬件设计工作，无训练过程。优化目标是保持与 IPC 参考软件一致的率失真性能，同时最小化延迟、资源占用和功耗。

## 实验关键数据

### 主实验

| 参数 | Method 0 (Baseline) | Method 1 (Proposed) |
|------|---------------------|---------------------|
| 平台 | Xilinx Artix-7, 100 MHz | 同左 |
| 吞吐量 (Mpixels/s) | 35.98 | **38.30** |
| 功耗 (mW) | 276 | 277 |
| 功耗效率 (Mpixels/s/W) | 130.36 | **138.27** |
| LUTs (K) | 13.93 | **12.89** |
| FFs (K) | 23.80 | **21.79** |
| DSPs | 17 | 17 |
| BRAM | 11 | 15 |

### 消融实验：模块资源占用

| 模块 | LUTs (K) | FFs (K) | DSPs | BRAM |
|------|----------|---------|------|------|
| 残差计算引擎 | 0.48 | 0.47 | 0 | 15 |
| GCLI_CAL（DV比较） | 11.63 | 19.98 | 17 | 0 |
| DV_UPDATE（DV比较） | 0.73 | 1.41 | 0 | 0 |

### 关键发现

- Method 1 相比 Method 0 吞吐量提升 6.4%，功耗效率提升 6.1%
- LUT 和 FF 资源分别减少 7.5% 和 8.4%，仅增加 4 个 BRAM
- DV 比较引擎中的 GCLI_CAL 模块贡献了绝大部分逻辑资源消耗（约 90% LUT），是优化重点
- 延迟为 73.01 ms，率失真性能与 IPC 参考软件一致

## 亮点与洞察

- **首次将 IPC DV 搜索搬上 FPGA**：填补了 JPEG XS IPC 硬件实现的空白
- **内存组织与访问模式协同设计**：Method 1 的核心洞察是让存储布局与计算的遍历顺序一致
- **四级流水线粒度恰当**：没有过度拆分流水线级数，在面积和吞吐之间找到平衡

## 局限与展望

- 仅在较小规模的 Artix-7 器件上验证，未在更高端 FPGA 或 ASIC 上评估
- 吞吐量 38.3 Mpixels/s 对于 4K 实时（约 500 Mpixels/s）仍有较大差距
- 未与完整 JPEG XS IPC 编码器集成测试，系统级瓶颈尚不明确
- 仅支持单一小波分解配置（5 水平 2 垂直），灵活性有限

## 相关工作与启发

- H.264/HEVC 运动估计 FPGA 实现提供了成熟的流水线和存储优化范式，但频域预测流需要全新存储组织
- JPEG XS 的 TDC（时域差分编码）与 IPC 互补，未来可能需要将两者硬件实现整合
- Group-aligned 内存组织思路可推广到其他需要按特定维度遍历的小波域处理任务

## 评分

- 新颖性: ⭐⭐⭐ 架构设计思路较为常规（流水线+内存优化），但在 JPEG XS IPC 领域是首创
- 实验充分度: ⭐⭐⭐ 仅一个 FPGA 平台，缺乏与同类硬件编码器的对比
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰，内存组织对比直观
- 价值: ⭐⭐⭐ 对 JPEG XS 硬件化有实际推动作用，但论文影响范围较窄

<!-- RELATED:START -->

## 相关论文

- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [Distilling Balanced Knowledge from a Biased Teacher](distilling_balanced_knowledge_from_a_biased_teacher.md)
- [Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)

<!-- RELATED:END -->
