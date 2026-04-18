---
title: >-
  [论文解读] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
description: >-
  [CVPR 2026][模型压缩][FPGA] 本文首次为 JPEG XS 标准中的 Intra Pattern Copy (IPC) 工具设计了 FPGA 硬件加速架构，通过四级流水线 DV 比较引擎和按 IPC Group 对齐的存储组织，在 Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277mW 功耗。
tags:
  - CVPR 2026
  - 模型压缩
  - FPGA
  - JPEG XS
  - Intra Pattern Copy
  - 位移矢量搜索
  - 流水线架构
---

# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**会议**: CVPR 2026  
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)  
**代码**: 无  
**领域**: 模型压缩 / 硬件实现  
**关键词**: FPGA, JPEG XS, Intra Pattern Copy, 位移矢量搜索, 流水线架构

## 一句话总结

本文首次为 JPEG XS 标准中的 Intra Pattern Copy (IPC) 工具设计了 FPGA 硬件加速架构，通过四级流水线 DV 比较引擎和按 IPC Group 对齐的存储组织，在 Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277mW 功耗。

## 研究背景与动机

**领域现状**：JPEG XS 是面向低延迟、低复杂度的图像压缩标准，用于远程桌面、KVM 等场景。IPC（Intra Pattern Copy）是其中的编码工具，在小波域进行帧内预测以减少屏幕内容的空间冗余。

**现有痛点**：IPC 中的位移矢量（DV）搜索模块需要遍历所有候选偏移并选择最优预测参考，计算密集度极高。虽然 H.264/HEVC 已有大量运动估计硬件实现，但这些设计在像素域操作，不适用于 JPEG XS 的小波域分组预测流程。

**核心矛盾**：DV 搜索的高计算复杂度与 JPEG XS 低延迟要求之间的矛盾。小波系数按 IPC Group/Unit 分组存储，访问模式高度离散，传统存储方式导致控制复杂度高和带宽利用率低。

**本文目标** 设计 DV 搜索的 FPGA 实现，使 IPC 能高效部署在硬件上。

**切入角度**：针对 IPC 特有的分组预测流程设计四级流水线和优化存储。

**核心 idea**：通过按 IPC Group 对齐的存储组织和四级流水线 DV 比较架构，实现 JPEG XS IPC 的高效硬件部署。

## 方法详解

### 整体框架

系统由两个核心引擎组成：残差计算引擎（从 DRAM 读取原始和重建的 IPC Unit，计算残差）和 DV 比较引擎（评估每个残差的比特代价并搜索最优 DV）。输入是 RCT + DWT 后的小波系数，输出是最优 DV。

### 关键设计

1. **四级流水线 DV 比较架构**:

    - 功能：将 DV 比较过程分解为四个流水线阶段并行执行
    - 核心思路：Stage 0 加载残差系数和计算配置参数（BandIdx, GrpSize, UnitWidth）；Stage 1 的 GetOrMask 模块计算组内按位 OR 掩码；Stage 2 的 CalGCLI 模块计算残差比特代价 BitsTest；Stage 3 的 Compare 模块比较并更新最优 DV (BestDV)
    - 设计动机：DV 比较是整个 DV 搜索的瓶颈，流水线化让连续 DV 候选的评估可以重叠执行，显著提升吞吐量

2. **按 IPC Group 对齐的存储组织 (Method 1)**:

    - 功能：重新组织 DRAM 上小波系数的存储布局，从按 Precinct 存储改为按 IPC Group/Unit 存储
    - 核心思路：将同一 Group 的 IPC Unit 顺序存储，每个 Unit 包含所有子带块。只需一个基地址 + 固定偏移即可加载整个 IPC Unit，支持突发访问模式。配合片上 TLB RAM 存储不同 Group 的块大小
    - 设计动机：原始按 Precinct 存储时，不同子带的系数分散在内存中，需逐个定位，控制复杂度高且带宽利用率低

3. **残差计算引擎**:

    - 功能：从 DRAM 读取原始和重建系数块，计算分组残差
    - 核心思路：通过 CMD 模块进行地址映射，数据流入 Q0-Q3（原始）和 C0-C3（重建）四组 FIFO。CTRL 模块管理同步读写，SIG_MAG_SUB 模块对符号-幅值格式的 32 位系数做四路并行减法计算残差，结果存入 R0-R3 残差 FIFO
    - 设计动机：残差计算需要同步访问原始和重建数据的相同 Group，FIFO 阵列 + CTRL 同步机制确保数据对齐

### 损失函数 / 训练策略

不适用（硬件设计，非学习方法）。设计目标是保持与 IPC 参考软件一致的率失真性能。

## 实验关键数据

### 主实验

| 参数 | Method 0 (Baseline) | Method 1 (本文) | 提升 |
|------|---------------------|-----------------|------|
| 吞吐量 (Mpixels/s) | 35.98 | 38.30 | +6.4% |
| 功耗 (mW) | 276 | 277 | ≈持平 |
| 功耗效率 (Mpixels/s/W) | 130.36 | 138.27 | +6.1% |
| LUTs (K) | 13.93 | 12.89 | -7.5% |
| FFs (K) | 23.80 | 21.79 | -8.4% |
| DSPs | 17 | 17 | 持平 |
| BRAM | 11 | 15 | +4 |

### 消融实验（模块资源占用）

| 模块 | LUTs (K) | FFs (K) | DSPs | BRAM |
|------|---------|---------|------|------|
| 残差计算引擎 | 0.48 | 0.47 | 0 | 15 |
| DV比较-GCLI_CAL | 11.63 | 19.98 | 17 | 0 |
| DV比较-DV_UPDATE | 0.73 | 1.41 | 0 | 0 |

### 关键发现
- DV 比较引擎中的 GCLI_CAL 模块占据绝大多数资源（90%+ LUT 和 FF），是面积瓶颈
- Method 1 存储方式在提升吞吐量的同时减少了 LUT/FF 用量（分别降 7.5%/8.4%），代价是增加 4 个 BRAM 用于 TLB
- 系统延迟 73.01ms，对屏幕内容编码场景可接受

## 亮点与洞察
- **首个 JPEG XS IPC 硬件实现**：填补了 JPEG XS 硬件加速的空白，为 ASIC 部署提供参考
- **存储组织优化思路通用**：按计算模式而非数据原始布局组织存储的思想可迁移到其他小波/变换域编码工具

## 局限与展望
- 仅实现了 DV 搜索模块，完整 IPC 框架（模式选择、补偿等）尚未在硬件上验证
- 目标平台 Artix-7 (XC7A35T) 较小，实际 ASIC 部署可能有不同权衡
- 未与其他图像编码标准（如 HEVC SCC）的硬件实现做性能对比
- 论文缺乏率失真性能与软件参考的详细偏差分析

## 相关工作与启发
- **vs H.264/HEVC 运动估计硬件**: 那些在像素域操作，使用 SAD/SATD 代价评估和固定块划分；本文在小波域操作，使用 GCLI 比特代价评估和分组预测
- **vs JPEG XS TDC**: TDC 做帧间时域预测，IPC 做帧内空域预测，两者互补

## 评分
- 新颖性: ⭐⭐⭐ 首个 IPC FPGA 实现但方法偏工程
- 实验充分度: ⭐⭐⭐ 资源和性能数据充分但缺少率失真对比
- 写作质量: ⭐⭐⭐ 结构清晰但部分细节不够
- 价值: ⭐⭐⭐ 对 JPEG XS 硬件化有直接实用价值
