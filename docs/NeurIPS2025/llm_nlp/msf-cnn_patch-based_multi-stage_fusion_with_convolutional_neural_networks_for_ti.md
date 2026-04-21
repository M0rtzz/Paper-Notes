---
title: >-
  [论文解读] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML
description: >-
  [NeurIPS 2025][LLM/NLP][TinyML] 提出 msf-CNN，一种基于有向无环图（DAG）最短路径算法的多阶段 patch-based 融合优化技术，通过高效搜索 CNN 的最优融合配置，在各种微控制器（ARM Cortex-M、RISC-V、ESP32）上实现比现有方法（MCUNetV2、StreamNet）减少 50%–87% 的峰值 RAM 使用，同时保持可控的计算开销。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - TinyML
  - CNN fusion
  - microcontroller
  - patch-based inference
  - memory optimization
  - DAG shortest path
  - IoT
---

# msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML

**会议**: NeurIPS 2025  
**arXiv**: [2505.11483](https://arxiv.org/abs/2505.11483)  
**代码**: [GitHub](https://github.com/TinyPART/msf-CNN)  
**领域**: model_compression  
**关键词**: TinyML, CNN fusion, microcontroller, patch-based inference, memory optimization, DAG shortest path, IoT

## 一句话总结

提出 msf-CNN，一种基于有向无环图（DAG）最短路径算法的多阶段 patch-based 融合优化技术，通过高效搜索 CNN 的最优融合配置，在各种微控制器（ARM Cortex-M、RISC-V、ESP32）上实现比现有方法（MCUNetV2、StreamNet）减少 50%–87% 的峰值 RAM 使用，同时保持可控的计算开销。

## 研究背景与动机

AI of Things（AIoT）推动将深度神经网络部署到微控制器（MCU）上，但 MCU 的资源极为受限：

**内存瓶颈巨大**：典型 IoT 设备 RAM < 50 KiB，Flash < 250 KiB（RFC7228），而量化 ResNet-34 单个卷积层就需约 414.72 KiB RAM

**Patch-based 融合的潜力**：融合技术可节省高达 95% 的 RAM，并将输入大小与内存使用解耦。但现有方案存在以下问题：
   - 融合块内部的重计算开销高
   - 输入尺寸限制影响医学影像、音频等应用
   - 实现高度绑定特定硬件（如 ARM Cortex-M7）和特定模型（如 MobileNetV2 的 inverted block）

**搜索空间未充分探索**：MCUNetV2 仅启发式地融合头部层，StreamNet 用暴力搜索但搜索空间有限，均未探索多融合块的潜力

## 方法详解

### 整体框架

msf-CNN 的核心思路：将 CNN 建模为数据节点 DAG，将融合优化转化为图上的最短路径问题，使用经典图算法求解最优融合配置。

流程：CNN → DAG 建模（节点=张量，边=算子/融合块）→ 编码 RAM 和 MAC 到边权重 → 图算法搜索最优路径 → 代码生成部署。

### 关键设计

**DAG 表示**：将 CNN 建模为 $G = (V, E)$，其中：
- 节点 $v_0, \ldots, v_n$ 表示连续层的输入/输出张量
- 边 $e = v_n \to v_{n+1}$ 为单层算子；$e = v_n \to v_{n+m}$（$m > 1$）为融合块
- 从 $v_0$ 到 $v_n$ 的完整计算路径 $S$ 对应一个融合配置

**RAM 编码**：每条边的 RAM 使用量：

$$P_{e_i} = I + O + Buf$$

其中 $I, O$ 为输入/输出张量大小，$Buf$ 为融合缓冲区大小（非融合层 $Buf = 0$）。完整路径的峰值 RAM：

$$P_S = \max_{j=1\ldots n} P_{e_{i_j}}$$

**计算开销编码**：路径总计算成本为 MAC 之和：

$$C_S = \sum_{j=1}^{n} C_{e_{i_j}}$$

计算开销因子 $F = C_S / C_{vanilla}$。

**双优化问题**：
- **P1**：最小化峰值 RAM，约束 $F < F_{max}$
- **P2**：最小化计算成本，约束 $P < P_{max}$

**P1 的图搜索策略**：迭代消除子图中 RAM 最大的边，在每个子图上用最短路径求解，构建候选方案集 $\{S_0, S_1, \ldots\}$，筛选满足约束的最优方案。复杂度从 $O(2^{V-2})$ 降至 $O(V^3)$。

**P2 的剪枝**：直接删除 RAM 超限的边，在剩余图上求最短路径。

**迭代运算优化**：
- **全局池化迭代化**：逐元素处理输入，$7\times7$ 池化 RAM 压缩到原来的 2%，无额外计算开销
- **全连接层迭代化**：将输入向量逐元素与权重列相乘并累加，$1024 \to 256$ 层 RAM 压缩到 20%

### 损失函数 / 训练策略

msf-CNN 是编译优化工具，不涉及模型训练。基于 microTVM v0.16.0 实现，将模型转换为中间表示后重写计算图，生成 C 代码通过 RIOT OS 部署到各种 MCU 板。

## 实验关键数据

### 主实验

**最小峰值 RAM 使用（kB）**：

| 方法 | MBV2-w0.35 | MN2-vww5 | MN2-320K |
|------|-----------|----------|----------|
| Vanilla（未融合） | 194.44 | 96 | 309.76 |
| MCUNetV2 | 63 | 45 | 215 |
| StreamNet | 66 | 44 | 208 |
| **msf-CNN** | **8.56** | **15.37** | **51.16** |

RAM 压缩比：msf-CNN 相比 Vanilla 减少 87%–96%，相比 MCUNetV2/StreamNet 减少 65%–87%。

**推理延迟（ms）在 msf-CNN 最小 RAM 配置下**：

| MCU | MBV2-w0.35 | MN2-vww5 | MN2-320K |
|-----|-----------|----------|----------|
| STM32F767 (Cortex-M7 216MHz) | 1996.8 (2.5×) | 1723.0 (3.4×) | 19329.9 (4.4×) |
| ESP32-S3 (Xtensa 240MHz) | 6748.2 | 5974.1 | 76763.6 |
| ESP32-C3 (RISC-V 160MHz) | 6792.7 | 6248.9 | 73713.8 |
| SiFive (RISC-V 320MHz) | 10000.0 | OOM | OOM |

### 消融实验

**不同约束下的优化解析结果（MBV2-w0.35）**：

| 约束类型 | 约束值 | 峰值 RAM (kB) | 计算开销 F |
|---------|--------|-------------|-----------|
| P1: $F_{max}$ | 1.1 | 67.91 | 1.1 |
| P1: $F_{max}$ | 1.3 | 21.29 | 1.3 |
| P1: $F_{max}$ | 1.5 | 同上 | 同上 |
| P1: $F_{max}$ | ∞ | 7.89 | 1.68 |
| P2: $P_{max}$ | 16 kB | 15.34 | 1.38 |
| P2: $P_{max}$ | 32 kB | 25.67 | 1.25 |
| P2: $P_{max}$ | 128 kB | 83.07 | 1.02 |
| P2: $P_{max}$ | 256 kB | 181.44 | 1.0 |

启发式方法（MCUNetV2 仅融合头部层）RAM 32.08 kB / F=1.59，而 msf-CNN 在相似 RAM 下可找到更低开销的方案。

### 关键发现

1. msf-CNN 甚至能将 MBV2-w0.35 部署到仅 16 kB RAM 的 SiFive 板上
2. MCU 架构对延迟影响显著：ESP32-S3（240MHz Xtensa）在大模型上反而比 ESP32-C3（160MHz RISC-V）更慢
3. 在合理约束（$F_{max} = 1.3$）下，msf-CNN 可在极小 RAM 开销增加下实现显著内存节省
4. 添加 CMSIS 后端后，msf-CNN 的 Pareto 前沿接近 StreamNet（后者针对 ARM 做了硬件特化优化）
5. 迭代全局池化和全连接层无额外计算开销即可进一步压缩 RAM

## 亮点与洞察

1. **图论建模的优雅性**：将融合搜索问题转化为经典的最短路径/minimax 路径问题，利用成熟图算法高效求解
2. **多融合块探索**：首次系统性地探索 CNN 中多个融合块的组合，而非仅融合头部层
3. **硬件通用性**：开源实现支持 ARM Cortex-M、RISC-V、ESP32 三种主流 MCU 架构
4. **搜索空间剪枝**：从指数级 $O(2^{V-2})$ 降至多项式 $O(V^3)$，使大型网络的优化在 PC 上数秒内完成
5. **迭代算子设计**：全局池化和全连接层的迭代化实现简单有效，可无缝接入融合块

## 局限与展望

1. **仅支持 CNN**：当前仅处理卷积神经网络，不支持注意力机制和 RNN
2. **缓存策略单一**：仅实现 H-Cache 方案，未集成 2D Cache（StreamNet 使用）等更灵活的缓存策略
3. **迭代粒度固定**：每次迭代仅产生一个输出元素，这一参数对内存和计算影响大但未被优化
4. **延迟高于 StreamNet**：缺乏硬件特化优化（CMSIS 指令集），纯通用实现延迟偏高
5. **非 MCU 平台未验证**：虽然理论上支持 CPU/GPU/FPGA，但实验仅限 MCU

## 相关工作与启发

- **MCUNetV2**：启发式融合头部层策略简单但次优，msf-CNN 证明系统搜索能找到更好方案
- **StreamNet**：2D 缓存策略有效减少重计算，msf-CNN 未来计划集成
- **TinyNAS / Once-for-All**：NAS 方法需要重新训练，而 msf-CNN 不需修改模型本身
- **启发**：DAG + 最短路径的建模思路可扩展到 attention 层、混合精度等其他优化维度

## 评分

- ⭐⭐⭐⭐ **创新性**：图论建模 + 多阶段融合搜索的思路新颖，将组合优化问题优雅地转化为经典图算法
- ⭐⭐⭐⭐ **实验充分性**：在 6 种硬件、3 个模型、多种约束条件下验证，覆盖全面
- ⭐⭐⭐⭐ **实用价值**：开源工具、多平台支持，对嵌入式 AI 工程师有直接帮助
- ⭐⭐⭐ **局限性**：仅限 CNN、缓存策略单一、未与 NAS 方法系统对比

**总评**: ⭐⭐⭐⭐ (3.5/5) — 扎实的系统优化工作，图论建模思路优雅，多平台验证令人信服。主要不足是仅限 CNN 架构和缓存策略的局限性。对 TinyML 领域有实际工程价值。

<!-- RELATED:START -->

## 相关论文

- [STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](../../CVPR2025/llm_nlp/staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [Opinion Maximization in Social Networks by Modifying Internal Opinions](opinion_maximization_in_social_networks_by_modifying_internal_opinions.md)
- [Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)

<!-- RELATED:END -->
