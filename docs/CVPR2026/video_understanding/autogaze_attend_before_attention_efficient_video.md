---
title: >-
  [论文解读] Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing
description: >-
  [CVPR 2026][视频理解][AutoGaze] 提出 AutoGaze，一个仅 3M 参数的轻量模块，在 ViT 之前自回归选择最少的多尺度 patch 集合以重建视频，去除 4×-100× 时空冗余，实现 ViT 最高 19× / MLLM 最高 10× 加速…
tags:
  - "CVPR 2026"
  - "视频理解"
  - "AutoGaze"
  - "自回归注视"
  - "多尺度patch选择"
  - "token压缩"
  - "长视频高分辨率"
---

# AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing

**会议**: CVPR 2026  
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**代码**: [autogaze.github.io](https://autogaze.github.io/)  
**领域**: 视频理解 / 高效推理  
**关键词**: 自回归注视, token压缩, 多尺度patch选择, 视频MLLM, 高分辨率长视频

## 一句话总结

提出 AutoGaze，一个仅 3M 参数的轻量模块，通过自回归地选择最小化重建损失的多尺度 patch 集合，在 ViT 之前移除视频中的冗余信息，实现 4×~100× 的 token 压缩和最高 19× 的 ViT 加速，使 MLLM 能够扩展至 1K 帧 4K 分辨率视频并在 VideoMME 上达到 67.0%。

## 研究背景与动机

**领域现状**：多模态大语言模型 (MLLM) 如 Qwen2.5-VL、NVILA 等已实现通用视频理解，但受限于计算成本，通常只能处理短时、低分辨率视频。人类视觉系统通过眼动注视 (saccade) 选择性关注信息丰富的区域，高效理解高 FPS 高分辨率视频流。

**现有痛点**：(1) 现有 MLLM 视频中每个像素和每帧的处理权重相同，忽视了视频中巨大的时空冗余（静态背景只需观看一次）；(2) 现有 token 压缩方法仅在 LLM 内部或 ViT-LLM 之间剪枝，ViT 仍需处理所有像素，成为扩展瓶颈；(3) 启发式方法（注意力分数）效果不如学习方法，而基于搜索推理的方法引入额外开销。

**核心矛盾**：长时高分辨率视频理解的能力需求与 ViT 处理所有像素的计算瓶颈之间的矛盾。

**本文目标**：在 ViT 之前高效地移除视频冗余 patch，同时保留足够信息用于下游理解。

**切入角度**：将 patch 选择建模为自回归序列生成问题，通过最小化重建损失来决定选择哪些 patch 以及何时停止。

**核心 idea**：让一个轻量模型学会"先看哪里"，模仿人类眼动注视机制，在 ViT 处理之前去除冗余。

## 方法详解

### 整体框架

AutoGaze 在视频处理流程的最前端工作：(1) 对视频的每一帧，卷积编码器提取帧级特征；(2) Transformer 解码器自回归地输出 patch 索引序列，同时预测当前重建损失；(3) 当预测重建损失低于用户指定阈值 $\epsilon$ 时自动停止当前帧的注视；(4) 仅将选中的多尺度 patch 送入下游 ViT 和 MLLM。

### 关键设计

1. **自回归注视 (Autoregressive Gazing)**:
    - 功能：逐帧逐 patch 地自回归选择信息最丰富的 patch，同时利用历史帧和已选 patch 避免重复选择
    - 核心思路：将 patch 选择形式化为序列到序列问题，词汇表为 patch 索引 $\{1, \dots, V\}$，解码器基于当前及历史帧特征和已选 patch 输出下一个待选 patch 的概率分布
    - 优化目标：$\min_{p_1^1, \dots, p_{N^T}^T} L(\mathbf{X}^{1:T}, \text{Recon}(\mathbf{X}^1[p_1^1], \dots, \mathbf{X}^T[p_{N^T}^T]))$
    - 自动停止：解码器额外预测每步的重建损失，当预测损失 $< \epsilon$ 时停止当前帧注视
    - 设计动机：自回归方式可利用前序帧的注视历史避免冗余选择（如静态背景已在前帧选过），这是启发式方法无法做到的

2. **多尺度注视 (Multi-Scale Gazing)**:
    - 功能：为不同详细程度的区域分配不同分辨率的 patch，低细节区域用粗粒度 patch 覆盖
    - 核心思路：解码器词汇表包含多个尺度的 patch，让模型自适应选择每个区域的最优尺度
    - 验证：实验证实 AutoGaze 在细粒度区域（高 Laplacian 方差）倾向选择更细尺度，在低细节区域使用粗尺度（Spearman $\rho = 0.12, p < 0.001$）
    - 设计动机：纯色区域可用低分辨率无损表示，无需浪费全分辨率 patch

3. **两阶段训练：NTP 预训练 + RL 后训练**:
    - 功能：先通过监督学习建立基础注视能力，再通过 RL 突破贪心搜索的次优性
    - NTP 预训练：在 250K 视频上通过贪心搜索收集近似最优注视序列，使用交叉熵损失 $L_{NTP} = -\sum_t \sum_k \log \pi_\theta(\tilde{p}_k^t | \mathbf{X}^{1:t}, \tilde{p}_{1:k-1}^t)$ 训练
    - RL 后训练：采用简化的 GRPO 算法，以重建损失的负值作为奖励，优势为 $G_k^t = \sum_{\tau=t}^T \gamma^{N^t-k+\sum_{s=t+1}^\tau N^s} \cdot (-l_{N^\tau}^\tau)$
    - 多 token 预测：采用多头同时输出多个 patch 索引，10 token 预测加速约 5× 注视过程，性能损失微小
    - 设计动机：贪心搜索数据只是次优解，RL 可发现更优的注视序列

4. **任意分辨率/时长推理**:
    - 功能：训练在 16 帧 224×224 视频上，推理时扩展至任意分辨率和时长
    - 核心思路：将视频切分为 $16 \times 224 \times 224$ 时空 tile，分别运行 AutoGaze 后合并注视位置
    - ViT 适配：对不同尺度的帧和位置编码分别插值，各尺度分别做 patch embedding 后拼接送入 ViT

### 损失函数 / 训练策略

- 重建模型：定制的 VideoMAE + block-causal attention
- 重建损失：像素重建损失 + 感知损失的加权和
- 训练数据：800K 视频（包括第一/第三人称、自然场景、文本丰富视频），其中 250K 配有贪心注视序列
- 视频采样：16 帧，224 分辨率
- 最优重建损失阈值：0.7（下游 MLLM 性能下降 < 0.5%）

## 实验关键数据

### 主实验 (与 SOTA MLLM 对比)

| 模型 | Max #F | Max Res | VideoMME (w/o) | VideoMME (w/) | MVBench | HLVid |
|------|--------|---------|---------------|--------------|---------|-------|
| GPT-4o | - | - | 71.9 | 77.2 | 64.6 | 49.3 |
| Qwen2.5-VL-7B | 48 | 896 | 65.1 | 71.6 | 69.6 | 48.1 |
| VideoChat-Flash | 10000 | 448 | 65.3 | 69.7 | 74.0 | 46.6 |
| NVILA-8B-Video | 256 | 448 | 64.2 | 70.0 | 68.1 | 42.5 |
| **+ AutoGaze** | **1024** | **3584** | **67.0** | **71.8** | 69.7 | **52.6** |
| 增益 (vs NVILA) | ×4 | ×8 | +2.8 | +1.8 | +1.6 | **+10.1** |

### 消融实验

| 训练策略 | 重建损失 | 注视比例 |
|---------|---------|---------|
| 无训练 | 0.7 | 0.263 |
| 仅 NTP 预训练 | 0.7 | 0.102 |
| 仅 RL 后训练 | 0.7 | 0.209 |
| **NTP + RL** | 0.7 | **0.094** |

| 多 token 预测数 | 多尺度 | 注视比例 | 推理延迟 |
|---------------|--------|---------|---------|
| 1 | ✓ | 0.074 | 0.949s |
| 5 | ✓ | 0.078 | 0.246s |
| 10 | ✓ | 0.094 | 0.193s |
| 10 | ✗ | 0.220 | 0.467s |

### 关键发现

- 30FPS 4K 分辨率视频仅需约 1% 的 patch 即可达到 0.7 重建损失阈值
- ViT 加速最高 19×，MLLM 加速最高 10×，使得 4K 分辨率视频处理成为可能
- 在 HLVid 高分辨率长视频基准上提升 10.1%（42.5→52.6），超越 GPT-4o（49.3%）
- AutoGaze 的注视行为具有良好的 OOD 泛化性：对未见语义（CCTV、机器人、物体替换视频）和风格迁移后的视频均保持一致的注视模式
- 对比现有 token 压缩方法：AutoGaze 是唯一同时改善 ViT 和 LLM 延迟的方法（ViT: 2.20s→0.55s）

## 亮点与洞察

- **从根本上解决瓶颈**：首次在 ViT 之前进行 token 压缩，打破了所有先前方法仅在 ViT 后压缩的格局
- **仿生设计**：自回归注视机制直接模仿人类视觉系统的扫视行为，利用时序信息避免重复注视
- **极致轻量**：3M 参数即实现强大的注视能力，注视本身的计算开销极低
- **首个高分辨率长视频基准 HLVid**：268 个问题，要求 1K-2K 分辨率才能解答，填补了评估空白
- **两阶段训练范式**：NTP+RL 的组合优于单独使用任一方法，RL 突破了贪心数据的天花板

## 局限与展望

- 在部分不需要高分辨率的基准上，过度扩展分辨率反而有害，需要自适应策略
- 依赖 VideoMAE 作为重建模型，重建质量本身可能限制注视决策质量
- HLVid 基准规模有限（268 QAs），代表性有待验证
- 多尺度 patch 输入要求对 ViT 进行修改（不同尺度分别做 patch embedding），对已部署模型的兼容性需要考量

## 相关工作与启发

- **NVILA** (Lin et al., 2024)：作为基座 MLLM，AutoGaze 通过其实现了 1K 帧 4K 分辨率的突破
- **GRPO** (Shao et al.)：用于注视策略的 RL 后训练，验证了 RL 在视觉 token 选择任务上的有效性
- **ToMe / FastV / LongVU**：先前的 token 压缩方法仅在 ViT/LLM 内部工作，AutoGaze 首次将压缩前置
- **启发**：自回归序列生成 + 重建损失作为奖励这一范式可扩展到图像、3D 点云等其他模态的 token 选择

## 评分 (⭐星级)

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐⭐** |
---
description: "AutoGaze：3M参数轻量模块，在ViT之前自回归选择多尺度关键patch去除视频冗余，实现4x-100x token缩减和最高19x加速，使MLLM可扩展到1K帧4K分辨率视频"
tags:
- CVPR 2026
- 视频理解
- token压缩
- 自回归
- 高效推理
- MLLM
---

# Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing

**会议**: CVPR 2026  
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**代码**: [https://autogaze.github.io/](https://autogaze.github.io/)  
**领域**: 视频理解 / 高效推理  
**关键词**: AutoGaze, 自回归注视, 多尺度patch选择, token压缩, 长视频高分辨率

## 一句话总结

提出 AutoGaze，一个仅 3M 参数的轻量模块，在 ViT 之前自回归选择最少的多尺度 patch 集合以重建视频，去除 4×-100× 时空冗余，实现 ViT 最高 19× / MLLM 最高 10× 加速，首次使 MLLM 可扩展到 1K 帧 4K 分辨率视频，在 VideoMME 上达到 67.0%。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）如 Qwen2.5-VL、NVILA 等推动了通用视频理解，但受限于 ViT 和 LLM 对每个像素的均等处理——视频存在大量时空冗余（静态背景、低变化区域）。

**现有痛点**：(1) 现有 token 缩减方法（FastV、ToMe、VisionZip 等）仅在 ViT 内部或 ViT-LLM 之间剪枝 token，ViT 本身仍需处理所有像素，是主要效率瓶颈；(2) 基于注意力分数的启发式方法效果不如学习式方法；(3) 涉及搜索和推理的方法引入额外开销，进一步限制可扩展性；(4) 现有基准只关注长视频但不关注高分辨率。

**核心矛盾**：人类通过扫视（saccade）选择性关注信息丰富的区域高效处理视频流，而模型均等处理每个像素——如何让模型在"看"之前先决定"看哪里"？

**本文目标**：在 ViT 之前去除视频的时空冗余 patch，从根本上降低视觉编码成本，使 MLLM 可扩展到长时间高分辨率视频。

**切入角度**：模拟人类注视行为——自回归预测哪些多尺度 patch 能以最少数量重建当前帧（在给定误差阈值内）。

**核心 idea**：在注意力机制之前先"注视"——用自回归模型选出最少的 patch，ViT 只需处理这些 patch。

## 方法详解

### 整体框架

AutoGaze 由卷积编码器和自回归 Transformer 解码器组成，仅 3M 参数。处理流程：(1) 依次编码每帧；(2) 基于当前帧特征和历史帧/注视信息，自回归解码 patch 索引；(3) 在解码每个 patch 时同时预测重建损失；(4) 当预测损失低于用户指定阈值时自动停止当前帧的注视；(5) 仅将选中的多尺度 patch 送入 ViT/MLLM。

### 关键设计

1. **自回归注视（Autoregressive Gazing）**：将 patch 选择建模为序列生成问题。给定 $T$ 帧视频 $\bm{X}^{1:T}$，模型输出每帧的 patch 索引序列 $p_{1:N^1}^1, \ldots, p_{1:N^T}^T$，其中 $p_k^t \in \{1, \ldots, V\}$。目标是找到使重建损失最小化的最少 patch 子集：$\min_{p_1^1, \ldots, p_{N^T}^T} L(\bm{X}^{1:T}, \text{Recon}(\bm{X}^1[p_1^1], \ldots, \bm{X}^T[p_{N^T}^T]))$。重建器为带块因果注意力的 VideoMAE，距离函数为像素重建损失和感知损失的加权和。设计动机：自回归方式允许模型参考前序帧和已选 patch 的历史，自然避免跨帧重复选择。

2. **自动决定注视长度**：在解码器上添加额外头，在解码每个 $p_k^t$ 时预测"用已选 patch 重建当前帧"的损失。当预测损失 $< \epsilon$（用户指定阈值）时自动停止当前帧注视，实现自适应的每帧 token 数量。不同视频/帧的冗余程度不同——静态帧只需极少 patch，运动剧烈帧需更多。

3. **多尺度注视**：解码器词表包含多个尺度的 patch（如 $224 \times 224$ 的 $1/4, 1/2, 1, 2$ 倍），让模型为不同细节程度的区域选择不同分辨率：纯色区域用粗尺度低分辨率覆盖，精细纹理区域用细尺度高分辨率捕捉。下游 ViT 通过对不同尺度分别做 patch embedding + 插值位置编码来接受多尺度输入。

4. **两阶段训练**：(a) **NTP 预训练**：在 250K 视频上收集贪心搜索得到的近似最优注视序列，用 next-token-prediction 交叉熵损失 $L_{NTP} = -\sum_t \sum_k \log \pi_\theta(\tilde{p}_k^t | \bm{X}^{1:t}, \ldots)$ 训练，同时用 $\ell_2$ 损失监督重建损失预测。(b) **RL 后训练**：用简化的 GRPO 算法以重建损失作为奖励进一步优化，$L_{GRPO} = -\sum_t \sum_k \frac{\pi_\theta(p_k^t)}{\pi_{\theta_{detached}}(p_k^t)} A_k^t$，发现比预训练上界更优的注视序列。

### 损失函数 / 训练策略

- 预训练数据：800K 视频（自中心、他中心、自然、文本丰富场景），采样为 16 帧 224 分辨率
- 贪心搜索收集注视序列：逐个穷举找重建损失最低的 patch，记录每步重建损失
- RL 后训练：在策略 on-policy 生成的注视序列上用重建损失负值作奖励，优势函数为折扣未来帧重建损失
- 多 token 预测：使用多个头同时输出多个 patch 索引，加速推理
- 推理时任意分辨率/时长：将视频切为 $16 \times 224 \times 224$ 时空瓦片，每个瓦片独立运行 AutoGaze 后合并

## 实验关键数据

### 主实验

| 模型 | 最大帧数 | 最大分辨率 | VideoMME(w/o sub) | VideoMME(w/ sub) | MVBench | L-VidBench | HLVid |
|------|---------|-----------|-------------------|------------------|---------|------------|-------|
| GPT-4o | - | - | 71.9 | 77.2 | 64.6 | 66.7 | 49.3 |
| Qwen2.5-VL-7B | 48 | 896 | 65.1 | 71.6 | 69.6 | 56.0 | 48.1 |
| VideoChat-Flash | 10000 | 448 | 65.3 | 69.7 | 74.0 | 64.7 | 46.6 |
| NVILA-8B-Video | 256 | 448 | 64.2 | 70.0 | 68.1 | 57.7 | 42.5 |
| **NVILA + AutoGaze** | **1024** | **3584** | **67.0** | **71.8** | 69.7 | **61.0** | **52.6** |
| vs NVILA 基线 | ×4 | ×8 | +2.8 | +1.8 | +1.6 | +3.3 | **+10.1** |

> AutoGaze 使 NVILA-8B 在帧数和分辨率上扩展 4x/8x，VideoMME 达 67.0%。HLVid 提升 10.1%，超越 GPT-4o (+3.3%)。

### 消融实验

| 类型 | 方法 | ViT延迟 | LLM延迟 | V-MME | L-Vid |
|------|------|---------|---------|-------|-------|
| - | 无缩减 | 2.20s | 1.42s | 53.4 | 51.1 |
| S-PA | 空间池化 | 2.20s | 0.18s | 51.5 | 47.2 |
| S-PA | ToMe | 2.23s | 0.11s | 51.5 | 49.3 |
| S-PD | FastV | 2.23s | 0.38s | 53.0 | 46.3 |
| T-PA | 时间池化 | 2.20s | 0.13s | 52.2 | 50.0 |
| **AutoGaze** | 学习式 | **降低** | **降低** | **同等/更优** | **同等/更优** |

> 现有方法仅降低 LLM 延迟（ViT 不变），AutoGaze 同时降低 ViT 和 LLM 延迟。在相同 6.25% token 选择率下全面优于启发式方法。

### 关键发现

- **自适应注视行为**：AutoGaze 自动关注运动区域（光流高的 patch 被更频繁选择）、用细尺度捕捉精细纹理、用粗尺度覆盖平滑区域
- **重建阈值 0.7 最优**：通常导致下游性能下降 <0.5%
- **冗余随 FPS/分辨率增大而增大**：30-FPS 4K 视频仅需 ~1% patch
- **OOD 泛化性**：在 CCTV 监控、机器人操作、风格迁移等训练分布外视频上，AutoGaze 依然稳健追踪变化区域
- **HLVid 新基准**：首个长时间高分辨率视频 QA 基准（268 QA，5分钟 4K 视频），验证了高分辨率理解的必要性

## 亮点与洞察

- **在注意力之前注意**：概念优雅——将 patch 选择从模型内部移到模型之前，从根本上解决 ViT 的计算瓶颈，而非在 ViT 输出后裁剪
- **仅 3M 参数**：overhead 极小（相比 ViT 的数百M参数），添加 AutoGaze 的边际成本几乎可忽略
- **NTP + RL 训练范式**：先用贪心搜索收集"教科书"注视序列做预训练，再用 RL 超越教科书——与 LLM 的训练范式高度一致
- **任意分辨率/时长推理**：通过时空瓦片化，在 16帧/224分辨率训练的模型可直接推理 1K帧/4K视频
- **HLVid 基准贡献**：填补了高分辨率长视频 QA 基准的空白

## 局限与展望

- 某些基准上扩展到过长/过高分辨率反而有害——需要自适应选择最优帧数/分辨率
- 当前训练数据仅 16帧/224分辨率，更大规模训练可能进一步提升
- AutoGaze 的注视决策是 prompt-agnostic 的，不根据用户问题调整关注区域——prompt-dependent 版本可能更优
- 多 token 预测的精度-速度权衡需进一步研究
- 未探索与 Flash Attention 等硬件级优化的协同效果

## 相关工作与启发

- **VideoMAE**：作为 AutoGaze 的重建器，提供了"从少量 patch 重建完整帧"的能力基础
- **ToMe / FastV / VisionZip**：都在 ViT 内部或之后做 token 缩减，AutoGaze 将缩减前移到 ViT 之前
- **NVILA**：AutoGaze 的下游 MLLM，验证了方法的通用性
- **启发**：(1) "在处理之前先做粗粒度筛选"的思路可推广到 3D 点云、音频等其他模态；(2) 自回归 patch 选择本质是视觉 token 的"压缩编码"，与信息论中的率失真理论有深层联系

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "在注意力之前注意"的概念简洁而深刻，从根本上解决ViT效率瓶颈
- 技术深度: ⭐⭐⭐⭐ NTP+RL训练、自适应停止、多尺度注视的设计精良
- 实验充分度: ⭐⭐⭐⭐⭐ 行为分析+效率测试+基准对比+消融+OOD测试+新基准
- 写作质量: ⭐⭐⭐⭐⭐ 图表精美，直觉清晰，人类注视的类比恰到好处
- 实用价值: ⭐⭐⭐⭐⭐ 直接解决MLLM处理长高分辨率视频的核心瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)

</div>

<!-- RELATED:END -->
