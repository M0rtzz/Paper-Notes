---
title: >-
  [论文解读] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection
description: >-
  [CVPR 2026][LoRA] 提出 FAAR，一种频率感知的多任务参数高效微调方法，通过 Performance-Driven Rank Shrinking (PDRS) 为每个任务和层动态选择最优秩，并设计 Task-Spectral Pyramidal Decoder (TS-PD) 利用 FFT 频率信息增强空间感知和跨任务一致性，以传统微调 1/9 的参数量实现更优性能。
tags:
  - CVPR 2026
  - LoRA
  - 自动秩选择
  - FFT
  - multi-task learning
  - PEFT
---

# FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection

**会议**: CVPR 2026  
**arXiv**: [2603.20403](https://arxiv.org/abs/2603.20403)  
**代码**: 有（论文中提到）  
**领域**: 参数高效微调 / 多任务学习  
**关键词**: LoRA, 自动秩选择, FFT, multi-task learning, PEFT

## 一句话总结

提出 FAAR，一种频率感知的多任务参数高效微调方法，通过 Performance-Driven Rank Shrinking (PDRS) 为每个任务和层动态选择最优秩，并设计 Task-Spectral Pyramidal Decoder (TS-PD) 利用 FFT 频率信息增强空间感知和跨任务一致性，以传统微调 1/9 的参数量实现更优性能。

## 研究背景与动机

多任务学习（MTL）旨在同时学习多个任务，共享表示以发现任务间的关系和结构。随着骨干模型参数量不断增长，传统全量微调变得越来越不可行。参数高效微调（PEFT），特别是基于低秩适应（LoRA）的方法成为主流。

然而，现有 LoRA-based MTL 方法存在两个核心局限：

**固定秩问题**：现有方法对所有层和所有任务使用统一的秩，这不符合直觉——不同任务可能需要不同的适应强度，不同层也需要不同程度的微调灵活性。深层需要更强的适应能力来处理任务特定的精细信息，而浅层可能只需少量调整。

**缺乏空间归纳偏置**：现有 LoRA-based MTL 策略忽视了跨任务交互在深层的作用。对于语义分割、深度估计、法线估计等密集视觉任务，强空间感知和跨任务几何一致性至关重要，但低秩适应本身缺乏这种能力。

FAAR 的解决思路：
- 通过动态秩收缩（PDRS）解决固定秩问题，让每个任务/每层自动找到最优秩
- 通过频率分析（TS-PD）引入廉价但有效的空间信息和跨任务关系

## 方法详解

### 整体框架

FAAR 基于冻结的 Swin Transformer 骨干，在注意力和 MLP 层放置 DoRA 适配器。每个 Transformer 阶段的最后一个块使用任务特定适配器，前面的块共享适配器。骨干之后接 Task-Spectral Pyramidal Decoder (TS-PD) 进行频率增强和跨任务对齐。整个训练过程由 PDRS 控制，动态减少适配器秩。

### 关键设计

1. **Performance-Driven Rank Shrinking (PDRS)**：

    - **秩掩码（Rank Masking）**：每次前向传播随机采样前缀大小 $b \in \{1, ..., r_{curr}\}$，构建二进制掩码 $m$，只让前 $b$ 个秩分量参与计算
        - $A^{eff} = \text{diag}(m) A$, $B^{eff} = B \text{diag}(m)$
        - 这迫使重要的秩-1 更新向低维方向集中
    - **覆盖策略（Coverage Strategy）**：
        - 每次反向传播计算每个活跃秩 $i$ 的重要性分数：$s_i = \frac{1}{2}(|\langle A_{:,i}^{eff}, \frac{\partial \mathcal{L}}{\partial A_{:,i}^{eff}} \rangle| + |\langle B_{i,:}^{eff}, \frac{\partial \mathcal{L}}{\partial B_{i,:}^{eff}} \rangle|)$
        - 通过 EMA 累积跨批次的分数：$\hat{s}_i \leftarrow \beta \hat{s}_{i-1} + (1-\beta) s_i$
        - 每个 epoch 末尾，按分数降序排列，选择满足覆盖率 $\rho$ 的最少秩数 $K$：$K = \min\{k : c(k) \geq \rho\}$
        - 未覆盖的秩从优化中永久删除
    - 设计动机：基于 MTL 损失的方向导数反映每个秩-1 分量的实际贡献，以性能为导向的收缩确保不损失关键更新

2. **DoRA 适配器（而非 LoRA）**：

    - DoRA 将低秩适应解耦为幅度和方向：$\text{Out}_i^{DoRA} = m_i \frac{W_i + \alpha B_i A_i}{\|W_i + \alpha B_i A_i\|_2} x + b_i$
    - 在极低秩下比 LoRA 更稳定，与 PDRS 的秩收缩配合更好
    - 实验验证：高秩时 DoRA 不一定优于 LoRA，但低秩时 DoRA 明显更好

3. **Task-Spectral Pyramidal Decoder (TS-PD)**：

    - **Channel-wise Spectral Filter (CW-SP)**：
        - 对每个任务特定特征进行 FFT，学习任务/分辨率特定的 2D 频率滤波矩阵 $W_t^{res}$
        - 通过逐元素乘法 $Y = W \odot FFT(I)$ 选择性增强/抑制不同频率
        - 逆 FFT 变换回特征空间后，用可学习的 scale/shift 参数调制
        - 设计动机：不同任务需要不同的频率信息——边缘检测依赖高频，深度估计利用高低频

    - **Cross-Task Consensus Alignment (XT-Cons)**：
        - 对于主任务，计算辅助任务频谱的平均表示 $F_{avg}$
        - 从主任务频谱提取高频和低频掩码 $M_{low}$, $M_{high}$
        - 计算对齐差异：$\Delta_{low,high} = M_{low,high} * (F_{avg} - FFT(X_i^{main}))$
        - 用可学习标量 $\alpha_{low,high}$ 缩放贡献
        - 设计动机：通过频域中辅助任务的"共识"来推动主任务表示的几何一致性，比直接在空间域交互更廉价

### 损失函数 / 训练策略

- MTL 损失：$L_{MTL} = \sum_{i=1}^T w \times L_i$
    - 语义分割、人体部件分割：像素交叉熵
    - 深度估计、法线估计：L1 损失
    - 显著性检测：平衡交叉熵
- 覆盖率参数：$\rho_{shared} = \rho_{task} = 0.95$
- 骨干：Swin-Tiny (ImageNet-1k 预训练)，解码器：HRNet
- 初始秩 $r_{init} = 64$，训练过程中动态收缩到约 $r_{global} \approx 5$
- 单张 NVIDIA A40，学习率 $5 \times 10^{-4}$，batch size 32

## 实验关键数据

### 主实验

**PASCAL-Context 数据集**（4 个任务）：

| 方法 | SemSeg (mIoU↑) | HumanParts (mIoU↑) | Saliency (mIoU↑) | Normals (rmse↓) | Δm (%) | 参数量(M) |
|------|----------------|---------------------|-------------------|-----------------|--------|-----------|
| Single Task | 67.21 | 61.93 | 62.35 | 17.97 | 0 | 112.62 |
| MTL Full FT | 67.56 | 60.24 | 65.21 | 16.64 | +2.23 | 30.06 |
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 | 8.34 |
| TADFormer (r=64) | 70.82 | 60.45 | 65.88 | 16.48 | +4.24 | 7.38 |
| **FAAR** | **72.02** | **61.25** | **66.11** | **16.35** | **+5.28** | **3.38** |

**NYUDv2 数据集**（3 个任务）：

| 方法 | SemSeg (mIoU↑) | Depth (rmse↓) | Normals (rmse↓) | Δm (%) | 参数量(M) |
|------|----------------|---------------|-----------------|--------|-----------|
| Single Task | 42.65 | 0.60 | 22.83 | 0 | 84.00 |
| MTL Full FT | 38.85 | 0.66 | 24.33 | -8.49 | 28.10 |
| TADFormer (r=64) | 40.85 | 0.64 | 27.48 | -10.42 | 8.90 |
| **FAAR** | **41.27** | **0.63** | **26.35** | **-7.88** | **2.85** |

### 消融实验

PASCAL-Context 上的组件消融：

| 配置 | SemSeg | HumanParts | Saliency | Normals | Δm |
|------|--------|------------|----------|---------|-----|
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 |
| + DoRA (高秩) | 67.55 | 60.00 | 64.70 | 17.20 | +1.36 |
| + PDRS w/ LoRA | 68.11 | 59.93 | 65.54 | 16.50 | +2.83 |
| + PDRS w/ DoRA (1) | 71.35 | 61.02 | 65.92 | 16.42 | +4.92 |
| + TS-PD (2) | 70.73 | 60.95 | 65.92 | 16.40 | +4.63 |
| **FAAR (1+2)** | **72.02** | **61.25** | **66.11** | **16.35** | **+5.28** |

### 关键发现

1. **秩收缩模式符合直觉**：任务特定层和深层倾向于保留更大的秩，因为它们处理更精细的任务特定信息；共享层和浅层的秩被大幅削减
2. **DoRA 在低秩时显著优于 LoRA**：高秩时 DoRA 性能反而下降（+1.36 vs +2.55），但经 PDRS 收缩到低秩后 DoRA 发挥巨大优势（+4.92）
3. **初始秩对最终性能影响不大**：$r_{init} \in \{16, 32, 64\}$ 时结果几乎相同，说明 PDRS 的搜索空间足够
4. **XT-Cons 的跨任务对齐有效**：在 TS-PD 基础上额外带来 +0.8% Δm 提升，验证了频域跨任务一致性的价值
5. **9倍参数节省**：FAAR (3.38M) vs MTL Full FT (30.06M)，同时性能更优

## 亮点与洞察

- **秩收缩以性能为导向**：不同于 AdaLoRA（基于奇异值重要性）或 DyLoRA（训练对低秩的鲁棒性），PDRS 直接用 MTL 损失的方向导数指导秩削减，更直接地与优化目标对齐
- **频率域作为跨任务桥梁**：首次在密集视觉 MTL 中利用 FFT。频率域自然区分了边缘/语义信息，为不同任务提供了有意义的共享基础
- **DoRA + 极低秩的协同效应**：高秩时 DoRA 不一定优于 LoRA，但当秩被 PDRS 动态压缩到极低值时，DoRA 的幅度-方向解耦变得关键
- **全任务同时改善**：FAAR 在所有 4 个 PASCAL 任务上均优于基线，不存在某些任务以牺牲其他任务为代价的情况

## 局限与展望

1. 在 NYUDv2 上所有 MTL PEFT 方法均未超过单任务训练，FAAR 也未完全解决小数据集上的 MTL 困难
2. 覆盖率参数 $\rho$ 仍需手动设定（虽然论文称 0.95 在验证集上选择，但不同数据集可能需要不同值）
3. 仅验证了 Swin-Tiny 骨干，对更大骨干（如 Swin-Base/Large）或 ViT 的效果未知
4. TS-PD 的频率滤波矩阵对每个分辨率和任务单独学习，任务数量增多时参数增长
5. 跨任务对齐仅在频域进行，空间域的交互可能提供额外互补信息

## 相关工作与启发

- **MTLoRA / TADFormer**：MTL 中的 LoRA 基线方法，使用固定秩
- **AdaLoRA / AutoLoRA / DyLoRA**：单任务下的自动秩选择方法，FAAR 将其扩展到多任务
- **FADA / NightAdapter**：频率适配器在域泛化/夜晚分割中的应用，启发了 TS-PD 的设计
- **DiTASK**：通过神经微分同胚适应奇异值的替代方案

## 评分

- 新颖性: ⭐⭐⭐⭐ （PDRS和TS-PD各自有新意，但都是已有思路的改进组合）
- 实验充分度: ⭐⭐⭐⭐ （两个数据集、详细消融、参数效率对比完整）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但公式和缩写较多影响可读性）
- 价值: ⭐⭐⭐⭐ （为MTL PEFT提供了实用且高效的解决方案，9倍参数节省很有吸引力）
