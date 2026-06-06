---
title: >-
  [论文解读] 统一不同生成顺序的掩码扩散模型
description: >-
  [ICML 2026][图像生成][掩码扩散模型] 提出统一框架 OeMDM 和学习型版本 LoMDM——通过显式建模"速度"（生成优先级）将随机掩码、自回归、块扩散模型统一在一个 NELBO 下，实现从零开始联合学习生成顺序和扩散骨干。
tags:
  - "ICML 2026"
  - "图像生成"
  - "掩码扩散模型"
  - "生成顺序"
  - "速度场"
  - "联合学习"
---

# 统一不同生成顺序的掩码扩散模型

**会议**: ICML 2026  
**arXiv**: [2602.02112](https://arxiv.org/abs/2602.02112)  
**代码**: 待确认  
**领域**: 扩散模型 / 文本生成 / 语言建模  
**关键词**: 掩码扩散模型, 生成顺序, 速度场, 联合学习

## 一句话总结
提出统一框架 OeMDM 和学习型版本 LoMDM——通过显式建模"速度"（生成优先级）将随机掩码、自回归、块扩散模型统一在一个 NELBO 下，实现从零开始联合学习生成顺序和扩散骨干。

## 研究背景与动机

**领域现状**：掩码扩散模型（MDMs）是自回归模型（ARMs）的潜在替代品，但生成质量严重依赖生成顺序。

**现有痛点**：既有方案要么硬编码顺序（如块状 L2R），要么为预训练 MDM 学习顺序策略——后者需额外计算且因两阶段优化导致次优解。

**核心矛盾**：MDM 本身对顺序无感知，统一噪声调度使所有位置去噪率相同，导致生成顺序完全随机；而有序方法各自为营无统一视角。

**本文目标**：（1）在单一框架下统一 MDM、ARM、块扩散；（2）直接从头联合学习生成顺序与扩散模型。

**切入角度**：将 NELBO 中隐含的生成速率显式化为"速度"函数，设计位置相关的自适应噪声调度。

**核心 idea**：用位置相关调度器替代全局固定调度，让扩散过程"知道"应该先生成哪些位置，通过速度匹配损失同时优化骨干和生成策略。

## 方法详解

### 整体框架
OeMDM 引入自由形式调度器 $\alpha_F: I \times [0,1] \to [0,1]^L$，让不同位置在前向过程获得不同噪声量。NELBO 分解为重建损失和速度不匹配损失两部分。LoMDM 在此基础上用神经网络 $\phi$、$\psi$ 参数化前向和反向速度，实现端到端联合学习。

### 关键设计

1. **速度场显式化**:

    - 功能：使隐含的生成顺序具体化为可优化的函数 $A(u,t) = -\partial_t\alpha_F(u,t) \oslash (1-\alpha_F(u,t))$，表示位置 $i$ 在时刻 $t$ 的去噪速度。
    - 核心思路：反向后验和去噪过程可改写为 $\text{Cat}((1-A^{(i)}dt)m + A^{(i)}dt \cdot x^{(i)})$，速度高的位置更早被还原。
    - 设计动机：解决 MDM 顺序无感知问题——显式速度让训练信号聚焦于高优先级 token，推导出有原则的 NELBO 分解。

2. **广义 NELBO 分解**:

    - 功能：将 OeMDM 目标函数分解为 $L_{\text{main}} + L_{\text{velocity}}$，其中 $L_{\text{main}}$ 按速度加权重建损失，$L_{\text{velocity}} = A(i)(\log A(i) - \log \hat{A}(i)) - (A(i) - \hat{A}(i)) \geq 0$。
    - 核心思路：$L_{\text{velocity}}$ 在前向速度 $A_\phi$ 和反向速度 $\hat{A}_\psi$ 对齐时为 0，强制两者学习到相同的生成顺序。
    - 设计动机：统一训练和推理——训练中学到的顺序直接用于生成，避免两阶段优化。

3. **参数高效的联合学习**:

    - 功能：复用扩散骨干 $\theta$ 的 Transformer 特征提取器，用轻量级 MLP+Transformer 层参数化 $\alpha_\phi(x,t)$、$\hat{\alpha}_\psi(z_t,t)$。
    - 核心思路：$\alpha^{(i)}_\phi(x,t) := 1 - t^{c_1 + c_2 \cdot [\text{NormSig}(g_\phi(f(x)))]_i}$，用归一化 Sigmoid 输出调制相对优先级。
    - 设计动机：避免引入大量新参数导致优化不稳定；通过 stop-gradient 技巧让调度器独立优化。

## 实验关键数据

### 主实验

| 数据集 | MDLM | BD3LM(L'=4) | GenMD4 | LoMDM | 提升 |
|--------|------|------------|--------|--------|------|
| LM1B   | 27.0 | -          | 26.9   | 25.4   | -1.5 vs MDLM |
| LM1B+packed | 31.8 | 28.2 | 30.0 | 27.2 | -4.6 vs MDLM |
| OWT    | 23.2 | 20.7       | 21.8   | 20.4   | -2.8 vs MDLM |

### 零样本泛化

| 数据集 | MDLM | BD3LM | LoMDM | vs MDLM |
|---------|------|--------|--------|---------|
| PTB     | 95.26 | 96.81 | 80.40  | ↓14.86 |
| WikiText | 32.83 | 31.31 | 27.82 | ↓5.01 |
| Lambada | 47.52 | 50.03 | 36.32  | ↓11.20 |

### 关键发现
- LoMDM 在 7/7 零样本数据集上超越 MDLM，6/7 领先所有扩散模型；在 4/7 数据集上击败自回归 Transformer。
- 生成 PPL（NFE=256）：LoMDM 73.98 vs MDLM 79.43。
- 消融（禁用推理调度 $c_2=0$）：生成 PPL 从 48.29 升至 59.34。

## 亮点与洞察
- **统一视角突破**：将 ARM、MDM、块扩散作为 OeMDM 在不同调度下的特例，用一个 NELBO 框架解释。
- **速度匹配巧妙设计**：$L_{\text{velocity}} \geq 0$ 的凸形式保证优化稳定，同时强制训推一致。
- **端到端联合学习**：相比 GenMD4 的冻结骨干+学习调度，LoMDM 同步优化，骨干得到顺序感知的训练信号。

## 局限与展望
- 训练成本——单次迭代需 3 倍 forward pass，绝对吞吐量略低。
- 调度器设计——参数化 $\alpha_\phi(x,t)$ 仍是手工设计的形式。
- 可扩展性——实验限于 LM1B/OWT 规模，大模型上的表现需验证。

## 相关工作与启发
- **vs MDLM/SEDD**：都用随机掩码但无顺序优化；LoMDM 通过显式调度器实现上下文感知的生成路径。
- **vs BD3LM**：硬编码 L2R 块结构；LoMDM 学到的顺序更灵活。
- **vs GenMD4**：都学调度器但 GenMD4 是两阶段；LoMDM 从零开始端到端优化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次用速度场统一离散扩散与自回归。
- 实验充分度: ⭐⭐⭐⭐⭐  覆盖 3 数据集 + 3 评估指标 + 详细消融。
- 写作质量: ⭐⭐⭐⭐  推导完整清晰。
- 价值: ⭐⭐⭐⭐⭐  为离散扩散文本生成提供原则性框架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[ICML 2026\] Let EEG Models Learn EEG](let_eeg_models_learn_eeg.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)

</div>

<!-- RELATED:END -->
