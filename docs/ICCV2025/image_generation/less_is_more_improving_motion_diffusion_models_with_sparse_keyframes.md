---
title: >-
  [论文解读] Less is More: Improving Motion Diffusion Models with Sparse Keyframes
description: >-
  [图像生成] 提出 sMDM，一种以稀疏关键帧为核心的运动扩散框架，通过 masking-interpolation 策略和 Visvalingam-Whyatt 关键帧选择算法，减少冗余帧处理，在文本对齐和运动质量上持续超越密集帧基线。
tags:
  - 图像生成
---

# Less is More: Improving Motion Diffusion Models with Sparse Keyframes

> **会议**: ICCV 2025
> **arXiv**: [2503.13859](https://arxiv.org/abs/2503.13859)
> **领域**: 动作生成·扩散模型·关键帧
> **关键词**: motion diffusion, sparse keyframes, masking-interpolation, Visvalingam-Whyatt, Lipschitz MLP

## 一句话总结

提出 sMDM，一种以稀疏关键帧为核心的运动扩散框架，通过 masking-interpolation 策略和 Visvalingam-Whyatt 关键帧选择算法，减少冗余帧处理，在文本对齐和运动质量上持续超越密集帧基线。

## 研究背景与动机

现有运动扩散模型（如 MDM）以密集帧序列作为输入和输出，面临两个核心问题：

**训练和推理的高复杂度**：self-attention 层的计算量随帧数二次增长，大规模运动数据集使训练成本高昂。实践中，运动扩散模型在小扩散步数设置下质量严重下降。

**缺乏可控性**：同时生成所有帧使得难以解释和控制模型行为。而专业动画师的工作流程是先定义稀疏关键帧再插值中间帧。

**核心动机**：受专业动画制作流程启发——动画师主要关注关键帧而非每一帧——sMDM 围绕稀疏的、几何意义上有意义的关键帧构建扩散框架。

## 方法详解

### 整体框架（Fig. 2）

sMDM 在 MDM 的 Transformer 基础上引入三个核心修改：

1. **Masking**：用二值关键帧 mask $\mathbf{M}$ 在 self-attention 中屏蔽非关键帧，注意力复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(K^2)$（$K \ll N$）
2. **Interpolation**：自注意力后对关键帧特征进行线性插值以重构非关键帧特征
3. **Lipschitz MLP**：将输入/输出线性层替换为 Lipschitz MLP，确保平滑插值

### 关键设计 1：关键帧选择

**训练时**：使用 Visvalingam-Whyatt 几何简化算法自动识别几何上最有意义的关键帧。该算法迭代移除"面积"最小（几何重要性最低）的帧，直到达到目标关键帧数量。使用 80% 的压缩率（保留 20% 的帧作为关键帧）。

**推理时（两阶段策略）**：
- **早期（$t > T' = \gamma \cdot T$）**：使用均匀 mask（关键帧均匀分布），提供稳定的去噪基线
- **晚期（$t \leq T'$）**：**动态 mask 更新** — 对中间帧 $\mathbf{x}_t$ 运用 Visvalingam-Whyatt 重新选择关键帧，使注意力聚焦于最关键的帧。$\gamma = 0.1$ 效果最佳。

### 关键设计 2：Lipschitz MLP

为确保特征空间中的线性插值产生平滑的运动输出，对输入/输出 MLP 施加 Lipschitz 约束：

$$\|g_\theta(y_1) - g_\theta(y_2)\|_p \leq \alpha \|y_1 - y_2\|_p$$

使用正弦激活函数（sine activations）替代 ReLU，更好地捕获高频运动细节（如突然的方向变化）。

### 损失函数

标准扩散重建损失加 Lipschitz 正则化：

$$\mathcal{L} = \|\mathbf{x}_0 - \hat{\mathbf{x}}_0\|^2 + \lambda \mathcal{L}_{lip}$$

其中 $\hat{\mathbf{x}}_0 = f_\theta(x_t, t, c)$，损失在**插值后的密集帧**上计算（而非仅关键帧）。

## 实验

### 主实验：文本到动作生成（Tab. 1, HumanML3D）

| 方法 | R@1 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ |
|------|-------|-------|-------|-----------|
| MDM | 0.320 | 0.611 | 0.544 | 5.566 |
| MLD | 0.481 | 0.772 | 0.473 | 3.196 |
| T2M-GPT | 0.606 | 0.838 | 12.475 | 16.812 |
| MoMask | 0.621 | 0.846 | 12.232 | 16.138 |
| ReMoDiffuse§ | 0.510 | 0.795 | 0.103 | 2.974 |
| **sMDM** | 0.494 | 0.776 | **0.130** | **3.051** |
| **sMDM-stella†** | **0.554** | **0.829** | **0.151** | **2.740** |

sMDM 使用相同的 CLIP 编码器即超越 MDM 大幅度；搭配 Stella-1.5B 编码器后（sMDM-stella），在所有指标上超越使用更大编码器的 MotionGPT、MotionLCM 和 MotionBase。

### 消融实验（Tab. 1 下半部分）

| 配置 | FID ↓ | R@3 ↑ | MM-Dist ↓ |
|------|-------|-------|-----------|
| 随机关键帧 | 0.249 | 0.777 | 3.086 |
| 无插值（仅关键帧损失） | 0.267 | 0.773 | 3.127 |
| 无 Lipschitz | 0.329 | 0.768 | 3.149 |
| **sMDM（完整）** | **0.130** | **0.776** | **3.051** |

关键发现：
- 几何关键帧选择远优于随机选择（FID 0.130 vs 0.249）
- 在插值后密集帧上计算损失比仅在关键帧上计算效果更好
- Lipschitz 约束对 FID 有显著贡献

### 不同扩散步数稳定性（Tab. 2）

| 步数 | 标准 FID | +动态采样 FID |
|------|---------|-------------|
| 1000 | 0.291 | **0.246** |
| 500 | 0.322 | **0.229** |
| 100 | 0.230 | **0.190** |
| 50 | 0.130 | **0.134** |
| 10 | 0.349 | 0.385 |

关键发现：动态 mask 更新在**大步数**配置下效果特别显著（FID 从 0.322 降至 0.229），这与空间引导技术（如 OmniControl）的发现一致。

### 长序列生成（Tab. 3, DoubleTake）

sPriorMDM 生成的运动更有表现力（EES 2.338 vs 1.856），更忠实于文本指令，但过渡区域 FID 略高——因为生成的运动更动态，过渡模式在训练集中较少见。

## 亮点与洞察

1. **稀疏训练范式**：本文证明了"少即是多"——关注 20% 的关键帧比处理所有帧效果更好，这与动画制作直觉一致
2. **Visvalingam-Whyatt 算法**的引入使关键帧选择具有几何意义，非随机或均匀采样
3. **即插即用**：sMDM 不改变 MDM 主体架构，仅通过 mask+插值+Lipschitz 实现提升，易于应用到其他扩散动作模型
4. 在极低扩散步数（10步）下仍保持良好质量，对实时应用有重要价值

## 局限性

- 关键帧压缩率作为固定超参数，不同运动类型可能需要不同的最优压缩率
- 动态 mask 更新在极低步数（10步）时效果不佳
- 线性插值可能不足以重建复杂的高频运动细节

## 相关工作

- 运动扩散：MDM、MLD、MotionDiffuse、FlowMDM
- 关键帧方法：CondMDI、KEYIN
- 实时控制：DiP、CLoSD、CAMDM

## 评分

- **新颖性**: ★★★★☆ — 稀疏关键帧训练运动扩散模型的思路新颖实用
- **技术深度**: ★★★★☆ — 各组件设计简洁有效，消融充分
- **实验质量**: ★★★★★ — 三类下游任务验证，多步数稳定性分析详尽
- **写作质量**: ★★★★☆ — 动机清晰，与动画工作流的类比直观

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](../../CVPR2025/image_generation/dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)

<!-- RELATED:END -->
