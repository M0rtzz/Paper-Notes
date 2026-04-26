---
title: >-
  [论文解读] Target-Aware Video Diffusion Models
description: >-
  [ICLR 2026][视频扩散模型] 提出 target-aware 视频扩散模型，仅需一张输入图像和目标物体的分割 mask，即可生成演员与指定目标交互的视频；核心创新是引入 [TGT] 特殊 token 并设计选择性交叉注意力损失，使模型关注目标的空间位置，在目标对齐和视频质量上全面超越基线。
tags:
  - ICLR 2026
  - 视频扩散模型
  - 目标感知
  - 交叉注意力损失
  - 人物交互
  - 动作规划
---

﻿---
title: >-
  [论文解读] Target-Aware Video Diffusion Models
description: >-
  [ICLR 2026][视频扩散模型] 提出 target-aware 视频扩散模型，仅需一张输入图像和目标物体的分割 mask，即可生成演员与指定目标交互的视频；核心创新是引入 [TGT] 特殊 token 并设计选择性交叉注意力损失，使模型关注目标的空间位置，在目标对齐和视频质量上全面超越基线。
tags:
  - ICLR 2026
  - 视频扩散模型
  - 目标感知
  - 交叉注意力损失
  - 人物交互
  - 动作规划
---

# Target-Aware Video Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2503.18950](https://arxiv.org/abs/2503.18950)  
**代码**: [taeksuu.github.io/tavid](https://taeksuu.github.io/tavid/)  
**领域**: 视频生成 / 人物-物体交互  
**关键词**: 视频扩散模型, 目标感知, 交叉注意力损失, 人物交互, 动作规划

## 一句话总结
提出 target-aware 视频扩散模型，仅需一张输入图像和目标物体的分割 mask，即可生成演员与指定目标交互的视频；核心创新是引入 [TGT] 特殊 token 并设计选择性交叉注意力损失，使模型关注目标的空间位置，在目标对齐和视频质量上全面超越基线。

## 研究背景与动机
视频扩散模型已展现出模拟复杂场景的显著能力，但要在实际应用中发挥作用，需要对内容和动作进行精确控制。现有的可控视频生成方法通常依赖密集的结构或运动线索（深度图、边缘图、光流、拖拽等）来引导演员的运动。这些方法对简单平移或视角变化有效，但对于**演员-目标交互场景**则存在根本困难——为演员提供结构性动作引导（如怎样伸手去拿桌上的杯子）非常困难。

另一个重要动机是：将视频扩散模型用作**高层动作规划器**。不同于将视频模型当"渲染器"（需密集运动输入），本文将其定位为"规划器"——仅给定目标位置就能生成合理的交互动作。这对机器人操控等下游应用具有重要意义。

核心 idea：仅用一个分割 mask 标记目标物体，让视频扩散模型的生成先验自主推断演员的合理交互动作。

## 方法详解

### 整体框架
输入：一张图像 $I$、目标物体的分割 mask $M$、描述期望动作的文本 prompt。输出：一段视频，其中演员准确地与 mask 指定的目标进行交互。基于 CogVideoX-5B-I2V 模型，通过 LoRA 微调。

### 关键设计

1. **Mask 条件注入**: 将二值分割 mask $M$ 下采样后与输入图像拼接在一起，作为额外的通道输入到扩散模型。通过扩展 image projection layer 的输入通道来支持额外的 mask 通道，新增权重初始化为零以保留预训练参数。这让模型能感知目标的空间位置，但仅此不足以保证目标感知——模型可能忽略 mask 信息。

2. **[TGT] Token 与交叉注意力损失**: 这是本文的核心创新。

    - 在文本 prompt 末尾添加 "The person interacts with [TGT] object."，引入特殊 token [TGT] 来编码目标的空间信息
    - 设计交叉注意力损失将 [TGT] token 的交叉注意力图与输入 mask 对齐：
    $\mathcal{L}_{\text{attn}} = \mathbb{E}[\|A(\mathbf{z}_t^0, [\text{TGT}]) - M\|_2^2]$
   其中 $A(\mathbf{z}_t^0, [\text{TGT}])$ 是第一帧视频潜变量与 [TGT] token 间的交叉注意力权重
    - 总训练目标为：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{attn}} \mathcal{L}_{\text{attn}}$（$\lambda_{\text{attn}} = 0.1$）
    - 推理时，将 [TGT] 前置到文本中指代目标的词语前，使模型利用 mask 提供的空间线索

3. **选择性交叉注意力损失**: 不是对所有 transformer block 和注意力区域无差别施加损失，而是精心选择：

    - **选择性 Transformer Block**: 通过实验评估发现第 5~23 个 block（共 42 个）的注意力图与分割 mask 最为语义对齐，每个训练步在这些 block 中选择 7 个施加损失
    - **选择性注意力区域**: MM-DiT 架构的注意力分为 4 种（text-to-text, T2V, V2T, video-to-video），其中 **V2T (video-to-text) cross-attention** 直接影响视频潜表示的值，效果最好。T2V 虽也编码语义信息但影响间接

4. **数据集构建**: 从 BEHAVE（简单人物-物体交互）和 Ego-Exo4D（复杂场景如烹饪、修车）数据集中提取 1290 个视频片段，每个片段满足：(1) 初始帧中演员存在但未与目标交互，(2) 后续帧中演员与目标发生交互。用 SAM 获取目标 mask，用 CogVLM2 生成文本 caption。

### 训练细节
- 基于 CogVideoX-5B-I2V，LoRA rank=128, α=64
- 仅训练 LoRA 层和 image projection layer，冻结其余参数
- 训练 2000 步，AdamW，lr=1e-4，batch size=4
- 4x NVIDIA A100 GPU，约 6 小时
- 推理：DPM 采样器，50 步，CFG=6，单卡 A100 约 4 分钟/视频

## 实验关键数据

### 主实验 — 目标对齐与视频质量

| 方法 | Hum. Eval. ↑ | User Pref. ↑ | Contact Score ↑ | SS | BC | DD | MS |
|------|-------------|-------------|----------------|----|----|----|----|
| CogVideoX | 0.592 | 0.456 | 28.4% | 0.914 | 0.903 | 0.950 | 0.988 |
| CogVideoX w. data | 0.692 | 0.596 | 36.2% | 0.915 | 0.900 | 0.956 | 0.990 |
| Attn. Mod. | 0.613 | 0.508 | 22.2% | 0.878 | 0.887 | 0.827 | 0.984 |
| **Ours** | **0.896** | **0.892** | **最高** | **0.938** | **0.914** | 0.956 | 0.905 |

### 消融实验

| 配置 | Contact Score | 说明 |
|------|--------------|------|
| $\lambda_{\text{attn}} = 0.0$（无注意力损失） | 0.688 | ≈ CogVideoX w. data，证明注意力损失至关重要 |
| $\lambda_{\text{attn}} = 0.01$ | 0.756 | 有改善但不充分 |
| $\lambda_{\text{attn}} = 0.1$（ours） | **0.896** | 最优平衡 |
| $\lambda_{\text{attn}} = 1.0$ | 0.904 | Contact 微升但视频质量下降 |
| 随机 Block 选择 | 0.840 | 不如语义选择 |
| 等间距 Block 选择 | 0.839 | 不如语义选择 |
| T2V Cross-Attn. | 0.784 | 不如 V2T |
| V2T Cross-Attn.（ours） | **0.896** | V2T 直接影响视频潜表示 |

### 关键发现
- 交叉注意力损失是实现目标感知的关键：$\lambda=0$ 时性能几乎等于仅用数据微调的 baseline
- V2T cross-attention 是正确的施加位置：它直接通过 value 的点积影响视频潜表示
- 语义选择 block（第 5-23 block 中每 3 个取 1 个）效果最好
- 场景中有多个同类物体时，mask 的优势尤为明显（文本无法区分，mask 可以精确指定）
- 模型泛化到非人类主体（如动物）的交互

## 亮点与洞察
- **最小化控制输入，最大化生成先验**: 仅用一个分割 mask（不需要密集轨迹或多帧引导）就能让模型自主推断合理的交互动作，充分利用了视频扩散模型的生成能力
- **[TGT] token 设计优雅**: 巧妙地利用 text token 来携带空间信息，不需要修改模型架构，只需加一个额外训练损失
- **选择性损失的分析很深入**: 系统性地分析了 MM-DiT 中不同 block 和注意力区域的语义特性，不是黑盒调参而是有原则的设计
- **两个下游应用亮眼**: 视频内容创作（导航+交互组合）和零样本 3D HOI 运动合成，展示了模型作为"动作规划器"的潜力

## 局限与展望
- 视频质量受限于基础开源模型（CogVideoX），闭源商业模型可能更好
- 训练数据的摄像机都是静态的，生成的视频倾向于固定机位
- 数据集仅 1290 个片段，扩大数据规模可能进一步提升泛化
- 当前只支持单个目标的 mask，多目标同时交互的扩展（虽有 [SRC]+[TGT] 的初步探索）仍需完善
- 生成的运动虽合理但可能在物理精确性上不足（如接触力学）
- 物理仿真学习部分的 3D pose 和场景尺度未完全对齐

## 相关工作与启发
- **与 ControlNet 系 方法的区别**: ControlNet 类方法需要每帧的密集条件（深度/边缘），适合精确控制简单运动；本文用单帧 mask 适合 HOI 场景
- **与 DragDiffusion 的比较**: 拖拽式方法对大幅运动失效，且无法生成复杂交互
- **与 Direct-a-Video 的区别**: 注意力调制方法不需训练但效果差——MM-DiT 中 softmax 的行归一化导致交叉注意力值的放大会破坏自注意力，产生时间不一致
- **启发**: 视频扩散模型内在蕴含了丰富的物理世界交互先验，关键是如何用最少的信号（一个 mask）来释放这些先验

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](../../CVPR2026/video_generation/tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](../../CVPR2025/video_generation/interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)

<!-- RELATED:END -->
