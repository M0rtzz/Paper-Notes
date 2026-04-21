---
title: >-
  [论文解读] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout
description: >-
  [CVPR 2026][自回归视频生成] 提出 ∞-RoPE，一个训练免调的推理时框架，通过 Block-Relativistic RoPE、KV Flush 和 RoPE Cut 三个组件，将仅在5秒视频上训练的自回归视频扩散模型扩展为支持无限时长生成、精细动作控制和电影级场景切换的系统。
tags:
  - CVPR 2026
  - 自回归视频生成
  - 位置编码
  - 无限长视频
  - 动作控制
  - 推理时方法
---

# Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

**会议**: CVPR 2026  
**arXiv**: [2511.20649](https://arxiv.org/abs/2511.20649)  
**代码**: [Project Page](https://infinity-rope.github.io)  
**领域**: 视频生成 / 扩散模型  
**关键词**: 自回归视频生成, 位置编码, 无限长视频, 动作控制, 推理时方法

## 一句话总结

提出 ∞-RoPE，一个训练免调的推理时框架，通过 Block-Relativistic RoPE、KV Flush 和 RoPE Cut 三个组件，将仅在5秒视频上训练的自回归视频扩散模型扩展为支持无限时长生成、精细动作控制和电影级场景切换的系统。

## 研究背景与动机

当前自回归视频扩散模型面临三大核心瓶颈：

**有限时间范围**：3D-RoPE 位置编码将生成限制在固定的 1024 帧内，超出后注意力质量急剧退化

**动作响应迟钝**：在长序列 rollout 中，prompt 变更无法立即生效，KV cache 中的旧语义持续影响生成

**缺乏场景跳转能力**：无法在单一生成流中实现电影式的不连续场景切换

**关键洞察**：在 Self-Forcing 范式下仅训练5秒片段的模型，实际上已经具备高动态的无限时长生成能力——瓶颈不在模型容量，而在位置编码的绝对索引机制。作者提出通过相对性的位置编码重参数化和 KV cache 管理来突破，无需任何额外训练。

## 方法详解

### 整体框架

∞-RoPE 基于 Self-Forcing 蒸馏得到的 Wan2.1-T2V-1.3B 模型（4步因果生成器），在推理时引入三个互联组件：
- **Block-Relativistic RoPE**：相对性时间位置编码，突破固定帧数限制
- **KV Flush**：KV cache 重置机制，实现即时 prompt 响应
- **RoPE Cut**：时间坐标不连续跳转，实现多镜头场景切换

### 关键设计

1. **Block-Relativistic RoPE（核心）**

   自回归生成以3帧为一个 block 推进：$\mathbf{B}_f = \{f-2, f-1, f\}$。传统绝对 RoPE 中 $i \gg f_{\text{limit}}$ 时进入未见过的位置区域导致失效。Block-Relativistic RoPE 将时间坐标定义为**移动的局部参考系**：

   $$\tilde{\mathbf{B}}_i = \begin{cases} \mathbf{B}_i, & \text{if } i \leq f_0 \\ \mathbf{B}_{f_0} = \{f_0-2, f_0-1, f_0\}, & \text{otherwise} \end{cases}$$

   当新 block 生成时，其 RoPE 索引始终被旋转到模型最大帧范围 $f_{\text{limit}}$ 内，而更早的 block 的时间相位被反向旋转以保持相对时间几何不变。设计动机：类似认知神经科学中的"语义化"（semanticization），远期记忆丧失精确时间标记但保留语义信息——最早缓存帧的时间坐标坍缩为共享最小索引 $\mathbf{B}_{\bar{1}} = \{1,1,1\}$。

2. **KV Flush（动作控制）**

   当 prompt 变更时，清空所有 KV cache，仅保留两个锚点：**全局 sink 帧**（稳定注意力归一化）和**最后生成帧**（保持局部时间连续性）。新动作直接在这两个最小锚点上条件化生成，实现零延迟的 prompt 响应。相比 no-cache（突兀变化）、full-cache（语义滞后）、KV re-cache（高延迟），KV Flush 在效率和可控性上均优。

3. **RoPE Cut（场景切换）**

   通过在时间 RoPE 坐标中引入受控的不连续跳转实现电影级多镜头切换。对当前 block $\mathbf{B}_f = \{f-2, f-1, f\}$，重新映射为：

   $$\mathbf{B}_{f \to f+\Delta} = \{f-2, f+\Delta-1, f+\Delta\}$$

   跳转后的帧被视为"过去上下文"，生成从新的原始时间位置重新开始。由于相对性公式中不存在绝对位置，坐标系随每次 cut 自行偏移，即使大跨度时间/语义跳转后仍能保持身份一致性。

### 损失函数 / 训练策略

∞-RoPE 是**纯推理时方法**，不涉及额外训练。底层 Self-Forcing 模型基于 Rectified Flow 公式训练：$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$，通过神经速度场 $v_\theta$ 参数化的 ODE 求解逆过程。实验固定 KV cache 大小为6，onset index $f_0=21$，CFG scale 3.0，timestep shift 5.0。

## 实验关键数据

### 主实验

VBench 评测，5秒和60秒视频生成（表格为60秒数据）：

| 模型 | Background Consistency | Dynamic Degree | Subject Consistency | Overall |
|------|----------------------|---------------|-------------------|---------|
| NOVA | 0.8806 | 0.12 | 0.7750 | 0.6901 |
| SkyReels-V2 | 0.8995 | 0.44 | 0.8499 | 0.7768 |
| CausVid | 0.8985 | 0.52 | 0.8675 | 0.7940 |
| Self-Forcing | 0.8784 | 0.32 | 0.8360 | 0.7715 |
| Rolling-Forcing | 0.9447 | 0.36 | 0.9409 | 0.8146 |
| **∞-RoPE** | **0.9490** | **0.52** | **0.9444** | **0.8298** |

120秒和240秒超长视频（240秒数据）：

| 模型 | Background Consistency | Dynamic Degree | Subject Consistency | Overall |
|------|----------------------|---------------|-------------------|---------|
| Rolling-Forcing | 0.9248 | 0.40 | 0.9080 | 0.8017 |
| **∞-RoPE** | **0.9361** | **0.64** | **0.9256** | **0.8309** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Block-Relativistic RoPE 开启 vs 关闭 | Self-Forcing 单独无法维持动态长视频 | 仅5秒训练模型+BRRoPE 即可生成高质量30s+ |
| KV cache 大小扫描 | Overall/Aesthetic/Dynamic 随 cache 变化 | 固定 cache 6 在各时长上达到最佳平衡 |
| KV Flush 对比 no-cache/full-cache/re-cache | 即时语义响应+平滑运动连续 | KV Flush 在效率和可控性上全面领先 |

### 关键发现

- ∞-RoPE 在所有时长（5s/60s/120s/240s）上的 Overall 分数均为最高或并列最高
- 关键优势在 **Subject Consistency** 和 **Background Consistency**，在超长视频中优势更加显著
- Dynamic Degree 在 240s 达到 **0.64**，远超其他方法（大多 0.24-0.40），说明长期生成不会退化为静止

## 亮点与洞察

- **认知科学启发的设计**：将远期帧的时间坐标坍缩为"语义记忆"，类比人类记忆中的 semanticization 过程
- **注意力图的可解释性**：通过 attention map 可视化清晰展示了 BRRoPE（对角带+sink列）、KV Flush（切断中间历史）、RoPE Cut（分裂为两个独立对角块）的不同结构
- **零训练开销**：作为纯推理时方法，可即插即用于任何 Self-Forcing 变体

## 局限与展望

- 依赖 Self-Forcing 蒸馏的基础模型，模型本身的生成质量上限不变
- 场景切换的语义连贯性依赖 sink 帧的全局信息，复杂场景下可能不足
- 仅在 1.3B 参数模型上验证，14B 级模型的效果未知

## 相关工作与启发

- **Self-Forcing / Self-Forcing++**：提供了自回归 rollout 训练范式，∞-RoPE 在其基础上实现推理时突破
- **Rolling Forcing**：渐进噪声窗口方法是主要竞争者，但仍受限于 RoPE 范围
- **FLEX**：后续工作引入频率感知 RoPE 调制，与本文互补

## 评分

- **新颖性**: ★★★★☆ — 位置编码的相对性重参数化思路巧妙，认知科学类比有启发
- **技术深度**: ★★★★☆ — 三个组件设计完整、互相配合，机理分析充分
- **实验充分度**: ★★★★☆ — VBench 多时长全面评测，但缺少用户研究
- **实用性**: ★★★★★ — 训练免调、即插即用，实际部署潜力大

<!-- RELATED:START -->

## 相关论文

- [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](../../CVPR2025/video_generation/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)
- [CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)

<!-- RELATED:END -->
