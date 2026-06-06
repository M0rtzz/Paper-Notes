---
title: >-
  [论文解读] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation
description: >-
  [ICLR 2026][视频生成][Joint Audio-Video Generation] 提出 JavisDiT++，一个面向联合音视频生成（JAVG）的简洁统一框架，通过模态特定 MoE 提升生成质量、时间对齐 RoPE 实现帧级同步、音视频 DPO 对齐人类偏好…
tags:
  - "ICLR 2026"
  - "视频生成"
  - "Joint Audio-Video Generation"
  - "DiT"
  - "Mixture-of-Experts"
  - "RoPE"
  - "DPO"
---

# JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation

**会议**: ICLR 2026  
**arXiv**: [2602.19163](https://arxiv.org/abs/2602.19163)  
**代码**: [GitHub](https://JavisVerse.github.io/JavisDiT2-page)  
**领域**: 视频生成  
**关键词**: Joint Audio-Video Generation, DiT, Mixture-of-Experts, RoPE, DPO

## 一句话总结

提出 JavisDiT++，一个面向联合音视频生成（JAVG）的简洁统一框架，通过模态特定 MoE 提升生成质量、时间对齐 RoPE 实现帧级同步、音视频 DPO 对齐人类偏好，基于 Wan2.1-1.3B 仅用约 1M 公开数据即达到 SOTA。

## 研究背景与动机

联合音视频生成（JAVG）要求模型从文本描述同时生成时间同步、语义对齐的视频和音频。当前开源方法与商业模型（如 Veo3）相比存在三方面差距：

**生成质量**：现有方法要么用统一 FFN 处理两模态（UniForm），导致模态信息损失；要么用双流 DiT（JavisDiT、UniVerse-1），架构复杂且扩展性差。

**时间同步**：JavisDiT 用 ST-Prior、UniVerse-1 用 Stitching 策略，均为隐式同步，不够精确且增加推理开销。

**人类偏好对齐**：现有 JAVG 方法未引入偏好优化，在美学和和谐度上与人类期望存在差距。JavisDiT++ 是首个将偏好对齐引入 JAVG 的工作。

## 方法详解

### 整体框架

基于 Wan2.1-1.3B-T2V 作为视频 backbone，采用三阶段训练：音频预训练 → 音视频 SFT → 音视频 DPO。使用 Rectified Flow 作为噪声调度器，视频 VAE 来自 Wan2.1，音频 VAE 来自 AudioLDM2，均冻结。

### 关键设计

1. **模态特定 MoE（MS-MoE）**：音频和视频 token 通过共享的多头自注意力层进行跨模态交互，然后分别经过各自的 FFN 层进行模态内信息聚合。设计思路类似 BAGEL，但按模态而非任务分配 token。虽然总参数从 1.3B 增至 2.1B，但每个 token 激活的参数仍为 1.3B，因此推理开销不增加。相比以下两种替代方案更优：

    - Shared-DiT + LoRA：音频质量受限于可训练容量不足
    - Shared-DiT + Full-FT：音频预训练阶段过多参数偏移，严重损害视频质量

2. **时间对齐 RoPE（TA-RoPE）**：在 3D 位置 ID 的第一维（时间维）上对音频和视频 token 强制绝对时间对齐。视频 token 的位置 ID 为 $(t, h, w)$，音频 token 的位置 ID 设为：

$$R_a(t, m) = \left(\left[t \cdot \frac{T_v}{T_a}\right], t + H, m + W\right)$$

其中 $[\cdot]$ 为取整操作，$H$、$W$ 的偏移保证音视频位置 ID 不重叠。这种设计无需物理重排 token 序列，通过位置 ID 操作即可在全注意力框架中实现时间对齐，零额外推理成本。

3. **音视频 DPO（AV-DPO）**：首创将偏好对齐引入 JAVG。核心贡献：

    - **奖励模型**：从三个维度评估——音频质量（AudioBox + ImageBind）、视频质量（VideoAlign + ImageBind）、音视频对齐（ImageBind + Syncformer）
    - **偏好数据构建**：30K 提示 × 3 对生成 + ground truth，按模态分别归一化排序后选取 winner-loser 对，确保 winner 在所有模态维度上都优于 loser（约得到 25K 对）
    - **模态感知损失**：分别计算音频和视频的 DPO 损失并加权：

$$\mathcal{L}_{\mathrm{DPO}}^{av} = -\mathbb{E}\left[\log\sigma\left(-\beta_v(\mathrm{Diff}_{\mathrm{policy}}^v - \mathrm{Diff}_{\mathrm{ref}}^v) - \beta_a(\mathrm{Diff}_{\mathrm{policy}}^a - \mathrm{Diff}_{\mathrm{ref}}^a)\right)\right]$$

### 损失函数 / 训练策略

- 音频预训练：780K 音频-文本对
- 音视频 SFT：330K 音视频-文本三元组，使用 Flow Matching 目标
- 音视频 DPO：25K 偏好对，搭配 Flow Matching 正则化防过拟合
- 支持 2-5 秒、240p-480p 不同纵横比

## 实验关键数据

### 主实验（JavisBench, 240p4s）

| 模型 | 参数量 | FVD↓ | FAD↓ | AV-IB↑ | JavisScore↑ | DeSync↓ | 推理时间 |
|------|--------|------|------|--------|-------------|---------|----------|
| JavisDiT | 3.1B | 204.1 | 7.2 | 0.197 | 0.154 | 1.039 | 30s |
| UniVerse-1 | 6.4B | 194.2 | 8.7 | 0.104 | 0.077 | 0.929 | 13s |
| **JavisDiT++** | **2.1B** | **141.5** | **5.5** | **0.198** | **0.159** | **0.832** | **10s** |

### 消融实验（JavisBench-mini）

| 配置 | FVD↓ | FAD↓ | JavisScore↑ | DeSync↓ | 说明 |
|------|------|------|-------------|---------|------|
| Shared-DiT + LoRA | 227.6 | 6.51 | 0.098 | 0.934 | LoRA 容量不足 |
| Shared-DiT + Full-FT | 269.3 | 5.66 | 0.137 | 0.945 | 视频质量下降 |
| **MS-MoE** | **221.3** | **5.51** | **0.153** | **0.807** | 最佳架构 |
| 无同步机制 | - | - | 0.142 | 0.942 | 基线 |
| ST-Prior | - | - | 0.145 | 0.863 | +6s 延迟 |
| **TA-RoPE** | - | - | **0.153** | **0.807** | 零额外成本 |
| 无 DPO | 221.3 | 5.51 | 0.153 | 0.807 | SFT 基线 |
| Modality-Micro DPO | **198.5** | 5.32 | **0.156** | **0.776** | 最佳 DPO 策略 |

### 关键发现

- MS-MoE 在保持视频质量的同时大幅提升音频质量，证明模态特定 FFN 的必要性
- TA-RoPE 以零推理成本实现的同步效果优于需要额外计算的 ST-Prior 和 FrameAttn
- AV-DPO 在客观指标上改进温和，但人类评价中 25% 以上偏好提升，捕捉到了指标难以衡量的美学偏好
- 模态感知的偏好对构建至关重要——模态不一致的 winner 选择会导致 DPO 退化

## 亮点与洞察

- 用更少参数（2.1B vs 6.4B）和更少数据（1M vs 大规模）超越了双流架构，说明统一简洁架构 + 精心设计的模块比暴力堆叠更有效
- TA-RoPE 的位置 ID 操纵思路优雅——利用全注意力框架的对称性，无需物理重排序列即可实现时间对齐
- 首次将 DPO 引入多模态联合生成，且设计了模态感知的偏好数据构建流程
- 推理仅比纯视频生成多 1.6% 开销，实用性极强

## 局限与展望

- 当前视频分辨率和时长受限（240-480p, 2-5s），离实际商用还有距离
- AV-DPO 的客观指标提升有限，奖励模型的评估能力可能是瓶颈
- 音频 VAE（AudioLDM2）不是为联合生成设计的，可能限制了音频多样性
- 仅在 Wan2.1-1.3B 上验证，更大或不同系列模型的扩展性未知
- 与 Veo3 等商业模型仍有差距，特别是在复杂场景的语义对齐上

## 相关工作与启发

- JavisDiT 和 UniVerse-1 的双流 DiT 方案被 MS-MoE 统一替代，说明共享注意力 + 模态 FFN 是更高效的范式
- AV-DPO 的模态感知偏好数据策略可推广到其他多模态对齐场景（音频+3D、视频+触觉等）
- 将 TA-RoPE 的时间对齐思路引入更多需要跨模态同步的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ TA-RoPE 和 AV-DPO 有新意，MS-MoE 相对常规
- 实验充分度: ⭐⭐⭐⭐⭐ 全面的架构对比、同步机制对比、DPO 策略对比、主观评估，ablation 非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但部分描述略冗长
- 价值: ⭐⭐⭐⭐ 为开源 JAVG 设立新 SOTA 和新标杆，AV-DPO 思路对社区有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](../../CVPR2026/video_generation/unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)
- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](../../CVPR2026/video_generation/uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)

</div>

<!-- RELATED:END -->
