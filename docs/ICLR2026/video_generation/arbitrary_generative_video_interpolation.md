---
title: >-
  [论文解读] Arbitrary Generative Video Interpolation
description: >-
  [ICLR 2026][视频生成][Video Frame Interpolation] ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。
tags:
  - "ICLR 2026"
  - "视频生成"
  - "Video Frame Interpolation"
  - "Generative VFI"
  - "RoPE"
  - "Temporal Conditioning"
  - "Any-length Generation"
---

# Arbitrary Generative Video Interpolation

**会议**: ICLR 2026  
**arXiv**: [2510.00578](https://arxiv.org/abs/2510.00578)  
**代码**: [项目主页](https://mcg-nju.github.io/ArbInterp-Web/)  
**领域**: 视频理解 / 视频生成  
**关键词**: Video Frame Interpolation, Generative VFI, RoPE, Temporal Conditioning, Any-length Generation

## 一句话总结

ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。

## 研究背景与动机

视频帧插值（Video Frame Interpolation, VFI）是视频生成领域的基础任务，给定起始帧和结束帧，生成中间过渡帧。近年来，基于扩散模型的生成式 VFI 方法（如 DynamiCrafter、TRF、GI 等）展示了生成高质量中间帧的能力。

然而现有生成式 VFI 方法存在两个关键限制：

**固定插值数量**: 现有方法只能一次性生成固定数量的中间帧（如一次生成 7 帧或 15 帧），无法灵活调整生成帧率或总序列时长。例如，用户可能需要在两帧之间插 2 帧（2x），也可能需要插 31 帧（32x），现有方法难以统一处理。

**长序列不连贯**: 当需要大量插值帧时（如 32x 插值），直接生成长序列面临显存和质量问题。分段生成是自然的解决方案，但不同片段之间的时空连贯性难以保证，容易出现运动不自然、外观不一致等问题。

ArbInterp 的目标是构建一个统一的生成式 VFI 框架，同时解决"任意时间戳"和"任意长度"两个挑战。

## 方法详解

### 整体框架

ArbInterp 在视频扩散模型之上同时打通两个尺度：单段内用 TaRoPE 把每一帧绑定到一个连续的归一化时间戳 $t\in[0,1]$，从而精确生成任意位置的中间帧；序列层面则把超长插值拆成若干短段顺序生成，再用外观-运动解耦的条件把相邻段缝合起来。前者解决"任意时间戳"，后者解决"任意长度"，二者合起来让单一模型覆盖从 2x 到 32x 乃至更高的全部插值需求。

### 关键设计

**1. 时间戳感知旋转位置编码（TaRoPE）：让模型生成任意连续时间位置的帧**

DynamiCrafter 等方法把帧编号成离散整数位置 $0,1,2,\dots$ 再喂给位置编码，结果只能产出等间距的固定帧——想单独要一帧 $t=0.3$ 的画面就无能为力。TaRoPE 的做法是把时间维度 RoPE 的旋转角度直接挂到目标时间戳上：传统 RoPE 在第 $m$ 个位置对频率 $\theta_k$ 施加旋转角 $m\theta_k$，TaRoPE 则用归一化时间戳替换离散索引，使旋转角变为 $t\cdot\theta_k$。这样位置编码从"逐格跳跃"变成"沿时间轴连续滑动"，模型对 $t=0.25,0.5,0.75$ 与对任意非均匀时间戳的响应是同一套连续映射。几乎零额外参数，却让一个模型既能 2x 也能 32x，倍率不再需要逐个单独训练。

**2. 分段式帧合成：把超长序列拆成可生成的短段**

32x 插值意味着一次要产出 31 帧中间帧，直接生成这样的长序列会撞上显存上限、画质也随长度退化。ArbInterp 改为顺序生成若干短段，每段只合成有限帧，并把前一段的末帧作为后一段的起始边界条件接力下去。这一步本身只解决"算得动"的问题——真正的难点是段与段的接缝如何不露痕迹，这交给下一个设计处理。

**3. 外观-运动解耦条件：让相邻段在外观和运动上都平滑接续**

如果只拿前一段最后一帧当条件硬接下一段，容易出现外观逐渐漂移或运动在接缝处突然跳变。ArbInterp 把段间一致性拆成两条正交的信号分别注入：外观条件取前一段的端点帧（起、止帧）作为视觉锚点，强制新段保持与已生成内容一致的纹理与风格；运动条件则复用归一化时间戳承载的时间语义，约束全局速度与方向在跨段时连续过渡。外观管"长得像"、运动管"动得顺"，两路解耦后各自的可控性都比混在一起更强，这也是长序列不累积可见退化的关键。

### 损失函数 / 训练策略

论文摘要与项目页未公开损失的具体形式，从其扩散框架可推断核心是标准去噪扩散损失。训练时对时间戳做随机采样，让模型把"任意 $t$"当成常态来学；同时混合不同段长与不同插值倍率，增强对任意长度、任意倍率场景的泛化。

## 实验关键数据

### 评估基准

作者构建了两个综合性基准：

1. **MultiInterp Benchmark**: 多尺度帧插值评估（2x, 4x, 8x, 16x, 32x），测试模型在不同插值倍率下的泛化能力
2. **StreamInterp Benchmark**: 流式/长序列插值评估，测试分段生成时的时空连贯性

### 主实验

| 方法 | 2x 质量 | 8x 质量 | 16x 质量 | 32x 质量 | 评估维度 |
|------|---------|---------|----------|----------|----------|
| DynamiCrafter | 基线 | 基线 | 基线 | 基线 | 固定帧数限制 |
| TRF | 对比 | 对比 | 对比 | 对比 | 固定位置编码 |
| GI | 对比 | 对比 | 对比 | 对比 | 生成式插值 |
| **ArbInterp** | **最优** | **最优** | **最优** | **最优** | 统一模型全覆盖 |

根据论文摘要，ArbInterp 在所有插值场景下均超越先前方法，展现更高保真度和更无缝的时空连续性。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o TaRoPE（固定位置编码） | 质量下降 | 无法适应不同倍率 |
| w/o 外观-运动解耦条件 | 段间不连贯 | 运动跳变、外观漂移 |
| 不同段长度 | 影响效率和质量权衡 | 较短段更灵活但可能累积误差 |

### 关键发现

1. **TaRoPE 的连续可控性**: 单一模型可以处理 2x 到 32x 的任意插值，无需针对每个倍率单独训练
2. **解耦条件的必要性**: 如果只用前一段末帧做条件（不区分外观和运动），长序列会出现渐进的质量退化
3. **从定量到定性的全面优势**: 在多个对比方法中，ArbInterp 不仅指标更好，视觉效果也更自然流畅

## 亮点与洞察

- **TaRoPE 方案优雅**: 将连续时间戳编码到 RoPE 中是一个简洁但有效的设计，几乎零额外参数即可实现任意时间戳控制。这个思路可以推广到其他需要连续化离散位置的生成任务。
- **解耦设计思想**: 外观一致性和运动连贯性是两个正交的需求，将它们解耦处理比混合处理更加可控。这种思路在视频编辑、视频续写等任务中也有借鉴意义。
- **实用性强**: 任意倍率 + 任意长度 = 一个模型适配所有帧插值需求，大幅降低部署复杂度。
- **基准构建**: MultiInterp 和 StreamInterp 两个 benchmark 的构建也是一个贡献，有助于后续工作的公平比较。

## 局限与展望

1. **依赖扩散模型的生成速度**: 生成式方法逐帧去噪的推理速度远慢于传统光流方法（如 RIFE、IFRNet），高倍率插值时的推理时间可能成为瓶颈
2. **分段累积误差**: 虽然有解耦条件策略，但超长序列（如 64x, 128x）下是否会出现渐进退化尚不确定
3. **场景多样性**: 从项目页面看，演示主要集中在驾驶和运动场景，复杂遮挡、场景切换等极端情况的表现未知
4. **与非生成式方法的对比**: 论文主要对比生成式 VFI 方法，与传统高效的光流 VFI 方法（RIFE 等）的全面定量对比会更有说服力
5. **训练数据需求**: 生成式模型通常需要大规模视频数据预训练，训练成本和数据来源值得关注

## 相关工作与启发

- **与 DynamiCrafter 的关系**: DynamiCrafter 是生成式 VFI 的代表方法之一，但受限于固定帧位置编码。ArbInterp 的 TaRoPE 直接解决了这个根本限制。
- **与 TRF、GI 的关系**: TRF（Time-Reversal Fusion）和 GI（Generative Interpolation）也尝试了生成式插值，但同样受固定长度约束。
- **RoPE 的时间维度扩展**: 原始 RoPE 在 LLM 中用于序列位置编码，ArbInterp 将其扩展到视频的时间维度并支持连续值，这种跨领域的技术迁移值得注意。
- **对视频生成的启发**: TaRoPE 和解耦条件策略不仅适用于帧插值，也可能对视频预测、视频续写等任务有帮助。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[ACL 2026\] TeachMaster: Generative Teaching via Code](../../ACL2026/video_generation/teachmaster_generative_teaching_via_code.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](../../ICCV2025/video_generation/motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](../../CVPR2026/video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video](../../CVPR2026/video_generation/physvid_physics_aware_local_conditioning_for_generative_video_models.md)

</div>

<!-- RELATED:END -->
