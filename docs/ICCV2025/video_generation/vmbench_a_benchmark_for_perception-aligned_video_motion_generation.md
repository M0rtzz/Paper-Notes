---
title: >-
  [论文解读] VMBench: A Benchmark for Perception-Aligned Video Motion Generation
description: >-
  [ICCV 2025][视频理解][Video Motion Evaluation] 提出 VMBench——首个面向视频运动质量评估的综合基准，包含五维感知对齐运动指标（PMM）和元信息引导的运动提示生成框架（MMPG），覆盖 969 类运动类型，在 Spearman 相关系数上比现有方法平均提升 35.3%。
tags:
  - ICCV 2025
  - 视频理解
  - Video Motion Evaluation
  - Human Perception Alignment
  - text-to-video
  - benchmark
  - Motion Quality
---

# VMBench: A Benchmark for Perception-Aligned Video Motion Generation

**会议**: ICCV 2025  
**arXiv**: [2503.10076](https://arxiv.org/abs/2503.10076)  
**代码**: https://github.com/GD-AIGC/VMBench (有)  
**领域**: 视频理解  
**关键词**: Video Motion Evaluation, Human Perception Alignment, text-to-video, benchmark, Motion Quality

## 一句话总结

提出 VMBench——首个面向视频运动质量评估的综合基准，包含五维感知对齐运动指标（PMM）和元信息引导的运动提示生成框架（MMPG），覆盖 969 类运动类型，在 Spearman 相关系数上比现有方法平均提升 35.3%。

## 研究背景与动机

文本到视频（T2V）生成模型发展迅速，但**运动质量评估**仍然是一个重大挑战。现有评估方法存在两个核心问题：

**问题一：运动指标与人类感知不对齐。** 当前运动评估主要局限于运动平滑度（如 VBench 用帧插值模型衡量），无法捕捉更复杂的运动缺陷——时空不一致、违反物理定律、物体变形消失等。Feature-based 指标（FID、FVD）忽略时序连贯性；Rule-based 指标（VBench）设计主观且片面；MLLM-based 方法（VideoScore）评分粒度过粗，且训练偏差导致忽略细微运动违规。

**问题二：运动提示多样性不足。** 现有 benchmark 的运动提示类型有限且简单，无法全面探索模型的运动生成能力。VMBench 覆盖 969 类运动——远超其他所有 benchmark。

本文的切入点是：**模拟人类感知运动的层次化过程**——先构建场景的整体理解（常识判断、运动平滑度），再关注运动细节（物体完整性、运动幅度、时序连贯性），从而设计出与人类感知真正对齐的评估指标。

## 方法详解

### 整体框架

VMBench 由两大核心组件构成：(1) **感知驱动运动评估指标（PMM）**——五个维度的细粒度指标；(2) **元信息引导运动提示生成（MMPG）**——覆盖六大运动模式的结构化提示库。两者结合构成完整的运动评估 benchmark。

### 关键设计

1. **常识遵循分数（Commonsense Adherence Score, CAS）**:

    - 功能：评估视频是否符合人类常识和物理规律
    - 核心思路：收集 10k 生成视频 → 通过 VideoReward 模型进行系统性成对比较建立感知基线 → 将偏好分数离散化为五级标签（Bad/Poor/Fair/Good/Perfect）→ 训练 VideoMAEv2（ViT-Giant 骨干）作为分类器。最终 CAS 通过 Mean Opinion Score 计算：$\text{CAS} = \sum_{i=1}^{5} p_i G(i)$，其中 $p_i$ 是各类别概率，$G(i)$ 将类别映射为质量权重
    - 设计动机：现有方法缺乏对整体场景合理性的判断。CAS 在消融实验中移除后准确率下降最大（-6.5%），证明其核心地位

2. **运动平滑度分数（Motion Smoothness Score, MSS）**:

    - 功能：检测时序伪影和运动模糊
    - 核心思路：利用 Q-Align 美学评分检测帧间质量退化，当连续帧间评分下降超过自适应阈值时判定为伪影帧。$\text{MSS} = 1 - \frac{1}{T}\sum_{t=2}^T \mathbb{I}(\Delta Q_t > \tau_s(t))$，其中 $\Delta Q_t = Q(f_{t-1}) - Q(f_t)$，$\tau_s(t)$ 是场景自适应阈值
    - 设计动机：先前指标用光流偏差或简单运动模型衡量平滑度，与人类感知脱节。自适应阈值允许高运动场景有更高的质量退化容忍度

3. **物体完整性分数（Object Integrity Score, OIS）**:

    - 功能：检测运动中物体的不合理变形
    - 核心思路：用 MMPose 检测主体关键点，分析帧间骨骼长度和关节角度变化，判断是否违反解剖学约束。$\text{OIS} = \frac{1}{F \cdot K}\sum_{f=1}^{F}\sum_{k=1}^{K}\mathbb{I}(\mathcal{D}_f^{(k)} \leq \tau^{(k)})$
    - 设计动机：已有方法（如 DINO 语义一致性）关注语义级别，忽视人眼敏感的形状变形问题

4. **感知幅度分数（Perceptible Amplitude Score, PAS）**:

    - 功能：在分离相机运动后估计主体运动幅度
    - 核心思路：GroundingDINO 定位主体 → GroundedSAM 稳定跟踪 → CoTracker 追踪关键点位移 → 根据场景类型设置感知阈值。$\text{PAS} = \frac{1}{T}\sum_{t=1}^T \min(\frac{\bar{D}_t}{\tau_s}, 1)$
    - 设计动机：传统 RAFT 光流将相机运动混入整体运动，导致估计偏高

5. **时序连贯性分数（Temporal Coherence Score, TCS）**:

    - 功能：检测物体异常消失/重现
    - 核心思路：GroundedSAM2 实例分割跟踪 → 对不连续存在的物体用 CoTracker 二次验证 → 规则过滤合理遮挡/入出画面场景。$\text{TCS} = 1 - \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\mathcal{A}_i \wedge \neg \mathcal{R})$
    - 设计动机：现有 CLIP/DINO 帧间余弦相似度无法区分自然运动和突变

6. **元信息引导提示生成（MMPG）**:

    - 功能：生成覆盖六大运动模式的多样化提示
    - 核心思路：三阶段流程——(a) 从 VidProm、Place365 等数据集提取主体/场景/动作元数据；(b) GPT-4o 随机组合元数据生成约 50k 候选提示并自验证；(c) DeepSeek-R1 + 人工联合验证，最终筛选出 1050 条高质量提示
    - 设计动机：确保物理合理性和动作多样性，六大运动模式包括流体动力学、生物运动、机械运动、天气现象、集体行为、能量传递

### 评估设置

评估了六个开源 T2V 模型（OpenSora、CogVideoX、OpenSora-Plan、Mochi 1、HunyuanVideo、Wan2.1），每个模型生成 1050 个视频。随机抽取 1200 个视频进行人工标注验证。

## 实验关键数据

### 主实验（指标与人类感知的 Spearman 相关性 ρ×100）

| 方法 | Avg. | CAS | MSS | OIS | PAS | TCS |
|------|------|-----|-----|-----|-----|-----|
| SSIM (Rule) | 1.6 | -0.9 | -12.1 | 8.3 | 17.8 | -4.8 |
| RAFT (Rule) | -1.7 | -0.7 | -17.0 | -16.6 | 47.7 | -21.9 |
| CLIP (Rule) | 15.0 | 21.5 | 36.5 | 31.7 | -42.7 | 28.0 |
| Dover Technical (Rule) | 20.6 | 40.2 | 32.6 | 34.5 | -6.2 | 2.2 |
| InternVideo2.5 (MLLM) | 26.9 | 22.7 | 21.9 | 29.6 | 44.3 | 15.8 |
| **PMM (Ours)** | **62.2** | **69.9** | **77.1** | **65.8** | **65.2** | **54.5** |

### 消融实验（移除单指标对预测准确率的影响）

| 配置 | 准确率(%) | 说明 |
|------|----------|------|
| 完整 PMM（5维全部） | 70.6 | 基准 |
| 移除 TCS | 66.9 | -3.7% |
| 移除 PAS | 68.7 | -1.9% |
| 移除 OIS | 65.2 | -5.4% |
| 移除 MSS | 64.6 | -6.0% |
| 移除 CAS | 64.1 | **-6.5%，影响最大** |
| 仅 CAS | 58.9 | 起点 |
| CAS + MSS | 66.1 | +7.2% |
| CAS + MSS + OIS | 67.3 | +1.2% |

### 关键发现

- PMM 在所有五个维度上均大幅领先 Rule-based 和 MLLM 方法。平均 Spearman 相关系数 62.2% vs 最佳 MLLM（InternVideo2.5）26.9%
- CAS（常识遵循）对整体评估贡献最大，移除后准确率降幅最大
- PAS（感知幅度）与其他维度呈负相关（ρ=-0.18 与 OIS），揭示了运动幅度和结构完整性之间的权衡关系
- Wan2.1 在 PMM 综合评分中最优（78.4%），表现最真实

## 亮点与洞察

- **首次从人类感知角度评估运动质量**：五维指标的设计严格遵循认知科学中运动感知的层次化过程（全局解析→局部细节）
- **指标的独立性和互补性**：PAS 与结构/时序指标的负相关性挑战了传统光流评估框架的假设，说明分离运动幅度评估的必要性
- **MLLM 在运动评估上的局限**：即使最强的 InternVideo2.5 平均相关性也仅 26.9%，说明通用多模态模型无法替代专门的运动评估工具

## 局限与展望

- 评估指标对齐的是**一般性**人类感知，无法完全覆盖个体差异和文化差异导致的感知偏好
- OIS 目前依赖关键点检测（MMPose），对非人/非动物物体的完整性评估有限
- TCS 的规则过滤可能无法覆盖所有合理的物体消失场景
- 1050 条提示虽然覆盖 969 类运动，但每类运动的样本较少
- 未涵盖多物体交互的复杂运动场景评估

## 相关工作与启发

- **VBench**：包含 RAFT、CLIP、DINO、AMT 等运动指标，但这些规则指标与人类感知的相关性极低（平均 -1.7~15.0%）
- **EvalCrafter**：使用 Dover Technical 和 Warping Error，同样对齐度有限
- **VideoScore / VideoPhy**：MLLM-based 方法，粒度过粗且偏差放大
- **启发**：将感知科学理论直接转化为计算指标是一个被低估的研究方向

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation](../../ACL2025/video_generation/vidcapbench_a_comprehensive_benchmark_of_video_captioning_for_controllable_text-.md)
- [MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

<!-- RELATED:END -->
