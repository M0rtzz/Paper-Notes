---
title: >-
  [论文解读] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL
description: >-
  [ICLR 2026][目标检测][AI生成视频检测] VidGuard-R1 是首个采用 GRPO（Group Relative Policy Optimization）强化学习微调 MLLM 的视频真伪检测器，通过构建 14 万无快捷方式的真/假视频对数据集，并设计时序伪影奖励和扩散步数质量奖励两种专用奖励机制，在自建数据集上达到 86.17% 准确率，在 GenVidBench 和 GenVideo 基准上实现 95%+ 的 SOTA 零样本检测性能，同时生成可解释的思维链推理。
tags:
  - ICLR 2026
  - 目标检测
  - AI生成视频检测
  - MLLM推理
  - GRPO
  - 时序伪影
  - 可解释取证
---

# VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL

**会议**: ICLR 2026  
**arXiv**: [2510.02282](https://arxiv.org/abs/2510.02282)  
**代码**: [项目页面](https://vidguard-r1.github.io/)  
**领域**: 视频真伪检测/推理  
**关键词**: AI生成视频检测, MLLM推理, GRPO, 时序伪影, 可解释取证

## 一句话总结

VidGuard-R1 是首个采用 GRPO（Group Relative Policy Optimization）强化学习微调 MLLM 的视频真伪检测器，通过构建 14 万无快捷方式的真/假视频对数据集，并设计时序伪影奖励和扩散步数质量奖励两种专用奖励机制，在自建数据集上达到 86.17% 准确率，在 GenVidBench 和 GenVideo 基准上实现 95%+ 的 SOTA 零样本检测性能，同时生成可解释的思维链推理。

## 研究背景与动机

**领域现状**：AI 视频生成模型（Sora、HunyuanVideo、Wan 等）的视频质量飞速提升，生成视频与真实视频的界限日益模糊，带来虚假信息传播、隐私侵犯、诈骗等严重社会风险，亟需准确且可解释的检测工具。

**现有痛点**：

1. **传统检测器局限性大**：早期 DeepFake 检测器仅针对面部伪造，无法泛化到开放域多场景视频；时空一致性方法容易被后处理绕过
2. **MLLM 直接应用效果差**：GPT-4o 等强大 MLLM 直接用于视频真伪判断时准确率仅约 57%，与随机猜测相差无几
3. **SFT 微调推理能力弱**：SFT 虽提升了检测准确率，但模型无法生成"为什么是假的"这样有意义的解释——推理能力不足
4. **现有数据集有快捷方式**：GenVideo、GenVidBench 等基准中真假视频在分辨率、帧率、码率、时长上存在系统性差异，模型利用元数据而非视觉真实性进行判断

**核心矛盾**：需要模型既能准确检测又能深度推理"假在哪里"，而 SFT 只能教会格式无法激发探索性推理。

**本文方案**：引入 GRPO 强化学习框架，通过多路径推理采样和组内排名，鼓励模型自主发现物理不一致性，并设计两种专用奖励信号引导时序推理和质量感知。

## 方法详解

### 整体框架

VidGuard-R1 采用两阶段训练：**阶段一** SFT 初始化（在 30K 视频上学习 CoT 推理格式）→ **阶段二** RL 强化推理（在 100K 视频上用 GRPO/DPO 进一步提升推理和检测能力）。基座模型为 Qwen2.5-VL-7B。

### 关键设计一：无快捷方式的训练数据构建

现有基准的致命缺陷是真假视频在低级特征上存在系统性差异（真实视频 >10 秒、假视频 <4 秒等），导致模型走捷径。本文构建 140K 视频数据集消除这些偏差：

- **真实视频来源**：InternVid（55K）+ ActivityNet（15K）
- **生成视频**：用 HunyuanVideo-I2V（50K）+ CogVideoX-5B（20K）从真实视频的首帧+文本描述生成对应假视频
- **标准化**：统一为 49 帧、8 FPS、720×480 分辨率、YUV420p 格式
- **CoT 标注**：用 Qwen-2.5-VL-72B 生成包含动作一致性、光照一致性、纹理伪影、物理合理性等维度的推理注释

### 关键设计二：GRPO-TA（时序伪影增强奖励）

标准 GRPO 善于利用局部视觉线索（像素失真、光照异常），但忽略时序不一致性。GRPO-TA 通过注入时序伪影来强化时序推理：

- **操作方式**：对视频随机执行片段重复或帧序反转（基于高斯分布选择操作区域）
- **非对称奖励设计**：检测到真实视频被篡改后的时序异常给予更高奖励 $\alpha_1 = 0.5$（更难发现），生成视频被篡改给较低奖励 $\alpha_2 = 0.3$（更易发现）
- **条件激活**：仅当原始视频预测正确且篡改视频的组准确率 $\tilde{p} > \mu = 0.8$ 时才添加额外奖励

奖励函数：

$$r_i^{\text{GRPO-TA}} = \begin{cases} r_i^{\text{GRPO}} + w_i, & \text{if } o_i \text{ correct and } \tilde{p} > \mu \\ r_i^{\text{GRPO}}, & \text{otherwise} \end{cases}$$

### 关键设计三：GRPO-Q（质量演化奖励）

利用扩散模型的内在属性——不同反向扩散步数产生不同质量的视频——来训练模型进行细粒度质量感知：

- **数据增强**：对 12K 真实视频，用扩散步数 10-50 生成 5 个质量级别（20%, 40%, 60%, 80%, 95%）的变体，共 72K 样本/模型
- **分级奖励**：精确匹配质量级别得满分 $\delta = 1$，不精确匹配按预测与真实步数的距离给予部分奖励 $g(o_i, y_i) = \delta \cdot (1 - |s(o_i) - s(y_i)|)$
- **超越二值判断**：模型不仅判真假，还能估计生成视频的质量退化程度

## 实验结果

### 主实验：自建数据集上的检测性能

| 方法 | 类型 | CogVideoX 准确率(%) | HunyuanVideo 准确率(%) |
|------|------|:---:|:---:|
| I3D | CNN | 64.78 | 62.13 |
| SlowFast | CNN | 77.87 | 77.03 |
| TimeSformer | Transformer | 78.53 | 74.55 |
| VideoSwin | Transformer | 76.81 | 79.71 |
| GPT-4o | MLLM | 56.81 | 57.42 |
| Qwen2.5-VL-7B | MLLM | 50.95 | 52.83 |
| VidGuard-R1 (CoT/SFT) | MLLM | 66.18 | 63.19 |
| VidGuard-R1 (DPO) | MLLM | 79.13 | 80.88 |
| VidGuard-R1 (GRPO) | MLLM | 81.30 | 81.90 |
| VidGuard-R1 (GRPO-TA) | MLLM | 82.17 | 83.72 |
| VidGuard-R1 (GRPO-Q) | MLLM | **84.32** | **86.17** |

关键观察：(1) Qwen2.5-VL-7B/GPT-4o 直接应用接近随机（~50-57%）；(2) SFT 将准确率提升至 66%，但仍不如传统视频模型；(3) GRPO 在 DPO 基础上再提 ~2%；(4) GRPO-TA 和 GRPO-Q 分别再提 ~2% 和 ~5%，证实专用奖励的有效性。

### 跨基准零样本泛化

| 方法 | GenVidBench 均值(%) | GenVideo 最优指标 |
|------|:---:|:---:|
| MViT V2 | 79.90 | - |
| GPT-4.1 mini | 59.62 | - |
| VidGuard-R1 (GRPO, 零样本) | 96.37 | F1: 0.97 |
| VidGuard-R1 (GRPO, 微调) | **97.53** | F1: **0.98** |

VidGuard-R1 在 GenVidBench 上零样本达到 96.37%，超过先前 SOTA（MViT V2, 79.90%）约 17 个百分点；在 GenVideo 上 F1 也大幅领先。微调后进一步提升至 97.53%。

### 消融实验：各训练阶段贡献

| 训练配置 | CogVideoX | HunyuanVideo | 增益来源 |
|----------|:---:|:---:|----------|
| SFT (CoT) | 66.18 | 63.19 | 基础推理格式 |
| + DPO | 79.13 | 80.88 | 偏好对齐 +15% |
| + GRPO | 81.30 | 81.90 | 组排名探索 +2% |
| + GRPO-TA | 82.17 | 83.72 | 时序推理 +1.8% |
| + GRPO-Q | 84.32 | 86.17 | 质量感知 +2.5% |

每个阶段都带来明确且一致的提升，其中从 SFT 到 DPO 的跳跃最大（~15%），说明偏好学习是关键；GRPO-Q 的质量分级奖励带来最强的增量提升。

## 论文评价

### 优点

1. **首创性**：首次将 GRPO 强化学习应用于 AI 生成视频检测，建立了"检测 + 解释"的范式
2. **奖励设计巧妙**：GRPO-TA 的非对称时序伪影奖励和 GRPO-Q 的扩散步数质量奖励都利用了生成模型的内在特性，针对性强
3. **数据集严谨**：通过标准化消除快捷方式，确保模型学习视觉真实性而非元数据差异
4. **泛化能力突出**：零样本即在 GenVidBench/GenVideo 上达到 95%+，远超之前所有方法

### 不足

1. 基座模型固定为 Qwen2.5-VL-7B，未验证在其他 MLLM 上的通用性
2. GRPO-Q 需要生成多种扩散步数的视频，数据构建成本高
3. 生成模型快速迭代，检测方法的持久有效性不确定

### 评分

⭐⭐⭐⭐ — 将推理型 RL 引入视频取证领域的开创性工作，方法设计精巧、实验充分，为可解释的 AI 安全检测提供了强有力的范式。

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Detection of AI-Generated Images](../../ECCV2024/object_detection/zero-shot_detection_of_ai-generated_images.md)
- [Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/object_detection/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](../../CVPR2026/object_detection/saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)
- [ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](../../ICCV2025/object_detection/chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)

<!-- RELATED:END -->
