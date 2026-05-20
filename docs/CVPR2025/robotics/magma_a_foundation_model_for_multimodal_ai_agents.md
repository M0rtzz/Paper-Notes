---
title: >-
  [论文解读] Magma: A Foundation Model for Multimodal AI Agents
description: >-
  [CVPR 2025][机器人][多模态代理] Magma 通过在图像上标注可交互区域（Set-of-Mark）和在视频中标注运动轨迹（Trace-of-Mark），将 UI 截图、机器人数据和人类操作视频统一到同一个预训练框架中，使单一模型同时具备多模态理解和跨域动作预测能力…
tags:
  - "CVPR 2025"
  - "机器人"
  - "多模态代理"
  - "视觉-语言-动作模型"
  - "UI导航"
  - "机器人操控"
  - "空间智能"
---

# Magma: A Foundation Model for Multimodal AI Agents

**会议**: CVPR 2025  
**arXiv**: [2502.13130](https://arxiv.org/abs/2502.13130)  
**代码**: [https://microsoft.github.io/Magma](https://microsoft.github.io/Magma) (开源)  
**领域**: Agent / Robotics / 多模态VLM  
**关键词**: 多模态代理、视觉-语言-动作模型、UI导航、机器人操控、空间智能

## 一句话总结
Magma 通过在图像上标注可交互区域（Set-of-Mark）和在视频中标注运动轨迹（Trace-of-Mark），将 UI 截图、机器人数据和人类操作视频统一到同一个预训练框架中，使单一模型同时具备多模态理解和跨域动作预测能力，在 UI 导航和机器人操控上均取得 SOTA。

## 研究背景与动机

**领域现状**：当前基于视觉-语言-动作（VLA）模型的 AI Agent 通常针对特定领域单独训练——数字世界有 Pix2ACT、WebGUM、Ferret-UI 做 UI 导航，物理世界有 RT-2、OpenVLA 做机器人操控。这些模型各自为政，无法跨域迁移。

**现有痛点**：(1) 大多数模型为了学习任务特定的动作策略，牺牲了通用多模态理解能力，泛化性差；(2) 数字世界和物理世界的输入域（2D截图 vs 3D场景）和动作空间（2D坐标 vs 7-DoF）差异巨大，简单混合训练会互相干扰；(3) 现有的视觉-语言-动作数据量有限，而海量的无标注视频和图文数据无法直接用于动作预训练。

**核心矛盾**：verbal intelligence（语义理解）和 spatial-temporal intelligence（空间时序推理）之间存在天然鸿沟——语义标注是文本形式，动作标注是空间坐标形式，两者目标函数冲突，简单联合训练反而掉点。

**本文目标** 如何用一个统一框架让模型同时学会"看懂"（多模态理解）和"动手"（动作规划），并能从海量无标注视频中获取动作监督信号？

**切入角度**：作者观察到，无论是 UI 还是机器人，动作的本质都是"在某个位置做某件事"——把所有可操作对象用标记（Mark）在图像上标出来，动作预测就变成了"选标记"而非"预测坐标"，大幅简化了搜索空间并统一了不同域的输出格式。

**核心 idea**：用 Set-of-Mark（静态图标注可操作区域）和 Trace-of-Mark（视频中追踪运动轨迹）作为代理任务，将异构的图像/视频/机器人数据统一到同一个"标记预测"界面下联合预训练。

## 方法详解

### 整体框架
Magma 采用标准 VLM 架构：ConvNeXt-XXLarge 作为视觉编码器处理任意分辨率图像/视频帧，LLaMA-3-8B 作为语言骨干做自回归解码。输入为一系列视觉观测 + 文本任务描述，输出为文本 token（语义回答）或空间 token（动作坐标/标记编号）。核心创新不在架构，而在预训练数据的标注方式和代理任务设计。

预训练数据包含四类：UI 截图（270 万，SeeClick + Vision2UI）、机器人轨迹（940 万，Open-X-Embodiment）、人类操作视频（2500 万，Epic-Kitchen + Ego4d + SomethingV2 等）、图文对（120 万，ShareGPT4V + LLaVA-1.5），总计约 3900 万样本。

### 关键设计

1. **Set-of-Mark (SoM) 动作定位**:

    - 功能：在单帧图像上标注所有可交互区域，将"预测像素坐标"转化为"选择标记编号"
    - 核心思路：给定一张图像，用领域检测器（UI 用 DOM 树/Android View Hierarchy，视频/机器人用 CoTracker 提取的运动点）提取 K 个候选区域，给每个区域画框并标上数字编号，模型只需预测"选哪个标记"而非具体坐标。公式化为 $o_t^{mark} = \pi(\mathcal{I}_t^M, \text{task}, \text{ctx})$，输出是标记集合的子集
    - 设计动机：直接预测像素坐标在 UI 和机器人之间格式完全不同（2D vs 7-DoF），且搜索空间巨大；用标记编号统一后，不同域的动作预测被归约为同一种"选择"任务，使跨域联合训练成为可能

2. **Trace-of-Mark (ToM) 动作规划**:

    - 功能：在视频中追踪标记点的未来运动轨迹，将无标注视频转化为可用的"动作"训练数据
    - 核心思路：给定视频片段，用 CoTracker 在首帧放置 $s^2$ 个网格点并追踪它们在后续帧的位置，过滤掉运动幅度小于阈值 $\epsilon$ 的静止点（视为背景），保留运动显著的前景轨迹。对有相机运动的视频（尤其是 ego-centric），先做单应性变换去除全局运动再提取前景。最终用 K-Means 聚类前景/背景轨迹，随机采样代表性点作为标记
    - 设计动机：与预测下一帧图像相比，预测轨迹用极少的 token 就能捕捉长时间范围的动作相关物体动态，同时忽略无关的环境内容；更重要的是，这让海量无动作标注的视频也能产生有用的训练信号

3. **ConvNeXt 全局编码 + 统一分辨率处理**:

    - 功能：支持从普通自然图像到高分辨率 UI 截图（最高 2000px）的任意分辨率输入
    - 核心思路：使用 ConvNeXt-XXLarge 做视觉编码器，天然支持任意分辨率而无需裁切/拼接。对高分辨率图像直接全局编码，不做局部裁切和全局-局部融合
    - 设计动机：UI 截图分辨率极高且信息分布在全图，而机器人/视频帧分辨率相对较低，用 CNN 的天然分辨率灵活性统一处理，比 ViT 的 patch 方案更简洁高效

### 损失函数 / 训练策略
标准自回归 next-token prediction loss。所有输出（语义文本、UI 坐标、机器人 7-DoF 动作、标记编号、轨迹坐标）都统一为语言 token：2D 坐标归一化后量化为 256 bins 的文本字典，机器人动作用 LLM 词表中最后 256 个极少使用的 token 表示。全参数微调（包括视觉编码器和 LLM），学习率 1e-5，最多 3 个 epoch。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | Magma-8B | 之前最佳 | 提升 |
|--------|------|------|----------|------|
| ScreenSpot-Mobile | 文本/图标准确率 | 60.4/58.5 | 78.0/52.0 (SeeClick) | 图标+6.5% |
| ScreenSpot-Desktop | 文本/图标准确率 | 75.3/52.9 | 72.2/30.0 (SeeClick) | 图标+22.9% |
| ScreenSpot-Web | 文本/图标准确率 | 69.1/52.0 | 55.7/32.5 (SeeClick) | 图标+19.5% |
| VisualWebBench Ele-G | 准确率 | 96.3 | 67.5 (GPT-4V) | +28.8% |
| SimplerEnv-Google Robot | 成功率 | 52.3 | 34.2 (RT-1-X) | +18.1% |
| SimplerEnv-Bridge | 成功率 | 35.4 | 15.9 (Octo) | +19.5% |
| Mind2Web Cross-Website | Step SR | 45.4 | 36.5 (GPT-4V-OmniParser) | +8.9% |
| AITW Overall | 准确率 | 67.3 | 59.3 (SeeClick) | +8.0% |

### 消融实验

| 配置 | SS-Overall | VWB-Ele-G | SE-Bridge | SE-Google |
|------|---------|---------|---------|---------|
| Magma-8B (UI only) | 57.7 | 68.5 | - | - |
| Magma-8B (OXE only) | - | - | 22.2 | 35.7 |
| Magma-8B (UI+OXE, w/o SoM+ToM) | 56.2 | 89.1 | 17.5 | 31.5 |
| Magma-8B (Full, w/o SoM+ToM) | 57.4 | 90.1 | 17.7 | 37.5 |
| Magma-8B (Full, w/ SoM+ToM) | **61.4** | **96.3** | **35.4** | **52.3** |

### 关键发现
- **SoM+ToM 是核心**：没有 SoM/ToM 时，简单混合 UI 和机器人数据反而导致两端性能下降（互相干扰）；加上 SoM/ToM 后性能大幅提升，UI +4pp，机器人 +17.8pp/+14.8pp
- **视频数据通过 ToM 才有用**：没有 ToM 的情况下加入视频数据仅带来微小提升（视频文本描述只增强 verbal intelligence），但 ToM 让视频成为空间智能的重要训练来源
- **空间推理能力显著增强**：在 VSR (65.1%) 和 SpatialEval 上大幅超越同级别 VLM，在 BLINK (41.0%) 上接近 GPT-4o 水平
- **Few-shot 机器人迁移极强**：在 LIBERO 上仅需 10 条轨迹微调即可大幅超过 OpenVLA，成功率翻倍

## 亮点与洞察
- **SoM/ToM 的统一接口设计非常巧妙**：通过"在图像上画标记"这一简单操作，将语义理解和动作预测的 gap 弥合——标记既是视觉 grounding 又是动作锚点，让不同域的数据产生正向迁移而非冲突
- **无标注视频变废为宝**：ToM 用点追踪+运动过滤就能从任意视频中提取动作级监督信号，这意味着海量 YouTube 视频都可以成为 Agent 预训练数据，scaling law 的天花板被大幅提高
- **ConvNeXt 代替 ViT 的思路**：面对 UI 截图的超高分辨率需求，不做复杂的裁切拼接，直接用 CNN 的分辨率无关性解决，简洁有效。这个思路可以迁移到任何需要处理不同分辨率输入的多模态任务

## 局限与展望
- 当前仅支持单步动作预测，长序列多步规划能力依赖外部循环，缺乏内在的 planning 能力
- ToM 依赖 CoTracker 的质量，对遮挡严重或快速运动的场景可能失效（虽然论文在 YouCook2 上验证了 precision=0.89，但更复杂场景未必稳健）
- 机器人端用最后 256 个 token 表示动作是一种 hack，无法表达连续动作空间的精细控制
- 预训练数据中 UI 和机器人数据占比相对较小（~15%），如果进一步 scale up 这两类数据可能获得更大收益
- 没有在 3D 导航任务（如 Habitat、AI2Thor）上验证，跨域泛化的边界不清楚

## 相关工作与启发
- **vs OpenVLA**: OpenVLA 只在 OXE 机器人数据上训练，domain-specific、泛化差；Magma 用异构数据 + SoM/ToM 学到了更通用的空间智能，零样本即超越 OpenVLA 在 SimplerEnv 上的表现（成功率翻倍）
- **vs SeeClick**: SeeClick 基于 Qwen-VL 专做 UI grounding；Magma 在 ScreenSpot 的图标类别上大幅领先（+19.5%），说明 ToM 从视频中学到的空间推理能力对 UI 理解也有帮助
- **vs GPT-4V-OmniParser**: GPT-4V 配合 OmniParser 在 ScreenSpot 上表现很好，但 Magma 在 VisualWebBench 元素定位上以 96.3% vs 67.5% 碾压，说明预训练的空间感知远超提示工程
- 这篇工作对"如何从视频中提取动作信号"给出了一个优雅的方案，可以启发视频理解和具身智能的交叉研究

## 评分
- 新颖性: ⭐⭐⭐⭐ SoM 已有前人工作，ToM 是自然延伸；但将两者统一到跨域预训练的框架设计很新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 UI/机器人/VL理解/空间推理四大类任务，消融充分，还有真实机器人验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation 到方法到实验一气呵成
- 价值: ⭐⭐⭐⭐⭐ 微软开源的统一 Agent 基础模型，具有很强的实用价值和后续影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](../../ICCV2025/robotics/adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)
- [\[ICLR 2026\] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](../../ICLR2026/robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](../../NeurIPS2025/robotics/mip_against_agent_malicious_image_patches_hijacking_multimod.md)

</div>

<!-- RELATED:END -->
