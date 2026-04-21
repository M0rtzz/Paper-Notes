---
title: >-
  [论文解读] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models
description: >-
  [CVPR 2026][视频世界模型] 本文提出StEvo-Bench，一个包含225个任务的benchmark，通过在视频生成过程中插入遮挡或相机转向来测试视频世界模型能否在不可观测期间继续正确演化场景状态，发现当前最先进模型（包括Veo 3、Sora 2 Pro等）的成功率不到10%，揭示了视频模型将状态演化与观察高度耦合的根本问题。
tags:
  - CVPR 2026
  - 视频世界模型
  - 状态演化
  - 遮挡测试
  - benchmark
  - 物理一致性
---

# Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models

**会议**: CVPR 2026  
**arXiv**: [2603.13215](https://arxiv.org/abs/2603.13215)  
**代码**: [项目主页](https://glab-caltech.github.io/STEVOBench/)  
**领域**: 视频理解 / 世界模型  
**关键词**: 视频世界模型, 状态演化, 遮挡测试, benchmark, 物理一致性

## 一句话总结
本文提出StEvo-Bench，一个包含225个任务的benchmark，通过在视频生成过程中插入遮挡或相机转向来测试视频世界模型能否在不可观测期间继续正确演化场景状态，发现当前最先进模型（包括Veo 3、Sora 2 Pro等）的成功率不到10%，揭示了视频模型将状态演化与观察高度耦合的根本问题。

## 研究背景与动机
1. **领域现状**：随着视频生成技术的飞速发展，人们开始将视频模型称为"世界模型"，期望它们能够模拟真实世界的物理过程。当前的视频世界模型包括通用视频生成模型（Veo 3, Sora 2 Pro, WAN 2.2等）和相机控制视频模型（Genie 3, HunyuanWorld等）。
2. **现有痛点**：现有benchmark只评估了世界模型的子集能力——物理直觉benchmark（VideoPhy）只测物理正确性，一致性benchmark（MIND）只测记忆/一致性，没有任何benchmark同时评估"当观察被中断时，状态是否继续正确演化"。
3. **核心矛盾**：真实世界中，物理过程不依赖于观察者——水在被遮挡时仍在流动，冰在不被看见时仍在融化。但视频模型通过生成像素帧来"模拟世界"，其内部状态可能与像素观察高度耦合。
4. **本文目标** 设计一套系统性的测试框架来回答：视频世界模型能否将状态演化与观察解耦？
5. **切入角度**：通过两种方式中断观察——在场景中插入遮挡物（纸板/窗帘/关灯）或控制相机转向——然后检查恢复观察时状态是否正确演化。
6. **核心 idea**：用"遮住-揭开"的实验范式系统性地证明当前视频世界模型无法实现状态演化与像素观察的解耦。

## 方法详解

### 整体框架
StEvo-Bench的pipeline分三个阶段：(1) 任务构建——用初始图像+文本提示驱动视频模型生成含遮挡的演化过程；(2) 控制验证——检查遮挡控制和动作控制是否成功执行；(3) 演化评估——在通过控制验证的视频上评估状态进展、物理合理性和一致性。

### 关键设计

1. **任务构建体系（6类225个任务）**:

    - 功能：覆盖现实世界中常见的物理演化过程
    - 核心思路：每个任务由初始图像和文本提示指定，涵盖六个演化类别：连续过程（水流/融化）、运动学（抛物线/自由落体）、关系变化（多米诺骨牌）、因果变化（开关灯）、状态转换（燃烧/膨胀）、预期行为（人/动物的常识行为）。对视频生成模型使用场景内遮挡（纸板/关灯），对相机控制模型使用相机转向轨迹
    - 设计动机：这些任务反映了真实世界agent日常面对的物理事件，且覆盖了多种不同类型的物理过程，确保评估的全面性

2. **自动验证器流水线（5个独立验证器）**:

    - 功能：自动评估生成视频的多个维度，并解耦不同的失败模式
    - 核心思路：使用Gemini 3.1 Pro作为VLM判官，构建5个专家验证器：(a) 观察控制验证器——检查遮挡是否成功；(b) 动作控制验证器——检查动作是否正确发生；(c) 状态进展验证器——检查状态是否在遮挡期间继续演化（使用unanimous-vote集成n=3）；(d) 物理合理性验证器——检查演化是否物理正确（多数投票）；(e) 一致性验证器——检查遮挡前后场景是否时间一致
    - 设计动机：将验证分解为独立的专家模块有两个好处：(1) 可以细粒度诊断失败原因；(2) 每个验证器只问一个简单的yes/no问题，比让VLM一次性评估多个方面更可靠

3. **评估协议设计**:

    - 功能：两阶段评估确保严格性
    - 核心思路：先检查控制是否成功（观察控制+动作控制），失败的排除。通过控制验证的视频再评估演化成功率，要求三个标准同时满足：状态有进展、物理合理、一致性保持。最终的task success = control success × evolution success
    - 设计动机：如果控制本身失败（如遮挡没生效），则无法评判演化是否正确，必须分阶段排除

### 验证器可靠性验证
招募3名标注员在180个视频上标注，通过Accuracy、ROC-AUC和MRA（模型排名一致性）三个指标评估。结果显示验证器与人类的一致性等于甚至超过人类标注员之间的一致性，证明自动评估的可靠性。

## 实验关键数据

### 主实验（各模型在StEvo-Bench上的表现 %）

| 模型类型 | 模型 | Success | Progress | Physics | Coherence |
|---------|------|---------|----------|---------|-----------|
| 视频模型 | Veo 3 | 8.7 | 17.4 | 82.6 | 66.5 |
| 视频模型 | Sora 2 Pro | 8.1 | 13.1 | 85.5 | 69.7 |
| 视频模型 | WAN 2.2 | 0.9 | 7.7 | 52.0 | 58.4 |
| 相机控制 | Genie 3 | 0.0 | 2.9 | 15.2 | 27.3 |
| 相机控制 | HY-WorldPlay | 0.0 | 0.0 | 72.2 | 88.2 |
| 相机控制 | GEN3C | 0.0 | 0.0 | 30.6 | 82.4 |

### 消融实验（全观察 vs 遮挡控制对比，Veo3 + Sora2 Pro平均）

| 条件 | State Progress | Task Success |
|------|---------------|-------------|
| 全程观察 | 84.6% | 46.2% |
| 加入观察控制 | 17.4% | 12.4% |

### 关键发现
- **所有模型成功率 < 10%**：最好的Veo 3也只有8.7%的综合成功率，揭示了当前视频世界模型的根本局限
- **进展停止是最普遍的失败模式**：加入遮挡后，状态进展率从84.6%暴跌到17.4%，说明模型确实"看不到就不演化"
- **一致性是第二大失败模式**：即使闭源顶级模型，一致性也只有~67%，遮挡移除后物体外观经常发生突变
- **相机控制模型更严重**：几乎所有相机控制模型的状态进展率接近0%，存在强烈的静态场景偏置
- **演化与相机控制互斥**：当相机控制模型能生成动态时，反而无法执行相机转向，反之亦然
- **记忆模块无助于状态演化**：VMem虽然能完美回忆初始帧，但无法推进状态演化，记忆架构鼓励的是外观记忆而非状态演化
- **训练数据偏置是根本原因之一**：相机控制模型训练数据以静态场景渲染为主（3DGS重建/UE场景），缺乏含丰富物理动态的视频

## 亮点与洞察
- **评估范式设计极具创意**：用"遮住-揭开"的实验方法论测试世界模型的"理解"能力，类似认知科学中对婴儿的object permanence测试。这个范式可以迁移到评估任何声称"理解世界"的AI系统
- **失败模式的解耦分析很有价值**：不是简单地报告"失败率"，而是将失败分解为进展停止、物理错误、一致性丧失三类，每一类都指向不同的改进方向
- **对视频世界模型架构的深刻洞察**：全对全双向注意力可能不适合处理遮挡帧，因为遮挡帧不提供状态演化信息。这暗示需要新的注意力机制来区分"信息帧"和"非信息帧"

## 局限与展望
- StEvo-Bench仅有225个任务，可能不足以覆盖所有物理过程类型
- 自动验证器依赖Gemini 3.1 Pro，本身可能存在偏置
- 仅测试了遮挡/关灯/转向三种观察中断方式，其他方式（如模糊、雾化）未探索
- 未提出解决方案，仅是诊断性工作
- 对于how to fix的讨论较为表面，指出了训练数据偏置但没有具体的解决方案

## 相关工作与启发
- **vs VideoPhy/VideoPhy2**: 只测物理正确性，不测遮挡下的状态演化，StEvo-Bench是更全面的测试
- **vs MIND**: 只测静态场景的记忆一致性，StEvo-Bench测动态过程的持续演化
- **vs WorldScore**: 综合但简单设置，StEvo-Bench专注于"未观察期间的演化"这一关键维度
- 这篇论文对做视频世界模型的研究者是重要的参考，指明了改进方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地测试视频世界模型的状态演化-观察解耦能力，实验设计极具创意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖11个SOTA模型（开源+闭源），有验证器可靠性分析和人类标注对比
- 写作质量: ⭐⭐⭐⭐⭐ 论文讲故事清晰流畅，失败模式分析深入，insight丰富
- 价值: ⭐⭐⭐⭐⭐ 指出了视频世界模型的根本性局限，对该领域有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [Enhancing Out-of-Distribution Detection with Extended Logit Normalization](enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)
- [TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)

<!-- RELATED:END -->
