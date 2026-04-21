---
title: >-
  [论文解读] Semantic Audio-Visual Navigation in Continuous Environments
description: >-
  [CVPR 2026][语音][音视觉导航] 本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。
tags:
  - CVPR 2026
  - 语音
  - 音视觉导航
  - 连续环境
  - 记忆增强
  - 目标推理
  - 具身智能
---

# Semantic Audio-Visual Navigation in Continuous Environments

**会议**: CVPR 2026  
**arXiv**: [2603.19660](https://arxiv.org/abs/2603.19660)  
**代码**: https://github.com/yichenzeng24/SAVN-CE  
**领域**: 具身导航 / 音频视觉  
**关键词**: 音视觉导航, 连续环境, 记忆增强, 目标推理, 具身智能

## 一句话总结
本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。

## 研究背景与动机
1. **领域现状**：音视觉导航（AVN）让具身智能体利用听觉和视觉线索在未知环境中导航到发声目标。语义音视觉导航（SAVN）进一步要求目标是语义上有意义的物体（如"椅子在吱嘎作响"），而非任意位置。
2. **现有痛点**：现有方法依赖预计算的房间脉冲响应（RIR），需要TB级存储空间，且智能体只能在离散网格点上移动（1米分辨率、4个固定朝向），严重限制了任务的真实性。
3. **核心矛盾**：离散环境使得观测在空间上不连续，智能体无法自由探索；而连续环境下目标声音可能间歇性静默甚至完全停止发声，导致目标信息丢失。
4. **本文目标** (a) 如何在连续环境中实现自由移动的音视觉导航？(b) 当目标声音消失时，如何维持稳定的目标表示？(c) 如何同时推断目标的空间位置和语义类别？
5. **切入角度**：作者观察到自运动线索（上一步动作+当前位姿）可以推断目标相对位置的动态变化，而通过情景记忆可以在声音消失后保持目标表示的时间连续性。
6. **核心 idea**：用记忆增强的 Transformer 编码器融合双耳音频、自运动线索和情景记忆，实现声音消失后的持续目标追踪。

## 方法详解

### 整体框架
MAGNet 由三个模块组成：(1) **多模态观测编码器**，将 RGB-D 图像、双耳音频、动作和位姿编码为紧凑嵌入并存入场景记忆；(2) **记忆增强目标描述网络（GDN）**，融合双耳特征、自运动信息和情景记忆，推断目标的空间-语义表示；(3) **上下文感知策略网络**，基于 Transformer 编码-解码架构，利用场景记忆预测下一步动作。

### 关键设计

1. **多模态观测编码器**:
    - 功能：将每个时间步的多模态观测编码为统一表示
    - 核心思路：分别用 ResNet-18 编码 RGB 和深度图，用嵌入层编码前一动作，用全连接层编码归一化位姿 $[x/d, y/d, \sin\theta, \cos\theta, t/t_{max}]$，用 STFT 将双耳波形转换为复数谱图并提取4通道声学特征（平均幅度谱、ICP相位差正弦/余弦、ILD）后通过3层卷积编码。所有嵌入拼接构成观测表示，并维护一个容量为 $N_s=150$ 的滑动窗口场景记忆。
    - 设计动机：连续环境中观测更密集（0.25s步长），需要高效编码多模态信息并保留长期历史。

2. **记忆增强目标描述网络（GDN）**:
    - 功能：在目标声音间歇或完全消失时维持稳定的目标表示
    - 核心思路：每步将双耳音频嵌入、动作嵌入、位姿嵌入通过 MLP 融合为 $m_t$，存入容量 $N_g=128$ 的情景记忆。记忆序列加入位置编码后送入因果 Transformer 编码器，产生两个输出：(a) 目标嵌入 $e_t^G$ 供策略网络使用；(b) ACCDDOA 格式的目标描述 $y_{ct} = [a_{ct}R_{ct}, d_{ct}]$（包含声音活动状态、方向向量、归一化距离）用于训练时的监督损失。
    - 设计动机：自运动线索可以精确推断目标方位变化（TurnLeft/TurnRight 使方位角±15°，MoveForward 影响方位和距离），而情景记忆保证了时间连续性。细粒度动作空间（0.25m前进/15°转弯）限制了步间位置变化，确保目标追踪的稳定性。

3. **上下文感知策略网络**:
    - 功能：基于历史和当前观测预测下一步动作
    - 核心思路：Transformer 编码器处理场景记忆 $M_{s,t}$ 捕获时间依赖，解码器以当前观测嵌入为查询、编码后的记忆为键值，生成上下文感知的潜状态 $s_t$。该状态分别送入 actor 和 critic 全连接层，预测动作分布和状态价值。
    - 设计动机：利用 Transformer 的注意力机制充分利用历史信息，使策略在部分可观测连续环境中做出连贯决策。

### 损失函数 / 训练策略
- GDN 使用 MSE 损失在线监督训练，利用 oracle ACCDDOA 标签，采用因果注意力防止未来信息泄露
- 策略网络使用 DD-PPO 训练，遵循 SAVi 的两阶段范式
- 奖励：成功 +10，到目标的测地距离变化的中间奖励，每步 -0.01 时间惩罚
- 每次迭代 150 步 rollout，在 128 CPU + 4 A800 GPU 上训练约 14 天

## 实验关键数据

### 主实验

| 方法 | SR↑ | SPL↑ | SNA↑ | DTG↓ | SWS↑ |
|------|-----|------|------|------|------|
| AV-Nav | 21.3 | 17.8 | 13.1 | 10.7 | 4.0 |
| SMT+Audio | 24.8 | 21.0 | 16.8 | 10.1 | 5.3 |
| SAVi | 25.6 | 21.2 | 17.3 | 10.1 | 6.0 |
| **MAGNet** | **37.7** | **32.9** | **27.4** | **8.0** | **10.6** |
| Oracle1 | 41.4 | 37.8 | 31.0 | 6.3 | 13.0 |
| Oracle2 | 75.0 | 63.7 | 51.9 | 4.2 | 48.4 |

Clean 环境下，MAGNet 相比 SAVi 提升 12.1% SR（绝对值），SWS 提升 4.6%。

### 消融实验

| 配置 | SR↑ | SPL↑ | SWS↑ |
|------|-----|------|------|
| w/o GDN | 32.4 | 27.9 | 6.3 |
| GDN w/o Memory | 33.9 | 29.8 | 8.9 |
| GDN w/o Self-motion | 34.3 | 30.4 | 7.8 |
| Full MAGNet | **37.7** | **32.9** | **10.6** |

### 关键发现
- 去掉 GDN 后 SR 降 5.3%，但仍超过所有 baseline，说明策略网络本身已很强
- 情景记忆的贡献（+3.8% SR）大于自运动线索（+1.9% SR），但两者结合效果最佳
- 干扰环境下所有方法性能下降，MAGNet 虽然 SR 仅从 37.7 降到 19.3，但 DSR（误触干扰源率）最高达 7.8%，说明声学相似干扰是主要瓶颈
- Oracle2 vs Oracle1 的巨大差距（75.0 vs 41.4）表明声音消失后目标信息丢失是核心挑战

## 亮点与洞察
- **ACCDDOA 格式统一描述目标**：将声音活动状态、到达方向单位向量和距离融合为一个紧凑的向量表示，非常优雅地将 SELD 任务与导航结合起来
- **自运动线索的物理可解释性**：TurnLeft/TurnRight 精确改变目标方位角 ±15°，这种显式的几何关系使得网络更容易学习声音消失后的目标位置更新
- **连续环境数据集构建**：声音起始时间从 [0,5]s 均匀采样，持续时间服从高斯分布（均值 15s，标准差 9s），测试集平均 oracle 动作数 78.49（离散设定仅 26.52），大幅提升了任务难度

## 局限与展望
- Oracle2 与 MAGNet 的巨大差距（75.0 vs 37.7）表明 GDN 仍有很大提升空间
- 干扰环境下 DSR 最高，说明对声学相似干扰的区分能力不足
- 仅支持单目标导航，未来可扩展到多目标、动态目标场景
- 训练成本高（14天，128 CPU + 4 GPU），限制了大规模实验

## 相关工作与启发
- **vs SAVi**: SAVi 依赖加权因子 λ 聚合历史估计，本文用 Transformer 情景记忆替代，在声音消失后表现更好
- **vs VLN-CE**: VLN-CE 有语言指令提供明确目标，SAVN-CE 需要从部分感知中推断目标，更具挑战性

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续环境 + 记忆增强 GDN 的组合较为新颖，但各组件设计较为standard
- 实验充分度: ⭐⭐⭐⭐⭐ 消融全面，GDN评估、因素分析、轨迹可视化俱全
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，任务定义明确
- 价值: ⭐⭐⭐⭐ 为音视觉导航提供了更真实的连续环境基准

<!-- RELATED:START -->

## 相关论文

- [Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/audio_speech/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation](../../CVPR2025/audio_speech/wav2sem_plug-and-play_audio_semantic_decoupling_for_3d_speech-driven_facial_anim.md)
- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](../../ICLR2026/audio_speech/redteamcua_adversarial_testing_agents.md)
- [Towards Open-Vocabulary Audio-Visual Event Localization](../../CVPR2025/audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)

<!-- RELATED:END -->
