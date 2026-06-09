---
title: >-
  [论文解读] Chain of World: World Model Thinking in Latent Motion (CoWVLA)
description: >-
  [CVPR 2026][机器人][VLA] 提出CoWVLA，统一世界模型VLA和隐动作VLA的优势：通过Latent Motion Extractor将视频分解为结构隐变量和运动隐变量，VLA在隐运动空间做世界模型预测而非重建冗余像素，配合Co-Fine-tuning交替生成关键帧和动作token…
tags:
  - "CVPR 2026"
  - "机器人"
  - "VLA"
  - "世界模型"
  - "隐运动建模"
  - "视频VAE"
  - "关键帧预测"
  - "动作量化"
---

# Chain of World: World Model Thinking in Latent Motion (CoWVLA)

**会议**: CVPR 2026  
**arXiv**: [2603.03195](https://arxiv.org/abs/2603.03195)  
**代码**: [https://fx-hit.github.io/cowvla-io](https://fx-hit.github.io/cowvla-io)  
**领域**: 机器人操作 / 视觉-语言-动作模型 / 世界模型  
**关键词**: [VLA, 世界模型, 隐运动建模, 视频VAE, 关键帧预测, 动作量化]  

## 一句话总结
提出CoWVLA，统一世界模型VLA和隐动作VLA的优势：通过Latent Motion Extractor将视频分解为结构隐变量和运动隐变量，VLA在隐运动空间做世界模型预测而非重建冗余像素，配合Co-Fine-tuning交替生成关键帧和动作token，LIBERO-LONG达95.2%超越π₀(85.2%)，SimplerEnv-WidowX avg 0.560超π₀(0.425)。

## 背景与动机
将世界模型引入VLA是近期的重要趋势，核心思想是让模型不仅预测动作，还能预测未来状态——"想象"动作执行后的世界会变成什么样。然而现有两类方法各有硬伤：

1. **World-model VLA（如GR-2、UniPi）**：直接在像素空间预测未来帧。问题是场景中大量像素是静止背景（桌面、墙壁、远处物体），模型将大量capacity浪费在重建这些冗余信息上。真正对机器人决策有用的是运动相关信息（物体位移、机械臂轨迹），而这在像素空间中只占极小比例。
2. **Latent-action VLA（如LAPA、latent action pretraining）**：将动作编码到隐空间，绕过显式动作标注的限制。但这类方法仅抽取动作的隐表示，缺乏对时间连续动态的建模——不能预测"接下来会发生什么"，也没有整合世界知识来做前瞻性推理。

核心矛盾：世界模型需要预测未来，但像素级预测太浪费；隐动作节省了capacity，但丢失了世界动态信息。

## 核心问题
如何在不重建冗余背景像素的前提下，让VLA具备世界模型的预测能力——即在隐运动空间而非像素空间进行世界模型推理？

## 方法详解

### 整体框架
CoWVLA分三个阶段：(1) 预训练Latent Motion Extractor——将视频分解为结构和运动两类隐变量；(2) VLA预训练——在隐运动空间做世界模型预测；(3) Co-Fine-tuning——交替生成关键帧(视觉token)和动作token(FAST量化)。

### 关键设计

1. **Latent Motion Extractor（Sec 3.1）**: 基于预训练视频VAE构建的运动分解器：

    - **Structure latent z_s**：捕获场景的静态结构信息（空间布局、物体外观）。从视频VAE的中间特征出发，用Q-Former（一组可学习的query token通过cross-attention聚合视频特征）提取。z_s编码的是"场景是什么样的"。
    - **Motion latent z_m**：捕获场景中运动变化——物体位移、机械臂轨迹、夹爪状态变化。提取方式独特：对视频VAE的时间差分特征在H和W方向分别做空间均值池化，得到两个向量，拼接形成z_m。H方向均值保留水平运动模式，W方向均值保留垂直运动模式，拼接后完整编码2D运动场。这种设计的优势在于z_m天然过滤了静态背景——均值池化消除了不变区域的贡献。
    - 训练目标：从z_s和z_m重建原始视频帧，确保分解是完整的（信息无损）。

2. **VLA预训练——隐运动空间的世界模型（Sec 3.2）**: 输入序列：[T, v_q^1, Q, v_q^f]，其中T是语言指令token，v_q^1是首帧视觉token（SigLIP编码），Q是可学习的motion query（对应z_m的预测位置），v_q^f是末帧视觉token。

    - **运动预测**：Q位置的MLP预测头输出ẑ_m，用MSE损失对齐真值z_m
    - **末帧重建**：v_q^f位置用交叉熵损失重建末帧视觉token
    - **因果注意力mask**：关键设计——Q token不能看到v_q^f（未来帧），确保运动预测是真正的"预测"而非"抄答案"。但v_q^f可以看到Q，因为末帧重建可以利用预测的运动信息
    - 总损失：L_pretrain = MSE(ẑ_m, z_m) + CE(visual tokens)
    - 这一阶段让VLA学会了"给定当前观测和指令，预测接下来的运动模式"——这就是隐运动空间的世界模型

3. **Co-Fine-tuning——交替关键帧与动作生成（Sec 3.3）**: 微调阶段的输入序列：[T, ṽ_q^1, Q, A_q^1, ṽ_q^2, Q, A_q^2, ...]，交替排列：

    - ṽ_q^i：第i个关键帧的视觉token
    - Q：motion query，持续聚合到目前为止的运动动态
    - A_q^i：第i步的动作token
    - **动作量化**：用FAST（Fast Action STate quantizer）将连续动作离散化为token序列，VLA以自回归方式预测
    - **关键帧量化**：用VQGAN将关键帧图像编码为离散visual token
    - **Q的累积聚合**：每个时间步的Q不仅预测当前运动，还通过cross-attention聚合之前所有时间步的信息，形成持续更新的动态表示——类似world model的隐状态

### 损失函数 / 训练策略
- 预训练：L = MSE(ẑ_m, z_m) + CE(v_q^f reconstruction)，在大规模机器人视频数据上训练
- Co-Fine-tuning：L = CE(action tokens) + CE(keyframe tokens) + MSE(motion prediction)，在任务特定数据上微调
- FAST量化器和VQGAN独立预训练后冻结
- 推理时：自回归生成action token → 解码为连续动作，同时生成关键帧token用于可视化/验证

## 实验关键数据

| 数据集 | 指标 | CoWVLA | π₀ | OpenVLA | HPT | 提升(vs π₀) |
|--------|------|--------|-----|---------|-----|-------------|
| LIBERO-Spatial | 成功率 | **96.8%** | 92.4% | 78.8% | — | **+4.4** |
| LIBERO-Object | 成功率 | **98.4%** | 94.0% | 88.4% | — | +4.4 |
| LIBERO-Goal | 成功率 | **95.2%** | 87.2% | 68.4% | — | +8.0 |
| LIBERO-Long | 成功率 | **95.2%** | 85.2% | 56.4% | — | **+10.0** |
| LIBERO-Avg | 成功率 | **96.4%** | 89.7% | 73.0% | — | **+6.7** |
| SimplerEnv-WidowX | Avg score | **0.560** | 0.425 | 0.268 | 0.308 | **+0.135** |
| SimplerEnv-Google Robot | Avg score | **0.504** | — | 0.248 | 0.480 | — |

### 消融实验要点
- 移除motion latent预训练：LIBERO-Avg从96.4%降至92.1%，说明隐运动空间的世界模型预训练是核心贡献
- 用像素级重建替换latent motion预测：性能降至90.8%，证实像素级重建确实浪费capacity
- 移除Co-Fine-tuning中的关键帧生成：降至93.7%，关键帧提供了有用的视觉锚点
- 移除Q-Former聚合（直接concat所有视频token）：z_s过于冗长，训练不稳定
- z_m提取方式对比：H/W方向均值拼接 > 全局均值池化 > 时间差分卷积，说明保留方向信息很重要

## 亮点
- **隐运动空间世界模型的概念突破**：不在像素空间重建未来帧，而在压缩的运动隐空间做预测——既保留了世界模型的前瞻推理能力，又消除了冗余背景重建的计算浪费
- **z_m的提取方式优雅**：H/W方向均值池化天然过滤静态背景，保留运动方向信息，设计极简但有效
- **因果mask设计精巧**：Q不能看未来帧（确保是预测），但未来帧可以看Q（利用预测的运动信息）——信息流向完全正确
- **Co-Fine-tuning的交替生成**：关键帧和动作token交替排列，Q持续聚合动态——类似world model的循环状态更新
- **LIBERO-Long大幅领先**：95.2% vs π₀的85.2%，+10%的提升说明世界模型思维对长序列任务至关重要

## 局限与展望
- 视频VAE是预训练后冻结的——如果VAE的运动-结构分解不够好，后续所有环节都受影响。端到端联合训练VAE可能进一步提升
- z_m的H/W方向均值池化丢失了精细的空间局部运动信息——对需要精确空间定位的任务（如穿针引线）可能不够
- 关键帧的选择策略未详细描述——均匀采样还是基于运动幅度自适应选取？不同策略可能显著影响性能
- SimplerEnv-Google Robot上的改进有限（0.504），可能因Google Robot任务与预训练数据分布差异较大
- FAST和VQGAN量化器引入了离散化误差——对精细动作（如旋转瓶盖）可能产生累积偏差

## 与相关工作的对比
- **vs GR-2 / UniPi (像素级世界模型VLA)**: 这类方法在像素空间做未来帧预测/生成，计算代价高且大量capacity浪费在静态背景上。CoWVLA在隐运动空间做预测，只关注运动相关信息
- **vs LAPA (隐动作预训练)**: LAPA提取动作隐表示但不做时间连续动态建模。CoWVLA的latent motion预测包含时间动态——不仅知道"做什么动作"，还知道"世界会怎么变"
- **vs π₀ (Flow matching VLA)**: π₀用flow matching做连续动作预测，没有世界模型组件。CoWVLA通过隐运动预测赋予π₀框架世界模型能力，LIBERO-Long +10%
- **vs AtomicVLA (技能规划VLA)**: AtomicVLA通过think-act切换做任务规划，CoWVLA通过世界模型做前瞻推理——两者解决多步任务的角度不同但互补：AtomicVLA是"先想清楚做什么"，CoWVLA是"先想象做后会怎样"

## 启发与关联
- **隐运动空间的通用性**：将视频分解为structure+motion的思路可以迁移到视频理解任务——如动作识别中用motion latent做分类可能比端到端更高效
- **Co-Fine-tuning范式**：交替生成不同模态token的训练方式可以应用到其他多模态任务——如图文交错生成、视频-音频联合生成
- **与扩散世界模型的结合**：CoWVLA的latent motion可以作为扩散世界模型的条件信号——先预测z_m（快速、低维），再用z_m引导像素级扩散生成高保真未来帧

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 隐运动空间世界模型是全新概念，z_m提取方式和Co-Fine-tuning交替生成均有创新
- 实验充分度: ⭐⭐⭐⭐ LIBERO全子集 + SimplerEnv两个机器人 + 完整消融，但缺少真实机器人实验
- 写作质量: ⭐⭐⭐⭐ 动机对比（像素重建 vs 隐运动预测）非常清晰，方法描述系统化
- 价值: ⭐⭐⭐⭐⭐ 为VLA世界模型指明了新方向——从像素空间走向隐运动空间，LIBERO和SimplerEnv大幅SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[ICLR 2026\] Sparse Imagination for Efficient Visual World Model Planning](../../ICLR2026/robotics/sparse_imagination_for_efficient_visual_world_model_planning.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](../../ICML2026/robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[CVPR 2025\] Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](../../CVPR2025/robotics/neural_motion_simulator_pushing_the_limit_of_world_models_in_reinforcement_learn.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)

</div>

<!-- RELATED:END -->
