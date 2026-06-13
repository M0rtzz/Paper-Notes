---
title: >-
  [论文解读] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide
description: >-
  [CVPR 2025][视频生成][视频扩散模型] VideoGuide 提出了一种无需额外训练的视频扩散模型增强框架，通过在反向扩散采样的早期阶段利用任意预训练视频扩散模型（或自身）作为教师，将教师模型的去噪样本与采样模型进行插值融合，显著提升视频的时序一致性而不损害图像质量。 1. 领域现状：文本到视频（T2V）扩散模型…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频扩散模型"
  - "时序一致性"
  - "无训练引导"
  - "教师-学生蒸馏"
  - "低通滤波"
---

# VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide

**会议**: CVPR 2025  
**arXiv**: [2410.04364](https://arxiv.org/abs/2410.04364)  
**代码**: [https://github.com/dohunlee1/videoguide](https://github.com/dohunlee1/videoguide)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 视频扩散模型, 时序一致性, 无训练引导, 教师-学生蒸馏, 低通滤波

## 一句话总结

VideoGuide 提出了一种无需额外训练的视频扩散模型增强框架，通过在反向扩散采样的早期阶段利用任意预训练视频扩散模型（或自身）作为教师，将教师模型的去噪样本与采样模型进行插值融合，显著提升视频的时序一致性而不损害图像质量。

## 研究背景与动机

1. **领域现状**：文本到视频（T2V）扩散模型已经取得了重要进展，但在生成视频时面临时序一致性与图像质量之间的权衡困境。AnimateDiff 在个性化方面灵活但时序一致性差，LaVie 提供了多功能级联生成但时序稳定性不足。
2. **现有痛点**：已有方法如 FreeInit 通过迭代噪声细化来改善时序一致性，但带来了严重的图像质量退化（纹理细节丢失）和极高的计算开销（推理时间大幅增加）。UniCtrl 等方法也存在图像质量下降或流程复杂的问题。
3. **核心矛盾**：改善时序一致性的方法往往以牺牲图像保真度或增加不可接受的计算成本为代价，无法两全。
4. **本文目标**：设计一种零训练、低开销的通用框架，能在不损失图像质量的前提下，增强任意预训练 T2V 模型的时序一致性。
5. **切入角度**：作者从优化问题的视角出发，将视频一致性增强重新表述为一个正则化目标——高质量视频样本应满足：经过随机扰动后，能被教师模型良好地重建。
6. **核心 idea**：利用教师 VDM 的去噪样本与学生 VDM 的去噪样本进行插值，辅以低通滤波器，仅在前几步推理中引导采样方向，即可使整个生成过程趋向更好的时序一致性。

## 方法详解

### 整体框架

VideoGuide 的流程紧密嵌入标准 DDIM 采样过程。在采样模型（Student）的反向扩散过程中，取中间潜变量 $z_t$，送入引导模型（Teacher）进行若干步去噪得到 $z_{0|t-\tau}$，将其与 Student 自身的去噪估计 $z_{0|t}$ 进行加权插值，生成融合后的 $z'$，再通过低通滤波器处理高频区域。这一引导操作仅在推理的前 $I$ 步执行，后续步骤完全由 Student 模型自主完成。

### 关键设计

1. **视频一致性引导（Video Consistency Guidance）**:
    - 功能：将时序一致性增强表述为优化问题并嵌入反向采样过程
    - 核心思路：定义正则化目标 $\ell(z_0;\psi,\epsilon,t) = \|\epsilon_\psi(\sqrt{\bar\alpha_t}z_0 + \sqrt{1-\bar\alpha_t}\epsilon, t) - \epsilon\|^2$，将其梯度下降步骤融入 DDIM 更新公式。经过数学推导，梯度项自然简化为去噪样本的线性插值：$z' = \beta \cdot z_{0|t} + (1-\beta) \cdot z_{0|t-\tau}$，其中 $\beta$ 控制插值权重。教师模型 $\psi$ 的端点通过多步反向采样近似 PF-ODE 端点来获得。
    - 设计动机：从理论角度证明了视频一致性可以通过优化框架来改善，最终的插值方案在理论上等价于梯度引导，但实现极其简洁高效。

2. **低通滤波器（Low-Pass Filter, LPF）**:
    - 功能：加速一致性收敛，防止长时间优化导致的图像退化
    - 核心思路：在扩散过程的早期时间步，对更新后的潜变量应用低通滤波器：$z_{t-1} = \text{LPF}_\gamma(z_{t-1}) + \text{HPF}_{1-\gamma}(\epsilon)$，保留低频结构信息，用高斯噪声替换高频部分。使用 Butterworth 滤波器，归一化频率 0.25，阶数 $n=4$。
    - 设计动机：研究表明扩散过程的早期主要建立低频结构，高频贡献不大。LPF 在迭代中持续施加（而非仅用于初始噪声），确保轨迹稳定性。当使用外部 VDM 时，LPF 额外起到防止域漂移的作用——只蒸馏时序稳定性而保留 Student 模型的独特特征。

3. **外部 VDM 引导与域对齐（External VDM Guidance）**:
    - 功能：支持任意预训练 VDM 作为教师进行即插即用的引导
    - 核心思路：不同 VDM 有不同的噪声调度和分布。先通过 renoising 将 Student 的去噪估计 $z_{0|t}^{(S)}$ 转换到教师的噪声域：$z_t^{(G)} = \sqrt{\bar\alpha_t^{(G)}} z_{0|t}^{(S)} + \sqrt{1-\bar\alpha_t^{(G)}} \epsilon$，然后教师进行 $\tau$ 步去噪得到 $z_{0|t-\tau}^{(G)}$，最终进行跨模型插值。
    - 设计动机：使得框架具有极高的灵活性——可以自由选择当前最强的开源 VDM 作为教师，提升较弱模型的质量。例如用 VideoCrafter2（时序一致性强）引导 AnimateDiff（个性化能力强），实现"取长补短"。

### 损失函数 / 训练策略

本方法完全无需训练，仅在推理时操作。关键超参数：插值权重 $\beta=0.5$，插值步数 $I=5$（前 5 步使用引导），教师采样步数 $\tau=10$，总共 DDIM 50 步。

## 实验关键数据

### 主实验

| 方法 | Subject Consistency↑ | Background Consistency↑ | Imaging Quality↑ | Motion Smoothness↑ |
|------|---------------------|------------------------|-------------------|-------------------|
| AnimateDiff | 0.9183 | 0.9437 | 0.6647 | 0.9547 |
| AnimateDiff + FreeInit | 0.9487 | 0.9604 | 0.6173 | 0.9705 |
| AnimateDiff + Ours (self) | 0.9520 | 0.9600 | 0.6566 | 0.9731 |
| AnimateDiff + Ours (VC2) | **0.9614** | **0.9664** | **0.6671** | **0.9772** |

| 方法 | Subject Consistency↑ | Background Consistency↑ | Imaging Quality↑ | Motion Smoothness↑ |
|------|---------------------|------------------------|-------------------|-------------------|
| LaVie | 0.9534 | 0.9599 | 0.6750 | 0.9658 |
| LaVie + FreeInit | 0.9625 | 0.9643 | 0.6533 | **0.9757** |
| LaVie + Ours (self) | 0.9629 | **0.9652** | 0.6780 | 0.9725 |
| LaVie + Ours (VC2) | **0.9635** | 0.9643 | **0.6796** | 0.9723 |

### 消融实验

| 配置 | Subject Consistency | Background Consistency | 说明 |
|------|-------------------|----------------------|------|
| β=0.9 | 0.9518 | 0.9599 | 插值权重过高，引导不足 |
| β=0.5 | **0.9614** | **0.9664** | 最优插值权重 |
| I=1 | 0.9524 | 0.9618 | 仅1步引导，效果有限 |
| I=5 | **0.9614** | **0.9664** | 5步引导效果饱和 |
| τ=1 | 0.9444 | 0.9558 | 教师仅去噪1步，估计不准 |
| τ=10 | **0.9614** | **0.9664** | 10步去噪提供更好的端点估计 |

### 关键发现

- 三个超参（β、I、τ）均与时序一致性正相关，但存在计算效率权衡
- 使用外部高性能 VDM（VideoCrafter2）作为教师比自引导效果更好
- 推理时间远优于 FreeInit：AnimateDiff 上 21.68s vs 51.98s（快 2.4×），LaVie 上 10.01s vs 30.18s（快 3.0×）
- Prior Distillation 效果：AnimateDiff 由于数据先验不足，生成"beetle"和"jaguar"时会偏向汽车，VideoGuide 通过教师模型的先验蒸馏可以纠正这一问题

## 亮点与洞察

- **优化视角的理论贡献**：将视频一致性增强重新表述为优化问题，最终推导出的插值方案在理论上等价于梯度引导。这一思路非常优雅，使得方法既有理论基础又极其简洁。
- **即插即用的协同框架**：不同 VDM 各有所长，VideoGuide 让它们能够"合作"——Student 保留个性化/可控性等特有能力，Teacher 贡献时序稳定性。这意味着随着新模型的出现，旧模型不会过时，反而能通过引导获得提升。
- **Prior Distillation 的发现**：通过教师模型的数据先验蒸馏，学生模型能生成原本不在其数据分布中的内容。这暗示了模型之间的"知识转移"可以在推理阶段无需训练地完成。

## 局限与展望

- 动态程度（Dynamic Degree）略有下降，时序一致性与运动幅度之间存在固有权衡
- 低通滤波器的参数选择（截止频率、阶数）需要手动调整
- 外部引导增加推理时间（虽比 FreeInit 快很多，但仍比基线慢约 2×）
- 未来可探索：自适应插值权重（根据时间步动态调整 β）、更高效的教师采样策略、扩展到更长视频和更高分辨率

## 相关工作与启发

- **vs FreeInit**: FreeInit 通过迭代细化初始噪声改善一致性，但每次迭代都需完整 DDIM 采样，计算量巨大且易损害图像质量。VideoGuide 仅在前几步插值，开销小且质量更好。
- **vs PYoCo**: PYoCo 设计新的噪声先验但需要大量微调，VideoGuide 完全免训练。
- **vs DPS/DDS**: VideoGuide 的理论推导借鉴了 DPS 和 DDS 的引导框架，但创新性地将其应用于视频时序一致性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 从优化角度推导出简洁的插值方案，理论优雅；但核心操作（去噪样本插值）本身并不复杂
- 实验充分度: ⭐⭐⭐⭐ 多模型、多指标、消融完整，但缺少用户研究和更多 VDM 的组合实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从优化到插值的推导链条完整
- 价值: ⭐⭐⭐⭐ 零训练立即可用，对视频生成社区有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[NeurIPS 2025\] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision](../../NeurIPS2025/video_generation/video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)
- [\[CVPR 2025\] Taming Teacher Forcing for Masked Autoregressive Video Generation](taming_teacher_forcing_for_masked_autoregressive_video_generation.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
