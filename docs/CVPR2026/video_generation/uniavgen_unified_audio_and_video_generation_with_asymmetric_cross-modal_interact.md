---
title: >-
  [论文解读] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions
description: >-
  [CVPR 2026][音视频联合生成] UniAVGen 提出了一个基于对称双分支 DiT 的音视频联合生成框架，通过**非对称跨模态交互机制**和**人脸感知调制模块**实现精确的时空同步，仅用 1.3M 训练样本就在唇音同步、音色一致性和情感一致性上全面超越使用 30M 数据的竞品。
tags:
  - CVPR 2026
  - 音视频联合生成
  - 跨模态交互
  - 扩散模型
  - 唇音同步
  - 人脸感知调制
---

# UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions

**会议**: CVPR 2026  
**arXiv**: [2511.03334](https://arxiv.org/abs/2511.03334)  
**代码**: https://mcg-nju.github.io/UniAVGen/ (项目页)  
**领域**: 多模态生成  
**关键词**: 音视频联合生成, 跨模态交互, 扩散模型, 唇音同步, 人脸感知调制

## 一句话总结
UniAVGen 提出了一个基于对称双分支 DiT 的音视频联合生成框架，通过**非对称跨模态交互机制**和**人脸感知调制模块**实现精确的时空同步，仅用 1.3M 训练样本就在唇音同步、音色一致性和情感一致性上全面超越使用 30M 数据的竞品。

## 研究背景与动机

1. **领域现状**：音视频联合生成是生成式 AI 的重要方向。商业系统（Veo3、Sora2、Wan2.5）已展现出色效果，但开源方法仍主要依赖解耦的两阶段管线——先生成无声视频再配音，或先生成音频再驱动视频合成。

2. **现有痛点**：两阶段方法的根本问题在于**模态解耦**——生成过程中音频和视频无法交互，导致语义一致性差、情感对齐弱、唇音同步不精确。现有的端到端联合生成方法（JavisDiT、UniVerse-1、Ovi）虽尝试解决该问题，但要么只支持环境声不支持人类语音，要么跨模态对齐效果有限。

3. **核心矛盾**：音频和视频在时间粒度、语义空间上存在天然的**不对称性**——视频的每个 latent 帧对应多个音频 token，反之亦然。现有方法忽略了这种不对称性，要么全局交互（收敛慢），要么对称时间对齐交互（上下文利用不足）。

4. **本文目标**（a）如何设计既收敛快又性能好的跨模态交互；（b）如何让交互聚焦在人脸等关键区域；（c）推理时如何增强跨模态关联信号。

5. **切入角度**：视频中的唇部运动受前后音素影响，而音频需要感知更精确的视频时间位置信息——两个方向的需求完全不同，应采用不对称设计。

6. **核心 idea**：用模态感知的非对称跨模态注意力 + 人脸感知软掩码 + 模态感知 CFG，在远少于竞品的训练数据上实现 SOTA 的音视频同步生成。

## 方法详解

### 整体框架
UniAVGen 采用**对称双分支联合合成架构**：视频分支使用 Wan 2.2-5B DiT 骨干，音频分支使用 Wan 2.1-1.3B 的架构模板（结构相同，仅通道数不同）。输入包括参考说话人图像、视频描述文本和语音文本内容，可选地接受参考音频和条件音视频。两个分支通过 Flow Matching 范式训练，各自预测速度场。

视频分支：视频以 16fps 处理，经 VAE 编码为 latent $z^v$，将参考图像和条件视频的 latent 拼接作为输入，文本通过 umT5 编码后经 cross-attention 注入。

音频分支：音频以 24kHz 采样后转为 Mel 频谱图作为 latent $z^a$，参考音频和条件音频同样拼接输入，语音文本通过 ConvNeXt blocks 提取特征后注入。

### 关键设计

1. **非对称跨模态交互 (Asymmetric Cross-Modal Interaction)**:

    - 功能：实现双向、时间对齐的跨模态注意力，兼顾收敛速度和性能
    - 核心思路：包含两个模态专用对齐器——**A2V 对齐器**为每个视频帧构建一个音频上下文窗口 $C_i^a$（包含前后 $w$ 帧的音频 token），然后做逐帧 cross-attention，让视频感知相邻音频的语义信息；**V2A 对齐器**则采用**时间邻域插值策略**，对每个音频 token $j$，根据其在视频帧间的相对位置 $\alpha = (j \bmod k)/k$ 对相邻两个视频帧做加权插值，得到平滑的视频上下文 $C_j^v$，再做 cross-attention。所有输出矩阵零初始化以避免训练初期破坏各模态的生成能力。
    - 设计动机：视频的唇部动作受前后音素影响（需要窗口上下文），而音频需要感知精确的连续时间位置（需要插值），两个方向的需求天然不对称。相比全局交互（收敛慢）和对称时间对齐交互（上下文受限），非对称设计在两者之间取得最佳平衡。

2. **人脸感知调制 (Face-Aware Modulation, FAM)**:

    - 功能：动态引导跨模态交互聚焦于人脸等显著区域
    - 核心思路：在每个交互层引入轻量级掩码预测头，对视频特征 $H^{v_l}$ 施加 LayerNorm + 仿射变换 + 线性投影 + Sigmoid，生成软掩码 $M^l \in (0,1)^{T \times N_v}$。A2V 方向用掩码做选择性更新 $H^{v_l} = H^{v_l} + M^l \odot \bar{H}^{v_l}$；V2A 方向用掩码增强视频显著区域的信息传递 $\hat{H}^{v_l} = M^l \odot \hat{H}^{v_l}$。掩码用 ground-truth 人脸 mask 监督，损失权重 $\lambda^m$ 从 0.1 线性衰减到 0。
    - 设计动机：人体音视频的关键语义耦合集中在面部，训练初期约束交互范围可加速收敛、避免背景干扰；后期逐步放松约束让模型学习更灵活的交互模式（衰减 $\lambda^m$），实验证明衰减策略在音色和情感一致性上优于固定权重。

3. **模态感知无分类器引导 (Modality-Aware CFG, MA-CFG)**:

    - 功能：在推理时显式增强跨模态关联信号
    - 核心思路：传统 CFG 是单模态的，无法增强跨模态依赖。MA-CFG 的核心洞察是用一次前向传播（去掉跨模态交互的条件信号，等价于单模态推理）得到无条件估计 $u_{\theta_v}$ 和 $u_{\theta_a}$，再用有跨模态交互的估计 $u_{\theta_{a,v}}$ 计算引导：$\hat{u}_v = u_{\theta_v} + s_v(u_{\theta_{a,v}} - u_{\theta_v})$。
    - 设计动机：传统 CFG 只引导文本条件，忽略了音频驱动视频或视频影响音频的信号。MA-CFG 显著增强了情感强度和运动动态性。

### 损失函数 / 训练策略

三阶段训练：Stage 1 仅训练音频分支（$\mathcal{L}^a$, 160k steps, batch=256）；Stage 2 联合训练两分支（$\mathcal{L}^{joint} = \mathcal{L}^v + \mathcal{L}^a + \lambda_m \mathcal{L}^m$, 30k steps, batch=32, lr=5e-6）；Stage 3 多任务学习（5种任务比例 4:1:1:2:2, 10k steps）。$\lambda_m$ 从 0.1 线性衰减至 0。

## 实验关键数据

### 主实验

| 方法 | 联合训练 | 训练样本 | PQ↑ | CU↑ | WER↓ | SC↑ | DD↑ | LS↑ | TC↑ | EC↑ |
|------|---------|---------|-----|-----|------|-----|-----|-----|-----|-----|
| OmniAvatar (两阶段) | ✗ | 21.1B | 8.15 | 7.41 | 0.152 | 0.987 | 0.000 | 6.34 | 0.454 | 0.349 |
| Ovi (联合) | ✓ | 30.7M | 6.03 | 6.01 | 0.216 | 0.972 | 0.360 | 6.48 | 0.828 | 0.558 |
| **UniAVGen** | ✓ | **1.3M** | **7.00** | **6.62** | **0.151** | 0.973 | **0.410** | 5.95 | **0.832** | **0.573** |

UniAVGen 用 23 倍少的数据超越 Ovi（30.7M vs 1.3M），在音频质量和音视频一致性上全面领先。

### 消融实验

| 交互设计 (A2V / V2A) | LS↑ | TC↑ | EC↑ |
|----------------------|-----|-----|-----|
| SGI / SGI (全局) | 3.46 | 0.667 | 0.459 |
| STI / STI (对称时间) | 3.73 | 0.685 | 0.472 |
| **ATI / ATI (非对称)** | **4.09** | **0.725** | **0.504** |

| FAM 配置 | LS↑ | TC↑ | EC↑ |
|----------|-----|-----|-----|
| 无 FAM | 3.89 | 0.705 | 0.489 |
| 无监督 FAM | 3.92 | 0.701 | 0.492 |
| 固定 $\lambda_m$ | 4.11 | 0.719 | 0.497 |
| **衰减 $\lambda_m$** | **4.09** | **0.725** | **0.504** |

### 关键发现
- **非对称交互贡献最大**：ATI 在所有指标上显著优于 SGI 和 STI，验证了模态专用设计的必要性
- **FAM 的监督信号很重要**：有监督 FAM 比无监督大幅提升一致性，说明约束掩码到人脸区域有效加速训练收敛
- **衰减策略优于固定权重**：逐步放松约束让模型学习更灵活的交互，TC 和 EC 进一步提升
- **多任务训练增强联合生成**：先联合训练再多任务（JFML）效果最好，多任务从一开始训练（MTO）收敛更慢
- 在 OOD 动漫图像上，UniAVGen 展现出强泛化能力，而 Ovi 唇部运动失败、UniVerse-1 几乎静止

## 亮点与洞察
- **非对称设计巧妙精准**：A2V 用窗口上下文考虑前后音素影响、V2A 用时间插值感知连续视频位置，完美匹配了两个方向的不同需求
- **FAM 的渐进放松策略**：用衰减的监督信号初期约束后期释放，是一种兼顾训练效率和模型灵活性的优雅方案，可迁移到其他需要区域聚焦的多模态任务
- **MA-CFG 将 CFG 推广到跨模态**：思路简洁（用单模态推理作为无条件基线），但效果显著，可直接应用于任何双模态生成系统

## 局限与展望
- 仅专注于**人体中心**的音视频生成，未覆盖通用场景（环境声、音乐等）
- 音频分支仅支持**英文语音**，多语言能力未验证
- 视频时长受限（训练数据据推测为短视频片段），长视频的一致性维持未探讨
- 评估中 TC 和 EC 使用 Gemini-2.5-Pro 打分，缺乏标准化的开源评测方法

## 相关工作与启发
- **vs Ovi**: 同为对称双塔架构，但 Ovi 使用对称全局交互缺乏模态专用设计，OOD 泛化差；UniAVGen 通过非对称交互和 FAM 在 23 倍少的数据上超越
- **vs UniVerse-1**: 拼接两个预训练模型，架构不对称导致拼接复杂性能有限；UniAVGen 从设计之初就统一架构
- **vs 两阶段方法**: 两阶段方法唇音同步好但动态性几乎为零（DD≈0），说明视频生成时完全不感知音频

## 评分
- 新颖性: ⭐⭐⭐⭐ 非对称交互和 FAM 衰减策略有新意，MA-CFG 是 CFG 的自然推广
- 实验充分度: ⭐⭐⭐⭐ 主实验+5组消融+多任务分析+OOD定性比较，但评测指标部分依赖闭源模型
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、图表丰富、动机推导逻辑性强
- 价值: ⭐⭐⭐⭐ 开源音视频联合生成的 SOTA，数据效率极高，但限于人体场景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICLR 2026\] BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](../../ICLR2026/video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)
- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)

<!-- RELATED:END -->
