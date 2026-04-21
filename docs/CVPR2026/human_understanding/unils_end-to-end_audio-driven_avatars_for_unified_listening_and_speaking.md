---
title: >-
  [论文解读] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
description: >-
  [CVPR 2026][人体理解][对话虚拟人] 提出首个端到端统一说话-倾听面部表情生成框架UniLS，通过两阶段训练范式（先学内在运动先验、再用双轨音频微调），仅需双方音频输入即可同时生成自然的说话和倾听面部动作，倾听指标提升高达44.1%。
tags:
  - CVPR 2026
  - 人体理解
  - 对话虚拟人
  - 说话-倾听统一生成
  - 音频驱动
  - 面部动画
  - 两阶段训练
---

# UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking

**会议**: CVPR 2026  
**arXiv**: [2512.09327](https://arxiv.org/abs/2512.09327)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 对话虚拟人, 说话-倾听统一生成, 音频驱动, 面部动画, 两阶段训练

## 一句话总结
提出首个端到端统一说话-倾听面部表情生成框架UniLS，通过两阶段训练范式（先学内在运动先验、再用双轨音频微调），仅需双方音频输入即可同时生成自然的说话和倾听面部动作，倾听指标提升高达44.1%。

## 研究背景与动机

1. **领域现状**：对话虚拟人领域的大多数系统仍然是单向的——要么只做说话生成（speech-driven），要么只做倾听生成（listener generation）。真正的交互需要虚拟人同时具备说话和倾听两种模式。

2. **现有痛点**：
    - **说话-only方法**（如FaceFormer、CodeTalker、ARTalk）：虽然能生成高质量的说话面部动作，但完全忽略了倾听行为，无法在对话场景中使用。
    - **唯一的说话-倾听方法DualTalk**：依赖说话者A的预计算面部序列来生成说话者B的动作，不是端到端的，需要额外的动作获取/生成流水线，无法实时部署。
    - **直接端到端联合训练失败**：导致倾听表情变得僵硬、不自然（"poker face"现象）。

3. **核心矛盾**：说话动作与音频有强相关性（音素-唇形对齐），但倾听动作与对方音频的相关性很弱——倾听时的眨眼、点头、微表情主要来自内在运动先验，而非外部语音信号。这种音频-动作相关性的不平衡导致联合训练中倾听分支崩溃到低方差的安全静态先验。

4. **本文目标** 设计一个端到端框架，仅用双方音频就能同时生成说话者A和B的说话+倾听面部动作，关键挑战是解决倾听表情的僵硬问题。

5. **切入角度**：将倾听行为重新理解为"内在运动先验"与"外部音频调制"的组合——人的倾听表情首先遵循自身的运动模式（如眨眼频率、微妙点头），然后才受外部语音的调制。

6. **核心 idea**：通过两阶段训练——先无音频学运动先验、再用双轨音频微调——将倾听的内在动力学和外部音频调制解耦学习，彻底解决端到端倾听僵硬问题。

## 方法详解

### 整体框架
UniLS基于chunk-based自回归模型生成面部动作。输入是说话者A和B的双轨音频，输出是FLAME参数（表情参数 $\psi$ 和姿态参数 $\theta$）表示的3D面部动作。训练分两个阶段：Stage 1学习无音频的运动先验，Stage 2进行音频驱动的微调。底层使用多尺度VQ codec将面部动作离散化。

### 关键设计

1. **多尺度VQ Codec**:

    - 功能：将连续的3D面部动作离散化为紧凑且具有表现力的离散表示，作为自回归生成器的监督目标。
    - 核心思路：Transformer encoder将输入动作chunk $M$ 提取为时序特征 $\mathbf{f}$，然后通过多尺度码本 $[k_1, k_2, ..., k_L]$（尺度为 $[1, 5, 25, 50, 100]$）逐层量化。每层先量化再插值：$\mathbf{c}^{(l+1)} = \text{Interp}(\text{Quant}(\mathbf{f}^{(l)}), k_l)$，残差 $\mathbf{f}^{(l+1)} = \mathbf{f}^{(l)} - \mathbf{c}^{(l+1)}$。码本256个条目，每个维度64。
    - 设计动机：多尺度量化在不同时间分辨率上逐步细化运动表示，既保证了高保真合成又维持了时间连贯性。

2. **Stage 1: 无音频生成器训练（运动先验学习）**:

    - 功能：训练一个不使用任何音频输入的自回归生成器，让模型学习自然面部运动的内在动力学先验。
    - 核心思路：输入仅包含运动chunk $M$ 和风格嵌入 $\mathbf{s}$（编码说话人特定运动特征），生成器 $\mathcal{G}$ 预测下一个chunk：$\hat{M}_{t:2t} = \mathcal{G}(M_{1:t}, \mathbf{s})$。使用自回归重建损失 $\mathcal{L} = \sum_{t=1}^{T} ||\hat{M}_{t:2t} - M_{t:2t}||$。训练数据包括新闻广播、访谈、直播和日常对话等多场景非配对数据（546.5小时）。
    - 设计动机：通过在无音频条件下学习，模型建立起对自然面部行为的内在先验——眨眼、微表情、微妙头部运动等。这些行为是倾听表情的基础，不应由音频驱动。

3. **Stage 2: 音频驱动微调（双轨音频调制）**:

    - 功能：在配对对话数据上微调生成器，使其根据双轨音频条件生成说话和倾听动作。
    - 核心思路：在每个Transformer块中新增两个cross-attention层——一个关注说话者A的音频（控制说话行为），另一个关注说话者B的音频（控制倾听行为）。新增的cross-attention从头训练，Stage 1骨干网络使用LoRA微调，保护已学到的运动先验不被覆盖。生成过程为 $\hat{M}_{t:2t} = \mathcal{G}(M_{1:t}, \mathbf{a}^A_{1:t}, \mathbf{a}^B_{1:t}, \mathbf{s})$。音频特征由冻结的wav2vec编码器提取。
    - 设计动机：双cross-attention设计让模型能区分自己的语音和对方的语音（单cross-attention混合音频会导致说话质量严重下降，LVE从5.83升至11.48）。LoRA微调保护运动先验，避免Stage 2过度拟合音频信号而丢失倾听多样性。

### 损失函数 / 训练策略
两个阶段均使用chunk-wise自回归重建损失。Stage 1在4卡H200上训练10 GPU小时，Stage 2训练30 GPU小时。使用AdamW优化器，学习率1e-4，batch size 128，训练200K迭代。

## 实验关键数据

### 主实验
在Seamless Interaction数据集测试集上评估说话和倾听性能。

| 方法 | LVE↓ | MHD↓ | 说话FDD↓ | 说话PDD↓ | 说话JDD↓ | 倾听FDD↓ | 倾听PDD↓ | 倾听JDD↓ | F-FID↓ | P-FID↓ |
|------|------|------|---------|---------|---------|---------|---------|---------|--------|--------|
| DiffPoseTalk | 9.48 | 2.96 | 32.66 | 7.89 | 1.40 | - | - | - | - | - |
| ARTalk | 7.46 | 2.12 | 31.64 | 7.66 | 1.19 | - | - | - | - | - |
| ARTalk* | 6.79 | 2.02 | 27.41 | 8.55 | 0.81 | 30.62 | 9.52 | 1.53 | 10.78 | 0.072 |
| DualTalk | 6.35 | 1.95 | 37.46 | 9.70 | 1.02 | 43.58 | 10.71 | 2.02 | 13.14 | 0.079 |
| **UniLS** | **5.83** | **1.89** | **18.41** | **4.67** | **0.71** | **17.12** | **4.75** | **0.98** | **4.30** | **0.038** |

### 消融实验

| 配置 | LVE↓ | MHD↓ | 倾听FDD↓ | F-FID↓ | 说明 |
|------|------|------|---------|--------|------|
| 单cross-attention | 11.48 | 3.00 | 27.46 | 5.97 | 混合音频导致说话严重退化 |
| 无Stage 1 | 6.32 | 2.02 | 25.64 | 5.97 | 无运动先验导致倾听僵硬 |
| 无多场景数据 | 6.26 | 1.99 | 17.81 | 4.62 | 数据多样性对倾听有帮助 |
| 完整UniLS | **5.83** | **1.89** | **17.12** | **4.30** | 所有组件协同最优 |

### 关键发现
- **倾听指标提升巨大**：UniLS的F-FID (4.30) 相比DualTalk (13.14) 降低了67.3%，P-FID降低51.9%，说明生成的倾听动作的分布与真实分布更接近。
- **说话精度也是SOTA**：LVE 5.83是最低的，说明唇形同步精度最好。
- **双cross-attention至关重要**：单cross-attention导致LVE几乎翻倍（5.83→11.48），因为模型无法区分混合音频流中的说话/倾听信号。
- **Stage 1不可或缺**：去掉Stage 1后倾听FDD从17.12升至25.64（增50%），验证了运动先验学习对解决倾听僵硬的核心作用。
- 用户研究中25人参与者对DualTalk的倾听偏好高达91.35%。

## 亮点与洞察
- **"音频-动作相关性不平衡"的分析非常精准**：通过t-SNE可视化清晰展示了说话音频与运动特征的强对齐vs倾听音频的弱对齐，定位了问题本质。这个insight比直接的方法设计更有价值。
- **两阶段解耦设计简洁有效**：先学运动先验再用LoRA微调，既保护了多样性又引入了音频控制，是一个非常优雅的方案。这种"先建先验再微调"的思路可以迁移到其他存在多模态不平衡的任务。
- **第一个端到端说话-倾听框架**：直接用双轨音频驱动，不需要预生成对方的面部序列，具有明确的实际应用价值（实时对话系统）。

## 局限与展望
- 目前只处理面部动作（FLAME参数），未涉及身体姿态和手势，完整的对话虚拟人需要全身动作。
- 倾听行为的评估指标主要是分布性指标（FID等），缺乏对倾听反应"时机"和"语义相关性"的评估——是否在正确时刻做出恰当反应。
- 训练需要大规模配对对话数据（657.5小时），数据获取成本较高。
- 未探讨多人对话场景（>2人）的扩展。

## 相关工作与启发
- **vs DualTalk**：DualTalk需要预计算说话者A的面部序列，非端到端，本文仅用双轨音频即可。倾听F-FID从13.14降至4.30。
- **vs ARTalk***：ARTalk*是作者适配的基线（添加额外音频输入），但倾听仍然僵硬，F-FID 10.78远高于UniLS的4.30。
- **vs FaceFormer/CodeTalker**：这些说话-only方法完全不处理倾听，UniLS在说话精度上也超越了它们。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个端到端说话-倾听框架，两阶段训练范式解决了倾听僵硬的根本问题
- 实验充分度: ⭐⭐⭐⭐ 定量+用户研究+消融充分，但缺乏时序语义对齐评估
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析透彻，t-SNE可视化直观
- 价值: ⭐⭐⭐⭐⭐ 对交互式虚拟人系统有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)
- [CryptoFace: End-to-End Encrypted Face Recognition](../../CVPR2025/human_understanding/cryptoface_end-to-end_encrypted_face_recognition.md)
- [Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](../../ICLR2026/human_understanding/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)
- [VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](../../ICML2025/human_understanding/sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)

<!-- RELATED:END -->
