---
title: >-
  [论文解读] Referring Atomic Video Action Recognition
description: >-
  [ECCV 2024][视频理解][动作识别] 提出"基于文本引用的原子视频动作识别"（RAVAR）新任务和 RefAVA 数据集（36,630 实例），以及 RefAtomNet 方法，通过跨流 agent 注意力融合视觉、文本和位置-语义三路 token，在 mAP 上比最佳基线 BLIPv2 提升 3.85%/3.17%。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "动作识别"
  - "referring expression"
  - "multi-stream fusion"
  - "注意力机制"
---

# Referring Atomic Video Action Recognition

**会议**: ECCV 2024  
**arXiv**: [2407.01872](https://arxiv.org/abs/2407.01872)  
**代码**: [https://ravar-dataset.github.io/](https://ravar-dataset.github.io/)  
**领域**: 视频理解 / 人体理解  
**关键词**: atomic action recognition, referring expression, multi-stream fusion, agent attention, video understanding

## 一句话总结
提出"基于文本引用的原子视频动作识别"（RAVAR）新任务和 RefAVA 数据集（36,630 实例），以及 RefAtomNet 方法，通过跨流 agent 注意力融合视觉、文本和位置-语义三路 token，在 mAP 上比最佳基线 BLIPv2 提升 3.85%/3.17%。

## 研究背景与动机
**领域现状**：原子视频动作识别（Atomic Video Action Recognition）关注人的最基本、不可再分的动作。现有工作（如 I3D、X3D、MViTv2、Hiera）在多人场景中要么手动裁剪特定人物区域，要么自动检测所有人并分别预测，需要大量前/后处理。

**现有痛点**：在实际应用中（如辅助系统、人机交互），用户往往只关心特定个体的动作。现有方法要么对所有人预测（低效），要么需手动裁剪（不实用）。缺乏利用自然语言描述来指定目标个体的机制。

**核心矛盾**：视频中包含大量无关视觉信息，会干扰模型对目标个体的关注。如何根据文本引用抑制不相关信息是核心挑战。

**本文目标**：定义 RAVAR 任务——给定视频和描述特定个体的文本（如"穿红色上衣的女性"），识别该个体的原子动作并给出位置。

**切入角度**：三流架构（视觉+文本+位置-语义），通过 agent attention 跨流融合抑制不相关信息。

**核心 idea**：引入位置-语义感知流（融合检测框坐标和目标类别语义），结合跨流 agent 注意力融合，精确定位文本描述的特定个体并识别其原子动作。

## 方法详解

### 整体框架
RefAtomNet 包含三个 token 流：(1) 视觉流——ViT 编码视频帧，QFormer 提取视觉 token；(2) 文本引用流——BERT 编码引用文本，QFormer 提取文本 token；(3) 位置-语义流——DETR 检测目标后，融合框坐标和类别语义嵌入。三路 token 通过 agent-based 跨流注意力融合后，经 MLP 头预测动作和边界框。

### 关键设计

1. **位置-语义感知 Token（Location-Semantic Aware Tokens）**

    - 功能：将场景中检测到的物体的位置和语义信息编码为 token，辅助定位
    - 核心思路：用冻结的 DETR 从关键帧检测 $N_o$ 个目标，得到边界框 $\mathbf{r}_{boxes} \in \mathbb{R}^{N_o \times 4}$ 和类别标签 $\mathbf{r}_{cats}$。类别标签过 BERT 编码得到语义嵌入，与框坐标拼接并投影：
    $\mathbf{t}^{LS} = \mathbf{P}_{LS}(\text{Concat}[\mathcal{V}_{RT}(\mathbf{r}_{cats}), \mathbf{r}_{boxes}])$
    - 设计动机：文本引用常包含位置信息（"左侧的人"），单纯视觉特征难以提供精确的空间关系。目标检测结果天然包含位置和语义，可作为定位辅助信号

2. **跨流 Agent 注意力融合（Cross-Stream Agent Attention Fusion）**

    - 功能：利用 agent token 跨流抑制不相关视觉信息
    - 核心思路：
        - 对每个流 $\phi \in \{RT, VT, LS\}$ 计算 Q、K、V 和 agent token $\mathbf{A}^\phi$
        - 对文本和位置-语义流做 agent attention：$\mathbf{M}_{QA}^\pi = \sigma_c(\text{MatMul}[\alpha \cdot \mathbf{A}_*^\pi, \mathbf{Q}^\pi])$
        - 跨流融合视觉流的 agent query 注意力：
    $\hat{\mathbf{M}}_{QA}^\gamma = \text{AVG}[\mathbf{M}_{QA}^\gamma, \sigma_c(\sum_\pi \mathbf{M}_{QA}^\pi) \cdot \mathbf{M}_{QA}^\gamma, \sigma_t(\sum_\pi \mathbf{M}_{QA}^\pi) \cdot \mathbf{M}_{QA}^\gamma]$
        - 类似地计算跨流 agent token 融合，最终聚合所有流：$\mathbf{t}_{agg} = \sum_\phi \mathbf{t}_*^\phi / N_s$
    - 设计动机：标准注意力无法有效区分不同流的重要性。Agent attention 通过中间 agent token 聚合关键信息并排除冗余，跨流应用可以用文本和位置-语义线索指导视觉流关注正确区域

3. **1D Sequential Agent Token 改造**

    - 功能：将原本为 2D 图像设计的 agent attention 适配为 1D 序列格式
    - 核心思路：用全连接层替代 2D 池化获取 agent token，去除深度卷积分支和 2D 位置编码
    - 设计动机：三路 token 来自不同模态，都是 1D 序列格式，原始 agent attention 的 2D 设计不适用

### 损失函数 / 训练策略
- BCE 损失（多标签动作分类）：$L_{BCE} = -\frac{1}{N_c}\sum_{i=1}^{N_c}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- MSE 损失（边界框回归）：$L_{MSE} = \sum_{j=1}^{4}(b_j - \hat{b}_j)^2$，权重 5
- BertAdam 优化器，lr=1e-4，batch=128，40 epoch
- 文本编码器冻结，214M 可训练参数

## 实验关键数据

### RefAVA 数据集

| 属性 | 数值 |
|------|------|
| 视频片段数 | 17,946（来自 127 部电影） |
| 标注实例数 | 36,630 |
| 训练/验证/测试 | 22,658 / 10,916 / 3,056 |
| 原子动作类别 | 80 类（物体操作+人际交互+人体运动） |
| 总帧数 | 1,615,140 |

### 主实验

| 方法 | 类别 | Val mIOU | Val mAP | Val AUROC | Test mIOU | Test mAP | Test AUROC |
|------|------|----------|---------|-----------|-----------|----------|------------|
| I3D | AAL | 0.00 | 44.04 | 57.77 | 0.00 | 44.64 | 62.71 |
| X3D | AAL | 0.26 | 44.45 | 59.09 | 0.27 | 46.34 | 64.51 |
| AskAnything | VQA | 20.09 | 51.42 | 66.12 | 22.35 | 52.25 | 69.35 |
| BLIPv2 | VTR | 32.99 | 52.13 | 66.56 | 32.75 | 53.19 | 69.92 |
| Su et al. | VOS | 23.71 | 52.17 | 66.67 | 26.02 | 53.20 | 70.19 |
| **RefAtomNet** | **Ours** | **38.22** | **55.98** | **69.73** | **36.42** | **57.52** | **73.95** |

### 消融实验

| 配置 | Val mIOU | Val mAP | Val AUROC | 说明 |
|------|----------|---------|-----------|------|
| w/o ALSAF（简单加法融合）| 27.30 | 50.70 | 65.31 | mIOU 下降 10.92 |
| w/o LSAS（去位置语义流）| 31.90 | 55.21 | 69.47 | 定位能力下降 |
| w/o CAAF（去跨流注意力融合）| 36.21 | 55.43 | 69.66 | 少量下降 |
| w/o CATF（去跨流 token 融合）| 35.01 | 53.83 | 67.71 | AUROC 下降 2.02 |
| **RefAtomNet (完整)** | **38.22** | **55.98** | **69.73** | — |

### 与其他融合机制比较

| 融合方式 | Val mIOU | Val mAP | Val AUROC |
|----------|----------|---------|-----------|
| Addition | 27.30 | 50.70 | 65.31 |
| Concatenation | 18.64 | 52.23 | 66.45 |
| AttentionBottleneck | 33.47 | 50.97 | 65.07 |
| **Ours (Agent Fusion)** | **38.22** | **55.98** | **69.73** |

### 关键发现
- AAL 方法（I3D、X3D 等）在 mIOU 上接近 0，说明其完全无法定位文本指定的个体
- VQA 和 VTR 基线受益于文本感知预训练，但对原子动作的细粒度预测仍不足
- 简单加法融合三流导致 mIOU 暴跌（38.22→27.30），证明不加区分地融合会引入大量不相关视觉干扰
- 位置-语义流对 mIOU 提升最大（31.90→38.22），说明检测框坐标+类别语义是定位引用个体的关键

## 亮点与洞察
- **新任务定义有意义**：RAVAR 直接将自然语言引用与原子动作识别结合，解决了多人场景中"识别谁做了什么"的实际需求。数据集基于 AVA 扩展，7 名标注者提供高质量文本引用。
- **Agent Attention 的跨流重定义**：将 agent attention 从 2D 图像适配为 1D 多流序列融合的思路具有创新性，通过 agent token 在不同模态间传递和过滤信息，比简单拼接或 attention bottleneck 有效得多。

## 局限与展望
- RefAVA 基于 AVA 数据集，80 类原子动作的覆盖面有限
- 文本引用不包含动作描述（只描述外观和位置），实际场景可能需要更灵活的引用方式
- 214M 可训练参数相对较多，推理效率未讨论
- 仅用关键帧做目标检测，缺乏时间维度的目标追踪

## 相关工作与启发
- **vs BLIPv2**：BLIPv2 作为最强 VTR 基线在 RAVAR 上 mAP 53.19，RefAtomNet 的 57.52 说明专门设计的位置-语义流和跨流融合对细粒度动作识别至关重要
- **vs RVOS（引用视频目标分割）**：RVOS 输入文本引用包含动作名称，RAVAR 的文本仅描述外观/位置不含动作，任务设定不同

## 评分
- 新颖性: ⭐⭐⭐⭐ 新任务+新数据集+新方法，任务定义有实际意义
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个基线方法覆盖 AAL/VQA/VTR/SF/VOS 五个领域
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务对比图直观
- 价值: ⭐⭐⭐⭐ 填补引用式原子动作识别空白，数据集和基准可推动后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[ECCV 2024\] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition](optimizing_factorized_encoder_models_time_and_memory_reduction_for_scalable_and_.md)
- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)

</div>

<!-- RELATED:END -->
