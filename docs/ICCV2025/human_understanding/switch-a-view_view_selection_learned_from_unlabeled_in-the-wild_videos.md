---
title: >-
  [论文解读] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos
description: >-
  [ICCV 2025][人体理解][视角选择] 提出 Switch-a-view 模型，通过从大规模无标注的互联网教学视频中学习视角切换模式（ego/exo），实现多视图教学视频的自动视角选择，无需显式的最佳视角标注。
tags:
  - ICCV 2025
  - 人体理解
  - 视角选择
  - 教学视频
  - 多视图视频
  - 弱监督学习
  - 自动摄像
---

# Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos

**会议**: ICCV 2025  
**arXiv**: [2412.18386](https://arxiv.org/abs/2412.18386)  
**代码**: [项目页面](https://vision.cs.utexas.edu/projects/switch_a_view/)  
**领域**: 人体理解  
**关键词**: 视角选择, 教学视频, 多视图视频, 弱监督学习, 自动摄像

## 一句话总结

提出 Switch-a-view 模型，通过从大规模无标注的互联网教学视频中学习视角切换模式（ego/exo），实现多视图教学视频的自动视角选择，无需显式的最佳视角标注。

## 研究背景与动机

教学视频（how-to video）在 YouTube、TikTok 等平台上拥有庞大的用户群体。一个高质量的教学视频需要在多个摄像机角度之间智能切换——例如演示如何修剪狗毛时，先用远景展示整体姿态，再切换到近景展示剪刀的操作细节。这种视角编排目前完全依赖人工手动完成，代价极高。

**核心挑战**在于标注数据的获取：互联网上有大量经过人工编辑的教学视频，但这些视频只包含最终选定的视角序列，被放弃的视角（"剪辑室地板上的素材"）并未保留，因此无法直接构建"最佳视角"的监督信号。

**关键洞察**：虽然这些视频没有显式的视角标注，但人类编辑者在制作时已经隐含了视角偏好——在特定的语言叙述和视觉内容下，选择了对应的 ego 或 exo 视角。因此可以通过伪标签的方式从这些视频中学习视角切换模式，然后将学到的知识迁移到有限标注的多视图场景中。

## 方法详解

### 整体框架

Switch-a-view 分为两个阶段：
1. **预训练阶段**：在大规模无标注的 HowTo100M 教学视频上学习视角切换检测任务（pretext task）
2. **微调阶段**：利用少量标注数据，将视角切换检测器迁移为视角选择器

### 关键设计

1. **视角伪标签生成器（View Pseudo-Labeler）**:

    - 功能：为无标注的教学视频自动生成 ego/exo 视角标签
    - 核心思路：首先使用场景检测器（PySceneDetect）将视频分割为连续镜头（shots），然后对每个镜头内的帧使用预训练的 ego-exo 分类器进行分类，最后通过帧级概率的平均聚合得到镜头级伪标签
    - 设计动机：直接对帧分类噪声较大，尤其是在场景边界处；结合场景检测器进行镜头级别的聚合可以有效降低噪声。分类器来自 Charades-Ego 上训练的 ego-exo 视角分类模型

2. **视角切换检测器（View-Switch Detector $D$）**:

    - 功能：在时刻 $t$ 预测未来 $\Delta$ 秒内应使用的视角类型
    - 核心思路：利用过去的视频帧 $F_{[:t]}$、叙述文本 $N_{[:t]}$、视角历史 $V_{[:t]}$ 以及下一段叙述 $N'_{(t,t+\Delta]}$ 作为输入，通过多模态融合预测下一视角：
    $D(F_{[:t]}, N_{[:t]}, V_{[:t]}, N'_{(t,t+\Delta]}) = V_{(t,t+\Delta]}$
    - 编码方式：
        - **帧编码**：使用 DINOv2 提取视觉特征，叠加视角嵌入和时间位置编码：$f_i = \mathcal{E}^F(F_i) + \mathcal{E}^V(V_i^F) + \mathcal{E}^{\mathcal{T}}(\mathcal{T}_i^F)$
        - **叙述编码**：使用 Llama 2 编码文本，同样叠加视角嵌入和时间编码：$n_i = \mathcal{E}^N(N_i) + \mathcal{E}^V(V_i^N) + \mathcal{E}^{\mathcal{T}}(\mathcal{T}_i^N)$
        - **特征聚合**：添加模态嵌入区分视觉/文本后，通过 8 层 Transformer 编码器对所有特征做 self-attention，再取 [CLS] token 输出送入 2 层 MLP 分类头
    - 设计动机：过去帧提供细粒度的视觉上下文，过去叙述提供活动步骤的高层语义，下一段叙述直接暗示所需视角（如"接下来让我们仔细看看..."暗示 ego 视角）。多模态融合比任何单一信号都更有效

3. **视角选择器（View Selector $S$）**:

    - 功能：在多视图场景中选择当前时间段的最佳视角
    - 核心思路：在检测器 $D$ 基础上扩展，额外接收预测区间内 ego 和 exo 两路同步帧，将其编码后追加到 Transformer 的输入序列中：
    $\ddot{V}_{(t,t+\Delta]} = \mathcal{H}(\mathcal{A}(f, n, n', f^G, f^X, c)[j_{\text{CLS}}])$
    - 设计动机：视角选择比视角切换需要更多信息——不仅需要预测"是否切换"，还需对比两路候选视角的内容质量。通过同时看到两路候选帧，模型可以做出更精准的判断

### 损失函数 / 训练策略

- **预训练阶段**：使用交叉熵损失训练检测器，标签来自伪标签：$\mathcal{L}^D = \mathcal{L}_{CE}(\hat{V}, \tilde{V})$
- **微调阶段**：用检测器参数初始化选择器，使用少量 Ego-Exo4D 标注数据微调：$\mathcal{L}^S = \mathcal{L}_{CE}(\ddot{V}, V)$
- 训练数据：3416 小时 HowTo100M 视频用于预训练，约 3.5 小时 Ego-Exo4D 视频（6634 样本）用于微调
- 配置：过去帧 8 秒，过去叙述 32 秒，预测窗口 $\Delta = 2$ 秒

## 实验关键数据

### 主实验

**视角切换检测**（Table 1）：

| 模型 | HT100M Acc | HT100M AUC | EgoExo4D AUC | 说明 |
|------|-----------|------------|-------------|------|
| Random | 52.0 | 52.0 | 49.3 | 随机基线 |
| Retrieval-F | 53.4 | 53.4 | 52.6 | InternVideo2 检索 |
| **Switch-a-view** | **59.4** | **63.8** | **56.4** | 本文方法 |

**视角选择**（Table 2）：

| 模型 | Accuracy | AUC | AP | 说明 |
|------|----------|-----|-----|------|
| LangView-bigData | 53.3 | 54.8 | 54.5 | SOTA，使用 98× 更多数据 |
| Ours w/o pretrain | 50.1 | 51.6 | 51.3 | 无预训练消融 |
| **Switch-a-view** | **54.0** | **57.3** | **56.0** | 本文方法 |

### 消融实验

| 配置 | HT100M AUC | EgoExo4D AUC | 说明 |
|------|-----------|-------------|------|
| 仅叙述 $N$ | 54.4 | 48.7 | 过去叙述单模态 |
| 仅下一叙述 $N'$ | 57.8 | - | 下一叙述比过去叙述更有用 |
| 仅帧 $F$ | - | - | 帧提供细粒度信息 |
| **完整模型** | **63.8** | **56.4** | 多模态融合最优 |

### 关键发现

1. 下一段叙述（$N'$）比过去叙述对预测更重要，因为它与预测时间窗口直接对齐
2. 零样本迁移到 Ego-Exo4D 即有效，说明从野外视频学到的视角偏好具有泛化性
3. 无预训练直接在少量标注上训练效果很差（51.6 vs 57.3 AUC），证明视角切换预训练是核心贡献
4. 人类标注者对最佳视角有显著共识（Cohen's kappa 0.65-0.70）

## 亮点与洞察

- **信号来源巧妙**：利用互联网上大量已编辑的教学视频作为弱监督信号，避免了昂贵的标注成本
- **任务分解合理**：先学习何时切换（pretext task），再学习切换到哪个视角，降低了直接学习视角选择的难度
- **多模态设计完整**：视觉帧+文本叙述+视角历史+时间编码，每种信号都有明确的贡献
- 标注者实验证明了"最佳视角"确实存在一致性标准，任务定义是合理的

## 局限与展望

- 目前仅考虑 ego/exo 二分类，实际场景中可能有多个 exo 相机需要进一步选择
- 伪标签依赖预训练的 ego-exo 分类器，其质量直接影响下游性能
- 预测窗口固定为 2 秒，不适应不同活动的自然节奏变化
- 在 Ego-Exo4D 上的绝对性能仍有较大提升空间（AUC 57.3%），说明任务本身非常具有挑战性

## 相关工作与启发

- 与 LangView (CVPR 2024) 的互补性值得关注：LangView 使用叙述文本做弱监督预训练，而本文使用视频编辑模式，两者结合可进一步提升
- 视角选择思路可推广到更多场景，如体育转播、在线教育、VR 内容制作
- 从 HowTo100M 学习到的视角偏好模式是否可迁移到其他领域（如烹饪→手工艺）值得探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角切换检测作为预训练任务新颖，伪标签策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融完整，人类标注实验严谨，但绝对性能有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 对自动视频编辑领域有实际意义，但绝对性能还需提升

<!-- RELATED:START -->

## 相关论文

- [Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [Multi-view Gaze Target Estimation](multi-view_gaze_target_estimation.md)
- [Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues](neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)
- [Understanding Co-speech Gestures in-the-wild](understanding_co-speech_gestures_in-the-wild.md)
- [Monocular Facial Appearance Capture in the Wild](monocular_facial_appearance_capture_in_the_wild.md)

<!-- RELATED:END -->
