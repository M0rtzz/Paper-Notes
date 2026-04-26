---
title: >-
  [论文解读] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling
description: >-
  [NeurIPS 2025][视频理解][在线音视频解析] 本文首次提出在线音视频事件解析（On-AVEP）范式，通过预测性未来建模框架 PreFM，利用伪未来序列增强当前上下文理解，同时借助模态无关的知识蒸馏和焦点时间优先策略，以仅 2.7% 的参数量超越离线 SOTA 方法 +9.3 的事件级平均 F1 分数。
tags:
  - NeurIPS 2025
  - 视频理解
  - 在线音视频解析
  - 预测性未来建模
  - 多模态融合
  - 实时视频理解
  - 知识蒸馏
---

# PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling

**会议**: NeurIPS 2025  
**arXiv**: [2505.23155](https://arxiv.org/abs/2505.23155)  
**代码**: [GitHub](https://github.com/XiaoYu-1123/PreFM)  
**领域**: 视频理解 / 音视频事件解析  
**关键词**: 在线音视频解析, 预测性未来建模, 多模态融合, 实时视频理解, 知识蒸馏

## 一句话总结

本文首次提出在线音视频事件解析（On-AVEP）范式，通过预测性未来建模框架 PreFM，利用伪未来序列增强当前上下文理解，同时借助模态无关的知识蒸馏和焦点时间优先策略，以仅 2.7% 的参数量超越离线 SOTA 方法 +9.3 的事件级平均 F1 分数。

## 研究背景与动机

音视频事件解析（AVEP）是多模态视频理解中的重要任务，需要同时处理纯音频、纯视觉和音视频联合事件。现有方法（如 UnAV、UniAV、CCNet）都是离线处理整个视频序列，虽然精确但模型巨大且需全视频输入，无法满足自动驾驶、可穿戴设备等场景的实时需求。

核心矛盾在于：在线设置下模型只能看到历史和当前信息，缺乏未来上下文导致歧义（例如一个人张嘴发声——是唱歌还是说话？），同时还要保证计算效率。

本文的切入角度是：通过预测性建模生成伪未来的多模态线索，让在线模型也能"预见"即将发生的事件，从而在保持轻量的同时获得更强的上下文理解能力。

## 方法详解

### 整体框架

PreFM 框架接收流式音频和视觉特征，输入为当前窗口 $L_c$ 长度的特征序列。核心包含三个模块：(1) 伪未来机制生成 $L_f$ 长度的预测未来序列；(2) 时间-模态交叉融合进行跨时间和跨模态的特征增强；(3) 训练阶段的模态无关鲁棒表示和焦点时间优先化策略。最终在时间步 $T$ 处输出事件预测。

### 关键设计

1. **通用混合注意力（UHA）**:

    - 作为所有融合操作的基础模块，接收目标查询序列 $Q$ 和多个上下文集合 $\{F_i\}$
    - 通过多头注意力将多个上下文信息聚合到查询中：$\text{UHA}(Q, \{F_i\}) = \text{FFN}(\text{LN}(Q + \sum_i \text{Attn}(Q, F_i, F_i)))$
    - 灵活支持自注意力、跨模态注意力和跨时间注意力的统一计算

2. **伪未来机制（Pseudo-Future Mechanism）**:

    - 首先对当前音频和视觉特征通过 UHA 进行初始跨模态融合
    - 然后使用可学习的 query token $Q^a, Q^v$ 对融合后的当前特征做注意力，生成伪未来序列 $\tilde{F}_f^a, \tilde{F}_f^v$
    - 设计动机：在线推理缺乏未来信息，通过预测建模补充关键的时序上下文

3. **时间-模态交叉融合（Temporal-Modality Cross Fusion）**:

    - 未来增强阶段：伪未来序列通过 UHA 同时与自身（自注意力）、另一模态伪未来（跨模态）和对应当前特征（跨时间）交互
    - 当前精炼阶段：增强后的伪未来信息反馈回当前表示，让当前特征获得"前瞻视野"
    - 最终共享分类头对当前和未来窗口分别生成事件预测

4. **模态无关鲁棒表示（MRR）**:

    - 使用冻结的 OnePeace 大模型将事件标签转为模态无关的文本特征作为蒸馏目标
    - 学生模型的音视频联合表示通过余弦相似度损失与教师特征对齐
    - 以轻量方式获得大模型的泛化知识，无需增加推理参数

5. **焦点时间优先化（Focal Temporal Prioritization）**:

    - 使用以当前时间 $T$ 为中心的高斯函数对不同时间步的损失进行加权
    - 越靠近当前时刻的预测权重越高，鼓励模型聚焦在最关键的当前决策上
    - 分别对当前窗口和伪未来窗口设置不同的高斯权重

### 损失函数 / 训练策略

总损失为加权 BCE 损失和 MRR 蒸馏损失的组合：$\mathcal{L} = \sum w_c \cdot \mathcal{L}_c + \sum w_f \cdot \mathcal{L}_f + \lambda \sum w \cdot \mathcal{L}_{mrr}$。训练时采用随机段采样策略，在视频中以 $L_c$ 步长加随机偏移生成训练目标时间点，增强数据多样性。60 个训练 epoch，其中前 10 个 epoch 为 warmup。

## 实验关键数据

### 主实验

**On-AVEL 任务（UnAV-100 数据集）**：

| 方法 | 特征 | 段级F1 | 段级mAP | 事件级Avg F1 | 参数量 | FLOPs |
|------|------|--------|---------|-------------|--------|-------|
| CCNet* (离线) | OnePeace | 65.0 | 70.6 | 58.3 | 238.8M | 72.1G |
| UniAV* (离线) | OnePeace | 59.2 | 70.0 | 52.9 | 130.8M | 22.7G |
| PreFM (在线) | CLIP+CLAP | 59.1 | 70.1 | 46.3 | **6.5M** | **0.4G** |
| PreFM+ (在线) | OnePeace | **62.4** | **70.6** | **51.5** | 13.8M | 0.5G |

**On-AVVP 任务（LLP 数据集）**：

| 方法 | 段级F1a/F1v/F1av | 事件级Avga/Avgv/Avgav | 参数量 |
|------|------------------|----------------------|--------|
| MM-CSE | 53.3/56.5/48.9 | 37.7/46.9/36.2 | 6.2M |
| PreFM | **60.0**/59.3/**53.3** | **46.3**/**50.6**/**41.2** | **3.3M** |

### 消融实验

| 配置 | 事件级Avg F1 | 说明 |
|------|-------------|------|
| 基础模型（无预测） | 42.1 | 仅使用当前窗口 |
| + 伪未来建模 | 44.5 | 添加未来序列预测 |
| + 交叉融合 | 45.3 | 跨时间模态增强 |
| + MRR蒸馏 | 45.8 | 知识蒸馏 |
| + 焦点时间优先化 | 46.3 | 完整PreFM |

### 关键发现
- PreFM 在线版本以 2.7% 的参数量（6.5M vs 238.8M）就超越了 CCNet 等需要全视频输入的离线方法在多项指标上的表现
- 推理速度达到 51.9 FPS（vs CCNet 的 7.5 FPS），延迟仅 19.3ms
- 伪未来建模是最关键的模块，贡献了主要的性能提升

## 亮点与洞察

- **范式创新**：首次定义并系统解决"在线音视频事件解析"问题，将 AVEL 和 AVVP 统一到在线流式处理框架中
- **效率-性能平衡极佳**：仅用 6.5M 参数在在线设置下逼近甚至超越 238.8M 参数的离线方法
- **UHA 模块设计精巧**：通过灵活的上下文列表统一了自注意力、跨模态和跨时间注意力，避免了层层堆叠不同类型注意力的开销
- **伪未来序列的跨模态交互**：不仅生成单一模态的未来预测，还通过音视频交叉关注减少伪未来的噪声

## 局限与展望

- 伪未来序列的质量取决于当前窗口的信息量，当场景发生突变时预测可能不准确
- 蒸馏目标依赖于事件标签生成文本 prompt，弱监督场景下性能可能受限
- 当前窗口长度 $L_c=10$ 秒和未来窗口 $L_f=5$ 秒为固定设置，缺乏自适应调整
- 未探索音频和视觉特征提取器的端到端微调

## 相关工作与启发

- **vs CCNet（离线）**: PreFM 在在线模式下用极少参数接近其性能，证明预测性建模可有效弥补未来信息的缺失
- **vs MAT/TPT（在线动作检测）**: PreFM 将在线视频理解从单一视觉模态扩展到音视频联合，UHA 的统一注意力设计比为每种交互分别设计模块更高效
- **对其他任务的启发**: 伪未来建模 + 焦点时间优先化的组合可迁移到其他在线感知任务（如在线目标检测、流式对话等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义 On-AVEP 问题，伪未来建模在音视频领域是新颖的应用
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、段级和事件级完整评估、参数/计算量/速度全面对比、消融充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观，问题定义明确
- 价值: ⭐⭐⭐⭐ 对实时多模态理解有重要参考价值，效率优势使其具有实际部署潜力

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](../../ICCV2025/video_understanding/hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](../../CVPR2025/video_understanding/question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[ECCV 2024\] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos](../../ECCV2024/video_understanding/spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)

<!-- RELATED:END -->
