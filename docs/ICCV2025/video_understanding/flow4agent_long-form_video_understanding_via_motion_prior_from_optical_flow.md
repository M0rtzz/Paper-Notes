---
title: >-
  [论文解读] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow
description: >-
  [ICCV 2025][视频理解][视频理解] Flow4Agent 首次将光流运动先验引入 LLM-based 视频理解，通过时域粒度优化（TGO）利用粗粒度光流聚类视频事件并用语义先验过滤冗余场景，通过运动 Token 剪枝（MTP）利用细粒度光流去除帧内静态冗余 token，在 VideoMME/MLVU/LongVideoBench 等长视频基准上取得领先表现。
tags:
  - ICCV 2025
  - 视频理解
  - 光流
  - motion prior
  - 剪枝
  - MLLM
  - temporal granularity
  - key content extraction
---

# Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow

**会议**: ICCV 2025  
**arXiv**: [2510.05836](https://arxiv.org/abs/2510.05836)  
**代码**: 待确认  
**领域**: 长视频理解 / 多模态大语言模型 / 光流先验  
**关键词**: long-form video understanding, optical flow, motion prior, token pruning, MLLM, temporal granularity, key content extraction

## 一句话总结
Flow4Agent 首次将光流运动先验引入 LLM-based 视频理解，通过时域粒度优化（TGO）利用粗粒度光流聚类视频事件并用语义先验过滤冗余场景，通过运动 Token 剪枝（MTP）利用细粒度光流去除帧内静态冗余 token，在 VideoMME/MLVU/LongVideoBench 等长视频基准上取得领先表现。

## 研究背景与动机

多模态大语言模型（MLLM）在短视频理解上已取得显著进展，但面对小时级长视频时存在根本挑战：

**均匀采样导致严重信息丢失**：对于一小时视频，均匀采样意味着每分钟分配不到一帧，大量关键内容被遗漏。

**密集采样带来严重冗余**：长视频中存在大量时间维度（不相关帧）和空间维度（同一场景内重复内容）的冗余信息，可能淹没 LLM 的有限上下文窗口。

**现有语义优先方法的局限**：目前提取关键内容的方法高度依赖语义先验（如 CLIP 检索、dense caption），存在两个问题：(a) 严重依赖用户指令的信息量，query 细节不足时失效；(b) 受制于先验模型本身的精度，CLIP 或 captioning 模型的错误会级联传导。

Motion 信息（光流）作为一种被忽视的视频理解先验，天然描述了场景中的动态变化，既不依赖用户指令的详细程度，也不需要密集字幕生成，可以以更低成本获取更鲁棒的关键视频内容。

## 方法详解

### 整体架构
Flow4Agent 基于标准视频 MLLM（如 LLaVA-Video-Qwen），在输入端增加两个模块来处理帧间和帧内冗余：TGO 处理帧间冗余（选择关键事件），MTP 处理帧内冗余（剪枝 token），两者协作从粗到细利用光流先验。

### 模块一：时域粒度优化 (TGO)

TGO 分为两个阶段：

**动态事件分割 (Dynamic Event Split, DES)**：
- **第一阶段（粗筛）**：将视频帧转到 HSV 色彩空间（对光照变化不敏感），计算相邻帧的均方误差，超过阈值θ标记为候选事件边界。
- **第二阶段（精确分割）**：对每个候选边界，在时间窗口内计算 M=3 帧的光流（使用 SeaRAFT，仅需少量迭代获取粗光流），若窗口内最大光流幅值超过阈值η，则确认为最终事件边界。
- 通过这种运动感知的分割，视频被划分为若干事件单元，每个单元内语义场景基本一致。

**事件中心跨模态查询 (Event-Center Cross-modal Query, ECQ)**：
- 选取每个事件的中间帧作为锚点帧，利用语义先验（SigLiP）与用户指令进行相似度计算。
- **关键创新**：不直接取 top-k 最相似的帧，而是引入统计假设检验约束：定义每个事件的显著性 α(Si)（基于 softmax 归一化的相似度），要求选出的事件集 S_out 满足 p-value < 0.05，同时最小化事件数量。
- 效果：query 信息丰富时，高度相关事件被独立选出；query 缺乏细节时，保守策略确保不遗漏重要场景，仅过滤完全无关的低显著性场景。

### 模块二：运动 Token 剪枝 (MTP)

针对帧内冗余，同一场景中大部分背景不变，只有少量前景变化是理解关键：

1. 计算当前帧与下一帧的精细光流（SeaRAFT，12 次迭代）。
2. 通过单应性矩阵补偿消除相机运动干扰。
3. 利用 U2-Net 显著性检测获取主要运动区域 mask。
4. 在运动区域内，按光流幅值取 top-50% 的像素对应的 token，生成最终有效 token mask。
5. 锚点帧保留全部 token 以维护完整上下文，仅对相邻帧应用 MTP。
6. 剪枝释放的 token 预算用于增加采样帧数，保持视觉上下文长度不变。

### 训练与推理
- 无需额外训练，Flow4Agent 是推理时的采样与剪枝策略，可即插即用到不同基础 MLLM。
- 图像输入分辨率 336，LLM 上下文长度 8k，初始采样 64 帧。
- 实验在 2 × A100 GPU 上进行。

## 实验关键数据

### 主要基准测试

| 模型 | 参数 | NextQA | EgoSchema | PercepTest | MLVU | L-VideoBench | VideoMME-Long | VideoMME-Overall |
|------|------|--------|-----------|------------|------|-------------|---------------|------------------|
| LLaVA-Video | 7B | 83.2 | 57.3 | 67.9 | 70.8 | 58.2 | 50.6 | 62.6 |
| Apollo | 7B | - | - | 67.3 | 70.9 | 58.5 | - | 61.3 |
| GPT-4V | - | - | 55.6 | - | - | 59.1 | 56.9 | 60.7 |
| **Flow4Agent** | **7B** | **84.0** | **61.4** | **69.6** | **71.4** | **60.4** | **54.2** | **64.7** |

- 在长视频基准（VideoMME-Long）上超越 LLaVA-Video 3.6%，超越 LLaVA-OneVision 7.5%
- 作为 7B 模型，在多数指标上超越 GPT-4V

### 跨模型泛化性

| 基础模型 | 原始 Overall | +Flow4Agent Overall | Long 提升 |
|----------|-------------|--------------------:|----------:|
| LLaVA-NeXT (7B) | 44.9 | 47.0 | +3.7 |
| LLaVA-OneVision (7B) | 58.2 | 59.9 | +3.2 |
| Qwen2-VL (7B) | 61.7 | 63.9 | +2.7 |
| LLaVA-Video (7B) | 62.6 | 64.7 | +3.6 |
| LLaVA-Video (72B) | 67.1 | 69.0 | +2.0 |

所有模型一致提升，长视频提升最显著。

### 消融实验

| DES | ECQ | MTP | Short | Medium | Long | Overall |
|:---:|:---:|:---:|------:|-------:|-----:|--------:|
| | | | 75.9 | 61.2 | 50.6 | 62.6 |
| ✓ | | | 77.0 | 61.7 | 50.8 | 63.2 |
| | ✓ | | 75.8 | 62.3 | 52.0 | 63.4 |
| ✓ | ✓ | | 77.1 | 62.2 | 52.9 | 64.0 |
| | | ✓ | 75.9 | 61.5 | 52.4 | 63.3 |
| ✓ | ✓ | ✓ | 77.2 | 62.6 | 54.2 | 64.7 |

- DES 主要提升短视频，ECQ 主要提升长视频，MTP 进一步增强长视频理解
- 三模块互补，联合使用效果最优

### 光流模型选择
Sea-RAFT (4 iter TGO / 12 iter MTP) 达到最优，比 NeuFlow 和 StreamFlow 分别高 1.2% 和 0.3%（Long 类别）。

## 亮点与洞察

- **首创性**：第一个将光流运动先验用于 LLM-based 视频理解的工作，开辟了运动信息辅助长视频理解的新方向。
- **粗细结合的光流使用**：TGO 用粗粒度光流（4 iter）快速聚类事件，MTP 用细粒度光流（12 iter）精确剪枝 token，计算开销分配合理。
- **统计假设检验的事件选择**：不是简单取 top-k，而是用 p-value < 0.05 的约束自适应选择事件数量，优雅地平衡了信息保留与冗余去除。
- **模型无关的即插即用设计**：无需训练，可直接增强任意视频 MLLM，从 7B 到 72B 一致有效。
- **帧效率极高**：在帧数受限时优势更加明显，证明运动先验真正帮助模型"选对帧"而非"多看帧"。

## 局限与展望

- 光流计算引入额外推理开销（SeaRAFT + U2-Net），论文未报告端到端延迟对比。
- HSV 阈值 θ 和光流阈值 η 需要手动设定，对不同类型视频可能需要调整。
- p-value = 0.05 的显著性水平选择较为固定，未讨论不同阈值的敏感性。
- MTP 的 top-50% 保留率固定，对运动稀少或密集场景可能不是最优比例。
- 未在流式推理或在线场景中验证，当前设计假设可访问完整视频。
- 光流方法本身对快速运动或遮挡场景可能不可靠，极端情况下可能引入噪声。

## 相关工作与启发

- **vs. VideoAgent**：VideoAgent 用 GPT-4 做密集字幕 + CLIP 检索关键帧，依赖强语义先验且成本高。Flow4Agent 用运动先验替代部分语义依赖，更鲁棒且更高效。
- **vs. LongVU**：LongVU 用 DINOv2 过滤空间冗余 token，但忽略运动信息。Flow4Agent 的 MTP 利用光流更精确地识别动态前景。
- **vs. MovieChat/Flash-VStream**：采用流式记忆/滑动窗口处理长视频。Flow4Agent 从信息选择角度出发，在有限上下文内塞入更多关键信息。
- 运动先验在传统视频理解（动作识别、数据筛选）中已有应用，本文首次将其迁移到 MLLM 范式，启发未来更多模态先验（如音频、深度）的融合探索。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次引入光流先验到 LLM 视频理解，TGO 中的假设检验事件选择设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个基准，5 个基础模型，详细消融（组件、帧数、光流模型、语义模型）
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统，图示直观
- 价值: ⭐⭐⭐⭐ 即插即用且一致有效，对长视频理解领域有实质推动

<!-- RELATED:START -->

## 相关论文

- [PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_o.md)
- [FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)

<!-- RELATED:END -->
