---
title: >-
  [论文解读] Online Video Understanding: OVBench and VideoChat-Online
description: >-
  [CVPR 2025][自动驾驶][在线视频理解] 本文从评估基准、模型架构和训练策略三个角度推进在线视频理解：提出 OVBench（包含 6 大任务类型 16 个子任务的在线视频 QA 基准），设计金字塔记忆库（PMB）高效压缩流式视频信息，并通过离线到在线的渐进训练构建 4B 参数的 VideoChat-Online 模型，在 OVBench 上超越 7B 离线模型 4.2%。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "在线视频理解"
  - "流式视频"
  - "金字塔记忆库"
  - "时空感知"
  - "基准测试"
---

# Online Video Understanding: OVBench and VideoChat-Online

**会议**: CVPR 2025  
**arXiv**: [2501.00584](https://arxiv.org/abs/2501.00584)  
**代码**: [https://videochat-online.github.io/](https://videochat-online.github.io/)  
**领域**: 自动驾驶  
**关键词**: 在线视频理解, 流式视频, 金字塔记忆库, 时空感知, 基准测试

## 一句话总结
本文从评估基准、模型架构和训练策略三个角度推进在线视频理解：提出 OVBench（包含 6 大任务类型 16 个子任务的在线视频 QA 基准），设计金字塔记忆库（PMB）高效压缩流式视频信息，并通过离线到在线的渐进训练构建 4B 参数的 VideoChat-Online 模型，在 OVBench 上超越 7B 离线模型 4.2%。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在离线视频理解上取得了显著进展，但现实应用（自动驾驶、AR 眼镜、人机交互）需要对连续在线视频流进行实时处理。现有模型和基准测试大多面向离线场景。

**现有痛点**：(1) 现有视频基准（如 MVBench、VideoMME）都在离线模式下评估，无法反映在线场景的独特需求——时间依赖的上下文、过去/当前/未来的多时态推理、实时时空交互；(2) 现有在线模型（Flash-VStream、VideoLLM-Online）缺乏合理的架构设计来平衡空间细节和时间跨度；(3) 没有专门针对在线视频的训练策略。

**核心矛盾**：在线视频流产生无限量的视觉信息，模型需要像人类认知一样保留关键信息并遗忘冗余信息，同时保持实时响应能力。

**本文目标**：构建完整的在线视频理解研究体系——从评估基准到模型架构再到训练范式。

**切入角度**：将在线视频的时态划分为过去/当前/未来三个维度，在此基础上定义 6 种核心能力（空间感知、时间感知、时空感知、过去记忆、时间幻觉验证、未来预测），系统化地设计评估和训练方案。

**核心 idea**：用金字塔式的多层记忆库实现空间-时间的渐进抽象——近帧保留高分辨率空间细节，远帧压缩为低分辨率时间摘要——配合离线到在线的课程学习训练策略。

## 方法详解

### 整体框架
VideoChat-Online 基于 InternVL2-4B 构建（InternViT-300M 视觉编码器 + Phi-3 语言模型）。流式视频输入经金字塔记忆库（PMB）压缩后送入 LLM。PMB 包含多层队列，每层有不同的采样率和分辨率：近帧保留完整空间细节，远帧逐步降低空间分辨率但保持时间覆盖。当某层满时，通过自适应帧驱逐（保留最不相似的帧）将被淘汰的帧降分辨率后传入下一层。训练采用"离线 → 在线"渐进范式。

### 关键设计

1. **金字塔记忆库（PMB）**:

    - 功能：在有限的视觉 token 预算内平衡空间和时间信息
    - 核心思路：将记忆分为 $n$ 层 $\{m_i\}$，每层有采样率 $r_i$（逐层递增）和分辨率 $\text{Res}_i = \text{Res}_1 / \beta^{i-1}$（逐层递减，$\beta=2$）。三个操作：(1) 流式写入：按采样率接收帧直到容量 $C_i$ 满；(2) 帧驱逐+下传：找到余弦相似度最高的相邻帧对，淘汰较旧的帧，平均池化降分辨率后传入 $m_{i+1}$；(3) 读出：按时间顺序读取所有层的帧。实际配置：3 层记忆，采样率 $\{1, 2, 8\}$，每帧 token 数 $\{256, 64, 16\}$
    - 设计动机：近帧空间细节对当前感知至关重要（高分辨率），远帧主要提供时间上下文（低分辨率够用）。相似帧驱逐策略有效去除冗余

2. **KVCache 兼容设计**:

    - 功能：避免记忆更新时的全量重计算
    - 核心思路：帧 token 进入记忆库的同时也写入 KVCache。帧驱逐时，删除被淘汰帧时间戳之后的所有 KVCache 条目：$\text{KVCache} \leftarrow \text{KVCache} \setminus \{t_i | t_i > \min(t_{f_a}, t_{f_b})\}$
    - 设计动机：现有内存压缩方法（如 MovieChat、FlashVStream）每次更新都需要重新处理整个压缩记忆，产生计算瓶颈。PMB 与 KVCache 同步，增量更新效率高

3. **离线到在线渐进训练**:

    - 功能：逐步增强模型的在线时空理解能力
    - 核心思路：收集 96K 高质量时空标注数据（涵盖密集字幕、步骤定位、目标跟踪等），转换为交错对话格式——沿时间线精心放置问题，区分过去/当前/未来时态。先在离线视频数据上训练建立基础视频理解能力，再联合在线数据微调
    - 设计动机：直接在在线数据上训练难以同时优化时空理解和时间/框预测能力，课程学习策略更稳定

### 损失函数 / 训练策略
标准的自回归语言建模损失。训练数据混合：离线数据（VideoChat2-IT、STAR、PerceptionTest）+ 图像数据（ShareGPT4V/4o）+ 多图数据（LLaVA-OneVision）+ 在线时空数据（96K）。输入 1 fps 采样，最大 64 帧。

## 实验关键数据

### OVBench 主实验

| 模型 | 参数量 | 设置 | FP | THV | PM | SP | STP | TP | 平均 |
|------|--------|------|-----|-----|-----|-----|-----|-----|------|
| Qwen2-VL | 7B | 滑窗 | 49.5 | 52.5 | 57.2 | 35.3 | 49.4 | 35.8 | 49.7 |
| Flash-Vstream | 7B | 流式 | 29.5 | 47.3 | 28.3 | 24.7 | 21.4 | 27.4 | 31.2 |
| **VideoChat-Online** | **4B** | **流式** | **46.8** | **61.4** | **55.7** | **54.1** | **48.5** | **56.9** | **54.9** |

### 消融实验

| 配置 | OVBench Avg | 说明 |
|------|-------------|------|
| 无 PMB（固定滑窗） | 47.2 | 缺乏长程记忆 |
| 单层记忆 | 49.8 | 无空间-时间分层 |
| 仅离线训练 | 48.5 | 缺乏在线时空数据 |
| **完整模型** | **54.9** | PMB + 渐进训练 |

### 关键发现
- VideoChat-Online (4B) 在流式设置下以 54.9% 超越 7B 离线模型 Qwen2-VL (49.7%)，且参数量更小
- 比最佳流式竞争者 Flash-Vstream (31.2%) 高出 23.7 个百分点，说明现有在线模型的架构和训练严重不足
- PMB 在过去记忆（PM）和时间幻觉验证（THV）任务上增益最大，因为这些任务依赖长程时间信息
- 时间感知（TP）任务中 Object Existence State 子任务的提升最显著（69.9% vs 下一最佳 46.9%），说明 PMB 的帧驱逐机制有效保留了关键时间信息

## 亮点与洞察
- **系统化的在线视频研究**：从基准、架构到训练的完整体系，填补了在线视频理解研究的空白
- **空间-时间渐进抽象**：金字塔记忆库的设计符合人类认知——对近期事件记忆细节，对远期事件保留梗概——是直觉且高效的方案
- **4B 即可超越 7B**：证明了针对性的架构设计和训练策略比盲目增加参数更重要

## 局限与展望
- OVBench 目前主要基于已有数据集改造，覆盖场景有限（缺少对话交互、多模态输入等）
- PMB 的帧驱逐策略基于相邻帧相似度，可能淘汰重要但与邻帧相似的关键帧
- 1 fps 的采样率对快速动作理解可能不足
- 未来可探索更细粒度的注意力机制替代简单的池化降分辨率

## 相关工作与启发
- **vs Flash-Vstream**: Flash-Vstream 用可学习记忆模块压缩流信息，但缺乏分层设计和专用训练数据，性能远低于 VideoChat-Online
- **vs VideoLLM-Online**: 该先驱工作在 OVBench 上几乎失效（9.6%），因为受限于单帧视觉 token 输入
- 金字塔记忆库的分层压缩思路可推广到其他需要处理长序列的场景（如长文档理解）

## 评分
- 新颖性: ⭐⭐⭐⭐ PMB 设计合理，OVBench 填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ 全面的基准评估、离线/在线双对比、消融研究
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务定义系统化
- 价值: ⭐⭐⭐⭐⭐ 为在线视频理解提供了完整的研究基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[CVPR 2025\] InteractionMap: Improving Online Vectorized HDMap Construction with Interaction](interactionmap_improving_online_vectorized_hdmap_construction_with_interaction.md)
- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](../../ICCV2025/autonomous_driving/embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

</div>

<!-- RELATED:END -->
