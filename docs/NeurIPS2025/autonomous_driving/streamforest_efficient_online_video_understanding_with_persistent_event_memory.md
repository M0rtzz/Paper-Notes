---
title: >-
  [论文解读] StreamForest: Efficient Online Video Understanding with Persistent Event Memory
description: >-
  [NeurIPS 2025][自动驾驶][流式视频理解] 本文提出 StreamForest 架构，通过"持久事件记忆森林"将流式视频帧自适应组织为多棵事件级树结构，结合"细粒度时空窗口"捕捉短期视觉线索，在 StreamingBench 上达到 77.3% 准确率…
tags:
  - "NeurIPS 2025"
  - "自动驾驶"
  - "流式视频理解"
  - "持久事件记忆"
  - "记忆树结构"
  - "视觉token压缩"
  - "多模态大模型"
---

# StreamForest: Efficient Online Video Understanding with Persistent Event Memory

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.24871](https://arxiv.org/abs/2509.24871)  
**代码**: [GitHub](https://github.com/MCG-NJU/StreamForest)  
**领域**: 自动驾驶 / 在线视频理解  
**关键词**: 流式视频理解, 持久事件记忆, 记忆树结构, 视觉token压缩, 多模态大模型

## 一句话总结

本文提出 StreamForest 架构，通过"持久事件记忆森林"将流式视频帧自适应组织为多棵事件级树结构，结合"细粒度时空窗口"捕捉短期视觉线索，在 StreamingBench 上达到 77.3% 准确率，并在极端压缩（仅 1024 visual tokens）下仍保留 96.8% 的性能。

## 研究背景与动机

多模态大语言模型（MLLMs）在离线视频理解上取得了显著进展，但在实时流式视频场景中面临两大挑战：(1) 持续到达的视频帧带来的历史特征存储压力；(2) 实时时空推理能力不足。

现有流式视频处理策略存在明显缺陷：采样阶段压缩（如大幅丢弃帧）导致细粒度时空推理能力丧失；存储阶段压缩（基于帧间相似度合并）容易因背景噪声遗漏关键前景动作，且过度局部合并引入时空不规则性。

本文的切入角度是在事件语义层面进行记忆管理——将视频自然分割为事件段，构建层次化的事件树结构，通过多维度惩罚函数引导自适应合并，既保留语义丰富性又控制存储开销。同时引入细粒度时空窗口关注当前时刻的详细视觉特征。

## 方法详解

### 整体框架

StreamForest 以 1 FPS 处理流式视频帧，包含两个核心组件：(1) 细粒度时空窗口（FSTW）负责当前时刻的高分辨率感知和短期记忆；(2) 持久事件记忆森林（PEMF）负责长期历史的层次化存储和自适应压缩。视觉编码器使用 SigLiP-so400M，LLM 使用 Qwen2-7B。收到用户查询时，PEMF 所有根节点特征和 FSTW 的全部视觉特征一起送入 LLM。

### 关键设计

1. **细粒度时空窗口（FSTW）**:

    - 实时感知：从当前帧直接采样高分辨率视觉特征（729 tokens），编码时空位置信息
    - 短期时空记忆：维持 $t_s$ 秒的帧缓存（18 帧，每帧 128 tokens），新帧到达时旧帧沿空间维度压缩
    - 计算帧间相似度，用于后续事件级分割
    - 缓存溢出时，通过局部最小帧间相似度位置切分出"元事件"（meta-event），下放到 PEMF
    - 元事件是一组相似连续帧的视觉 token 集合，作为 PEMF 的独立节点

2. **持久事件记忆森林（PEMF）**:

    - 与传统帧级压缩不同，PEMF 在事件语义层面进行层次化组织
    - 通过树结构管理事件节点：当 long-term memory token 数量超过上限 $L_q$ 时，选择惩罚最低的相邻节点对进行合并
    - 合并使用 ToMe（Token Merging）方法，将选中节点对的 visual tokens 压缩到原来总数的一半
    - 三重惩罚函数共同引导合并决策，确保自适应性

3. **三重罚函数设计**:

    - **相似度惩罚 $P_s$**: 基于双部图匹配计算两个事件节点间 token 的余弦相似度，选 top-k 最高相似分的均值，$P_s = 1 - \text{avg}$。鼓励合并高度相似的冗余事件
    - **合并次数惩罚 $P_m$**: $P_m = (c_i + c_{i+1}) / (2c_{max})$。惩罚被反复合并的节点，防止累积信息损失导致的时空不一致
    - **时间距离惩罚 $P_t$**: $P_t = 1 - (d_i + d_{i+1})/2$。越近的事件保留越详细，越远的允许更激进压缩
    - 总惩罚：$P = w_s P_s + w_m P_m + w_t P_t$（默认 0.4, 0.4, 0.2）
    - 退化分析：仅用 $P_s$ 退化为相似度压缩；仅用 $P_m$ 退化为均匀下采样；仅用 $P_t$ 退化为 FIFO

4. **OnlineIT 训练数据集**:

    - OnlineIT-general（32K）：整合多个流式视频理解数据集，解决时空分布偏移导致的幻觉
    - OnlineIT-drive（89K）：自动驾驶场景的流式 QA 数据，覆盖实时定位、静态/动态交通实体理解和风险评估

5. **ODV-Bench 基准**:

    - 面向自动驾驶的流式视频理解基准，包含静态目标、动态目标和多智能体交互事件三类任务
    - 半自动构建流水线：YOLO 检测 + VLLM 标注 + 人工验证

### 损失函数 / 训练策略

五阶段训练策略：前三阶段遵循离线长视频 MLLM 训练范式（VideoChat-Flash），第四阶段用 OnlineIT 微调得到基础 StreamForest，可选第五阶段用 OnlineIT-Drive 微调得到 StreamForest(FT-drive)。32 张 A100 GPU 训练。

## 实验关键数据

### 主实验

**在线视频理解基准**：

| 方法 | 规模 | StreamingBench | OVBench | OVO-Bench |
|------|------|---------------|---------|-----------|
| VideoChat-Online | 4B | - | 62.9 | - |
| Dispider | 7B | - | 52.7 | - |
| Flash-VStream | 7B | - | 40.2 | - |
| StreamForest | 7B | **77.3** | **62.3** | **55.6** |

**ODV-Bench（自动驾驶场景）**：

| 方法 | 静态目标Avg | 动态目标Avg | 事件级Avg | 总体 |
|------|-----------|-----------|---------|------|
| Qwen2.5-VL-7B | 48.3 | 57.5 | 59.4 | 55.6 |
| StreamForest | 51.5 | 62.3 | 63.8 | 59.9 |
| StreamForest(FT-drive) | **62.6** | **64.0** | **67.5** | **65.0** |
| Human | 95.9 | 88.2 | 92.5 | 91.4 |

### 消融实验

| 配置 | 平均准确率 | 说明 |
|------|----------|------|
| 默认 8192 tokens | 100% (基线) | 完整设置 |
| 4096 tokens | ~99% | 轻度压缩几乎无损 |
| 2048 tokens | ~98% | 中度压缩保持良好 |
| 1024 tokens | **96.8%** | 极端压缩仍保留绝大部分性能 |
| 仅 $P_s$ | 退化 | 相似度压缩 |
| 仅 $P_m$ | 退化 | 类似均匀下采样 |
| 仅 $P_t$ | 退化 | 类似 FIFO |

### 关键发现
- StreamForest 在极端压缩（1024 tokens）下仅损失 3.2% 性能，证明事件级记忆管理的有效性
- 在离线视频基准上也能匹配甚至超越 SOTA 离线模型，说明流式处理不牺牲理解质量
- 三重惩罚函数中，合并次数惩罚和相似度惩罚的权重最高（各 0.4），表明防止过度合并和冗余消除同等重要
- 自动驾驶微调（FT-drive）能显著提升驾驶场景表现（+5.1 整体准确率），但与人类仍有 26 个百分点差距

## 亮点与洞察

- **事件级记忆树**设计优雅，与人类对视频的认知方式（以事件为单位记忆）高度一致
- **三重惩罚函数**的设计兼顾了内容冗余、信息保真和时间重要性，且参数可调控退化为多种已知策略
- **96.8% 保留率**的极端压缩实验是说服力极强的结果，直接展示了方法的鲁棒性
- **ODV-Bench** 填补了自动驾驶流式视频理解评估的空白

## 局限与展望

- 固定 1 FPS 处理速率可能无法满足更高帧率的应用需求（如快速运动场景）
- 事件边界检测基于帧间相似度的局部最小值，对渐变场景可能不够鲁棒
- ToMe 合并虽高效但会持续丢失细节，长时间运行后早期事件的信息可能严重降级
- 当前仅评估了 7B 规模模型，更大规模模型的效果未知
- ODV-Bench 与人类水平差距巨大（59.9 vs 91.4），自动驾驶场景的理解仍需大幅提升

## 相关工作与启发

- **vs VideoChat-Online**: StreamForest 在所有在线基准上全面超越，得益于事件级记忆管理（vs VideoChat-Online 的静态层次记忆）
- **vs Flash-VStream**: Flash-VStream 使用相似度压缩策略，在 PEMF 中仅相当于 $P_s$ 单因子的退化版本
- **对流式 AI Agent 的启发**: 事件级记忆管理可直接应用于需要长期记忆的 AI Agent（如机器人、直播助手等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 事件记忆森林是新颖的记忆管理范式，三重惩罚函数设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个在线基准 + 离线基准全面评测，极端压缩实验令人印象深刻
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽，但部分公式符号可以更统一
- 价值: ⭐⭐⭐⭐⭐ 对流式视频理解领域有重要推动作用，ODV-Bench 和 OnlineIT 是有价值的社区资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Online Video Understanding: OVBench and VideoChat-Online](../../CVPR2025/autonomous_driving/online_video_understanding_ovbench_and_videochat-online.md)
- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](../../ICLR2026/autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[NeurIPS 2025\] Aha: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead](aha_predicting_what_matters_next_online_highlight_detection.md)

</div>

<!-- RELATED:END -->
