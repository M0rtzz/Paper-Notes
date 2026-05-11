---
title: >-
  [论文解读] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding
description: >-
  [CVPR 2026][视频理解][流式视频理解] 提出 FluxMem，一个无需训练的流式视频理解框架，通过层级化记忆设计（短期/中期/长期）和两个自适应 token 压缩模块（TAS 去时间冗余 + SDC 去空间冗余），在丢弃 60-70% 视觉 token 的同时在 StreamingBench 和 OVO-Bench 上取得新 SOTA。
tags:
  - CVPR 2026
  - 视频理解
  - 流式视频理解
  - 层级记忆
  - 视觉token压缩
  - 自适应阈值
  - training-free
---

# FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2603.02096](https://arxiv.org/abs/2603.02096)  
**代码**: [https://github.com/YiwengXie/FluxMem](https://github.com/YiwengXie/FluxMem)  
**领域**: 视频理解  
**关键词**: 流式视频理解, 层级记忆, 视觉token压缩, 自适应阈值, training-free

## 一句话总结
提出 FluxMem，一个无需训练的流式视频理解框架，通过层级化记忆设计（短期/中期/长期）和两个自适应 token 压缩模块（TAS 去时间冗余 + SDC 去空间冗余），在丢弃 60-70% 视觉 token 的同时在 StreamingBench 和 OVO-Bench 上取得新 SOTA。

## 研究背景与动机
多模态大模型在离线视频理解中表现优异，但实际应用（机器人操控、自动驾驶、智能眼镜）需要实时处理连续视频流。流式视频理解的核心挑战是如何在有限的计算/内存预算下有效记忆长期时序上下文，并在查询到达时因果地生成响应。

**现有方法的不足**：

**KV cache 管理**（ReKV、LiveVLM）：在 LLM prefill 阶段才做去重，此前的视觉编码已消耗大量计算

**查询引导过滤**（TimeChat-Online）：依赖文本查询来选择视觉内容，但流式场景中查询可能在视频之后才到达，无法提前过滤

**固定压缩策略**：现有 token 压缩方法对所有帧使用统一的裁剪/合并策略，忽略了记忆的时间依赖性——近帧需要保留高分辨率细节用于当前推理，远帧可以更激进地压缩

**核心 idea**：模仿人类记忆的衰减特性，设计层级化记忆（短/中/长期），近期记忆保持完整，远期记忆逐级压缩。压缩阈值不用人工调参，而是基于 Otsu's method 自适应确定。

## 方法详解

### 整体框架
FluxMem 将视觉上下文分为三级记忆，以级联方式处理流式帧：
- **短期记忆 $\mathcal{M}^s$**（容量 $c_s$=8 帧）：完整保留所有视觉 token，用于当前瞬时感知
- **中期记忆 $\mathcal{M}^m$**（容量 $c_m$=64 帧）：通过 TAS 去除时间冗余后的 token
- **长期记忆 $\mathcal{M}^l$**：通过 SDC 进一步合并空间冗余后的紧凑表示

当新帧进入时，短期记忆溢出的帧经 TAS 压缩后进入中期；中期溢出的帧经 SDC 压缩后进入长期。查询到达时，三级记忆拼接送入 LLM。

### 关键设计
1. **时间邻接选择（Temporal Adjacency Selection, TAS）**:

    - 功能：在短期→中期边界去除时间冗余 token
    - 核心思路：对每个空间位置 $(h,w)$，计算该 token 与相邻帧在 $3 \times 3$ 窗口内的最小余弦距离：
    $s_{t,h,w}^{-} = \min_{(i,j) \in \mathcal{N}_{3\times 3}(h,w)} d(v_{t,h,w}, v_{t-1,i,j})$
    $s_{t,h,w}^{+} = \min_{(i,j) \in \mathcal{N}_{3\times 3}(h,w)} d(v_{t,h,w}, v_{t+1,i,j})$
      分别用 Otsu's method 计算前向/后向阈值 $\Theta_t^{-}$ 和 $\Theta_t^{+}$。token 只要对前帧或后帧之一有显著变化就保留：$(s_{t,h,w}^{-} > \Theta_t^{-}) \lor (s_{t,h,w}^{+} > \Theta_t^{+})$
    - 设计动机：(1) $3 \times 3$ 窗口搜索容忍轻微的运动/抖动，避免因微小位移丢失重要 token；(2) 双向比较（union 操作）确保"从前帧来的新内容"和"对后帧将消失的内容"都被保留；(3) 严格因果，单遍扫描，$\mathcal{O}(HW)$ 复杂度

2. **空间域合并（Spatial Domain Consolidation, SDC）**:

    - 功能：在中期→长期边界合并空间冗余区域
    - 核心思路：对 TAS 保留的 token 集合，在原始 $3 \times 3$ 空间邻域内构建稀疏图——距离 $\leq \Theta_t$（Otsu 阈值）的 token 之间连边。通过 union-find 求连通分量 $\{C_{t,k}\}_k$，每个分量替换为其均值锚点：
    $a_{t,k} = \frac{1}{|C_{t,k}|} \sum_{(i,j) \in C_{t,k}} v_{t,i,j}$
    - 设计动机：(1) 仅在 TAS 已过滤的稀疏 token 集上操作，图天然稀疏，union-find 近线性时间；(2) 局部相似区域合并为重心锚点，保留必要信息的同时大幅减少 token 数

3. **自适应 Otsu 阈值**:

    - 功能：为 TAS 和 SDC 自动确定每帧的压缩阈值
    - 核心思路：Otsu's method 找到使类间方差最大的二值化阈值：
    $\Theta_t = \arg\max_{\theta} [\omega_1(\theta)\omega_2(\theta)(\mu_1(\theta) - \mu_2(\theta))^2]$
      TAS 中分析时间相似度分数的分布，SDC 中分析空间距离分布
    - 设计动机：固定阈值在静态场景过度保留（浪费）、在动态场景过度丢弃（损失信息）。Otsu 是经典非参数方法，零额外参数，运动剧烈时自动抬高阈值（保留更多），静态时自动降低阈值（更激进压缩）

4. **主动响应触发（Proactive Response Triggering）**:

    - 功能：无额外开销地检测场景切换，触发 LLM 主动输出
    - 核心思路：复用 TAS 的后向得分统计——计算超过阈值的 token 比例 $r_t^{-} = \frac{1}{HW}\sum_{h,w} \mathbf{1}[s_{t,h,w}^{-} > \Theta_t^{-}]$，当 $r_t^{-} > \gamma$ 时触发
    - 设计动机：场景切换时大量 token 变化，$r_t^{-}$ 自然升高，零额外计算

### 损失函数 / 训练策略
- **完全无需训练**：FluxMem 是即插即用的推理时模块
- 基于 Qwen2.5-VL-7B 实现
- 在线设置：1 fps 采样，每帧 256 token，最多 256 帧
- 离线设置：1 fps，每帧 64 token，最多 1024 帧

## 实验关键数据

### 主实验

| 方法 | 类型 | StreamingBench (real-time) | OVO-Bench (real-time) | OVO-Bench (overall) | VideoMME | MLVU |
|------|------|--------------------------|----------------------|---------------------|----------|------|
| Qwen2.5-VL (baseline) | 离线 | 73.9 | 63.3 | 49.8 | 63.3 | 67.9 |
| TimeChat-Online | 训练式 | 75.3 | 61.4 | 47.6 | 63.3 | 65.4 |
| StreamForest | 训练式 | 77.3 | 61.2 | 55.6 | 61.9 | 69.6 |
| ViSpeak | 训练式 | 74.4 | 66.3 | — | — | — |
| **FluxMem** | **无训练** | **76.4** | **67.2** | **53.3** | **65.3** | **73.1** |

FluxMem 在在线任务上超越所有 training-based 方法（StreamingBench 76.4 vs StreamForest 77.3，OVO-Bench 67.2 vs ViSpeak 66.3），且在离线视频 MLVU 上达 73.1（比 baseline +5.2）。

### 消融实验

| 记忆配置 | Token 丢弃率 | MLVU | VideoMME | StreamingBench | 平均 |
|---------|-------------|------|---------|---------------|------|
| S only | 0% | 67.8 | 63.3 | 73.9 | 68.3 |
| M only | 43.2% | 69.9 | 65.5 | 74.7 | 70.0 |
| L only | 85.1% | 70.9 | 62.0 | 75.9 | 69.6 |
| S+M+L (完整) | **64.3%** | **73.1** | **65.3** | **76.4** | **71.6** |

| 效率指标 | 数据集 | Baseline | FluxMem | 改善 |
|---------|--------|----------|---------|------|
| 延迟 (ms) | OVO-Bench | 2701 | 812 | ↓69.9% |
| GPU 峰值内存 (GB) | OVO-Bench | 35.8 | 23.5 | ↓34.5% |
| 延迟 (ms) | MLVU | 3614 | 2014 | ↓44.3% |
| 准确率 | OVO-Bench | 49.8 | 53.3 | +3.5 |

### 关键发现
- **层级互补性**：M+L 组合 MLVU 得分 73.1 远超单用 M（69.9）或 L（70.9），TAS 和 SDC 分别捕捉时间变化和空间结构，互为补充
- **短期记忆对在线任务关键**：S+L 在 StreamingBench 达 77.0，高于 S alone（73.9）和 L alone（75.9），近帧细节对即时感知不可或缺
- **自适应阈值远优于固定阈值**：中期记忆中自适应阈值在 42.8% 丢弃率下达到固定阈值在 29.4% 丢弃率下的同等精度，压缩效率提高 45%
- **FluxMem 在 50-70% Token 丢弃范围内一致优于所有对比方法**（FIFO、Uniform、Random、DTD）

## 亮点与洞察
- **完全无需训练**的设计是最大亮点——即插即用，适配任何 MLLM，免除了 SFT 的数据收集和训练成本
- Otsu's method 用于 token 压缩是巧妙的类比：将"哪些 token 该保留/丢弃"转化为经典的二值分割问题
- 层级记忆设计优雅地映射了视频信息的时间衰减特性
- TAS 的双向布局 + SDC 的 union-find 合并，每帧仅增加 4.1ms 额外开销

## 局限与展望
- 层级记忆的容量划分（短期 8, 中期 64）是手动设置的，不同任务/视频可能有不同最优配置
- Otsu 假设分数分布为双峰分布，对于单调或多峰分布可能不是最优分割
- 仅基于 Qwen2.5-VL-7B 验证，对其他 MLLM（LLaVA、InternVL）的适配性未测试
- 在极高帧率（如 30fps）下的表现未知——当前实验均为 1fps
- 空间合并使用均值锚点可能丢失细节纹理信息，对需要空间精度的任务（如 OCR）可能不利

## 与相关工作的对比
- **LiveVLM / ReKV（KV cache 管理）**：这些方法在 LLM prefill 阶段才做去重，视觉编码器已经处理了所有 token，计算浪费。FluxMem 在 token 进入 LLM 之前就完成压缩，直接减少 LLM 的输入长度，从根源降低延迟
- **TimeChat-Online（查询引导过滤）**：依赖文本查询来选择视觉 token，但流式场景中查询可能远在帧到达之后。FluxMem 的 TAS/SDC 完全基于视觉内容自身的信息密度做压缩，不需要文本条件
- **StreamForest（训练式）**：StreamForest 在 StreamingBench 上达 77.3 略高于 FluxMem 的 76.4，但需要专门的 SFT 数据和训练。FluxMem 零训练即插即用，且在 OVO-Bench 和 MLVU 上反超
- **FastV / DTD（静态 token 裁剪）**：这些方法对所有帧使用统一的裁剪策略，不区分近帧/远帧。FluxMem 的层级设计让近帧保留完整细节、远帧激进压缩，更符合视频信息的时间衰减特性
- **StreamMem**：同样采用记忆层级设计，但需要训练一个记忆控制器模块。FluxMem 用经典的 Otsu 阈值替代了可学习的门控，简洁且泛化性更好

## 启发与关联
- **层级记忆范式的通用性**：短/中/长期记忆的设计不限于视频理解，可以迁移到流式文档理解（近段落保留完整→旧段落压缩摘要）、实时对话（近轮完整→旧轮压缩）等序列场景
- **Otsu 用于信息密度分割的启发**：将 "保留 vs 丢弃" 建模为二值分类是一个优雅的类比。类似地，自适应量化（如图像编码中的自适应比特分配）也可以用 Otsu 来决定每个区域的量化精度
- **与 token merging (ToMe) 的关系**：SDC 的 union-find 合并与 ToMe 的二部图匹配思路相似，但 SDC 仅在稀疏 token 集上操作效率更高。两者结合可能进一步提升压缩率
- **主动触发机制的扩展**：TAS 的副产品——场景变化检测——可用于视频分割、关键帧提取等下游任务，实现多任务共享计算

## 评分
- 新颖性: ⭐⭐⭐⭐ 层级记忆 + Otsu 自适应阈值的组合新颖，但单个组件（TAS/SDC）的技术复杂度不高
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 benchmark（2 在线 + 3 离线）、效率分析、层级消融、阈值分析、多方法对比，非常完整
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，Algorithm 伪代码完整，图表丰富且信息量大
- 价值: ⭐⭐⭐⭐⭐ training-free + 显著效率提升 + SOTA 性能，对流式视频理解领域有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)

</div>

<!-- RELATED:END -->
