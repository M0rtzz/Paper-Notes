---
title: >-
  [论文解读] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding
description: >-
  [ICCV 2025][视频理解][Video Temporal Grounding] 提出TimeExpert——首个基于MoE的Video-LLM框架，通过**任务感知动态门控**和**token自适应路由**将时间戳、显著性分数和文本描述路由到专门的专家，配合任务依赖辅助损失，在Dense Video Captioning、Moment Retrieval和Video Highlight Detection三类VTG任务上全面超越SOTA。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Video Temporal Grounding"
  - "MoE"
  - "动态路由"
  - "Video-LLM"
  - "Dense Video Captioning"
  - "Moment Retrieval"
---

# TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding

**会议**: ICCV 2025  
**arXiv**: [2508.01699](https://arxiv.org/abs/2508.01699)  
**代码**: [项目页面](https://mwxely.github.io/projects/yang2025time/index)  
**领域**: 视频理解 / 时序定位  
**关键词**: Video Temporal Grounding, MoE, 动态路由, Video-LLM, Dense Video Captioning, Moment Retrieval

## 一句话总结

提出TimeExpert——首个基于MoE的Video-LLM框架，通过**任务感知动态门控**和**token自适应路由**将时间戳、显著性分数和文本描述路由到专门的专家，配合任务依赖辅助损失，在Dense Video Captioning、Moment Retrieval和Video Highlight Detection三类VTG任务上全面超越SOTA。

## 研究背景与动机

Video Temporal Grounding (VTG) 旨在根据文本查询精确定位视频中的事件时间段。VTG输出包含三个异质组件：**时间戳**、**显著性分数**和**文本描述**。现有方法面临根本性局限：

### 现有方法的三层缺陷

**通用Video-LLM的时间粗粒度**：VideoQA等粗粒度任务的成功难以迁移到需要精确时间定位的VTG任务——缺乏显式时间建模机制

**VTG专用方法的共享参数瓶颈**：TimeChat、TRACE等方法虽引入时间token，但将时间戳、分数和文本token**不加区分地**通过同一LLM处理，共享参数导致**任务干扰**：
   - 时间戳预测需要精确的数值回归能力
   - 显著性评分需要全局重要性判断
   - 文本生成需要语义理解和语言组织
   这三种能力要求截然不同的特征表示

**静态计算分配**：所有token获得相同的计算资源，忽略了不同任务token重要性的差异

### 隐式任务偏好的发现

TimeExpert的动机源于一个关键观察：即使在**未显式训练**专家特化的vanilla MoE中，某些专家已经表现出对特定任务token的隐式偏好（图4）。例如某个专家持续被score token激活。这暗示**显式强化**这一偏好可以大幅提升性能。

## 方法详解

### 整体框架

TimeExpert的核心改进集中在LLM backbone的替换——从单一LLM换为MoE解码器，同时引入独立的时间编码器、分数编码器和对应的解码头。

1. **视觉编码器**：轻量ViT (438M参数)，每帧压缩为8个visual token（slot-based token compression）
2. **时间/分数编码器**：独立的tokenizer（11个数字token + 分隔token + 切换token）
3. **MoE解码器**：替代单一LLM，实现动态专家路由

### 任务感知动态门控 (Task-aware Dynamic Gating)

**Vanilla MoE的问题**：固定top-k选择缺乏灵活性，且对所有token一视同仁。

TimeExpert的门控函数引入两个创新：

**1. 余弦相似度代替线性投影**：

$$s(\mathbf{x}) = \cos(\mathbf{x}, \mathbf{W}_g)$$

**2. 任务激活率加权**：

$$g(\mathbf{x}) = \text{sign}\left(\sigma\left(\frac{s(\mathbf{x}) + \alpha A_t}{1 + \alpha}\right) - \sigma(\mathbf{G})\right)$$

其中：
- $A_t$：该任务token类型的**历史激活率**——被频繁激活的专家更可能接收同类token
- $\alpha$：任务重要性缩放系数
- $\mathbf{G} \in \mathbb{R}^K$：**可学习阈值**——只有相似度超过阈值才路由到该专家
- 通过straight-through estimator使sign函数可微

**关键效果**：不同token激活的专家数量可以不同（adaptive-k），时间戳token可能激活较多专家（需要精确处理），而文本token可能激活较少。

### Token自适应路由

路由机制包含三个动态组件：

**1. 任务级路由记录**：
- 记录每个专家的激活时间戳 $\mathbf{R}_E \in \mathbb{R}^K$
- 聚合未被激活的token嵌入 $\mathbf{R}_S \in \mathbb{R}^d$
- 维护每种任务token的激活率 $A_t$

**2. 自适应专家添加**：当大量任务token无法激活任何专家时，添加新专家：

$$\mathbf{W}_{g,K+1} = \frac{\mathbf{R}_S}{\|\mathbf{R}_S\|}, \quad \mathbf{G}_{K+1} = 0$$

新专家的表示向量初始化为未匹配token的平均嵌入。

**3. 冗余专家剪枝**：激活率低于阈值 $\tau_{\min}$ 的专家被移除：

$$\mathcal{E}_{\text{remove}} = \{e \mid A_e < \tau_{\min}\}$$

### 任务依赖辅助损失

$$\mathcal{L}_{\text{aux}} = \lambda_1 \sum_{e=1}^{K}\left(\frac{A_e}{\sum_j A_j} - \frac{N_e}{\sum_j N_j}\right)^2 + \lambda_2 \sum_{e=1}^{K}\|\mathbf{w}_{g,e}\|_2^2$$

- **左项（任务感知集中）**：鼓励高激活率专家处理更多同类task token
- **右项（激活正则化）**：防止单个专家过度激活
- 与传统负载均衡损失的区别：不追求均匀分配，而是**强化专业化**

### 三阶段训练

| 阶段 | 目标 | 数据规模 |
|------|------|---------|
| Stage 1: 任务模块预训练 | 视觉压缩层+任务编码器+任务头 | 1.9M |
| Stage 2: MoE解码器预训练 | 专家路由对齐VTG任务token | 0.9M |
| Stage 3: 监督微调 | 全模型联合优化 | 2.3M |

## 实验

### 零样本VTG性能 (表2)

| 方法 | 激活参数 | DVC-SODAc | DVC-F1 | MR-R@1₀.₅ | MR-R@1₀.₇ | VHD-mAP | VHD-HIT@1 |
|------|---------|-----------|--------|-----------|-----------|---------|-----------|
| TimeChat | 7B | 1.2 | 12.6 | 32.2 | 13.4 | 14.5 | 23.9 |
| TRACE | 7B | 2.2 | 22.4 | 40.3 | 19.4 | 26.8 | 42.7 |
| **TimeExpert** | **~4-6B** | **2.5** | **23.6** | **42.8** | **20.3** | **29.6** | **46.9** |

TimeExpert在**更少激活参数**下全面超越TRACE：MR R@1₀.₅ +2.5%，VHD HIT@1 +4.2%。

### 微调VTG性能 (表3)

| 方法 | DVC-CIDEr | DVC-F1 | MR-R@1₀.₅ | MR-R@1₀.₇ |
|------|-----------|--------|-----------|-----------|
| TRACE | 35.5 | 31.8 | 61.7 | 41.4 |
| **TimeExpert** | **39.0** | **33.5** | **64.1** | **43.3** |

CIDEr +3.5, F1 +1.7, R@1₀.₅ +2.4。

### 消融实验 (表5)

| 配置 | DVC-SODAc | MR-R@1₀.₅ | VHD-HIT@1 |
|------|-----------|-----------|-----------|
| w/o token-adaptive routing | 2.1 | 40.5 | 42.6 |
| w/o task-dependent loss | 2.4 | 41.3 | 45.2 |
| Vanilla MoE (k=2) | 2.3 | 42.1 | 45.8 |
| Vanilla MoE (k=6) | 2.5 | 42.8 | 46.9 |
| **TimeExpert (adaptive k)** | **2.5** | **42.8** | **46.9** |

- 移除token-adaptive routing后VHD HIT@1降4.3%，影响最大
- adaptive-k与k=6性能相当，但计算更高效（平均激活更少专家）
- 帧数从8增至128带来显著提升，验证VTG对时间分辨率的需求

## 亮点与洞察

1. **首次在VTG中发现并利用"隐式任务偏好"**：MoE专家天然趋向处理特定类型的token，显式强化这一趋势可大幅提升性能
2. **动态k比固定k更优雅**：不同token根据自身需求激活不同数量的专家，既高效又精确
3. **专家动态增删机制**为MoE研究提供了新视角：不是固定专家数量后训练，而是让专家数量在训练中自适应变化
4. 独立编码器处理时间/分数/文本彻底解耦了异质任务——直接用文本tokenizer处理时间token会导致指令跟随能力崩溃

## 局限性

1. 三阶段训练+5.1M数据的成本较高，且数据经过大量人工筛选和重标注
2. 基座模型ARIA的MoE架构限制了方法的通用性——是否适用于非MoE基座尚不清楚
3. 专家增删策略的阈值（ $\tau_{\min}$ 等）需要调优
4. 仅评估了分钟级视频，对小时级长视频（如电影）的适用性未验证

## 相关工作

- **VTG**: TimeChat, VTimeLLM, HawkEye, TRACE, VTG-LLM
- **Video-LLM**: ARIA, LLaVA-Video, Share-GPT4Video
- **MoE**: DeepSeekMoE, Switch Transformer, Llama-MoE

## 评分

- 创新性：⭐⭐⭐⭐⭐ — 任务感知动态门控+专家增删+任务依赖损失的完整MoE创新链
- 实用性：⭐⭐⭐⭐ — 统一处理三类VTG子任务，但训练成本较高
- 实验充分度：⭐⭐⭐⭐⭐ — 零样本+微调、四数据集、三类任务、详细消融
- 写作质量：⭐⭐⭐⭐ — 图表清晰，公式严谨

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](../../CVPR2025/video_understanding/seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)

</div>

<!-- RELATED:END -->
