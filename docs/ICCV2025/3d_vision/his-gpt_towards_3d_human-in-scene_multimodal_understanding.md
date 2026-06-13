---
title: >-
  [论文解读] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding
description: >-
  [ICCV 2025][3D视觉][人体-场景理解] 提出 HIS-QA 任务和 HIS-Bench 基准以及首个 3D 人体-场景联合理解基础模型 HIS-GPT，通过辅助交互模块（AInt）和布局-轨迹位置编码（LTP）捕获人与3D场景的精细交互，在16项子任务上大幅超越 GPT-4o 等基线。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "人体-场景理解"
  - "3D多模态"
  - "大语言模型"
  - "人体运动"
  - "问答基准"
---

# HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding

**会议**: ICCV 2025  
**arXiv**: [2503.12955](https://arxiv.org/abs/2503.12955)  
**代码**: [ZJHTerry18/HumanInScene](https://github.com/ZJHTerry18/HumanInScene)  
**领域**: 3D视觉  
**关键词**: 人体-场景理解, 3D多模态, 大语言模型, 人体运动, 问答基准

## 一句话总结

提出 HIS-QA 任务和 HIS-Bench 基准以及首个 3D 人体-场景联合理解基础模型 HIS-GPT，通过辅助交互模块（AInt）和布局-轨迹位置编码（LTP）捕获人与3D场景的精细交互，在16项子任务上大幅超越 GPT-4o 等基线。

## 研究背景与动机

**3D场景-语言理解和人体运动理解** 分别取得了显著进展，但二者的联合理解——即在3D场景中理解人的状态和行为——仍然是一个严重欠缺探索的领域。这一能力对具身智能至关重要：

1. **场景模型忽略人体**：现有 3D 场景 LLM（如 LL3DA、Chat-Scene）只能理解场景布局和物体，无法处理人体运动序列，无法识别"坐在椅子上"等人-物交互行为。

2. **人体模型忽略环境**：现有 3D 人体 LLM（如 MotionGPT、AvatarGPT）仅分析孤立的人体姿态/运动，无法感知"面朝电视"等需要场景上下文才能回答的状态。

3. **缺乏联合评估基准**：现有基准要么关注场景 QA（SQA3D），要么只关注运动描述（Motion-X），从未同时整合场景和人体模态进行开放式语言理解。

**核心洞察**：联合理解人与场景需要两个关键能力——(a) 捕获人体运动与周围物体间的交互线索（如接触、空间关系），(b) 将场景空间布局和人体运动轨迹进行时空对齐。HIS-GPT 通过 AInt 和 LTP 两个模块分别攻克这两个挑战。

## 方法详解

### 整体框架

HIS-GPT 接收三元输入：3D 场景点云 $\mathcal{S} \in \mathbb{R}^{P \times 6}$、人体运动序列 $\mathcal{M} = \{M_i\}_{i=1}^T$（SMPL 姿态序列）和文本指令 $\mathcal{I}$。场景和运动分别经独立编码器提取嵌入，然后通过 AInt 和 LTP 增强交互信息，最终投影至 LLM 进行自回归文本生成。

### 场景编码器

采用预训练的 3D 编码器（Uni3D）提取物体级特征。先用 3D 场景分割器（Mask3D）从点云中提取物体点云，再编码为 $\{s_i \in \mathbb{R}^d\}_{i=1}^N$，$N$ 为检测到的物体数。

### 运动编码器

采用运动 VQ-VAE（MotionGPT）将运动序列映射到离散码本上，得到运动嵌入 $\{m_t \in \mathbb{R}^d\}_{t=1}^T$。

### 辅助交互模块（AInt）

由于场景和运动嵌入独立生成，缺乏交互线索。AInt 通过三个辅助任务注入交互信息：

**（1）活动分类**：融合运动嵌入与空间最近 $k$ 个物体的特征后，预测人体活动类别。首先对第 $t$ 帧的运动嵌入执行场景上下文融合：

$$\tilde{m}_t = m_t + \text{Avg}(s_{t_1}, \ldots, s_{t_k})$$

然后通过 MLP + Softmax 分类，使用交叉熵损失 $\mathcal{L}_{act}$。

**（2）空间关系检测**：定义 8 种人-物空间关系类别（如"面朝"），预测第 $i$ 个物体与第 $t$ 帧运动之间的空间关系：

$$\mathcal{L}_{spa} = \sum_{i,t} \text{CE}(p_{it}^s, \text{SM}(W_s^{spa}(s_i) \cdot W_m^{spa}(m_t)))$$

**（3）接触检测**：预测物体是否与特定人体部位接触（二分类），使用 BCE 损失：

$$\mathcal{L}_{cont} = \sum_{i,t} \text{BCE}(p_{it}^c, \sigma(W_s^{cont}(s_i) \cdot W_m^{cont}(m_t)))$$

### 布局-轨迹位置编码（LTP）

传统位置编码仅建模 token 序列关系，忽略了人与场景之间的复杂时空关系。LTP 通过空间傅里叶变换（SF）和时间傅里叶变换（TF）编码3D坐标和时间信息：

$$SF(\mu) = \text{sincos}(\phi_{SF} \cdot 2\pi\mu), \quad TF(t) = \text{sincos}(\phi_{TF} \cdot 2\pi t)$$

- **运动位置编码**：$e_t^m = SF(\mu_t) + TF(t)$，基于第 $t$ 帧运动的3D位置和时间戳
- **场景位置编码**：$e_i^s = SF(\mu_i) + \frac{1}{T}\sum_t TF(t)$，物体在整个运动序列中持续存在，故对时间做平均

最终特征为 $f_i^s = s_i + e_i^s$, $f_t^m = m_t + e_t^m$。

### 训练策略与损失函数

两阶段训练：

- **Stage 1 — 模态对齐**：使用 HIS 描述、场景描述、运动描述进行对齐训练，总损失 $\mathcal{L} = \mathcal{L}_{llm} + \lambda_{act}\mathcal{L}_{act} + \lambda_{spa}\mathcal{L}_{spa} + \lambda_{cont}\mathcal{L}_{cont}$
- **Stage 2 — HIS 指令微调**：用 70 万指令微调样本，仅使用 $\mathcal{L}_{llm}$

训练数据共 6 万视觉描述 + 70 万指令微调样本，覆盖 750+ 场景。

## 实验

### 主实验：HIS-Bench 评估

| 方法 | Activity | Spatial | HoI | Analysis | Prediction | Dialogue | Planning | **Avg.** |
|------|----------|---------|-----|----------|------------|----------|----------|----------|
| LL3DA | 6.5 | 9.1 | 8.7 | 11.9 | 5.3 | 4.7 | 0.4 | 6.7 |
| Chat-Scene | - | - | - | - | - | - | - | 8.2 |
| GPT-4o | 30.2 | 25.8 | 36.6 | 35.5 | 20.5 | 36.5 | 35.0 | 31.3 |
| Qwen-VL-max | 28.7 | 17.6 | 37.1 | 13.4 | 14.5 | 33.0 | 21.5 | 23.5 |
| LLaVA-Video | 13.8 | 11.3 | 24.9 | 17.7 | 13.3 | 20.8 | 14.1 | 16.3 |
| **HIS-GPT** | **44.6** | **42.1** | **55.5** | **41.0** | **50.3** | **53.2** | **53.9** | **48.7** |

**关键发现**：
- HIS-GPT 平均分 48.7，超越最强基线 GPT-4o（31.3）达 17.4 分
- 在需要精细空间理解的任务（HP、CP）上优势尤为明显
- Vision LLM 因强指令跟随能力和视觉泛化性表现最好，但仍远不如 HIS-GPT
- 3D 场景 LLM 因训练语料无人体数据表现最差

### 消融实验

| 配置 | AInt(act/spa/cont) | PE | Act. | Spa. | HoI. | **Avg.** |
|------|-----------|-----|------|------|------|----------|
| Baseline | 无 | sine | 41.8 | 34.7 | 45.8 | 43.0 |
| +AInt | ✓✓✓ | sine | 43.5 | 35.3 | 51.0 | 44.1 |
| +LTP | 无 | LTP | 43.5 | 38.8 | 50.3 | 46.0 |
| **+AInt+LTP** | ✓✓✓ | LTP | **44.6** | **42.1** | **55.5** | **48.7** |

- **AInt 贡献**：平均分 +1.1；act/spa/cont 分别提升 Activity +1.3、Spatial +0.9、HoI +1.7
- **LTP 贡献**：平均分 +3.0
- **联合使用**：平均分 +5.7，说明两模块互补性强

**训练策略消融**：Stage 1 加入 scene+motion caption 数据后，平均分提升 2.9，验证模态对齐的重要性。

## 亮点与洞察

1. **开创性任务定义**：首次提出 HIS-QA 任务和 HIS-Bench 基准，填补了 3D 人体-场景联合理解的空白，包含 3 大能力、7 核心任务、16 子任务、800 道题
2. **精巧的交互建模**：AInt 模块通过三个辅助任务显式学习人-场景交互线索（活动、空间关系、接触），无需额外推理开销
3. **统一的时空对齐**：LTP 模块通过傅里叶位置编码将场景空间布局和运动轨迹对齐到统一坐标系，提升跨模态感知
4. **全面的标注流水线**：结合 3D 分割工具、视频描述模型和规则算法自动生成多面向标注，降低人工成本

## 局限性

1. 当前训练数据仅来自 PROX 和 GIMO 两个 HIS 数据集，场景和活动多样性有限
2. HIS-Bench 中部分任务（聚焦分析、情境分析、导航）仍需人工标注，扩展性受限
3. 场景编码器和运动编码器冻结训练，可能限制了交互特征的深度融合

## 相关工作

- **3D 场景 LLM**：LL3DA、Chat-Scene 等在场景 QA 和 Grounding 上表现优异，但无法处理人体模态
- **3D 人体 LLM**：MotionGPT、AvatarGPT 能理解和生成运动，但缺乏环境上下文
- **情境场景理解**：SQA3D 等假设通过文本或第一人称视角确定主体位置，缺乏全身姿态表示

## 评分

- **创新性**：⭐⭐⭐⭐⭐（新任务、新基准、新方法的完整开创性工作）
- **实用性**：⭐⭐⭐⭐（具身 AI、家庭机器人等应用价值大）
- **实验充分度**：⭐⭐⭐⭐（多基线对比 + 多维消融，但仅在自建基准上评估）
- **论文质量**：⭐⭐⭐⭐（结构清晰，方法描述详尽）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)

</div>

<!-- RELATED:END -->
