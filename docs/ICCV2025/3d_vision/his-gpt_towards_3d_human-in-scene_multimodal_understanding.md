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
**关键词**: 人体场景理解, 3D场景问答, 多模态大语言模型, 人体动作, 人场交互

## 一句话总结

提出 HIS-GPT，首个面向3D人-场景联合理解的多模态大语言模型，通过辅助交互模块(AInt)和布局-轨迹位置编码(LTP)捕获人场交互线索，并构建首个系统性基准 HIS-Bench，在HIS-QA任务上大幅超越GPT-4o等基线。

## 研究背景与动机

**人体场景理解 (Human-In-Scene, HIS)** 要求智能体在3D场景中联合理解人体状态、行为及其与环境的交互关系，是具身智能的核心能力。当前存在三个关键瓶颈：

**3D场景LLM与人体LLM割裂**：3D场景LLM（如LL3DA、Chat-Scene）擅长场景描述和定位，但无法处理人体动作序列；3D人体LLM（如AvatarGPT）能理解姿态和动作，但忽略环境上下文。两者的分离使得无法回答"人在场景中做什么"这类需要联合理解的问题。

**缺乏HIS理解基准**：现有3D问答基准（如SQA3D）仅关注场景理解，动作基准（如MoVid-Bench）仅关注动作理解，没有一个基准同时整合场景和人体模态来评估人场交互理解能力。

**人场交互建模困难**：在3D空间中精确建模人体与场景物体之间的空间关系、接触状态和活动语义极具挑战，现有模型缺乏专门的交互建模机制。

**核心洞察**：要实现真正的人场理解，需要一个统一框架同时编码3D场景和人体运动，并通过专门的交互建模模块捕获两者之间的细粒度关联。HIS-GPT正是为此设计的首个基础模型。

## 方法详解

### 整体框架

HIS-GPT以3D场景点云 $\mathcal{S} \in \mathbb{R}^{P \times 6}$、人体运动序列 $\mathcal{M} = \{M_i\}_{i=1}^{T}$（SMPL姿态序列）和文本指令 $\mathcal{I}$ 为输入。框架包含四个核心组件：

- **场景编码器**：使用预训练3D编码器提取物体级场景嵌入 $\{s_i \in \mathbb{R}^d\}_{i=1}^{N}$
- **运动编码器**：使用Motion VQ-VAE将人体运动映射为运动嵌入 $\{m_t \in \mathbb{R}^d\}_{t=1}^{T}$
- **辅助交互模块 (AInt)**：通过辅助任务增强场景和运动嵌入中的交互线索
- **布局-轨迹位置编码 (LTP)**：编码空间布局和时间轨迹信息为位置嵌入

最终增强后的场景特征 $F^s$ 和运动特征 $F^m$ 被投影并与文本指令拼接，送入LLM自回归生成答案。

### 辅助交互模块 (AInt)

AInt引入三个辅助训练任务来增强人场交互建模：

**（1）活动分类**：通过场景上下文融合增强运动嵌入，对每个运动帧 $m_t$ 寻找空间上最近的 $k$ 个物体并融合其嵌入：

$$\tilde{m}_t = m_t + \text{Avg}(s_{t_1}, \ldots, s_{t_k})$$

然后通过MLP预测活动类别，使用交叉熵损失 $\mathcal{L}_{act}$ 监督。

**（2）空间关系检测**：定义8类人-物空间关系（如"面对"），通过线性投影预测物体 $s_i$ 与运动帧 $m_t$ 之间的空间关系，损失为：

$$\mathcal{L}_{spa} = \sum_{i,t} \text{CE}\left(p^s_{it}, \text{SM}(W^{spa}_s(s_i) \cdot W^{spa}_m(m_t))\right)$$

**（3）接触检测**：预测物体与人体特定身体部位是否接触，使用二值交叉熵损失：

$$\mathcal{L}_{cont} = \sum_{i,t} \text{BCE}\left(p^c_{it}, \sigma(W^{cont}_s(s_i) \cdot W^{cont}_m(m_t))\right)$$

### 布局-轨迹位置编码 (LTP)

传统MLLM中的位置编码仅建模token序列关系，忽略了3D时空结构。LTP通过空间傅里叶变换(SF)和时间傅里叶变换(TF)编码3D坐标和时间信息：

$$SF(\mu) = \text{sincos}(\phi_{SF} \cdot 2\pi\mu), \quad TF(t) = \text{sincos}(\phi_{TF} \cdot 2\pi t)$$

对运动帧生成位置编码 $e^m_t = SF(\mu_t) + TF(t)$；对场景物体生成 $e^s_i = SF(\mu_i) + \frac{1}{T}\sum_t TF(t)$（物体在整个运动序列中持续存在，故时间编码取平均）。最终：$f^s_i = s_i + e^s_i$，$f^m_t = m_t + e^m_t$。

### 损失函数

两阶段训练：
- **阶段1（模态对齐）**：$\mathcal{L} = \mathcal{L}_{llm} + \lambda_{act}\mathcal{L}_{act} + \lambda_{spa}\mathcal{L}_{spa} + \lambda_{cont}\mathcal{L}_{cont}$，其中 $\lambda_{act}=0.5, \lambda_{spa}=0.5, \lambda_{cont}=0.1$
- **阶段2（指令微调）**：仅使用 $\mathcal{L}_{llm}$，确保指令跟随质量

## 实验

### 主实验：HIS-Bench评测

| 方法 | Activity | Spatial | HoI | Analysis | Prediction | Dialogue | Planning | Avg. |
|------|----------|---------|-----|----------|------------|----------|----------|------|
| LL3DA | 6.5 | 9.1 | 8.7 | 11.9 | 5.3 | 4.7 | 0.4 | 6.7 |
| Chat-Scene | 9.2 | 5.2 | 3.0 | 24.3 | 14.7 | 3.7 | 6.7 | 8.2 |
| GPT-4o | 30.2 | 25.8 | 36.6 | 35.5 | 20.5 | 36.5 | 25.2 | 31.3 |
| LLaVA-Video | 13.8 | 11.3 | 24.9 | 17.7 | 13.3 | 20.8 | 14.1 | 16.3 |
| **HIS-GPT** | **44.6** | **42.1** | **55.5** | **41.0** | **50.3** | **53.2** | **53.9** | **48.7** |

**关键发现**：HIS-GPT平均分48.7，超越最强基线GPT-4o达17.4分。在需要精确空间理解的任务（如Human Position、Contact Part）和预测/规划任务上优势尤为明显。

### 消融实验

| 模型配置 | AInt(act/spa/cont) | PE | Activity | Spatial | HoI | Avg. |
|---------|-------------------|-----|----------|---------|-----|------|
| 基线 | — | sine | 41.8 | 34.7 | 45.8 | 43.0 |
| +AInt | ✓✓✓ | sine | 43.5 | 35.3 | 51.0 | 44.1 |
| +LTP | — | LTP | 43.5 | 38.8 | 50.3 | 46.0 |
| **+AInt+LTP** | **✓✓✓** | **LTP** | **44.6** | **42.1** | **55.5** | **48.7** |

**关键发现**：
- AInt模块贡献+1.1分，其中接触检测(cont)对HoI任务提升最大(+1.7)
- LTP模块贡献+3.0分，对空间关系理解提升显著(+4.1)
- 两模块联合使用获得+5.7分的互补增益，表明交互建模与时空编码的协同效应
- 训练策略消融证实两阶段训练的必要性，场景和运动描述数据在阶段1中贡献+2.9分

## 亮点与洞察

1. **任务与基准的开创性**：首次定义HIS-QA任务并构建系统性基准HIS-Bench（800题，3大能力、7核心任务、16子任务），填补了人场联合理解评估的空白
2. **交互建模的有效分解**：将复杂的人场交互分解为活动分类、空间关系和接触检测三个可学习的辅助任务，既提供了监督信号又增强了嵌入质量
3. **时空位置编码的新范式**：LTP模块通过傅里叶变换统一编码场景布局和运动轨迹，在两个独立编码的模态之间建立了全局时空对齐，值得推广到其他多模态3D任务
4. **Vision LLM的失败提供反面证据**：GPT-4o虽然具有强大的视觉理解能力，但在渲染视频上的HIS理解仍远不及直接处理3D输入的HIS-GPT，证明了3D原生表示的不可替代性

## 局限性

- 训练数据依赖PROX和GIMO两个相对小规模的HIS数据集，场景多样性有限
- 场景编码器和运动编码器冻结训练，可能限制了特征的适应性
- HIS-Bench的部分任务（焦点分析、情景分析、导航）需要人工标注，扩展成本较高
- 尚未探索HIS-GPT在动作生成、导航规划等下游具身任务中的实际应用

## 相关工作

- **3D场景语言理解**：LL3DA、Chat-Scene等3D场景LLM，SQA3D等基准
- **3D人体语言理解**：AvatarGPT、MotionGPT等人体动作LLM，Motion-X等数据集
- **情境场景理解**：SIG3D等利用文本或第一人称视角建立主体位置的方法，但缺乏全身姿态表示

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首次定义HIS-QA任务，提出首个HIS基础模型，AInt和LTP模块设计新颖
- **技术质量**: ⭐⭐⭐⭐ — 方法设计合理，消融实验充分，但训练数据规模有限
- **实验充分度**: ⭐⭐⭐⭐ — 基准全面，对比了多种基线，消融覆盖关键组件
- **表达清晰度**: ⭐⭐⭐⭐ — 任务定义清晰，图示直观，但附录信息量大增加阅读负担
- **综合评分**: 8.5/10
# HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding

**会议**: ICCV 2025  
**arXiv**: [2503.12955](https://arxiv.org/abs/2503.12955)  
**代码**: [ZJHTerry18/HumanInScene](https://github.com/ZJHTerry18/HumanInScene)  
**领域**: 3d_vision  
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
