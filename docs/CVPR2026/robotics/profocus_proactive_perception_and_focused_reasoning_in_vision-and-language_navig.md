---
title: >-
  [论文解读] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
description: >-
  [CVPR 2026][机器人][VLN] 提出 ProFocus，一个免训练的渐进式框架，通过主动感知（将全景图转为语义地图+LLM 生成针对性视觉查询）和聚焦推理（BD-MCTS 从大量历史路点中筛选 top-k 高价值候选），在 R2R 和 REVERIE 基准上达到零样本方法的 SOTA。
tags:
  - CVPR 2026
  - 机器人
  - VLN
  - proactive perception
  - MCTS
  - zero-shot navigation
  - LLM agent
---

# ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation

**会议**: CVPR 2026  
**arXiv**: [2603.05530](https://arxiv.org/abs/2603.05530)  
**代码**: 无  
**领域**: Robotics / Vision-Language Navigation  
**关键词**: VLN, proactive perception, MCTS, zero-shot navigation, LLM agent

## 一句话总结
提出 ProFocus，一个 training-free 框架，通过推理引导的主动感知（构建语义地图并迭代生成定向视觉查询）和分支多样化蒙特卡洛树搜索（BD-MCTS，筛选 top-k 高价值路点实现聚焦推理），在 R2R 和 REVERIE 上达到零样本 VLN 的 SOTA。

## 研究背景与动机
**领域现状**：Vision-and-Language Navigation（VLN）让智能体根据自然语言指令在环境中导航。基础模型的发展催生了两条路线：微调适应（如 NaviLLM，需大量数据但泛化差）和零样本推理（如 NavGPT、MapGPT，直接利用 LLM/VLM 的推理能力）。

**现有痛点**：(1) **被动视觉感知**——现有方法全量处理全景图或多视图输入，大量冗余视觉 token 导致注意力弥散，遮蔽了指令关键的细粒度线索（如物体颜色、纹理、空间关系）；(2) **无焦点推理**——所有历史路点被无差别对待，随着轨迹变长，模型难以从大量历史上下文中隔离关键线索，导致决策效率低下。

**核心矛盾**：VLM 的视觉 token 膨胀与精细属性识别需求的矛盾；LLM 的历史上下文无限增长与有效决策所需的聚焦推理之间的矛盾。

**本文目标**：让导航智能体主动获取任务相关视觉信息（而非被动接收全部输入），并在大量历史路点中聚焦于高价值候选进行推理。

**切入角度**：受人类认知启发（人类不会均匀回顾所有过往状态，而是修剪低价值分支、选择性重放任务相关轨迹），构建感知-推理闭环 + 价值驱动的路点优先级。

**核心 idea**：主动感知（"看什么"由推理决定）+ 聚焦推理（BD-MCTS 筛出 top-k 路点）= 高效零样本 VLN。

## 方法详解

### 整体框架
ProFocus 采用三个专业化智能体：编排智能体 $\mathcal{A}_{\text{orch}}^{\theta}$（LLM，空间推理和语义评估）、感知智能体 $\mathcal{A}_{\text{perc}}^{\phi}$（VLM，细粒度感知）、决策智能体 $\mathcal{A}_{\text{dec}}^{\psi}$（LLM，对 top-k 候选推理）。每个时间步执行两个核心机制：(1) 推理引导的主动感知——将全景观测转为结构化语义地图，迭代生成定向查询直到信息充分；(2) 基于 BD-MCTS 的聚焦推理——维护全局搜索树，筛选 top-k 高价值路点，决策智能体仅对这些候选做深度推理。

### 关键设计

1. **以自我为中心的语义地图 + 推理驱动感知闭环**:

    - 功能：将冗余全景观测压缩为结构化文本表示，并迭代获取指令相关的精细视觉属性
    - 核心思路：全景图分为 $K$ 个方向视图，VLM 检测所有物体的边界框和类别 $\{(\mathbf{b}_i, \text{obj}_i)\}_{i=1}^{N_t}$，结合单目深度估计 $d_i$ 和朝向角 $h_i = \pi \cdot (x_1 + x_2 - F) / F$ 构建语义地图 $\mathcal{C}_t = \{(h_i, \text{obj}_i(\mathbf{b}_i), d_i)\}$。编排智能体基于语义地图、轨迹历史和指令生成定向查询 $(q, R_{\text{focus}}^t) = \mathcal{A}_{\text{orch}}^{\theta}(\mathcal{C}_t, \tau_t, \mathcal{I}, \mathcal{H}_{\text{query}})$，感知智能体仅在焦点区域内做细粒度分析 $a_i^t = \mathcal{A}_{\text{perc}}^{\phi}(\mathcal{O}_t|_{R_{\text{focus}}^t}, q)$。循环直到编排智能体判断信息充分
    - 设计动机：被动感知处理全部视觉 token 造成注意力弥散，主动查询机制减少 token 数、增强属性识别、按需适应导航需求

2. **分支多样化蒙特卡洛树搜索（BD-MCTS）**:

    - 功能：从大量历史路点中筛选 top-k 高价值候选，使决策智能体聚焦推理
    - 核心思路：维护搜索树 $\mathcal{T} = \langle V_{\mathcal{T}}, E_{\mathcal{T}}, Q, N \rangle$。三阶段执行：**Phase I 扩展**——发现新路点并用语义值初始化 $Q(u) \leftarrow V_{\text{sem}}(u)$（替代传统 MCTS 的随机 rollout）；**Phase II 反向传播**——沿当前路径增量更新 $Q(v) \leftarrow Q(v) + (R_t - Q(v))/N(v)$，低奖励触发回溯、高奖励强化前进；**Phase III Top-k 选择**——计算路径聚合值并加距离惩罚 $\text{Score}(v) = V_{\text{path}}(v) - \lambda \cdot d_{\mathcal{G}}(v_t, v) / \max d$，约束每个父节点最多贡献 2 个孩子以保证分支多样性
    - 设计动机：标准 MCTS 选单一最优动作，VLN 需要考虑多条候选路线；均匀处理所有历史路点导致注意力弥散，top-k 筛选使 LLM 聚焦于最有价值的决策上下文

3. **语义值评估与信息融合**:

    - 功能：量化新发现路点与导航指令的语义相关度
    - 核心思路：感知闭环终止后，编排智能体综合积累的视觉信息评估新路点语义值 $V_{\text{sem}}(u) = \mathcal{A}_{\text{orch}}^{\theta}(\mathcal{I}, \tau_t, \{a_i^t\}_{i=1}^{n_t}) \in [0,1]$。所有信息整合为多模态上下文 $\mathcal{S}_t = \{\mathcal{I}, \tau_t, \mathcal{C}_t, \{a_i^t\}\}$ 存入记忆库 $\mathcal{M}$，决策智能体通过索引 top-k 路点的路径标识符从 $\mathcal{M}$ 检索相关历史上下文
    - 设计动机：为 BD-MCTS 提供基于多模态上下文的节点价值估计，替代传统 MCTS 的随机模拟

### 损失函数 / 训练策略
ProFocus 是 training-free 框架，无需训练或微调。使用现成的 LLM（Qwen3-Max / DeepSeek-V3）和 VLM（Qwen3-VL-Max / GLM-4.5V）。

## 实验关键数据

### 主实验
R2R validation unseen（零样本 VLN 方法对比）：

| 方法 | NE↓ | OSR↑ | SR↑ | SPL↑ |
|------|-----|------|-----|------|
| NavGPT (GPT-4) | 6.46 | 42.0 | 34.0 | 29.0 |
| MapGPT (GPT-4V) | 5.63 | 57.6 | 43.7 | 34.8 |
| DiscussNav (GPT-4) | 5.32 | 61.0 | 43.0 | 40.0 |
| NavGPT† (Q3) | 4.82 | 57.5 | 47.0 | 38.4 |
| MapGPT† (GLM) | 5.00 | 70.7 | 41.4 | 30.8 |
| **ProFocus (DS3+GLM)** | 5.21 | 63.0 | **50.0** | **41.2** |
| **ProFocus (Q3+Q3VL)** | **4.92** | **65.0** | 52.5 | 39.8 |

REVERIE validation unseen：

| 方法 | OSR↑ | SR↑ | SPL↑ |
|------|------|-----|------|
| MapGPT (GPT-4V) | 36.8 | 31.6 | 20.3 |
| MapGPT† (GLM) | 50.5 | 37.1 | 24.7 |
| **ProFocus (DS3+GLM)** | **57.1** | 36.9 | **25.9** |
| **ProFocus (Q3+Q3VL)** | 51.7 | **40.0** | 24.8 |

### 消融实验
R2R validation unseen（Q3+Q3VL 配置）：

| 配置 | NE↓ | OSR↑ | SR↑ | SPL↑ |
|------|-----|------|-----|------|
| ProFocus (完整) | 4.92 | 65.0 | 52.0 | 39.8 |
| w/o BD-MCTS | 6.19 | 53.7 | 50.0 | 38.4 |
| w/o Proactive Perception | 5.97 | 54.0 | 48.0 | 34.4 |

REVERIE validation unseen（Q3+Q3VL 配置）：

| 配置 | OSR↑ | SR↑ | SPL↑ |
|------|------|-----|------|
| ProFocus (完整) | 51.7 | 40.0 | 24.8 |
| w/o BD-MCTS | 48.9 | 37.8 | 27.4 |
| w/o Proactive Perception | 34.0 | 30.0 | 18.5 |

### 关键发现
- 主动感知对 SPL 影响最大（R2R 上 -5.4%），说明定向查询显著提升路径效率
- 去除 BD-MCTS 导致 OSR 大幅下降（REVERIE 上 -17.7%），说明 top-k 筛选对探索能力至关重要
- 在 R2R 最长 30 条轨迹上，ProFocus 达 50% SR，验证了长轨迹鲁棒性
- 两种基础模型配置（DS3+GLM 和 Q3+Q3VL）均持续优于对应基线，说明改进来自框架而非特定模型

## 亮点与洞察
- "看什么由推理决定"的主动感知闭环设计精巧——将全景图的 token 爆炸问题转化为按需查询
- BD-MCTS 巧妙地解决了 VLN 中"所有历史都重要但不能全看"的难题，top-k + 分支多样性约束兼顾深度和广度
- 语义值替代随机 rollout 的设计避免了 NeRF/VLN 环境中模拟的高成本
- training-free + 即插即用不同 LLM/VLM 组合，实际部署友好

## 局限与展望
- 依赖强大的商用 API（Qwen3-Max、DeepSeek-V3），推理成本较高
- 每步多次 VLM 调用（构建语义地图 + 迭代查询）可能导致延迟问题
- 仅在 R2R 和 REVERIE 上测试，未验证连续动作空间或真实机器人部署
- BD-MCTS 的超参数（top-k 值、$\lambda$、分支约束）的敏感性分析不够充分

## 相关工作与启发
- NavGPT 将全景转为文本描述再用 GPT-4 决策，但被动感知导致信息丢失
- MapGPT 引入 SAM 做视觉负担分离，但仍然一次性处理所有视觉输入
- MCTS 在 AlphaGo 中大获成功，BD-MCTS 的贡献在于适配图结构导航（处理环路）和用语义值替代 rollout
- 主动感知的思路可推广到其他 VLM 任务（如 embodied QA、对话导航）

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动感知闭环 + BD-MCTS 的组合在 VLN 中首次出现
- 实验充分度: ⭐⭐⭐⭐ 两个 benchmark、多模型配置、完整消融和定性分析
- 写作质量: ⭐⭐⭐⭐ 公式化严谨，框架图清晰
- 价值: ⭐⭐⭐⭐ 为 training-free VLN 提供了有效的感知-推理范式
# ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation

**会议**: CVPR 2026  
**arXiv**: [2603.05530](https://arxiv.org/abs/2603.05530)  
**代码**: 无  
**领域**: 机器人 / 视觉语言导航  
**关键词**: VLN, proactive perception, MCTS, zero-shot navigation, LLM agent

## 一句话总结
提出 ProFocus，一个免训练的渐进式框架，通过主动感知（将全景图转为语义地图+LLM 生成针对性视觉查询）和聚焦推理（BD-MCTS 从大量历史路点中筛选 top-k 高价值候选），在 R2R 和 REVERIE 基准上达到零样本方法的 SOTA。

## 研究背景与动机
**领域现状**：视觉语言导航 (VLN) 要求智能体根据自然语言指令在物理环境中导航。基于基础模型的 VLN 方法通过后训练适应或零样本提示展现了良好前景，但普遍存在两个关键缺陷。

**现有痛点**：(1) 被动视觉感知——VLM 驱动的方法统一处理全景或多视角视觉输入，冗余信息膨胀视觉 token 数量，导致注意力在无关特征上扩散，遮蔽指令相关的细粒度线索；(2) 无聚焦推理——两种范式都接收大量未优先级排序的历史上下文，包含过去的观测和路点，长轨迹历史稀释注意力，阻碍精确推理。

**核心矛盾**：导航需要选择性感知（只获取任务相关信息）和聚焦推理（只关注高价值历史路点），但现有方法在两方面都是"全量处理"。

**本文目标**：如何主动获取任务相关的视觉观测以减少感知冗余，如何在大量历史上下文中聚焦推理高价值路点。

**切入角度**：通过 LLM-VLM 协作建立闭环感知-推理循环，用 MCTS 变体从全局历史中筛选关键路点。

**核心 idea**：LLM 根据语义地图判断"需要知道什么"并生成视觉查询，VLM 在指定区域执行细粒度感知，再用 BD-MCTS 从海量历史中聚焦 top-k 高价值路点进行决策。

## 方法详解

### 整体框架
ProFocus 包含三个专用智能体：编排智能体 $\mathcal{A}_{\mathrm{orch}}^{\boldsymbol{\theta}}$（LLM，空间推理和语义评估）、感知智能体 $\mathcal{A}_{\mathrm{perc}}^{\boldsymbol{\phi}}$（VLM，细粒度感知）、决策智能体 $\mathcal{A}_{\mathrm{dec}}^{\boldsymbol{\psi}}$（LLM，基于 top-k 候选推理）。输入全景图和指令，输出导航动作。

### 关键设计

1. **以自我为中心的语义地图 (Ego-centric Semantic Map)**:

    - 功能：将全景观测转化为结构化的文本表示，编码物体位置、深度和方向关系
    - 核心思路：全景图分为 $K$ 个方向视图，VLM 并行检测所有物体 $\{(\boldsymbol{b}_i, \textit{obj}_i)\}_{i=1}^{N_t}$，用单目深度估计获取深度 $d_i$，计算朝向角 $h_i = \pi \cdot (\frac{x_1 + x_2 - F}{F})$，构建语义地图 $\mathcal{C}_t = \{(h_i, \textit{obj}_i(\boldsymbol{b}_i), d_i)\}_{i=1}^{N_t}$，格式化为自然语言文本
    - 设计动机：将视觉信息压缩为结构化文本，使 LLM 能进行空间推理（如"左边的物体"），避免处理大量原始像素

2. **推理驱动的主动感知循环 (Reasoning-Driven Perception Loop)**:

    - 功能：根据任务需求主动获取指令相关的视觉信息，而非被动处理全部输入
    - 核心思路：编排智能体根据语义地图、轨迹历史和指令生成视觉查询 $\boldsymbol{q}$ 和聚焦区域 $\boldsymbol{R}_{\text{focus}}^t$：$(\boldsymbol{q}, \boldsymbol{R}_{\text{focus}}^t) = \mathcal{A}_{\mathrm{orch}}^{\boldsymbol{\theta}}(\mathcal{C}_t, \boldsymbol{\tau}_t, \mathcal{I}, \mathcal{H}_{\text{query}})$。感知智能体在聚焦区域执行细粒度分析 $\boldsymbol{a}_i^t = \mathcal{A}_{\mathrm{perc}}^{\boldsymbol{\phi}}(\boldsymbol{\mathcal{O}}_t|_{\boldsymbol{R}_{\text{focus}}^t}, \boldsymbol{q})$。循环迭代直到信息充足 $s_t = \text{sufficient}$
    - 设计动机：减少视觉 token（仅感知指令相关区域），增强属性识别（通过针对性查询获取细粒度细节），自适应感知

3. **分支多样 MCTS (Branch-Diverse MCTS, BD-MCTS)**:

    - 功能：从大量历史路点中筛选 top-k 高价值候选，引导决策智能体聚焦推理
    - 核心思路：维护搜索树 $\mathcal{T} = \langle \boldsymbol{V}_{\mathcal{T}}, \boldsymbol{E}_{\mathcal{T}}, Q, N \rangle$，分三阶段——(I) 用语义值 $V_{\text{sem}}(u)$ 替代随机 rollout 初始化新节点；(II) 沿路径反向传播动态精修 $Q(v) \leftarrow Q(v) + \frac{R_t - Q(v)}{N(v)}$；(III) 路径聚合打分 $\text{Score}(v) = V_{\text{path}}(v) - \lambda \cdot \frac{d_{\mathcal{G}}(v_t, v)}{\max_{u} d_{\mathcal{G}}(v_t, u)}$ 结合距离惩罚，每个父节点贡献最多 2 个子节点保持分支多样性
    - 设计动机：标准 MCTS 选择单一最优动作，BD-MCTS 选择多样的 top-k 候选；距离惩罚确保物理可达性；分支多样性确保覆盖不同探索方向

### 损失函数 / 训练策略
ProFocus 是**完全免训练 (training-free)** 的框架，无须微调或训练任何模型。使用现成的 LLM（Qwen3-Max / DeepSeek-V3）和 VLM（Qwen3-VL-Max / GLM-4.5V）。

## 实验关键数据

### 主实验
R2R validation unseen set 结果：

| 方法 | NE↓ | OSR↑ | SR↑ | SPL↑ |
|------|-----|------|-----|------|
| NavGPT (GPT-4) | 6.46 | 42.0 | 34.0 | 29.0 |
| MapGPT (GPT-4V) | 5.63 | 57.6 | 43.7 | 34.8 |
| MSNav (GPT-4o) | 5.24 | 65.0 | 46.0 | 40.0 |
| **ProFocus (Q3+Q3VL)** | **4.92** | **65.0** | **52.5** | **39.8** |
| **ProFocus (DS3+GLM)** | 5.21 | 63.0 | 50.0 | 41.2 |

### 消融实验

| 配置 | SR↑ | SPL↑ | 说明 |
|------|-----|------|------|
| NavGPT† (DS3) | 36.0 | 28.1 | 无主动感知，无聚焦推理 |
| MapGPT† (GLM) | 41.4 | 30.8 | 基于VLM的被动感知 |
| ProFocus (DS3+GLM) | 50.0 | 41.2 | +主动感知+BD-MCTS |
| ProFocus (Q3+Q3VL) | 52.5 | 39.8 | 更强基础模型进一步提升 |

### 关键发现
- SR 从 NavGPT 的 36.0% 提升至 52.5%，绝对提升 16.5 个百分点
- 主动感知通过减少视觉 token 和增强细粒度属性识别同时提高效率和准确率
- BD-MCTS 的分支多样性约束对避免局部最优至关重要
- 免训练框架在零样本设置下已超过部分训练方法

## 亮点与洞察
- 三智能体协作分工明确——编排（规划）、感知（执行）、决策（推理），符合人类导航认知过程
- "主动感知"对比"被动感知"的提升说明：少而精的视觉信息优于多而杂
- BD-MCTS 将人类的优先记忆访问机制引入 VLN——人类也不会均匀回忆所有历史状态
- 免训练框架意味着可即插即用更强的 LLM/VLM

## 局限与展望
- API 调用开销——每步导航需多次 LLM/VLM 调用，延迟较高
- 语义地图依赖物体检测和深度估计的质量
- 评估依赖特定 LLM/VLM，更换模型可能需要重新调整 prompt
- 仅评估 R2R 和 REVERIE，未涉及更挑战性的连续环境导航

## 相关工作与启发
- **NavGPT**：将全景场景转为文本描述，被动感知
- **MapGPT**：利用 VLM 地图表示，但仍被动处理所有视图
- **AO-Planner**：使用 SAM 的视觉可达性提示，但缺乏主动查询机制
- 启发：主动感知+聚焦推理的范式可推广到其他需要长程历史推理的任务，如对话系统、长文档问答

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动感知循环和 BD-MCTS 的结合新颖，闭环感知-推理设计巧妙
- 实验充分度: ⭐⭐⭐⭐ R2R 和 REVERIE 两个标准基准，多种模型配置对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式化严谨
- 价值: ⭐⭐⭐⭐ 免训练框架实用性强，可作为 LLM/VLM 在导航任务中的通用增强模块

<!-- RELATED:START -->

## 相关论文

- [Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)
- [Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)

<!-- RELATED:END -->
