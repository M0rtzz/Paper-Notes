---
title: >-
  [论文解读] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System
description: >-
  [CVPR 2026][自动驾驶][end-to-end driving] 提出KnowVal端到端自驾系统，通过三大核心解决知识推理和价值对齐缺失：(1)Retrieval-guided Open-world Perception融合标准3D检测+VL-SAMv2长尾物体+VLM场景理解…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "end-to-end driving"
  - "knowledge graph"
  - "value model"
  - "world model"
  - "open-world perception"
  - "VLM"
  - "retrieval-augmented planning"
---

# KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System

**会议**: CVPR 2026  
**arXiv**: [2512.20299](https://arxiv.org/abs/2512.20299)  
**代码**: 待确认  
**领域**: 自动驾驶 / 知识增强规划  
**关键词**: end-to-end driving, knowledge graph, value model, world model, open-world perception, VLM, retrieval-augmented planning

## 一句话总结

提出KnowVal端到端自驾系统，通过三大核心解决知识推理和价值对齐缺失：(1)Retrieval-guided Open-world Perception融合标准3D检测+VL-SAMv2长尾物体+VLM场景理解；(2)Perception-guided Knowledge Retrieval从驾驶知识图谱（交通法/防御驾驶/道德规范）检索相关知识；(3)World Model预测未来状态+Value Model（human-preference训练）评估轨迹价值，实现可解释决策。nuScenes最低碰撞率，Bench2Drive/NVISIM SOTA。

## 研究背景与动机

端到端自动驾驶近年发展迅速，从感知到规划用单一模型完成，避免了模块间的误差累积。但现有端到端方法存在两个根本性缺陷：

**缺乏知识推理能力**：当前模型以数据驱动为主，学到的是统计模式而非驾驶知识。面对训练数据未覆盖的长尾场景（如施工区域特殊标志、不常见的交通手势、道德困境），模型无法像人类驾驶员那样调用交通法规、防御驾驶常识来做出合理决策

**缺乏价值对齐**：模型的优化目标（如模仿学习的轨迹L2距离）与人类对"好驾驶"的价值判断之间存在gap。人类认为好的驾驶不仅是到达目的地，还包括乘客舒适性、对弱势道路使用者的礼让、遵守社会规范等——这些无法通过简单的轨迹匹配来学习

**具体场景举例**：
- 一个施工区域放置了临时限速标志和锥桶→标准3D检测器不认识这些物体→需要开放世界感知+知识检索来理解"这是施工区，应减速"
- 一辆救护车在后方鸣笛→模型需要知道"应靠边让行"这一交通法规→需要知识图谱支持
- 前方行人犹豫是否过马路→需要"防御驾驶"知识指导减速观察→需要知识推理

现有方法如DriveVLM/LMDrive引入了VLM，但主要用于场景描述而非知识推理；UniAD等统一框架虽然端到端，但规划仍缺乏价值引导。

## 方法详解

### 整体架构

KnowVal包含三个协同工作的核心模块，形成感知→知识检索→规划的闭环：

```
Camera/LiDAR → Open-world Perception → Perception Verbalizer
                                              ↓
                     Knowledge Graph ← Knowledge Retrieval
                                              ↓
                   World Model → Value Model → Planning Decision
```

### 模块1：Retrieval-guided Open-world Perception

**目标**：构建超越封闭类别集的全方位感知能力。

三层感知体系：

**Layer 1 — Specialized Perception（标准3D检测）**：
- 使用成熟的3D目标检测器（如BEVFormer/StreamPETR）检测常见交通参与者（车辆、行人、骑行者等）
- 提供精确的3D bounding box、速度、朝向等结构化信息
- 这是传统端到端方法已有的能力，作为感知基座

**Layer 2 — Open-ended 3D Perception（长尾物体感知）**：
- 基于VL-SAMv2和OpenAD等开放词汇检测器
- 检测Specialized Perception无法覆盖的长尾物体：施工锥桶、临时标志、路面坑洞、遗落物体等
- VL-SAMv2利用视觉-语言对齐能力，无需为每个新类别标注训练数据
- 输出包含物体类别描述、2D/3D位置、置信度

**Layer 3 — Abstract Concept Understanding（抽象概念理解）**：
- 使用VLM（如GPT-4V/InternVL）对场景进行高层语义理解
- 提取无法用bounding box表示的抽象信息：道路状态（湿滑/积水）、天气条件、交通密度、整体场景氛围（紧张/平稳）
- 输出为结构化的场景属性描述

**三层互补逻辑**：Layer 1提供精确的结构化检测；Layer 2覆盖Layer 1的盲区（长尾物体）；Layer 3提供超越物体级别的场景语义。

### 模块2：Perception-guided Knowledge Retrieval

**目标**：根据感知结果从预构建的驾驶知识图谱中检索相关知识条目，为规划提供知识支撑。

**驾驶知识图谱构建**：

预构建三类知识库：
1. **交通法规知识**：结构化的交通法条目——限速规则、路权优先级、特殊区域规定等
2. **防御驾驶知识**：经验性安全驾驶规则——跟车距离、盲区风险、恶劣天气应对策略等
3. **道德规范知识**：驾驶伦理准则——弱势道路使用者保护、紧急让行规则、道德困境处理原则等

每条知识以(触发条件, 知识内容, 建议动作)三元组存储，并附有向量化索引。

**Perception Verbalizer（感知→文本转换器）**：

将三层感知模块的结构化输出转换为自然语言query：

$$q = \text{Verbalizer}(\text{Layer1\_output}, \text{Layer2\_output}, \text{Layer3\_output})$$

例如：检测到"前方10m有施工锥桶，道路变窄，左侧有临时标志" → 生成query "施工区域道路变窄处的驾驶规则和安全注意事项"

**知识检索过程**：

$$\mathcal{K}_{\text{relevant}} = \text{LLM-Retrieve}(q, \mathcal{G}_{\text{knowledge}})$$

使用LLM作为检索器，从知识图谱中检索最相关的$k$条知识条目。

**双向反馈机制（关键创新）**：检索模块不仅向规划端输出知识，还会**回传需要进一步感知的元素**给感知模块。例如：检索到"施工区域应注意临时信号灯"→通知感知模块额外关注信号灯检测。这形成了感知↔知识检索的闭环。

### 模块3：Planning with World Model + Value Model

**World Model — 未来状态预测**：

给定当前感知状态$s_t$和候选动作$a_t$，World Model预测未来$H$步的状态序列：

$$\hat{s}_{t+1:t+H} = f_{\text{world}}(s_t, a_t, \mathcal{K}_{\text{relevant}})$$

World Model将检索到的知识$\mathcal{K}_{\text{relevant}}$作为额外条件输入，使预测不仅基于物理动力学，还考虑知识约束（如"施工区限速30"会影响预测其他车辆的行为模式）。

**Value Model — 轨迹价值评估**：

$$V(\tau) = f_{\text{value}}(\hat{s}_{t+1:t+H}, \mathcal{K}_{\text{relevant}})$$

Value Model在**human-preference dataset**上训练，学习人类对驾驶轨迹的价值偏好：
- 训练数据：人类评估者对成对轨迹进行偏好排序（类似RLHF）
- 评估维度：安全性、舒适性、效率、合规性等
- 输出：每条候选轨迹的标量价值分数

**最终决策**：

$$a^* = \arg\max_{a \in \mathcal{A}} V(f_{\text{world}}(s_t, a, \mathcal{K}_{\text{relevant}}))$$

在候选动作空间中选择使Value Model评分最高的轨迹。决策过程可解释——可以追溯"为什么选择这条轨迹"（哪些知识被检索、Value Model在哪些维度给出高分）。

### 系统特性

- **兼容现有架构**：KnowVal的各模块可插入任意端到端自驾框架，不限定底层感知或规划架构
- **可解释性**：每个决策都可回溯知识来源和价值评估，便于调试和安全审计

## 实验关键数据

### 评估基准

- **nuScenes**：标准自驾感知+规划基准
- **Bench2Drive**：综合闭环驾驶评估基准
- **NVISIM**：NVIDIA仿真环境

### nuScenes 规划实验

| 方法 | 碰撞率↓ | L2 (1s) | L2 (3s) |
|------|---------|---------|---------|
| UniAD | — | — | — |
| VAD | — | — | — |
| KnowVal | **最低** | **SOTA** | **SOTA** |

KnowVal在nuScenes上实现**最低碰撞率**，显著优于现有端到端方法。碰撞率的大幅降低主要归功于知识检索提供的安全驾驶规则和Value Model的安全偏好。

### Bench2Drive 闭环评估

在Bench2Drive的多种驾驶场景（城市、高速、交叉口、恶劣天气）中均达到SOTA水平。尤其在需要复杂推理的场景（如无保护左转、施工区域通行）中，KnowVal的优势更为明显——这正是知识检索发挥作用的典型场景。

### NVISIM 仿真测试

在NVIDIA仿真环境中验证了KnowVal的实时性和鲁棒性，取得SOTA结果。

### 消融实验

| 配置 | 碰撞率 | 说明 |
|------|--------|------|
| 仅Specialized Perception | baseline | 标准端到端基线 |
| +Open-world Perception | 下降 | 长尾物体感知减少意外碰撞 |
| +Knowledge Retrieval | 显著下降 | 知识指导带来安全性提升 |
| +Value Model | **最低** | 价值对齐进一步优化轨迹选择 |

每个模块均带来独立正向贡献，Value Model的增益在安全性维度最为突出。

### 知识检索效果分析

在长尾场景中，Knowledge Retrieval的作用尤为显著——在包含非常规交通元素的场景中，碰撞率降低超过30%。这验证了"数据驱动+知识驱动"融合的必要性。

## 亮点与洞察

- **三层感知体系设计完善**：从精确的结构化检测→开放词汇长尾检测→抽象概念理解，覆盖了自动驾驶感知的全谱段需求。这是当前最完整的感知能力定义
- **感知↔知识检索的双向闭环**：不是单向的"感知→检索"，而是知识检索可以反向指导感知关注点。这种双向反馈机制模拟了人类驾驶中"先看到→联想知识→再仔细确认"的认知过程
- **Value Model引入human-preference**：类似LLM领域的RLHF思路迁移到自动驾驶，将"好驾驶"从轨迹L2距离泛化为多维度的人类偏好评估。这是自驾领域价值对齐的开创性尝试
- **知识图谱的三类知识划分清晰**：交通法规（硬约束）、防御驾驶（软约束）、道德规范（伦理约束）覆盖了从法律到伦理的完整驾驶知识体系
- **兼容性强**：不绑定特定的感知/规划架构，各模块可独立插拔，降低了集成门槛

## 局限与展望

1. **知识图谱的维护和更新**：交通法规随地区和时间变化，知识图谱需要持续更新。如何实现知识图谱的自动化更新和版本管理是实际部署的关键挑战
2. **VLM推理延迟**：Layer 3的VLM场景理解和知识检索中的LLM推理都有较高延迟。在要求实时响应（<100ms）的自动驾驶场景中，如何在保证推理质量的同时满足时延要求需要进一步优化
3. **Value Model的偏好泛化**：human-preference dataset的标注规模和多样性直接影响Value Model的泛化能力。不同文化背景下的驾驶偏好可能差异很大，单一数据集难以覆盖
4. **开放世界感知的误报问题**：VL-SAMv2等开放词汇检测器在开放场景中可能产生大量误报，如何有效过滤误报而不丢失真正的长尾物体是一个trade-off
5. **仿真到真实的迁移**：主要实验在nuScenes和仿真环境中完成，真实道路部署中的知识检索准确性和Value Model的鲁棒性需要进一步验证

## 相关工作与启发

- **UniAD (CVPR 2023)**：统一端到端自驾框架 → KnowVal在此基础上增加了知识推理和价值对齐维度
- **DriveVLM (ECCV 2024)**：引入VLM用于驾驶场景理解 → KnowVal进一步构建了完整的知识检索和利用管线
- **RAG (Retrieval-Augmented Generation)**：NLP领域的检索增强范式 → KnowVal将RAG思想巧妙迁移到自动驾驶的知识检索场景
- **RLHF (Reinforcement Learning from Human Feedback)**：LLM领域的价值对齐方法 → KnowVal的Value Model是RLHF在自驾规划中的自然对应
- **OpenAD / VL-SAMv2**：开放词汇3D检测前沿工作 → 作为KnowVal开放世界感知的基础组件
- **启发**：知识图谱+RAG+价值对齐的框架可推广到其他需要知识推理的具身智能任务（如机器人操作、无人机导航）。Value Model的人类偏好学习思路对整个自动驾驶规划领域具有启示意义

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.5 | 知识图谱+RAG+Value Model的组合在自驾领域是开创性的，系统设计完整而有远见 |
| 实用性 | 3.5 | 概念先进但VLM/LLM推理延迟和知识图谱维护增加了实际部署复杂度 |
| 实验充分度 | 4.0 | 三个基准SOTA，消融完整，缺少实时性能分析和真实道路测试 |
| 写作质量 | 4.0 | 系统架构描述清晰，模块间关系明确，动机论述有力，部分实验细节可更充分 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[ECCV 2024\] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](../../ECCV2024/autonomous_driving/visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[ICCV 2025\] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving](../../ICCV2025/autonomous_driving/adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)

</div>

<!-- RELATED:END -->
