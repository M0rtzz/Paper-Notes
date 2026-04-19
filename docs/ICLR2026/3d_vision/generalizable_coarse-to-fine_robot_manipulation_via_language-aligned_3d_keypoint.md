---
title: >-
  [论文解读] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints
description: >-
  [ICLR 2026][3D视觉][Robot Manipulation] CLAP（Coarse-to-fine Language-Aligned manipulation Policy）通过任务分解、VLM微调的3D关键点预测和3D感知表征三个核心组件，实现了对新指令和新环境的强泛化能力，在 GemBench 上以 1/5 的训练数据比 SOTA 高出 12%。
tags:
  - ICLR 2026
  - 3D视觉
  - Robot Manipulation
  - Coarse-to-Fine Policy
  - 3D Keypoints
  - VLM Fine-tuning
  - Language Grounding
---

# Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints

**会议**: ICLR 2026  
**arXiv**: [2509.23575](https://arxiv.org/abs/2509.23575)  
**代码**: 无  
**领域**: 3D视觉 / 机器人操作  
**关键词**: Robot Manipulation, Coarse-to-Fine Policy, 3D Keypoints, VLM Fine-tuning, Language Grounding

## 一句话总结

CLAP（Coarse-to-fine Language-Aligned manipulation Policy）通过任务分解、VLM微调的3D关键点预测和3D感知表征三个核心组件，实现了对新指令和新环境的强泛化能力，在 GemBench 上以 1/5 的训练数据比 SOTA 高出 12%。

## 研究背景与动机

分层的粗到细（Coarse-to-Fine）策略在机器人3D操作任务中展现了巨大潜力。其基本思路是：粗分支（coarse branch）预测一个感兴趣区域（Region of Interest），然后细分支（fine branch）在该区域内执行精确的动作预测。这种层次化设计显著提升了样本效率和操作精度。

然而即使引入了预训练模型增强，现有的分层策略仍然面临**泛化性不足**的核心问题：

**对新指令的泛化**: 当给出训练时未见过的自然语言指令时（如"拿起红色的杯子"→"把蓝色的碗放到架子上"），策略往往失败

**对环境变化的泛化**: 物体位置、外观、背景等变化都可能导致策略崩溃

**样本效率**: 现有方法通常需要大量演示轨迹来学习每个任务

这些问题的根源在于：粗分支缺乏对语言语义的深度理解，且表征缺乏3D空间的结构化信息。

## 方法详解

### 整体框架

CLAP 是一个分层的操作策略框架，包含三个相互配合的核心组件：

1. 首先通过任务分解将复杂指令拆解为子任务序列
2. 然后使用微调的 VLM 从当前观察中预测与子任务对齐的 3D 关键点
3. 最后基于 3D 感知表征在关键点附近执行精确操作

### 关键设计

1. **任务分解（Task Decomposition）**:

    - **功能**: 将自然语言指令分解为有序的子任务步骤
    - **核心思路**: 利用大型语言模型（LLM）或规则化方法将复杂的操作指令分解为原子化的步骤。例如，"把杯子放到盘子旁边"可以分解为"1.接近杯子 → 2.抓取杯子 → 3.移动到盘子旁 → 4.放下杯子"。
    - **设计动机**: 复杂指令的直接端到端映射需要大量数据。分解后，每个子任务更简单、更通用——"抓取"这个原子动作可以在多种场景中复用。这种组合性是泛化的关键来源。

2. **VLM 微调用于 3D 关键点预测（VLM Fine-tuning for 3D Keypoint Prediction）**:

    - **功能**: 微调视觉-语言模型（VLM），使其根据当前视觉观察和子任务描述预测目标物体的 3D 关键点位置
    - **核心思路**: 取预训练 VLM（如基于 CLIP 的模型），在机器人操作数据上微调。输入为 RGB 图像和子任务文本描述，输出为 3D 空间中的关键点坐标。关键点是语言对齐的——对"抓取红色杯子"和"推动红色杯子"，预测的关键点位置不同（分别对应杯柄和杯侧面）。
    - **设计动机**: VLM 拥有丰富的视觉-语言先验知识，知道"杯子"长什么样、"抓取"应该作用在哪个部位。微调使这些先验适应机器人操作场景，同时保留对新概念的泛化能力。3D 关键点而非 2D 关键点的预测确保了对深度和空间关系的感知。

3. **3D 感知表征（3D-Aware Representation）**:

    - **功能**: 构建面向操作的 3D 空间表征
    - **核心思路**: 结合多视角 RGB 图像和深度信息，构建以预测关键点为中心的 3D 局部特征。细分支的动作预测基于这个 3D 表征进行，而非原始图像。
    - **设计动机**: 机器人操作本质上是 3D 任务——物体的抓取姿态、放置位置都定义在 3D 空间中。纯 2D 表征缺乏深度信息，在视角变化或物体遮挡时容易失败。3D 表征提供了更鲁棒的空间推理基础。

### 损失函数 / 训练策略

- **VLM 微调**: 关键点预测通常使用回归损失（L1或L2距离），将预测的3D坐标与标注的关键点坐标对齐
- **策略学习**: 细分支使用行为克隆（Behavior Cloning），在关键点附近学习从3D表征到末端执行器动作的映射
- **数据高效**: 仅需 10 条真实演示即可训练出有效策略（real-world 实验），远少于常规方法所需的数百条演示

## 实验关键数据

### 实验设置
- 仿真基准：GemBench（专为泛化评估设计的操作基准）
- 真实实验：真实机器人平台，10条演示
- 评价指标：操作成功率
- 泛化维度：新指令、新物体外观、新环境布局

### 主实验
| 方法 | GemBench 平均成功率 | 训练轨迹数 | 说明 |
|------|-------------------|-----------|------|
| SOTA（基线最优） | ~X% | ~5N | 需要大量演示 |
| **CLAP** | **X + 12%** | **N（1/5）** | 显著更高成功率 + 更少数据 |

CLAP 在 GemBench 上比 SOTA 方法平均成功率高出 12 个百分点，同时仅使用 1/5 的训练轨迹。

### 真实机器人实验
| 设置 | 成功率 | 说明 |
|------|--------|------|
| 训练场景 | 高成功率 | 10条演示即可学会 |
| 新指令 | 成功泛化 | 语言对齐的关键点正确识别新目标 |
| 新环境 | 成功泛化 | 3D表征对布局变化鲁棒 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无任务分解 | 成功率下降 | 复杂指令直接处理效果差 |
| 无VLM微调（直接用预训练VLM） | 成功率下降 | 预训练VLM对操作场景不够适配 |
| 2D表征替代3D表征 | 成功率下降 | 缺乏深度信息影响精确操作 |

### 关键发现

1. **三个组件缺一不可**: 任务分解、VLM 微调、3D 表征各自贡献了不同维度的泛化能力
2. **极低数据需求**: 10条演示在真实场景即可工作，这对实际部署非常有价值
3. **语言对齐是关键**: 关键点不仅是空间位置，还携带语义信息——同一物体对不同指令产生不同关键点

## 亮点与洞察

- **"少量数据+强泛化"的理想组合**: 通过充分利用预训练 VLM 的先验知识，将样本需求压到极低水平同时保持强泛化
- **层次化设计清晰**: 粗分支（VLM 关键点预测）和细分支（3D 局部动作预测）分工明确，各司其职
- **语言与3D空间的桥接**: 通过 VLM 微调将语言语义映射到 3D 关键点，是连接 NLP 和机器人操作的有效桥梁
- **实用导向**: 10条演示即可部署的特性使得该方法具有很高的实际应用价值

## 局限与展望

1. **任务分解的鲁棒性**: 如果 LLM 的分解不准确（如遗漏关键步骤或顺序错误），整个流水线会失败
2. **关键点的表达能力**: 单个 3D 关键点可能不足以描述复杂操作（如需要双手协调、多点接触的任务）
3. **VLM 微调数据**: 虽然策略学习数据需求低，但 VLM 微调可能仍需可观的标注数据
4. **动态环境**: 当前方法似乎面向静态或缓变环境，对快速动态场景（如移动物体）的适应性未知
5. **长horizon任务**: 任务分解产生的子任务序列如果很长，误差累积可能成为问题
6. **开放词汇的极限**: 虽然对新指令有泛化，但对全新概念类别（训练时完全未见的物体类型）的泛化边界未被充分探索

## 相关工作与启发

- **与 PerAct/RVT 的关系**: PerAct 和 RVT 使用体素化3D表征进行操作，但缺乏语言引导的关键点机制。CLAP 的粗到细设计是对这类方法的有效补充。
- **与 SayCan/Code-as-Policies 的关系**: 这些方法用 LLM 做任务规划但不涉及底层操作策略的泛化。CLAP 的任务分解类似但更关注执行层面。
- **VLM 在机器人中的应用趋势**: RT-2、Octo 等工作也在将 VLM 用于机器人，但多为端到端方式。CLAP 的层次化方式（VLM → 关键点 → 局部策略）提供了一种更可控且数据高效的替代方案。
- **3D 关键点的普适性**: 关键点作为操作的中间表征具有很好的通用性，未来可以探索更丰富的关键点表示（如带方向的关键点、关键点图等）。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)
- [3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image](../../CVPR2026/3d_vision/3d-fixer_coarse-to-fine_in-place_completion_for_3d_scenes_from_a_single_image.md)
- [RoboPearls: Editable Video Simulation for Robot Manipulation](../../ICCV2025/3d_vision/robopearls_editable_video_simulation_for_robot_manipulation.md)
- [VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](../../NeurIPS2025/3d_vision/shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)

<!-- RELATED:END -->
