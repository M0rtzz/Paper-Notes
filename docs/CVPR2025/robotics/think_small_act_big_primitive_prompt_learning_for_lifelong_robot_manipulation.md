---
title: >-
  [论文解读] Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation
description: >-
  [CVPR 2025][机器人][原始提示学习] 提出 Primitive Prompt Learning (PPL)，通过将运动原语编码为可复用的提示向量，结合光流感知的 Motion-Aware Prompting（MAP）实现跨技能运动原语共享，用冻结-扩展机制支持终身机器人操作学习，在 LIBERO 和真实世界中均优于 LoRA、经验回放等基线。
tags:
  - CVPR 2025
  - 机器人
  - 原始提示学习
  - 终身学习
  - 光流
  - 扩散策略
  - 灾难性遗忘
---

# Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation

**会议**: CVPR 2025  
**arXiv**: [2504.00420](https://arxiv.org/abs/2504.00420)  
**代码**: 无  
**领域**: 机器人操作 / 持续学习  
**关键词**: 原始提示学习, 终身学习, 光流, 扩散策略, 灾难性遗忘

## 一句话总结

提出 Primitive Prompt Learning (PPL)，通过将运动原语编码为可复用的提示向量，结合光流感知的 Motion-Aware Prompting（MAP）实现跨技能运动原语共享，用冻结-扩展机制支持终身机器人操作学习，在 LIBERO 和真实世界中均优于 LoRA、经验回放等基线。

## 研究背景与动机

**领域现状**：机器人操作策略通常在固定任务集合上训练，面对新任务时需要重新训练。实际应用中机器人需要持续学习新技能而不遗忘旧技能（终身学习）。现有终身学习方法（经验回放、LoRA 等）要么需要存储旧数据，要么无法有效迁移知识。

**现有痛点**：不同操作技能之间存在共享的运动原语（如"抓取"动作在拿杯子和拿香蕉中相似），但现有方法只通过语义相似性（文本嵌入）发现任务间关联，忽略了运动层面的共享结构。语义不同但运动相似的任务（"grasp mug" vs "place banana"）之间的知识迁移被遗漏。

**核心矛盾**：终身学习需要平衡旧知识保留和新知识获取。参数共享多→迁移好但遗忘严重；参数隔离多→遗忘少但迁移弱。

**本文目标** 找到共享的运动原语，让它们在多任务预训练中学到并在终身学习中冻结复用。

**切入角度**：用光流提取运动信息，与 CLIP 文本嵌入结合形成查询向量。光流捕捉的运动模式（"向下抓取"、"向前推动"）跨越语义边界，是发现运动原语共享性的关键。

**核心 idea**：光流+语义联合查询发现跨任务运动原语 → 编码为提示向量 → 预训练冻结后终身复用。

## 方法详解

### 整体框架

两阶段框架：Stage 1 在多技能数据上预训练扩散 Transformer 策略+原始提示向量。提示通过前缀方式注入 MSA 层的 Key/Value。Stage 2 学习新技能时冻结预训练提示，新增终身学习提示并用注意力加权机制混合两种提示。MAP 模块将光流和文本嵌入结合为查询向量，通过余弦相似度选择相关的提示组件。

### 关键设计

1. **Motion-Aware Prompting (MAP)**:

    - 功能：同时利用运动和语义信息发现跨任务的共享原语
    - 核心思路：用 RAFT 算法提取视频中的光流 $F$，特征化后得到 $\Phi(F)$；用 CLIP 编码任务描述 $E_{\text{CLIP}}(T)$。两者融合为 MAP 查询 $\text{MAP}(T,F) = f_{\text{prompt}}(E_{\text{CLIP}}(T), \Phi(F))$。光流捕捉低层运动模式（方向、速度、轨迹），CLIP 捕捉高层语义
    - 设计动机：纯文本查询只能发现语义相似任务间的关联（如"拿杯子"和"拿瓶子"），无法发现语义不同但运动相似的任务（如"抓杯子"和"放香蕉"都涉及手臂下移+夹爪闭合）。消融显示 MAP 使提示权重分布反映了运动共享模式

2. **前缀提示学习（Prefix Prompt）**:

    - 功能：以最小参数量为策略网络注入技能知识
    - 核心思路：提示 $p \in \mathbb{R}^{L_p \times D}$ 分为 $\{p^K, p^V\}$ 对，前缀拼接到 MSA 的 Key 和 Value 序列中：$f_{P-T}(\mathbf{p}, \mathbf{h}) = \text{MSA}(h_Q, [\mathbf{p}_K; h_K], [\mathbf{p}_V; h_V])$，只更新提示参数而冻结主干网络
    - 设计动机：提示学习的参数量远小于全量微调或 LoRA，且提示可以独立冻结/扩展，天然适合终身学习的知识管理

3. **冻结-扩展的终身学习机制**:

    - 功能：学新技能时不遗忘旧技能
    - 核心思路：新任务到来时冻结所有预训练提示，新增一组终身提示。MAP 查询用注意力加权机制从冻结提示和新提示中选择相关组件：$\alpha_m = \cos\_\text{sim}(\text{MAP}(T,F) \odot A, K_m)$，$p = \sum_m \alpha_m P_m$。只更新新提示的参数
    - 设计动机：冻结提示保留旧知识不受干扰，新提示捕捉新任务特有的运动模式。注意力机制让选择过程可微分且自适应

### 损失函数 / 训练策略

行为克隆损失：$\hat\theta = \min_\theta \sum_k \mathbb{E}_{s_t, a_t \sim \mathcal{D}_k}[\sum_t \mathcal{L}(\pi(a|s_t, T_k; \theta), a_k^t)]$，通过扩散策略参数化。预训练使用 MimicGen + LIBERO 数据，每技能 200 条人类演示。终身学习阶段只更新新提示参数。

## 实验关键数据

### 主实验

LIBERO 终身学习（7 个顺序任务）的前向/后向迁移：

| 方法 | 平均 FWT | 平均 BWT |
|------|---------|---------|
| Sequential | 低 | 严重遗忘 |
| Experience Replay | 中 | 负 BWT |
| LoRA | 中 | 中 |
| **PPL (Ours)** | **0.83±0.03** | **0.78±0.09** |

真实世界实验（Franka Panda，9 个技能）：

| 方法 | 预训练均值 | 终身学习均值 |
|------|-----------|------------|
| Diffusion-Transformer | 0.42±0.09 | - |
| MoE | 0.73±0.08 | - |
| **PPL** | **0.84±0.05** | **0.68±0.05** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 文本-only 查询 | 只发现语义相似任务间的关联 |
| 文本+光流查询 (MAP) | 额外发现运动相似任务间的关联 |
| 无预训练提示 | 终身学习性能显著下降 |
| 极端光照变化 | 光流质量下降导致性能退化 (0.83→0.61) |

### 关键发现
- **光流发现运动原语共享**：MAP 查询使语义不同但运动相似的任务（如"抓杯子"和"放香蕉"）共享更高的提示权重，这是纯文本查询无法实现的
- **提示数量不是越多越好**：过多提示引入噪声，最优数量需要平衡覆盖度和精度
- **训练效率接近 LoRA**：PPL 速度与 LoRA 相当，但性能接近 MoE（同时获得效率和质量）
- **后期任务遗忘更明显**：Task 7 的 BWT 降至 0.43，说明终身学习序列越长遗忘风险越大
- **光照鲁棒性不足**：暖→冷→暗的光照变化使光流方案从 0.83 降到 0.61，纯文本方案反而更稳定

## 亮点与洞察

- **MAP 的双模态查询**：将"看起来像什么"（语义）和"怎么动的"（运动）结合来发现原语共享性，这种直觉非常自然——人类学新技能时也会同时参考动作模式和任务描述
- **冻结-扩展的简洁性**：不需要复杂的正则化项来防遗忘，只需冻结旧提示+扩展新提示。注意力加权自动处理新旧知识的融合
- **提示作为运动原语的载体**：每个提示向量对应一种运动原语，提示选择权重可以可视化任务间的运动共享模式

## 局限与展望

- **光照敏感性**：光流在极端光照变化下失效，限制了真实环境中的部署可靠性。深度或 3D 场景流可能更鲁棒
- **后期任务遗忘**：7 个任务序列末端的 BWT 已经明显下降（0.43），更长的终身学习序列可能面临更大挑战
- **仅桌面操作**：实验限于 Franka Panda 的桌面操作任务，移动操作或双臂协作未涉及
- **提示数量需要手动调参**：最优提示数量取决于任务复杂度，缺乏自适应机制
- **单步光流**：RAFT 只提取相邻帧的光流，无法捕获长程运动模式

## 相关工作与启发

- **vs LoRA**: LoRA 在终身学习中每个任务都修改主干权重，虽然修改量小但累积修改可能导致漂移。PPL 冻结主干+只修改提示，本质上更安全
- **vs 经验回放**: 经验回放需要存储旧数据且在长序列中 BWT 为负（实际遗忘）。PPL 不需要存储旧数据，靠冻结提示保留知识
- **vs MoE**: MoE 在多技能预训练中性能好但参数量大。PPL 以提示学习的轻量方式达到接近 MoE 的性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 光流+语义联合发现运动原语是有趣的创新，冻结-扩展提示的终身学习设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实世界、多基线对比、消融充分，但任务规模有限（最多 9 个技能）
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，可视化分析有说服力
- 价值: ⭐⭐⭐⭐ 为机器人终身学习提供了轻量且有效的方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ACL 2025\] DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](../../ACL2025/robotics/drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)
- [\[CVPR 2025\] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2025\] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)

<!-- RELATED:END -->
