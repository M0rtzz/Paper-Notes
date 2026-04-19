---
title: >-
  [论文解读] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences
description: >-
  [ICLR 2026][机器人][VLM空间推理] 系统评估了 VLM 对机器人运动路径的空间推理能力，提出 4 种图像查询方法用于让 VLM 根据用户自然语言描述选择最佳运动路径，发现 Qwen2.5-VL 零样本准确率达 71.4%，且微调后小模型可获显著提升。
tags:
  - ICLR 2026
  - 机器人
  - VLM空间推理
  - 机器人运动规划
  - 运动偏好
  - 路径选择
  - 视觉语言模型评估
---

# Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences

**会议**: ICLR 2026  
**arXiv**: [2603.13100](https://arxiv.org/abs/2603.13100)  
**代码**: 无  
**领域**: 机器人  
**关键词**: VLM空间推理, 机器人运动规划, 运动偏好, 路径选择, 视觉语言模型评估  

## 一句话总结

系统评估了 VLM 对机器人运动路径的空间推理能力，提出 4 种图像查询方法用于让 VLM 根据用户自然语言描述选择最佳运动路径，发现 Qwen2.5-VL 零样本准确率达 71.4%，且微调后小模型可获显著提升。

## 研究背景与动机

在人机交互场景中，用户需要以自然语言表达对机器人运动的偏好，例如"远离窗户移动"或"沿曲线路径行进"。VLM 具备的语义知识和空间推理能力可能增强机器人规划器对新任务的泛化能力。

然而，现有研究的关键空白在于：

**运动偏好被忽视**：现有 VLM-机器人研究主要关注任务规划（做什么），而非运动规划（怎么走）——路径的拓扑属性和风格偏好从未被系统研究

**空间推理能力未知**：VLM 是否真的能理解"远离/靠近某物体"、"曲线/直线路径"等空间约束，缺乏定量评估

**集成路径不明**：将 VLM 集成到运动规划 pipeline 中的最佳方式（图像查询策略）尚未探索

## 方法详解

### 整体框架

提出一个"生成-评分-选择"的 pipeline 将 VLM 集成到运动规划中：

1. **多样路径生成**：使用 BiRRT + PRM 采样 n=50 条候选路径
2. **路径聚类**：K-means 聚类将候选路径分组，选取每簇最接近质心的路径
3. **图像可视化**：将候选路径渲染为彩色圆点轨迹叠加在场景图像上
4. **VLM 评分**：向 VLM 提供图像和用户指令，要求评分并选择最匹配的路径

### 关键设计

**四种图像查询方法**：

| 查询方法 | 描述 | 查询次数 | 图像数量 |
|----------|------|----------|----------|
| **Single-image trajectory** | 所有候选路径用不同颜色绘制在同一张图上 | 1次 | 1张 |
| **Multi-image trajectory** | 每条路径单独一张图，逐张查询VLM | k次 | k张 |
| **Single-image + visual context** | 先让VLM生成图像描述，再结合描述做选择 | 2次 | 1张 |
| **Screenshot gallery** | 为每条路径生成机器人沿路径运动的截图序列 | 1次 | 1张（gallery） |

**两类运动偏好**：

- **物体邻近偏好（Object proximity）**：描述与环境物体的空间关系，如"远离灯"、"从物体B和C之间穿过"
- **路径风格偏好（Path style）**：描述路径几何属性，如"直线"、"曲线"、"锯齿形"、"最短路径"

**数据集构建**：在 iGibson 仿真环境中构建了 558 个带语言约束的运动规划问题（126 个导航 + 432 个操控），并手动标注了每个问题的 ground truth 路径。

### 损失函数 / 训练策略

评估以零样本为主。微调实验使用 SFT（监督微调）在 98 个训练样本上对 LLaVa1.5-7B 和 Qwen2.5-VL-7B 进行微调，测试集含 28 个问题。

## 实验关键数据

### 主实验

**不同查询方法的总体准确率（导航任务，Qwen2.5-VL-72B）**：

| 查询方法 | 准确率 | Token消耗 |
|----------|--------|-----------|
| Single-image trajectory | **71.4%** | 687.3 |
| Multi-image trajectory | ~55% | 高 |
| Single-image + visual context | ~68% | 中 |
| Screenshot gallery | 略高于随机 | 高 |

**不同 VLM 在导航任务上的表现（Single-query 方法）**：

| 模型 | 邻近偏好 | 路径风格 | 总体 |
|------|----------|----------|------|
| **Qwen2.5-VL-72B** | **74.4%** | **63.9%** | **71.4%** |
| GPT-4o | 较低 | 较低 | 低于Qwen |
| LLaVa1.5 | 最低 | 最低 | ~随机 |

**操控任务表现**：

| 模型 | 邻近偏好 | 路径风格 |
|------|----------|----------|
| Qwen2.5-VL-72B | **66.3%** | 65.5% |
| GPT-4o | 较低 | **69.5%** |

### 消融实验

**微调效果（28 个测试问题）**：

| 模型 | 微调前（零样本） | 微调后 | 提升 |
|------|------------------|--------|------|
| Qwen2.5-VL-7B | ~55% | **75%** | +20% |
| LLaVa1.5-7B | ~15% | ~75% | **+60%** |
| Qwen2.5-VL-3B | ~45% | ~55% | +10% |

**Token 数量与准确率的关系**：准确率随 token 数量（即图像大小）近似线性增长。在 200-800 token 范围内，Qwen2.5-VL-7B 和 72B 均呈现线性趋势。

### 关键发现

1. **Single-image 最优**：将所有路径放在同一张图中让 VLM 能进行相对比较，优于逐张评分
2. **邻近偏好优于路径风格**：邻近类型准确率一致高于路径风格类型，因为 VLM 更擅长理解物体空间关系而非路径几何属性
3. **导航优于操控**：导航任务总体成功率（71.4%）高于操控任务（65.5%），因为操控场景中路径差异更细微
4. **Visual context 无效**：额外生成视觉描述对大模型无帮助，可能引入冗余信息，反而与模型内置的上下文追踪冲突
5. **微调效率高**：仅 98 个样本即可带来显著提升，说明 VLM 架构具备快速适配新运动偏好的能力

## 亮点与洞察

1. **问题视角新颖**：首次系统研究 VLM 对机器人运动路径（而非目标/子目标）的空间推理能力，从任务规划扩展到运动质量控制
2. **评估框架完整**：4 种查询方法 × 3 种 VLM × 2 类偏好 × 2 种任务类型的全面交叉评估
3. **实用发现**：最简单的 single-image 方法在效果和计算成本上双赢，为实际部署提供清晰指导
4. **微调潜力大**：少量样本微调即可大幅提升小模型性能，降低部署门槛

## 局限与展望

1. **VLM 幻觉问题**：模型偶尔选择不存在的颜色路径（如选了"红色"但没有红色路径）
2. **最短/最长路径识别差**：这类问题恰好是经典最优规划器（RRT*, PRM*）的强项，应融合使用
3. **仅限仿真环境**：实际机器人场景中的视觉复杂度、遮挡和动态变化未涉及
4. **准确率尚不足以直接部署**：71.4% 的准确率在安全关键场景中仍需人类介入
5. **未考虑动态障碍物和多步规划**：数据集均为静态场景的单步路径选择

## 相关工作与启发

- **与 SayCan/PaLM-E 的区别**：那些工作关注任务规划（做什么），本工作关注运动属性（怎么走），互为补充
- **与 MotionGPT 的区别**：MotionGPT 生成人体运动，本工作用 VLM 评估机器人运动，且使用图像作为中间表示
- **与 IMPACT 的区别**：IMPACT 识别可接触物体，本工作调节运动风格/约束
- **启发**：VLM 作为运动质量评判器 + 经典规划器生成候选的混合范式值得深入探索

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统评估 VLM 对机器人运动空间推理能力
- 实验充分度: ⭐⭐⭐⭐ — 查询方法、VLM 类型、偏好类型、任务类型覆盖全面
- 写作质量: ⭐⭐⭐ — 结构清晰但部分图表呈现可改进
- 价值: ⭐⭐⭐⭐ — 为 VLM 融入运动规划 pipeline 提供了扎实的基准和可行性证据

<!-- RELATED:START -->

## 相关论文

- [DAWN: Pixel Motion Diffusion is What We Need for Robot Control](../../CVPR2026/robotics/dawn_pixel_motion_diffusion_robot_control.md)
- [ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)
- [Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](../../ICCV2025/robotics/interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)

<!-- RELATED:END -->
